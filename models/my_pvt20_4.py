import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.pvt import ( Mlp, Attention, PatchEmbed, Block, DropPath, to_2tuple, trunc_normal_,register_model, _cfg)
import math
import matplotlib.pyplot as plt
from models.smpl_head import build_smpl_head
import utils.config as cfg
from models.my_pvt2520_9 import Bottleneck, conv1x1, conv3x3
''' HR local feature '''

vis = False
# vis = True


def get_grid_loc(B, H, W, device):
    y_g, x_g = torch.arange(H, device=device).float(), torch.arange(W, device=device).float()
    y_g = 2 * ((y_g + 0.5) / H) - 1
    x_g = 2 * ((x_g + 0.5) / W) - 1
    y_map, x_map = torch.meshgrid(y_g, x_g)
    xy_map = torch.stack((x_map, y_map), dim=-1)

    loc = xy_map.reshape(-1, 2)[None, ...].repeat([B, 1, 1])
    return loc


def get_loc(x, H, W, grid_stride):
        B = x.shape[0]
        device = x.device
        y_g, x_g = torch.arange(H, device=device).float(), torch.arange(W, device=device).float()
        y_g = 2 * ((y_g + 0.5) / H) - 1
        x_g = 2 * ((x_g + 0.5) / W) - 1
        y_map, x_map = torch.meshgrid(y_g, x_g)
        xy_map = torch.stack((x_map, y_map), dim=-1)

        loc = xy_map.reshape(-1, 2)[None, ...].repeat([B, 1, 1])

        # split into grid and adaptive tokens
        pos = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        tmp = pos.reshape([H, W])
        pos_grid = tmp[grid_stride // 2:H:grid_stride, grid_stride // 2:W:grid_stride]
        pos_grid = pos_grid.reshape([-1])
        mask = torch.ones(pos.shape, dtype=torch.bool, device=pos.device)
        mask[pos_grid] = 0
        pos_ada = torch.masked_select(pos, mask)

        x_grid = torch.index_select(x, 1, pos_grid)
        x_ada = torch.index_select(x, 1, pos_ada)
        loc_grid = torch.index_select(loc, 1, pos_grid)
        loc_ada = torch.index_select(loc, 1, pos_ada)

        x = torch.cat([x_grid, x_ada], 1)
        loc = torch.cat([loc_grid, loc_ada], 1)
        N_grid = x_grid.shape[1]
        return x, loc, N_grid


def extract_local_feature(src, loc, kernel_size=(3, 3)):
    B, C, H, W = src.shape
    B, N, _ = loc.shape

    h, w = kernel_size
    x = torch.arange(w, device=loc.device, dtype=loc.dtype)
    x = (x - 0.5 * (w-1)) * 2 / W
    y = torch.arange(h, device=loc.device, dtype=loc.dtype)
    y = (y - 0.5 * (h-1)) * 2 / H
    y, x = torch.meshgrid(y, x)
    grid = torch.stack([x, y], dim=-1)
    grid = loc[:, :, None, None, :] + grid[None, None, ...]     # (B, N, h, w, 2)

    loc_feature = F.grid_sample(src, grid.flatten(2, 3))        # (B, C, N, h * w)
    loc_feature = loc_feature.reshape(B, C, N, h, w)            # (B, C, N, h, w)
    loc_feature = loc_feature.permute(0, 2, 1, 3, 4).contiguous()            # (B, N, C, h, w)
    return loc_feature.flatten(0, 1)                            # (B * N, C, h, w)


def extract_neighbor_feature(src, loc, kernel_size=(3, 3)):
    B, C, H, W = src.shape
    B, N, _ = loc.shape

    h, w = kernel_size
    x = torch.arange(w, device=loc.device, dtype=loc.dtype)
    x = (x - 0.5 * (w-1)) * 2 / W
    y = torch.arange(h, device=loc.device, dtype=loc.dtype)
    y = (y - 0.5 * (h-1)) * 2 / H
    y, x = torch.meshgrid(y, x)
    grid = torch.stack([x, y], dim=-1)
    grid = loc[:, :, None, None, :] + grid[None, None, ...]     # (B, N, h, w, 2)
    loc_feature = F.grid_sample(src, grid.flatten(2, 3))        # (B, C, N, h * w)
    loc_feature = loc_feature.permute(0, 2, 3, 1)               # (B, N, h * w, C)
    return loc_feature


def gumble_top_k(x, k, dim, T=1, p_value=1e-6):
    # Noise
    noise = torch.rand_like(x)
    noise = -1 * (noise + p_value).log()
    noise = -1 * (noise + p_value).log()
    # add
    x = x / T + noise
    _, index_k = torch.topk(x, k, dim)
    return index_k


def guassian_filt(x, kernel_size=3, sigma=2):
    channels = x.shape[1]

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size, device=x.device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size).contiguous()
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).contiguous()
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    paddding = int((kernel_size - 1) // 2)

    y = F.conv2d(
        input=x,
        weight=gaussian_kernel,
        stride=1,
        padding=paddding,
        dilation=1,
        groups=channels
    )
    return y


def reconstruct_feature(feature, mask, kernel_size, sigma):
    if kernel_size < 3:
        return feature
    feature = feature * mask
    out = guassian_filt(torch.cat([feature, mask], dim=1),
                        kernel_size=kernel_size, sigma=sigma)
    C = out.shape[1] - 1
    feature_inter = out[:, :C]
    mask_inter = out[:, C:]
    # tmp = mask_inter.min()
    feature_inter = feature_inter / (mask_inter + 1e-6)
    mask_inter = (mask_inter > 0).float()
    feature_inter = feature_inter * mask_inter
    out = feature + (1 - mask) * feature_inter
    return out


def token2map(x, loc, map_size, kernel_size, sigma, return_mask=False):
    H, W = map_size
    B, N, C = x.shape
    loc = loc.clamp(-1, 1)
    loc = 0.5 * (loc + 1) * torch.FloatTensor([W, H]).to(loc.device)[None, None, :] - 0.5
    loc = loc.round().long()
    loc[..., 0] = loc[..., 0].clamp(0, W-1)
    loc[..., 1] = loc[..., 1].clamp(0, H-1)
    idx = loc[..., 0] + loc[..., 1] * W
    idx = idx + torch.arange(B)[:, None].to(loc.device) * H * W

    out = x.new_zeros(B*H*W, C+1)
    out.index_add_(dim=0, index=idx.reshape(B*N),
                   source=torch.cat([x, x.new_ones(B, N, 1)], dim=-1).reshape(B*N, C+1))
    out = out.reshape(B, H, W, C+1).permute(0, 3, 1, 2).contiguous()
    assert out.shape[1] == C+1
    feature = out[:, :C, :, :]
    mask = out[:, C:, :, :]

    # try:
    #     feature, mask = out[:, :C, :, :], out[:, C:, :, :]
    # except:
    #     info = 'out shape: ' + str(out.shape) + ' C: ' + str(C)
    #     print(info)
    #     print(info)
    #     raise KeyError(info)

    # del out

    feature = feature / (mask + 1e-6)
    mask = (mask > 0).float()
    feature = feature * mask
    feature = reconstruct_feature(feature, mask, kernel_size, sigma)
    if return_mask:
        return feature, mask
    return feature


def map2token(feature_map, loc_xy, mode='bilinear', align_corners=False):
    B, N, _ = loc_xy.shape
    # B, C, H, W = feature_map.shape
    # loc_xy = loc_xy.type(feature_map.dtype) * 2 - 1
    loc_xy = loc_xy.unsqueeze(1).type(feature_map.dtype)
    tokens = F.grid_sample(feature_map, loc_xy, mode=mode, align_corners=align_corners)
    tokens = tokens.permute(0, 2, 3, 1).squeeze(1).contiguous()
    return tokens


def show_tokens(x, out, N_grid=14*14):
    import matplotlib.pyplot as plt
    IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
    x = x * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
    # for i in range(x.shape[0]):
    for i in range(1):
        img = x[i].permute(1, 2, 0).detach().cpu()
        ax = plt.subplot(1, len(out)+2, 1)
        ax.clear()
        ax.imshow(img)
        for lv in range(len(out)):
            ax = plt.subplot(1, len(out)+2, lv+2+(lv > 0))
            ax.clear()
            ax.imshow(img, extent=[0, 1, 0, 1])
            loc = out[lv][1]
            loc = 0.5 * loc + 0.5
            loc_grid = loc[i, :N_grid].detach().cpu().numpy()
            ax.scatter(loc_grid[:, 0], 1 - loc_grid[:, 1], c='blue', s=0.4+lv*0.1)
            loc_ada = loc[i, N_grid:].detach().cpu().numpy()
            ax.scatter(loc_ada[:, 0], 1 - loc_ada[:, 1], c='red', s=0.4+lv*0.1)
    return


def show_conf(conf, loc):
    H = int(conf.shape[1]**0.5)
    if H == 56:
        conf = F.softmax(conf, dim=1)
        conf_map = token2map(conf,  map_size=[H, H], loc=loc, kernel_size=3, sigma=2)
        lv = 3
        ax = plt.subplot(1, 6, lv)
        ax.clear()
        ax.imshow(conf_map[0, 0].detach().cpu())







class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class MyMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MyDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, loc, H, W, kernel_size, sigma):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dwconv(x, loc, H, W, kernel_size, sigma)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MyDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, loc, H, W, kernel_size, sigma):
        B, N, C = x.shape
        x = token2map(x, loc, [H, W], kernel_size=kernel_size, sigma=sigma)
        # x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        # x = x.flatten(2).transpose(1, 2)
        x = map2token(x, loc)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        #if sr_ratio > 1:
            # self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        #if self.sr_ratio > 1:
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #else:
        #    kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MyAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        #if sr_ratio > 1:
            # self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, x_source, loc_source, H, W, conf_source=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        h, w = H // self.sr_ratio, W // self.sr_ratio
        x_source = token2map(x_source, loc_source, [h, w], 1, 1)
        x_source = self.sr(x_source)
        x_source = x_source.reshape(B, C, -1).permute(0, 2, 1)
        x_source = self.norm(x_source)
        x_source = self.act(x_source)
        if conf_source is not None:
            conf_source = token2map(conf_source, loc_source, [h, w], 1, 1)
            conf_source = conf_source.reshape(B, 1, -1).permute(0, 2, 1)

        _, Ns, _ = x_source.shape
        kv = self.kv(x_source).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if conf_source is not None:
            conf_source = conf_source.squeeze(-1)[:, None, None, :]
            attn = attn + conf_source
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class MyBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MyAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, x_source, loc, loc_source, H, W, conf_source=None):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x_source), loc_source, H, W, conf_source))
        kernel_size = self.attn.sr_ratio + 1
        x = x + self.drop_path(self.mlp(self.norm2(x), loc, H, W, kernel_size, 2))
        return x


# from partialconv2d import PartialConv2d
class DownLayer(nn.Module):
    """ Down sample
    """
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate, down_block):
        super().__init__()
        # self.sample_num = sample_num
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out

        self.block = down_block
        # self.pos_drop = nn.Dropout(p=drop_rate)
        # self.gumble_sigmoid = GumbelSigmoid()
        # temperature of confidence weight
        self.register_buffer('T', torch.tensor(1.0, dtype=torch.float))
        self.T_min = 1
        self.T_decay = 0.9998
        self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=1, padding=1)
        # self.conv = PartialConv2d(embed_dim, self.block.dim_out, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.conf = nn.Linear(self.dim_out, 1)

    def forward(self, x, pos, H, W, N_grid):
        # x, mask = token2map(x, pos, [H, W], 1, 2, return_mask=True)
        # x = self.conv(x, mask)

        kernel_size = self.block.attn.sr_ratio + 1
        x = token2map(x, pos, [H, W], kernel_size, 2)
        x = self.conv(x)
        x = map2token(x, pos)
        B, N, C = x.shape
        # sample_num = max(math.ceil(N * self.sample_ratio) - N_grid, 0)
        sample_num = max(math.ceil(N * self.sample_ratio), 0)

        x_grid = x[:, :N_grid]
        x_ada = x[:, N_grid:]
        pos_grid = pos[:, :N_grid]
        pos_ada = pos[:, N_grid:]

        conf = self.conf(self.norm(x))
        conf_ada = conf[:, N_grid:]
        if vis:
            show_conf(conf, pos)
        # temperature
        # T = self.T if self.training else self.T_min
        T = self.T
        self.T = (self.T * self.T_decay).clamp(self.T_min, 1.0)

        # _, index_down = torch.topk(conf_ada, self.sample_num, 1)
        index_down = gumble_top_k(conf_ada, sample_num, 1, T=T)
        # conf = F.softmax(conf, dim=1) * N
        # conf = F.sigmoid(conf)
        # conf = self.gumble_sigmoid(conf)

        x_down = torch.gather(x_ada, 1, index_down.expand([B, sample_num, C]))
        pos_down = torch.gather(pos_ada, 1, index_down.expand([B, sample_num, 2]))

        x_down = torch.cat([x_grid, x_down], 1)
        pos_down = torch.cat([pos_grid, pos_down], 1)

        x_down = self.block(x_down, x, pos_down, pos, H, W, conf)
        return x_down, pos_down


class LocalNet(nn.Module):
    def __init__(self, layers=3,
                 block=Bottleneck, norm_layer=nn.BatchNorm2d,
                 zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 ):
        super().__init__()
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers)
        self.layer2 = self._make_layer(block, 32, 1, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class MyPVT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], pretrained=None, alpha=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.grid_stride = sr_ratios[0]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])

        # self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
        #                                       embed_dim=embed_dims[1])
        # self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
        #                                       embed_dim=embed_dims[2])
        # self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
        #                                       embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.down_layers1 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[0], dim_out=embed_dims[1], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[1], num_heads=num_heads[1],
                                            mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[0])
                                      )
        self.block2 = nn.ModuleList([MyBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(1, depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.down_layers2 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[1], dim_out=embed_dims[2], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[2], num_heads=num_heads[2],
                                            mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[1]))
        self.block3 = nn.ModuleList([MyBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(1, depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.down_layers3 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[2], dim_out=embed_dims[3], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[3], num_heads=num_heads[3],
                                            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[2]))
        self.block4 = nn.ModuleList([MyBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(1, depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.head_type = kwargs['head_type'] if 'head_type' in kwargs else 'hmr'
        self.head = build_smpl_head(embed_dims[3], self.head_type)

        self.local1_conv1 = LocalNet()
        self.local1_conv2 = nn.Conv2d(in_channels=embed_dims[0]+128, out_channels=embed_dims[0], kernel_size=1)

        self.local2_conv1 = LocalNet()
        self.local2_fc = nn.Linear(in_features=embed_dims[1]+128, out_features=embed_dims[1])

        self.local3_conv1 = LocalNet()
        self.local3_fc = nn.Linear(in_features=embed_dims[2]+128, out_features=embed_dims[2])

        self.local4_conv1 = LocalNet()
        self.local4_fc = nn.Linear(in_features=embed_dims[3]+128, out_features=embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[0]
        self.down_layers1.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[1] - 1):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[1] - 1
        self.down_layers2.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[2] - 1):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[2] - 1
        self.down_layers3.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[3] - 1):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if vis:
            outs = []
            img = x
        img = x

        B = x.shape[0]
        x = F.avg_pool2d(x, kernel_size=2)
        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)

        local = self.local1_conv1(img)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.local1_conv2(torch.cat([x, local], dim=1))
        x = x.flatten(2).permute(0, 2, 1)

        x, loc, N_grid = get_loc(x, H, W, self.grid_stride)
        if vis: outs.append((x, loc, [H, W]))

        # stage 2
        x, loc = self.down_layers1(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block2:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm2(x)

        local = self.local2_conv1(img)
        local = map2token(local, loc)
        x = torch.cat([x, local], dim=-1)
        x = self.local2_fc(x)
        if vis: outs.append((x, loc, [H, W]))

        # stage 3
        x, loc = self.down_layers2(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block3:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm3(x)
        local = self.local3_conv1(img)
        local = map2token(local, loc)
        x = torch.cat([x, local], dim=-1)
        x = self.local3_fc(x)
        if vis: outs.append((x, loc, [H, W]))

        # stage 4
        x, loc = self.down_layers3(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block4:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm4(x)
        local = self.local4_conv1(img)
        local = map2token(local, loc)
        x = torch.cat([x, local], dim=-1)
        x = self.local4_fc(x)
        if vis:
            outs.append((x, loc, [H, W]))
            show_tokens(img, outs, N_grid)

        if self.head_type == 'tcmr':
            return x
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

@register_model
def mypvt20_4_small(pretrained=False, **kwargs):
    model = MyPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


class MyPVTb(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], pretrained=None, alpha=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.grid_stride = sr_ratios[0]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])

        # self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
        #                                       embed_dim=embed_dims[1])
        # self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
        #                                       embed_dim=embed_dims[2])
        # self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
        #                                       embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.down_layers1 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[0], dim_out=embed_dims[1], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[1], num_heads=num_heads[1],
                                            mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[0])
                                      )
        self.block2 = nn.ModuleList([MyBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(1, depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.down_layers2 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[1], dim_out=embed_dims[2], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[2], num_heads=num_heads[2],
                                            mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[1]))
        self.block3 = nn.ModuleList([MyBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(1, depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.down_layers3 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[2], dim_out=embed_dims[3], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[3], num_heads=num_heads[3],
                                            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[2]))
        self.block4 = nn.ModuleList([MyBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(1, depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.head_type = kwargs['head_type'] if 'head_type' in kwargs else 'hmr'
        self.head = build_smpl_head(embed_dims[3], self.head_type)

        self.local1_conv1 = LocalNet()
        self.local1_conv2 = nn.Conv2d(in_channels=embed_dims[0]+128, out_channels=embed_dims[0], kernel_size=1)

        self.local2_conv1 = LocalNet()
        self.local2_fc = nn.Linear(in_features=embed_dims[1]+128, out_features=embed_dims[1])

        self.local3_conv1 = LocalNet()
        self.local3_fc = nn.Linear(in_features=embed_dims[2]+128, out_features=embed_dims[2])

        # self.local4_conv1 = LocalNet()
        # self.local4_fc = nn.Linear(in_features=embed_dims[3]+128, out_features=embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[0]
        self.down_layers1.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[1] - 1):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[1] - 1
        self.down_layers2.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[2] - 1):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[2] - 1
        self.down_layers3.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[3] - 1):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if vis:
            outs = []
            img = x
        img = x

        B = x.shape[0]
        x = F.avg_pool2d(x, kernel_size=2)
        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)

        local = self.local1_conv1(img)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.local1_conv2(torch.cat([x, local], dim=1))
        x = x.flatten(2).permute(0, 2, 1)

        x, loc, N_grid = get_loc(x, H, W, self.grid_stride)
        if vis: outs.append((x, loc, [H, W]))

        # stage 2
        x, loc = self.down_layers1(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block2:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm2(x)

        local = self.local2_conv1(img)
        local = map2token(local, loc)
        x = torch.cat([x, local], dim=-1)
        x = self.local2_fc(x)
        if vis: outs.append((x, loc, [H, W]))

        # stage 3
        x, loc = self.down_layers2(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block3:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm3(x)
        local = self.local3_conv1(img)
        local = map2token(local, loc)
        x = torch.cat([x, local], dim=-1)
        x = self.local3_fc(x)
        if vis: outs.append((x, loc, [H, W]))

        # stage 4
        x, loc = self.down_layers3(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block4:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm4(x)
        # local = self.local4_conv1(img)
        # local = map2token(local, loc)
        # x = torch.cat([x, local], dim=-1)
        # x = self.local4_fc(x)
        if vis:
            outs.append((x, loc, [H, W]))
            show_tokens(img, outs, N_grid)

        if self.head_type == 'tcmr':
            return x
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

@register_model
def mypvt20_4b_small(pretrained=False, **kwargs):
    model = MyPVTb(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


class MyPVTc(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], pretrained=None, alpha=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.grid_stride = sr_ratios[0]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])

        # self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
        #                                       embed_dim=embed_dims[1])
        # self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
        #                                       embed_dim=embed_dims[2])
        # self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
        #                                       embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.down_layers1 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[0], dim_out=embed_dims[1], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[1], num_heads=num_heads[1],
                                            mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[0])
                                      )
        self.block2 = nn.ModuleList([MyBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(1, depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.down_layers2 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[1], dim_out=embed_dims[2], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[2], num_heads=num_heads[2],
                                            mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[1]))
        self.block3 = nn.ModuleList([MyBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(1, depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.down_layers3 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[2], dim_out=embed_dims[3], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[3], num_heads=num_heads[3],
                                            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[2]))
        self.block4 = nn.ModuleList([MyBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(1, depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.head_type = kwargs['head_type'] if 'head_type' in kwargs else 'hmr'
        self.head = build_smpl_head(embed_dims[3], self.head_type)

        # self.local1_conv1 = LocalNet()
        # self.local1_conv2 = nn.Conv2d(in_channels=embed_dims[0]+128, out_channels=embed_dims[0], kernel_size=1)

        self.local2_conv1 = LocalNet()
        self.local2_fc = nn.Linear(in_features=embed_dims[1]+128, out_features=embed_dims[1])

        # self.local3_conv1 = LocalNet()
        # self.local3_fc = nn.Linear(in_features=embed_dims[2]+128, out_features=embed_dims[2])

        # self.local4_conv1 = LocalNet()
        # self.local4_fc = nn.Linear(in_features=embed_dims[3]+128, out_features=embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[0]
        self.down_layers1.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[1] - 1):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[1] - 1
        self.down_layers2.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[2] - 1):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[2] - 1
        self.down_layers3.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[3] - 1):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if vis:
            outs = []
            img = x
        img = x

        B = x.shape[0]
        x = F.avg_pool2d(x, kernel_size=2)
        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)

        # local = self.local1_conv1(img)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = self.local1_conv2(torch.cat([x, local], dim=1))
        # x = x.flatten(2).permute(0, 2, 1)

        x, loc, N_grid = get_loc(x, H, W, self.grid_stride)
        if vis: outs.append((x, loc, [H, W]))

        # stage 2
        x, loc = self.down_layers1(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2

        local = self.local2_conv1(img)
        local = map2token(local, loc)
        x = torch.cat([x, local], dim=-1)
        x = self.local2_fc(x)

        for blk in self.block2:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm2(x)


        if vis: outs.append((x, loc, [H, W]))

        # stage 3
        x, loc = self.down_layers2(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block3:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm3(x)
        # local = self.local3_conv1(img)
        # local = map2token(local, loc)
        # x = torch.cat([x, local], dim=-1)
        # x = self.local3_fc(x)
        if vis: outs.append((x, loc, [H, W]))

        # stage 4
        x, loc = self.down_layers3(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block4:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm4(x)
        # local = self.local4_conv1(img)
        # local = map2token(local, loc)
        # x = torch.cat([x, local], dim=-1)
        # x = self.local4_fc(x)
        if vis:
            outs.append((x, loc, [H, W]))
            show_tokens(img, outs, N_grid)

        if self.head_type == 'tcmr':
            return x
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

@register_model
def mypvt20_4c_small(pretrained=False, **kwargs):
    model = MyPVTc(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


class MyPVTd(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], pretrained=None, alpha=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.grid_stride = sr_ratios[0]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])

        # self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
        #                                       embed_dim=embed_dims[1])
        # self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
        #                                       embed_dim=embed_dims[2])
        # self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
        #                                       embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.down_layers1 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[0], dim_out=embed_dims[1], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[1], num_heads=num_heads[1],
                                            mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[0])
                                      )
        self.block2 = nn.ModuleList([MyBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(1, depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.down_layers2 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[1], dim_out=embed_dims[2], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[2], num_heads=num_heads[2],
                                            mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[1]))
        self.block3 = nn.ModuleList([MyBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(1, depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.down_layers3 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[2], dim_out=embed_dims[3], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[3], num_heads=num_heads[3],
                                            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[2]))
        self.block4 = nn.ModuleList([MyBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(1, depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.head_type = kwargs['head_type'] if 'head_type' in kwargs else 'hmr'
        self.head = build_smpl_head(embed_dims[3], self.head_type)

        # self.local1_conv1 = LocalNet()
        # self.local1_conv2 = nn.Conv2d(in_channels=embed_dims[0]+128, out_channels=embed_dims[0], kernel_size=1)

        self.local2_conv1 = LocalNet()
        self.local2_fc = nn.Linear(in_features=embed_dims[1]+128, out_features=embed_dims[1])

        # self.local3_conv1 = LocalNet()
        # self.local3_fc = nn.Linear(in_features=embed_dims[2]+128, out_features=embed_dims[2])

        # self.local4_conv1 = LocalNet()
        # self.local4_fc = nn.Linear(in_features=embed_dims[3]+128, out_features=embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[0]
        self.down_layers1.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[1] - 1):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[1] - 1
        self.down_layers2.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[2] - 1):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[2] - 1
        self.down_layers3.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[3] - 1):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if vis:
            outs = []
            img = x
        img = x

        B = x.shape[0]
        x = F.avg_pool2d(x, kernel_size=2)
        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)

        # local = self.local1_conv1(img)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = self.local1_conv2(torch.cat([x, local], dim=1))
        # x = x.flatten(2).permute(0, 2, 1)

        x, loc, N_grid = get_loc(x, H, W, self.grid_stride)
        if vis: outs.append((x, loc, [H, W]))

        # stage 2
        x, loc = self.down_layers1(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2

        local = self.local2_conv1(img)
        local = map2token(local, loc)
        x = self.local2_fc(torch.cat([x, local], dim=-1)) + x

        for blk in self.block2:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm2(x)


        if vis: outs.append((x, loc, [H, W]))

        # stage 3
        x, loc = self.down_layers2(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block3:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm3(x)
        # local = self.local3_conv1(img)
        # local = map2token(local, loc)
        # x = torch.cat([x, local], dim=-1)
        # x = self.local3_fc(x)
        if vis: outs.append((x, loc, [H, W]))

        # stage 4
        x, loc = self.down_layers3(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block4:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm4(x)
        # local = self.local4_conv1(img)
        # local = map2token(local, loc)
        # x = torch.cat([x, local], dim=-1)
        # x = self.local4_fc(x)
        if vis:
            outs.append((x, loc, [H, W]))
            show_tokens(img, outs, N_grid)

        if self.head_type == 'tcmr':
            return x
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

@register_model
def mypvt20_4d_small(pretrained=False, **kwargs):
    model = MyPVTd(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model

# For test
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = mypvt20_4_small(drop_path_rate=0.1).to(device)
    model.reset_drop_path(0.1)

    empty_input = torch.rand([2, 3, 448, 448], device=device)
    del device

    output = model(empty_input)
    # tmp = output.sum()
    # print(tmp)

    print('Finish')

