import sys
sys.path.insert(0, 'index_process')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_utils import DropPath, to_2tuple, trunc_normal_
from .tcformer_utils import token2map, map2token, index_points
try:
    from index_process.function import f_attn, f_weighted_sum
except:
    print('index attention is not supported')
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
# from mmcv.cnn.bricks.transformer import build_dropout


# Mlp with dwconv
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
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Attention module with spatial reduction layer
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
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
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

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# Transformer blocks
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


# The first conv layer
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


# depth-wise conv
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


# Mlp for dynamic tokens
class TCMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = TCDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
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

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, loc_orig, idx_agg, agg_weight, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# DWConv for dynamic tokens
class TCDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv_skip = nn.Conv1d(dim, dim, 1, bias=False, groups=dim)

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        B, N, C = x.shape
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])
        x_map = self.dwconv(x_map)
        # x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + \
        #     self.dwconv_skip(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = self.dwconv_skip(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = x + map2token(x_map, N, loc_orig, idx_agg, agg_weight)
        return x


# Attention for dynamic tokens
class TCAttention(nn.Module):
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
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
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

    def forward(self, x, loc_orig, x_source, idx_agg_source, H, W, conf_source=None):
        B, N, C = x.shape
        Ns = x_source.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            if conf_source is None:
                conf_source = x_source.new_zeros(B, Ns, 1)
            tmp = torch.cat([x_source, conf_source], dim=-1)
            tmp, _ = token2map(tmp, None, loc_orig, idx_agg_source, [H, W])
            x_source = tmp[:, :C]
            conf_source = tmp[:, C:]

            x_source = self.sr(x_source)
            _, _, h, w = x_source.shape
            x_source = x_source.reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_source = self.norm(x_source)
            conf_source = F.avg_pool2d(conf_source, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            conf_source = conf_source.reshape(B, 1, -1).permute(0, 2, 1).contiguous()

        kv = self.kv(x_source).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if conf_source is not None:
            conf_source = conf_source.squeeze(-1)[:, None, None, :]
            attn = attn + conf_source
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# Transformer block for dynamic tokens
class TCBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TCAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TCMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, idx_agg, agg_weight, loc_orig,
                x_source, idx_agg_source, agg_weight_source, H, W, conf_source=None):
        x1 = x + self.drop_path(self.attn(self.norm1(x),
                                          loc_orig,
                                          self.norm1(x_source),
                                          idx_agg_source,
                                          H, W, conf_source))

        x2 = x1 + self.drop_path(self.mlp(self.norm2(x1),
                                          loc_orig,
                                          idx_agg,
                                          agg_weight,
                                          H, W))
        return x2


# Local window attention for dynamic tokens
class TCWindowAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=(7, 7), rpe=False,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
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

        self.window_size = window_size
        self.rpe = rpe
        if self.rpe:
            print('Not support relative position embedding yet.')

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

        if self.rpe:
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, tar_dict, src_dict):
        x = tar_dict['x']
        loc_orig = tar_dict['loc_orig']
        idx_agg = tar_dict['idx_agg']
        agg_weight = tar_dict['agg_weight']

        x_source = src_dict['x']
        idx_agg_source = src_dict['idx_agg']
        H, W = src_dict['map_size']
        conf_source = src_dict['conf'] if 'conf' in src_dict.keys() else None
        N0 = idx_agg_source.shape[1]
        device = x.device


        B, N, C = x.shape
        Ns = x_source.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # determine window
        with torch.no_grad():
            idx_map = torch.arange(H*W, device=x.device).reshape(H, W)
            hw, ww = self.window_size
            pad_w = (ww - W % ww) % ww
            pad_h = (hw - H % hw) % hw
            H_pad = H + pad_h
            W_pad = W + pad_w
            nh, nw = H_pad // hw, W_pad // ww

            # padded region use a False token
            idx_map_pad = idx_map.new_ones(H_pad, W_pad) * H * W
            idx_map_pad[pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W] = idx_map

            token_window = idx_map_pad.view(nh, hw, nw, ww).permute(0, 2, 1, 3)
            token_window = token_window.reshape(1, nh * nw, hw * ww).expand(B, -1, -1)

            # determine the relation between window and token.
            window_map = torch.arange(nh*nw, device=x.device).float().reshape(nh, nw)
            window_map = F.interpolate(window_map[None, None, :, :], size=[H_pad, W_pad]).long()
            window_map = window_map[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W]
            if H*W == N and Ns == N and N == N0:
                idx_window = window_map.reshape(1, N).expand(B, -1)
            else:

                tmp_xy = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(x.device)[None, None, :] - 0.5
                x = tmp_xy[:, :, 0]
                y = tmp_xy[:, :, 1]

                x_grid = x.round().long().clamp(min=0, max=W - 1)
                y_grid = y.round().long().clamp(min=0, max=H - 1)
                idx_tmp = window_map[0, 0, y_grid.reshape(-1), x_grid.reshape(-1)]
                idx_batch = torch.arange(B, device=device)[:, None].expand(B, N0)

                coor = torch.stack([idx_batch.reshape(-1),
                                    idx_agg.reshape(-1),
                                    idx_tmp.reshape(-1)],
                                   dim=0)

                part_weight = torch.sparse.FloatTensor(
                    coor, agg_weight.reshape(-1), torch.Size([B, N, nh*nw])).to_dense()
                idx_window = part_weight.argmax(dim=-1)

                # # for debug only
                # print('FOR DEBUG ONLY!')
                # import matplotlib.pyplot as plt
                # tmp = token2map(idx_window.float()[..., None], None, loc_orig, idx_agg, tar_dict['map_size'])[0]
                # tmp = tmp.detach().cpu()[0, 0]
                # plt.imshow(tmp)

            idx_K = index_points(token_window, idx_window)

        # transfer x,source to feature map
        if conf_source is None:
            conf_source = x_source.new_zeros(B, Ns, 1)
        source_map = torch.cat([x_source, conf_source], dim=-1)
        source_map, _ = token2map(source_map, None, loc_orig, idx_agg_source, [H, W])
        source_map = source_map.flatten(2).permute(0, 2, 1)

        x_source = source_map[..., :C]
        conf_source = source_map[..., C:]

        # add a fake token at the end.
        x_source = torch.cat([x_source, x_source.new_zeros(B, 1, C)], dim=1)

        # print('no conf mask')
        # conf_source = torch.cat([conf_source, conf_source.new_ones(B, 1, 1) * 0], dim=1)
        conf_source = torch.cat([conf_source, conf_source.new_ones(B, 1, 1) * (-float('Inf'))], dim=1)

        # calculate
        kv = self.kv(x_source).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]     # B, num_head, N, Ch

        q = q.flatten(0, 1)
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)
        idx_K_t = idx_K[:, None, :, :].expand(-1, self.num_heads, -1, -1).flatten(0, 1)

        attn = f_attn(q * self.scale, k, idx_K_t)                 # B * num_heads, N, K
        conf = index_points(conf_source, idx_K).squeeze(-1)     # B, N, K
        attn = attn + conf[:, None, ...].expand(-1, self.num_heads, -1, -1).flatten(0, 1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = f_weighted_sum(attn, v, idx_K_t)        # B * num_head, N, C
        out = out.reshape(B, self.num_heads, N, C // self.num_heads).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# MlpBN for dynamic tokens
class TCMlpDWBN(nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels=None,
            out_channels=None,
            act_cfg=dict(type='GELU', inplace=True),
            dw_act_cfg=dict(type='GELU', inplace=True),
            drop=0.0,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.fc1 = build_conv_layer(
            conv_cfg,
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.act1 = build_activation_layer(act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, hidden_channels)[1]
        self.dw3x3 = build_conv_layer(
            conv_cfg,
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_channels,
        )
        self.dwconv_skip = nn.Conv2d(hidden_channels, hidden_channels, 1,
                                     bias=False, groups=hidden_channels)

        self.act2 = build_activation_layer(dw_act_cfg)

        self.norm2 = build_norm_layer(norm_cfg, hidden_channels)[1]
        self.fc2 = build_conv_layer(
            conv_cfg,
            hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.act3 = build_activation_layer(act_cfg)
        self.norm3 = build_norm_layer(norm_cfg, out_channels)[1]
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        B, N, C = x.shape

        x_ = x.permute(0, 2, 1)[..., None]      # fake 2D
        x_ = self.fc1(x_)
        x_ = self.norm1(x_)
        x_ = self.act1(x_)

        # real 2D
        x_map = token2map(x_.squeeze(-1).permute(0, 2, 1), None, loc_orig, idx_agg, [H, W])[0]
        x_map = self.dw3x3(x_map)
        x_map = map2token(x_map, N, loc_orig, idx_agg, agg_weight)

        # fake 2D
        x_ = self.dwconv_skip(x_)
        x_ = x_ + x_map.permute(0, 2, 1)[..., None]
        x_ = self.norm2(x_)
        x_ = self.act2(x_)
        x_ = self.drop(x_)
        x_ = self.fc2(x_)
        x_ = self.norm3(x_)
        x_ = self.act3(x_)
        x_ = self.drop(x_)

        # back to token
        x_ = x_.squeeze(-1).permute(0, 2, 1).contiguous()
        return x_


# Transformer block for dynamic tokens with local window attention
class TCWinBlock(nn.Module):
    expansion=1
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_cfg=dict(type='GELU'),
                 dw_act_cfg=None,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 conv_cfg=None,
                 mlp_norm_cfg=dict(type='BN', requires_grad=True),
                 attn_type='window',
                 num_parts=(1, 1),
                 after_cluster=False,
                 ):
        super().__init__()
        self.dim = in_channels
        self.out_dim = out_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.attn_type = attn_type
        if attn_type == 'window':
            self.attn = TCWindowAttention(
                self.dim,
                window_size=(window_size, window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
            )
        else:
            self.attn = TCPartAttention(
                self.dim,
                num_parts=num_parts,
                after_cluster=after_cluster,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
            )

        self.norm1 = build_norm_layer(norm_cfg, self.dim)[1]
        self.norm2 = build_norm_layer(norm_cfg, self.out_dim)[1]
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(self.dim * mlp_ratio)
        dw_act_cfg = dw_act_cfg or act_cfg
        self.mlp = TCMlpDWBN(
            in_channels=self.dim,
            hidden_channels=mlp_hidden_dim,
            out_channels=self.out_dim,
            act_cfg=act_cfg,
            dw_act_cfg=dw_act_cfg,
            drop=drop,
            conv_cfg=conv_cfg,
            norm_cfg=mlp_norm_cfg,
        )

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

    def forward(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            dst_dict, src_dict = inputs
        else:
            dst_dict, src_dict = inputs, None

        x = dst_dict['x']
        loc_orig = dst_dict['loc_orig']
        idx_agg = dst_dict['idx_agg']
        agg_weight = dst_dict['agg_weight']
        H, W = dst_dict['map_size']

        # norm1
        dst_dict['x'] = self.norm1(dst_dict['x'])
        if src_dict is None:
            src_dict = dst_dict
        else:
            src_dict['x'] = self.norm1(src_dict['x'])

        # attn
        x1 = x + self.drop_path(self.attn(dst_dict, src_dict))

        # mlp
        x2 = x1 + self.drop_path(self.mlp(self.norm2(x1),
                                          loc_orig,
                                          idx_agg,
                                          agg_weight,
                                          H, W))
        out_dict = {
            'x': x2,
            'idx_agg': idx_agg,
            'agg_weight': agg_weight,
            'loc_orig': loc_orig,
            'map_size': (H, W),
        }
        return out_dict


# part attention for dynamic tokens
class TCPartAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_parts=(1, 1), after_cluster=False, rpe=False,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
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

        self.num_parts = num_parts
        self.after_cluster = after_cluster
        self.rpe = rpe
        if self.rpe:
            print('Not support relative position embedding yet.')

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

        if self.rpe:
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward_qkv(self, q, kv, conf=None):
        B, N, C = q.shape
        N_kv = kv.shape[1]
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]     # B, num_head, N, Ch

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if conf is not None:
            conf = conf.squeeze(-1)[:, None, None, :]
            attn = attn + conf
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_before_cluster(self, tar_dict, src_dict):
        x = tar_dict['x']
        H, W = tar_dict['map_size']
        x_source = src_dict['x']
        conf_source = src_dict['conf'] if 'conf' in src_dict.keys() else None
        B, N, C = x.shape
        Ns = x_source.shape[1]
        Hs, Ws = src_dict['map_size']

        # transfer x, x_source, conf to 2D map
        # x_map = token2map(x, None, loc_orig, idx_agg, [H, W])
        x_map = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x_source = x_source.reshape(B, Hs, Ws, -1).permute(0, 3, 1, 2).contiguous()
        if conf_source is None:
            conf_source = x_source.new_zeros(B, Ns, 1)
        conf_source = conf_source.reshape(B, Hs, Ws, -1).permute(0, 3, 1, 2).contiguous()

        if Hs != H or Ws != W:
            x_source = F.adaptive_avg_pool2d(x_source, [H, W])
            conf_source = F.adaptive_avg_pool2d(conf_source, [H, W])


        # pad feature map and conf
        nh, nw = self.num_parts
        pad_h = (nh - H % nh) % nh
        pad_w = (nw - W % nw) % nw
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x_map, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
            x_source_pad = F.pad(x_source, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
            conf_pad = F.pad(conf_source, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                             mode='constant', value=-float('Inf'))
        else:
            x_pad = x_map
            x_source_pad = x_source
            conf_pad = conf_source

        # reshape
        B, _, H_pad, W_pad = x_pad.shape
        h, w = H_pad // nh, W_pad // nw
        q = self.part_reshape(x_pad, nh, nw, h, w)
        kv = self.part_reshape(x_source_pad, nh, nw, h, w)
        conf = self.part_reshape(conf_pad, nh, nw, h, w)

        out = self.forward_qkv(q, kv, conf)

        out = out.reshape(B, nh, nw, h, w, C)
        out = out.permute(0, 1, 3, 2, 4, 5)     # B, nh, h, nw, w, C
        out = out.reshape(B, H_pad, W_pad, C)
        if pad_h > 0 or pad_h > 0:
            out = out[:, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W, :].contiguous()
        out = out.flatten(1, 2).contiguous()
        return out

    def part_reshape(self, x, nh, nw, h, w):
        B = x.shape[0]
        C = x.shape[1]
        x = x.view(B, C, nh, h, nw, w)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()    # B, nh, nw, h, w, C
        x = x.reshape(B * nh * nw, h * w, C)
        return x

    def forward_after_cluster(self, tar_dict, src_dict):
        x = tar_dict['x']
        x_source = src_dict['x']
        conf_source = src_dict['conf'] if 'conf' in src_dict.keys() else None
        idx_agg = src_dict['idx_agg']
        N0 = idx_agg.shape[1]
        B, N, C = x.shape
        Ns = x_source.shape[1]

        nh, nw = self.num_parts
        num_parts = nh * nw

        if conf_source is None:
            conf_source = x_source.new_zeros(B, Ns, 1)

        q = x.reshape(B * num_parts, N // num_parts, C)

        if Ns == N0:
            # first cluster, source need pad
            H, W = src_dict['map_size']

            x_source = x_source.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            conf_source = conf_source.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            # pad feature map and conf
            pad_h = (nh - H % nh) % nh
            pad_w = (nw - W % nw) % nw
            if pad_h > 0 or pad_w > 0:
                x_source_pad = F.pad(x_source, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
                conf_pad = F.pad(conf_source, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                                 mode='constant', value=-float('Inf'))
            else:
                x_source_pad = x_source
                conf_pad = conf_source

            # reshape
            B, _, H_pad, W_pad = x_source_pad.shape
            h, w = H_pad // nh, W_pad // nw
            kv = self.part_reshape(x_source_pad, nh, nw, h, w)
            conf = self.part_reshape(conf_pad, nh, nw, h, w)

        else:
            # late stage, only need split
            kv = x_source.reshape(B*num_parts, Ns // num_parts, -1)
            conf = conf_source.reshape(B * num_parts, Ns // num_parts, -1)

        out = self.forward_qkv(q, kv, conf)
        out = out.reshape(B, N, C)
        return out

    def forward(self, tar_dict, src_dict):
        if self.after_cluster:
            return self.forward_after_cluster(tar_dict, src_dict)
        else:
            return self.forward_before_cluster(tar_dict, src_dict)
