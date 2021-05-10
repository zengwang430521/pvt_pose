import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .pvt import ( Mlp, Attention, PatchEmbed, Block, DropPath, to_2tuple, trunc_normal_,register_model, _cfg)
from .smpl_head import HMRHead
import utils.config as cfg


class MyAttention(nn.Module):
    def __init__(self, dim, dim_out=-1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        if dim_out < 0:
            dim_out = dim
        # assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert dim_out % num_heads == 0, f"dim_out {dim_out} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, x_source, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_source = x_source.permute(0, 2, 1).reshape(B, C, H, W)
            x_source = self.sr(x_source).reshape(B, C, -1).permute(0, 2, 1)
            x_source = self.norm(x_source)
        _, Ns, _ = x_source.shape

        k = self.k(x_source).reshape(B, Ns, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_source).reshape(B, Ns, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)
        # kv = self.kv(x_source).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim_out)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MyBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, dim_out=None, sr_ratio=1, alpha=1):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.norm1 = norm_layer(dim)
        self.attn = MyAttention(
            dim,
            dim_out=dim_out,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_fc = False
        if dim_out != dim:
            self.use_fc = True
            self.fc = nn.Linear(dim, dim_out)
        self.alpha = alpha

    def forward(self, x, x_source, H, W):
        if self.use_fc:
            x = self.fc(x) + self.drop_path(self.attn(self.norm1(x), self.norm1(x_source), H, W)) * self.alpha
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x_source), H, W)) * self.alpha

        x = x + self.drop_path(self.mlp(self.norm2(x))) * self.alpha
        return x


class DownLayer(nn.Module):
    """ Down sample
    """
    def __init__(self, sample_num, embed_dim, drop_rate, down_block):
        super().__init__()
        self.sample_num = sample_num
        self.norm = nn.LayerNorm(embed_dim)
        self.conf = nn.Linear(embed_dim, 1)
        self.block = down_block
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x, pos, pos_embed, H, W):
        B, N, C = x.shape
        assert self.sample_num <= N
        # FIXME: Should we add position embedding before or after down sampling ?

        # pos_feature = torch.index_select(pos_embed, 1, pos.reshape(-1))
        # pos_feature = pos_feature.reshape(B, N, -1)
        # x = x + pos_feature
        # x = self.pos_drop(x)

        conf = self.conf(self.norm(x))
        conf = F.softmax(conf, dim=1) * N
        _, index_down = torch.topk(conf, self.sample_num, 1)
        x_down = torch.gather(x, 1, index_down.expand([B, self.sample_num, C]))
        pos_down = torch.gather(pos, 1, index_down.squeeze(-1))
        x_down, pos_down = x_down.contiguous(), pos_down.contiguous()

        x = x * conf
        x_down = self.block(x_down, x, H, W)

        pos_feature = torch.index_select(pos_embed, 1, pos_down.reshape(-1))
        pos_feature = pos_feature.reshape(B, self.sample_num, -1)
        x_down += pos_feature
        x_down = self.pos_drop(x_down)

        return x_down, pos_down


class MyPyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], alpha=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.alpha = alpha

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[1]))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[2]))
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[3]))

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        sample_num = self.patch_embed1.num_patches
        cur = 0

        # stage 1
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])  # , alpha=alpha)
            for i in range(depths[0])])
        cur += depths[0]

        # stage 2
        sample_num = sample_num // 4
        self.down_layers1 = DownLayer(sample_num=sample_num, embed_dim=embed_dims[0], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[0], dim_out=embed_dims[1], num_heads=num_heads[1],
                                            mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[0], alpha=alpha))

        self.block2 = nn.ModuleList([MyBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], alpha=alpha)
            for i in range(1, depths[1])])
        cur += depths[1]

        # stage 3
        sample_num = sample_num // 4
        self.down_layers2 = DownLayer(sample_num=sample_num, embed_dim=embed_dims[1], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[1], dim_out=embed_dims[2], num_heads=num_heads[2],
                                            mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[1], alpha=alpha))
        self.block3 = nn.ModuleList([MyBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], alpha=alpha)
            for i in range(1, depths[2])])
        cur += depths[2]

        # stage 4
        sample_num = sample_num // 4
        self.down_layers3 = DownLayer(sample_num=sample_num, embed_dim=embed_dims[2], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[2], dim_out=embed_dims[3], num_heads=num_heads[3],
                                            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[2], alpha=alpha))
        self.block4 = nn.ModuleList([MyBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], alpha=alpha)
            for i in range(1, depths[3])])
        self.norm = norm_layer(embed_dims[3])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]

        # stage 1 Unchanged
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)

        # stage 2
        pos = torch.arange(x.shape[1], dtype=torch.long, device=x.device)[None, :].repeat([B, 1])
        x, pos = self.down_layers1(x, pos, self.pos_embed2, H, W)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block2:
            x = blk(x, x, H, W)

        # stage 3
        x, pos = self.down_layers2(x, pos, self.pos_embed3, H, W)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block3:
            x = blk(x, x, H, W)

        # stage 4
        x, pos = self.down_layers3(x, pos, self.pos_embed4, H, W)     # down sample
        H, W = H // 2, W // 2
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.block4:
            x = blk(x, x, H, W)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def mypvt_small(pretrained=False, **kwargs):
    model = MyPyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def mypvt_small_2(pretrained=False, **kwargs):
    model = MyPyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 1, 1, 1], **kwargs)
    model.default_cfg = _cfg()

    return model








class MyAttention2(nn.Module):
    def __init__(self, dim, dim_out=-1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        if dim_out < 0:
            dim_out = dim
        # assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert dim_out % num_heads == 0, f"dim_out {dim_out} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        if not dim_out == dim:
            self.k2 = nn.Linear(dim_out, dim_out, bias=qkv_bias)
            self.v2 = nn.Linear(dim_out, dim_out, bias=qkv_bias)
        else:
            self.k2, self.v2 = self.k, self.v

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward_grid(self, x, x_source, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)

        _, Ns, _ = x_source.shape

        k = self.k(x_source).reshape(B, Ns, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_source).reshape(B, Ns, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)
        # kv = self.kv(x_source).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim_out)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_ada(self, x, x_source, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)

        _, Ns, _ = x_source.shape

        k = self.k2(x_source).reshape(B, Ns, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)
        v = self.v2(x_source).reshape(B, Ns, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)
        # kv = self.kv(x_source).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim_out)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    '''
    x: Tuple of 2 tensor (x_grid, x_ada)
    x_grid: Feature of grid location.
    x_ada: Feature of ada location.
    x_sourceL Tuple of 2 tensor, the same as x.
    '''
    def forward(self, x, x_source):
        x_grid, x_ada = x
        # merge grid and ada nodes in x_source
        x_source = torch.cat(x_source, dim=1)
        # gather feature into x_grid
        x_grid = self.forward_grid(x_grid, x_source, 1, 1)
        # get feature from new x_grid
        x_ada = self.forward_ada(x_ada, x_grid, 1, 1)
        return (x_grid, x_ada)


def tuple_add(x, y):
    x = [x_i + y_i for (x_i, y_i) in zip(x, y)]
    return x


def tuple_forward(net, x):
    N_grid = x[0].shape[1]
    N_ada = x[1].shape[2]
    x = torch.cat(x, dim=1)
    x = net(x)
    x = (x[:, :N_grid], x[:, N_grid:])
    return x


def tuple_times(x, alpha):
    return [x_i * alpha for x_i in x]


class MyBlock2(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, dim_out=None, sr_ratio=1, alpha=1):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.norm1 = norm_layer(dim)
        self.attn = MyAttention2(
            dim,
            dim_out=dim_out,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_fc = False
        if dim_out != dim:
            self.use_fc = True
            self.fc = nn.Linear(dim, dim_out)
        self.alpha = alpha

    def forward(self, x, x_source, H=1, W=1):
        y = self.attn(tuple_forward(self.norm1, x),
                      tuple_forward(self.norm1, x_source))
        y = tuple_forward(self.drop_path, y)
        y = tuple_times(y, self.alpha)
        if self.use_fc:
            x = tuple_add(tuple_forward(self.fc, x), y)
        else:
            x = tuple_add(x, y)

        y = tuple_forward(self.norm2, x)
        y = tuple_forward(self.mlp, y)
        y = tuple_forward(self.drop_path, y)
        y = tuple_times(y, self.alpha)
        x = tuple_add(x, y)
        return x

        # if self.use_fc:
        #     x = self.fc(x) + self.drop_path(self.attn(self.norm1(x), self.norm1(x_source), H, W)) * self.alpha
        # else:
        #     x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x_source), H, W)) * self.alpha
        # x = x + self.drop_path(self.mlp(self.norm2(x))) * self.alpha
        # return x


class DownLayer2(nn.Module):
    """ Down sample
    """
    def __init__(self, sample_num, embed_dim, drop_rate, down_block, norm_layer=nn.LayerNorm):
        super().__init__()
        self.sample_num = sample_num
        self.norm = norm_layer(embed_dim)
        self.conf = nn.Linear(embed_dim, 1)
        self.block = down_block
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x, pos, pos_embed, H=1, W=1):
        x_grid, x_ada = x
        pos_grid, pos_ada = pos
        B, N_g, C = x_grid.shape
        B, N, C = x_ada.shape
        assert self.sample_num <= N

        # only x_ada needs down sample
        # FIXME: Should we add position embedding before or after down sampling ?
        conf = self.conf(self.norm(x_ada))
        conf = F.softmax(conf, dim=1) * N
        _, index_down = torch.topk(conf, self.sample_num, 1)
        x_down = torch.gather(x_ada, 1, index_down.expand([B, self.sample_num, C]))
        pos_down = torch.gather(pos_ada, 1, index_down.squeeze(-1))
        x_down, pos_down = x_down.contiguous(), pos_down.contiguous()

        x_ada = x_ada * conf
        x = self.block((x_grid, x_down), (x_grid, x_ada))
        pos_feature = (torch.index_select(pos_embed, 1, pos_grid.reshape(-1)).reshape(B, N_g, -1),
                       torch.index_select(pos_embed, 1, pos_down.reshape(-1)).reshape(B, self.sample_num, -1))
        x = tuple_add(x, pos_feature)
        x = tuple_forward(self.pos_drop, x)
        return x, pos


class MyPVT2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 1, 1, 1], alpha=1, grid_stride=8):
        super().__init__()
        self.grid_stride = int(grid_stride)
        self.num_classes = num_classes
        self.depths = depths
        self.alpha = alpha

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[1]))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[2]))
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[3]))

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        sample_num = self.patch_embed1.num_patches
        cur = 0

        # stage 1, unchanged
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])  # , alpha=alpha)
            for i in range(depths[0])])
        cur += depths[0]

        # stage 2
        sample_num = sample_num // 4
        self.down_layers1 = DownLayer2(sample_num=sample_num, embed_dim=embed_dims[0], drop_rate=drop_rate,
                                       norm_layer=norm_layer,
                                       down_block=MyBlock2(
                                            dim=embed_dims[0], dim_out=embed_dims[1], num_heads=num_heads[1],
                                            mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=1, alpha=alpha))

        self.block2 = nn.ModuleList([MyBlock2(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=1, alpha=alpha)
            for i in range(1, depths[1])])
        cur += depths[1]

        # stage 3
        sample_num = sample_num // 4
        self.down_layers2 = DownLayer2(sample_num=sample_num, embed_dim=embed_dims[1], drop_rate=drop_rate,
                                       norm_layer=norm_layer,
                                       down_block=MyBlock2(
                                            dim=embed_dims[1], dim_out=embed_dims[2], num_heads=num_heads[2],
                                            mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=1, alpha=alpha))
        self.block3 = nn.ModuleList([MyBlock2(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=1, alpha=alpha)
            for i in range(1, depths[2])])
        cur += depths[2]

        # stage 4
        sample_num = sample_num // 4
        self.down_layers3 = DownLayer2(sample_num=sample_num, embed_dim=embed_dims[2], drop_rate=drop_rate,
                                       norm_layer=norm_layer,
                                       down_block=MyBlock2(
                                            dim=embed_dims[2], dim_out=embed_dims[3], num_heads=num_heads[3],
                                            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=1, alpha=alpha))
        self.block4 = nn.ModuleList([MyBlock2(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=1, alpha=alpha)
            for i in range(1, depths[3])])
        self.norm = norm_layer(embed_dims[3])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.head = HMRHead(embed_dims[3], cfg.SMPL_MEAN_PARAMS, 3)


        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]

        # stage 1 Unchanged
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)

        # stage 2
        # generate pos index of each node
        pos = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        # split x into grid and adaptive nodes
        tmp = pos.reshape([H, W])
        grid_stride = self.grid_stride
        pos_grid = tmp[grid_stride // 2:H:grid_stride, grid_stride // 2:W:grid_stride]
        pos_grid = pos_grid.reshape([-1])
        mask = torch.ones(pos.shape, dtype=torch.bool, device=pos.device)
        mask[pos_grid] = 0
        pos_ada = torch.masked_select(pos, mask)

        x_grid = torch.index_select(x, 1, pos_grid)
        x_ada = torch.index_select(x, 1, pos_ada)
        x = (x_grid, x_ada)

        pos_grid = pos_grid[None, :].repeat([B, 1])
        pos_ada = pos_ada[None, :].repeat([B, 1])
        pos = (pos_grid, pos_ada)

        x, pos = self.down_layers1(x, pos, self.pos_embed2, H, W)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block2:
            x = blk(x, x, H, W)

        # stage 3
        x, pos = self.down_layers2(x, pos, self.pos_embed3, H, W)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block3:
            x = blk(x, x, H, W)

        # stage 4
        x, pos = self.down_layers3(x, pos, self.pos_embed4, H, W)     # down sample
        H, W = H // 2, W // 2
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = (torch.cat((cls_tokens, x[0]), dim=1), x[1])

        for blk in self.block4:
            x = blk(x, x, H, W)
        x_grid = x[0]
        x_grid = self.norm(x_grid)
        return x_grid[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def mypvt2_small(pretrained=False, **kwargs):
    model = MyPVT2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 1, 1, 1], grid_stride=8, **kwargs)
    model.default_cfg = _cfg()

    return model


# For test
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = mypvt2_small().to(device)
    empty_input = torch.rand([2, 3, 224, 224], device=device)
    output = model(empty_input)
    print('Finish')
