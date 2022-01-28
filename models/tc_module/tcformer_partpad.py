import torch
import torch.nn as nn
from functools import partial
import math
from .tc_layers import Block, TCBlock, OverlapPatchEmbed
from .tcformer_utils import (
    get_grid_loc, show_tokens_merge, get_initial_loc_neighbor,
    DPC_flops, token2map_flops, map2token_flops, sra_flops,
    load_checkpoint, get_root_logger)
from .transformer_utils import trunc_normal_
# from .ctm_block import CTM as CTM
from .ctm_block import CTM_partpad as CTM
from mmpose.models.builder import BACKBONES
from mmpose.utils import get_root_logger
# from mmcv.runner import load_checkpoint


vis = False

'''
partwise token merge with padding
'''

class TCFormer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4,
                 k=5, pretrained=None, nh_list=[4, 2, 1], nw_list=[4, 2, 1],
                 use_agg_weight=True, agg_weight_detach=False,
                 **kwargs
                 ):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.depths = depths
        self.sample_ratio = 0.25
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.nh_list = nh_list
        self.nw_list = nw_list

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        # In stage 1, use the standard transformer blocks
        for i in range(1):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # In stage 2~4, use TCBlock for dynamic tokens
        for i in range(1, num_stages):
            ctm = CTM(sample_ratio=0.25, embed_dim=embed_dims[i-1], dim_out=embed_dims[i],
                      drop_rate=drop_rate, k=k,
                      nh=self.nh_list[i-1], nw=self.nw_list[i-1],
                      nh_list=self.nh_list if i == 1 else None,
                      nw_list=self.nw_list if i == 1 else None,
                      use_agg_weight=use_agg_weight, agg_weight_detach=agg_weight_detach,
                      down_block=TCBlock(
                            dim=embed_dims[i], num_heads=num_heads[i],
                            mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                            norm_layer=norm_layer, sr_ratio=sr_ratios[i]))

            # Because CTM contains a transformer block, we reduce the depth in the next stage by 1.
            # The transformer block in CTM can be regarded as the first block of the next stage.
            block = nn.ModuleList([TCBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(1, depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"ctm{i}", ctm)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.init_weights(pretrained)
        self.count = 0

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        outs = []
        img = x
        # stage 1
        i = 0
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        x, H, W = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)

        B, N, _ = x.shape
        device = x.device
        idx_agg = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        loc_orig = get_grid_loc(B, H, W, x.device)
        idx_k_loc = get_initial_loc_neighbor(H, W, x.device)
        outs.append({'x': x,
                     'map_size': [H, W],
                     'loc_orig': loc_orig,
                     'idx_agg': idx_agg,
                     'agg_weight': agg_weight})

        for i in range(1, self.num_stages):
            ctm = getattr(self, f"ctm{i}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, idx_agg, agg_weight, idx_k_loc = ctm(x, loc_orig, idx_agg, agg_weight, H, W, idx_k_loc)  # down sample
            H, W = H // 2, W // 2
            for j, blk in enumerate(block):
                x = blk(x, idx_agg, agg_weight, loc_orig, x, idx_agg, agg_weight, H, W, conf_source=None)

            x = norm(x)
            outs.append({'x': x,
                         'map_size': [H, W],
                         'loc_orig': loc_orig,
                         'idx_agg': idx_agg,
                         'agg_weight': agg_weight})

        if vis:
            show_tokens_merge(img, outs, self.count)
            self.count += 1

        return outs
        # return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def get_extra_flops(self, H, W):
        flops = 0
        h, w = H // 4, W // 4
        N0 = h * w
        N = N0
        for stage in range(4):
            depth, sr, dim = self.depths[stage], self.sr_ratios[stage], self.embed_dims[stage]
            mlp_r = self.mlp_ratios[stage]
            dim_up = self.embed_dims[stage-1]

            if stage > 0:
                # cluster flops
                flops += DPC_flops(N, dim)
                flops += map2token_flops(N0, dim_up) + token2map_flops(N0, dim)
                N = N * self.sample_ratio
                h, w = h // 2, w // 2

            # attn flops
            flops += sra_flops(h, w, sr, dim) * depth

            if stage > 0:
                # map, token flops
                flops += (map2token_flops(N0, dim) + map2token_flops(N0, dim * mlp_r) + token2map_flops(N0, dim * mlp_r)) * depth

        return flops


@BACKBONES.register_module()
class tcformer_partpad_light(TCFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            k=5, **kwargs)

@BACKBONES.register_module()
class tcformer_partpad_small(TCFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            nh_list=[4, 2, 1], nw_list=[4, 2, 1], use_agg_weight=True, agg_weight_detach=True,
            k=5, **kwargs)

@BACKBONES.register_module()
class tcformer_partpad_small2(TCFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            nh_list=[8, 4, 2], nw_list=[8, 4, 2], use_agg_weight=True, agg_weight_detach=True,
            k=5, **kwargs)


@BACKBONES.register_module()
class tcformer_partpad_large(TCFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            k=5, **kwargs)
