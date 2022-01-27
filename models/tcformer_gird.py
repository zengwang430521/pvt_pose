import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .pvt_v2 import (Block, DropPath, DWConv, OverlapPatchEmbed,
                    to_2tuple, trunc_normal_, register_model, _cfg,
                     cfg, build_smpl_head)
from .utils_mine import (
    get_grid_loc,
    gumble_top_k, index_points,
    map2token_agg_fast_nearest,  # map2token_agg_mat_nearest, map2token_agg_sparse_nearest
    show_tokens_merge, show_conf_merge, merge_tokens, merge_tokens_agg_dist, token2map_agg_mat,
    tokenconv_sparse, token_remerge_part, token_cluster_grid
    # farthest_point_sample
)

from .utils_mine import token_cluster_density_fixbug as token_cluster_density
# from utils_mine import token2map_agg_sparse as token2map_agg_mat

vis = False
# vis = True





'''
do not select tokens, merge tokens. weight NOT clamp, conf do not clamp
merge feature, but not merge locs, reserve all locs.
inherit weights when map2token, which can regarded as tokens merge
farthest_point_sample DOWN, N_grid = 0, feature distance merge
token2map nearest + skip token conv (this must be used together.)
try to make it faster

dist_assign, No ada dc
'''




class MyMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MyDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
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

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, loc_orig, idx_agg, agg_weight, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MyDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv_skip = nn.Conv1d(dim, dim, 1, bias=False, groups=dim)

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        B, N, C = x.shape
        x_map, _ = token2map_agg_mat(x, None, loc_orig, idx_agg, [H, W])
        x_map = self.dwconv(x_map)
        x = map2token_agg_fast_nearest(x_map, N, loc_orig, idx_agg, agg_weight) + \
            self.dwconv_skip(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        return x


class MyAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
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

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
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

    def forward(self, x, loc_orig, x_source, idx_agg_source, H, W, conf_source=None):
        B, N, C = x.shape
        Ns = x_source.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if not self.linear:
            if self.sr_ratio > 1:
                if conf_source is None:
                    conf_source = x_source.new_zeros(B, Ns, 1)
                tmp = torch.cat([x_source, conf_source], dim=-1)
                tmp, _ = token2map_agg_mat(tmp, None, loc_orig, idx_agg_source, [H, W])
                x_source = tmp[:, :C]
                conf_source = tmp[:, C:]

                x_source = self.sr(x_source)
                _, _, h, w = x_source.shape
                x_source = x_source.reshape(B, C, -1).permute(0, 2, 1).contiguous()
                x_source = self.norm(x_source)
                conf_source = F.avg_pool2d(conf_source, kernel_size=self.sr_ratio, stride=self.sr_ratio)
                conf_source = conf_source.reshape(B, 1, -1).permute(0, 2, 1).contiguous()

        else:
            print('error!')

        kv = self.kv(x_source).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
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


class MyBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MyAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

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



# from partialconv2d import PartialConv2d
class DownLayer(nn.Module):
    """ Down sample
    """

    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate, down_block,
                 k=5, dist_assign=True, ada_dc=False, use_conf=False, conf_scale=0.25, conf_density=False):
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
        self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=2, padding=1)
        self.conv_skip = nn.Linear(embed_dim, dim_out, bias=False)
        # self.conv = PartialConv2d(embed_dim, self.block.dim_out, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.conf = nn.Linear(self.dim_out, 1)

        # for density clustering
        self.k = k
        self.dist_assign = dist_assign
        self.ada_dc = ada_dc
        self.use_conf = use_conf
        self.conf_scale = conf_scale
        self.conf_density = conf_density

    def forward(self, x, pos_orig, idx_agg, agg_weight, H, W, N_grid, nh_list, nw_list, lv):
        # x, mask = token2map(x, pos, [H, W], 1, 2, return_mask=True)
        # x = self.conv(x, mask)

        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        if N0 == N and N == H * W:
            x_map = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:
            x_map, _ = token2map_agg_mat(x, None, pos_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token_agg_fast_nearest(x_map, N, pos_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        input_dict = {'x': x,
                      'loc_orig': pos_orig,
                      'idx_agg': idx_agg,
                      'agg_weight': agg_weight,
                      'map_size': [H, W]}

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio) - N_grid, 0)
        if sample_num < N_grid:
            sample_num = N_grid

        # x_down, idx_agg_down, agg_weight_down = token_remerge_part(
        #     input_dict, Ns=sample_num, weight=weight, k=self.k,
        #     nh_list=nh_list, nw_list=nw_list, level=lv,
        #     output_tokens=True, first_cluster=lv == 1,
        #     ignore_density=False)

        x_down, idx_agg_down, weight_t = token_cluster_grid(
            input_dict, Ns=sample_num, conf=None, weight=weight, k=self.k)

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]



        x_down = self.block(x_down, idx_agg_down, agg_weight_down, pos_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        if vis:
            # show_conf_merge(conf, None, pos_orig, idx_agg)
            pass
        return x_down, idx_agg_down, agg_weight_down


class MyPVT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False,
                 k=5, dist_assign=True, ada_dc=False, use_conf=False, conf_scale=0.25, conf_density=False,
                 pretrained=None,
                 nh_list=[1,1,1,1], nw_list=[1,1,1,1],
                 **kwargs
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.nh_list = nh_list
        self.nw_list = nw_list

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(1):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        for i in range(1, num_stages):
            down_layers = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[i-1], dim_out=embed_dims[i],
                                    drop_rate=drop_rate,
                                    down_block=MyBlock(
                                        dim=embed_dims[i], num_heads=num_heads[i],
                                        mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                        norm_layer=norm_layer, sr_ratio=sr_ratios[i], linear=linear),
                                    k=k, dist_assign=dist_assign, ada_dc=ada_dc, use_conf=use_conf,
                                    conf_scale=conf_scale, conf_density=conf_density
                                    )
            block = nn.ModuleList([MyBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(1, depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"down_layers{i}", down_layers)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.head_type = kwargs['head_type'] if 'head_type' in kwargs else 'hmr'
        self.head = build_smpl_head(embed_dims, self.head_type)
        self.return_list = 'att' in self.head_type
        self.batch_count = 0
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
            img = x
        outs = []

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
        N_grid = 0
        idx_agg = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        loc_orig = get_grid_loc(B, H, W, x.device)

        outs.append((x, None, [H, W], loc_orig, idx_agg, agg_weight))

        for i in range(1, self.num_stages):
            down_layers = getattr(self, f"down_layers{i}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, idx_agg, agg_weight = down_layers(x, loc_orig, idx_agg, agg_weight, H, W, N_grid,
                                                 self.nh_list, self.nw_list, i)  # down sample
            H, W = H // 2, W // 2

            for j, blk in enumerate(block):
                x = blk(x, idx_agg, agg_weight, loc_orig, x, idx_agg, agg_weight, H, W, conf_source=None)

            x = norm(x)
            outs.append((x, None, [H, W], loc_orig, idx_agg, agg_weight))

        if vis:
            show_tokens_merge(img, outs, N_grid, self.batch_count)

        if self.return_list:
            return outs
        else:
            return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x




@register_model
def tcformer_grid_small(pretrained=False, **kwargs):
    model = MyPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        k=5, dist_assign=True, ada_dc=False, use_conf=False, conf_scale=0,
        **kwargs)
    model.default_cfg = _cfg()
    return model


