import torch
import torch.nn as nn
import math
from .tcformer_utils import (
    token2map, map2token,
    token_cluster_merge, token_cluster_hir, token_cluster_dpc_hir, token_cluster_lsh,
    token_cluster_app, token_cluster_app2, token_cluster_app3, token_cluster_near,
    token_cluster_nms, token_cluster_grid, token_cluster_part, token_cluster_part_pad,
    token_cluster_part_follow
    )

# CTM block
class CTM(nn.Module):
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

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        x_down, idx_agg_down, weight_t = token_cluster_merge(
            x, sample_num, idx_agg, weight, True, k=self.k
        )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down


class CTM_hir(nn.Module):
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

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        x_down, idx_agg_down, weight_t = token_cluster_hir(
            x, sample_num, idx_agg, conf, weight=weight, return_weight=True
        )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down


class CTM_dpchir(nn.Module):
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

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        x_down, idx_agg_down, weight_t = token_cluster_dpc_hir(
            x, sample_num, idx_agg, weight=weight, return_weight=True, k=self.k,
        )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down


class CTM_lsh(nn.Module):
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

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        x_down, idx_agg_down, weight_t = token_cluster_lsh(
            x, sample_num, idx_agg, weight=weight, return_weight=True, k=self.k,
        )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down


class CTM_app(nn.Module):
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

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        # # ONLY FOR DEBUG
        # x_down, idx_agg_down, weight_t = token_cluster_merge(
        #     x, sample_num, idx_agg, weight, True, k=self.k
        # )

        input_dict = {'x': x,
                      'idx_agg': idx_agg,
                      'agg_weight': agg_weight,
                      'loc_orig': loc_orig,
                      'map_size': [H*2, W*2]}

        x_down, idx_agg_down, weight_t = token_cluster_app(
            input_dict, sample_num, weight=weight, return_weight=True, k=self.k,
        )


        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down


class CTM_app2(nn.Module):
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

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W, idx_k_loc):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        # # # ONLY FOR DEBUG
        # x_down, idx_agg_down, weight_t = token_cluster_merge(
        #     x, sample_num, idx_agg, weight, True, k=self.k
        # )

        if N <= 256:
            x_down, idx_agg_down, weight_t = token_cluster_merge(
                x, sample_num, idx_agg, weight, True, k=self.k
            )
            idx_k_loc_down = None
        else:
            input_dict = {'x': x,
                          'idx_agg': idx_agg,
                          'agg_weight': agg_weight,
                          'loc_orig': loc_orig,
                          'map_size': [H*2, W*2],
                          'idx_k_loc': idx_k_loc}

            x_down, idx_agg_down, weight_t, idx_k_loc_down = token_cluster_app2(
                input_dict, sample_num, weight=weight, k=self.k,
            )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down, idx_k_loc_down


class CTM_app3(nn.Module):
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

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W, idx_k_loc):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        # # # ONLY FOR DEBUG
        # x_down, idx_agg_down, weight_t = token_cluster_merge(
        #     x, sample_num, idx_agg, weight, True, k=self.k
        # )

        if N <= 256:
            x_down, idx_agg_down, weight_t = token_cluster_merge(
                x, sample_num, idx_agg, weight, True, k=self.k
            )
            idx_k_loc_down = None
        else:
            input_dict = {'x': x,
                          'idx_agg': idx_agg,
                          'agg_weight': agg_weight,
                          'loc_orig': loc_orig,
                          'map_size': [H*2, W*2],
                          'idx_k_loc': idx_k_loc}

            x_down, idx_agg_down, weight_t, idx_k_loc_down = token_cluster_app3(
                input_dict, sample_num, weight=weight, k=self.k,
            )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down, idx_k_loc_down


class CTM_app2a(nn.Module):
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

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W, idx_k_loc):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        # # # ONLY FOR DEBUG
        # x_down, idx_agg_down, weight_t = token_cluster_merge(
        #     x, sample_num, idx_agg, weight, True, k=self.k
        # )

        input_dict = {'x': x,
                      'idx_agg': idx_agg,
                      'agg_weight': agg_weight,
                      'loc_orig': loc_orig,
                      'map_size': [H*2, W*2],
                      'idx_k_loc': idx_k_loc}

        x_down, idx_agg_down, weight_t, idx_k_loc_down = token_cluster_app2(
            input_dict, sample_num, weight=weight, k=self.k,
        )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down, idx_k_loc_down


class CTM_near(CTM):
    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W, idx_k_loc):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        # # # ONLY FOR DEBUG
        # x_down, idx_agg_down, weight_t = token_cluster_merge(
        #     x, sample_num, idx_agg, weight, True, k=self.k
        # )

        input_dict = {'x': x,
                      'idx_agg': idx_agg,
                      'agg_weight': agg_weight,
                      'loc_orig': loc_orig,
                      'map_size': [H*2, W*2],
                      'idx_k_loc': idx_k_loc}

        x_down, idx_agg_down, weight_t, idx_k_loc_down = token_cluster_near(
            input_dict, sample_num, weight=weight, k=self.k,
        )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down, idx_k_loc_down


class CTM_nms(CTM):
    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W, idx_k_loc):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        # # # ONLY FOR DEBUG
        # x_down, idx_agg_down, weight_t = token_cluster_merge(
        #     x, sample_num, idx_agg, weight, True, k=self.k
        # )
        if N <= 256:
            x_down, idx_agg_down, weight_t = token_cluster_merge(
                x, sample_num, idx_agg, weight, True, k=self.k
            )
            idx_k_loc_down = None
        else:
            input_dict = {'x': x,
                          'idx_agg': idx_agg,
                          'agg_weight': agg_weight,
                          'loc_orig': loc_orig,
                          'map_size': [H*2, W*2],
                          'idx_k_loc': idx_k_loc}

            x_down, idx_agg_down, weight_t, idx_k_loc_down = token_cluster_nms(
                input_dict, sample_num, weight=weight, k=self.k, conf=conf
            )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down, idx_k_loc_down


class CTM_grid(CTM):
    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W, idx_k_loc, use_grid=False):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        # # # ONLY FOR DEBUG
        # x_down, idx_agg_down, weight_t = token_cluster_merge(
        #     x, sample_num, idx_agg, weight, True, k=self.k
        # )
        if not use_grid:
            x_down, idx_agg_down, weight_t = token_cluster_merge(
                x, sample_num, idx_agg, weight, True, k=self.k
            )
            idx_k_loc_down = None
        else:
            input_dict = {'x': x,
                          'idx_agg': idx_agg,
                          'agg_weight': agg_weight,
                          'loc_orig': loc_orig,
                          'map_size': [H*2, W*2],
                          'idx_k_loc': idx_k_loc}

            x_down, idx_agg_down, weight_t, idx_k_loc_down = token_cluster_grid(
                input_dict, sample_num, weight=weight, k=self.k, conf=conf
            )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down, idx_k_loc_down


class CTM_part(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate, down_block,
                 k=5, dist_assign=True, ada_dc=False, use_conf=False, conf_scale=0.25,
                 conf_density=False, part_ratio=1,
                 ):
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
        self.part_ratio = part_ratio

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W, idx_k_loc):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)
        input_dict = {'x': x,
                      'idx_agg': idx_agg,
                      'agg_weight': agg_weight,
                      'loc_orig': loc_orig,
                      'map_size': [H*2, W*2],
                      'idx_k_loc': idx_k_loc}
        h, w = H * 2 // self.part_ratio, W * 2 // self.part_ratio

        x_down, idx_agg_down, weight_t = token_cluster_part(
            input_dict, sample_num, weight=weight, k=self.k, h=h, w=w
        )
        idx_k_loc_down = None

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down, idx_k_loc_down

# part wise merge with padding
class CTM_partpad(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate, down_block,
                 k=5, dist_assign=True, ada_dc=False, use_conf=False, conf_scale=0.25,
                 conf_density=False, nh=1, nw=1, nh_list=None, nw_list=None,
                 use_agg_weight=True, agg_weight_detach=False,
                 ):
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

        # for partwise
        self.nh = nh
        self.nw = nw or nh
        self.nh_list = nh_list
        self.nw_list = nw_list or nh_list
        self.use_agg_weight = use_agg_weight
        self.agg_weight_detach = agg_weight_detach

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W, idx_k_loc):
        # do not use any agg weight to save memory
        if not self.use_agg_weight:
            agg_weight = None

        if agg_weight is not None and self.agg_weight_detach:
            agg_weight = agg_weight.detach()

        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        B, N, C = x.shape
        input_dict = {'x': x,
                      'idx_agg': idx_agg,
                      'agg_weight': agg_weight,
                      'loc_orig': loc_orig,
                      'map_size': [H, W],
                      'idx_k_loc': idx_k_loc}
        sample_num = max(math.ceil(N * self.sample_ratio), 1)
        nh, nw = self.nh, self.nw
        num_part = nh * nw
        sample_num = round(sample_num // num_part) * num_part

        if self.nh_list is not None and self.nw_list is not None:
            x_down, idx_agg_down, weight_t = token_cluster_part_pad(
                input_dict, sample_num, weight=weight, k=self.k,
                nh_list=self.nh_list, nw_list=self.nw_list
            )
        else:
            x_down, idx_agg_down, weight_t = token_cluster_part_follow(
                input_dict, sample_num, weight=weight, k=self.k, nh=nh, nw=nw
            )
        idx_k_loc_down = None

        if agg_weight is not None:
            agg_weight_down = agg_weight * weight_t
            agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]
            if self.agg_weight_detach:
                agg_weight_down = agg_weight_down.detach()
        else:
            agg_weight_down = None

        _, _, H, W = x_map.shape
        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down, idx_k_loc_down


