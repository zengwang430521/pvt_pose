import sys
sys.path.insert(0, 'index_process')
import torch
# from torch_sparse import spmm, coalesce
from mmcv.utils import get_logger
from mmcv.runner import _load_checkpoint, load_state_dict
import logging
import re
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
try:
    from torch_cluster import fps
except:
    print('no torch_cluster')
# try:
#     from function import f_distance
# except:
#     print('no f_distance')


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].


    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}
    # load state_dict
    _ = load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger


def get_grid_loc(B, H, W, device):
    y_g, x_g = torch.arange(H, device=device).float(), torch.arange(W, device=device).float()
    y_g = 2 * ((y_g + 0.5) / H) - 1
    x_g = 2 * ((x_g + 0.5) / W) - 1
    y_map, x_map = torch.meshgrid(y_g, x_g)
    xy_map = torch.stack((x_map, y_map), dim=-1)

    loc = xy_map.reshape(-1, 2)[None, ...].repeat([B, 1, 1])
    return loc


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



def token2map(x, loc, loc_orig, idx_agg, map_size, weight=None):
    H, W = map_size
    B, N, C = x.shape
    N0 = loc_orig.shape[1]
    if N0 < N * H * W:
        return token2map_sparse2(x, loc, loc_orig, idx_agg, map_size, weight)
    else:
        return token2map_dense(x, loc, loc_orig, idx_agg, map_size, weight)


def map2token(feature_map, N, loc_orig, idx_agg, agg_weight=None):
    dtype = feature_map.dtype
    B, C, H, W = feature_map.shape
    N0 = loc_orig.shape[1]
    if N0 < N * H * W:
        return map2token_sparse2(feature_map, N, loc_orig, idx_agg, agg_weight)
    else:
        return map2token_dense(feature_map, N, loc_orig, idx_agg, agg_weight)


def token_downup(target_dict, source_dict):
    x_s = source_dict['x']
    x_t = target_dict['x']
    idx_agg_s = source_dict['idx_agg']
    B, T, C = x_t.shape
    B, S, C = x_s.shape
    N0 = idx_agg_s.shape[1]
    if N0 < T * S:
        return token_downup_sparse2(target_dict, source_dict)
    else:
        return token_downup_dense(target_dict, source_dict)




# use torch_sparse
def token2map_sparse1(x, loc, loc_orig, idx_agg, map_size, weight=None):
    H, W = map_size
    B, N, C = x.shape
    N0 = loc_orig.shape[1]
    device = x.device
    dtype = x.dtype
    if N0 == N and N == H * W:
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous(), None

    loc_orig = loc_orig.clamp(-1, 1)
    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    loc_orig = loc_orig.round().long()
    loc_orig[..., 0] = loc_orig[..., 0].clamp(0, W-1)
    loc_orig[..., 1] = loc_orig[..., 1].clamp(0, H-1)
    idx_HW_orig = loc_orig[..., 0] + loc_orig[..., 1] * W
    idx_HW_orig = idx_HW_orig + torch.arange(B)[:, None].to(device) * H * W

    idx_tokens = idx_agg + torch.arange(B)[:, None].to(device) * N

    coor = torch.stack([idx_HW_orig, idx_tokens], dim=0).reshape(2, B*N0)
    if weight is None:
        weight = x.new_ones(B, N, 1)
    value = index_points(weight, idx_agg).reshape(B*N0)

    # print('only for debug!')
    value = value.detach()  # to save memory!

    all_weight = spmm(coor, value, B*H*W, B*N, x.new_ones(B*N, 1)) + 1e-6
    value = value / all_weight[idx_HW_orig.reshape(-1), 0]

    x_out = spmm(coor, value, B*H*W, B*N, x.reshape(B*N, C))
    x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    all_weight = all_weight.reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()

    return x_out, all_weight


def map2token_sparse1(feature_map, N, loc_orig, idx_agg, agg_weight=None):

    dtype = feature_map.dtype
    B, C, H, W = feature_map.shape
    device = feature_map.device
    N0 = loc_orig.shape[1]

    if N0 == N and N == H * W:
        return feature_map.flatten(2).permute(0, 2, 1).contiguous()

    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    x = loc_orig[:, :, 0]
    y = loc_orig[:, :, 1]

    h, w = H, W
    x_grid = x.round().long().clamp(min=0, max=w - 1)
    y_grid = y.round().long().clamp(min=0, max=h - 1)
    idx_HW_orig = (y_grid * w + x_grid).detach()
    index_batch = torch.arange(B, device=device)[:, None].expand(B, N0)

    # use sparse matrix
    idx_agg = idx_agg + index_batch * N
    idx_HW_orig = idx_HW_orig + index_batch * H * W

    indices = torch.stack([idx_agg, idx_HW_orig], dim=0).reshape(2, -1)

    if agg_weight is None:
        value = feature_map.new_ones(B * N0)
    else:
        value = agg_weight.reshape(B * N0).type(feature_map.dtype)

    # print('only for debug!')
    value = value.detach()  # to save memory!

    all_weight = spmm(indices, value, B*N, B*H*W, feature_map.new_ones([B*H*W, 1])) + 1e-6
    value = value / all_weight[idx_agg.reshape(-1), 0]
    out = spmm(indices, value, B*N, B*H*W,
               feature_map.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C))
    out = out.reshape(B, N, C)
    return out


def token_downup_sparse1(target_dict, source_dict):
    x_s = source_dict['x']
    x_t = target_dict['x']
    idx_agg_s = source_dict['idx_agg']
    idx_agg_t = target_dict['idx_agg']
    agg_weight_t = target_dict['agg_weight']
    B, T, C = x_t.shape
    B, S, C = x_s.shape
    N0 = idx_agg_s.shape[1]

    idx_agg_t = idx_agg_t + torch.arange(B, device=x_s.device)[:, None] * T
    idx_agg_s = idx_agg_s + torch.arange(B, device=x_s.device)[:, None] * S

    coor = torch.stack([idx_agg_t, idx_agg_s], dim=0).reshape(2, B*N0)
    weight = agg_weight_t
    if weight is None:
        weight = x_s.new_ones(B, N0, 1)
    weight = weight.reshape(-1)
    # print('only for debug!')
    weight = weight.detach()  # to save memory!

    all_weight = spmm(coor, weight, B*T, B*S, x_s.new_ones(B*S, 1)) + 1e-6
    weight = weight / all_weight[(idx_agg_t).reshape(-1), 0]
    x_out = spmm(coor, weight, B*T, B*S, x_s.reshape(B*S, C))
    x_out = x_out.reshape(B, T, C)
    return x_out


# use torch.sparse
def token2map_sparse2(x, loc, loc_orig, idx_agg, map_size, weight=None):
    H, W = map_size
    B, N, C = x.shape
    N0 = loc_orig.shape[1]
    device = x.device
    dtype = x.dtype
    if N0 == N and N == H * W:
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous(), None

    loc_orig = loc_orig.clamp(-1, 1)
    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    loc_orig = loc_orig.round().long()
    loc_orig[..., 0] = loc_orig[..., 0].clamp(0, W-1)
    loc_orig[..., 1] = loc_orig[..., 1].clamp(0, H-1)
    idx_HW_orig = loc_orig[..., 0] + loc_orig[..., 1] * W
    idx_HW_orig = idx_HW_orig + torch.arange(B)[:, None].to(device) * H * W

    idx_tokens = idx_agg + torch.arange(B)[:, None].to(device) * N

    coor = torch.stack([idx_HW_orig, idx_tokens], dim=0).reshape(2, B*N0)
    if weight is None:
        weight = x.new_ones(B, N, 1)
    value = index_points(weight, idx_agg).reshape(B*N0)

    with torch.cuda.amp.autocast(enabled=False):
        value = value.detach().float()
        A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))
        all_weight = A @ x.new_ones(B*N, 1).type(torch.float32) + 1e-6
        value = value / all_weight[idx_HW_orig.reshape(-1), 0]
        A = torch.sparse.FloatTensor(coor, value, torch.Size([B*H*W, B*N]))
        x_out = A @ x.reshape(B*N, C).type(torch.float32)

    x_out = x_out.type(x.dtype)
    x_out = x_out.reshape(B, H, W, C).permute(0, 3,  1, 2).contiguous()
    all_weight = all_weight.reshape(B, H, W, 1).permute(0, 3,  1, 2).contiguous().type(x.dtype)
    return x_out, all_weight


def map2token_sparse2(feature_map, N, loc_orig, idx_agg, agg_weight=None):

    dtype = feature_map.dtype
    B, C, H, W = feature_map.shape
    device = feature_map.device
    N0 = loc_orig.shape[1]

    if N0 == N and N == H * W:
        return feature_map.flatten(2).permute(0, 2, 1).contiguous()

    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    x = loc_orig[:, :, 0]
    y = loc_orig[:, :, 1]

    h, w = H, W
    x_grid = x.round().long().clamp(min=0, max=w - 1)
    y_grid = y.round().long().clamp(min=0, max=h - 1)
    idx_HW_orig = (y_grid * w + x_grid).detach()
    index_batch = torch.arange(B, device=device)[:, None].expand(B, N0)

    # use sparse matrix
    idx_agg = idx_agg + index_batch * N
    idx_HW_orig = idx_HW_orig + index_batch * H * W

    indices = torch.stack([idx_agg, idx_HW_orig], dim=0).reshape(2, -1)

    if agg_weight is None:
        value = feature_map.new_ones(B * N0)
    else:
        value = agg_weight.reshape(B * N0).type(feature_map.dtype)

    with torch.cuda.amp.autocast(enabled=False):
        value = value.detach().float()  # sparse mm do not support gradient for sparse matrix
        A = torch.sparse_coo_tensor(indices, value, (B * N, B *H * W))
        all_weight = A @ torch.ones([B*H*W, 1], device=device, dtype=torch.float32) + 1e-6
        value = value / all_weight[idx_agg.reshape(-1), 0]

        A = torch.sparse_coo_tensor(indices, value, (B * N, B *H * W))
        out = A @ feature_map.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C).float()
        out = out.reshape(B, N, C)
    out = out.type(feature_map.dtype)
    return out


def token_downup_sparse2(target_dict, source_dict):
    x_s = source_dict['x']
    x_t = target_dict['x']
    idx_agg_s = source_dict['idx_agg']
    idx_agg_t = target_dict['idx_agg']
    agg_weight_t = target_dict['agg_weight']
    B, T, C = x_t.shape
    B, S, C = x_s.shape
    N0 = idx_agg_s.shape[1]

    idx_agg_t = idx_agg_t + torch.arange(B, device=x_s.device)[:, None] * T
    idx_agg_s = idx_agg_s + torch.arange(B, device=x_s.device)[:, None] * S

    coor = torch.stack([idx_agg_t, idx_agg_s], dim=0).reshape(2, B*N0)
    weight = agg_weight_t
    if weight is None:
        weight = x_s.new_ones(B, N0, 1)
    weight = weight.reshape(-1)

    with torch.cuda.amp.autocast(enabled=False):
        weight = weight.float().detach()    # sparse mm do not support grad for sparse mat
        A = torch.sparse.FloatTensor(coor, weight, torch.Size([B*T, B*S]))
        all_weight = A.type(torch.float32) @ x_s.new_ones(B*S, 1).type(torch.float32) + 1e-6
        weight = weight / all_weight[(idx_agg_t).reshape(-1), 0]

        A = torch.sparse.FloatTensor(coor, weight, torch.Size([B*T, B*S]))
        x_out = A.type(torch.float32) @ x_s.reshape(B*S, C).type(torch.float32)
        x_out = x_out.reshape(B, T, C).type(x_s.dtype)
    return x_out


# dense matmul
def token2map_dense(x, loc, loc_orig, idx_agg, map_size, weight=None):
    H, W = map_size
    B, N, C = x.shape
    N0 = loc_orig.shape[1]
    device = x.device
    dtype = x.dtype
    if N0 == N and N == H * W:
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous(), None

    loc_orig = loc_orig.clamp(-1, 1)
    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    loc_orig = loc_orig.round().long()
    loc_orig[..., 0] = loc_orig[..., 0].clamp(0, W-1)
    loc_orig[..., 1] = loc_orig[..., 1].clamp(0, H-1)
    idx_HW_orig = loc_orig[..., 0] + loc_orig[..., 1] * W

    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N0)

    coor = torch.stack([idx_batch, idx_HW_orig, idx_agg], dim=0).reshape(3, B*N0)
    if weight is None:
        weight = x.new_ones(B, N, 1)
    value = index_points(weight, idx_agg).reshape(B*N0)
    value = value.detach()

    A = torch.sparse.FloatTensor(coor, value, torch.Size([B, H * W, N])).to_dense()
    all_weight = A @ x.new_ones([B, N, 1]) + 1e-6
    A = A / all_weight
    x_out = A @ x

    x_out = x_out.reshape(B, H, W, C).permute(0, 3,  1, 2).contiguous()
    all_weight = all_weight.reshape(B, H, W, 1).permute(0, 3,  1, 2).contiguous().type(x.dtype)
    return x_out, all_weight


def map2token_dense(feature_map, N, loc_orig, idx_agg, agg_weight=None):

    dtype = feature_map.dtype
    B, C, H, W = feature_map.shape
    device = feature_map.device
    N0 = loc_orig.shape[1]

    if N0 == N and N == H * W:
        return feature_map.flatten(2).permute(0, 2, 1).contiguous()

    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    x = loc_orig[:, :, 0]
    y = loc_orig[:, :, 1]

    h, w = H, W
    x_grid = x.round().long().clamp(min=0, max=w - 1)
    y_grid = y.round().long().clamp(min=0, max=h - 1)
    idx_HW_orig = (y_grid * w + x_grid).detach()
    index_batch = torch.arange(B, device=device)[:, None].expand(B, N0)

    # use sparse matrix

    indices = torch.stack([index_batch, idx_agg, idx_HW_orig], dim=0).reshape(3, -1)

    if agg_weight is None:
        value = feature_map.new_ones(B * N0)
    else:
        value = agg_weight.reshape(B * N0).type(feature_map.dtype)

    value = value.detach()  # sparse mm do not support gradient for sparse matrix
    A = torch.sparse_coo_tensor(indices, value, (B, N, H * W)).to_dense()
    all_weight = A @ feature_map.new_ones([B, H*W, 1]) + 1e-6
    A = A / all_weight
    out = A @ feature_map.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
    out = out.reshape(B, N, C)

    out = out.type(feature_map.dtype)
    return out


def token_downup_dense(target_dict, source_dict):
    x_s = source_dict['x']
    x_t = target_dict['x']
    idx_agg_s = source_dict['idx_agg']
    idx_agg_t = target_dict['idx_agg']
    agg_weight_t = target_dict['agg_weight']
    B, T, C = x_t.shape
    B, S, C = x_s.shape
    N0 = idx_agg_s.shape[1]

    index_batch = torch.arange(B, device=x_s.device)[:, None].expand(B, N0)

    coor = torch.stack([index_batch, idx_agg_t, idx_agg_s], dim=0).reshape(3, B*N0)
    weight = agg_weight_t
    if weight is None:
        weight = x_s.new_ones(B, N0, 1)
    weight = weight.reshape(-1)

    weight = weight.detach()
    A = torch.sparse.FloatTensor(coor, weight, torch.Size([B, T, S])).to_dense()
    all_weight = A.type(torch.float32) @ x_s.new_ones(B, S, 1) + 1e-6
    A = A / all_weight
    x_out = A @ x_s

    x_out = x_out.reshape(B, T, C).type(x_s.dtype)
    return x_out




def DPC_flops(N, C):
    flops = 0
    flops += N * N * C  # dist_matrix
    flops += N * 5  # density
    flops += N * N  # dist indicator
    flops += N * C  # gather
    return flops


def DPC_part_flops(N, Np, C):
    flops = 0

    flops += N * Np * C  # dist_matrix
    flops += N * 5  # density
    flops += N * Np  # dist
    flops += N * C  # gather
    return flops


def map2token_flops(N0, C):
    return N0 * (2 + 1 + 1 + C)


def token2map_flops(N0, C):
    return N0 * (2 + 1 + 1 + C)


def downup_flops(N0, C):
    return N0 * (2 + 1 + 1 + C)


# flops for attention
def sra_flops(h, w, r, dim):
    return 2 * h * w * (h // r) * (w // r) * dim


def gumble_top_k(x, k, dim, T=1, p_value=1e-6):
    # Noise
    noise = torch.rand_like(x)
    noise = -1 * (noise + p_value).log()
    noise = -1 * (noise + p_value).log()
    # add
    x = x / T + noise
    _, index_k = torch.topk(x, k, dim)
    return index_k



# DPC-KNN based token clustering and token feature averaging
def token_cluster_merge(x, Ns, idx_agg, weight=None, return_weight=False, k=5):
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape

    if weight is None:
        weight = x.new_ones(B, N, 1)

    with torch.no_grad():
        dist_matrix = torch.cdist(x, x)
        # normalize dist_matrix for stable
        # dist_matrix = dist_matrix / (dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] + 1e-6)
        dist_matrix = dist_matrix / (C ** 0.5)


        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

        density = density + torch.rand(density.shape, device=device, dtype=density.dtype) * 1e-6


        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)

        # get distance indicator
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1-mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=Ns, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)
        idx_agg_t = dist_matrix.argmin(dim=1)

        # make sure selected centers merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns

    # # only for debug
    # print('for debug only!')
    # loc_orig = get_grid_loc(x.shape[0], 64, 64, x.device)
    # show_conf_merge(density[:, :, None], None, loc_orig, idx_agg, n=1, vmin=None)
    # show_conf_merge(dist[:, :, None], None, loc_orig, idx_agg, n=2, vmin=None)
    # show_conf_merge(score[:, :, None], None, loc_orig, idx_agg, n=3, vmin=None)

    # normalize the weight
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, idx_agg, weight_t
    return x_out, idx_agg


def token_cluster_hir(x, Ns, idx_agg, conf, weight=None, return_weight=False, **kwargs):
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    conf = conf.squeeze(-1)
    if weight is None:
        weight = x.new_ones(B, N, 1)

    with torch.no_grad():
        index_down = gumble_top_k(conf, Ns, dim=1)

        if N <= 256:
            '''nearest assign'''
            centers = index_points(x, index_down)
            dist_matrix = torch.cdist(x, centers)
            idx_agg_t = dist_matrix.argmin(dim=2)
        else:

            Nr = int(math.sqrt(Ns))
            K = int(2 * Ns / Nr)
            index_rough_center = index_down[:, :Nr]

            centers = index_points(x, index_down)
            rough_centers = index_points(x, index_rough_center)

            dist_matrix1 = torch.cdist(rough_centers, centers, p=2)
            _, idx_k_rough = torch.topk(-dist_matrix1, k=K, dim=-1)

            idx_tmp = torch.cdist(x, rough_centers, p=2).argmin(axis=2)
            idx_k = index_points(idx_k_rough, idx_tmp)

            with torch.cuda.amp.autocast(enabled=False):
                '''I only support float, float, int Now'''
                dist_k = f_distance(x.float(), centers.float(), idx_k.int())

            idx_tmp = dist_k.argmin(dim=2)
            idx_agg_t = torch.gather(idx_k, -1, idx_tmp[:,:, None])
            idx_agg_t = idx_agg_t.squeeze(-1)

        # make sure selected tokens merge to itself
        if index_down is not None:
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
            idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
            idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns


    # # # for debug only
    # loc_orig = get_grid_loc(x.shape[0], 56, 56, x.device)
    # show_conf_merge(density[:, :, None], None, loc_orig, idx_agg, n=1, vmin=None)
    # show_conf_merge(dist[:, :, None], None, loc_orig, idx_agg, n=2, vmin=None)
    # show_conf_merge(score[:, :, None], None, loc_orig, idx_agg, n=3, vmin=None)
    # show_conf_merge(conf[:, :, None], None, loc_orig, idx_agg, n=4, vmin=None)
    # if use_conf:
    #     show_conf_merge(score_log[:, :, None], None, loc_orig, idx_agg, n=5)


    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, idx_agg, weight_t
    return x_out, idx_agg


# use dpc to determine center and use hir for cluster
# just for comparison
def token_cluster_dpc_hir(x, Ns, idx_agg, weight=None, return_weight=False, k=5):
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape

    if weight is None:
        weight = x.new_ones(B, N, 1)

    with torch.no_grad():
        dist_matrix = torch.cdist(x, x)
        # normalize dist_matrix for stable
        dist_matrix = dist_matrix / (dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] + 1e-6)

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)

        # get distance indicator
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1-mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=Ns, dim=-1)

        # if N <= 256:
        if N < 0:
            # assign tokens to the nearest center
            dist_matrix = index_points(dist_matrix, index_down)
            idx_agg_t = dist_matrix.argmin(dim=1)
        else:
            # assign tokens to the nearest center use hir way
            Nr = int(math.sqrt(Ns))
            K = int(2 * Ns / Nr)
            index_rough_center = index_down[:, :Nr]

            centers = index_points(x, index_down)
            rough_centers = index_points(x, index_rough_center)

            dist_matrix1 = torch.cdist(rough_centers, centers, p=2)
            _, idx_k_rough = torch.topk(-dist_matrix1, k=K, dim=-1)

            idx_tmp = torch.cdist(x, rough_centers, p=2).argmin(axis=2)
            idx_k = index_points(idx_k_rough, idx_tmp)

            with torch.cuda.amp.autocast(enabled=False):
                '''I only support float, float, int Now'''
                dist_k = f_distance(x.float(), centers.float(), idx_k.int())

            idx_tmp = dist_k.argmin(dim=2)
            idx_agg_t = torch.gather(idx_k, -1, idx_tmp[:, :, None])
            idx_agg_t = idx_agg_t.squeeze(-1)


        # make sure selected centers merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns


    # normalize the weight
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, idx_agg, weight_t
    return x_out, idx_agg


def token_cluster_lsh(x, Ns, idx_agg, weight=None, return_weight=False,  **kwargs):
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    if weight is None:
        weight = x.new_ones(B, N, 1)

    Nbit = math.ceil(math.log2(Ns))
    Ns = 2**Nbit
    with torch.no_grad():
        weight_proj = torch.rand([C, Nbit], dtype=dtype, device=device)
        x_proj = torch.matmul((x - x.mean(dim=-1, keepdim=True)), weight_proj)
        tmp = 2**torch.arange(Nbit, device=device)
        idx_agg_t = ((x_proj > 0) * tmp[None, None, :]).sum(dim=-1)

        index_down = None
        # make sure selected tokens merge to itself
        if index_down is not None:
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
            idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
            idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns

    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, idx_agg, weight_t
    return x_out, idx_agg


def show_tokens_merge(x, out, count=0):
    # import matplotlib.pyplot as plt
    IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
    x = x * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
    save_x = False
    save_img = False
    save_fig = True

    if save_x:
        save_dict = {
            'x': x,
            'out': out
        }
        fname = f'vis/{count}.pth'
        torch.save(save_dict, fname)

    B, _, h, w = x.shape
    h, w = h // 4, w//4
    device = x.device
    color_map = F.avg_pool2d(x, kernel_size=4)


    N0 = h*w
    # num_fig = len(out) + 2
    num_fig = 6

    for i in range(1):
        img = x[i].permute(1, 2, 0).detach().cpu()
        ax = plt.subplot(1, num_fig, 1)

        ax.clear()
        ax.imshow(img)
        plt.axis('off')

        if save_img:
            fname = f'vis/{count}_img.png'
            import cv2
            cv2.imwrite(fname, img.numpy()[:, :, ::-1] * 255)

        lv = len(out) - 1

        x = out[lv]['x']
        idx_agg = out[lv]['idx_agg']
        loc_orig = out[lv]['loc_orig']

        B, N, _ = x.shape

        # tmp = torch.arange(N, device=x.device)[None, :, None].expand(B, N, 1).float()
        tmp = torch.rand([N, 3], device=x.device)[None, :, :].expand(B, N, 3).float()
        H, W, _ = img.shape
        idx_map, _ = token2map(tmp, loc_orig, loc_orig, idx_agg, [H // 4, W // 4])
        idx_map = F.interpolate(idx_map, [H, W], mode='nearest')
        # idx_map = idx_map[0].permute(1, 2, 0).detach().cpu().float()
        ax = plt.subplot(1, num_fig, num_fig)
        ax.imshow(idx_map[0].permute(1, 2, 0).detach().cpu().float())
        plt.axis('off')

        for lv in range(len(out)):

            x = out[lv]['x']
            idx_agg = out[lv]['idx_agg']
            loc_orig = out[lv]['loc_orig']
            agg_weight = out[lv]['agg_weight']
            B, N, _ = x.shape

            token_c = map2token(color_map, N, loc_orig, idx_agg, agg_weight)
            # token_c = torch.rand([N, 3], device=x.device)[None, :, :].expand(B, N, 3).float()

            idx_map, _ = token2map(token_c, loc_orig, loc_orig, idx_agg, [H // 4, W // 4])
            idx_map_grid = F.avg_pool2d(color_map, kernel_size=2**lv)

            idx_map_our = idx_map
            idx_map_our = F.interpolate(idx_map, [H*4, W*4], mode='nearest')
            idx_map_grid = F.interpolate(idx_map_grid, [H * 4, W * 4], mode='nearest')

            sharpen = torch.FloatTensor([   [0, -1, 0],
                                            [-1, 4, -1],
                                            [0, -1, 0]])
            sharpen = sharpen[None, None, :, :].to(idx_map.device).expand([3,1,3,3])

            mask_our = F.conv2d(F.pad(idx_map_our, [1, 1, 1, 1], mode='replicate'), sharpen, groups=3)
            mask_grid = F.conv2d(F.pad(idx_map_grid, [1, 1, 1, 1], mode='replicate'), sharpen, groups=3)

            mask_our = (mask_our.abs() > 0).float()
            mask_grid = (mask_grid.abs() > 0).float()
            # for t in range(lv - 1):
            for t in range(1):
                kernel = torch.FloatTensor([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]])
                kernel = kernel[None, None, :, :].to(idx_map.device).expand([3, 1, 3, 3])

                mask_our = F.conv2d(F.pad(mask_our, [1, 1, 1, 1], mode='replicate'), kernel, groups=3)
                mask_grid = F.conv2d(F.pad(mask_grid, [1, 1, 1, 1], mode='replicate'), kernel, groups=3)

            idx_map_our = (idx_map_our + mask_our * 10).clamp(0, 1)
            idx_map_grid = (idx_map_grid + mask_grid * 10).clamp(0, 1)

            if save_img:
                fname = f'vis/{count}_{lv}.png'
                import cv2
                cv2.imwrite(fname, idx_map_our[0].permute(1, 2, 0).detach().cpu().float().numpy()[:, :, ::-1] * 255)

                fname = f'vis/{count}_{lv}_grid.png'
                import cv2
                cv2.imwrite(fname, idx_map_grid[0].permute(1, 2, 0).detach().cpu().float().numpy()[:, :, ::-1] * 255)

            ax = plt.subplot(1, num_fig, lv+2)
            ax.clear()
            ax.imshow(idx_map_our[0].permute(1, 2, 0).detach().cpu().float())
            plt.axis('off')

    # plt.show()
    if save_fig:
        fname = f'vis/{count}.jpg'
        plt.savefig(fname, dpi=400)

    return


def show_conf_merge(conf, loc, loc_orig, idx_agg, l=2, c=5, n=0, vmin=0, vmax=7):
    H0 = 32
    H = int(conf.shape[1]**0.5)
    if n <= 0:
        n = int(math.log2(H0 / H) + 7 + 0)

    # conf = F.softmax(conf, dim=1)
    # conf = conf.exp()
    # conf = conf - conf.min(dim=1, keepdim=True)[0]
    conf_map, _ = token2map(conf, loc, loc_orig, idx_agg, [H0, H0])
    ax = plt.subplot(l, c, n)
    ax.clear()
    if vmax is not None and vmin is not None:
        ax.imshow(conf_map[0, 0].detach().cpu().float(), vmin=vmin, vmax=vmax)
    else:
        ax.imshow(conf_map[0, 0].detach().cpu().float())
    # plt.colorbar()


# approximate DPC-KNN
def token_cluster_app(input_dict, Ns, weight=None, return_weight=False, k=5):
    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    loc_orig = input_dict['loc_orig']
    H, W = input_dict['map_size']

    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    N0 = idx_agg.shape[1]

    if weight is None:
        weight = x.new_ones(B, N, 1)

    with torch.no_grad():
        if agg_weight is None:
            agg_weight = x.new_ones(B, N0, 1)
        scale_factor = N ** 0.25
        Nr = int(math.sqrt(Ns))
        K = max(2 * Nr, k)
        h, w = int(round(H/scale_factor)), int(round(W/scale_factor))
        x_rough, _ = token2map(x, None, loc_orig, idx_agg, [h, w], weight=agg_weight)

        # # just for debug
        # x_re = map2token(x_rough, N, loc_orig, idx_agg, agg_weight)
        # dist_debug = torch.cdist(x, x_re)
        # plt.imshow(dist_debug[0].detach().float().cpu())
        # err = torch.norm(x-x_re, p=2, dim=-1, keepdim=True)
        # err_map, _ = token2map(err, None, loc_orig, idx_agg, [H, W], weight=agg_weight)
        # plt.imshow(err_map[0, 0].detach().float().cpu())

        x_rough = x_rough.flatten(2).permute(0, 2, 1)
        dist_matrix1 = torch.cdist(x, x_rough)

        _, idx_k_rough = torch.topk(-dist_matrix1, k=K, dim=1)      # nearest tokens for every rough token
        idx_k_rough = idx_k_rough.permute(0, 2, 1)
        idx_tmp = dist_matrix1.argmin(axis=2)                       # nearest rough token for each token
        idx_k = index_points(idx_k_rough, idx_tmp)                  # approximate nearest tokens for each token

        with torch.cuda.amp.autocast(enabled=False):
            '''I only support float, float, int Now'''
            dist_k = f_distance(
                x.float().contiguous(),
                x.float().contiguous(),
                idx_k.int().contiguous())

        # get local density
        dist_k = dist_k.type(dist_matrix1.dtype)
        dist_matrix = torch.cat([dist_matrix1, dist_k], dim=-1)
        dist_matrix = dist_matrix / (C ** 0.5)

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

        # # distance indicator, JUST FOR DEBUG
        # mask = density[:, None, :] > density[:, :, None]
        # dist_matrix = torch.cdist(x, x)
        # mask = mask.type(x.dtype)
        # dist, index_parent = (dist_matrix * mask +
        #                       dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)

        # distance indicator
        density_rough, _ = token2map(density[:, :, None], None, loc_orig, idx_agg, [h, w], weight=agg_weight)
        density_rough = density_rough.flatten(2).permute(0, 2, 1)
        density_rough = density_rough[:, None, :, 0].expand(-1, N, -1)
        density_k = index_points(density[:, :, None], idx_k).squeeze(-1)
        density_matrix = torch.cat([density_rough, density_k], dim=-1)
        mask = density_matrix > density[:, :, None]
        # idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, N)
        # mask[idx_batch.reshape(-1), idx_tmp.reshape(-1)] = False
        mask = mask.type(x.dtype)

        # just for debug
        # dist_matrix = dist_matrix[:, :, h*w:]
        # mask = mask[:, :, h*w:]
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=Ns, dim=-1)

        # grouping
        # assign tokens to the nearest center
        centers = index_points(x, index_down)
        dist_matrix = torch.cdist(centers, x)
        idx_agg_t = dist_matrix.argmin(dim=1)

        # make sure selected centers merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns

    # # # for debug only
    loc_orig = get_grid_loc(x.shape[0], 32, 32, x.device)
    show_conf_merge(density[:, :, None], None, loc_orig, idx_agg, n=1+5, vmin=None)
    show_conf_merge(dist[:, :, None], None, loc_orig, idx_agg, n=2+5, vmin=None)
    show_conf_merge(score[:, :, None], None, loc_orig, idx_agg, n=3+5, vmin=None)


    '''merge'''
    # normalize the weight
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, idx_agg, weight_t
    return x_out, idx_agg


# approximate DPC-KNN
# failed try
# def token_cluster_app2(input_dict, Ns, weight=None, return_weight=False, k=5):
#     x = input_dict['x']
#     idx_agg = input_dict['idx_agg']
#     agg_weight = input_dict['agg_weight']
#     loc_orig = input_dict['loc_orig']
#     H, W = input_dict['map_size']
#
#     dtype = x.dtype
#     device = x.device
#     B, N, C = x.shape
#     N0 = idx_agg.shape[1]
#
#     if weight is None:
#         weight = x.new_ones(B, N, 1)
#
#     with torch.no_grad():
#         if agg_weight is None:
#             agg_weight = x.new_ones(B, N0, 1)
#         scale_factor = N ** 0.25
#         h, w = int(round(H/scale_factor)), int(round(W/scale_factor))
#         K = max(int(2 * math.sqrt(N)), k)
#         # x_rough, _ = token2map(x, None, loc_orig, idx_agg, [h, w], weight=agg_weight)
#         # x_rough = x_rough.flatten(2).permute(0, 2, 1)
#
#         batch = torch.arange(B, device=x.device)[:, None].expand(B, N)
#         with torch.cuda.amp.autocast(enabled=False):
#             # h = h * 2
#             idx_rough = fps(x.flatten(0, 1).type(torch.float32), batch.flatten(0, 1), ratio=h*w/N).reshape(B, -1).to(device)
#             idx_rough = idx_rough - torch.arange(B, device=device)[:, None] * N
#             x_rough = index_points(x, idx_rough)
#
#
#         Nr = x_rough.shape[1]
#         dist_matrix1 = torch.cdist(x, x_rough)
#
#         _, idx_k_rough = torch.topk(-dist_matrix1, k=K, dim=1)      # nearest tokens for every rough token
#         idx_k_rough = idx_k_rough.permute(0, 2, 1)
#
#         idx_tmp = dist_matrix1.argmin(axis=2)                       # nearest rough token for each token
#         idx_k = index_points(idx_k_rough, idx_tmp)                  # approximate nearest tokens for each token
#
#         # _, idx_tmp = dist_matrix1.topk(k=k, dim=-1, largest=False)      # nearest rough token for each token
#         # idx_k = index_points(idx_k_rough, idx_tmp).reshape(B, N, -1)    # approximate nearest tokens for each token
#
#         '''add neareset neighbor in space'''
#         # y_map, x_map = torch.meshgrid(torch.torch.arange(7, device=device)-3, torch.torch.arange(7, device=device)-3)
#         # y_map, x_map = y_map.reshape(-1), x_map.reshape(-1)
#         # idx_off = y_map * W + x_map
#         # K2 = 49
#         # idx_space = torch.arange(N, device=device)[None, :, None].expand(B, N, K2) +\
#         #             idx_off[None, None, :].expand(B, N, K2)
#         # idx_space = idx_space.clamp(0, N-1).long()
#         #
#         # idx_k = torch.cat([idx_k, idx_space], dim=-1)
#         # K1 = K
#         # K = K + K2
#
#
#
#         #####################################################
#         # just for debug
#         # tmp = x.new_zeros([B, N, 1])
#         # idx_batch = torch.arange(B)[:, None, None].expand(B, N, K)
#         # tmp[idx_batch.reshape(-1), idx_k.reshape(-1), 0] = (torch.arange(N, device=device) == N-1)[None, :, None].expand(B, N, K).reshape(-1).float()
#         # show_conf_merge(tmp, None, loc_orig, idx_agg, n=4 + 5, vmin=None)
#         #
#         # # tmp = x.new_zeros([B, N, 1])
#         # # idx_batch = torch.arange(B)[:, None, None].expand(B, h*w, K)
#         # # tmp[idx_batch.reshape(-1), idx_k_rough.reshape(-1), 0] = (torch.arange(h*w,device=device)[None, :, None]>-1).expand(B, -1, K).reshape(-1).float()
#         # # show_conf_merge(tmp, None, loc_orig, idx_agg, n=4 + 5, vmin=None)
#         #
#         #
#         # tmp = x.new_zeros([B, N, 1])
#         # idx_batch = torch.arange(B)[:, None, None].expand(B, h*w, 1)
#         # idx_rough = idx_k_rough[:, :, 0].reshape(B, -1)
#         # tmp[idx_batch.reshape(-1), idx_rough.reshape(-1), :] = 1
#         # show_conf_merge(tmp, None, loc_orig, idx_agg, n=5 + 5, vmin=None)
#         #
#         #####################################################
#
#
#         # idx_rough = idx_k_rough[:, :, 0].reshape(B, -1)
#         # idx_part = torch.cat([idx_k, idx_rough[:, None, :].expand(-1, N, -1)], dim=-1)
#         idx_part = idx_k
#
#         #
#         # # make sure idx_k do not include the token itself.
#         # mask_k = idx_k == torch.arange(N, device=device)[None, :, None]
#         # mask_k2 = mask_k.sum(dim=-1)
#         # idx_k[mask_k] = idx_k[mask_k]
#
#         with torch.cuda.amp.autocast(enabled=False):
#             '''I only support float, float, int Now'''
#             dist_part = f_distance(
#                 x.float().contiguous(),
#                 x.float().contiguous(),
#                 idx_part.int().contiguous())
#
#         # get local density
#         dist_part = dist_part.type(dtype) / (C**0.5)
#         dist_k = dist_part[:, :, :K]
#         idx_k = idx_part[:, :, :K]
#
#         # make sure selected k-nearest neighbor do not include the token itself.
#         mask_k = idx_k == torch.arange(N, device=device)[None, :, None]
#         dist_k = dist_k * (~mask_k) + (dist_k.max()+1) * mask_k
#
#         dist_nearest, index_nearest = torch.topk(dist_k, k=k, dim=-1, largest=False)
#         density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
#
#         # dist indicator
#         density_k_rough = index_points(density[:, :, None], idx_k_rough).squeeze(-1)
#         idx_den = density_k_rough.argmax(dim=-1)
#         idx_batch = torch.arange(B)[:, None].expand(B, Nr)
#         idx_nr = torch.arange(h*w)[None, :].expand(B, Nr)
#         idx_den = idx_k_rough[idx_batch.reshape(-1), idx_nr.reshape(-1), idx_den.reshape(-1)].reshape(B, Nr)
#         x_den = index_points(x, idx_den)
#         dist_den = torch.cdist(x, x_den) / (C**0.5)
#         density_den = index_points(density[:, :, None], idx_den).squeeze(-1)
#
#
#         dist_matrix = dist_part
#         density_matrix = index_points(density[:, :, None], idx_part).squeeze(-1)
#
#         dist_matrix = torch.cat([dist_matrix, dist_den], dim=-1)
#         density_matrix = torch.cat([density_matrix, density_den[:, None, :].expand(B, N, Nr)], dim=-1)
#
#         # dist_matrix = dist_den
#         # density_matrix = density_den[:, None, :].expand(B, N, Nr)
#         #
#
#         mask = density_matrix > density[:, :, None]
#         mask = mask.type(x.dtype)
#         dist, index_parent = (dist_matrix * mask +
#                               dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)
#
#         # # distance indicator, JUST FOR DEBUG
#         # mask = density[:, None, :] > density[:, :, None]
#         # dist_matrix = torch.cdist(x, x) / (C ** 0.5)
#         # mask = mask.type(x.dtype)
#         # dist2, index_parent = (dist_matrix * mask +
#         #                       dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)
#
#
#         # select clustering center according to score
#         score = dist * density
#         _, index_down = torch.topk(score, k=Ns, dim=-1)
#
#         # # # for debug only
#         show_conf_merge(density[:, :, None], None, loc_orig, idx_agg, n=1 + 5, vmin=None)
#         show_conf_merge(dist[:, :, None], None, loc_orig, idx_agg, n=2 + 5, vmin=None)
#         show_conf_merge(score[:, :, None], None, loc_orig, idx_agg, n=3 + 5, vmin=None)
#
#
#         # grouping
#         # assign tokens to the nearest center
#         centers = index_points(x, index_down)
#         dist_matrix = torch.cdist(centers, x)
#         idx_agg_t = dist_matrix.argmin(dim=1)
#
#         # make sure selected centers merge to itself
#         idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
#         idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
#         idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
#
#         idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns
#
#
#
#     '''merge'''
#     # normalize the weight
#     all_weight = weight.new_zeros(B * Ns, 1)
#     all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
#     all_weight = all_weight + 1e-6
#     norm_weight = weight / all_weight[idx]
#
#     # average token features
#     x_out = x.new_zeros(B * Ns, C)
#     source = x * norm_weight
#     x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
#     x_out = x_out.reshape(B, Ns, C)
#
#     idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
#     if return_weight:
#         weight_t = index_points(norm_weight, idx_agg)
#         return x_out, idx_agg, weight_t
#     return x_out, idx_agg




'''failed'''
'''select k nearest neighbor according to distance with a single point'''
def token_cluster_app3(input_dict, Ns, weight=None, return_weight=False, k=5):
    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    loc_orig = input_dict['loc_orig']
    H, W = input_dict['map_size']

    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    N0 = idx_agg.shape[1]

    if weight is None:
        weight = x.new_ones(B, N, 1)
    if agg_weight is None:
        agg_weight = x.new_ones(B, N0, 1)

    with torch.no_grad():
        x_ref = x[:, [0], :]
        dist_ref = torch.cdist(x, x_ref).squeeze(-1)
        _, idx_sort = dist_ref.sort(dim=-1)
        idx_back = idx_sort.argsort(dim=1)

        # x_sort = index_points(x, idx_sort)
        K = max(int(8 * math.sqrt(N)), k)
        # K = int(N)
        idx_k = torch.arange(K, device=device)[None, None, :].expand(B, N, K) - 0.5 * K \
                + torch.arange(N, device=device)[None, :, None].expand(B, N, K)
        idx_k = idx_k.clamp(0, N-1).long()

        idx_k = index_points(idx_back[:, :, None], idx_k).squeeze(-1)
        idx_k = index_points(idx_k, idx_back)


        with torch.cuda.amp.autocast(enabled=False):
            '''I only support float, float, int Now'''
            dist_k = f_distance(
                x.float().contiguous(),
                x.float().contiguous(),
                idx_k.int().contiguous())
            dist_k = dist_k.type(dtype) / (C**0.5)

        dist_part = dist_k
        idx_part = idx_k

        # dist_part = index_points(dist_k, idx_back)
        # idx_part = index_points(idx_sort[:, :, None], idx_k).squeeze(-1)
        # idx_part = index_points(idx_part, idx_back)

        # get local density
        dist_k = dist_part[:, :, :K]
        idx_k = idx_part[:, :, :K]
        # make sure selected k-nearest neighbor do not include the token itself.
        mask_k = idx_k == torch.arange(N, device=device)[None, :, None]
        dist_k = dist_k * (~mask_k) + (dist_k.max()+1) * mask_k

        dist_nearest, index_nearest = torch.topk(dist_k, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

        # dist indicator
        dist_matrix = dist_part
        density_matrix = index_points(density[:, :, None], idx_part).squeeze(-1)

        mask = density_matrix > density[:, :, None]
        mask = mask.type(x.dtype)
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)

        # # distance indicator, JUST FOR DEBUG
        # mask = density[:, None, :] > density[:, :, None]
        # dist_matrix = torch.cdist(x, x) / (C ** 0.5)
        # mask = mask.type(x.dtype)
        # dist2, index_parent = (dist_matrix * mask +
        #                       dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)


        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=Ns, dim=-1)

        # # # for debug only
        # print('debug only!')
        # show_conf_merge(density[:, :, None], None, loc_orig, idx_agg, n=1 + 5, vmin=None)
        # show_conf_merge(dist[:, :, None], None, loc_orig, idx_agg, n=2 + 5, vmin=None)
        # show_conf_merge(score[:, :, None], None, loc_orig, idx_agg, n=3 + 5, vmin=None)


        # grouping
        # assign tokens to the nearest center
        centers = index_points(x, index_down)
        dist_matrix = torch.cdist(centers, x)
        idx_agg_t = dist_matrix.argmin(dim=1)

        # make sure selected centers merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns



    '''merge'''
    # normalize the weight
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    weight_t = index_points(norm_weight, idx_agg)
    return x_out, idx_agg, weight_t, None


def agg_loc(input_dict):
    x = input_dict['x']
    loc_orig = input_dict['loc_orig']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']

    B, N, _ = x.shape
    device, dtype = x.device, x.dtype
    N0 = loc_orig.shape[1]
    if N == N0:
        return loc_orig

    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N0)
    idx_loc = torch.arange(N0, device=device)[None, :].expand(B, N0)

    # use sparse matrix
    idx_agg = idx_agg + idx_batch * N
    idx_loc = idx_loc + idx_batch * N0

    indices = torch.stack([idx_agg, idx_loc], dim=0).reshape(2, -1)

    if agg_weight is None:
        value = torch.ones(B * N0, device=device, dtype=dtype)
    else:
        value = agg_weight.reshape(B * N0)  # .type(torch.float32)

    all_weight = spmm(indices, value, B * N, B * N0, x.new_ones([B * N0, 1])) + 1e-6
    value = value / all_weight[idx_agg.reshape(-1), 0]
    loc = spmm(indices, value, B * N, B * N0, loc_orig.reshape(B * N0, 2))
    loc = loc.reshape(B, N, 2)
    return loc


def token_cluster_app2(input_dict, Ns, weight=None, k=5):
    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    loc_orig = input_dict['loc_orig']
    H, W = input_dict['map_size']
    idx_k_loc = input_dict['idx_k_loc']

    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    N0 = idx_agg.shape[1]

    idx_k_loc = idx_k_loc.expand(B, -1, -1)

    if weight is None:
        weight = x.new_ones(B, N, 1)

    with torch.no_grad():
        if agg_weight is None:
            agg_weight = x.new_ones(B, N0, 1)

        scale_factor = N ** 0.25
        h, w = int(round(H/scale_factor)), int(round(W/scale_factor))
        K = max(int(2 * math.sqrt(N)), k)
        x_rough, _ = token2map(x, None, loc_orig, idx_agg, [h, w], weight=agg_weight)
        x_rough = x_rough.flatten(2).permute(0, 2, 1)

        Nr = x_rough.shape[1]
        dist_matrix1 = torch.cdist(x, x_rough)

        _, idx_k_rough = torch.topk(-dist_matrix1, k=K, dim=1)      # nearest tokens for every rough token
        idx_k_rough = idx_k_rough.permute(0, 2, 1)

        idx_tmp = dist_matrix1.argmin(axis=2)                       # nearest rough token for each token
        idx_k_fea = index_points(idx_k_rough, idx_tmp)              # approximate nearest tokens for each token

        idx_k = torch.cat([idx_k_fea, idx_k_loc], dim=-1)

        # density is calculated using idx_k_fea
        with torch.cuda.amp.autocast(enabled=False):
            '''I only support float, float, int Now'''
            dist_k = f_distance(
                x.float().contiguous(),
                x.float().contiguous(),
                idx_k.int().contiguous())
            dist_k = dist_k.type(dtype) / (C ** 0.5)

        '''get local density'''
        # the neighbor tokens can NOT include itself and duplicate tokens.

        # in order to make sure no duplicate tokens, we only use idx_k_fea for density,
        dist_fea = dist_k[:, :, : K]

        # make sure selected k-nearest neighbor do not include the token itself.
        mask_fea = idx_k_fea == torch.arange(N, device=device)[None, :, None]
        dist_fea = dist_fea * (~mask_fea) + (dist_fea.max()+1) * mask_fea

        dist_nearest, _ = torch.topk(dist_fea, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

        '''dist indicator'''
        # the neighbor tokens can include itself and duplicate tokens.

        dist_matrix = dist_k
        density_matrix = index_points(density[:, :, None], idx_k).squeeze(-1)

        # add some rough tokens with high density
        density_k_rough = index_points(density[:, :, None], idx_k_rough).squeeze(-1)
        idx_den = density_k_rough.argmax(dim=-1)
        idx_batch = torch.arange(B)[:, None].expand(B, Nr)
        idx_nr = torch.arange(h*w)[None, :].expand(B, Nr)
        idx_den = idx_k_rough[idx_batch.reshape(-1), idx_nr.reshape(-1), idx_den.reshape(-1)].reshape(B, Nr)
        x_den = index_points(x, idx_den)
        dist_den = torch.cdist(x, x_den) / (C**0.5)
        density_den = index_points(density[:, :, None], idx_den).squeeze(-1)

        dist_matrix = torch.cat([dist_matrix, dist_den], dim=-1)
        density_matrix = torch.cat([density_matrix, density_den[:, None, :].expand(B, N, Nr)], dim=-1)

        mask = density_matrix > density[:, :, None]
        mask = mask.type(x.dtype)
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)

        score = dist * density
        _, index_down = torch.topk(score, k=Ns, dim=-1)

        # # # for debug only
        # print('for debug only!')
        # show_conf_merge(density[:, :, None], None, loc_orig, idx_agg, n=1 + 5, vmin=None)
        # show_conf_merge(dist[:, :, None], None, loc_orig, idx_agg, n=2 + 5, vmin=None)
        # show_conf_merge(score[:, :, None], None, loc_orig, idx_agg, n=3 + 5, vmin=None)

        ''' tokens grouping '''
        # assign tokens to the nearest center
        centers = index_points(x, index_down)
        dist_matrix = torch.cdist(x, centers)
        idx_agg_t = dist_matrix.argmin(dim=-1)

        # make sure selected centers merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns


        '''get idx_k_loc for the next stage'''
        # first change idx
        idx_k_loc_new = index_points(idx_agg_t[:, :, None], idx_k_loc).squeeze(-1)

        # direct use the idx_k_loc of the center
        idx_k_loc_new = index_points(idx_k_loc_new, index_down)

    '''merge'''
    # normalize the weight
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    weight_t = index_points(norm_weight, idx_agg)

    return x_out, idx_agg, weight_t, idx_k_loc_new


def token_cluster_near(input_dict, Ns, weight=None, k=5):
    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    loc_orig = input_dict['loc_orig']
    H, W = input_dict['map_size']
    idx_k_loc = input_dict['idx_k_loc']

    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    N0 = idx_agg.shape[1]

    idx_k_loc = idx_k_loc.expand(B, -1, -1)

    if weight is None:
        weight = x.new_ones(B, N, 1)

    with torch.no_grad():
        if agg_weight is None:
            agg_weight = x.new_ones(B, N0, 1)


        scale_factor = N ** 0.25
        h, w = int(round(H/scale_factor)), int(round(W/scale_factor))
        K = max(int(2 * math.sqrt(N)), k)
        x_rough, _ = token2map(x, None, loc_orig, idx_agg, [h, w], weight=agg_weight)
        x_rough = x_rough.flatten(2).permute(0, 2, 1)

        Nr = x_rough.shape[1]
        dist_matrix1 = torch.cdist(x, x_rough)

        _, idx_k_rough = torch.topk(-dist_matrix1, k=K, dim=1)      # nearest tokens for every rough token
        idx_k_rough = idx_k_rough.permute(0, 2, 1)

        # idx_tmp = dist_matrix1.argmin(axis=2)                       # nearest rough token for each token
        # idx_k_fea = index_points(idx_k_rough, idx_tmp)              # approximate nearest tokens for each token
        #
        # idx_k = torch.cat([idx_k_fea, idx_k_loc], dim=-1)

        dist_all = torch.cdist(x, x)
        _, idx_k_gt = torch.topk(dist_all, k=K, dim=-1, largest=False)
        idx_k = idx_k_gt
        idx_k_fea = idx_k_gt


        # density is calculated using idx_k_fea
        with torch.cuda.amp.autocast(enabled=False):
            '''I only support float, float, int Now'''
            dist_k = f_distance(
                x.float().contiguous(),
                x.float().contiguous(),
                idx_k.int().contiguous())
            dist_k = dist_k.type(dtype) / (C ** 0.5)

        '''get local density'''
        # the neighbor tokens can NOT include itself and duplicate tokens.

        # in order to make sure no duplicate tokens, we only use idx_k_fea for density,
        dist_fea = dist_k[:, :, : K]

        # make sure selected k-nearest neighbor do not include the token itself.
        mask_fea = idx_k_fea == torch.arange(N, device=device)[None, :, None]
        dist_fea = dist_fea * (~mask_fea) + (dist_fea.max()+1) * mask_fea

        dist_nearest, _ = torch.topk(dist_fea, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

        '''dist indicator'''
        # the neighbor tokens can include itself and duplicate tokens.

        dist_matrix = dist_k
        density_matrix = index_points(density[:, :, None], idx_k).squeeze(-1)

        # add some rough tokens with high density
        density_k_rough = index_points(density[:, :, None], idx_k_rough).squeeze(-1)
        idx_den = density_k_rough.argmax(dim=-1)
        idx_batch = torch.arange(B)[:, None].expand(B, Nr)
        idx_nr = torch.arange(h*w)[None, :].expand(B, Nr)
        idx_den = idx_k_rough[idx_batch.reshape(-1), idx_nr.reshape(-1), idx_den.reshape(-1)].reshape(B, Nr)
        x_den = index_points(x, idx_den)
        dist_den = torch.cdist(x, x_den) / (C**0.5)
        density_den = index_points(density[:, :, None], idx_den).squeeze(-1)

        dist_matrix = torch.cat([dist_matrix, dist_den], dim=-1)
        density_matrix = torch.cat([density_matrix, density_den[:, None, :].expand(B, N, Nr)], dim=-1)

        mask = density_matrix > density[:, :, None]
        mask = mask.type(x.dtype)
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)

        score = dist * density
        _, index_down = torch.topk(score, k=Ns, dim=-1)

        # # # # for debug only
        # print('for debug only!')
        # show_conf_merge(density[:, :, None], None, loc_orig, idx_agg, n=1 + 5, vmin=None)
        # show_conf_merge(dist[:, :, None], None, loc_orig, idx_agg, n=2 + 5, vmin=None)
        # show_conf_merge(score[:, :, None], None, loc_orig, idx_agg, n=3 + 5, vmin=None)

        ''' tokens grouping '''
        # assign tokens to the nearest center
        centers = index_points(x, index_down)
        dist_matrix = torch.cdist(x, centers)
        idx_agg_t = dist_matrix.argmin(dim=-1)

        # make sure selected centers merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns


        '''get idx_k_loc for the next stage'''
        # first change idx
        idx_k_loc_new = index_points(idx_agg_t[:, :, None], idx_k_loc).squeeze(-1)

        # direct use the idx_k_loc of the center
        idx_k_loc_new = index_points(idx_k_loc_new, index_down)

    '''merge'''
    # normalize the weight
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    weight_t = index_points(norm_weight, idx_agg)

    return x_out, idx_agg, weight_t, idx_k_loc_new


def conf_nms(conf_map, k):
    if k <= 1:
        return conf_map
    dtype, device = conf_map.dtype, conf_map.device
    pad = int((k - 1) / 2)
    tmp = torch.arange(k, device=device).type(dtype) - pad
    y_map, x_map = torch.meshgrid(tmp, tmp)
    dist_map = torch.stack([x_map, y_map], dim=-1).norm(p=2, dim=-1)
    kernel = 1 / dist_map
    kernel[pad, pad] = 0
    kernel = -kernel / kernel.sum()
    kernel[pad, pad] = 1
    kernel = kernel[None, None, :, :]
    out = F.conv2d(F.pad(conf_map, [pad, pad, pad, pad], mode='replicate'), kernel)
    return out


def token_cluster_nms(input_dict, Ns, conf, weight=None, k=5):
    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    loc_orig = input_dict['loc_orig']
    H, W = input_dict['map_size']
    idx_k_loc = input_dict['idx_k_loc']

    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    N0 = idx_agg.shape[1]

    with torch.no_grad():
        conf_map, _ = token2map(conf, None, loc_orig, idx_agg, [H, W], agg_weight)
        nms_k = int((H * W) ** 0.25 // 2 * 2 - 1)
        conf_map = conf_nms(conf_map, nms_k)
        score = map2token(conf_map, N, loc_orig, idx_agg, agg_weight).squeeze(-1)
        _, index_down = torch.topk(score, k=Ns, dim=-1)

        ''' tokens grouping '''
        # assign tokens to the nearest center
        centers = index_points(x, index_down)
        dist_matrix = torch.cdist(x, centers)
        idx_agg_t = dist_matrix.argmin(dim=-1)

        # make sure selected centers merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns

    '''merge'''
    # normalize the weight
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    weight_t = index_points(norm_weight, idx_agg)

    return x_out, idx_agg, weight_t, None


def token_cluster_grid(input_dict, Ns, conf, weight=None, k=5):
    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    loc_orig = input_dict['loc_orig']
    H, W = input_dict['map_size']
    # idx_k_loc = input_dict['idx_k_loc']

    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    N0 = idx_agg.shape[1]
    if weight is None:
        weight = x.new_ones(B, N, 1)

    w_map, _ = token2map(weight, None, loc_orig, idx_agg, [H, W], agg_weight)
    mean_w = F.avg_pool2d(w_map, kernel_size=2)
    mean_w = F.interpolate(mean_w, [H, W], mode='nearest')
    norm_weight = w_map / (mean_w + 1e-6)
    norm_weight = map2token(norm_weight, N, loc_orig, idx_agg, agg_weight)
    weight_t = norm_weight / 4
    weight_t = index_points(weight_t, idx_agg)


    x_map, _ = token2map(x*norm_weight, None, loc_orig, idx_agg, [H, W], agg_weight)
    x_map = F.avg_pool2d(x_map, kernel_size=2)
    x_out = x_map.flatten(2).permute(0, 2, 1)

    # follow token2map process
    H, W = H // 2, W // 2
    loc_orig = loc_orig.clamp(-1, 1)
    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    loc_orig = loc_orig.round().long()
    loc_orig[..., 0] = loc_orig[..., 0].clamp(0, W-1)
    loc_orig[..., 1] = loc_orig[..., 1].clamp(0, H-1)
    idx_HW_orig = loc_orig[..., 0] + loc_orig[..., 1] * W
    idx_agg_down = idx_HW_orig

    return x_out, idx_agg_down, weight_t, None


# find the spatial neighbors of tokens
def get_spatial_neighbor(loc, k=49):
    # reorder to grid structure
    B, N, _ = loc.shape
    dist_matrix = torch.cdist(loc, loc)
    _, idx_k = torch.topk(dist_matrix, dim=-1, largest=False, k=k)
    return idx_k


def get_initial_loc_neighbor(H, W, device, k=7):
    pad = int((k - 1) / 2)
    tmp = torch.arange(k, device=device) - pad
    y_map, x_map = torch.meshgrid(tmp, tmp)
    y_map, x_map = y_map.reshape(-1), x_map.reshape(-1)
    idx_off = y_map * W + x_map

    idx_k_loc = torch.arange(H * W, device=device).reshape(H, W, 1)
    idx_k_loc = idx_k_loc - idx_off[None, None, :]
    idx_k_loc = idx_k_loc[pad:-pad, pad:-pad, :]
    idx_k_loc = F.pad(idx_k_loc.permute(2, 0, 1).unsqueeze(0).float(),
                      [pad, pad, pad, pad], mode='replicate').long()
    idx_k_loc = idx_k_loc.flatten(2).permute(0, 2, 1)
    return idx_k_loc


'''part wise clustering'''
def token_cluster_part(input_dict, Ns, weight=None, k=5, h=-1, w=-1):
    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    loc_orig = input_dict['loc_orig']
    H, W = input_dict['map_size']
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    N0 = idx_agg.shape[1]

    if weight is None:
        weight = x.new_ones(B, N, 1)

    if (h < 0 and w < 0) or (h >= H and w >= W):
        # no part seg
        return token_cluster_merge(x, Ns, idx_agg, weight=weight, return_weight=True, k=k)
    else:
        # can be equally splited
        if H % h == 0 and W % w == 0:
            p_h, p_w = H // h, W // w
            num_part = p_h * p_w
            N_p = N // num_part
            Ns_p = Ns // num_part

            part_map = torch.arange(p_h * p_w, device=device).reshape(1, 1, p_h, p_w).expand(B, -1, -1, -1)
            # no need to use weight here
            token_part = map2token(part_map.float(), N, loc_orig, idx_agg, agg_weight).round().long()
            idx_sort = token_part.argsort(dim=1).squeeze(-1)
            idx_back = idx_sort.argsort(dim=1)

            x_sort = index_points(x, idx_sort)
            x_sort = x_sort.reshape(B * num_part, N_p, C)

            dist_matrix = torch.cdist(x_sort, x_sort, p=2) / (C ** 0.5)

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

            # get relative-separation distance
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist, index_parent = (dist_matrix * mask +
                                  dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)

            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=Ns_p, dim=-1)

            # # for debug only
            # print('only for debug')
            # den = density.reshape(B, N, 1)
            # den = index_points(den, idx_back)
            # den = den.reshape(B, H, W)
            # plt.imshow(den[0].detach().cpu())
            #
            # tmp = dist.reshape(B, N, 1)
            # tmp = index_points(tmp, idx_back)
            # tmp = tmp.reshape(B, H, W)
            # plt.imshow(tmp[0].detach().cpu())

            # assign tokens to the nearest center
            dist_matrix = index_points(dist_matrix, index_down)
            idx_agg_t = dist_matrix.argmin(dim=1)

            # make sure selected centers merge to itself
            idx_batch = torch.arange(B * num_part, device=x.device)[:, None].expand(B * num_part, Ns_p)
            idx_tmp = torch.arange(Ns_p, device=x.device)[None, :].expand(B * num_part, Ns_p)
            idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

            # tansfer index_down and idx_agg_t to the original sort
            # index_down = index_down.reshape(B, num_part, Ns_p) + torch.arange(num_part)[:, None, :] * N_p
            # index_down = index_down.reshape(B, num_part*Ns_p)
            # index_down = index_points(idx_sort[:, :, None], index_down).squeeze(-1)

            idx_agg_t = idx_agg_t.reshape(B, num_part, N_p) + torch.arange(num_part, device=device)[None, :, None] * Ns_p
            idx_agg_t = idx_agg_t.reshape(B, num_part*N_p)
            # idx_agg_t = index_points(idx_sort[:, :, None], idx_agg_t)
            idx_agg_t = index_points(idx_agg_t[:, :, None], idx_back).squeeze(-1)

            idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns
        else:
            raise NotImplementedError('tokens can not be splitted equally!')

    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    weight_t = index_points(norm_weight, idx_agg)

    return x_out, idx_agg, weight_t




# sort and pad tokens
def token_cluster_part_pad(input_dict, Ns, weight=None, k=5, nh_list=[1, 1], nw_list=[1, 1]):
    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    loc_orig = input_dict['loc_orig']
    H, W = input_dict['map_size']
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    N0 = idx_agg.shape[1]

    # assert multiple stage seg is compatiable
    assert len(nh_list) == len(nw_list)
    for i in range(len(nh_list) - 1):
        assert nh_list[i] % nh_list[i + 1] == 0 and \
               nw_list[i] % nw_list[i + 1] == 0
    nh, nw = nh_list[0], nw_list[0]

    if (nh <= 1 and nw <= 1):
        # no part seg
        return token_cluster_merge(x, Ns, idx_agg, weight=weight, return_weight=True, k=k)


    with torch.no_grad():
        # reshape to feature map
        x_pad = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # pad feature map
        pad_h = (nh - H % nh) % nh
        pad_w = (nw - W % nw) % nw
        x_pad = F.pad(x_pad, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        pad_mask = x.new_ones([1, 1, H, W])
        pad_mask = F.pad(pad_mask,
                         [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                         mode='constant', value=0)

        # sort padded tokens
        _, _, H_pad, W_pad = x_pad.shape
        token_part = x.new_zeros([1, 1, H_pad, W_pad])
        for i in range(len(nh_list)-1, -1, -1):
            nh_t, nw_t = nh_list[i], nw_list[i]
            part_t = torch.arange(nh_t*nw_t, device=device, dtype=x.dtype)
            part_t = part_t.reshape(1, 1, nh_t, nw_t)
            part_t = F.interpolate(part_t, size=(H_pad, W_pad), mode='nearest')
            token_part = token_part * nh_t * nw_t + part_t

        token_part = token_part.reshape(H_pad * W_pad)
        idx_sort = token_part.argsort(dim=0)
        idx_back = idx_sort.argsort(dim=0)

        x_pad = x_pad.flatten(2).permute(0, 2, 1)
        pad_mask = pad_mask.flatten(2).permute(0, 2, 1)

        x_sort = index_points(x_pad, idx_sort[None, :].expand(B, -1))
        pad_mask_sort = index_points(pad_mask, idx_sort[None, :].expand(1, -1)).expand(B, -1, -1)
        num_part = nh * nw
        N_p = (H_pad // nh) * (W_pad // nw)
        Ns_p = round(Ns / num_part)
        Ns = Ns_p * num_part

        x_sort = x_sort.reshape(B * num_part, N_p, C)
        dist_matrix = torch.cdist(x_sort, x_sort, p=2) / (C ** 0.5)
        pad_mask_sort = pad_mask_sort.reshape(B*num_part, N_p, 1)

        # in order to not affect cluster, masked tokens distance should be max
        dist_matrix = dist_matrix * pad_mask_sort + dist_matrix.max() * (1-pad_mask_sort)

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

        # add a small random noise for the situation where some tokens have totally the same feature
        # (for the images with balck edges)
        density = density + torch.rand(density.shape, device=device, dtype=density.dtype) * 1e-6

        # masked tokens density should be 0
        density = density * pad_mask_sort.squeeze(-1)

        # get relative-separation distance
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=Ns_p, dim=-1)

        # only for debug
        # print('for debug only!')
        # show_conf_merge(index_points(density.reshape(B, num_part * N_p, 1), idx_back[None, :].expand(B, -1)), None, loc_orig, idx_agg, n=1, vmin=None)
        # show_conf_merge(index_points(dist.reshape(B, num_part * N_p, 1), idx_back[None, :].expand(B, -1)), None, loc_orig, idx_agg, n=2, vmin=None)
        # show_conf_merge(index_points(score.reshape(B, num_part * N_p, 1), idx_back[None, :].expand(B, -1)), None, loc_orig, idx_agg, n=3, vmin=None)
        #

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)
        idx_agg_t = dist_matrix.argmin(dim=1)

        # make sure selected centers merge to itself
        idx_batch = torch.arange(B * num_part, device=x.device)[:, None].expand(B * num_part, Ns_p)
        idx_tmp = torch.arange(Ns_p, device=x.device)[None, :].expand(B * num_part, Ns_p)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        # tansfer index_down and idx_agg_t to the original sort
        idx_agg_t = idx_agg_t.reshape(B, num_part, N_p) + torch.arange(num_part, device=device)[None, :, None] * Ns_p
        idx_agg_t = idx_agg_t.reshape(B, num_part * N_p)
        idx_agg_t = index_points(idx_agg_t[:, :, None], idx_back[None, :].expand(B, -1)).squeeze(-1)

        # remve padded tokens
        idx_agg_t = idx_agg_t.reshape(B, H_pad, W_pad)
        idx_agg_t = idx_agg_t[:, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W].contiguous()
        idx_agg_t = idx_agg_t.reshape(B, N)


    # merge tokens
    if weight is None:
        weight = x.new_ones(B, N, 1)

    idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns

    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    weight_t = index_points(norm_weight, idx_agg)

    return x_out, idx_agg, weight_t


def token_cluster_part_follow(input_dict, Ns, weight=None, k=5, nh=1, nw=1):
    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    loc_orig = input_dict['loc_orig']
    H, W = input_dict['map_size']
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    N0 = idx_agg.shape[1]

    if (nh <= 1 and nw <= 1):
        # no part seg
        return token_cluster_merge(x, Ns, idx_agg, weight=weight, return_weight=True, k=k)

    with torch.no_grad():
        # can be equally splited
        num_part = nh * nw
        N_p = N // num_part
        assert N % num_part == 0
        Ns_p = round(Ns // num_part)
        Ns = Ns_p * num_part

        x_sort = x
        x_sort = x_sort.reshape(B * num_part, N_p, C)

        dist_matrix = torch.cdist(x_sort, x_sort, p=2) / (C ** 0.5)

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

        # add a small random noise for the situation where some tokens have totally the same feature
        # (for the images with balck edges)
        density = density + torch.rand(density.shape, device=device, dtype=density.dtype) * 1e-6


        # get relative-separation distance
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=Ns_p, dim=-1)

        dist_matrix = index_points(dist_matrix, index_down)
        idx_agg_t = dist_matrix.argmin(dim=1)

        # make sure selected centers merge to itself
        idx_batch = torch.arange(B * num_part, device=x.device)[:, None].expand(B * num_part, Ns_p)
        idx_tmp = torch.arange(Ns_p, device=x.device)[None, :].expand(B * num_part, Ns_p)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        # tansfer index_down and idx_agg_t to the original sort
        idx_agg_t = idx_agg_t.reshape(B, num_part, N_p) + torch.arange(num_part, device=device)[None, :, None] * Ns_p
        idx_agg_t = idx_agg_t.reshape(B, num_part*N_p)

    if weight is None:
        weight = x.new_ones(B, N, 1)
    idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    weight_t = index_points(norm_weight, idx_agg)

    return x_out, idx_agg, weight_t


# ATS: Adaptive Token Sampling For Efficient Vision Transformers
def token_cluster_ats(input_dict, Ns, score, weight=None):
    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    # loc_orig = input_dict['loc_orig']
    H, W = input_dict['map_size']
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    N0 = idx_agg.shape[1]

    with torch.no_grad():
        # print('ONLY FOR DEBUG!')
        # score = score.new_ones([B, N])

        score = score.squeeze(-1)
        score = score / score.sum(dim=-1, keepdim=True)
        sum_score = score.cumsum(dim=-1)

        # equal gap leads to very bad cluster performance.
        sample_score = (torch.arange(Ns, device=device).type(dtype) + 0.5) / Ns

        # idx_tmp = torch.arange(H*W, device=device).float().reshape(H, W)
        # idx_tmp = F.interpolate(idx_tmp[None, None, :, :], size=[H//2, W//2], mode='nearest')
        # sample_score = (idx_tmp.reshape(-1) + 0.5) / (H*W)
        # Ns = sample_score.shape[0]

        tmp_matrix = (sample_score[None, :, None] - sum_score[:, None, :]).abs()   # B, Ns, N
        index_down = tmp_matrix.argmin(dim=-1)

        x_down = index_points(x, index_down)
        dist_matrix = torch.cdist(x_down, x)
        idx_agg_t = dist_matrix.argmin(dim=1)

        # there may be duplicate tokens, so we do NOT need to make sure selected centers merge to itself
        # idx_batch = torch.arange(B * num_part, device=x.device)[:, None].expand(B * num_part, Ns_p)
        # idx_tmp = torch.arange(Ns_p, device=x.device)[None, :].expand(B * num_part, Ns_p)
        # idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns
    if weight is None:
        weight = x.new_ones(B, N, 1)

    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    weight_t = index_points(norm_weight, idx_agg)

    return x_out, idx_agg, weight_t


# gaussian filtering
def gaussian_filt(x, kernel_size=5, sigma=None):
    if kernel_size < 3:
        return x

    if sigma is None:
        sigma = 0.6 * kernel_size

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
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).contiguous()
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    pad = int((kernel_size - 1) // 2)

    x = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    y = F.conv2d(
        input=x,
        weight=gaussian_kernel,
        stride=1,
        padding=0,
        dilation=1,
        groups=channels
    )
    return y


# gaussian filtering
def avg_filt(x, kernel_size=5):
    if kernel_size < 3:
        return x

    channels = x.shape[1]

    gaussian_kernel = x.new_ones([kernel_size, kernel_size])
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).contiguous()
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    pad = int((kernel_size - 1) // 2)

    x = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    y = F.conv2d(
        input=x,
        weight=gaussian_kernel,
        stride=1,
        padding=0,
        dilation=1,
        groups=channels
    )
    return y


def pca_feature(x):
    with torch.cuda.amp.autocast(enabled=False):
        U, S, V = torch.pca_lowrank(x[0].float(), q=3)
        tmp = x @ V
        tmp = tmp - tmp.min(dim=1, keepdim=True)[0]
        tmp = tmp / tmp.max(dim=1, keepdim=True)[0]
        # tmp = tmp / tmp.max()
    return tmp


def get_idx_agg(x_sort, Ns_p, k=5, pad_mask_sort=None, ignore_density=False):
    C = x_sort.shape[-1]

    dist_matrix = torch.cdist(x_sort, x_sort, p=2) / (C ** 0.5)

    if pad_mask_sort is not None:
        # in order to not affect cluster, masked tokens distance should be max
        dist_matrix = dist_matrix * pad_mask_sort + dist_matrix.max() * (1 - pad_mask_sort)

    # get local density
    dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

    # add a small random noise for the situation where some tokens have totally the same feature
    # (for the images with balck edges)
    density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

    if pad_mask_sort is not None:
        # masked tokens density should be 0
        density = density * pad_mask_sort.squeeze(-1)

    # get relative-separation distance
    mask = density[:, None, :] > density[:, :, None]
    mask = mask.type(x_sort.dtype)
    dist, index_parent = (dist_matrix * mask +
                          dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1 - mask)).min(dim=-1)

    # select clustering center according to score
    score = dist if ignore_density else dist * density

    _, index_down = torch.topk(score, k=Ns_p, dim=-1)

    dist_matrix = index_points(dist_matrix, index_down)
    idx_agg_t = dist_matrix.argmin(dim=1)

    # make sure selected centers merge to itself
    B = x_sort.shape[0]
    idx_batch = torch.arange(B, device=x_sort.device)[:, None].expand(B, Ns_p)
    idx_tmp = torch.arange(Ns_p, device=x_sort.device)[None, :].expand(B, Ns_p)
    idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
    return idx_agg_t, density, dist, score


def token_remerge_part(input_dict, Ns, weight=None, k=5, nh_list=[1, 1, 1, 1], nw_list=[1, 1, 1, 1],
                       level=0, output_tokens=False, first_cluster=False, ignore_density=False):
    # assert multiple stage seg is compatiable
    assert len(nh_list) == len(nw_list)
    for i in range(len(nh_list) - 1):
        assert nh_list[i] % nh_list[i + 1] == 0 and \
               nw_list[i] % nw_list[i + 1] == 0

    nh, nw = nh_list[level], nw_list[level]

    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    loc_orig = input_dict['loc_orig']
    H, W = input_dict['map_size']
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    N0 = idx_agg.shape[1]

    # # only for debug
    # print('for debug only!')
    # if not output_tokens:
    #     # tmp = pca_feature(x)
    #     # tmp = token2map(tmp, None, loc_orig, idx_agg, [H, W])[0]
    #     # plt.imshow(tmp[0].detach().cpu().float().permute(1, 2, 0))
    #     # nh, nw = 1, 1
    #     x = x / C
    #     ignore_density = True

    # get clustering and merge way
    with torch.no_grad():
        if (nh <= 1 and nw <= 1):
            # no part seg
            idx_agg_t, density, dist, score = get_idx_agg(x, Ns, k, pad_mask_sort=None, ignore_density=ignore_density)
            # only for debug
            # if not output_tokens:
            #     print('for debug only!')
            #     show_conf_merge(density[..., None], None, loc_orig, idx_agg, n=1, vmin=None)
            #     show_conf_merge(dist[..., None], None, loc_orig, idx_agg, n=2, vmin=None)
            #     show_conf_merge(score[..., None], None, loc_orig, idx_agg, n=3, vmin=None)
            #     t=0

        elif first_cluster:
            '''we need to sort tokens in the first cluster process'''

            # reshape to feature map
            x_pad = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

            # pad feature map
            pad_h = (nh - H % nh) % nh
            pad_w = (nw - W % nw) % nw
            x_pad = F.pad(x_pad, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
            pad_mask = x.new_ones([1, 1, H, W])
            pad_mask = F.pad(pad_mask,
                             [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                             mode='constant', value=0)

            # sort padded tokens
            _, _, H_pad, W_pad = x_pad.shape
            token_part = x.new_zeros([1, 1, H_pad, W_pad])
            for i in range(len(nh_list)-1, level-1, -1):
                nh_t, nw_t = nh_list[i], nw_list[i]
                part_t = torch.arange(nh_t*nw_t, device=device, dtype=x.dtype)
                part_t = part_t.reshape(1, 1, nh_t, nw_t)
                part_t = F.interpolate(part_t, size=(H_pad, W_pad), mode='nearest')
                token_part = token_part * nh_t * nw_t + part_t

            token_part = token_part.reshape(H_pad * W_pad)
            idx_sort = token_part.argsort(dim=0)
            idx_back = idx_sort.argsort(dim=0)

            x_pad = x_pad.flatten(2).permute(0, 2, 1)
            pad_mask = pad_mask.flatten(2).permute(0, 2, 1)

            x_sort = index_points(x_pad, idx_sort[None, :].expand(B, -1))
            pad_mask_sort = index_points(pad_mask, idx_sort[None, :].expand(1, -1)).expand(B, -1, -1)
            num_part = nh * nw
            N_p = (H_pad // nh) * (W_pad // nw)
            Ns_p = round(Ns / num_part)
            Ns = Ns_p * num_part

            x_sort = x_sort.reshape(B * num_part, N_p, C)
            pad_mask_sort = pad_mask_sort.reshape(B*num_part, N_p, 1)



            idx_agg_t, density, dist, score = get_idx_agg(x_sort, Ns_p, k, pad_mask_sort, ignore_density=ignore_density)

            # # only for debug
            # if not output_tokens:
            #     print('for debug only!')
            #     show_conf_merge(index_points(density.reshape(B, num_part * N_p, 1), idx_back[None, :].expand(B, -1)), None, loc_orig, idx_agg, n=1, vmin=None)
            #     show_conf_merge(index_points(dist.reshape(B, num_part * N_p, 1), idx_back[None, :].expand(B, -1)), None, loc_orig, idx_agg, n=2, vmin=None)
            #     show_conf_merge(index_points(score.reshape(B, num_part * N_p, 1), idx_back[None, :].expand(B, -1)), None, loc_orig, idx_agg, n=3, vmin=None)
            #


            # tansfer index_down and idx_agg_t to the original sort
            idx_agg_t = idx_agg_t.reshape(B, num_part, N_p) + torch.arange(num_part, device=device)[None, :, None] * Ns_p
            idx_agg_t = idx_agg_t.reshape(B, num_part * N_p)
            idx_agg_t = index_points(idx_agg_t[:, :, None], idx_back[None, :].expand(B, -1)).squeeze(-1)

            # remve padded tokens
            idx_agg_t = idx_agg_t.reshape(B, H_pad, W_pad)
            idx_agg_t = idx_agg_t[:, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W].contiguous()
            idx_agg_t = idx_agg_t.reshape(B, N)

        else:

            # can be equally splited
            num_part = nh * nw
            N_p = N // num_part
            assert N % num_part == 0
            Ns_p = round(Ns // num_part)
            Ns = Ns_p * num_part

            x_sort = x
            x_sort = x_sort.reshape(B * num_part, N_p, C)

            idx_agg_t, density, dist, score = get_idx_agg(x_sort, Ns_p, k, pad_mask_sort=None, ignore_density=ignore_density)

            # # only for debug
            # if not output_tokens:
            #     print('for debug only!')
            #     show_conf_merge(density.reshape(B, num_part * N_p, 1), None, loc_orig, idx_agg, n=1, vmin=None)
            #     show_conf_merge(dist.reshape(B, num_part * N_p, 1), None, loc_orig, idx_agg, n=2, vmin=None)
            #     show_conf_merge(score.reshape(B, num_part * N_p, 1), None, loc_orig, idx_agg, n=3, vmin=None)



            # tansfer index_down and idx_agg_t to the original sort
            idx_agg_t = idx_agg_t.reshape(B, num_part, N_p) + torch.arange(num_part, device=device)[None, :,
                                                              None] * Ns_p
            idx_agg_t = idx_agg_t.reshape(B, num_part * N_p)

    # merge tokens
    if weight is None:
        weight = x.new_ones(B, N, 1)

    idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns

    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    if output_tokens:
        # average token features
        x_out = x.new_zeros(B * Ns, C)
        source = x * norm_weight
        x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
        x_out = x_out.reshape(B, Ns, C)
    else:
        x_out = x.new_zeros(B, Ns, C)   # empty tokens

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    weight_t = index_points(norm_weight, idx_agg)

    agg_weight_down = agg_weight * weight_t
    agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

    return x_out, idx_agg, agg_weight_down

