import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
# try:
#     from torch_cluster import fps
# except (ImportError, ModuleNotFoundError):
#     print('no torch cluster')


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


def get_loc_new(x, H, W, grid_stride):
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
    # pos_grid = tmp[grid_stride // 2:H:grid_stride, grid_stride // 2:W:grid_stride]
    pos_grid = tmp[0:H:grid_stride, 0:W:grid_stride]
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


def token2map_partical(x, loc, map_size, conf=None, method=0):
    H, W = map_size
    B, N, C = x.shape
    loc = loc.clamp(-1, 1)
    loc = 0.5 * (loc + 1) * torch.FloatTensor([W, H]).to(loc.device)[None, None, :] - 0.5
    loc = loc.round().long()
    loc[..., 0] = loc[..., 0].clamp(0, W-1)
    loc[..., 1] = loc[..., 1].clamp(0, H-1)
    idx = loc[..., 0] + loc[..., 1] * W
    idx = idx + torch.arange(B)[:, None].to(loc.device) * H * W
    if conf is None:
        out = x.new_zeros(B * H * W, C + 1)
        weight = x.new_ones(B, N, 1)
        tmp = torch.cat([x, weight], dim=-1)
        out.index_add_(dim=0, index=idx.reshape(B*N), source=tmp.reshape(B*N, C+1))
        out = out.reshape(B, H, W, C + 1).permute(0, 3, 1, 2).contiguous()
        feature = out[:, :C, :, :]
        weight = out[:, C:, :, :]
        feature = feature / (weight + 1e-6)
        mask = (weight > 0).float()
    else:
        conf = conf - conf.max(dim=1, keepdim=True)[0]
        if method == 0:
            # 1 as weight, mean feature, mean conf as mask
            out = x.new_zeros(B * H * W, C + 2)
            conf = conf.exp()
            weight = x.new_ones(B, N, 1)
            tmp = torch.cat([x, conf], dim=-1)
            tmp = tmp * weight
            tmp = torch.cat([tmp, weight], dim=-1)
            out.index_add_(dim=0, index=idx.reshape(B * N), source=tmp.reshape(B * N, C + 2))
            out = out.reshape(B, H, W, C + 2).permute(0, 3, 1, 2).contiguous()

            feature = out[:, :C, :, :]
            conf = out[:, C:C+1, :, :]
            weight = out[:, C+1:, :, :]
            feature = feature / (weight + 1e-6)
            mask = conf / (weight + 1e-6)
        elif method == 1:
            # conf as weight, weighted mean feature, weighted mean conf as mask
            out = x.new_zeros(B * H * W, C + 2)
            conf = conf.exp()
            weight = conf
            tmp = torch.cat([x, conf], dim=-1)
            tmp = tmp * weight
            tmp = torch.cat([tmp, weight], dim=-1)
            out.index_add_(dim=0, index=idx.reshape(B * N), source=tmp.reshape(B * N, C + 2))
            out = out.reshape(B, H, W, C + 2).permute(0, 3, 1, 2).contiguous()

            feature = out[:, :C, :, :]
            conf = out[:, C:C+1, :, :]
            weight = out[:, C+1:, :, :]
            feature = feature / (weight + 1e-6)
            mask = conf / (weight + 1e-6)
    return feature, mask


def token2map(x, loc, map_size, kernel_size, sigma, return_mask=False):
    # if kernel_size == 1:
    #     x, loc = x[:, :16], loc[:, :16]

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

    # t = mask.min()
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
        ax = plt.subplot(2, 5, 1)
        ax.clear()
        ax.imshow(img)
        # ax = plt.subplot(2, 5, 6)
        # ax.clear()
        # ax.imshow(img)
        for lv in range(len(out)):
            ax = plt.subplot(2, 5, lv+2)
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
    lv = int(math.log2(28 / H) + 7 + 1)

    # conf = F.softmax(conf, dim=1)
    conf = conf.exp()
    conf_map = token2map(conf,  map_size=[H, H], loc=loc, kernel_size=3, sigma=2)
    ax = plt.subplot(2, 5, lv)
    ax.clear()
    ax.imshow(conf_map[0, 0].detach().cpu())


def token2critcal(x, loc, loc_critical, return_mask=False):
    B, N, C = x.shape
    k = loc_critical.shape[1]
    dists = square_distance(loc, loc_critical)
    idx = dists.argmin(dim=-1)

    idx = idx + torch.arange(B)[:, None].to(loc.device) * k
    out = x.new_zeros(B * k, C + 1)

    out.index_add_(dim=0, index=idx.reshape(B * N),
                   source=torch.cat([x, x.new_ones(B, N, 1)], dim=-1).reshape(B * N, C + 1))
    out = out.reshape(B, k, C + 1)
    feature = out[:, :, :C]
    mask = out[:, :, C:]
    feature = feature / (mask + 1e-6)
    mask = (mask > 0).float()
    feature = feature * mask

    if return_mask:
        return feature, mask
    return feature


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    dist = src.unsqueeze(2) - dst.unsqueeze(1)
    dist = (dist**2).sum(dim=-1)
    return dist


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


def inter_points(x_src, loc_src, loc_tar):
    B, N, _ = loc_tar.shape

    dists = square_distance(loc_tar, loc_src)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :3], idx[:, :, :3]     # [B, N, 3]

    dist_recip = 1.0 / (dists + 1e-6)

    one_mask = dists == 0
    zero_mask = one_mask.sum(dim=-1) > 0
    dist_recip[zero_mask, :] = 0
    dist_recip[one_mask] = 1
    # t = one_mask.max()

    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    x_tar = torch.sum(index_points(x_src, idx) * weight.view(B, N, 3, 1), dim=2)
    return x_tar


def get_critical_idx(x, k=49):
    # x： [B, N, C]
    value, idx = x.max(dim=1)
    tmp = (x >= value[:, None, :]) * x
    tmp, _ = tmp.max(dim=-1)
    _, idx = torch.topk(tmp, k, -1)
    return idx


def get_gaussian_kernel(kernel_size, sigma, device):
    x_coord = torch.arange(kernel_size, device=device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size).contiguous()
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).contiguous()
    return gaussian_kernel


def get_sample_grid(weight_map):
    B, _, H, W = weight_map.shape
    max_size = max(H, W)
    device = weight_map.device
    dtype = weight_map.dtype

    kernel_size = 2 * max_size - 1
    pad_size = max_size - 1

    kernel_gaussian = get_gaussian_kernel(kernel_size, sigma=3, device=device)

    h, w = kernel_size, kernel_size
    x = torch.arange(w, device=device, dtype=dtype)
    x = (x - 0.5 * (w-1)) * 2 / W
    y = torch.arange(h, device=device, dtype=dtype)
    y = (y - 0.5 * (h-1)) * 2 / H
    y, x = torch.meshgrid(y, x)
    kernel_delta = torch.stack([x, y], dim=-1)
    kernel_delta = kernel_delta.permute(2, 0, 1).unsqueeze(1)

    kernel = torch.cat([kernel_gaussian * kernel_delta, kernel_gaussian], dim=0)

    weight_map = F.pad(weight_map, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
    tmp = F.conv2d(weight_map, kernel, stride=1, padding=0)
    loc_delta, norm_weight = tmp[:, :2], tmp[:, 2:]
    loc_delta = loc_delta / (norm_weight + 1e-6)

    y_g, x_g = torch.arange(H, device=device).float(), torch.arange(W, device=device).float()
    y_g = 2 * ((y_g + 0.5) / H) - 1
    x_g = 2 * ((x_g + 0.5) / W) - 1
    y_map, x_map = torch.meshgrid(y_g, x_g)
    loc = torch.stack((x_map, y_map), dim=-1)
    loc = loc.permute(2, 0, 1)[None, ...]

    loc = loc + loc_delta
    loc = loc.clamp(-1, 1)
    return loc


def get_sample_grid2(weight_map, loc_init):
    B, _, H, W = weight_map.shape
    max_size = max(H, W)
    device = weight_map.device
    dtype = weight_map.dtype

    kernel_size = 2 * max_size - 1
    pad_size = max_size - 1

    kernel_gaussian = get_gaussian_kernel(kernel_size, sigma=3, device=device)

    h, w = kernel_size, kernel_size
    x = torch.arange(w, device=device, dtype=dtype)
    x = (x - 0.5 * (w-1)) * 2 / W
    y = torch.arange(h, device=device, dtype=dtype)
    y = (y - 0.5 * (h-1)) * 2 / H
    y, x = torch.meshgrid(y, x)
    kernel_delta = torch.stack([x, y], dim=-1)
    kernel_delta = kernel_delta.permute(2, 0, 1).unsqueeze(1)

    kernel = torch.cat([kernel_gaussian * kernel_delta, kernel_gaussian], dim=0)

    weight_map = F.pad(weight_map, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
    tmp = F.conv2d(weight_map, kernel, stride=1, padding=0)
    loc_delta, norm_weight = tmp[:, :2], tmp[:, 2:]
    loc_delta = loc_delta / (norm_weight + 1e-6)

    loc_delta = map2token(loc_delta, loc_init)
    loc = loc_init + loc_delta
    loc = loc.clamp(-1, 1)
    return loc


def merge_tokens_old(x, loc, loc_down, weight=None):
    B, N, C = x.shape
    Ns = loc_down.shape[1]

    dists = square_distance(loc, loc_down)
    idx = dists.argmin(axis=2)
    idx = idx + torch.arange(B)[:, None].to(loc.device) * Ns

    if weight is None:
        weight = x.new_ones(B, N, 1)
    tmp = x.new_zeros(B*Ns, C+3)
    source = torch.cat([x * weight, loc * weight, weight], dim=-1)
    source = source.to(x.device).type(x.dtype)
    tmp.index_add_(dim=0, index=idx.reshape(B*N), source=source.reshape(B*N, C+3))
    tmp = tmp.reshape(B, Ns, C+3)

    x_out = tmp[..., :C]
    loc_out = tmp[..., C:C+2]
    norm_weight = tmp[:, :, C+2:]

    # assert norm_weight.min() > 0
    # assert norm_weight.min() > 0

    # print(norm_weight.min())
    if norm_weight.min() <= 0:
        print('norm_weight: '); print(norm_weight.min())
        err_idx = (norm_weight <=0).nonzero()
        print('err_idx: '); print(err_idx)
        bid = err_idx[0, 0]
        print('loc: '); print(loc[bid])
        print('loc down: '); print(loc_down[bid])
        print('idx:'); print(idx[bid])
        print('weight:'); print(weight[bid])
        print('norm_weight:'); print(norm_weight[bid])

        err_mseg = f'norm_weight.min(): {norm_weight.min()}' + \
                   f'err_idx: {err_idx}' + \
                   f'loc: {loc[bid]}' + \
                   f'loc_down: {loc_down}' + \
                   f'idx: {idx[bid]}' + \
                   f'weight: {weight[bid]}' \
                   + f'norm_weight: {norm_weight[bid]}'
        print(err_mseg)
        raise ValueError(err_mseg)

    assert norm_weight.min() > 0

    x_out = x_out / (norm_weight + 1e-4)
    loc_out = loc_out / (norm_weight + 1e-4)

    # t1 = weight.min()
    # t2 = norm_weight.min()

    if torch.isnan(x_out).any():
        save_dict = {
            'x': x,
            'loc': loc,
            'loc_down':loc_down,
            'idx': idx,
            'weight':weight,
            'norm_weight':norm_weight
        }
        for key in save_dict.keys():
            save_dict[key] = save_dict[key].detach().cpu()
        torch.save(save_dict, 'debug_merge.pth')

        with open('debug.txt', 'a') as f:
            f.writelines('merge tokens:')
            f.writelines('merge tokens:')
            f.writelines('merge tokens:')
            f.writelines('merge tokens:')
            f.writelines('norm_weight_min: '); f.writelines(str(norm_weight.min()))
            err_idx = torch.isnan(x_out).nonzero()
            f.writelines('err_idx: '); f.writelines(str(err_idx))
            bid = err_idx[0, 0]
            f.writelines('loc: '); f.writelines(str(loc[bid]))
            f.writelines('loc down: '); f.writelines(str(loc_down[bid]))
            f.writelines('idx:'); f.writelines(str(idx[bid]))
            f.writelines('weight:'); f.writelines(str(weight[bid]))
            f.writelines('norm_weight:'); f.writelines(str(norm_weight[bid]))

            err_mseg = f'norm_weight.min(): {norm_weight.min()}' + \
                       f'err_idx: {err_idx}' + \
                       f'loc: {loc[bid]}' + \
                       f'loc_down: {loc_down}' + \
                       f'idx: {idx[bid]}' + \
                       f'weight: {weight[bid]}' \
                       + f'norm_weight: {norm_weight[bid]}'
            f.writelines(err_mseg)
        raise ValueError(err_mseg)
        exit(-1)

    return x_out, loc_out


def merge_tokens(x, loc, loc_down, weight=None):
    B, N, C = x.shape
    Ns = loc_down.shape[1]

    dists = square_distance(loc, loc_down)
    idx = dists.argmin(axis=2)
    idx = idx + torch.arange(B)[:, None].to(loc.device) * Ns

    if weight is None:
        weight = x.new_ones(B, N, 1)
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-4
    norm_weight = weight / all_weight[idx]

    tmp = x.new_zeros(B * Ns, C + 2)
    source = torch.cat([x * norm_weight, loc * norm_weight], dim=-1)
    source = source.to(x.device).type(x.dtype)
    tmp.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C + 2))
    tmp = tmp.reshape(B, Ns, C + 2)

    x_out = tmp[..., :C]
    loc_out = tmp[..., C:]

    if torch.isinf(x_out).any():
        save_dict = {
            'x': x,
            'loc': loc,
            'loc_down': loc_down,
            'idx': idx,
            'weight': weight,
            'norm_weight': norm_weight,
            'all_weight': all_weight
        }
        for key in save_dict.keys():
            save_dict[key] = save_dict[key].detach().cpu()
        torch.save(save_dict, 'debug_merge.pth')

    return x_out, loc_out

# '''for debug'''
#
# conf_map = torch.ones(2, 1, 28, 28) * 0.5
# conf_map[0, 0, 7:14, 7:14] = 5
# conf_map[0, 0, 3:6, 10:13] = 10
# conf_map[1, 0, 1, 10] = 5
# # conf_map = torch.rand(2, 1, 28, 28)
# # conf_map = guassian_filt(conf_map)
#
# loc = get_sample_grid(conf_map)
# loc = loc.reshape(2, 2, -1).permute(0, 2,  1)
#
#
# ax = plt.subplot(1, 2, 1)
# ax.imshow(conf_map[0, 0].detach().cpu(), extent=[-1, 1, 1, -1])
# ax.scatter(loc[0, :, 0], loc[0, :, 1], c='red', s=0.5)
#
# ax = plt.subplot(1, 2, 2)
# ax.imshow(conf_map[1, 0].detach().cpu(), extent=[-1, 1, 1, -1])
# ax.scatter(loc[1, :, 0], loc[1, :, 1], c='red', s=0.5)
#
# plt.show()
# t = 0


def token2map_with_conf(x, loc, map_size, kernel_size, sigma, conf=None):
    # if kernel_size == 1:
    #     x, loc = x[:, :16], loc[:, :16]
    B, N, C = x.shape

    if conf is None:
        conf = x.new_zeros(B, N, 1)
    weight = conf.exp()

    H, W = map_size
    loc = loc.clamp(-1, 1)
    loc = 0.5 * (loc + 1) * torch.FloatTensor([W, H]).to(loc.device)[None, None, :] - 0.5
    loc = loc.round().long()
    loc[..., 0] = loc[..., 0].clamp(0, W-1)
    loc[..., 1] = loc[..., 1].clamp(0, H-1)
    idx = loc[..., 0] + loc[..., 1] * W
    idx = idx + torch.arange(B)[:, None].to(loc.device) * H * W

    all_weight = weight.new_zeros(B*H*W, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B*N), source=weight.reshape(B*N, 1))
    norm_weight = weight / all_weight[idx]

    out = x.new_zeros(B*H*W, C+1)
    source = torch.cat([x * norm_weight, conf * norm_weight], dim=-1)
    out.index_add_(dim=0, index=idx.reshape(B*N),
                   source=source.reshape(B*N, C+1))
    out = out.reshape(B, H, W, C+1).permute(0, 3, 1, 2).contiguous()
    all_weight = all_weight.reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()

    out, mask = reconstruct_feature2(out, all_weight, kernel_size, sigma)

    feature = out[:, :C, :, :]
    conf = out[:, C:, :, :]
    conf = conf + (1 - mask.type(conf.dtype)) * (-10)
    return feature, conf, mask


def reconstruct_feature2(feature, weight, kernel_size, sigma):
    mask = (weight > 0)
    if kernel_size < 3:
        return feature, mask
    tmp = weight.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    weight = weight / (tmp+1e-6)
    C = feature.shape[1]
    out = guassian_filt(torch.cat([feature * weight, weight], dim=1),
                        kernel_size=kernel_size, sigma=sigma)
    feature_inter = out[:, :C]
    weight_inter = out[:, C:]
    feature_inter = feature_inter / (weight_inter + 1e-6)
    mask_inter = (weight_inter > 0)
    feature_inter = feature_inter * mask_inter.type(feature.dtype)
    out = feature + (1 - mask.type(feature.dtype)) * feature_inter

    return out, mask_inter


def merge_tokens2(x, loc, loc_down, weight=None):
    """
    merge tokens with 2 tokens
    """

    B, N, C = x.shape
    Ns = loc_down.shape[1]
    K = 2

    dists = square_distance(loc, loc_down)
    idx = dists.sort(dim=2)[1]
    idx = idx[:, :, :K]
    # idx = torch.zeros(B, N, K).long().to(x.device)
    idx = idx + torch.arange(B)[:, None, None].to(loc.device) * Ns

    if weight is None:
        weight = x.new_ones(B, N, 1)

    all_weight = weight.new_zeros(B * Ns, 1)
    weight = weight.expand(B, N, K)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N * K), source=weight.reshape(B * N * K, 1))

    all_weight = all_weight + 1e-4
    t_w = all_weight[idx.reshape(-1), 0].reshape(B, N, K)
    norm_weight = weight / t_w

    tmp = x.new_zeros(B * Ns, C + 2)
    source = torch.cat([x.unsqueeze(-2) * norm_weight.unsqueeze(-1),
                        loc.unsqueeze(-2) * norm_weight.unsqueeze(-1)], dim=-1)
    source = source.to(x.device).type(x.dtype)
    tmp.index_add_(dim=0, index=idx.reshape(B * N * K), source=source.reshape(B * N * K, C + 2))
    tmp = tmp.reshape(B, Ns, C + 2)

    x_out = tmp[..., :C]
    loc_out = tmp[..., C:]

    if torch.isinf(x_out).any():
        save_dict = {
            'x': x,
            'loc': loc,
            'loc_down': loc_down,
            'idx': idx,
            'weight': weight,
            'norm_weight': norm_weight,
            'all_weight': all_weight
        }
        for key in save_dict.keys():
            save_dict[key] = save_dict[key].detach().cpu()
        torch.save(save_dict, 'debug_merge.pth')

    return x_out, loc_out


def merge_tokens_conv(x, loc, loc_down, weight, conv_weight, bias):
    B, N, C_in = x.shape
    C_out, C_in, kh, kw = conv_weight.shape
    Ns = loc_down.shape[1]

    x_o = torch.einsum('bni,oihw->bnohw', x, conv_weight)

    dists = square_distance(loc, loc_down)
    idx = dists.argmin(axis=2)
    idx = idx + torch.arange(B)[:, None].to(loc.device) * Ns

    if weight is None:
        weight = x.new_ones(B, N, 1)
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-4
    norm_weight = weight / all_weight[idx]

    loc_out = loc.new_zeros(B * Ns, 2)
    source = loc * norm_weight
    loc_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, 2))
    loc_delta = loc - loc_out[idx]

    loc_out = loc_out.reshape(B, Ns, 2)

    delta_matrix = loc_out[:, :, None, :] - loc[:, None, :, :]
    mask = delta_matrix.new_zeros(B * Ns, N)
    tmp = torch.arange(N)[None, :].expand(B, N).to(x.device)
    mask[idx.reshape(-1), tmp.reshape(-1)] = 1
    mask = mask.reshape(B, Ns, N, 1)
    delta_matrix = delta_matrix * mask

    box_scale = delta_matrix.abs().max(dim=2)[0]
    box_scale = box_scale.max(dim=-1)[0] + 1e-4
    box_scale = box_scale.reshape(-1, 1)
    loc_delta = loc_delta / box_scale[idx]

    x = map2token(x_o.flatten(0, 1),
                  loc_delta.reshape(B * N, 1, 2))
    x = x.reshape(B, N, 1, C_out).squeeze(-2)

    x_out = x.new_zeros(B * Ns, C_out)
    source = x * norm_weight
    source = source.to(x.device).type(x.dtype)
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C_out))
    x_out = x_out.reshape(B, Ns, C_out)

    x_o = x_o[:, :, :, 1, 1]
    if bias is not None:
        x_out = x_out + bias[None, None, :]
        x_o = x_o + bias[None, None, :]

    return x_out, loc_out, x_o


def merge_tokens_agg(x, loc, loc_down, idx_agg, weight=None, return_weight=False):
    B, N, C = x.shape
    Ns = loc_down.shape[1]

    dists = square_distance(loc, loc_down)
    idx_agg_t = dists.argmin(axis=2)
    idx = idx_agg_t + torch.arange(B)[:, None].to(loc.device) * Ns

    if weight is None:
        weight = x.new_ones(B, N, 1)
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-4
    norm_weight = weight / all_weight[idx]

    tmp = x.new_zeros(B * Ns, C + 2)
    source = torch.cat([x * norm_weight, loc * norm_weight], dim=-1)
    source = source.to(x.device).type(x.dtype)
    tmp.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C + 2))
    tmp = tmp.reshape(B, Ns, C + 2)

    x_out = tmp[..., :C]
    loc_out = tmp[..., C:]
    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    if torch.isinf(x_out).any():
        save_dict = {
            'x': x,
            'loc': loc,
            'loc_down': loc_down,
            'idx': idx,
            'weight': weight,
            'norm_weight': norm_weight,
            'all_weight': all_weight
        }
        for key in save_dict.keys():
            save_dict[key] = save_dict[key].detach().cpu()
        torch.save(save_dict, 'debug_merge.pth')

    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, loc_out, idx_agg, weight_t
    return x_out, loc_out, idx_agg



# def token2map_agg(x, loc, loc_orig, idx_agg, map_size):
#     # x = torch.rand(2, 4, 3)
#     # loc = torch.rand(2, 4, 2)
#     # loc_orig = torch.rand(2, 7, 2)
#     # idx_agg = (torch.rand(2, 7) * 3).long()
#     # map_size = [5, 5]
#
#     H, W = map_size
#     B, N, C = x.shape
#     N0 = loc_orig.shape[1]
#     device = x.device
#     loc_orig = loc_orig.clamp(-1, 1)
#     loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
#     loc_orig = loc_orig.round().long()
#     loc_orig[..., 0] = loc_orig[..., 0].clamp(0, W-1)
#     loc_orig[..., 1] = loc_orig[..., 1].clamp(0, H-1)
#     idx_HW_orig = loc_orig[..., 0] + loc_orig[..., 1] * W
#     idx_HW_orig = idx_HW_orig + torch.arange(B)[:, None].to(device) * H * W
#
#     weight = x.new_ones(B, N, 1)
#     source = index_points(torch.cat([x, weight], dim=-1), idx_agg)
#     out = x.new_zeros(B*H*W, C+1)
#     out.index_add_(dim=0, index=idx_HW_orig.reshape(B*N0),
#                    source=source.reshape(B*N0, C+1))
#     x_out = out[:, :C]
#     all_weight = out[:, C:]
#     x_out = x_out / (all_weight + 1e-6)
#
#     x_out = x_out.reshape(B, H, W, C)
#     return x_out, all_weight
#
#
# def map2token_agg(feature_map, loc_xy, loc_orig, idx_agg, mode='bilinear', align_corners=False):
#     N = loc_xy.shape[1]
#     B, N0, _ = loc_orig.shape
#     C = feature_map.shape[-1]
#     device = feature_map.device
#
#     loc_orig = loc_orig.unsqueeze(1).type(feature_map.dtype)
#     tokens_orig = F.grid_sample(feature_map, loc_orig, mode=mode, align_corners=align_corners)
#     tokens_orig = tokens_orig.permute(0, 2, 3, 1).squeeze(1).contiguous()
#
#     idx_tokens = idx_agg + torch.arange(B)[:, None].to(device) * N
#
#     out = feature_map.new_zeros(B * N, C + 1)
#     weight = tokens_orig.new_ones(B, N0, 1)
#     source = torch.cat([tokens_orig, weight], dim=-1)
#     out.index_add_(dim=0, index=idx_tokens.reshape(B * N0),
#                    source=source.reshape(B * N0, C + 1))
#     tokens = out[:, :C]
#     all_weight = out[:, C:]
#     tokens = tokens / (all_weight + 1e-6)
#     tokens = tokens.reshape(B, N, C)
#     return tokens
#


def token2map_agg_sparse(x, loc, loc_orig, idx_agg, map_size, weight=None, kernel=1, sigma=2):
    # x = torch.rand(2, 4, 3).half()
    # loc = torch.rand(2, 4, 2)
    # loc_orig = torch.rand(2, 7, 2)
    # idx_agg = (torch.rand(2, 7) * 3).long()
    # map_size = [5, 5]
    # weight = None

    H, W = map_size
    B, N, C = x.shape
    N0 = loc_orig.shape[1]
    device = x.device
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

    A = torch.sparse.FloatTensor(coor, value, torch.Size([B*H*W, B*N]))

    all_weight = A.type(torch.float32) @ x.new_ones(B*N, 1).type(torch.float32) + 1e-6
    all_weight = all_weight.type(x.dtype)
    value = value / all_weight[idx_HW_orig.reshape(-1), 0]

    if kernel > 1 and C > N:
        A = x.new_zeros(B * H * W, N)
        A[idx_HW_orig.reshape(-1), idx_agg.reshape(-1)] = value.reshape(-1)
        A = A.reshape(B, H, W, N).permute(0, 3, 1, 2)
        A = guassian_filt(A, kernel, sigma)
        A = A.flatten(2).permute(0, 2, 1)
        x_out = A @ x
        x_out = x_out.type(x.dtype)
        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        all_weight = all_weight.reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
    else:
        A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))
        x_out = A.type(torch.float32) @ x.reshape(B*N, C).type(torch.float32)
        x_out = x_out.type(x.dtype)
        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        all_weight = all_weight.reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
        if kernel > 1:
            x_out = guassian_filt(x_out, kernel, sigma)

    return x_out, all_weight


def token2map_agg_mat(x, loc, loc_orig, idx_agg, map_size, weight=None, agg_weight=None):
    # x = torch.rand(2, 4, 3).half()
    # loc = torch.rand(2, 4, 2)
    # loc_orig = torch.rand(2, 7, 2)
    # idx_agg = (torch.rand(2, 7) * 3).long()
    # map_size = [5, 5]
    # weight = None

    dtype = x.dtype
    H, W = map_size
    B, N, C = x.shape
    N0 = loc_orig.shape[1]
    device = x.device

    if N0 == N and N == H * W:
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous(), x.new_ones(B, 1, H, W)

    loc_orig = loc_orig.clamp(-1, 1)
    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    loc_orig = loc_orig.round().long()
    loc_orig[..., 0] = loc_orig[..., 0].clamp(0, W-1)
    loc_orig[..., 1] = loc_orig[..., 1].clamp(0, H-1)
    idx_HW_orig = loc_orig[..., 0] + loc_orig[..., 1] * W

    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N0)
    if weight is None:
        weight = x.new_ones(B, N, 1)
    value = index_points(weight, idx_agg).reshape(B*N0)
    if agg_weight is not None:
        value = value * agg_weight.reshape(B*N0).type(x.dtype)

    A = x.new_zeros(B, H*W, N)
    A[idx_batch.reshape(-1), idx_HW_orig.reshape(-1), idx_agg.reshape(-1)] = value.reshape(-1)
    all_weight = (A.sum(dim=-1, keepdim=True) +1e-6)
    A = A / all_weight
    x_out = A @ x

    x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    all_weight = all_weight.reshape(B, H, W, 1).permute(0, 3,  1, 2).contiguous()
    return x_out, all_weight


# def map2token_agg_sparse(feature_map, loc, loc_orig, idx_agg, weight=None):
#
#     ''' sparse can not be multiply with sparse'''
#     feature_map = torch.rand(2, 3, 5, 5)
#     loc = torch.rand(2, 4, 2)
#     loc_orig = torch.rand(2, 7, 2) - 0.5
#     idx_agg = (torch.rand(2, 7) * 3).long()
#     weight = None
#
#
#     B, C, H, W = feature_map.shape
#     device = feature_map.device
#     N = loc.shape[1]
#     N0 = loc_orig.shape[1]
#
#
#     loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
#     x = loc_orig[:, :, 0].reshape(-1)
#     y = loc_orig[:, :, 1].reshape(-1)
#
#     h, w = H, W
#
#     x_grid = x
#     x_lo = x_grid.floor().long().clamp(min=0, max=w - 1)
#     x_hi = (x_lo + 1).clamp(max=w - 1)
#     x_grid = torch.min(x_hi.float(), x_grid)
#     x_w = x_grid - x_lo.float()
#
#     y_grid = y
#     y_lo = y_grid.floor().long().clamp(min=0, max=h - 1)
#     y_hi = (y_lo + 1).clamp(max=h - 1)
#     y_grid = torch.min(y_hi.float(), y_grid)
#     y_w = y_grid - y_lo.float()
#
#     w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
#     w_ylo_xhi = x_w * (1.0 - y_w)
#     w_yhi_xlo = (1.0 - x_w) * y_w
#     w_yhi_xhi = x_w * y_w
#
#     i_ylo_xlo = (y_lo * w + x_lo).detach()
#     i_ylo_xhi = (y_lo * w + x_hi).detach()
#     i_yhi_xlo = (y_hi * w + x_lo).detach()
#     i_yhi_xhi = (y_hi * w + x_hi).detach()
#
#     idx_HW_orig = torch.stack([i_ylo_xlo, i_ylo_xhi, i_yhi_xlo, i_yhi_xhi], dim=1)
#     idx_HW_orig = idx_HW_orig + torch.arange(B)[:, None].expand(B, N0).to(device).reshape(B*N0, 1) * H * W
#
#     idx_tokens_orig = torch.arange(B*N0).to(device)[:, None].expand(B*N0, 4)
#     value = torch.stack([w_ylo_xlo, w_ylo_xhi, w_yhi_xlo, w_yhi_xhi], dim=1)
#
#     coor = torch.stack([idx_tokens_orig.reshape(-1), idx_HW_orig.reshape(-1)], dim=0)
#     value = value.reshape(-1)
#     A = torch.sparse.FloatTensor(coor, value, torch.Size([B*N0, B*H*W]))   # [B*N0, B*H*W]
#
#     idx_tokens = idx_agg + torch.arange(B)[:, None].to(device) * N
#     idx_tokens = idx_tokens.reshape(-1)
#     coor1 = torch.stack([idx_tokens, torch.arange(B*N0).to(device)], dim=0)
#     if weight is None:
#         weight = feature_map.new_ones(B, N0, 1)
#     value1 = weight.reshape(-1)
#
#     A1 = torch.sparse.FloatTensor(coor1, value1, torch.Size([B*N, B*N0]))  # [B*N, B*N0]
#     A = A1 @ A                                                           # [B*N, B*HW]
#
#     A2 = feature_map.new_zeros(B*N, B*H*W)
#     A2.index_add(dim=0, index=idx_tokens, source=A)
#
#     return tokens

def map2token_agg_mat_nearest(feature_map, loc, loc_orig, idx_agg, weight=None):
    ''' realized by 2 attention matrix'''
    # feature_map = torch.rand(2, 3, 5, 5)
    # loc = torch.rand(2, 4, 2)
    # loc_orig = torch.rand(2, 7, 2) - 0.5
    # idx_agg = (torch.rand(2, 7) * 3).long()
    # weight = None

    B, C, H, W = feature_map.shape
    device = feature_map.device
    N = loc.shape[1]
    N0 = loc_orig.shape[1]

    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    x = loc_orig[:, :, 0].reshape(-1)
    y = loc_orig[:, :, 1].reshape(-1)

    h, w = H, W
    x_grid = x.round().long().clamp(min=0, max=w - 1)
    y_grid = y.round().long().clamp(min=0, max=h - 1)
    idx_HW_orig = (y_grid * w + x_grid).detach()
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N0)
    idx_tokens_orig = torch.arange(N0, device=device)[None, :].expand(B, N0)
    value = feature_map.new_ones(B, N0)

    # this will cause error on edges where the four pixel is the same one.
    # the weight is not the sum but the last one (usually 0)
    # A = feature_map.new_zeros(B, N0, H*W)
    # A[idx_batch.reshape(-1), idx_tokens_orig.reshape(-1), idx_HW_orig.reshape(-1)] = value.reshape(-1)
    #

    indices = torch.stack([idx_batch.reshape(-1), idx_tokens_orig.reshape(-1), idx_HW_orig.reshape(-1)], dim=0)
    A = torch.sparse_coo_tensor(indices, value.reshape(-1), (B, N0, H*W))
    A = A.to_dense()


    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N0)
    idx_tokens_orig = torch.arange(N0, device=device)[None, :].expand(B, N0)
    if weight is None:
        weight = feature_map.new_ones(B, N0, 1)

    indices = torch.stack([idx_batch.reshape(-1), idx_agg.reshape(-1), idx_tokens_orig.reshape(-1)], dim=0)
    A1 = torch.sparse_coo_tensor(indices, weight.reshape(-1).type(feature_map.dtype), (B, N, N0))
    A1 = A1.to_dense()
    A1 = A1 / (A1.sum(dim=-1, keepdim=True) +1e-6)

    # A1 = feature_map.new_zeros(B, N, N0)
    # A1[idx_batch.reshape(-1), idx_agg.reshape(-1), idx_tokens_orig.reshape(-1)] = weight.reshape(-1).type(feature_map.dtype)
    # A1 = A1 / (A1.sum(dim=-1, keepdim=True) +1e-6)

    A = A1 @ A

    tokens = A @ feature_map.flatten(2).permute(0, 2, 1)

    return tokens


def map2token_agg_mat(feature_map, loc, loc_orig, idx_agg, weight=None):
    ''' realized by 2 attention matrix'''
    # feature_map = torch.rand(2, 3, 5, 5)
    # loc = torch.rand(2, 4, 2)
    # loc_orig = torch.rand(2, 7, 2) - 0.5
    # idx_agg = (torch.rand(2, 7) * 3).long()
    # weight = None

    B, C, H, W = feature_map.shape
    device = feature_map.device
    N = loc.shape[1]
    N0 = loc_orig.shape[1]

    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    x = loc_orig[:, :, 0].reshape(-1)
    y = loc_orig[:, :, 1].reshape(-1)

    h, w = H, W

    x_grid = x
    x_lo = x_grid.floor().long().clamp(min=0, max=w - 1)
    x_hi = (x_lo + 1).clamp(max=w - 1)
    x_grid = torch.min(x_hi.float(), x_grid)
    x_w = x_grid - x_lo.float()

    y_grid = y
    y_lo = y_grid.floor().long().clamp(min=0, max=h - 1)
    y_hi = (y_lo + 1).clamp(max=h - 1)
    y_grid = torch.min(y_hi.float(), y_grid)
    y_w = y_grid - y_lo.float()

    w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
    w_ylo_xhi = x_w * (1.0 - y_w)
    w_yhi_xlo = (1.0 - x_w) * y_w
    w_yhi_xhi = x_w * y_w

    i_ylo_xlo = (y_lo * w + x_lo).detach()
    i_ylo_xhi = (y_lo * w + x_hi).detach()
    i_yhi_xlo = (y_hi * w + x_lo).detach()
    i_yhi_xhi = (y_hi * w + x_hi).detach()

    idx_HW_orig = torch.stack([i_ylo_xlo, i_ylo_xhi, i_yhi_xlo, i_yhi_xhi], dim=1)
    idx_batch = torch.arange(B, device=device)[:, None, None].expand(B, N0, 4)
    idx_tokens_orig = torch.arange(N0, device=device)[None, :, None].expand(B, N0, 4)
    value = torch.stack([w_ylo_xlo, w_ylo_xhi, w_yhi_xlo, w_yhi_xhi], dim=1).type(feature_map.dtype)


    # this will cause error on edges where the four pixel is the same one.
    # the weight is not the sum but the last one (usually 0)
    # A = feature_map.new_zeros(B, N0, H*W)
    # A[idx_batch.reshape(-1), idx_tokens_orig.reshape(-1), idx_HW_orig.reshape(-1)] = value.reshape(-1)
    #

    indices = torch.stack([idx_batch.reshape(-1), idx_tokens_orig.reshape(-1), idx_HW_orig.reshape(-1)], dim=0)
    A = torch.sparse_coo_tensor(indices, value.reshape(-1), (B, N0, H*W))
    A = A.to_dense()


    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N0)
    idx_tokens_orig = torch.arange(N0, device=device)[None, :].expand(B, N0)
    if weight is None:
        weight = feature_map.new_ones(B, N0, 1)

    indices = torch.stack([idx_batch.reshape(-1), idx_agg.reshape(-1), idx_tokens_orig.reshape(-1)], dim=0)
    A1 = torch.sparse_coo_tensor(indices, weight.reshape(-1).type(feature_map.dtype), (B, N, N0))
    A1 = A1.to_dense()
    A1 = A1 / (A1.sum(dim=-1, keepdim=True) +1e-6)

    # A1 = feature_map.new_zeros(B, N, N0)
    # A1[idx_batch.reshape(-1), idx_agg.reshape(-1), idx_tokens_orig.reshape(-1)] = weight.reshape(-1).type(feature_map.dtype)
    # A1 = A1 / (A1.sum(dim=-1, keepdim=True) +1e-6)

    A = A1 @ A

    tokens = A @ feature_map.flatten(2).permute(0, 2, 1)

    return tokens


def map2token_agg_mat_bug(feature_map, loc, loc_orig, idx_agg, weight=None):
    ''' realized by 2 attention matrix'''
    # feature_map = torch.rand(2, 3, 5, 5)
    # loc = torch.rand(2, 4, 2)
    # loc_orig = torch.rand(2, 7, 2) - 0.5
    # idx_agg = (torch.rand(2, 7) * 3).long()
    # weight = None

    B, C, H, W = feature_map.shape
    device = feature_map.device
    N = loc.shape[1]
    N0 = loc_orig.shape[1]

    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    x = loc_orig[:, :, 0].reshape(-1)
    y = loc_orig[:, :, 1].reshape(-1)

    h, w = H, W

    x_grid = x
    x_lo = x_grid.floor().long().clamp(min=0, max=w - 1)
    x_hi = (x_lo + 1).clamp(max=w - 1)
    x_grid = torch.min(x_hi.float(), x_grid)
    x_w = x_grid - x_lo.float()

    y_grid = y
    y_lo = y_grid.floor().long().clamp(min=0, max=h - 1)
    y_hi = (y_lo + 1).clamp(max=h - 1)
    y_grid = torch.min(y_hi.float(), y_grid)
    y_w = y_grid - y_lo.float()

    w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
    w_ylo_xhi = x_w * (1.0 - y_w)
    w_yhi_xlo = (1.0 - x_w) * y_w
    w_yhi_xhi = x_w * y_w

    i_ylo_xlo = (y_lo * w + x_lo).detach()
    i_ylo_xhi = (y_lo * w + x_hi).detach()
    i_yhi_xlo = (y_hi * w + x_lo).detach()
    i_yhi_xhi = (y_hi * w + x_hi).detach()

    idx_HW_orig = torch.stack([i_ylo_xlo, i_ylo_xhi, i_yhi_xlo, i_yhi_xhi], dim=1)
    idx_batch = torch.arange(B, device=device)[:, None, None].expand(B, N0, 4)
    idx_tokens_orig = torch.arange(N0, device=device)[None, :, None].expand(B, N0, 4)
    value = torch.stack([w_ylo_xlo, w_ylo_xhi, w_yhi_xlo, w_yhi_xhi], dim=1).type(feature_map.dtype)


    # this will cause error on edges where the four pixel is the same one.
    # the weight is not the sum but the last one (usually 0)
    A = feature_map.new_zeros(B, N0, H*W)
    A[idx_batch.reshape(-1), idx_tokens_orig.reshape(-1), idx_HW_orig.reshape(-1)] = value.reshape(-1)


    # indices = torch.stack([idx_batch.reshape(-1), idx_tokens_orig.reshape(-1), idx_HW_orig.reshape(-1)], dim=0)
    # A = torch.sparse_coo_tensor(indices, value.reshape(-1), (B, N0, H*W))
    # A = A.to_dense()


    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N0)
    idx_tokens_orig = torch.arange(N0, device=device)[None, :].expand(B, N0)
    if weight is None:
        weight = feature_map.new_ones(B, N0, 1)

    indices = torch.stack([idx_batch.reshape(-1), idx_agg.reshape(-1), idx_tokens_orig.reshape(-1)], dim=0)
    A1 = torch.sparse_coo_tensor(indices, weight.reshape(-1).type(feature_map.dtype), (B, N, N0))
    A1 = A1.to_dense()

    # A1 = feature_map.new_zeros(B, N, N0)
    # A1[idx_batch.reshape(-1), idx_agg.reshape(-1), idx_tokens_orig.reshape(-1)] = weight.reshape(-1).type(feature_map.dtype)
    # A1 = A1 / (A1.sum(dim=-1, keepdim=True) +1e-6)

    A = A1 @ A

    tokens = A @ feature_map.flatten(2).permute(0, 2, 1)

    return tokens


'''merge according to feature cosine similarity'''
def merge_tokens_agg_cosine(x, loc, index_down, x_down, idx_agg, weight=None, return_weight=False):
    B, N, C = x.shape
    Ns = x_down.shape[1]

    cos_sim = F.cosine_similarity(x[:, :, None, :], x_down[:, None, :, :], dim=-1)
    idx_agg_t = cos_sim.argmax(axis=2)

    # make sure selected tokens merge to itself
    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
    idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
    idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    idx = idx_agg_t + torch.arange(B)[:, None].to(loc.device) * Ns

    if weight is None:
        weight = x.new_ones(B, N, 1)
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-4
    norm_weight = weight / all_weight[idx]

    tmp = x.new_zeros(B * Ns, C + 2)
    source = torch.cat([x * norm_weight, loc * norm_weight], dim=-1)
    source = source.to(x.device).type(x.dtype)
    tmp.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C + 2))
    tmp = tmp.reshape(B, Ns, C + 2)

    x_out = tmp[..., :C]
    loc_out = tmp[..., C:]
    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    if torch.isinf(x_out).any():
        save_dict = {
            'x': x,
            'loc': loc,
            'index_down': index_down,
            'x_down': x_down,
            'idx': idx,
            'weight': weight,
            'norm_weight': norm_weight,
            'all_weight': all_weight
        }
        for key in save_dict.keys():
            save_dict[key] = save_dict[key].detach().cpu()
        torch.save(save_dict, 'debug_merge_cosine.pth')

    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, loc_out, idx_agg, weight_t
    return x_out, loc_out, idx_agg


'''merge according to feature distance'''
def merge_tokens_agg_dist(x, loc, index_down, x_down, idx_agg, weight=None, return_weight=False):
    B, N, C = x.shape
    Ns = x_down.shape[1]

    # cos_sim = F.cosine_similarity(x[:, :, None, :], x_down[:, None, :, :], dim=-1)
    # idx_agg_t = cos_sim.argmax(axis=2)

    # dist = x.unsqueeze(2) - x_down.unsqueeze(1)
    # dist = dist.norm(p=2, dim=-1)
    # idx_agg_t = dist.argmin(axis=2)
    idx_agg_t = torch.cdist(x, x_down, p=2).argmin(axis=2)

    # make sure selected tokens merge to itself
    if index_down is not None:
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    idx = idx_agg_t + torch.arange(B)[:, None].to(loc.device) * Ns

    if weight is None:
        weight = x.new_ones(B, N, 1)
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-4
    norm_weight = weight / all_weight[idx]

    tmp = x.new_zeros(B * Ns, C + 2)
    source = torch.cat([x * norm_weight, loc * norm_weight], dim=-1)
    source = source.to(x.device).type(x.dtype)
    tmp.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C + 2))
    tmp = tmp.reshape(B, Ns, C + 2)

    x_out = tmp[..., :C]
    loc_out = tmp[..., C:]
    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    if torch.isinf(x_out).any():
        save_dict = {
            'x': x,
            'loc': loc,
            'index_down': index_down,
            'x_down': x_down,
            'idx': idx,
            'weight': weight,
            'norm_weight': norm_weight,
            'all_weight': all_weight
        }
        for key in save_dict.keys():
            save_dict[key] = save_dict[key].detach().cpu()
        torch.save(save_dict, 'debug_merge_cosine.pth')

    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, loc_out, idx_agg, weight_t
    return x_out, loc_out, idx_agg

'''clean merge'''
def merge_tokens_agg_dist2(x, index_down, x_down, idx_agg, weight=None):
    B, N, C = x.shape
    Ns = x_down.shape[1]

    # cos_sim = F.cosine_similarity(x[:, :, None, :], x_down[:, None, :, :], dim=-1)
    # idx_agg_t = cos_sim.argmax(axis=2)

    # dist = x.unsqueeze(2) - x_down.unsqueeze(1)
    # dist = dist.norm(p=2, dim=-1)
    # idx_agg_t = dist.argmin(axis=2)
    idx_agg_t = torch.cdist(x, x_down, p=2).argmin(axis=2)

    # make sure selected tokens merge to itself
    if index_down is not None:
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    idx = idx_agg_t + torch.arange(B)[:, None].to(x.device) * Ns

    if weight is None:
        weight = x.new_ones(B, N, 1)
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    tmp = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    source = source.to(x.device).type(x.dtype)
    tmp.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C))
    tmp = tmp.reshape(B, Ns, C)
    x_out = tmp[..., :C]
    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    weight_t = index_points(norm_weight, idx_agg)
    return x_out, idx_agg, weight_t


'''merge according to qkv'''
def merge_tokens_agg_qkv(q, k, v, index_down, idx_agg, weight=None, return_weight=False):
    B, N, C = v.shape
    Ns = q.shape[1]

    scale = q.shape[-1] ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale

    # gumble argmax
    p_value = 1e-6
    noise = torch.rand_like(attn)
    noise = -1 * (noise + p_value).log()
    noise = -1 * (noise + p_value).log()
    idx_agg_t = (attn + noise).argmax(axis=1)
    # idx_agg_t = attn.argmax(axis=1)

    # make sure selected tokens merge to itself
    idx_batch = torch.arange(B, device=v.device)[:, None].expand(B, Ns)
    idx_tmp = torch.arange(Ns, device=v.device)[None, :].expand(B, Ns)
    idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    idx_batch = torch.arange(B, device=v.device)[:, None].expand(B, N)
    idx_tokens = torch.arange(N, device=v.device)[None, :].expand(B, N)
    indices = torch.stack([idx_batch.reshape(-1), idx_agg_t.reshape(-1), idx_tokens.reshape(-1)], dim=0)
    value = v.new_ones(B*N)
    mask = torch.sparse_coo_tensor(indices, value, (B, Ns, N))
    mask = mask.to_dense()

    attn = attn.softmax(dim=-1)
    attn = mask * attn
    if weight is not None:
        attn = attn * weight[:, None, :]
    attn = attn / attn.sum(dim=-1, keepdim=True)

    x_out = (attn @ v).transpose(1, 2).reshape(B, Ns, C)
    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    if return_weight:
        norm_weight = attn[indices[0], indices[1], indices[2]]
        norm_weight = norm_weight.reshape(B, N, 1)
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, idx_agg, weight_t
    return x_out, idx_agg



def conf_resample(conf_map, N):
    B, C, H, W = conf_map.shape
    conf = conf_map.flatten(2).permute(0, 2, 1)
    loc = get_grid_loc(B, H, W, conf_map.device)

    index_down = gumble_top_k(conf, N, 1, T=1)
    loc_down = torch.gather(loc, 1, index_down.expand([B, N, 2]))
    return loc_down


'''copied from point net ++'''
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.cdist(xyz, centroid).squeeze(-1)
        # dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def farthest_point_sample_try(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    dists = torch.cdist(xyz, xyz)
    idx_tmp = torch.arange(N, device=device)
    dists[:, idx_tmp, idx_tmp] = dists.max() + 1
    dists = dists.min(dim=-1)[0]
    _, index = torch.topk(dists, k=npoint, dim=-1)
    return index


def feature_try_sample(xyz, npoint):
    """
    find the feature far away from the mean
    """
    # dist = dist.norm(p=2, dim=-1)
    # idx_agg_t = dist.argmin(axis=2)
    dist = (xyz - xyz.mean(dim=1, keepdim=True)).norm(p=2, dim=-1)
    _, centroids = torch.topk(dist, npoint, dim=1)
    return centroids


'''merge according to feature distance'''
def merge_tokens_agg_dist_multi(x, index_down, x_down, weight=None, k=3):
    B, N, C = x.shape
    Ns = x_down.shape[1]
    if weight is None:
        weight = x.new_ones(B, N, 1)

    dists, idx_agg_t = (-1 * torch.cdist(x, x_down, p=2)).topk(k, dim=2)

    dist_recip = 1.0 / (dists + 1e-6)
    one_mask = dists == 0
    zero_mask = one_mask.sum(dim=-1) > 0
    dist_recip[zero_mask, :] = 0
    dist_recip[one_mask] = 1
    norm = torch.sum(dist_recip, dim=2, keepdim=True)  #+ 1e-6
    weight = (dist_recip / norm) * weight

    # weight = weight.expand(B, N, k)

    idx_batch = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
    idx_token = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)

    indices = torch.stack([idx_batch.reshape(-1), idx_agg_t.reshape(-1), idx_token.reshape(-1)], dim=0)
    A = torch.sparse_coo_tensor(indices,  weight.reshape(-1), (B, Ns, N))
    A = A.to_dense()
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

    x_down = A @ x

    return x_down, A


# # normalize for 2 matrix, so every orig token share the same weight when get feature map
# def token2map_Agg2(x, loc_orig, Agg, map_size, weight=None):
#     H, W = map_size
#     B, N, C = x.shape
#     N0 = loc_orig.shape[1]
#     device = x.device
#     loc_orig = loc_orig.clamp(-1, 1)
#     loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
#     loc_orig = loc_orig.round().long()
#     loc_orig[..., 0] = loc_orig[..., 0].clamp(0, W-1)
#     loc_orig[..., 1] = loc_orig[..., 1].clamp(0, H-1)
#     idx_HW_orig = loc_orig[..., 0] + loc_orig[..., 1] * W
#     idx_batch = torch.arange(B, device=device)[:, None].expand(B, N0)
#     idx_token_orig = torch.arange(N0, device=device)[None, :].expand(B, N0)
#
#     indices = torch.stack([idx_batch.reshape(-1), idx_HW_orig.reshape(-1), idx_token_orig.reshape(-1)], dim=0)
#     A = torch.sparse_coo_tensor(indices, x.new_ones(B * N0), (B, H*W, N0))
#     A = A.to_dense()                    # B, HW, N0
#     A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
#
#     Agg = Agg * weight          # B, N, N0
#     Agg = Agg / Agg.sum(dim=1, keepdim=True)   # normalize along N axis
#
#     A = A @ Agg.permute(0, 2, 1)        # B, HW, N
#
#     x_out = A @ x
#     x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
#     return x_out
#

# normalize for the last matrix, so orig token share the DIFFERENT weight when get feature map

def token2map_Agg(x, Agg, loc_orig, map_size, weight=None):
    H, W = map_size
    B, N, C = x.shape
    N0 = loc_orig.shape[1]
    device = x.device
    loc_orig = loc_orig.clamp(-1, 1)
    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    loc_orig = loc_orig.round().long()
    loc_orig[..., 0] = loc_orig[..., 0].clamp(0, W - 1)
    loc_orig[..., 1] = loc_orig[..., 1].clamp(0, H - 1)
    idx_HW_orig = loc_orig[..., 0] + loc_orig[..., 1] * W
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N0)
    idx_token_orig = torch.arange(N0, device=device)[None, :].expand(B, N0)

    indices = torch.stack([idx_batch.reshape(-1), idx_HW_orig.reshape(-1), idx_token_orig.reshape(-1)], dim=0)
    A = torch.sparse_coo_tensor(indices, x.new_ones(B * N0), (B, H * W, N0))
    A = A.to_dense()  # B, HW, N0
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

    if weight is None:
        weight = x.new_ones(B, N, 1)
    Agg = Agg * weight  # B, N, N0
    Agg = Agg / Agg.sum(dim=1, keepdim=True)  # normalize along N axis
    A = A @ Agg.permute(0, 2, 1)  # B, HW, N

    x_out = A @ x
    x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    return x_out


def map2token_Agg(feature_map, Agg, loc_orig):
    B, C, H, W = feature_map.shape
    device = feature_map.device
    _, N, N0 = Agg.shape

    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    x = loc_orig[:, :, 0].reshape(-1)
    y = loc_orig[:, :, 1].reshape(-1)

    h, w = H, W
    x_grid = x.round().long().clamp(min=0, max=w - 1)
    y_grid = y.round().long().clamp(min=0, max=h - 1)
    idx_HW_orig = (y_grid * w + x_grid).detach()
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N0)
    idx_tokens_orig = torch.arange(N0, device=device)[None, :].expand(B, N0)
    value = feature_map.new_ones(B, N0)

    indices = torch.stack([idx_batch.reshape(-1), idx_tokens_orig.reshape(-1), idx_HW_orig.reshape(-1)], dim=0)
    A = torch.sparse_coo_tensor(indices, value.reshape(-1), (B, N0, H*W))
    A = A.to_dense()        # B, N0, H*W (already normalized)

    Agg = Agg / (Agg.sum(dim=-1, keepdim=True) + 1e-6)  # normalize

    A = Agg @ A     # B, N, H*W
    tokens = A @ feature_map.flatten(2).permute(0, 2, 1)
    return tokens


def show_tokens_merge_multi(x, out, N_grid=14*14):
    import matplotlib.pyplot as plt
    IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
    x = x * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
    # for i in range(x.shape[0]):
    for i in range(1):
        img = x[i].permute(1, 2, 0).detach().cpu()
        ax = plt.subplot(3, 5, 1)
        ax.clear()
        ax.imshow(img)
        for lv in range(len(out)):
            x_token, map_size, Agg, loc_orig, A = out[lv]
            B, N, _ = Agg.shape
            tmp = torch.rand([N, 3], device=x.device)[None, :, :].expand(B, N, 3).float()
            # tmp = x_token[:, :, :3]
            # # max = tmp.max(dim=-1, keepdim=True)[0]
            # # min = tmp.min(dim=-1, keepdim=True)[0]
            # max, min = tmp.max(), tmp.min()
            # tmp = (tmp - min) / (max-min)
            H, W, _ = img.shape
            x_map = token2map_Agg(tmp, Agg, loc_orig, [H//4, W//4])
            x_map = x_map[i].permute(1, 2, 0).detach().cpu().float()
            ax = plt.subplot(3, 5, lv+2)
            ax.clear()
            ax.imshow(x_map)

            ax = plt.subplot(3, 5, lv+2 + 10)
            ax.clear()
            ax.imshow(A[i].float().detach().cpu())

    return


def show_conf_merge_multi(conf, Agg, loc_orig):
    H = int(conf.shape[1]**0.5)
    lv = int(math.log2(28 / H) + 7 + 0)

    conf = conf - conf.min(dim=1, keepdim=True)[0]
    conf_map = token2map_Agg(conf, Agg, loc_orig, [28, 28])

    ax = plt.subplot(3, 5, lv)
    ax.clear()
    ax.imshow(conf_map[0, 0].detach().cpu().float(), vmin=0, vmax=7)


def show_tokens_merge0(x, out, N_grid=14*14):
    # import matplotlib.pyplot as plt
    IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
    x = x * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN

    B, _, h, w = x.shape
    h, w = h // 4, w//4
    device = x.device
    # y_g, x_g = torch.arange(h, device=device).float(), torch.arange(w, device=device).float()
    # y_g = 1 * ((y_g + 0.5) / h) - 0
    # x_g = 1 * ((x_g + 0.5) / w) - 0
    # y_map, x_map = torch.meshgrid(y_g, x_g)
    # color_map = torch.stack((x_map, y_map, x_map*0), dim=-1)
    # color_map = color_map.permute(2, 0, 1).unsqueeze(0).expand(B, 3, h, w).float()

    color_map = F.avg_pool2d(x, kernel_size=4)

    # color_map = torch.rand([1, 3, h, w], device=x.device).expand(B, 3, h, w).float()



    # for i in range(x.shape[0]):
    for i in range(1):
        img = x[i].permute(1, 2, 0).detach().cpu()
        ax = plt.subplot(2, 5, 1)
        ax.clear()
        ax.imshow(img)
        # ax = plt.subplot(2, 5, 6)
        # ax.clear()
        # ax.imshow(img)

        for lv in range(len(out)):
            ax = plt.subplot(2, 5, lv+2)
            ax.clear()
            # ax.imshow(img, extent=[0, 1, 0, 1])
            # loc = out[lv][1]
            # loc = 0.5 * loc + 0.5
            # loc_grid = loc[i, :N_grid].detach().cpu().numpy()
            # ax.scatter(loc_grid[:, 0], 1 - loc_grid[:, 1], c='blue', s=0.4+lv*0.1)
            # loc_ada = loc[i, N_grid:].detach().cpu().numpy()
            # ax.scatter(loc_ada[:, 0], 1 - loc_ada[:, 1], c='red', s=0.4+lv*0.1)
            loc_orig = out[lv][3]
            idx_agg = out[lv][4]
            agg_weight = out[lv][5]
            x = out[lv][0]
            B, N, _ = x.shape

            # tmp = torch.arange(N, device=loc.device)[None, :, None].expand(B, N, 1).float()
            tmp = torch.rand([N, 3], device=x.device)[None, :, :].expand(B, N, 3).float()
            # tmp = map2token_agg_fast_nearest(color_map, N, loc_orig, idx_agg, agg_weight)

            H, W, _ = img.shape
            idx_map, _ = token2map_agg_mat(tmp, loc_orig, loc_orig, idx_agg, [H//4, W//4])
            idx_map = idx_map[i].permute(1, 2, 0).detach().cpu().float()
            ax.imshow(idx_map)
    # plt.show()

    return


def show_conf_merge(conf, loc, loc_orig, idx_agg):
    N = int(conf.shape[1])
    lv = int(math.log2(64*48 / N) / 2)

    # conf = F.softmax(conf, dim=1)
    conf = conf - conf.min(dim=1, keepdim=True)[0]
    conf_map, _ = token2map_agg_mat(conf, loc, loc_orig, idx_agg, [64 // 2**lv, 48 // 2**lv])
    ax = plt.subplot(2, 5, lv+7)
    ax.clear()
    ax.imshow(conf_map[0, 0].detach().cpu(), vmin=0, vmax=7)


def get_merge_way(input_dict, sample_num):
    x = input_dict['x']
    idx_agg = input_dict['idx_agg']
    agg_weight = input_dict['agg_weight']
    with torch.no_grad():
        index_down = farthest_point_sample(x, sample_num)
    x_down = index_points(x, index_down)

    B, N, C = x.shape
    Ns = x_down.shape[1]
    idx_agg_t = torch.cdist(x, x_down, p=2).argmin(axis=2)

    # make sure selected tokens merge to itself
    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
    idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
    idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    idx = idx_agg_t + torch.arange(B)[:, None].to(x.device) * Ns
    weight = x.new_ones(B, N, 1)
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-4
    norm_weight = weight / all_weight[idx]

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    weight_t = index_points(norm_weight, idx_agg)

    agg_weight_down = agg_weight * weight_t
    agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

    return idx_agg, agg_weight_down



'''for speed'''

def tokenconv_sparse(conv_layer, loc_orig, x, idx_agg, agg_weight, map_size, token_weight=None):
    H, W = map_size
    B, N, C = x.shape
    N0 = loc_orig.shape[1]
    device = x.device
    loc_orig = loc_orig.clamp(-1, 1)
    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    loc_orig = loc_orig.round().long()
    loc_orig[..., 0] = loc_orig[..., 0].clamp(0, W-1)
    loc_orig[..., 1] = loc_orig[..., 1].clamp(0, H-1)
    idx_HW_orig = loc_orig[..., 0] + loc_orig[..., 1] * W

    idx_HW_orig = idx_HW_orig + torch.arange(B)[:, None].to(device) * H * W
    idx_tokens = idx_agg + torch.arange(B)[:, None].to(device) * N

    # token to map
    coor = torch.stack([idx_HW_orig, idx_tokens], dim=0).reshape(2, B*N0)

    if token_weight is None:
        value = x.new_ones(B*N0)
    else:
        value = index_points(token_weight, idx_agg).reshape(B*N0)

    # if token_weight is None:
    #     agg_weight = x.new_ones(B, N, 1)
    # value = index_points(token_weight, idx_agg).reshape(B * N0)

    with torch.cuda.amp.autocast(enabled=False):
        value = value.type(torch.float32)
        A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))
        all_weight = A.type(torch.float32) @ x.new_ones(B*N, 1).type(torch.float32) + 1e-6

        value = value / all_weight[idx_HW_orig.reshape(-1), 0]

        A = torch.sparse.FloatTensor(coor, value, torch.Size([B*H*W, B*N]))
        x_out = A.type(torch.float32) @ x.reshape(B*N, C).type(torch.float32)

    x_out = x_out.type(x.dtype)
    x_out = x_out.reshape(B, H, W, C).permute(0, 3,  1, 2).contiguous()

    # conv
    x_out = conv_layer(x_out)

    # map to token
    coor = torch.stack([idx_tokens, idx_HW_orig], dim=0).reshape(2, B*N0)
    with torch.cuda.amp.autocast(enabled=False):
        value = agg_weight.reshape(-1).type(torch.float32)
        A = torch.sparse_coo_tensor(coor, value, (B * N, B * H * W))
        all_weight = A @ torch.ones([B * H * W, 1], device=x.device, dtype=torch.float32) + 1e-6

        value = value / all_weight[idx_tokens.reshape(-1), 0]

        A = torch.sparse_coo_tensor(coor, value, (B * N, B*H*W))
        tokens = A @ x_out.permute(0, 2, 3, 1).reshape(B*H*W, C).type(torch.float32)
    tokens = tokens.type(x.dtype)
    tokens = tokens.reshape(B, N, C)
    return tokens



def token_cluster_dist(x, Ns, idx_agg, weight=None, return_weight=False):
    device = x.device
    B, N, C = x.shape

    # dists_matrix = torch.cdist(x, x)
    # index_down = torch.zeros(B, Ns, dtype=torch.long).to(device)
    # distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # for i in range(Ns):
    #     index_down[:, i] = farthest
    #     dist = dists_matrix[batch_indices, farthest,:]
    #     mask = dist < distance
    #     distance[mask] = dist[mask]
    #     farthest = torch.max(distance, -1)[1]
    # dists_matrix = index_points(dists_matrix, index_down)
    # idx_agg_t = dists_matrix.argmin(axis=1)
    # idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns

    with torch.no_grad():
        sample_ratio = Ns / N
        batch = torch.arange(B, device=x.device)[:, None].expand(B, N)
        with torch.cuda.amp.autocast(enabled=False):
            index_down = fps(x.flatten(0, 1).type(torch.float32), batch.flatten(0, 1), ratio=sample_ratio)
        index_down = index_down.reshape(B, -1)
        Ns = index_down.shape[1]
        index_down = index_down - torch.arange(B, device=device)[:, None] * N
        x_down = index_points(x, index_down)

        idx_agg_t = torch.cdist(x_down, x).argmin(axis=1)
        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns

    # batch_down = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
    # idx = nearest(x.flatten(0, 1), x_down.flatten(0, 1),
    #                     batch.flatten(0, 1), batch_down.flatten(0, 1))
    # idx = idx.reshape(B, N)
    # idx_agg_t = idx - torch.arange(B, device=device)[:, None] * Ns

    if weight is None:
        weight = x.new_ones(B, N, 1)

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



def map2token_agg_fast_nearest(feature_map, N, loc_orig, idx_agg, agg_weight=None):

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

    indices = torch.stack([index_batch, idx_agg, idx_HW_orig], dim=0).reshape(3, -1)

    if agg_weight is None:
        value = torch.ones(B * N0, device=feature_map.device, dtype=torch.float32)
    else:
        value = agg_weight.reshape(B * N0).type(torch.float32)

    A = torch.sparse_coo_tensor(indices, value, (B, N, H * W)).to_dense()
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

    tokens = A @ feature_map.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
    return tokens


# def map2token_agg_sparse_nearest(feature_map, N, loc_orig, idx_agg, agg_weight=None):
#     ''' realized by 2 attention matrix'''
#     # feature_map = torch.rand(2, 3, 5, 5)
#     # loc = torch.rand(2, 4, 2)
#     # loc_orig = torch.rand(2, 7, 2) - 0.5
#     # idx_agg = (torch.rand(2, 7) * 3).long()
#     # weight = None
#
#     B, C, H, W = feature_map.shape
#     device = feature_map.device
#     N0 = loc_orig.shape[1]
#
#     if N0 == N and N == H * W:
#         return feature_map.flatten(2).permute(0, 2, 1)
#
#     loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
#     x = loc_orig[:, :, 0]
#     y = loc_orig[:, :, 1]
#
#     h, w = H, W
#     x_grid = x.round().long().clamp(min=0, max=w - 1)
#     y_grid = y.round().long().clamp(min=0, max=h - 1)
#     idx_HW_orig = (y_grid * w + x_grid).detach()
#
#     idx_HW_orig = idx_HW_orig + torch.arange(B)[:, None].to(device) * H * W
#     idx_tokens = idx_agg + torch.arange(B)[:, None].to(device) * N
#
#     indices = torch.stack([idx_tokens.reshape(-1), idx_HW_orig.reshape(-1)], dim=0)
#
#     with torch.cuda.amp.autocast(enabled=False):
#         if agg_weight is None:
#             value = torch.ones(B * N0, device=feature_map.device, dtype=torch.float32)
#         else:
#             value = agg_weight.reshape(B * N0).type(torch.float32)
#
#         A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
#
#         all_weight = A @ torch.ones([B * H * W, 1], device=feature_map.device, dtype=torch.float32) + 1e-6
#         value = value / all_weight[idx_tokens.reshape(-1), 0]
#
#         A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
#         tokens = A @ feature_map.permute(0, 2, 3, 1).reshape(B * H * W, C).type(torch.float32)
#     tokens = tokens.type(x.dtype)
#     tokens = tokens.reshape(B, N, C)
#     return tokens


'''density clustering'''
'''
Study on density peaks clustering based on k-nearest neighbors and
principal component analysis
'''


def token_cluster_density_old(x, Ns, idx_agg, weight=None, return_weight=False, conf=None,
                          k=3, dist_assign=False, ada_dc=False, use_conf=False, conf_scale=0.25,
                          conf_density=False):
    # import torch
    # x = torch.rand(2, 1000, 64)
    # Ns = 250
    # k = 3
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape

    if weight is None:
        weight = x.new_ones(B, N, 1)
    if conf is not None:
        conf = conf.squeeze(-1)

    with torch.no_grad():
        dist_matrix = torch.cdist(x, x)
        # normalize dist_matrix for stable
        dist_matrix = dist_matrix / (dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] + 1e-6)

        # get density
        if conf_density:
            density = conf.exp()
        else:
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1)
            if ada_dc:
                '''
                Adaptive density peak clustering based on K-nearest neighbors with aggregating strategy
                '''
                uk = dist_nearest[:, :, -1].mean(dim=-1)
                tmp = dist_nearest[:, :, -1] - uk[:, None]
                tmp = (tmp ** 2).mean(dim=-1) * (N - 1) / N
                tmp = tmp ** 0.5
                dc = uk + tmp
                density = -(dist_nearest / dc[:, None, None])**2
                density = density.exp().sum(dim=-1)
            else:
                density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
                # density = -(dist_nearest ** 2).mean(dim=-1)
                # density = density - density.min(dim=1, keepdim=True)
                # density = density.exp()

        # get dist
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1-mask)).min(dim=-1)

        # select center according to score
        score = dist * density
        if use_conf and conf is not None:
            ##TODO: make this combination learnable
            # we need to limit the influence of weight
            score_log = score.log()
            # conf = conf.squeeze(-1)
            conf_scale = conf_scale * (score_log.max(dim=1)[0] - score_log.min(dim=1)[0]) / (conf.max(dim=1)[0] - conf.min(dim=1)[0] + 1e-6)
            conf_scale = conf_scale.clamp(0, 1)
            score_log = score_log + conf * conf_scale[:, None]
            _, index_down = torch.topk(score_log, k=Ns, dim=-1)
        else:
            _, index_down = torch.topk(score, k=Ns, dim=-1)

        if not dist_assign:
            '''assign way in paper'''
            # assign the center first
            idx_agg_t = torch.zeros([B, N], dtype=torch.long, device=device) - 1
            idx_batch = torch.arange(B, device=device)[:, None].expand(B, Ns)
            idx_Ns = torch.arange(Ns, device=device)[None:, ].expand(B, Ns)
            idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_Ns.reshape(-1)
            index_parent[idx_batch.reshape(-1), index_down.reshape(-1)] = index_down.reshape(-1)

            # assign the point to its parent point cluster
            ind_tmp = torch.argsort(density, dim=-1, descending=True)
            idx_batch = torch.arange(B, device=device)
            for i in range(N):
                child = ind_tmp[:, i]
                parent = index_parent[idx_batch, child]
                idx_agg_t[idx_batch, child] = idx_agg_t[idx_batch, parent]
        else:
            '''nearest assign'''
            dist_matrix = index_points(dist_matrix, index_down)
            idx_agg_t = dist_matrix.argmin(dim=1)

        # # make sure selected tokens merge to itself
        # if index_down is not None:
        #     idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        #     idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        #     idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

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



def token_cluster_density_fixbug(x, Ns, idx_agg, weight=None, return_weight=False, conf=None,
                          k=5, dist_assign=False, ada_dc=False, use_conf=False, conf_scale=0.25,
                          conf_density=False):
    # import torch
    # x = torch.rand(2, 1000, 64)
    # Ns = 250
    # k = 3
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape

    if weight is None:
        weight = x.new_ones(B, N, 1)
    if conf is not None:
        conf = conf.squeeze(-1)

    with torch.no_grad():
        dist_matrix = torch.cdist(x, x)
        # normalize dist_matrix for stable
        dist_matrix = dist_matrix / (dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] + 1e-6)
        # idx_tmp = torch.arange(N, device=x.device)
        # dist_matrix[:, idx_tmp, idx_tmp] = 0

        # get density
        if conf_density:
            density = conf.exp()
        else:
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1)
            if ada_dc:
                '''
                Adaptive density peak clustering based on K-nearest neighbors with aggregating strategy
                '''
                uk = dist_nearest[:, :, -1].mean(dim=-1)
                tmp = dist_nearest[:, :, -1] - uk[:, None]
                tmp = (tmp ** 2).mean(dim=-1) * (N - 1) / N
                tmp = tmp ** 0.5
                dc = uk + tmp
                density = -(dist_nearest / dc[:, None, None])**2
                density = density.exp().sum(dim=-1)
            else:
                density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
                # density = -(dist_nearest ** 2).mean(dim=-1)
                # density = density - density.min(dim=1, keepdim=True)
                # density = density.exp()

        # get dist
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1-mask)).min(dim=-1)

        # select center according to score
        score = dist * density
        if use_conf and conf is not None:
            ##TODO: make this combination learnable
            # we need to limit the influence of weight
            score_log = score.log()
            # conf = conf.squeeze(-1)
            conf_scale = conf_scale * (score_log.max(dim=1)[0] - score_log.min(dim=1)[0]) / (conf.max(dim=1)[0] - conf.min(dim=1)[0] + 1e-6)
            conf_scale = conf_scale.clamp(0, 1)
            score_log = score_log + conf * conf_scale[:, None]
            _, index_down = torch.topk(score_log, k=Ns, dim=-1)
        else:
            _, index_down = torch.topk(score, k=Ns, dim=-1)

        if not dist_assign:
            '''assign way in paper'''
            # assign the center first
            idx_agg_t = torch.zeros([B, N], dtype=torch.long, device=device) - 1
            idx_batch = torch.arange(B, device=device)[:, None].expand(B, Ns)
            idx_Ns = torch.arange(Ns, device=device)[None:, ].expand(B, Ns)
            idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_Ns.reshape(-1)
            index_parent[idx_batch.reshape(-1), index_down.reshape(-1)] = index_down.reshape(-1)

            # assign the point to its parent point cluster
            ind_tmp = torch.argsort(density, dim=-1, descending=True)
            idx_batch = torch.arange(B, device=device)
            for i in range(N):
                child = ind_tmp[:, i]
                parent = index_parent[idx_batch, child]
                idx_agg_t[idx_batch, child] = idx_agg_t[idx_batch, parent]
        else:
            '''nearest assign'''
            dist_matrix = index_points(dist_matrix, index_down)
            idx_agg_t = dist_matrix.argmin(dim=1)

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



def token_cluster_density(x, Ns, idx_agg, weight=None, return_weight=False, conf=None,
                          k=3, dist_assign=False, ada_dc=False, use_conf=False, conf_scale=0.25,
                          conf_density=False):
    # import torch
    # x = torch.rand(2, 1000, 64)
    # Ns = 250
    # k = 3
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape

    if weight is None:
        weight = x.new_ones(B, N, 1)
    if conf is not None:
        conf = conf.squeeze(-1)

    with torch.no_grad():
        dist_matrix = torch.cdist(x, x)
        # normalize dist_matrix for stable
        dist_matrix = dist_matrix / (dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] + 1e-6)
        idx_tmp = torch.arange(N, device=x.device)
        dist_matrix[:, idx_tmp, idx_tmp] = 0

        # get density
        if conf_density:
            density = conf.exp()
        else:
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1)
            if ada_dc:
                '''
                Adaptive density peak clustering based on K-nearest neighbors with aggregating strategy
                '''
                uk = dist_nearest[:, :, -1].mean(dim=-1)
                tmp = dist_nearest[:, :, -1] - uk[:, None]
                tmp = (tmp ** 2).mean(dim=-1) * (N - 1) / N
                tmp = tmp ** 0.5
                dc = uk + tmp
                density = -(dist_nearest / dc[:, None, None])**2
                density = density.exp().sum(dim=-1)
            else:
                density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
                # density = -(dist_nearest ** 2).mean(dim=-1)
                # density = density - density.min(dim=1, keepdim=True)
                # density = density.exp()

        # get dist
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1-mask)).min(dim=-1)

        # select center according to score
        score = dist * density
        if use_conf and conf is not None:
            ##TODO: make this combination learnable
            # we need to limit the influence of weight
            score_log = score.log()
            # conf = conf.squeeze(-1)
            conf_scale = conf_scale * (score_log.max(dim=1)[0] - score_log.min(dim=1)[0]) / (conf.max(dim=1)[0] - conf.min(dim=1)[0] + 1e-6)
            conf_scale = conf_scale.clamp(0, 1)
            score_log = score_log + conf * conf_scale[:, None]
            _, index_down = torch.topk(score_log, k=Ns, dim=-1)
        else:
            _, index_down = torch.topk(score, k=Ns, dim=-1)

        if not dist_assign:
            '''assign way in paper'''
            # assign the center first
            idx_agg_t = torch.zeros([B, N], dtype=torch.long, device=device) - 1
            idx_batch = torch.arange(B, device=device)[:, None].expand(B, Ns)
            idx_Ns = torch.arange(Ns, device=device)[None:, ].expand(B, Ns)
            idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_Ns.reshape(-1)
            index_parent[idx_batch.reshape(-1), index_down.reshape(-1)] = index_down.reshape(-1)

            # assign the point to its parent point cluster
            ind_tmp = torch.argsort(density, dim=-1, descending=True)
            idx_batch = torch.arange(B, device=device)
            for i in range(N):
                child = ind_tmp[:, i]
                parent = index_parent[idx_batch, child]
                idx_agg_t[idx_batch, child] = idx_agg_t[idx_batch, parent]
        else:
            '''nearest assign'''
            dist_matrix = index_points(dist_matrix, index_down)
            idx_agg_t = dist_matrix.argmin(dim=1)

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


def downup_sparse(target_dict, source_dict):
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
        A = torch.sparse.FloatTensor(coor, weight, torch.Size([B*T, B*S]))
        all_weight = A.type(torch.float32) @ x_s.new_ones(B*S, 1).type(torch.float32) + 1e-6
        # all_weight = A @ x_s.new_ones(B*S, 1) + 1e-6
        all_weight = all_weight.type(x_s.dtype)
        weight = weight / all_weight[(idx_agg_t).reshape(-1), 0]

    with torch.cuda.amp.autocast(enabled=False):
        A = torch.sparse.FloatTensor(coor, weight, torch.Size([B*T, B*S]))
        x_out = A.type(torch.float32) @ x_s.reshape(B*S, C).type(torch.float32)
        x_out = x_out.reshape(B, T, C).type(x_s.dtype)
    return x_out


def downup(target_dict, source_dict):
    x_s = source_dict['x']
    x_t = target_dict['x']
    idx_agg_s = source_dict['idx_agg']
    idx_agg_t = target_dict['idx_agg']
    agg_weight_t = target_dict['agg_weight']
    B, T, _ = x_t.shape
    B, S, C = x_s.shape
    N0 = idx_agg_s.shape[1]
    idx_batch = torch.arange(B, device=idx_agg_t.device)[:, None].expand(B, N0)

    if agg_weight_t is None:
        weight = x_s.new_ones(B*N0)
    else:
        weight = agg_weight_t.reshape(-1)

    coor = torch.stack([idx_batch, idx_agg_t, idx_agg_s], dim=0).reshape(3, B*N0)
    A = torch.sparse.FloatTensor(coor, weight, torch.Size([B, T, S]))
    A = A.to_dense()
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
    x_out = A @ x_s
    return x_out



def vis_tokens_and_grid():
    # color = torch.rand(1, 10000, 3)
    # color48 = index_points(color, farthest_point_sample(color, 48))

    # idx_dict={
    #     # 18880: [0, 10, 11],
    #     # 260:[0, 17, 2],
    #     # 563: [0, 22, 8],
    #     993:[17, 27, 13],
    #     1272: [31, 37, 18],
    #     1736:[0, 35, 14],
    #     2405: [0, 12, 15],
    #     2950:[0, 16, 14],
    #     3590:[27, 43, 7],
    #     4710:[0, 12, 21],
    # }

    color_begin = torch.tensor([[0, 0, 1],
                                [1, 0, 0],
                                [0, 1, 0]]).float()
    # for count in idx_dict.keys():
    # for count in [913,918,921,1079,1272, 2424, 2556, 2802,2838,3318,5055,6308, 5378, 5484, 5951, 6018]:


    # for count in [62, 87, 250, 563, 581,740,968,993, 1114,1762,1900, 2085,2098,2147, 2251, 2557,2571,2821,
    #               2839,2950, 3207,3217, 3332, 3580, 3635, 3675,3779,3954,4054,4222, 4710, 4786, 4810, 5055,
    #               913, 918, 921, 1079, 1272, 2424, 2556, 2802, 2838, 3318, 5055, 6308, 5378, 5484, 5951, 6018]:
    # for count in [3332, 5055, 3675, 3779, 2251, 2950, 1762, 3954, 6018, 5484, 3635]:
    # for count in [3332, 5055, 3779]:
    for count in [1272, 2424, 3318]:
    # for count in [935, 1547]:

        fname = f'vis/{count}.pth'

        # color = torch.rand(1, 10000, 3)
        # color48 = index_points(color, farthest_point_sample(color, 48))


        data = torch.load(fname)
        x = data['x']
        out = data['out']
        B, _, h, w = x.shape
        h, w = h // 4, w // 4
        device = x.device
        N0 = h * w

        img = x[0].permute(1, 2, 0).detach().cpu()
        ax = plt.subplot(1, 6, 1)
        ax.clear()
        ax.imshow(img)
        color_map = F.avg_pool2d(x, kernel_size=4)



        fname = f'vis/{count}_img.png'
        import cv2
        cv2.imwrite(fname, img.numpy()[:, :, ::-1] * 255)


        lv = len(out) - 1
        ax = plt.subplot(1, 6, 6)
        ax.clear()
        loc_orig = out[lv][3]
        idx_agg = out[lv][4]
        agg_weight = out[lv][5]
        x = out[lv][0]
        B, N, _ = x.shape

        tmp = torch.arange(N, device=x.device)[None, :, None].expand(B, N, 1).float()
        H, W, _ = img.shape
        idx_map, _ = token2map_agg_mat(tmp, loc_orig, loc_orig, idx_agg, [H // 4, W // 4])
        idx_map = F.interpolate(idx_map, [H, W], mode='nearest')
        idx_map = idx_map[0].permute(1, 2, 0).detach().cpu().float()
        ax.imshow(idx_map)
        # ax.imshow(idx_map / 48 * 0.5 + img * 0.5)
        # tmp_map = img.clone()
        # tmp_map[:, :, 2] = idx_map[:, :, 0] / 48
        # ax.imshow(tmp_map)


        pixel_num = x.new_zeros(N)
        pixel_num.index_add_(0, idx_agg[0], x.new_ones(N0))
        # idx_begin = pixel_num.argsort()

        # token_c = x.new_ones(B, N, 3) * 0.8
        # token_c[:, :, :] = color48.to(x.device)
        #
        # idx_begin = idx_dict[count]
        # color_begin = torch.tensor([[0, 0, 1],
        #                             [1, 0, 0],
        #                             [0, 1, 0]]).float().to(x.device)
        # # token_c = x.new_zeros(B, N, 3)
        # # token_c = x.new_ones(B, N, 3) * 0.8
        # token_c[0, idx_begin] = color_begin


        # color_begin = torch.tensor([[0, 0, 1],
        #                             [1, 0, 0],
        #                             [0, 1, 0]]).float().to(x.device)
        # # token_c = x.new_zeros(B, N, 3)
        # token_c[0, idx_begin] = color_begin
        # src_dict = {
        #     'x': token_c,
        #     'idx_agg': idx_agg,
        #     'agg_weight': agg_weight,
        #     'loc_orig': loc_orig
        # }



        # factors = [.2, .4, .6]
        for lv in range(len(out)-1, -1, -1):
        # for lv in range(len(out) - 1, len(out) - 2, -1):

            loc_orig = out[lv][3]
            idx_agg = out[lv][4]
            agg_weight = out[lv][5]
            x = out[lv][0]
            B, N, _ = x.shape

            # if lv < len(out) - 1:
            #     dst_dict = {
            #         'x': x,
            #         'idx_agg': idx_agg,
            #         'agg_weight': agg_weight,
            #         'loc_orig': loc_orig
            #     }
            #
            #     token_c = downup(dst_dict, src_dict)
            #     factor = factors[lv]
            #     token_c = token_c + (torch.rand([B, N, 3], device=x.device)-0.5) * factor * (token_c.sum(dim=-1, keepdim=True) > 0).float()
            #     token_c = token_c.clamp(0, 1)
            #
            # src_dict = {
            #     'x': token_c,
            #     'idx_agg': idx_agg,
            #     'agg_weight': agg_weight,
            #     'loc_orig': loc_orig
            # }

            token_c = map2token_agg_fast_nearest(color_map, N, loc_orig, idx_agg, agg_weight)
            # token_c = token_c - token_c.min(dim=1, keepdim=True)[0]
            # token_c = token_c / token_c.max(dim=1, keepdim=True)[0]

            H, W, _ = img.shape
            idx_map, _ = token2map_agg_mat(token_c, loc_orig, loc_orig, idx_agg, [H // 4, W // 4])


            # # sharpen
            # idx_map_s = F.interpolate(idx_map, [H*4, W*4], mode='nearest')
            #
            #
            #
            #
            # # sharpen = torch.FloatTensor([   [0, -2, 0],
            # #                                 [-2, 9, -2],
            # #                                 [0, -2, 0]])
            # sharpen = torch.FloatTensor([   [0, -1, 0],
            #                                 [-1, 5, -1],
            #                                 [0, -1, 0]])
            # sharpen = sharpen[None, None, :, :].to(idx_map.device).expand([3,1,3,3])
            # # idx_map = F.conv2d(F.pad(idx_map, [1,1, 1,1], mode='replicate'), sharpen, groups=3)
            # for t in range(1):
            #     idx_map_s = F.conv2d(F.pad(idx_map_s, [1,1, 1,1], mode='replicate'), sharpen, groups=3)
            # fname = f'vis/{count}_{lv}.png'
            # import cv2
            # cv2.imwrite(fname, idx_map_s[0].permute(1, 2, 0).detach().cpu().float().numpy()[:, :, ::-1]*255)
            #
            #
            #
            #
            # idx_map = F.interpolate(idx_map, [H, W], mode='nearest')
            # idx_map = idx_map[0].permute(1, 2, 0).detach().cpu().float()
            # ax = plt.subplot(1, 6, lv+2)
            # ax.imshow(idx_map*0.7 + img*0.3)
            #
            #
            #
            # # idx_map_l = token2map_agg_mat(token_c, loc_orig, loc_orig, idx_agg, [H // 32, W // 32])
            #
            # # dst_dict = {
            # #     'x': x,
            # #     'idx_agg': idx_agg,
            # #     'agg_weight': agg_weight,
            # #     'loc_orig': loc_orig
            # # }
            # #
            # # idx_begin_new = []
            # # for idx in idx_begin:
            # #     mask = src_dict['idx_agg'] == idx
            # #     weight = x.new_zeros(N)
            # #     # weight.index_add_(0, idx_agg[0], agg_weight[0, :, 0] * mask[0].float())
            # #     weight.index_add_(0, idx_agg[0], mask[0].float())
            # #     idx_begin_new.append(weight.argmax().item())
            # # idx_begin = idx_begin_new
            # # token_c = x.new_ones(B, N, 3) * 0.8
            # # token_c[0, idx_begin] = color_begin
            #
            #
            #
            # # idx_begin_new = []
            # # for idx in idx_begin:
            # #     mask = src_dict['idx_agg'] == idx
            # #     weight = x.new_zeros(N)
            # #     # weight.index_add_(0, idx_agg[0], agg_weight[0, :, 0] * mask[0].float())
            # #     # weight.index_add_(0, idx_agg[0], mask[0].float())
            # #     idx_begin_new.append(weight.argmax().item())
            #
            # # idx_map_l = color48.reshape(1, 8, 6, 3).permute(0, 3, 1, 2)
            # idx_map_l = F.avg_pool2d(color_map, kernel_size=8)
            # idx_map_l = F.interpolate(idx_map_l, [H * 4, W * 4], mode='nearest')
            #
            # for t in range(1):
            #     idx_map_l = F.conv2d(F.pad(idx_map_l, [1,1, 1,1], mode='replicate'), sharpen, groups=3)
            #
            # fname = f'vis/{count}_{lv}_grid.png'
            # import cv2
            # cv2.imwrite(fname, idx_map_l[0].permute(1, 2,0).detach().cpu().float().numpy()[:, :, ::-1] * 255)








            # sharpen
            idx_map_s = F.interpolate(idx_map, [H*4, W*4], mode='nearest')
            idx_map_l = F.avg_pool2d(color_map, kernel_size=2**lv)
            idx_map_l = F.interpolate(idx_map_l, [H * 4, W * 4], mode='nearest')

            sharpen = torch.FloatTensor([   [0, -1, 0],
                                            [-1, 4, -1],
                                            [0, -1, 0]])
            sharpen = sharpen[None, None, :, :].to(idx_map.device).expand([3,1,3,3])


            mask_s = F.conv2d(F.pad(idx_map_s, [1, 1, 1, 1], mode='replicate'), sharpen, groups=3)
            mask_l = F.conv2d(F.pad(idx_map_l, [1, 1, 1, 1], mode='replicate'), sharpen, groups=3)

            mask_s = (mask_s.abs() > 0).float()
            mask_l = (mask_l.abs() > 0).float()
            for t in range(lv-1):
                kernel = torch.FloatTensor([[0, 1, 0],
                                            [1, 1, 1],
                                            [0, 1, 0]])
                kernel = kernel[None, None, :, :].to(idx_map.device).expand([3, 1, 3, 3])

                mask_s = F.conv2d(F.pad(mask_s, [1, 1, 1, 1], mode='replicate'), kernel, groups=3)
                mask_l = F.conv2d(F.pad(mask_l, [1, 1, 1, 1], mode='replicate'), kernel, groups=3)

            idx_map_s = (idx_map_s + mask_s*10).clamp(0, 1)
            idx_map_l = (idx_map_l + mask_l*10).clamp(0, 1)

            fname = f'vis/{count}_{lv}.png'
            import cv2
            cv2.imwrite(fname, idx_map_s[0].permute(1, 2, 0).detach().cpu().float().numpy()[:, :, ::-1]*255)

            fname = f'vis/{count}_{lv}_grid.png'
            import cv2
            cv2.imwrite(fname, idx_map_l[0].permute(1, 2,0).detach().cpu().float().numpy()[:, :, ::-1] * 255)











        fname = f'vis/{count}.jpg'
        plt.savefig(fname, dpi=800)

    return




def show_tokens_merge(x, out, N_grid=14*14, count=0):
    # import matplotlib.pyplot as plt
    IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
    x = x * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN

    # save_dict = {
    #     'x': x,
    #     'out': out
    # }
    # fname = f'vis/{count}.pth'
    # torch.save(save_dict, fname)

    B, _, h, w = x.shape
    h, w = h // 4, w//4
    device = x.device
    color_map = F.avg_pool2d(x, kernel_size=4)


    N0 = h*w

    for i in range(1):
        img = x[i].permute(1, 2, 0).detach().cpu()
        ax = plt.subplot(1, 6, 1)
        ax.clear()
        ax.imshow(img)

        fname = f'vis/{count}_img.png'
        import cv2
        cv2.imwrite(fname, img.numpy()[:, :, ::-1] * 255)

        lv = 3
        loc_orig = out[lv][3]
        idx_agg = out[lv][4]
        agg_weight = out[lv][5]
        x = out[lv][0]
        B, N, _ = x.shape

        tmp = torch.arange(N, device=x.device)[None, :, None].expand(B, N, 1).float()
        H, W, _ = img.shape
        idx_map, _ = token2map_agg_mat(tmp, loc_orig, loc_orig, idx_agg, [H // 4, W // 4])
        idx_map = F.interpolate(idx_map, [H, W], mode='nearest')
        # idx_map = idx_map[0].permute(1, 2, 0).detach().cpu().float()
        ax = plt.subplot(1, 6, 6)
        ax.imshow(idx_map[0].permute(1, 2, 0).detach().cpu().float())

        for lv in range(len(out)):

            loc_orig = out[lv][3]
            idx_agg = out[lv][4]
            agg_weight = out[lv][5]
            x = out[lv][0]
            B, N, _ = x.shape


            token_c = map2token_agg_fast_nearest(color_map, N, loc_orig, idx_agg, agg_weight)
            idx_map, _ = token2map_agg_mat(token_c, loc_orig, loc_orig, idx_agg, [H // 4, W // 4])


            idx_map_our = F.interpolate(idx_map, [H*4, W*4], mode='nearest')
            idx_map_grid = F.avg_pool2d(color_map, kernel_size=2**lv)
            idx_map_grid = F.interpolate(idx_map_grid, [H * 4, W * 4], mode='nearest')

            sharpen = torch.FloatTensor([   [0, -1, 0],
                                            [-1, 4, -1],
                                            [0, -1, 0]])
            sharpen = sharpen[None, None, :, :].to(idx_map.device).expand([3,1,3,3])


            mask_our = F.conv2d(F.pad(idx_map_our, [1, 1, 1, 1], mode='replicate'), sharpen, groups=3)
            mask_grid = F.conv2d(F.pad(idx_map_grid, [1, 1, 1, 1], mode='replicate'), sharpen, groups=3)

            mask_our = (mask_our.abs() > 0).float()
            mask_grid = (mask_grid.abs() > 0).float()
            for t in range(lv - 1):
                kernel = torch.FloatTensor([[0, 1, 0],
                                            [1, 1, 1],
                                            [0, 1, 0]])
                kernel = kernel[None, None, :, :].to(idx_map.device).expand([3, 1, 3, 3])

                mask_our = F.conv2d(F.pad(mask_our, [1, 1, 1, 1], mode='replicate'), kernel, groups=3)
                mask_grid = F.conv2d(F.pad(mask_grid, [1, 1, 1, 1], mode='replicate'), kernel, groups=3)

            idx_map_our = (idx_map_our + mask_our * 10).clamp(0, 1)
            idx_map_grid = (idx_map_grid + mask_grid * 10).clamp(0, 1)

            # fname = f'vis/{count}_{lv}.png'
            # import cv2
            # cv2.imwrite(fname, idx_map_our[0].permute(1, 2, 0).detach().cpu().float().numpy()[:, :, ::-1] * 255)
            #
            # fname = f'vis/{count}_{lv}_grid.png'
            # import cv2
            # cv2.imwrite(fname, idx_map_grid[0].permute(1, 2, 0).detach().cpu().float().numpy()[:, :, ::-1] * 255)

            ax = plt.subplot(1, 6, lv+2)
            ax.clear()
            ax.imshow(idx_map_our[0].permute(1, 2, 0).detach().cpu().float())

            import cv2
            fname = f'vis/{count}_{lv}.png'
            cv2.imwrite(fname, idx_map_our[0].permute(1, 2, 0).detach().cpu().float().numpy()[:, :, ::-1]*255)
            fname = f'vis/{count}_{lv}_grid.png'
            cv2.imwrite(fname, idx_map_grid[0].permute(1, 2,0).detach().cpu().float().numpy()[:, :, ::-1] * 255)





    # plt.show()
    fname = f'vis/{count}.jpg'
    plt.savefig(fname, dpi=200)


    return




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



