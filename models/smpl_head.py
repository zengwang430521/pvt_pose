import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import utils.config as config
import math
from .geometric_layers import rot6d_to_rotmat


class SimpleSMPLHead(nn.Module):

    def __init__(self, cfg, in_channels=512):
        super().__init__()
        self.cfg = cfg

        npose = 6
        if self.cfg.pose_head == 'share':
            self.decpose = nn.Linear(in_channels, npose)
        else:
            self.decpose = nn.Conv1d(in_channels*24, npose*24, kernel_size=1, groups=24)

        self.decshape = nn.Linear(in_channels, 10)
        self.deccam = nn.Linear(in_channels, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

    def forward(self, x,):
        bs, s, q, c = x.shape
        pred_cam = self.deccam(x[:, :, 0])
        pred_shape = self.decshape(x[:, :, 1])

        if self.cfg.pose_head == 'share':
            pred_pose = self.decpose(x[:, :, 2:])
        else:
            x_p = x[:, :, 2:]   # bs, s, 24, 256
            x_p = x_p.reshape([bs, s, -1]).permute(0, 2, 1)   # bs, 24*256, s
            pred_pose = self.decpose(x_p)     # bs, 24*6, s
            pred_pose = pred_pose.reshape(bs, 24, 6, s).permute(0, 3, 1, 2).contiguous()    # bs, s, 24, 6

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(bs, s, 24, 3, 3)
        return (pred_rotmat, pred_shape, pred_cam)


class HMRSMPLHead(nn.Module):

    def __init__(self, cfg, in_channels=512):
        super().__init__()
        self.cfg = cfg

        npose = 6
        in_channels = in_channels + npose * 24 + 10 + 3

        if self.cfg.pose_head == 'share':
            self.decpose = nn.Linear(in_channels, npose)
        else:
            self.decpose = nn.Conv1d(in_channels, npose, kernel_size=1)

        self.decshape = nn.Linear(in_channels, 10)
        self.deccam = nn.Linear(in_channels, 3)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        smpl_mean_params = config.SMPL_MEAN_PARAMS
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        s, bs, q, c = x.shape

        if init_pose is None:
            init_pose = self.init_pose.expand(s, bs, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(s, bs, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(s, bs, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 2)
            if self.cfg.pose_head == 'share':
                pred_pose = self.decpose(xc[:, :, 2:]) + pred_pose
            else:
                pred_pose = self.decpose(xc[:, :, 2:].permute(0, 2, 1)).permute(0, 2, 1) + pred_pose
            pred_shape = self.decshape(xc[:, :, 1]) + pred_shape
            pred_pose = self.decpose(xc[:, :, 2:]) + pred_pose

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(s, bs, 24, 3, 3)
        return (pred_rotmat, pred_shape, pred_cam)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


import utils.config as cfg
class HMRHead(nn.Module):
    """SMPL parameters regressor head of simple baseline paper
    ref: Angjoo Kanazawa. ``End-to-end Recovery of Human Shape and Pose''.

    Args:
        in_channels (int): Number of input channels
        in_res (int): The resolution of input feature map.
        smpl_mean_parameters (str): The file name of the mean SMPL parameters
        n_iter (int): The iterations of estimating delta parameters
    """

    def __init__(self, in_channels, smpl_mean_params=cfg.SMPL_MEAN_PARAMS, n_iter=3):
        super().__init__()

        self.in_channels = in_channels
        self.n_iter = n_iter

        npose = 24 * 6
        nbeta = 10
        ncam = 3
        hidden_dim = 1024

        self.fc1 = nn.Linear(in_channels + npose + nbeta + ncam, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(hidden_dim, npose)
        self.decshape = nn.Linear(hidden_dim, nbeta)
        self.deccam = nn.Linear(hidden_dim, ncam)

        # Load mean SMPL parameters
        if smpl_mean_params is None:
            init_pose = torch.zeros([1, npose])
            init_shape = torch.zeros([1, nbeta])
            init_cam = torch.FloatTensor([[1, 0, 0]])
        else:
            mean_params = np.load(smpl_mean_params)
            init_pose = torch.from_numpy(
                mean_params['pose'][:]).unsqueeze(0).float()
            init_shape = torch.from_numpy(
                mean_params['shape'][:]).unsqueeze(0).float()
            init_cam = torch.from_numpy(
                mean_params['cam']).unsqueeze(0).float()
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x):
        """Forward function.

        x is the image feature map and is expected to be in shape (batch size x
        channel number x height x width)
        """
        batch_size = x.shape[0]
        # extract the global feature vector by average along
        # spatial dimension.
        if x.dim() == 4:
            x = x.mean(dim=-1).mean(dim=-1)

        init_pose = self.init_pose.expand(batch_size, -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for _ in range(self.n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        out = (pred_rotmat, pred_shape, pred_cam)
        return out

    def init_weights(self):
        """Initialize model weights."""
        xavier_init(self.decpose, gain=0.01)
        xavier_init(self.decshape, gain=0.01)
        xavier_init(self.deccam, gain=0.01)


##########################
# CMR Head
##########################

class FCBlock(nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""

    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc_block(x)


class FCResBlock(nn.Module):
    """Residual block using fully-connected layers."""

    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCResBlock, self).__init__()
        self.fc_block = nn.Sequential(nn.Linear(in_size, out_size),
                                      nn.BatchNorm1d(out_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_size, out_size),
                                      nn.BatchNorm1d(out_size))

    def forward(self, x):
        return F.relu(x + self.fc_block(x))



try:
    from torch_batch_svd import svd
    gpu_svd = True
    print('load gpu svd success!')
except (ImportError, ModuleNotFoundError):
    gpu_svd = False
    print('load gpu svd fail!')


class CMRHead(nn.Module):

    def __init__(self, in_channels, use_cpu_svd=True):
        super().__init__()
        self.layers = nn.Sequential(FCBlock(in_channels, 1024),
                                    FCResBlock(1024, 1024),
                                    FCResBlock(1024, 1024),
                                    nn.Linear(1024, 24 * 3 * 3 + 10 + 3))
        self.use_cpu_svd = use_cpu_svd

    def forward(self, x):
        """Forward pass.
        Input:
            x: size = (B, 1723*6)
        Returns:
            SMPL pose parameters as rotation matrices: size = (B,24,3,3)
            SMPL shape parameters: size = (B,10)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.layers(x)
        rotmat = x[:, :24*3*3].view(-1, 24, 3, 3).contiguous()
        betas = x[:, 24*3*3:-3].contiguous()
        camera = x[:, -3:].contiguous()

        rotmat = rotmat.view(-1, 3, 3).contiguous()
        orig_device = rotmat.device


        if gpu_svd:
            U, S, V = svd(rotmat)
        else:
            if self.use_cpu_svd:
                rotmat = rotmat.cpu()
                U, S, V = batch_svd(rotmat)
                U, S, V = U.to(orig_device), S.to(orig_device), V.to(orig_device)
            else:
                U, S, V = batch_svd(rotmat)

        rotmat = torch.matmul(U, V.transpose(1, 2))
        # det = torch.zeros(rotmat.shape[0], 1, 1).to(rotmat.device)
        # with torch.no_grad():
        #     for i in range(rotmat.shape[0]):
        #         det[i] = torch.det(rotmat[i])
        # det2 = torch.det(rotmat)[:, None, None]
        # err = det2 - det
        # err = err.sum()

        det = torch.det(rotmat)[:, None, None]

        rotmat = rotmat * det
        rotmat = rotmat.view(batch_size, 24, 3, 3)
        rotmat = rotmat.to(orig_device)
        out = (rotmat, betas, camera)
        return out


def batch_svd(A):
    """Wrapper around torch.svd that works when the input is a batch of matrices."""
    U_list = []
    S_list = []
    V_list = []
    for i in range(A.shape[0]):
        U, S, V = torch.svd(A[i])
        U_list.append(U)
        S_list.append(S)
        V_list.append(V)
    U = torch.stack(U_list, dim=0)
    S = torch.stack(S_list, dim=0)
    V = torch.stack(V_list, dim=0)
    return U, S, V



# Transformer based head
from models.pvt_utils.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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

    def forward(self, x, x_source):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        _, Ns, _ = x_source.shape
        kv = self.kv(x_source).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                              4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, dim_out=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_source = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim_out,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, x_source):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1_source(x_source)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


import torch.nn.init as init

class PoseLayer(nn.Module):
    def __init__(self, in_channels=512, out_channels=9, n=24):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(n, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = (x[..., None] * self.weight[None, ...]).sum(dim=2)
        x = x + self.bias[None, ...]
        return x


class TCMRHead(nn.Module):

    def __init__(self, in_channels=512, hidden_dim=512, n_block=3, use_cpu_svd=True):
        super().__init__()
        self.use_cpu_svd = use_cpu_svd
        self.num_queries = 24 + 1 + 1
        self.in_channels = in_channels
        self.query_embed = nn.Embedding(self.num_queries, in_channels)
        self.pre_block = Block(dim=in_channels, num_heads=8, dim_out=hidden_dim)
        self.blocks = nn.ModuleList([Block(dim=hidden_dim, num_heads=8) for _ in range(n_block-1)])

        # self.pose_layers = nn.ModuleList([nn.Linear(hidden_dim, 3*3) for _ in range(24)])
        self.pose_layers = PoseLayer(hidden_dim, 3*3, 24)

        self.shape_layer = nn.Linear(hidden_dim, 10)
        self.camera_layer = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        B = x.shape[0]
        q = self.query_embed.weight[None, :, :].expand([B, -1, -1])
        x = self.pre_block(q, x)
        for block in self.blocks:
            x = block(x, x)
        B, N, C = x.shape

        # rotmat = []
        # for n in range(24):
        #     rotmat.append(self.pose_layers[n](x[:, n, :]))
        # rotmat = torch.stack(rotmat, dim=1)

        rotmat = self.pose_layers(x[:, :24, :])

        rotmat = rotmat.view(-1, 3, 3).contiguous()
        orig_device = rotmat.device
        if gpu_svd:
            U, S, V = svd(rotmat)
        else:
            if self.use_cpu_svd:
                rotmat = rotmat.cpu()
                U, S, V = batch_svd(rotmat)
                U, S, V = U.to(orig_device), S.to(orig_device), V.to(orig_device)
            else:
                U, S, V = batch_svd(rotmat)

        rotmat = torch.matmul(U, V.transpose(1, 2))
        det = torch.det(rotmat)[:, None, None]
        rotmat = rotmat * det
        rotmat = rotmat.view(B, 24, 3, 3)
        rotmat = rotmat.to(orig_device)

        betas = self.shape_layer(x[:, -2, :])
        camera = self.camera_layer(x[:, -1, :])
        out = (rotmat, betas, camera)
        return out


head_dict = {
    'hmr': HMRHead,
    'cmr': CMRHead,
    'tcmr': TCMRHead
}


def build_smpl_head(in_channels=512, head_type='hmr'):
    return head_dict[head_type](in_channels=in_channels)