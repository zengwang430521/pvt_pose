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


class HMRHead(nn.Module):
    """SMPL parameters regressor head of simple baseline paper
    ref: Angjoo Kanazawa. ``End-to-end Recovery of Human Shape and Pose''.

    Args:
        in_channels (int): Number of input channels
        in_res (int): The resolution of input feature map.
        smpl_mean_parameters (str): The file name of the mean SMPL parameters
        n_iter (int): The iterations of estimating delta parameters
    """

    def __init__(self, in_channels, smpl_mean_params=None, n_iter=3):
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


