import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import utils.config as config
import math


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


class ConvBottleNeck(nn.Module):
    """
    the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_channels, out_channels, nl_layer=nn.ReLU(inplace=True), norm_type='GN'):
        super(ConvBottleNeck, self).__init__()
        self.nl_layer = nl_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)

        if norm_type == 'BN':
            affine = True
            # affine = False
            self.norm1 = nn.BatchNorm2d(out_channels // 2, affine=affine)
            self.norm2 = nn.BatchNorm2d(out_channels // 2, affine=affine)
            self.norm3 = nn.BatchNorm2d(out_channels, affine=affine)
        elif norm_type == 'SYBN':
            affine = True
            # affine = False
            self.norm1 = nn.SyncBatchNorm(out_channels // 2, affine=affine)
            self.norm2 = nn.SyncBatchNorm(out_channels // 2, affine=affine)
            self.norm3 = nn.SyncBatchNorm(out_channels, affine=affine)
        else:
            self.norm1 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
            self.norm2 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
            self.norm3 = nn.GroupNorm(out_channels // 8, out_channels)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        residual = x

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.nl_layer(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.nl_layer(y)

        y = self.conv3(y)
        y = self.norm3(y)

        if self.in_channels != self.out_channels:
            residual = self.skip_conv(residual)
        y += residual
        y = self.nl_layer(y)
        return y


class ConvSMPLRegressor(nn.Module):

    def __init__(self, cfg, in_channels=512, k_size=7):
        super(ConvSMPLRegressor, self).__init__()
        self.cfg = cfg

        self.in_channels = in_channels
        self.conv = nn.Sequential(
            ConvBottleNeck(in_channels, in_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBottleNeck(in_channels, in_channels * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBottleNeck(in_channels * 2, in_channels * 4),
            nn.AvgPool2d(kernel_size=k_size, stride=k_size)
        )

        npose = 24 * 6
        self.fc1 = nn.Linear(in_channels * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        smpl_mean_params = config.SMPL_MEAN_PARAMS
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, densepose_outputs, feature_map, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        x = feature_map
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.conv(x)
        xf = xf.view(xf.size(0), self.in_channels * 4)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        return (pred_rotmat, pred_shape, pred_cam)
