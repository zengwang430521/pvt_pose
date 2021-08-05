from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from torchvision.utils import make_grid
from train.base_trainer import BaseTrainer
from datasets import create_dataset
from models import SMPL
from models.geometric_layers import orthographic_projection, rodrigues, proj_2d
import utils.config as cfg

from smplify.prior import MaxMixturePrior
import os
from smplify.losses import camera_fitting_loss, body_fitting_loss
from utils import constants
import numpy as np
from utils import config
from smplx import SMPL as _SMPL
from smplx.lbs import vertices2joints
import cv2
import math
import utils.misc as utils


class MeshLoss(nn.Module):
    def __init__(self, options, device):
        super().__init__()
        self.options = options
        self.device = device

        # prepare SMPL model
        self.smpl = SMPL().to(self.device)
        self.female_smpl = SMPL(cfg.FEMALE_SMPL_FILE).to(self.device)
        self.male_smpl = SMPL(cfg.MALE_SMPL_FILE).to(self.device)

        # Create loss functions
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_keypoints_3d = nn.L1Loss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

    def apply_smpl(self, pose, shape):
        flag_stage = False
        if shape.dim() == 3:  # s, bs, 10
            bs, s, _ = shape.shape
            flag_stage = True
            pose = pose.reshape(bs * s, 24, 3, 3)
            shape = shape.reshape(bs * s, 10)

        vertices = self.smpl(pose, shape)
        if flag_stage:
            vertices = vertices.reshape(bs, s, 6890, 3)
        return vertices

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        """Compute SMPL parameter loss for the examples that SMPL annotations are available."""
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]

        # bs, s, 24, 3, 3
        bs, s, _ = pred_betas_valid.shape
        gt_rotmat_valid = rodrigues(gt_pose[has_smpl == 1].view(-1, 3)).view(bs, 24, 3, 3)
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            gt_rotmat_valid = gt_rotmat_valid.unsqueeze(1).expand_as(pred_rotmat_valid)
            gt_betas_valid = gt_betas_valid.unsqueeze(1).expand_as(pred_betas_valid)
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def error_adaptive_weight(self, fit_joint_error):
        weight = (1 - 10 * fit_joint_error)
        weight[weight <= 0] = 0
        return weight

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, weight=None):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the weight
        The available keypoints are different for each dataset.
        """
        if gt_keypoints_2d.shape[2] == 3:
            conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        else:
            conf = 1

        if weight is not None:
            weight = weight[:, None, None]
            conf = conf * weight

        gt_keypoints_2d = gt_keypoints_2d[:, :, :-1].unsqueeze(1).expand_as(pred_keypoints_2d)
        conf = conf.unsqueeze(1)
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d)).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, weight=None):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the weight
        """
        if gt_keypoints_3d.shape[2] == 3:
            tmp = gt_keypoints_3d.new_ones(gt_keypoints_3d.shape[0], gt_keypoints_3d.shape[1], 1)
            gt_keypoints_3d = torch.cat((gt_keypoints_3d, tmp), dim=2)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]

        if weight is not None:
            weight = weight[has_pose_3d == 1, None, None]
            conf = conf * weight

        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            # Align the origin of the first 24 keypoints with the pelvis.
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]

            pred_pelvis = (pred_keypoints_3d[:, :, 2, :] + pred_keypoints_3d[:, :, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, :, None, :]

            # # Align the origin of the first 24 keypoints with the pelvis.
            # gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            # pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            # gt_keypoints_3d[:, :24, :] = gt_keypoints_3d[:, :24, :] - gt_pelvis[:, None, :]
            # pred_keypoints_3d[:, :24, :] = pred_keypoints_3d[:, :24, :] - pred_pelvis[:, None, :]
            #
            # # Align the origin of the 24 SMPL keypoints with the root joint.
            # gt_root_joint = gt_keypoints_3d[:, 24]
            # pred_root_joint = pred_keypoints_3d[:, 24]
            # gt_keypoints_3d[:, 24:, :] = gt_keypoints_3d[:, 24:, :] - gt_root_joint[:, None, :]
            # pred_keypoints_3d[:, 24:, :] = pred_keypoints_3d[:, 24:, :] - pred_root_joint[:, None, :]

            gt_keypoints_3d = gt_keypoints_3d.unsqueeze(1).expand_as(pred_keypoints_3d)
            conf = conf.unsqueeze(1)
            return (conf * self.criterion_keypoints_3d(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, weight=None):
        """
        Compute 3D SMPL keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the weight
        """
        if gt_keypoints_3d.shape[2] == 3:
            tmp = gt_keypoints_3d.new_ones(gt_keypoints_3d.shape[0], gt_keypoints_3d.shape[1], 1)
            gt_keypoints_3d = torch.cat((gt_keypoints_3d, tmp), dim=2)

        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]

        if weight is not None:
            weight = weight[has_pose_3d == 1, None, None]
            conf = conf * weight

        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_root_joint = gt_keypoints_3d[:, 0, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_root_joint[:, None, :]

            pred_root_joint = pred_keypoints_3d[:, :, 0, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_root_joint[:, :, None, :]
            gt_keypoints_3d = gt_keypoints_3d.unsqueeze(1).expand_as(pred_keypoints_3d)
            conf = conf.unsqueeze(1)

            return (conf * self.criterion_keypoints_3d(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl, weight=None):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]

        if weight is not None:
            weight = weight[has_smpl == 1, None, None].unqueeze(1)
        else:
            weight = 1

        if len(gt_vertices_with_shape) > 0:
            gt_vertices_with_shape = gt_vertices_with_shape.unsqueeze(1).expand_as(pred_vertices_with_shape)
            loss = self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
            loss = (loss * weight).mean()
            return loss
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def forward(self, pred_para, input_batch, return_vis=False):
        """Training step."""
        dtype = torch.float32

        # load predicted smpl paras
        if pred_para[-1].dim() == 2:
            pred_para = (p.unsqueeze(1) for p in pred_para)
        pred_pose, pred_shape, pred_camera = pred_para
        bs, s,  _ = pred_shape.shape
        pred_vertices = self.apply_smpl(pred_pose, pred_shape)

        # Grab data from the batch
        gt_keypoints_2d = input_batch['keypoints'].to(self.device)
        gt_keypoints_3d = input_batch['pose_3d'].to(self.device)
        has_pose_3d = input_batch['has_pose_3d'].to(self.device)
        gt_keypoints_2d_smpl = input_batch['keypoints_smpl'].to(self.device)
        gt_keypoints_3d_smpl = input_batch['pose_3d_smpl'].to(self.device)
        has_pose_3d_smpl = input_batch['has_pose_3d_smpl'].to(self.device)
        gt_pose = input_batch['pose'].to(self.device)
        gt_betas = input_batch['betas'].to(self.device)
        has_smpl = input_batch['has_smpl'].to(self.device)
        gender = input_batch['gender'].to(self.device)

        batch_size = pred_shape.shape[0]
        gt_vertices = gt_pose.new_zeros([batch_size, 6890, 3])
        with torch.no_grad():
            gt_vertices[gender < 0] = self.smpl(gt_pose[gender < 0], gt_betas[gender < 0])
            gt_vertices[gender == 0] = self.male_smpl(gt_pose[gender == 0], gt_betas[gender == 0])
            gt_vertices[gender == 1] = self.female_smpl(gt_pose[gender == 1], gt_betas[gender == 1])

        # compute losses
        losses = {}
        sampled_vertices = pred_vertices
        if self.options.adaptive_weight:
            # Get the confidence of the GT mesh, which is used as the weight of loss item.
            # The confidence is related to the fitting error and for the data with GT SMPL parameters,
            # the confidence is 1.0
            fit_joint_error = input_batch['fit_joint_error']
            ada_weight = self.error_adaptive_weight(fit_joint_error).type(dtype)
        else:
            ada_weight = None

        '''loss on mesh'''
        if self.options.lam_mesh > 0:
            loss_mesh = self.shape_loss(sampled_vertices, gt_vertices, has_smpl, ada_weight) * self.options.lam_mesh
            losses['mesh'] = loss_mesh

        '''loss on joints'''
        weight_key = sampled_vertices.new_ones(batch_size)
        if self.options.gtkey3d_from_mesh:
            # For the data without GT 3D keypoints but with SMPL parameters, we can
            # get the GT 3D keypoints from the mesh. The confidence of the keypoints
            # is related to the confidence of the mesh.
            gt_keypoints_3d_mesh = self.smpl.get_train_joints(gt_vertices)
            gt_keypoints_3d_mesh = torch.cat([gt_keypoints_3d_mesh,
                                              gt_keypoints_3d_mesh.new_ones([batch_size, 24, 1])],
                                             dim=-1)
            valid = has_smpl > has_pose_3d
            gt_keypoints_3d[valid] = gt_keypoints_3d_mesh[valid]
            has_pose_3d[valid] = 1
            if ada_weight is not None:
                weight_key[valid] = ada_weight[valid]

        sampled_joints_3d = self.smpl.get_train_joints(sampled_vertices.view(bs*s, 6890, 3)).view(bs, s, -1, 3)
        loss_keypoints_3d = self.keypoint_3d_loss(sampled_joints_3d, gt_keypoints_3d, has_pose_3d, weight_key)
        loss_keypoints_3d = loss_keypoints_3d * self.options.lam_key3d
        losses['key3D'] = loss_keypoints_3d

        sampled_joints_2d = \
            orthographic_projection(sampled_joints_3d.view(bs*s, -1, 3),
                                    pred_camera.view(bs*s, -1))[:, :, :2].view(bs, s, -1, 2)
        loss_keypoints_2d = self.keypoint_loss(sampled_joints_2d, gt_keypoints_2d) * self.options.lam_key2d
        losses['key2D'] = loss_keypoints_2d

        # We add the 24 joints of SMPL model for the training on SURREAL dataset.
        if self.options.use_smpl_joints:
            weight_key_smpl = sampled_vertices.new_ones(batch_size)
            if self.options.gtkey3d_from_mesh:
                gt_keypoints_3d_mesh = self.smpl.get_smpl_joints(gt_vertices)
                gt_keypoints_3d_mesh = torch.cat([gt_keypoints_3d_mesh,
                                                  gt_keypoints_3d_mesh.new_ones([batch_size, 24, 1])],
                                                 dim=-1)
                valid = has_smpl > has_pose_3d_smpl
                gt_keypoints_3d_smpl[valid] = gt_keypoints_3d_mesh[valid]
                has_pose_3d_smpl[valid] = 1
                if ada_weight is not None:
                    weight_key_smpl[valid] = ada_weight[valid]

            sampled_joints_3d_smpl = self.smpl.get_smpl_joints(sampled_vertices.view(bs*s, -1, 3)).view(bs, s, -1, 3)
            loss_keypoints_3d_smpl = self.smpl_keypoint_3d_loss(sampled_joints_3d_smpl, gt_keypoints_3d_smpl,
                                                                has_pose_3d_smpl, weight_key_smpl)
            loss_keypoints_3d_smpl = loss_keypoints_3d_smpl * self.options.lam_key3d_smpl
            losses['key3D_smpl'] = loss_keypoints_3d_smpl

            sampled_joints_2d_smpl = orthographic_projection(sampled_joints_3d_smpl.view(bs*s, -1, 3),
                                    pred_camera.view(bs*s, -1))[:, :, :2].view(bs, s, -1, 2)

            loss_keypoints_2d_smpl = self.keypoint_loss(sampled_joints_2d_smpl,
                                                        gt_keypoints_2d_smpl) * self.options.lam_key2d_smpl
            losses['key2D_smpl'] = loss_keypoints_2d_smpl

        '''SMPL paras regression loss'''
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_pose, pred_shape, gt_pose, gt_betas, has_smpl)
        loss_regr_pose = loss_regr_pose * self.options.lam_smpl_pose
        loss_regr_betas = loss_regr_betas * self.options.lam_smpl_beta
        losses['pose'] = loss_regr_pose
        losses['beta'] = loss_regr_betas

        # for visualize
        vis_data = None
        if return_vis:
            data = {}
            vis_num = min(4, batch_size)
            data['image'] = input_batch['img_orig'][0:vis_num].detach()
            data['gt_vert'] = gt_vertices[0:vis_num].detach()
            data['gt_joint'] = gt_keypoints_2d[0:vis_num].detach()
            data['pred_vert'] = sampled_vertices[0:vis_num, -1].detach()
            data['pred_cam'] = pred_camera[0:vis_num, -1].detach()
            data['pred_joint'] = sampled_joints_2d[0:vis_num, -1].detach()
            vis_data = data

        return losses, vis_data

# Without data selection
class MeshLoss2(nn.Module):
    def __init__(self, options, device):
        super().__init__()
        self.options = options
        self.device = device

        # prepare SMPL model
        self.smpl = SMPL().to(self.device)
        self.female_smpl = SMPL(cfg.FEMALE_SMPL_FILE).to(self.device)
        self.male_smpl = SMPL(cfg.MALE_SMPL_FILE).to(self.device)

        # Create loss functions
        self.criterion_shape = nn.L1Loss(reduction='none').to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_keypoints_3d = nn.L1Loss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss(reduction='none').to(self.device)

    def apply_smpl(self, pose, shape):
        flag_stage = False
        if shape.dim() == 3:  # s, bs, 10
            bs, s, _ = shape.shape
            flag_stage = True
            pose = pose.reshape(bs * s, 24, 3, 3)
            shape = shape.reshape(bs * s, 10)

        vertices = self.smpl(pose, shape)
        if flag_stage:
            vertices = vertices.reshape(bs, s, 6890, 3)
        return vertices

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl, weight=None):
        """Compute SMPL parameter loss for the examples that SMPL annotations are available."""
        B, S, _ = pred_betas.shape
        gt_rotmat = rodrigues(gt_pose.view(-1, 3)).view(B, 24, 3, 3)
        if weight is None:
            weight = pred_betas.new_ones(B)
        weight = weight * has_smpl

        gt_rotmat = gt_rotmat.unsqueeze(1)
        gt_betas = gt_betas.unsqueeze(1)

        loss_pose = self.criterion_regr(pred_rotmat, gt_rotmat)
        loss_pose = (weight * loss_pose.mean(dim=[1, 2, 3, 4])).mean()
        loss_betas = self.criterion_regr(pred_betas, gt_betas)
        loss_betas = (weight * loss_betas.mean(dim=[1, 2])).mean()
        return loss_pose, loss_betas

    def error_adaptive_weight(self, fit_joint_error):
        weight = (1 - 10 * fit_joint_error)
        weight[weight <= 0] = 0
        return weight

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, weight=None):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the weight
        The available keypoints are different for each dataset.
        """
        B = pred_keypoints_2d.shape[0]
        if gt_keypoints_2d.shape[2] == 3:
            conf = gt_keypoints_2d[:, :, [-1]]
        else:
            conf = 1

        if weight is None:
            weight = pred_keypoints_2d.new_ones(B)
        weight = weight[:, None, None]
        conf = conf * weight

        gt_keypoints_2d = gt_keypoints_2d[:, :, :-1].unsqueeze(1)
        conf = conf.unsqueeze(1)
        loss = self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d)
        loss = (conf * loss).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, weight=None):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the weight
        """
        B, K, C = gt_keypoints_3d.shape
        if C == 4:
            conf = gt_keypoints_3d[:, :, [-1]]
        else:
            conf = 1

        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]

        if weight is None:
            weight = pred_keypoints_3d.new_ones(B)
        weight = weight * has_pose_3d
        weight = weight[:, None, None]
        conf = conf * weight

        # Align the origin of the first 24 keypoints with the pelvis.
        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]

        pred_pelvis = (pred_keypoints_3d[:, :, 2, :] + pred_keypoints_3d[:, :, 3, :]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, :, None, :]

        gt_keypoints_3d = gt_keypoints_3d.unsqueeze(1)
        conf = conf.unsqueeze(1)
        loss = self.criterion_keypoints_3d(pred_keypoints_3d, gt_keypoints_3d)
        loss = (conf * loss).mean()
        return loss

    def smpl_keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, weight=None):
        """
        Compute 3D SMPL keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the weight
        """

        B, K, C = gt_keypoints_3d.shape
        if C == 4:
            conf = gt_keypoints_3d[:, :, [-1]]
        else:
            conf = 1

        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]

        if weight is None:
            weight = pred_keypoints_3d.new_ones(B)
        weight = weight * has_pose_3d
        weight = weight[:, None, None]
        conf = conf * weight

        # Align the origin of the first 24 keypoints with the pelvis.
        gt_root_joint = gt_keypoints_3d[:, 0, :]
        gt_keypoints_3d = gt_keypoints_3d - gt_root_joint[:, None, :]

        pred_root_joint = pred_keypoints_3d[:, :, 0, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_root_joint[:, :, None, :]

        gt_keypoints_3d = gt_keypoints_3d.unsqueeze(1)
        conf = conf.unsqueeze(1)
        loss = self.criterion_keypoints_3d(pred_keypoints_3d, gt_keypoints_3d)
        loss = (conf * loss).mean()
        return loss

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl, weight=None):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        B, S, N, C = pred_vertices.shape
        if weight is None:
            weight = pred_vertices.new_ones(B)
        weight = weight * has_smpl
        weight = weight[:, None, None, None]
        gt_vertices = gt_vertices.unsqueeze(1).expand_as(pred_vertices)
        loss = self.criterion_shape(pred_vertices, gt_vertices)
        loss = (loss * weight).mean()
        return loss

    def forward(self, pred_para, input_batch, return_vis=False):
        """Training step."""
        dtype = torch.float32

        # load predicted smpl paras
        if pred_para[-1].dim() == 2:
            pred_para = (p.unsqueeze(1) for p in pred_para)
        pred_pose, pred_shape, pred_camera = pred_para
        bs, s,  _ = pred_shape.shape
        pred_vertices = self.apply_smpl(pred_pose, pred_shape)

        # Grab data from the batch
        gt_keypoints_2d = input_batch['keypoints'].to(self.device)
        gt_keypoints_3d = input_batch['pose_3d'].to(self.device)
        has_pose_3d = input_batch['has_pose_3d'].to(self.device)
        gt_keypoints_2d_smpl = input_batch['keypoints_smpl'].to(self.device)
        gt_keypoints_3d_smpl = input_batch['pose_3d_smpl'].to(self.device)
        has_pose_3d_smpl = input_batch['has_pose_3d_smpl'].to(self.device)
        gt_pose = input_batch['pose'].to(self.device)
        gt_betas = input_batch['betas'].to(self.device)
        has_smpl = input_batch['has_smpl'].to(self.device)
        gender = input_batch['gender'].to(self.device)

        batch_size = pred_shape.shape[0]
        gt_vertices = gt_pose.new_zeros([batch_size, 6890, 3])
        with torch.no_grad():
            gt_vertices[gender < 0] = self.smpl(gt_pose[gender < 0], gt_betas[gender < 0])
            gt_vertices[gender == 0] = self.male_smpl(gt_pose[gender == 0], gt_betas[gender == 0])
            gt_vertices[gender == 1] = self.female_smpl(gt_pose[gender == 1], gt_betas[gender == 1])

        # compute losses
        losses = {}
        sampled_vertices = pred_vertices
        if self.options.adaptive_weight:
            # Get the confidence of the GT mesh, which is used as the weight of loss item.
            # The confidence is related to the fitting error and for the data with GT SMPL parameters,
            # the confidence is 1.0
            fit_joint_error = input_batch['fit_joint_error'].to(self.device)
            ada_weight = self.error_adaptive_weight(fit_joint_error).type(dtype)
        else:
            ada_weight = None

        '''loss on mesh'''
        if self.options.lam_mesh > 0:
            loss_mesh = self.shape_loss(sampled_vertices, gt_vertices, has_smpl, ada_weight) * self.options.lam_mesh
            losses['mesh'] = loss_mesh

        '''loss on joints'''
        weight_key = sampled_vertices.new_ones(batch_size)
        if self.options.gtkey3d_from_mesh:
            # For the data without GT 3D keypoints but with SMPL parameters, we can
            # get the GT 3D keypoints from the mesh. The confidence of the keypoints
            # is related to the confidence of the mesh.
            gt_keypoints_3d_mesh = self.smpl.get_train_joints(gt_vertices)
            gt_keypoints_3d_mesh = torch.cat([gt_keypoints_3d_mesh,
                                              gt_keypoints_3d_mesh.new_ones([batch_size, 24, 1])],
                                             dim=-1)
            valid = has_smpl > has_pose_3d
            gt_keypoints_3d[valid] = gt_keypoints_3d_mesh[valid]
            has_pose_3d[valid] = 1
            if ada_weight is not None:
                weight_key[valid] = ada_weight[valid]

        sampled_joints_3d = self.smpl.get_train_joints(sampled_vertices.view(bs*s, 6890, 3)).view(bs, s, -1, 3)
        loss_keypoints_3d = self.keypoint_3d_loss(sampled_joints_3d, gt_keypoints_3d, has_pose_3d, weight_key)
        loss_keypoints_3d = loss_keypoints_3d * self.options.lam_key3d
        losses['key3D'] = loss_keypoints_3d

        sampled_joints_2d = \
            orthographic_projection(sampled_joints_3d.view(bs*s, -1, 3),
                                    pred_camera.view(bs*s, -1))[:, :, :2].view(bs, s, -1, 2)
        loss_keypoints_2d = self.keypoint_loss(sampled_joints_2d, gt_keypoints_2d) * self.options.lam_key2d
        losses['key2D'] = loss_keypoints_2d

        # We add the 24 joints of SMPL model for the training on SURREAL dataset.
        if self.options.use_smpl_joints:
            weight_key_smpl = sampled_vertices.new_ones(batch_size)
            if self.options.gtkey3d_from_mesh:
                gt_keypoints_3d_mesh = self.smpl.get_smpl_joints(gt_vertices)
                gt_keypoints_3d_mesh = torch.cat([gt_keypoints_3d_mesh,
                                                  gt_keypoints_3d_mesh.new_ones([batch_size, 24, 1])],
                                                 dim=-1)
                valid = has_smpl > has_pose_3d_smpl
                gt_keypoints_3d_smpl[valid] = gt_keypoints_3d_mesh[valid]
                has_pose_3d_smpl[valid] = 1
                if ada_weight is not None:
                    weight_key_smpl[valid] = ada_weight[valid]

            sampled_joints_3d_smpl = self.smpl.get_smpl_joints(sampled_vertices.view(bs*s, -1, 3)).view(bs, s, -1, 3)
            loss_keypoints_3d_smpl = self.smpl_keypoint_3d_loss(sampled_joints_3d_smpl, gt_keypoints_3d_smpl,
                                                                has_pose_3d_smpl, weight_key_smpl)
            loss_keypoints_3d_smpl = loss_keypoints_3d_smpl * self.options.lam_key3d_smpl
            losses['key3D_smpl'] = loss_keypoints_3d_smpl

            sampled_joints_2d_smpl = orthographic_projection(sampled_joints_3d_smpl.view(bs*s, -1, 3),
                                    pred_camera.view(bs*s, -1))[:, :, :2].view(bs, s, -1, 2)

            loss_keypoints_2d_smpl = self.keypoint_loss(sampled_joints_2d_smpl,
                                                        gt_keypoints_2d_smpl) * self.options.lam_key2d_smpl
            losses['key2D_smpl'] = loss_keypoints_2d_smpl

        '''SMPL paras regression loss'''
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_pose, pred_shape, gt_pose, gt_betas, has_smpl, ada_weight)
        loss_regr_pose = loss_regr_pose * self.options.lam_smpl_pose
        loss_regr_betas = loss_regr_betas * self.options.lam_smpl_beta
        losses['pose'] = loss_regr_pose
        losses['beta'] = loss_regr_betas

        # for visualize
        vis_data = None
        if return_vis:
            data = {}
            vis_num = min(4, batch_size)
            data['image'] = input_batch['img_orig'][0:vis_num].detach()
            data['gt_vert'] = gt_vertices[0:vis_num].detach()
            data['gt_joint'] = gt_keypoints_2d[0:vis_num].detach()
            data['pred_vert'] = sampled_vertices[0:vis_num, -1].detach()
            data['pred_cam'] = pred_camera[0:vis_num, -1].detach()
            data['pred_joint'] = sampled_joints_2d[0:vis_num, -1].detach()
            vis_data = data

        return losses, vis_data




def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4


def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    # mask_c1 = mask_d2 * (1 - mask_d0_d1)
    # mask_c2 = (1 - mask_d2) * mask_d0_nd1
    # mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)

    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def estimate_translation_np(S, joints_2d, joints_conf, focal_length=5000, img_size=224):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length,focal_length])
    # optical center
    center = np.array([img_size/2., img_size/2.])

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(S, joints_2d, focal_length=5000., img_size=224.):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """

    device = S.device
    # Use only joints 25:49 (GT joints)
    S = S[:, 25:, :].cpu().numpy()
    joints_2d = joints_2d[:, 25:, :].cpu().numpy()
    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size)
    return torch.from_numpy(trans).to(device)


class FitsDict():
    """ Dictionary keeping track of the best fit per image in the training set """
    def __init__(self, options, device, dataset_infos):
        self.device = device
        self.options = options
        self.dataset_infos = dataset_infos
        # array used to flip SMPL pose parameters
        self.flipped_parts = torch.tensor(constants.SMPL_POSE_FLIP_PERM, dtype=torch.int64).to(device)

        # Load dictionary state
        all_fits = torch.zeros([0, 72+10], dtype=torch.float32)
        for ds_info in dataset_infos:
            ds_name = ds_info['ds_name']
            ds_len = ds_info['len']
            try:
                dict_file = os.path.join(options.checkpoint_dir, ds_name + '_fits.npy')
                fits = torch.from_numpy(np.load(dict_file))
            except IOError:
                try:
                    # Dictionary does not exist, so populate with static fits
                    dict_file = os.path.join(config.STATIC_FITS_DIR, ds_name + '_fits.npy')
                    fits = torch.from_numpy(np.load(dict_file))
                except IOError:
                    fits = torch.zeros(ds_len, 82)
            all_fits = torch.cat([all_fits, fits], dim=0)
        self.all_fits = all_fits

    def save(self):
        """ Save dictionary state to disk """
        for ds_info in self.dataset_infos:
            ds_name = ds_info['ds_name']
            begin_idx = ds_info['begin_idx']
            ds_len = ds_info['len']
            dict_file = os.path.join(self.options.checkpoint_dir, ds_name + '_fits.npy')
            np.save(dict_file, self.fits_dict[begin_idx:begin_idx+ds_len].cpu().numpy())

    def get_para(self, opt_ind, rot, is_flipped):
        """ Retrieve dictionary entries """
        params = self.all_fits[opt_ind.cpu(), :].clone()
        pose = params[:, :72].to(self.device)
        betas = params[:, 72:].to(self.device)
        # Apply flipping and rotation
        pose = self.flip_pose(self.rotate_pose(pose, rot), is_flipped)
        return pose, betas

    def flip_pose(self, pose, is_flipped):
        """flip SMPL pose parameters"""
        is_flipped = is_flipped.byte()
        pose_f = pose.clone()
        pose_f[is_flipped, :] = pose[is_flipped][:, self.flipped_parts]
        # we also negate the second and the third dimension of the axis-angle representation
        pose_f[is_flipped, 1::3] *= -1
        pose_f[is_flipped, 2::3] *= -1
        return pose_f

    def rotate_pose(self, pose, rot):
        """Rotate SMPL pose parameters by rot degrees"""
        batch_size = pose.shape[0]
        pose = pose.clone()
        pi = torch.tensor(math.pi, device=self.device)
        cos = torch.cos(-pi * rot / 180.)
        sin = torch.sin(-pi * rot / 180.)
        zeros = torch.zeros_like(cos, device=cos.device)
        r3 = torch.zeros(cos.shape[0], 1, 3, device=cos.device)
        r3[:,0,-1] = 1
        R = torch.cat([torch.stack([cos, -sin, zeros], dim=-1).unsqueeze(1),
                       torch.stack([sin, cos, zeros], dim=-1).unsqueeze(1),
                       r3], dim=1)
        global_pose = pose[:, :3]
        global_pose_rotmat = angle_axis_to_rotation_matrix(global_pose)
        global_pose_rotmat_3b3 = global_pose_rotmat[:, :3, :3]
        global_pose_rotmat_3b3 = torch.matmul(R, global_pose_rotmat_3b3)
        global_pose_rotmat[:, :3, :3] = global_pose_rotmat_3b3
        global_pose_rotmat = global_pose_rotmat[:, :-1, :-1]

        # Convert rotation matrices to axis-angle
        global_pose = torch.cat([global_pose_rotmat,
                                 torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device).view(1, 3, 1).expand(batch_size, -1, -1)],
                                dim=-1)

        global_pose = rotation_matrix_to_angle_axis(global_pose).contiguous()
        # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
        global_pose[torch.isnan(global_pose)] = 0.0
        pose[:, :3] = global_pose
        return pose

    def update(self, opt_idx, pose, betas, mask):
        if mask.max() == 0:
            return
        mask = mask.cpu()
        opt_idx = opt_idx[mask]
        params = torch.cat([pose, betas], dim=-1).cpu()
        opt_idx = opt_idx.cpu()
        params = params[mask]
        self.all_fits[opt_idx, :] = params
        return


# With SPIN fitting
class MeshLoss3(MeshLoss2):
    def __init__(self, options, device, dataset_infos):
        super().__init__(options, device)
        self.pose_prior = MaxMixturePrior(prior_folder='data',
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        self.num_iters = 50
        self.step_size = 1e-2
        self.focal_length = 5000.0
        self.loss_threhold = 100.0 * (self.options.img_res / 224.0) ** 2

        self.smplx = _SMPL(config.SMPL_MODEL_DIR,
                         create_transl=False).to(self.device)

        # tmp = self.smplx()
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32, device=device))
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]

        self.fits_dict = FitsDict(options, device, dataset_infos)

    def get_opt_joints(self, smpl_output):
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        return joints

    def optimize(self, init_para, keypoints_2d):
        pred_rotmat, init_betas, init_camera = init_para
        batch_size = pred_rotmat.shape[0]

        # Convert predicted rotation matrices to axis-angle
        pred_rotmat_hom = torch.cat(
            [pred_rotmat.detach().view(-1, 3, 3).detach(),
             torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device).view(1, 3, 1).expand(
                batch_size * 24, -1, -1)], dim=-1)
        init_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
        # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
        init_pose[torch.isnan(init_pose)] = 0.0

        # Make camera translation a learnable parameter
        init_cam_t = torch.stack([init_camera[:, 1],
                                  init_camera[:, 2],
                                  2 * self.focal_length/(self.options.img_res * init_camera[:, 0] + 1e-9)],
                                 dim=-1)
        camera_translation = init_cam_t.detach().clone()
        camera_center = 0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device)

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_2d = (joints_2d + 1) * 0.5 * self.options.img_res
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]
        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters):
            smpl_output = self.smplx(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = self.get_opt_joints(smpl_output)

            loss = camera_fitting_loss(model_joints, camera_translation,
                                       init_cam_t, camera_center,
                                       joints_2d, joints_conf, focal_length=self.focal_length)
            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # Fix camera translation after optimizing camera
        camera_translation.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, betas, global_orient]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        for i in range(self.num_iters):
            smpl_output = self.smplx(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = self.get_opt_joints(smpl_output)

            loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                     joints_2d, joints_conf, self.pose_prior,
                                     focal_length=self.focal_length)
            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smplx(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            vertices = smpl_output.vertices
            model_joints = self.get_opt_joints(smpl_output)

            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        vertices = vertices.detach()
        joints = model_joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        out = {
            'vertices': vertices,
            'joints': joints,
            'pose': pose,
            'betas': betas,
            'camera_translation': camera_translation,
            'reprojection_loss': reprojection_loss
        }
        return out

    def get_old_opt_loss(self, opt_pose, opt_betas, keypoints_2d):
        batch_size = opt_pose.shape[0]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
        opt_output = self.smplx(betas=opt_betas, body_pose=opt_pose[:,3:], global_orient=opt_pose[:,:3])
        opt_joints = self.get_opt_joints(opt_output)

        # Get joint confidence
        keypoints_2d[..., :-1] = (keypoints_2d[..., :-1] + 1) * 0.5 * self.options.img_res
        camera_center = 0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device)

        opt_cam_t = estimate_translation(opt_joints, keypoints_2d, focal_length=self.focal_length, img_size=self.options.img_res)

        with torch.no_grad():
            joints_2d = keypoints_2d[:, :, :2]
            joints_conf = keypoints_2d[:, :, -1]
            reprojection_loss = body_fitting_loss(opt_pose[:, 3:], opt_betas, opt_joints, opt_cam_t, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')
        return reprojection_loss

    def forward(self, pred_para, input_batch, return_vis=False):
        """Training step."""
        dtype = torch.float32

        # load predicted smpl paras
        if pred_para[-1].dim() == 2:
            pred_para = (p.unsqueeze(1) for p in pred_para)
        pred_pose, pred_shape, pred_camera = pred_para
        bs, s,  _ = pred_shape.shape
        pred_vertices = self.apply_smpl(pred_pose, pred_shape)

        # Grab data from the batch
        gt_keypoints_2d = input_batch['keypoints'].to(self.device)
        gt_keypoints_op_2d = input_batch['keypoints_op'].to(self.device)
        gt_keypoints_3d = input_batch['pose_3d'].to(self.device)
        has_pose_3d = input_batch['has_pose_3d'].to(self.device)
        gt_keypoints_2d_smpl = input_batch['keypoints_smpl'].to(self.device)
        gt_keypoints_3d_smpl = input_batch['pose_3d_smpl'].to(self.device)
        has_pose_3d_smpl = input_batch['has_pose_3d_smpl'].to(self.device)
        gt_pose = input_batch['pose'].to(self.device)
        gt_betas = input_batch['betas'].to(self.device)
        has_smpl = input_batch['has_smpl'].to(self.device).float()
        gender = input_batch['gender'].to(self.device)

        batch_size = pred_shape.shape[0]

        # spin optimize
        # opt_valid = has_smpl < 1
        opt_valid = has_smpl < 1
        # opt_valid = has_smpl < 2; print('debug')
        update_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        if opt_valid.sum() > 0:
            keypoints2d = torch.cat([gt_keypoints_op_2d, gt_keypoints_2d], dim=1)

            opt_idx = input_batch['opt_idx'][opt_valid]
            rot = input_batch['rot'][opt_valid].to(self.device)
            flip = input_batch['flip'][opt_valid].to(self.device)
            old_pose, old_shape = self.fits_dict.get_para(opt_idx, rot, flip)
            old_loss = self.get_old_opt_loss(old_pose, old_shape, keypoints2d)
            joints_conf = keypoints2d[:, :, -1]
            old_loss = old_loss.sum(dim=-1) / (joints_conf ** 2).sum(dim=-1)

            init_pose = pred_pose[opt_valid, -1].detach()
            init_shape = pred_shape[opt_valid, -1].detach()
            init_camera = pred_camera[opt_valid, -1].detach()
            opt_res = self.optimize((init_pose, init_shape, init_camera), keypoints2d)
            new_loss = opt_res['reprojection_loss']
            new_loss = new_loss.sum(dim=-1) / (joints_conf ** 2).sum(dim=-1)

            opt_update = new_loss < old_loss
            # opt_update[:] = True; print('debug')
            opt_pose = opt_res['pose'] * opt_update[:, None] + old_pose * (~opt_update[:, None])
            opt_betas = opt_res['betas'] * opt_update[:, None] + old_shape * (~opt_update[:, None])
            opt_loss = new_loss * opt_update + old_loss * (~opt_update)
            opt_has_smpl = opt_loss < self.loss_threhold

            gt_pose[opt_valid] = opt_pose
            gt_betas = opt_betas
            has_smpl[opt_valid] = opt_has_smpl.float()
            gender[opt_valid] = -1
            update_mask[opt_valid] = opt_update

            # update fits dict
            all_idx = input_batch['opt_idx'].to(self.device)
            up_paras = [all_idx, gt_pose, gt_betas, update_mask]
            up_paras = utils.all_gather(up_paras)
            up_idx = torch.cat([t[0] for t in up_paras], dim=0)
            up_pose = torch.cat([t[1] for t in up_paras], dim=0)
            up_betas = torch.cat([t[2] for t in up_paras], dim=0)
            up_mask = torch.cat([t[3] for t in up_paras], dim=0)
            self.fits_dict.update(up_idx, up_pose, up_betas, up_mask)
        else:
            pass

        # loss
        gt_vertices = gt_pose.new_zeros([batch_size, 6890, 3])
        with torch.no_grad():
            gt_vertices[gender < 0] = self.smpl(gt_pose[gender < 0], gt_betas[gender < 0])
            gt_vertices[gender == 0] = self.male_smpl(gt_pose[gender == 0], gt_betas[gender == 0])
            gt_vertices[gender == 1] = self.female_smpl(gt_pose[gender == 1], gt_betas[gender == 1])


        # compute losses
        losses = {}
        sampled_vertices = pred_vertices
        if self.options.adaptive_weight:
            # Get the confidence of the GT mesh, which is used as the weight of loss item.
            # The confidence is related to the fitting error and for the data with GT SMPL parameters,
            # the confidence is 1.0
            # ada_weight = self.error_adaptive_weight(fit_joint_error).type(dtype)
            print('not supported yet')
        else:
            ada_weight = None

        '''loss on mesh'''
        if self.options.lam_mesh > 0:
            loss_mesh = self.shape_loss(sampled_vertices, gt_vertices, has_smpl, ada_weight) * self.options.lam_mesh
            losses['mesh'] = loss_mesh

        '''loss on joints'''
        weight_key = sampled_vertices.new_ones(batch_size)
        if self.options.gtkey3d_from_mesh:
            # For the data without GT 3D keypoints but with SMPL parameters, we can
            # get the GT 3D keypoints from the mesh. The confidence of the keypoints
            # is related to the confidence of the mesh.
            gt_keypoints_3d_mesh = self.smpl.get_train_joints(gt_vertices)
            gt_keypoints_3d_mesh = torch.cat([gt_keypoints_3d_mesh,
                                              gt_keypoints_3d_mesh.new_ones([batch_size, 24, 1])],
                                             dim=-1)
            valid = has_smpl > has_pose_3d
            gt_keypoints_3d[valid] = gt_keypoints_3d_mesh[valid]
            has_pose_3d[valid] = 1
            if ada_weight is not None:
                weight_key[valid] = ada_weight[valid]

        sampled_joints_3d = self.smpl.get_train_joints(sampled_vertices.view(bs*s, 6890, 3)).view(bs, s, -1, 3)
        loss_keypoints_3d = self.keypoint_3d_loss(sampled_joints_3d, gt_keypoints_3d, has_pose_3d, weight_key)
        loss_keypoints_3d = loss_keypoints_3d * self.options.lam_key3d
        losses['key3D'] = loss_keypoints_3d

        sampled_joints_2d = proj_2d(sampled_joints_3d.view(bs*s, -1, 3),
                                    pred_camera.view(bs*s, -1))[:, :, :2].view(bs, s, -1, 2)

        loss_keypoints_2d = self.keypoint_loss(sampled_joints_2d, gt_keypoints_2d) * self.options.lam_key2d
        losses['key2D'] = loss_keypoints_2d

        # We add the 24 joints of SMPL model for the training on SURREAL dataset.
        if self.options.use_smpl_joints:
            weight_key_smpl = sampled_vertices.new_ones(batch_size)
            if self.options.gtkey3d_from_mesh:
                gt_keypoints_3d_mesh = self.smpl.get_smpl_joints(gt_vertices)
                gt_keypoints_3d_mesh = torch.cat([gt_keypoints_3d_mesh,
                                                  gt_keypoints_3d_mesh.new_ones([batch_size, 24, 1])],
                                                 dim=-1)
                valid = has_smpl > has_pose_3d_smpl
                gt_keypoints_3d_smpl[valid] = gt_keypoints_3d_mesh[valid]
                has_pose_3d_smpl[valid] = 1
                if ada_weight is not None:
                    weight_key_smpl[valid] = ada_weight[valid]

            sampled_joints_3d_smpl = self.smpl.get_smpl_joints(sampled_vertices.view(bs*s, -1, 3)).view(bs, s, -1, 3)
            loss_keypoints_3d_smpl = self.smpl_keypoint_3d_loss(sampled_joints_3d_smpl, gt_keypoints_3d_smpl,
                                                                has_pose_3d_smpl, weight_key_smpl)
            loss_keypoints_3d_smpl = loss_keypoints_3d_smpl * self.options.lam_key3d_smpl
            losses['key3D_smpl'] = loss_keypoints_3d_smpl

            sampled_joints_2d_smpl = proj_2d(sampled_joints_3d_smpl.view(bs*s, -1, 3),
                                    pred_camera.view(bs*s, -1))[:, :, :2].view(bs, s, -1, 2)

            loss_keypoints_2d_smpl = self.keypoint_loss(sampled_joints_2d_smpl,
                                                        gt_keypoints_2d_smpl) * self.options.lam_key2d_smpl
            losses['key2D_smpl'] = loss_keypoints_2d_smpl

        '''SMPL paras regression loss'''
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_pose, pred_shape, gt_pose, gt_betas, has_smpl, ada_weight)
        loss_regr_pose = loss_regr_pose * self.options.lam_smpl_pose
        loss_regr_betas = loss_regr_betas * self.options.lam_smpl_beta
        losses['pose'] = loss_regr_pose
        losses['beta'] = loss_regr_betas

        # for visualize
        vis_data = None
        if return_vis:
            data = {}
            vis_num = min(4, batch_size)
            data['image'] = input_batch['img_orig'][0:vis_num].detach()
            data['gt_vert'] = gt_vertices[0:vis_num].detach()
            data['gt_joint'] = gt_keypoints_2d[0:vis_num].detach()
            data['pred_vert'] = sampled_vertices[0:vis_num, -1].detach()
            data['pred_cam'] = pred_camera[0:vis_num, -1].detach()
            data['pred_joint'] = sampled_joints_2d[0:vis_num, -1].detach()
            vis_data = data

        return losses, vis_data






class JointEvaluator(nn.Module):
    def __init__(self, options, device):
        super().__init__()
        self.options = options
        self.device = device

        # prepare SMPL model
        self.smpl = SMPL().to(self.device)
        self.female_smpl = SMPL(cfg.FEMALE_SMPL_FILE).to(self.device)
        self.male_smpl = SMPL(cfg.MALE_SMPL_FILE).to(self.device)
        self.joint_mapper = cfg.J24_TO_J17 if options.val_dataset == 'mpi-inf-3dhp' else cfg.J24_TO_J14
        self.pred_joints = []
        self.gt_joints = []
        self.mpjpe = []

    def apply_smpl(self, pose, shape):
        flag_stage = False
        if shape.dim() == 3:  # s, bs, 10
            bs, s, _ = shape.shape
            flag_stage = True
            pose = pose.reshape(bs * s, 24, 3, 3)
            shape = shape.reshape(bs * s, 10)

        vertices = self.smpl(pose, shape)
        if flag_stage:
            vertices = vertices.reshape(bs, s, 6890, 3)
        return vertices

    def forward(self, pred_para, input_batch):
        """Training step."""
        dtype = torch.float32
        # load predicted smpl paras
        if pred_para[-1].dim() == 3:
            pred_para = (p[:, -1] for p in pred_para)
        pred_pose, pred_shape, pred_camera = pred_para
        pred_vertices = self.apply_smpl(pred_pose, pred_shape)
        pred_joints_3d = self.smpl.get_train_joints(pred_vertices)[:, self.joint_mapper]

        if self.options.val_dataset == '3dpw':
            gt_pose = input_batch['pose'].to(self.device)
            gt_betas = input_batch['betas'].to(self.device)
            gender = input_batch['gender'].to(self.device)
            batch_size = pred_shape.shape[0]
            gt_vertices = gt_pose.new_zeros([batch_size, 6890, 3])
            with torch.no_grad():
                gt_vertices[gender < 0] = self.smpl(gt_pose[gender < 0], gt_betas[gender < 0])
                gt_vertices[gender == 0] = self.male_smpl(gt_pose[gender == 0], gt_betas[gender == 0])
                gt_vertices[gender == 1] = self.female_smpl(gt_pose[gender == 1], gt_betas[gender == 1])
            gt_joints_3d = self.smpl.get_train_joints(gt_vertices)[:, self.joint_mapper]
        else:
            gt_joints_3d = input_batch['pose_3d'][:, self.joint_mapper, :3].to(self.device)

        gt_pelvis = (gt_joints_3d[:, [2]] + gt_joints_3d[:, [3]]) / 2
        gt_joints_3d = gt_joints_3d - gt_pelvis
        pred_pelvis = (pred_joints_3d[:, [2]] + pred_joints_3d[:, [3]]) / 2
        pred_joints_3d = pred_joints_3d - pred_pelvis
        return pred_joints_3d, gt_joints_3d
