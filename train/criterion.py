from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from torchvision.utils import make_grid
from train.base_trainer import BaseTrainer
from datasets import create_dataset
from models import SMPL
from models.geometric_layers import orthographic_projection, rodrigues
import utils.config as cfg


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



