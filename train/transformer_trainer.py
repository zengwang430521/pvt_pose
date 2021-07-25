"""
This file includes the full training procedure.
Codes are adapted from https://github.com/nkolot/GraphCMR
"""
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
from utils import CheckpointDataLoader, CheckpointSaver
import sys
import time
from tqdm import tqdm
import numpy as np
import utils.config as cfg
from models.transformer_net import TNet
from models.pvt import pvt_tiny, pvt_small, pvt_medium, pvt_large, pvt2048_small
from models.my_pvt import mypvt2_small
from models.pvt_nc import pvt_nc_tiny, pvt_nc_small, pvt_nc_medium, pvt_nc_large, pvt_nc2_small
from models.pvt_impr1 import pvt_small_impr1_peg
from models.hmr import HMR
from utils.vis import visualize_vert
from models.my_pvt9 import mypvt9_small
from models.my_pvt14_3 import mypvt14_3_small
from models.my_pvt20 import mypvt20_small, mypvt20c_small
from models.my_pvt21 import mypvt21_small
from models.my_pvt22 import mypvt22_small
from models.my_pvt25c import mypvt2520_small, mypvt2520_2_small
from models.my_pvt2520_3 import mypvt2520_3_small
from models.my_pvt25g import mypvt2520g_small, mypvt25g_small
from models.my_pvt23 import mypvt2320_small


from models.pvt_impr8 import pvt_small_impr8_peg


model_dict = {
    'TMR': TNet,
    'hmr': HMR,
    'pvt_tiny': pvt_tiny,
    'pvt_small': pvt_small,
    'pvt_medium': pvt_medium,
    'pvt_large': pvt_large,
    'pvt_nc_tiny': pvt_nc_tiny,
    'pvt_nc_small': pvt_nc_small,
    'pvt_nc2_small': pvt_nc2_small,
    'pvt_nc_medium': pvt_nc_medium,
    'pvt_nc_large': pvt_nc_large,
    'pvt2048_small': pvt2048_small,

    'mypvt2_small': mypvt2_small,
    'mypvt9_small': mypvt9_small,
    'mypvt20_small': mypvt20_small,
    'mypvt20c_small': mypvt20c_small,
    'mypvt21_small': mypvt21_small,
    'mypvt22_small': mypvt22_small,
    'mypvt2320_small': mypvt2320_small,
    'mypvt2520_small': mypvt2520_small,
    'mypvt2520_2_small': mypvt2520_2_small,
    'mypvt2520_3_small': mypvt2520_3_small,
    'mypvt2520g_small': mypvt2520g_small,
    'mypvt25g_small': mypvt25g_small,

    'mypvt14_3_small': mypvt14_3_small,
    'pvt_small_impr1_peg': pvt_small_impr1_peg,
    'pvt_small_impr8_peg': pvt_small_impr8_peg,

}


class TransformerTrainer(BaseTrainer):
    def init_fn(self):
        model_class = model_dict[self.options.model]
        if 'pvt' in self.options.model:
            self.TNet = model_class().to(self.device)
        else:
            self.TNet = model_class(self.options).to(self.device)

        # create training dataset
        self.train_ds = create_dataset(self.options.dataset, self.options, use_IUV=False)

        # prepare SMPL model
        self.smpl = SMPL().to(self.device)
        self.female_smpl = SMPL(cfg.FEMALE_SMPL_FILE).to(self.device)
        self.male_smpl = SMPL(cfg.MALE_SMPL_FILE).to(self.device)

        # Setup an optimizer
        self.optimizer = torch.optim.Adam(
            params=list(self.TNet.parameters()),
            lr=self.options.lr,
            betas=(self.options.adam_beta1, 0.999),
            weight_decay=self.options.wd)
        self.models_dict = {'TNet': self.TNet}
        self.optimizers_dict = {'optimizer': self.optimizer}

        # Create loss functions
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_keypoints_3d = nn.L1Loss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

        # LSP indices from full list of keypoints
        self.to_lsp = list(range(14))
        if self.options.use_renderer:
            from utils.renderer import Renderer, visualize_reconstruction, vis_mesh
            self.renderer = Renderer(faces=self.smpl.faces.cpu().numpy())
        else:
            self.renderer = None

        # Optionally start training from a pretrained checkpoint
        # Note that this is different from resuming training
        # For the latter use --resume
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    def apply_smpl(self, pose, shape):
        flag_stage = False
        if shape.dim() == 3:  # s, bs, 10
            bs, s, _ = shape.shape
            flag_stage = True
            pose = pose.reshape(bs * s, 24, 3, 3)
            shape = shape.reshape(bs * s, 10)

        if pose.is_cuda and self.options.ngpu > 1:
            vertices = data_parallel(self.smpl, (pose, shape), range(self.options.ngpu))
        else:
            vertices = self.smpl(pose, shape)

        if flag_stage:
            vertices = vertices.reshape(bs, s, 6890, 3)
        return vertices

    def train_step(self, input_batch):
        """Training step."""
        dtype = torch.float32
        self.TNet.train()

        # Grab data from the batch
        gt_keypoints_2d = input_batch['keypoints']
        gt_keypoints_3d = input_batch['pose_3d']
        has_pose_3d = input_batch['has_pose_3d']

        gt_keypoints_2d_smpl = input_batch['keypoints_smpl']
        gt_keypoints_3d_smpl = input_batch['pose_3d_smpl']
        has_pose_3d_smpl = input_batch['has_pose_3d_smpl']

        gt_pose = input_batch['pose']
        gt_betas = input_batch['betas']
        has_smpl = input_batch['has_smpl']
        images = input_batch['img']
        gender = input_batch['gender']

        batch_size = images.shape[0]
        gt_vertices = images.new_zeros([batch_size, 6890, 3])
        if images.is_cuda and self.options.ngpu > 1:
            with torch.no_grad():
                gt_vertices[gender < 0] = data_parallel(
                    self.smpl, (gt_pose[gender < 0], gt_betas[gender < 0]), range(self.options.ngpu))
                gt_vertices[gender == 0] = data_parallel(
                    self.male_smpl, (gt_pose[gender == 0], gt_betas[gender == 0]), range(self.options.ngpu))
                gt_vertices[gender == 1] = data_parallel(
                    self.female_smpl, (gt_pose[gender == 1], gt_betas[gender == 1]), range(self.options.ngpu))
            pred_para = data_parallel(self.TNet, images, range(self.options.ngpu))
        else:
            with torch.no_grad():
                gt_vertices[gender < 0] = self.smpl(gt_pose[gender < 0], gt_betas[gender < 0])
                gt_vertices[gender == 0] = self.male_smpl(gt_pose[gender == 0], gt_betas[gender == 0])
                gt_vertices[gender == 1] = self.female_smpl(gt_pose[gender == 1], gt_betas[gender == 1])
            pred_para = self.TNet(images)

        if pred_para[-1].dim() == 2:
            pred_para = (p.unsqueeze(1) for p in pred_para)
        pred_pose, pred_shape, pred_camera = pred_para
        bs, s,  _ = pred_shape.shape
        pred_vertices = self.apply_smpl(pred_pose, pred_shape)

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

        loss_total = sum(loss for loss in losses.values())
        # Do backprop
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        # for visualize
        if (self.step_count + 1) % self.options.summary_steps == 0:
            data = {}
            vis_num = min(4, batch_size)
            data['image'] = input_batch['img_orig'][0:vis_num].detach()
            data['gt_vert'] = gt_vertices[0:vis_num].detach()
            data['gt_joint'] = gt_keypoints_2d[0:vis_num].detach()

            data['pred_vert'] = sampled_vertices[0:vis_num, -1].detach()
            data['pred_cam'] = pred_camera[0:vis_num, -1].detach()
            data['pred_joint'] = sampled_joints_2d[0:vis_num, -1].detach()
            self.vis_data = data

        # Pack output arguments to be used for visualization in a list
        out_args = {key: losses[key].detach().item() for key in losses.keys()}
        out_args['total'] = loss_total.detach().item()
        self.loss_item = out_args

        return out_args

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

    def train_summaries(self, batch, epoch):
        """Tensorboard logging."""
        for key in self.loss_item.keys():
            self.summary_writer.add_scalar('loss_' + key, self.loss_item[key], self.step_count)

        gt_keypoints_2d = self.vis_data['gt_joint'].cpu().numpy()
        pred_vertices = self.vis_data['pred_vert']
        pred_keypoints_2d = self.vis_data['pred_joint']
        pred_camera = self.vis_data['pred_cam']
        dtype = pred_camera.dtype
        rend_imgs = []
        vis_size = pred_vertices.shape[0]
        # Do visualization for the first 4 images of the batch
        for i in range(vis_size):
            img = self.vis_data['image'][i].cpu().numpy().transpose(1, 2, 0)
            H, W, C = img.shape

            # Get LSP keypoints from the full list of keypoints
            gt_keypoints_2d_ = gt_keypoints_2d[i, self.to_lsp]
            pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, self.to_lsp]
            vertices = pred_vertices[i].cpu().numpy()
            cam = pred_camera[i].cpu().numpy()
            if self.renderer is not None:
                from utils.renderer import visualize_reconstruction, vis_mesh
                # Visualize reconstruction and detected pose
                rend_img = visualize_reconstruction(img, self.options.img_res, gt_keypoints_2d_, vertices,
                                                    pred_keypoints_2d_, cam, self.renderer)
                rend_img = rend_img.transpose(2, 0, 1)

                if 'gt_vert' in self.vis_data.keys():
                    rend_img2 = vis_mesh(img, self.vis_data['gt_vert'][i].cpu().numpy(), cam, self.renderer, color='blue')
                    rend_img2 = rend_img2.transpose(2, 0, 1)
                    rend_img = np.concatenate((rend_img, rend_img2), axis=2)

            else:
                rend_img = visualize_vert(img, self.options.img_res, gt_keypoints_2d_, vertices,
                                          pred_keypoints_2d_, cam, self.renderer)
                rend_img = rend_img.transpose(2, 0, 1)

            rend_imgs.append(torch.from_numpy(rend_img))
        rend_imgs = make_grid(rend_imgs, nrow=1)
        # Save results in Tensorboard
        self.summary_writer.add_image('imgs', rend_imgs, self.step_count)

        if 'gt_vert' in self.vis_data.keys():
            vert_image = self.vis_data['image'].float().clone()
            vis_res = self.options.img_res

            pred_vertices = self.vis_data['pred_vert'].detach().clone()
            vert_2d = orthographic_projection(pred_vertices, pred_camera)
            index_batch = torch.arange(vert_2d.shape[0]).unsqueeze(1).expand([-1, vert_2d.shape[1]])
            index_batch = index_batch.reshape(-1, 1)
            vert_2d = vert_2d.reshape(-1, 2)
            valid = (vert_2d[:, 0] >= -1) * (vert_2d[:, 0] <= 1) * (vert_2d[:, 1] >= -1) * (vert_2d[:, 1] <= 1)
            vert_2d = vert_2d[valid, :]
            index_batch = index_batch[valid]
            vert_2d = 0.5 * (vert_2d + 1) * (vis_res - 1)
            vert_2d = vert_2d.long().clamp(min=0, max=vis_res - 1)
            vert_image[index_batch[:, 0], :, vert_2d[:, 1], vert_2d[:, 0]] = 0.5

            vert_image_gt = self.vis_data['image'].float().clone()
            gt_vertices = self.vis_data['gt_vert'].detach().clone()
            vert_2d = orthographic_projection(gt_vertices, pred_camera)
            index_batch = torch.arange(vert_2d.shape[0]).unsqueeze(1).expand([-1, vert_2d.shape[1]])
            index_batch = index_batch.reshape(-1, 1)
            vert_2d = vert_2d.reshape(-1, 2)
            valid = (vert_2d[:, 0] >= -1) * (vert_2d[:, 0] <= 1) * (vert_2d[:, 1] >= -1) * (vert_2d[:, 1] <= 1)
            vert_2d = vert_2d[valid, :]
            index_batch = index_batch[valid]
            vert_2d = 0.5 * (vert_2d + 1) * (vis_res - 1)
            vert_2d = vert_2d.long().clamp(min=0, max=vis_res - 1)
            vert_image_gt[index_batch[:, 0], :, vert_2d[:, 1], vert_2d[:, 0]] = 0.5

            vert_image = torch.cat([vert_image_gt, vert_image], dim=-1)
            vert_image = make_grid(vert_image, nrow=1)
            self.summary_writer.add_image('vert', vert_image, self.step_count)

        import matplotlib.pyplot as plt
        plt.imshow(rend_imgs.permute(1, 2, 0).cpu().numpy())
        plt.imshow(vert_image.permute(1, 2, 0).cpu().numpy())
        t=0




    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in range(self.epoch_count, self.options.num_epochs):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)

            # Iterate over all batches in an epoch
            batch_len = len(self.train_ds) // self.options.batch_size
            data_stream = tqdm(train_data_loader, desc='Epoch ' + str(epoch),
                               total=len(self.train_ds) // self.options.batch_size,
                               initial=train_data_loader.checkpoint_batch_idx)
            for step, batch in enumerate(data_stream, train_data_loader.checkpoint_batch_idx):
                if time.time() < self.endtime:

                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    loss_dict = self.train_step(batch)
                    self.step_count += 1

                    tqdm_info = 'Epoch:%d| %d/%d ' % (epoch, step, batch_len)
                    for k, v in loss_dict.items():
                        tqdm_info += ' %s:%.4f' % (k, v)
                    data_stream.set_description(tqdm_info)

                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(step, epoch)

                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0 and self.step_count > 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step + 1,
                                                   self.options.batch_size, train_data_loader.sampler.dataset_perm,
                                                   self.step_count)
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()

                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step,
                                               self.options.batch_size, train_data_loader.sampler.dataset_perm,
                                               self.step_count)
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

            # load a checkpoint only on startup, for the next epochs just iterate over the dataset as usual
            self.checkpoint = None
            # save checkpoint after each 10 epoch
            if (epoch + 1) % 10 == 0:
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0,
                                           self.options.batch_size, None, self.step_count)

        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0,
                                   self.options.batch_size, None, self.step_count, checkpoint_filename='final')
        return
