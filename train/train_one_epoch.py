from typing import Iterable
import torch
import utils.misc as utils
from utils.vis import visualize_vert
from torchvision.utils import make_grid
from models.geometric_layers import orthographic_projection
from utils.pose_utils import reconstruction_error
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dataloader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    options, summary_writer=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = options.summary_steps

    data_iter = iter(dataloader)

    for _ in metric_logger.log_every(range(len(dataloader)), print_freq, header):
        return_vis = False
        if summary_writer is not None:
            summary_writer.iter_num += 1
            if summary_writer.iter_num % options.summary_steps == 0:
                return_vis = True
                
        input_batch = data_iter.next()
        images = input_batch['img'].to(device)
        pred_para = model(images)
        loss_dict, vis_data = criterion(pred_para, input_batch, return_vis)

        losses = sum(loss_dict[k] for k in loss_dict)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced = {f'{k}': v for k, v in loss_dict_reduced.items()}
        losses_reduced = sum(loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if return_vis:
            write_summary(summary_writer, loss_dict, vis_data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def write_summary(summary_writer, loss_dict, vis_data):
    step_count = summary_writer.iter_num
    loss_item = {key: loss_dict[key].detach().item() for key in loss_dict.keys()}
    loss_item['total'] = sum(loss_dict[k] for k in loss_dict)
    """Tensorboard logging."""
    for key in loss_item.keys():
        summary_writer.add_scalar('loss_' + key, loss_item[key], step_count)

    to_lsp = list(range(14))        # LSP indices from full list of keypoints
    gt_keypoints_2d = vis_data['gt_joint'].cpu().numpy()
    pred_vertices = vis_data['pred_vert']
    pred_keypoints_2d = vis_data['pred_joint']
    pred_camera = vis_data['pred_cam']
    rend_imgs = []
    vis_num = pred_vertices.shape[0]

    # skeleton
    for i in range(vis_num):
        img = vis_data['image'][i].cpu().numpy().transpose(1, 2, 0)
        H, W, C = img.shape

        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        # pred_keypoints_2d_ = pred_keypoints_2d.cpu().clamp(0, H).numpy()[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().clamp(-1, 1).numpy()[i, to_lsp]
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        rend_img = visualize_vert(img, H, gt_keypoints_2d_, vertices,
                                  pred_keypoints_2d_, cam, None)
        rend_img = rend_img.transpose(2, 0, 1)

        rend_imgs.append(torch.from_numpy(rend_img))
    rend_imgs = make_grid(rend_imgs, nrow=1)
    summary_writer.add_image('imgs', rend_imgs, step_count)

    # vertices
    if 'gt_vert' in vis_data.keys():
        vert_image = vis_data['image'].float().clone()
        vis_res = H

        pred_vertices = vis_data['pred_vert'].detach().clone()
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

        vert_image_gt = vis_data['image'].float().clone()
        gt_vertices = vis_data['gt_vert'].detach().clone()
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
        summary_writer.add_image('vert', vert_image, step_count)


@torch.no_grad()
def evaluate(model, evaluator, dataloader, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('MPJPE', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
    # metric_logger.add_meter('MPJPE_PA', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
    # metric_logger.add_meter('MPJPE_scale', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
    # metric_logger.add_meter('MPJPE_rot', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
    # metric_logger.add_meter('MPJPE_tran', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

    # metric_logger.add_meter('MPJPE_spin', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))
    # metric_logger.add_meter('MPJPE_PA_spin', utils.SmoothedValue(window_size=1, fmt='{global_avg:.2f}'))

    header = 'Test:'
    # mpjpe = []
    # mpjpe_pa = []
    # mpjpe_scale, mpjpe_rot, mpjpe_tran = [], [], []
    # mpjpe_spin, mpjpe_pa_spin = [], []

    print_freq = 20
    data_iter = iter(dataloader)
    for _ in metric_logger.log_every(range(len(dataloader)), print_freq, header):
        with torch.no_grad():
            input_batch = data_iter.next()
            images = input_batch['img'].to(device)
            pred_para = model(images)
            pred_joints_3d, gt_joints_3d, pred_joints_3d_spin, gt_joints_3d_spin, pred_vertices, gt_vertices = evaluator(pred_para, input_batch)

            # pred_joints_3d = utils.all_gather(pred_joints_3d)
            # pred_joints_3d = [p.to(device) for p in pred_joints_3d]
            # pred_joints_3d = torch.cat(pred_joints_3d, dim=0)
            #
            # gt_joints_3d = utils.all_gather(gt_joints_3d)
            # gt_joints_3d = [g.to(device) for g in gt_joints_3d]
            # gt_joints_3d = torch.cat(gt_joints_3d, dim=0)

            error = torch.sqrt(((pred_joints_3d - gt_joints_3d) ** 2).sum(dim=-1)).mean(dim=-1).detach().cpu().numpy() * 1000
            error_pa = reconstruction_error(pred_joints_3d.cpu().numpy(), gt_joints_3d.cpu().numpy(), reduction=None) * 1000
            # error_scale = reconstruction_error(pred_joints_3d.cpu().numpy(), gt_joints_3d.cpu().numpy(), reduction=None, skip=['rot', 'tran']) * 1000
            # error_rot = reconstruction_error(pred_joints_3d.cpu().numpy(), gt_joints_3d.cpu().numpy(), reduction=None, skip=['scale', 'tran']) * 1000
            # error_tran = reconstruction_error(pred_joints_3d.cpu().numpy(), gt_joints_3d.cpu().numpy(), reduction=None, skip=['scale', 'rot']) * 1000

            # mpjpe.append(error)
            # mpjpe_pa.append(error_pa)
            # mpjpe_scale.append(error_scale)
            # mpjpe_rot.append(error_rot)
            # mpjpe_tran.append(error_tran)


            '''spin eval'''
            # pred_joints_3d_spin = utils.all_gather(pred_joints_3d_spin)
            # pred_joints_3d_spin = [p.to(device) for p in pred_joints_3d_spin]
            # pred_joints_3d_spin = torch.cat(pred_joints_3d_spin, dim=0)

            # gt_joints_3d_spin = utils.all_gather(gt_joints_3d_spin)
            # gt_joints_3d_spin = [g.to(device) for g in gt_joints_3d_spin]
            # gt_joints_3d_spin = torch.cat(gt_joints_3d_spin, dim=0)

            error_spin = torch.sqrt(((pred_joints_3d_spin - gt_joints_3d_spin) ** 2).sum(dim=-1)).mean(
                dim=-1).detach().cpu().numpy() * 1000
            error_pa_spin = reconstruction_error(pred_joints_3d_spin.cpu().numpy(), gt_joints_3d_spin.cpu().numpy(),
                                                 reduction=None) * 1000

            # mpjpe_spin.append(error_spin)
            # mpjpe_pa_spin.append(error_pa_spin)


            # metric_logger.update(MPJPE=float(error.mean()),
            #                      MPJPE_PA=float(error_pa.mean()),
            #                      # MPJPE_scale=float(error_scale.mean()),
            #                      # MPJPE_rot=float(error_rot.mean()),
            #                      # MPJPE_tran=float(error_tran.mean()),
            #                      MPJPE_spin=float(error_spin.mean()),
            #                      MPJPE_PA_spin=float(error_pa_spin.mean()),
            #                      )

            '''
            vertices
            '''
            error_vertices = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).detach().cpu().numpy() * 1000

            batch_size = images.shape[0]
            metric_logger.meters['MPJPE'].update(error.mean().item(), n=batch_size)
            metric_logger.meters['MPJPE_PA'].update(error_pa.mean().item(), n=batch_size)
            metric_logger.meters['MPJPE_spin'].update(error_spin.mean().item(), n=batch_size)
            metric_logger.meters['MPJPE_PA_spin'].update(error_pa_spin.mean().item(), n=batch_size)
            metric_logger.meters['MPVPE'].update(error_vertices.mean().item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    stats = dict(MPJPE=float(metric_logger.MPJPE.global_avg),
                 MPJPE_PA=float(metric_logger.MPJPE_PA.global_avg),
                 MPJPE_spin=float(metric_logger.MPJPE_spin.global_avg),
                 MPJPE_PA_spin=float(metric_logger.MPJPE_PA_spin.global_avg),
                 MPVPE=float(metric_logger.MPVPE.global_avg),
                 )

    # stats = dict(MPJPE=float(np.concatenate(mpjpe, axis=0).mean()),
    #              MPJPE_PA=float(np.concatenate(mpjpe_pa, axis=0).mean()),
    #              # MPJPE_scale=float(np.concatenate(mpjpe_scale, axis=0).mean()),
    #              # MPJPE_rot=float(np.concatenate(mpjpe_rot, axis=0).mean()),
    #              # MPJPE_tran=float(np.concatenate(mpjpe_tran, axis=0).mean()),
    #              MPJPE_spin=float(np.concatenate(mpjpe_spin, axis=0).mean()),
    #              MPJPE_PA_spin=float(np.concatenate(mpjpe_pa_spin, axis=0).mean()),
    #              )

    return stats
