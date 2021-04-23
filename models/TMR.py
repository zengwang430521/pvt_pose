import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from train.transformer_trainer import model_dict
from .smpl import SMPL


def build_model(options):
    model_class = model_dict[options.model]
    if 'pvt' in options.model:
        model = model_class()
    else:
        model = model_class(options)
    return model


class TMR(nn.Module):
    def __init__(self, options, pretrained_checkpoint=None, ngpu=1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.options = options
        self.ngpu = ngpu
        self.to_lsp = list(range(14))

        self.TNet = build_model(options).to(self.device)

        self.smpl = SMPL().to(self.device)

        if pretrained_checkpoint is not None:
            checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')
            try:
                self.TNet.load_state_dict(checkpoint['TNet'])
                print('Checkpoint loaded')
            except KeyError:
                print('loading failed')

    def forward(self, images, detach=True):
        out_dict = {}

        if detach:
            with torch.no_grad():
                if self.ngpu > 1 and images.shape[0] % self.ngpu == 0:
                    pred_para = data_parallel(self.TNet, images, range(self.options.ngpu))
                    if pred_para[-1].dim() == 3:
                        pred_para = (p[:, -1] for p in pred_para)
                    pred_pose, pred_shape, pred_camera = pred_para
                    pred_vertices = data_parallel(self.smpl, (pred_pose, pred_shape), range(self.options.ngpu))
                else:
                    pred_para = self.TNet(images)
                    if pred_para[-1].dim() == 3:
                        pred_para = (p[:, -1] for p in pred_para)
                    pred_pose, pred_shape, pred_camera = pred_para
                    pred_vertices = self.smpl(pred_pose, pred_shape)
        else:
            if self.ngpu > 1 and images.shape[0] % self.ngpu == 0:
                pred_para = data_parallel(self.TNet, images, range(self.options.ngpu))
                if pred_para[-1].dim() == 3:
                    pred_para = (p[:, -1] for p in pred_para)
                pred_pose, pred_shape, pred_camera = pred_para
                pred_vertices = data_parallel(self.smpl, (pred_pose, pred_shape), range(self.options.ngpu))
            else:
                pred_para = self.TNet(images)
                if pred_para[-1].dim() == 3:
                    pred_para = (p[:, -1] for p in pred_para)
                pred_pose, pred_shape, pred_camera = pred_para
                pred_vertices = self.smpl(pred_pose, pred_shape)

        out_dict['pred_vertices'] = pred_vertices
        out_dict['camera'] = pred_camera
        # out_dict['uv_map'] = pred_uv_map
        # out_dict['dp_map'] = pred_dp
        return out_dict
