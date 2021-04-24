import torch
import torch.nn as nn
from .resnet import resnet50
from .smpl_head import HMRHead


class HMR(nn.Module):

    def __init__(self, options):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.options = options
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = None
        self.smpl_head = HMRHead(2048)

    def forward(self, images):
        feature = self.backbone(images)
        pred_para = self.smpl_head(feature)
        return pred_para
