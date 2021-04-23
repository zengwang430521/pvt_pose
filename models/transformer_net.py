import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from .resnet import resnet50backbone
from .transformer import Transformer
from .layers import ConvBottleNeck, HgNet
from .position_encoding import build_position_encoding
from .smpl_head import SimpleSMPLHead


class TNet(nn.Module):

    def __init__(self, options):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.options = options

        self.backbone = ResBackbone(out_lv=self.options.out_lv).to(self.device)
        args = self.options
        self.transformer = Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=args.intermediate_loss,
        )

        self.num_queries = 24+1+1
        channel_list = [3, 64, 256, 512, 1024, 2048]
        in_channels = channel_list[self.options.out_lv]
        hidden_dim = self.transformer.d_model
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.pos_embed = build_position_encoding(self.options)

        self.smpl_head = SimpleSMPLHead(options, hidden_dim)

    def forward(self, images):
        feature, global_vector = self.backbone(images)
        pos = self.pos_embed(feature)
        mask = None
        hs = self.transformer(self.input_proj(feature), mask, self.query_embed.weight, pos)[0]

        hs = hs.permute([1, 0, 2, 3])   # bs * s * 26 * c
        pred_para = self.smpl_head(hs)
        return pred_para


    #     out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
    #     if self.aux_loss:
    #         out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
    #     return out
    #
    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_boxes': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

# DPNet returns densepose result
class ResBackbone(nn.Module):

    def __init__(self, out_lv=2, norm_type='BN'):
        super().__init__()
        nl_layer = nn.ReLU(inplace=True)
        self.out_lv = out_lv
        # image encoder
        self.resnet = resnet50backbone(pretrained=True)

        dp_layers = []
        #              [224, 112, 56, 28,    14,    7]
        channel_list = [3,   64, 256, 512, 1024, 2048]
        for i in range(out_lv, 5):
            in_channels = channel_list[i + 1]
            out_channels = channel_list[i]

            dp_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                    ConvBottleNeck(in_channels=in_channels, out_channels=out_channels, nl_layer=nl_layer, norm_type=norm_type)
                )
            )
        self.dp_layers = nn.ModuleList(dp_layers)

    def forward(self, image):
        codes, features = self.resnet(image)
        # output feature
        dp_feature = features[-1]
        for i in range(len(self.dp_layers) - 1, -1, -1):
            dp_feature = self.dp_layers[i](dp_feature)
            dp_feature = dp_feature + features[i - 1 + len(features) - len(self.dp_layers)]
        return dp_feature, codes