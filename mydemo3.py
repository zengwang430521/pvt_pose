"""
Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python demo.py --checkpoint=data/models/model_checkpoint_h36m_up3d_extra2d.pt --img=examples/im1010.png --openpose=examples/im1010_openpose.json
```
Example with predefined Bounding Box
```
python demo.py --checkpoint=data/models/model_checkpoint_h36m_up3d_extra2d.pt --img=examples/im1010.png --bbox=examples/im1010_bbox.json
```
Example with cropped and centered image
```
python demo.py --checkpoint=data/models/model_checkpoint_h36m_up3d_extra2d.pt --img=examples/im1010.png
```

Running the previous command will save the results in ```examples/im1010_{gcnn,smpl,gcnn_side,smpl_side}.png```. The files ```im1010_gcnn``` and ```im1010_smpl``` show the overlayed reconstructions of the non-parametric and parametric shapes respectively. We also render side views, saved in ```im1010_gcnn_side.png``` and ```im1010_smpl_side.png```.
"""


'''
Just for images in paper

'''

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
from utils.imutils import crop
from utils.renderer import Renderer
import utils.config as cfg
from collections import namedtuple
from os.path import join, exists
from utils.objfile import read_obj


from models.TMR import build_model
from datasets.datasets import create_dataset, create_val_dataset
from utils.train_options import DDPTrainOptions

import datetime
import json
import random
import time
import numpy as np
import torch
import utils.misc as utils
import utils.samplers as samplers
from torch.utils.data import DataLoader
from train.train_one_epoch import train_one_epoch, evaluate
from pathlib import Path
from train.criterion import MeshLoss2, JointEvaluator, MeshLoss3, MeshLoss4
from models.TMR import build_model
from datasets.datasets import create_dataset, create_val_dataset
from utils.train_options import DDPTrainOptions
from tensorboardX import SummaryWriter
import os
from copy import deepcopy
from tqdm import tqdm




def notback(x):
    IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], device=x.device)
    IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], device=x.device)
    x = x * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
    return x.max().item() >= 1 / 255.0


def no_black_edge(img):
    return notback(img[0, :, 0, 0]) \
            and notback(img[0, :, 0, -1]) \
            and notback(img[0, :, -1, 0]) \
            and notback(img[0, :, -1, -1])


if __name__ == '__main__':

    options = DDPTrainOptions().parse_args()

    # Setup renderer for visualization
    _, faces = read_obj('data/reference_mesh.obj')
    renderer = Renderer(faces=np.array(faces) - 1)

    device = torch.device('cuda')
    model = build_model(options)
    model.to(device)
    model.eval()

    if options.resume_from:
        if os.path.exists(options.resume_from):
            checkpoint = torch.load(options.resume_from, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))
            print('resume finished.')
        else:
            print('NOTICE: ' + options.resume_from + ' not exists!')

    model.eval()
    # evaluator = JointEvaluator(options, device)


    dataset_val = create_val_dataset(options.val_dataset, options)
    options_hr = deepcopy(options)
    options_hr.img_res = 800
    dataset_val_hr = create_val_dataset(options.val_dataset, options_hr)

    # evaluator.smpl = evaluator.male_smpl
    # for idx in [4660, 4670, 4880, 5220, 6950, 10390, 10570, 10850, 17170]:
    # for idx in [53, 63, 66, 71, 77, 102, 128, 137, 140, 183, 214, 217, 218, 219, 243, 254]:
    # for idx in range(len(dataset_val)):
    for idx in [4860, 4950, 6280, 6290, 6550, 7620, 10170, 11690, 11710, 11720, 12050, 12220]:

        if idx % 10 != 0:
            continue
        data_batch = dataset_val[idx]
        img = data_batch['img'].unsqueeze(0).to(device)
        if not no_black_edge(img):
            continue

        with torch.no_grad():
            model.batch_count = idx
            pred_para = model(img)
