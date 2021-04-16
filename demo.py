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


import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
from models.dmr import DMR
from utils.imutils import crop
from utils.renderer import Renderer
import utils.config as cfg
from collections import namedtuple
from os.path import join, exists
from utils.objfile import read_obj

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to pretrained checkpoint')
parser.add_argument('--config', default=None, help='Path to config file containing model architecture etc.')
parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')


def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
    # tmp = cv2.imread(img_file)
    img = cv2.imread(img_file)[:,:,::-1].copy()  # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

if __name__ == '__main__':
    args = parser.parse_args()


    # Load model
    if args.config is None:
        tmp = args.checkpoint.split('/')[:-2]
        tmp.append('config.json')
        args.config = '/' + join(*tmp)

    with open(args.config, 'r') as f:
        options = json.load(f)
        options = namedtuple('options', options.keys())(**options)

    model = DMR(options, args.checkpoint)
    model.eval()

    # Setup renderer for visualization
    _, faces = read_obj('data/reference_mesh.obj')
    renderer = Renderer(faces=np.array(faces) - 1)

    # Preprocess input image and generate predictions
    img, norm_img = process_image(args.img, args.bbox, args.openpose, input_res=cfg.INPUT_RES)
    with torch.no_grad():
        out_dict = model(norm_img.to(model.device))
        pred_vertices = out_dict['pred_vertices']
        pred_camera = out_dict['camera']

    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*cfg.FOCAL_LENGTH/(cfg.INPUT_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()
    img = img.permute(1,2,0).cpu().numpy()

    # Render non-parametric shape
    img_render = renderer.render(pred_vertices,
                               camera_t=camera_translation,
                               img=img, use_bg=True, body_color='pink')

    # Render side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    center = pred_vertices.mean(axis=0)
    center_smpl = pred_vertices.mean(axis=0)
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center

    # Render non-parametric shape
    img_render_side = renderer.render(rot_vertices,
                               camera_t=camera_translation,
                               img=np.ones_like(img), use_bg=True, body_color='pink')
    
    # Render parametric shape
    outfile = args.img.split('.')[0] if args.outfile is None else args.outfile

    # Save reconstructions
    cv2.imwrite(outfile + '_render.png', 255 * img_render[:,:,::-1])
    cv2.imwrite(outfile + '_render_side.png', 255 * img_render_side[:,:,::-1])
