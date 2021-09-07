import json
import numpy as np
import torch
import os
from train.criterion import rotation_matrix_to_angle_axis
from tqdm import tqdm

# eft_lspet = '/home/wzeng/mydata/eft_fit/LSPet_ver01.json'
# lspet_npz = '/home/wzeng/mydata/DecoMR/hr-lspet_train.npz'
# lspet_out = lspet_npz[:-4] + '_eft.npz'
#
# with open(eft_lspet, 'r') as f:
#     eft_data = json.load(f)
#
# data = np.load(lspet_npz)
#
# imgnames = data['imgname']
# new_data_dict = {key: data[key] for key in data.keys()}
# del new_data_dict['fit_errors']
# new_data_dict['pose'] = np.zeros([len(imgnames), 72])
# new_data_dict['shape'] = np.zeros([len(imgnames), 10])
# new_data_dict['has_smpl'] = np.zeros(len(imgnames))
#
# for item in tqdm(eft_data['data']):
#     pose_eft = np.array(item['parm_pose'])
#     # Convert rotation matrices to axis-angle
#     rotmat = torch.tensor(pose_eft).float()
#     rotmat_hom = torch.cat(
#         [rotmat.view(-1, 3, 3),
#          torch.tensor([0, 0, 1], dtype=torch.float32, device=rotmat.device).view(1, 3, 1).expand(24, -1, -1)],
#         dim=-1)
#     pose_eft = rotation_matrix_to_angle_axis(rotmat_hom).contiguous().view(-1)
#
#     # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
#     pose_eft[torch.isnan(pose_eft)] = 0.0
#     pose_eft = pose_eft.numpy()
#
#     beta_eft = np.array(item['parm_shape'])
#     img_eft = item['imageName']
#
#     idx = np.argwhere(imgnames == img_eft)
#     if len(idx) == 1:
#         idx = idx[0][0]
#         img = imgnames[idx]
#         assert img_eft == img
#         new_data_dict['pose'][idx] = pose_eft
#         new_data_dict['shape'][idx] = beta_eft
#         new_data_dict['has_smpl'][idx] = 1
#         print(idx)
#     else:
#         print('new img !!!')
#         part_eft = np.array(item['gt_keypoint_2d'])
#         part_eft = part_eft[25:, :]     # remove 25 openpose joints
#         scale_eft = np.array('bbox_scale')
#         center_eft = np.array('bbox_center')
#
#         new_data_dict['imgname'] = np.concatenate((new_data_dict['imgname'], np.array(img_eft)), axis=0)
#         new_data_dict['center'] = np.concatenate((new_data_dict['center'], center_eft[None, :]), axis=0)
#         new_data_dict['scale'] = np.concatenate((new_data_dict['scale'], scale_eft[None, :]), axis=0)
#         new_data_dict['part'] = np.concatenate((new_data_dict['part'], scale_eft[None, :]), axis=0)
#         new_data_dict['iuvnames'] = np.concatenate((new_data_dict['part'], np.array('')), axis=0)
#         new_data_dict['pose'] = np.concatenate((new_data_dict['pose'], pose_eft[None, :]), axis=0)
#         new_data_dict['shape'] = np.concatenate((new_data_dict['shape'], beta_eft[None, :]), axis=0)
#         new_data_dict['has_smpl'] = np.concatenate((new_data_dict['has_smpl'], np.ones(1)), axis=0)
#
#
# np.savez(lspet_out, **new_data_dict)


# eft_mpii = '/home/wzeng/mydata/eft_fit/MPII_ver01.json'
# mpii_npz = '/home/wzeng/mydata/DecoMR/mpii_train.npz'
# mpii_out = mpii_npz[:-4] + '_eft.npz'
#
# with open(eft_mpii, 'r') as f:
#     eft_data = json.load(f)
#
# data = np.load(mpii_npz)
#
# imgnames = data['imgname']
# new_data_dict = {key: data[key] for key in data.keys()}
# del new_data_dict['fit_errors']
# new_data_dict['pose'] = np.zeros([len(imgnames), 72])
# new_data_dict['shape'] = np.zeros([len(imgnames), 10])
# new_data_dict['has_smpl'] = np.zeros(len(imgnames))
#
# for item in tqdm(eft_data['data']):
#     pose_eft = np.array(item['parm_pose'])
#     # Convert rotation matrices to axis-angle
#     rotmat = torch.tensor(pose_eft).float()
#     rotmat_hom = torch.cat(
#         [rotmat.view(-1, 3, 3),
#          torch.tensor([0, 0, 1], dtype=torch.float32, device=rotmat.device).view(1, 3, 1).expand(24, -1, -1)],
#         dim=-1)
#     pose_eft = rotation_matrix_to_angle_axis(rotmat_hom).contiguous().view(-1)
#
#     # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
#     pose_eft[torch.isnan(pose_eft)] = 0.0
#     pose_eft = pose_eft.numpy()
#
#     beta_eft = np.array(item['parm_shape'])
#     img_eft = item['imageName']
#     img_eft = 'images/' + img_eft
#
#     idx = np.argwhere(imgnames == img_eft)
#     if len(idx) == 1:
#         idx = idx[0][0]
#         img = imgnames[idx]
#         assert img_eft == img
#         new_data_dict['pose'][idx] = pose_eft
#         new_data_dict['shape'][idx] = beta_eft
#         new_data_dict['has_smpl'][idx] = 1
#         print(idx)
#     elif len(idx) > 1:
#         print('multi bbox in 1 image')
#         center_eft = np.array(item['bbox_center'])
#         centers = new_data_dict['center'][idx, :][:, 0, :]
#         tmp = centers - center_eft[None, :]
#         tmp = np.abs(tmp).sum(axis=1)
#         id_t = np.argmin(tmp)
#         idx = idx[id_t, 0]
#
#         img = imgnames[idx]
#         center = new_data_dict['center'][idx]
#         assert img_eft == img
#         assert np.allclose(center, center_eft)
#         new_data_dict['pose'][idx] = pose_eft
#         new_data_dict['shape'][idx] = beta_eft
#         new_data_dict['has_smpl'][idx] = 1
#         print(idx)
#
#     else:
#         print('new img !!!')
#         part_eft = np.array(item['gt_keypoint_2d'])
#         part_eft = part_eft[25:, :]     # remove 25 openpose joints
#         scale_eft = np.array(item['bbox_scale'])
#         center_eft = np.array(item['bbox_center'])
#
#         new_data_dict['imgname'] = np.concatenate((new_data_dict['imgname'], np.array(img_eft)), axis=0)
#         new_data_dict['center'] = np.concatenate((new_data_dict['center'], center_eft[None, :]), axis=0)
#         new_data_dict['scale'] = np.concatenate((new_data_dict['scale'], scale_eft[None, :]), axis=0)
#         new_data_dict['part'] = np.concatenate((new_data_dict['part'], scale_eft[None, :]), axis=0)
#         new_data_dict['iuvnames'] = np.concatenate((new_data_dict['part'], np.array('')), axis=0)
#         new_data_dict['pose'] = np.concatenate((new_data_dict['pose'], pose_eft[None, :]), axis=0)
#         new_data_dict['shape'] = np.concatenate((new_data_dict['shape'], beta_eft[None, :]), axis=0)
#         new_data_dict['has_smpl'] = np.concatenate((new_data_dict['has_smpl'], np.ones(1)), axis=0)
#
#
# np.savez(mpii_out, **new_data_dict)



# eft_coco = '/home/wzeng/mydata/eft_fit/COCO2014-Part-ver01.json'
# coco_npz = '/home/wzeng/mydata/DecoMR/coco_2014_train.npz'
# coco_out = coco_npz[:-4] + '_eft.npz'
#
# with open(eft_coco, 'r') as f:
#     eft_data = json.load(f)
#
# data = np.load(coco_npz)
#
# imgnames = data['imgname']
# new_data_dict = {key: data[key] for key in data.keys()}
# del new_data_dict['fit_errors']
# new_data_dict['pose'] = np.zeros([len(imgnames), 72])
# new_data_dict['shape'] = np.zeros([len(imgnames), 10])
# new_data_dict['has_smpl'] = np.zeros(len(imgnames))
#
#
# for item in tqdm(eft_data['data']):
#     pose_eft = np.array(item['parm_pose'])
#     # Convert rotation matrices to axis-angle
#     rotmat = torch.tensor(pose_eft).float()
#     rotmat_hom = torch.cat(
#         [rotmat.view(-1, 3, 3),
#          torch.tensor([0, 0, 1], dtype=torch.float32, device=rotmat.device).view(1, 3, 1).expand(24, -1, -1)],
#         dim=-1)
#     pose_eft = rotation_matrix_to_angle_axis(rotmat_hom).contiguous().view(-1)
#
#     # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
#     pose_eft[torch.isnan(pose_eft)] = 0.0
#     pose_eft = pose_eft.numpy()
#
#     beta_eft = np.array(item['parm_shape'])
#     img_eft = item['imageName']
#     img_eft = 'train2014/' + img_eft
#
#     idx = np.argwhere(imgnames == img_eft)
#     if len(idx) == 1:
#         idx = idx[0][0]
#         img = imgnames[idx]
#         assert img_eft == img
#         annot_id = new_data_dict['id'][idx]
#         annot_id_eft = item['annotId']
#         assert annot_id == annot_id_eft
#
#         new_data_dict['pose'][idx] = pose_eft
#         new_data_dict['shape'][idx] = beta_eft
#         new_data_dict['has_smpl'][idx] = 1
#         print(idx)
#     elif len(idx) > 1:
#         print('multi bbox in 1 image')
#         center_eft = np.array(item['bbox_center'])
#         centers = new_data_dict['center'][idx, :][:, 0, :]
#         tmp = centers - center_eft[None, :]
#         tmp = np.abs(tmp).sum(axis=1)
#         id_t = np.argmin(tmp)
#         idx = idx[id_t, 0]
#
#         img = imgnames[idx]
#         center = new_data_dict['center'][idx]
#         assert img_eft == img
#         annot_id = new_data_dict['id'][idx]
#         annot_id_eft = item['annotId']
#         assert annot_id == annot_id_eft
#         assert np.allclose(center, center_eft)
#         new_data_dict['pose'][idx] = pose_eft
#         new_data_dict['shape'][idx] = beta_eft
#         new_data_dict['has_smpl'][idx] = 1
#         print(idx)
#
#     else:
#         exit(1)
#         print('new img !!!')
#         part_eft = np.array(item['gt_keypoint_2d'])
#         part_eft = part_eft[25:, :]     # remove 25 openpose joints
#         scale_eft = np.array(item['bbox_scale'])
#         center_eft = np.array(item['bbox_center'])
#
#         new_data_dict['imgname'] = np.concatenate((new_data_dict['imgname'], np.array(img_eft)), axis=0)
#         new_data_dict['center'] = np.concatenate((new_data_dict['center'], center_eft[None, :]), axis=0)
#         new_data_dict['scale'] = np.concatenate((new_data_dict['scale'], scale_eft[None, :]), axis=0)
#         new_data_dict['part'] = np.concatenate((new_data_dict['part'], scale_eft[None, :]), axis=0)
#         new_data_dict['iuvnames'] = np.concatenate((new_data_dict['part'], np.array('')), axis=0)
#         new_data_dict['pose'] = np.concatenate((new_data_dict['pose'], pose_eft[None, :]), axis=0)
#         new_data_dict['shape'] = np.concatenate((new_data_dict['shape'], beta_eft[None, :]), axis=0)
#         new_data_dict['has_smpl'] = np.concatenate((new_data_dict['has_smpl'], np.ones(1)), axis=0)
#
#
# np.savez(coco_out, **new_data_dict)
#




# eft_coco = '/home/wzeng/mydata/eft_fit/COCO2014-Part-ver01.json'
# coco_npz = '/home/wzeng/mydata/DecoMR/coco_2014_train.npz'
# coco_out = coco_npz[:-4] + '_eft.npz'

eft_coco = '/home/wzeng/mydata/eft_fit/COCO2014-All-ver01.json'
coco_npz = '/home/wzeng/mydata/DecoMR/coco_2014_train.npz'
coco_out = coco_npz[:-4] + '_eft_all.npz'

with open(eft_coco, 'r') as f:
    eft_data = json.load(f)

data = np.load(coco_npz)

imgnames = data['imgname']
new_data_dict = {key: data[key] for key in data.keys()}
del new_data_dict['fit_errors']
new_data_dict['pose'] = np.zeros([len(imgnames), 72])
new_data_dict['shape'] = np.zeros([len(imgnames), 10])
new_data_dict['has_smpl'] = np.zeros(len(imgnames))


out_num = 0
for item in tqdm(eft_data['data']):
    pose_eft = np.array(item['parm_pose'])
    # Convert rotation matrices to axis-angle
    rotmat = torch.tensor(pose_eft).float()
    rotmat_hom = torch.cat(
        [rotmat.view(-1, 3, 3),
         torch.tensor([0, 0, 1], dtype=torch.float32, device=rotmat.device).view(1, 3, 1).expand(24, -1, -1)],
        dim=-1)
    pose_eft = rotation_matrix_to_angle_axis(rotmat_hom).contiguous().view(-1)

    # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
    pose_eft[torch.isnan(pose_eft)] = 0.0
    pose_eft = pose_eft.numpy()

    beta_eft = np.array(item['parm_shape'])
    img_eft = item['imageName']
    img_eft = 'train2014/' + img_eft
    annot_id_eft = item['annotId']
    idx = np.argwhere(new_data_dict['id'] == annot_id_eft)

    if len(idx) == 1:
        idx = idx[0][0]
        img = imgnames[idx]
        annot_id = new_data_dict['id'][idx]
        center = new_data_dict['center'][idx]
        scale = new_data_dict['scale'][idx]
        part = new_data_dict['part'][idx]

        scale_eft = item['bbox_scale']
        center_eft = np.array(item['bbox_center'])
        part_eft = np.array(item['gt_keypoint_2d'])
        part_eft = part_eft[25:, :]     # remove 25 openpose joints

        assert img_eft == img
        assert annot_id == annot_id_eft
        assert scale == scale_eft
        assert np.allclose(center, center_eft)
        assert np.allclose(part, part_eft)

        new_data_dict['pose'][idx] = pose_eft
        new_data_dict['shape'][idx] = beta_eft
        new_data_dict['has_smpl'][idx] = 1
        # print(idx)
    elif len(idx) > 1:
        print('error: the same id occurs twice!')
        exit(1)
    else:
        out_num += 1
        # print('new img !!!')
        part_eft = np.array(item['gt_keypoint_2d'])
        part_eft = part_eft[25:, :]     # remove 25 openpose joints
        scale_eft = np.array([item['bbox_scale']])
        center_eft = np.array(item['bbox_center'])

        new_data_dict['imgname'] = np.concatenate((new_data_dict['imgname'], np.array([img_eft])), axis=0)
        new_data_dict['center'] = np.concatenate((new_data_dict['center'], center_eft[None, :]), axis=0)
        new_data_dict['scale'] = np.concatenate((new_data_dict['scale'], scale_eft), axis=0)
        new_data_dict['part'] = np.concatenate((new_data_dict['part'], part_eft[None, :]), axis=0)
        new_data_dict['iuv_names'] = np.concatenate((new_data_dict['iuv_names'], np.array([''])), axis=0)
        new_data_dict['pose'] = np.concatenate((new_data_dict['pose'], pose_eft[None, :]), axis=0)
        new_data_dict['shape'] = np.concatenate((new_data_dict['shape'], beta_eft[None, :]), axis=0)
        new_data_dict['has_smpl'] = np.concatenate((new_data_dict['has_smpl'], np.ones(1)), axis=0)
        new_data_dict['id'] = np.concatenate((new_data_dict['id'], np.array([annot_id_eft])), axis=0)

        print(out_num)


np.savez(coco_out, **new_data_dict)

print(out_num)
print(out_num)




'''

COCO NPZ: 28344 

COCO PART: 
    28062 in npz, 0 out npz


COCO all:
    74834 = 27923 in npz + 46911 out npz

'''