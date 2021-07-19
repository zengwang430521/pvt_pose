'''
Copied from https://github.com/nkolot/SPIN
'''

import os
import sys
import cv2
import glob
import h5py
import json
import numpy as np
import scipy.io as sio
import scipy.misc
# from .read_openpose import read_openpose


def extract_train_frame(dataset_path, train_file, test_file):
    data = np.load(train_file)
    imgname_list = data['imgname']

    data = np.load(test_file)
    imgname_list += data['imgname']

    # training data
    user_list = range(1, 9)
    seq_list = range(1, 3)
    vid_list = list(range(3)) + list(range(4, 9))

    for user_i in user_list:
        for seq_i in seq_list:
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            for j, vid_i in enumerate(vid_list):

                # image folder
                imgs_path = os.path.join(seq_path,
                                         'imageFrames',
                                         'video_' + str(vid_i))

                # extract frames from video file
                # if doesn't exist
                if not os.path.isdir(imgs_path):
                    os.makedirs(imgs_path)

                # video file
                vid_file = os.path.join(seq_path,
                                        'imageSequence',
                                        'video_' + str(vid_i) + '.avi')
                vidcap = cv2.VideoCapture(vid_file)

                # process video
                frame = 0
                while 1:
                    # extract all frames
                    success, image = vidcap.read()
                    if not success:
                        break
                    frame += 1
                    # image name
                    imgname = os.path.join(imgs_path,
                                           'frame_%06d.jpg' % frame)
                    # save image
                    if imgname in imgname_list:
                        cv2.imwrite(imgname, image)


if __name__ == '__main__':
    dataset_path = '/mnt/lustre/zengwang/data/3dhp/3dhp/mpi_inf_3dhp'
    train_file = '../../data/datasets/npz/mpi_inf_3dhp_train.npz'
    test_file = '../../data/datasets/npz/mpi_inf_3dhp_train.npz'
    extract_train_frame(dataset_path, train_file, test_file)



