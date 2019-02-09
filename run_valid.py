# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

def loadtxt(path):
    with open(path, "r") as fptr:
        lines = fptr.readlines()[1:25]
        x = np.array([float(l.split(" ")[1]) for l in lines])
        y = np.array([float(l.split(" ")[2]) for l in lines])
        z = np.array([float(l.split(" ")[3]) for l in lines])
        selection = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 17, 19, 20]
        x = x[selection]
        y = y[selection]
        z = z[selection]
    return x, y, z

def loadKinect(dir = "/home/narvis/Dev/data_kinect/pose_data/"):
    directory = os.fsencode(dir)
    allFiles = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            allFiles.append(filename)

    lsorted = sorted(allFiles, key=lambda x: int(x.split('_')[1]))

    poses_kinect = []
    for i in lsorted:
        path = os.path.join(dir, i)
        x, y, z = loadtxt(path)
        poses_kinect.append(np.transpose([x, y, z]))

    poses_kinect = np.array(poses_kinect)
    return poses_kinect


print('Loading dataset...')
dataset_path = 'data/data_3d_h36m.npz'
from common.h36m_dataset import Human36mDataset
dataset = Human36mDataset(dataset_path)


print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        positions_3d = []
        for cam in anim['cameras']:
            pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
            pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
            positions_3d.append(pos_3d)
        anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
# Stuff we need to define
path2dKeypoints = 'data/data_2d_detections.npz'
width_of = 1920
height_of = 1080
manual_fps = 29
#########

keypoints = np.load(path2dKeypoints)
keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
kps_left = [1, 3, 5, 7, 9, 11, 13, 15]
kps_right = [2, 4, 6, 8, 10, 12, 14, 16]
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()
# keypoints['S1']['Directions 1'][0].shape == 1384,17,2 should be 3 though
# when loading the file provided py pavllo as example the shape is fine (1384,17,3) and the joints left = <class 'list'>: [4, 5, 6, 11, 12, 13] and right = <class 'list'>: [1, 2, 3, 14, 15, 16]
subject = 'S1'
action = 'Directions 1'


for cam_idx, kps in enumerate(keypoints[subject][action]):
    # Normalize camera frame
    cam = dataset.cameras()[subject][cam_idx]
    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=width_of, h=height_of)
    keypoints[subject][action][cam_idx] = kps

subjects_train = ['S1','S5','S6','S7','S8']
subjects_semi = []
subjects_test = ['S9', 'S11']


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    subject = 'S1'
    action = 'Directions 1'

                
    poses_2d = keypoints[subject][action]
    for i in range(len(poses_2d)): # Iterate across cameras
        out_poses_2d.append(poses_2d[i])


    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    
    stride = 1
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    

    return out_camera_params, out_poses_3d, out_poses_2d

action_filter = None
    
cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

filter_widths = list([3,3,3,3,3])

model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 17,
                            filter_widths=filter_widths, causal=False, dropout=0.25, channels=1024,
                            dense=False)


receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side

causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
#    model_pos_train = model_pos_train.cuda()
    

chk_filename = os.path.join('checkpoint', 'd-pt-243.bin')
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
print('This model was trained for {} epochs'.format(checkpoint['epoch']))
#    model_pos_train.load_state_dict(checkpoint['model_pos'])
model_pos.load_state_dict(checkpoint['model_pos'])
    
test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))


def evaluate(test_generator, action=None, return_predictions=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos,
                                                                                         inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----' + action + '----')
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev



print('Rendering...')
my_action = 'Directions 1'
#input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
input_keypoints = keypoints["S1"]["Directions 1"][0].copy()

ground_truth = None

gen = UnchunkedGenerator(None, None, [input_keypoints],
                         pad=pad, causal_shift=causal_shift, augment=True,
                         kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
prediction = evaluate(gen, return_predictions=True)



# Invert camera transformation
cam = dataset.cameras()['S1'][0]

for subject in dataset.cameras():
    if 'orientation' in dataset.cameras()[subject][0]:
        rot = dataset.cameras()[subject][0]['orientation']
        break
prediction = camera_to_world(prediction, R=rot, t=0)
# We don't have the trajectory, but at least we can rebase the height
prediction[:, :, 2] -= np.min(prediction[:, :, 2])

predictionsKinect = loadKinect(dir = "/home/narvis/study/TobiKinectRawDataTest/P1A3")


anim_output = {'Video3D': prediction}


input_keypoints = image_coordinates(input_keypoints[..., :2], w=width_of, h=height_of)



#np.savez('out_3D_vp3d', anim_output['Video3D'])

from common.visualization import render_animation_valid
render_animation_valid(predictionsKinect, input_keypoints, anim_output,
                 dataset.skeleton(), manual_fps, 3000, cam['azimuth'], "outputs/tesst.mp4",
                 limit=-1, downsample=1, size=5,
                 input_image_folder='/home/narvis/study/TobiKinectRawDataTest/P1A3', viewport=(width_of, height_of),
                 input_video_skip=0)


