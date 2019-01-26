# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch

import os
import errno

from common.camera import *
from common.model import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.utils import deterministic_random
from common.skeleton import Skeleton

args = parse_args()
print(args)

try:
	# Create checkpoint directory if it does not exist
	os.makedirs(args.checkpoint)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)


print('Loading 2D detections...')

keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = (kps_left, kps_right) #list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()



for subject in keypoints.keys():
	for action in keypoints[subject]:
		for cam_idx, kps in enumerate(keypoints[subject][action]):
			# Normalize camera frame
			kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=960, h=720)
			keypoints[subject][action][cam_idx] = kps

subjects_test = args.subjects_test.split(',')

def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
	out_poses_3d = []
	out_poses_2d = []
	out_camera_params = []
	for subject in subjects:
		for action in keypoints[subject].keys():
			if action_filter is not None:
				found = False
				for a in action_filter:
					if action.startswith(a):
						found = True
						break
				if not found:
					continue

			poses_2d = keypoints[subject][action]
			for i in range(len(poses_2d)):  # Iterate across cameras
				out_poses_2d.append(poses_2d[i])

	if len(out_camera_params) == 0:
		out_camera_params = None
	if len(out_poses_3d) == 0:
		out_poses_3d = None

	stride = args.downsample
	if subset < 1:
		for i in range(len(out_poses_2d)):
			n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
			start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
			out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
			if out_poses_3d is not None:
				out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
	elif stride > 1:
		# Downsample as requested
		for i in range(len(out_poses_2d)):
			out_poses_2d[i] = out_poses_2d[i][::stride]
			if out_poses_3d is not None:
				out_poses_3d[i] = out_poses_3d[i][::stride]

	return out_camera_params, out_poses_3d, out_poses_2d


action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
	print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

filter_widths = [int(x) for x in args.architecture.split(',')]


model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], poses_valid_2d[0].shape[-2],
						  filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
						  dense=args.dense)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2  # Padding on each side
if args.causal:
	print('INFO: Using causal convolutions')
	causal_shift = pad
else:
	causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
	model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
	model_pos = model_pos.cuda()

if args.resume or args.evaluate:
	chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
	print('Loading checkpoint', chk_filename)
	checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
	print('This model was trained for {} epochs'.format(checkpoint['epoch']))
	model_pos.load_state_dict(checkpoint['model_pos'])

test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
									pad=pad, causal_shift=causal_shift, augment=False,
									kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
									joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

# Evaluate

def evaluate(test_generator, action=None, return_predictions=False):
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


if args.render:
	print('Rendering...')

	input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()

	ground_truth = None
	print('INFO: this action is unlabeled. Ground truth will not be rendered.')

	gen = UnchunkedGenerator(None, None, [input_keypoints],
							 pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
							 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

	prediction = evaluate(gen, return_predictions=True)
	rot = np.asarray([ 0.14070565, -0.15007018, -0.7552408, 0.62232804]).astype(np.float32)
	prediction = camera_to_world(prediction, R=rot, t=0)
	# We don't have the trajectory, but at least we can rebase the height
	prediction[:, :, 2] -= np.min(prediction[:, :, 2])

	anim_output = {'Reconstruction': prediction}
	if ground_truth is not None and not args.viz_no_ground_truth:
		anim_output['Ground truth'] = ground_truth

	input_keypoints = image_coordinates(input_keypoints[..., :2], w=960, h=720)

	from common.visualization import render_animation

	human36m_skeleton = Skeleton(parents=[-1,  0,  1,  2,  0, 4,  5,  0,  7,  8,  9,  8, 11, 12, 8, 14, 15],
	   joints_left=[4, 5,6 ,11, 12,13],
	   joints_right=[1,2,3,14,15,16])

	render_animation(input_keypoints, anim_output,
					 human36m_skeleton, 10, args.viz_bitrate, 70, args.viz_output,
					 limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
					 input_video_path=args.viz_video, viewport=(960, 720),
					 input_video_skip=args.viz_skip)
