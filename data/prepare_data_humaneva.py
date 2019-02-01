# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import zipfile
import numpy as np
import h5py
import re
from glob import glob
from shutil import rmtree
from data_utils import suggest_metadata, suggest_pose_importer

import sys
sys.path.append('../')
from common.utils import wrap
from itertools import groupby

subjects = ['Train/S1', 'Train/S2', 'Train/S3', 'Validate/S1', 'Validate/S2', 'Validate/S3']

cam_map = {
    'C1': 0,
    'C2': 1,
    'C3': 2,
}

# Frame numbers for train/test split
# format: [start_frame, end_frame[ (inclusive, exclusive)
index = {
    'Train/S1': {
        'Walking 1': (590, 1203),
        'Jog 1': (367, 740),
        'ThrowCatch 1': (473, 945),
        'Gestures 1': (395, 801),
        'Box 1': (385, 789),
    },
    'Train/S2': {
        'Walking 1': (438, 876),
        'Jog 1': (398, 795),
        'ThrowCatch 1': (550, 1128),
        'Gestures 1': (500, 901),
        'Box 1': (382, 734),
    },
    'Train/S3': {
        'Walking 1': (448, 939),
        'Jog 1': (401, 842),
        'ThrowCatch 1': (493, 1027),
        'Gestures 1': (533, 1102),
        'Box 1': (512, 1021),
    },
    'Validate/S1': {
        'Walking 1': (5, 590),
        'Jog 1': (5, 367),
        'ThrowCatch 1': (5, 473),
        'Gestures 1': (5, 395),
        'Box 1': (5, 385),
    },
    'Validate/S2': {
        'Walking 1': (5, 438),
        'Jog 1': (5, 398),
        'ThrowCatch 1': (5, 550),
        'Gestures 1': (5, 500),
        'Box 1': (5, 382),
    },
    'Validate/S3': {
        'Walking 1': (5, 448),
        'Jog 1': (5, 401),
        'ThrowCatch 1': (5, 493),
        'Gestures 1': (5, 533),
        'Box 1': (5, 512),
    },
}

# Frames to skip for each video (synchronization)
sync_data = {
    'S1': {
        'Walking 1': (82, 81, 82),
        'Jog 1': (51, 51, 50),
        'ThrowCatch 1': (61, 61, 60),
        'Gestures 1': (45, 45, 44),
        'Box 1': (57, 57, 56),
    },
    'S2': {
        'Walking 1': (115, 115, 114),
        'Jog 1': (100, 100, 99),
        'ThrowCatch 1': (127, 127, 127),
        'Gestures 1': (122, 122, 121),
        'Box 1': (119, 119, 117),
    },
    'S3': {
        'Walking 1': (80, 80, 80),
        'Jog 1': (65, 65, 65),
        'ThrowCatch 1': (79, 79, 79),
        'Gestures 1': (83, 83, 82),
        'Box 1': (1, 1, 1),
    },
    'S4': {}
}

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
        
    parser = argparse.ArgumentParser(description='HumanEva dataset converter')
    
    parser.add_argument('-p', '--path', default='', type=str, metavar='PATH', help='path to the processed HumanEva dataset')
    parser.add_argument('--convert-3d', action='store_true', help='convert 3D mocap data')
    parser.add_argument('--convert-2d', default='', type=str, metavar='PATH', help='convert user-supplied 2D detections')
    parser.add_argument('-o', '--output', default='', type=str, metavar='PATH', help='output suffix for 2D detections (e.g. detectron_pt_coco)')
    
    args = parser.parse_args()
        
    if not args.convert_2d and not args.convert_3d:
        print('Please specify one conversion mode')
        exit(0)
        
 
    if args.path:
        print('Parsing HumanEva dataset from', args.path)
        output = {}
        output_2d = {}
        frame_mapping = {}
        
        from scipy.io import loadmat
        
        num_joints = None
        
        for subject in subjects:
            output[subject] = {}
            output_2d[subject] = {}
            split, subject_name = subject.split('/')
            if subject_name not in frame_mapping:
                frame_mapping[subject_name] = {}
            
            file_list = glob(args.path + '/' + subject + '/*.mat')
            for f in file_list:
                action = os.path.splitext(os.path.basename(f))[0]
                    
                # Use consistent naming convention
                canonical_name = action.replace('_', ' ')
                
                hf = loadmat(f)
                positions = hf['poses_3d']
                positions_2d = hf['poses_2d'].transpose(1, 0, 2, 3) # Ground-truth 2D poses
                assert positions.shape[0] == positions_2d.shape[0] and positions.shape[1] == positions_2d.shape[2]
                assert num_joints is None or num_joints == positions.shape[1], "Joint number inconsistency among files"
                num_joints = positions.shape[1]
                
                # Sanity check for the sequence length
                assert positions.shape[0] == index[subject][canonical_name][1] - index[subject][canonical_name][0]
                
                # Split corrupted motion capture streams into contiguous chunks
                # e.g. 012XX567X9 is split into "012", "567", and "9".
                all_chunks = [list(v) for k, v in groupby(positions, lambda x: np.isfinite(x).all())]
                all_chunks_2d = [list(v) for k, v in groupby(positions_2d, lambda x: np.isfinite(x).all())]
                assert len(all_chunks) == len(all_chunks_2d)
                current_index = index[subject][canonical_name][0]
                chunk_indices = []
                for i, chunk in enumerate(all_chunks):
                    next_index = current_index + len(chunk)
                    name = canonical_name + ' chunk' + str(i)
                    if np.isfinite(chunk).all():
                        output[subject][name] = np.array(chunk, dtype='float32') / 1000
                        output_2d[subject][name] = list(np.array(all_chunks_2d[i], dtype='float32').transpose(1, 0, 2, 3))
                    chunk_indices.append((current_index, next_index, np.isfinite(chunk).all(), split, name))
                    current_index = next_index
                assert current_index == index[subject][canonical_name][1]
                if canonical_name not in frame_mapping[subject_name]:
                    frame_mapping[subject_name][canonical_name] = []
                frame_mapping[subject_name][canonical_name] += chunk_indices
        
        metadata = suggest_metadata('humaneva' + str(num_joints))
        output_filename = 'data_3d_' + metadata['layout_name']
        output_prefix_2d = 'data_2d_' + metadata['layout_name'] + '_'
        
        if args.convert_3d:
            print('Saving...')
            np.savez_compressed(output_filename, positions_3d=output)
            np.savez_compressed(output_prefix_2d + 'gt', positions_2d=output_2d, metadata=metadata)
            print('Done.')
        
    else:
        print('Please specify the dataset source')
        exit(0)
        
    if args.convert_2d:
        if not args.output:
            print('Please specify an output suffix (e.g. detectron_pt_coco)')
            exit(0)
            
        import_func = suggest_pose_importer(args.output)
        metadata = suggest_metadata(args.output)
            
        print('Parsing 2D detections from', args.convert_2d)
        
        output = {}
        file_list = glob(args.convert_2d + '/S*/*.avi.npz')
        for f in file_list:
            path, fname = os.path.split(f)
            subject = os.path.basename(path)
            assert subject.startswith('S'), subject + ' does not look like a subject directory'
            
            m = re.search('(.*) \\((.*)\\)', fname.replace('_', ' '))
            action = m.group(1)
            camera = m.group(2)
            camera_idx = cam_map[camera]
            
            keypoints = import_func(f)
            assert keypoints.shape[1] == metadata['num_joints']
            
            if action in sync_data[subject]:
                sync_offset = sync_data[subject][action][camera_idx] - 1
            else:
                sync_offset = 0

            if subject in frame_mapping and action in frame_mapping[subject]:
                chunks = frame_mapping[subject][action]
                for (start_idx, end_idx, labeled, split, name) in chunks:
                    canonical_subject = split + '/' + subject
                    if not labeled:
                        canonical_subject = 'Unlabeled/' + canonical_subject
                    if canonical_subject not in output:
                        output[canonical_subject] = {}
                    kps = keypoints[start_idx+sync_offset:end_idx+sync_offset]
                    assert len(kps) == end_idx - start_idx, "Got len {}, expected {}".format(len(kps), end_idx - start_idx)
                    
                    if name not in output[canonical_subject]:
                        output[canonical_subject][name] = [None, None, None]
                    
                    output[canonical_subject][name][camera_idx] = kps.astype('float32')
            else:
                canonical_subject = 'Unlabeled/' + subject
                if canonical_subject not in output:
                    output[canonical_subject] = {}
                if action not in output[canonical_subject]:
                        output[canonical_subject][action] = [None, None, None]
                output[canonical_subject][action][camera_idx] = keypoints.astype('float32')
                
        print('Saving...')
        np.savez_compressed(output_prefix_2d + args.output, positions_2d=output, metadata=metadata)
        print('Done.')