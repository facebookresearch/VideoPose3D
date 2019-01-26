# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import zipfile
import tarfile
import numpy as np
import h5py
from glob import glob
from shutil import rmtree

import sys
sys.path.append('../')
from common.h36m_dataset import Human36mDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates
from common.utils import wrap

output_filename_pt = 'data_2d_h36m_sh_pt_mpii'
output_filename_ft = 'data_2d_h36m_sh_ft_h36m'
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
cam_map = {
    '54138969': 0,
    '55011271': 1,
    '58860488': 2,
    '60457274': 3,
}

metadata = {
    'num_joints': 16,
    'keypoints_symmetry': [
        [3, 4, 5, 13, 14, 15],
        [0, 1, 2, 10, 11, 12],
    ]
}

def process_subject(subject, file_list, output):
    if subject == 'S11':
        assert len(file_list) == 119, "Expected 119 files for subject " + subject + ", got " + str(len(file_list))
    else:
        assert len(file_list) == 120, "Expected 120 files for subject " + subject + ", got " + str(len(file_list))
        
    for f in file_list:
        action, cam = os.path.splitext(os.path.basename(f))[0].replace('_', ' ').split('.')
        
        if subject == 'S11' and action == 'Directions':
            continue # Discard corrupted video
        
        if action not in output[subject]:
            output[subject][action] = [None, None, None, None]
        
        with h5py.File(f) as hf:
            positions = hf['poses'].value
            output[subject][action][cam_map[cam]] = positions.astype('float32')    

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
        
    parser = argparse.ArgumentParser(description='Human3.6M dataset downloader/converter')
    
    parser.add_argument('-pt', '--pretrained', default='', type=str, metavar='PATH', help='convert pretrained dataset')
    parser.add_argument('-ft', '--fine-tuned', default='', type=str, metavar='PATH', help='convert fine-tuned dataset')
    
    args = parser.parse_args()
        
    if args.pretrained:
        print('Converting pretrained dataset from', args.pretrained)
        print('Extracting...')
        with zipfile.ZipFile(args.pretrained, 'r') as archive:
            archive.extractall('sh_pt')
        
        print('Converting...')
        output = {}
        for subject in subjects:
            output[subject] = {}
            file_list = glob('sh_pt/h36m/' + subject + '/StackedHourglass/*.h5')
            process_subject(subject, file_list, output)
        
        print('Saving...')
        np.savez_compressed(output_filename_pt, positions_2d=output, metadata=metadata)
        
        print('Cleaning up...')
        rmtree('sh_pt')
        
        print('Done.')
        
    if args.fine_tuned:
        print('Converting fine-tuned dataset from', args.fine_tuned)
        print('Extracting...')
        with tarfile.open(args.fine_tuned, 'r:gz') as archive:
            archive.extractall('sh_ft')
        
        print('Converting...')
        output = {}
        for subject in subjects:
            output[subject] = {}
            file_list = glob('sh_ft/' + subject + '/StackedHourglassFineTuned240/*.h5')
            process_subject(subject, file_list, output)
        
        print('Saving...')
        np.savez_compressed(output_filename_ft, positions_2d=output, metadata=metadata)
        
        print('Cleaning up...')
        rmtree('sh_ft')
        
        print('Done.')
