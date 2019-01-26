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

output_prefix_2d = 'data_2d_h36m_'
cam_map = {
    '54138969': 0,
    '55011271': 1,
    '58860488': 2,
    '60457274': 3,
}

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
        
    parser = argparse.ArgumentParser(description='Human3.6M dataset converter')
    
    parser.add_argument('-i', '--input', default='', type=str, metavar='PATH', help='input path to 2D detections')
    parser.add_argument('-o', '--output', default='', type=str, metavar='PATH', help='output suffix for 2D detections (e.g. detectron_pt_coco)')
    
    args = parser.parse_args()
        
    if not args.input:
        print('Please specify the input directory')
        exit(0)
        
    if not args.output:
        print('Please specify an output suffix (e.g. detectron_pt_coco)')
        exit(0)

    import_func = suggest_pose_importer(args.output)
    metadata = suggest_metadata(args.output)

    print('Parsing 2D detections from', args.input)

    output = {}
    file_list = glob(args.input + '/S*/*.mp4.npz')
    for f in file_list:
        path, fname = os.path.split(f)
        subject = os.path.basename(path)
        assert subject.startswith('S'), subject + ' does not look like a subject directory'

        if '_ALL' in fname:
            continue
        
        m = re.search('(.*)\\.([0-9]+)\\.mp4\\.npz', fname)
        action = m.group(1)
        camera = m.group(2)
        camera_idx = cam_map[camera]
        
        if subject == 'S11' and action == 'Directions':
            continue # Discard corrupted video
        
        # Use consistent naming convention
        canonical_name = action.replace('TakingPhoto', 'Photo') \
                               .replace('WalkingDog', 'WalkDog')

        keypoints = import_func(f)
        assert keypoints.shape[1] == metadata['num_joints']
        
        if subject not in output:
            output[subject] = {}
        if canonical_name not in output[subject]:
            output[subject][canonical_name] = [None, None, None, None]
        output[subject][canonical_name][camera_idx] = keypoints.astype('float32')

    print('Saving...')
    np.savez_compressed(output_prefix_2d + args.output, positions_2d=output, metadata=metadata)
    print('Done.')