# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

mpii_metadata = {
    'layout_name': 'mpii',
    'num_joints': 16,
    'keypoints_symmetry': [
        [3, 4, 5, 13, 14, 15],
        [0, 1, 2, 10, 11, 12],
    ]
}

coco_metadata = {
    'layout_name': 'coco',
    'num_joints': 17,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
    ]
}

h36m_metadata = {
    'layout_name': 'h36m',
    'num_joints': 17,
    'keypoints_symmetry': [
        [4, 5, 6, 11, 12, 13],
        [1, 2, 3, 14, 15, 16],
    ]
}

humaneva15_metadata = {
    'layout_name': 'humaneva15',
    'num_joints': 15,
    'keypoints_symmetry': [
        [2, 3, 4, 8, 9, 10],
        [5, 6, 7, 11, 12, 13]
    ]
}

humaneva20_metadata = {
    'layout_name': 'humaneva20',
    'num_joints': 20,
    'keypoints_symmetry': [
        [3, 4, 5, 6, 11, 12, 13, 14],
        [7, 8, 9, 10, 15, 16, 17, 18]
    ]
}

def suggest_metadata(name):
    names = []
    for metadata in [mpii_metadata, coco_metadata, h36m_metadata, humaneva15_metadata, humaneva20_metadata]:
        if metadata['layout_name'] in name:
            return metadata
        names.append(metadata['layout_name'])
    raise KeyError('Cannot infer keypoint layout from name "{}". Tried {}.'.format(name, names))

def import_detectron_poses(path):
    # Latin1 encoding because Detectron runs on Python 2.7
    data = np.load(path, encoding='latin1')
    kp = data['keypoints']
    bb = data['boxes']
    results = []
    for i in range(len(bb)):
        if len(bb[i][1]) == 0:
            assert i > 0
            # Use last pose in case of detection failure
            results.append(results[-1])
            continue
        best_match = np.argmax(bb[i][1][:, 4])
        keypoints = kp[i][1][best_match].T.copy()
        results.append(keypoints)
    results = np.array(results)
    return results[:, :, 4:6] # Soft-argmax
    #return results[:, :, [0, 1, 3]] # Argmax + score
    
    
def import_cpn_poses(path):
    data = np.load(path)
    kp = data['keypoints']
    return kp[:, :, :2]
    
    
def import_sh_poses(path):
    import h5py
    with h5py.File(path) as hf:
        positions = hf['poses'].value
    return positions.astype('float32')
    
def suggest_pose_importer(name):
    if 'detectron' in name:
        return import_detectron_poses
    if 'cpn' in name:
        return import_cpn_poses
    if 'sh' in name:
        return import_sh_poses
    raise KeyError('Cannot infer keypoint format from name "{}". Tried detectron, cpn, sh.'.format(name))
