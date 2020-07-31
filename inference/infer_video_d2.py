# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Perform inference on a single video or all videos with a certain extension
(e.g., .mp4) in a folder.
"""

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import subprocess as sp
import numpy as np
import time
import argparse
import sys
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: mp4)',
        default='mp4',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    for line in pipe.stdout:
        w, h = line.decode().strip().split(',')
        return int(w), int(h)

def read_video(filename):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    while True:
        data = pipe.stdout.read(w*h*3)
        if not data:
            break
        yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def main(args):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)
    predictor = DefaultPredictor(cfg)
    

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for video_name in im_list:
        out_name = os.path.join(
                args.output_dir, os.path.basename(video_name)
            )
        print('Processing {}'.format(video_name))

        boxes = []
        segments = []
        keypoints = []

        for frame_i, im in enumerate(read_video(video_name)):
            t = time.time()
            outputs = predictor(im)['instances'].to('cpu')
            
            print('Frame {} processed in {:.3f}s'.format(frame_i, time.time() - t))

            has_bbox = False
            if outputs.has('pred_boxes'):
                bbox_tensor = outputs.pred_boxes.tensor.numpy()
                if len(bbox_tensor) > 0:
                    has_bbox = True
                    scores = outputs.scores.numpy()[:, None]
                    bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
            if has_bbox:
                kps = outputs.pred_keypoints.numpy()
                kps_xy = kps[:, :, :2]
                kps_prob = kps[:, :, 2:3]
                kps_logit = np.zeros_like(kps_prob) # Dummy
                kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
                kps = kps.transpose(0, 2, 1)
            else:
                kps = []
                bbox_tensor = []
                
            # Mimic Detectron1 format
            cls_boxes = [[], bbox_tensor]
            cls_keyps = [[], kps]
            
            boxes.append(cls_boxes)
            segments.append(None)
            keypoints.append(cls_keyps)

        
        # Video resolution
        metadata = {
            'w': im.shape[1],
            'h': im.shape[0],
        }
        
        np.savez_compressed(out_name, boxes=boxes, segments=segments, keypoints=keypoints, metadata=metadata)


if __name__ == '__main__':
    setup_logger()
    args = parse_args()
    main(args)
