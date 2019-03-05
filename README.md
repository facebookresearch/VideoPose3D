# This is the 'in the wild' fork of 3D human pose estimation in video with temporal convolutions and semi-supervised training
With this repository you can run the VideoPose3D code on your own data. run_wild.py enables you to run the code on 2D poses that you created yourself using detectron. I did not use CPN 2D pose refinemend as discussed [here](https://github.com/facebookresearch/VideoPose3D/issues/2#issuecomment-443502874). 

To run this:
If you want to run this on your own videos you have to do step 1. and 2. otherwise go directly to 4. VideoPose3D part

### Detectron part
1. Find a video you like and download it.
2. Use ffmpeg to split it into individual frames in 'detectron_tools/' you can find the modified infer_simple.py file witch helps you to export the 2D poses. And the detectron_tools.txt that shows and example on how to use ffmpeg on your video.
2. Run detecton:
	- Download the [config-file](https://github.com/facebookresearch/Detectron/blob/master/configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml) (specified by --cfg flag) you will need it for detectron.
	- Download the weights file with the coco keypoints! ([weights file](https://github.com/tobiascz/VideoPose3D/issues/2))
	- Replace pathToYourWeightFileFrom and run detectron with your arguments:
```
python infer_simple.py --cfg /detectron/videopose3d/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml --output-dir demo/scating_vis --image-ext jpg --wts
	/pathToYourWeightFileFrom2.2/keypoints_coco_2014_train:keypoints_coco_2014_valminusminival/model_final.pkl
demo/splitted_scating
```
### VideoPose3D part
4. Move the data_2d_detections.npz file, that you created in step 2, into VideoPose3D/data or just use the one I created from the ice scating video.
5. When you run the program make sure you choose the right video file! The ice scating video is located at InTheWildData/out_cutted.mp4
6. Download the checkpoint file provided by the authors and move it to your VideoPose3D/checkpoint folder. [Help](https://s3.amazonaws.com/video-pose-3d/d-pt-243.bin)
7. Prepare the data_3d_h36m.npz file in the data directory. [Help](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md#setup-from-preprocessed-dataset)
8. Now it is time to run the thing!

My arguments for VideoPose3D:
```
python run_wild -k detections -arc 3,3,3,3,3 -c checkpoint --evaluate d-pt-243.bin --render --viz-subject S1 --viz-action Directions --viz-video InTheWildData/out_cutted.mp4 --viz-camera 0 --viz-output output_scater.mp4 --viz-size 5 --viz-downsample 1 --viz-skip 9
```

### Result

![](https://user-images.githubusercontent.com/8737489/52120837-f0f4e880-261d-11e9-8827-77a869f37d6a.gif)

# 3D human pose estimation in video with temporal convolutions and semi-supervised training
<p align="center"><img src="images/convolutions_anim.gif" width="50%" alt="" /></p>


This is the implementation of the approach described in the paper:
> Dario Pavllo, Christoph Feichtenhofer, David Grangier, and Michael Auli. [3D human pose estimation in video with temporal convolutions and semi-supervised training](https://arxiv.org/abs/1811.11742). In *arXiv*, 2018.

More demos are available at https://dariopavllo.github.io/VideoPose3D

<p align="center"><img src="images/demo_yt.gif" width="70%" alt="" /></p>

![](images/demo_temporal.gif)

### Results on Human3.6M
Under Protocol 1 (mean per-joint position error) and Protocol 2 (mean-per-joint position error after rigid alignment).

| 2D Detections | BBoxes | Blocks | Receptive Field | Error (P1) | Error (P2) |
|:-------|:-------:|:-------:|:-------:|:-------:|:-------:|
| CPN | Mask R-CNN  | 4 | 243 frames | **46.8 mm** | **36.5 mm** |
| CPN | Ground truth | 4 | 243 frames | 47.1 mm | 36.8 mm |
| CPN | Ground truth | 3 | 81 frames | 47.7 mm | 37.2 mm |
| CPN | Ground truth | 2 | 27 frames | 48.8 mm | 38.0 mm |
| Mask R-CNN | Mask R-CNN | 4 | 243 frames | 51.6 mm | 40.3 mm |
| Ground truth | -- | 4 | 243 frames | 37.2 mm | 27.2 mm |

## Quick start
To get started as quickly as possible, follow the instructions in this section. This should allow you train a model from scratch, test our pretrained models, and produce basic visualizations. For more detailed instructions, please refer to [`DOCUMENTATION.md`](DOCUMENTATION.md).

### Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python 3+ distribution
- PyTorch >= 0.4.0

Optional:
- Matplotlib, if you want to visualize predictions. Additionally, you need *ffmpeg* to export MP4 videos, and *imagemagick* to export GIFs.
- MATLAB, if you want to experiment with HumanEva-I (you need this to convert the dataset). 

### Dataset setup
You can find the instructions for setting up the Human3.6M and HumanEva-I datasets in [`DATASETS.md`](DATASETS.md). For this short guide, we focus on Human3.6M. You are not required to setup HumanEva, unless you want to experiment with it.

In order to proceed, you must also copy CPN detections (for Human3.6M) and/or Mask R-CNN detections (for HumanEva).

### Evaluating our pretrained models
The pretrained models can be downloaded from AWS. Put `pretrained_h36m_cpn.bin` (for Human3.6M) and/or `pretrained_humaneva15_detectron.bin` (for HumanEva) in the `checkpoint/` directory (create it if it does not exist).
```sh
mkdir checkpoint
cd checkpoint
wget https://s3.amazonaws.com/video-pose-3d/pretrained_h36m_cpn.bin
wget https://s3.amazonaws.com/video-pose-3d/pretrained_humaneva15_detectron.bin
cd ..
```

These models allow you to reproduce our top-performing baselines, which are:
- 46.8 mm for Human3.6M, using fine-tuned CPN detections, bounding boxes from Mask R-CNN, and an architecture with a receptive field of 243 frames.
- 28.6 mm for HumanEva-I (on 3 actions), using pretrained Mask R-CNN detections, and an architecture with a receptive field of 27 frames. This is the multi-action model trained on 3 actions (Walk, Jog, Box).

To test on Human3.6M, run:
```
python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin
```

To test on HumanEva, run:
```
python run.py -d humaneva15 -k detectron_pt_coco -str Train/S1,Train/S2,Train/S3 -ste Validate/S1,Validate/S2,Validate/S3 -a Walk,Jog,Box --by-subject -c checkpoint --evaluate pretrained_humaneva15_detectron.bin
```

[`DOCUMENTATION.md`](DOCUMENTATION.md) provides a precise description of all command-line arguments.

### Training from scratch
If you want to reproduce the results of our pretrained models, run the following commands.

For Human3.6M:
```
python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3
```
By default the application runs in training mode. This will train a new model for 80 epochs, using fine-tuned CPN detections. Expect a training time of 24 hours on a high-end Pascal GPU. If you feel that this is too much, or your GPU is not powerful enough, you can train a model with a smaller receptive field, e.g.
- `-arc 3,3,3,3` (81 frames) should require 11 hours and achieve 47.7 mm. 
- `-arc 3,3,3` (27 frames) should require 6 hours and achieve 48.8 mm.

You could also lower the number of epochs from 80 to 60 with a negligible impact on the result.

For HumanEva:
```
python run.py -d humaneva15 -k detectron_pt_coco -str Train/S1,Train/S2,Train/S3 -ste Validate/S1,Validate/S2,Validate/S3 -b 128 -e 1000 -lrd 0.996 -a Walk,Jog,Box --by-subject
```
This will train for 1000 epochs, using Mask R-CNN detections and evaluating each subject separately.
Since HumanEva is much smaller than Human3.6M, training should require about 50 minutes.

### Semi-supervised training
To perform semi-supervised training, you just need to add the `--subjects-unlabeled` argument. In the example below, we use ground-truth 2D poses as input, and train supervised on just 10% of Subject 1 (specified by `--subset 0.1`). The remaining subjects are treated as unlabeled data and are used for semi-supervision.
```
python run.py -k gt --subjects-train S1 --subset 0.1 --subjects-unlabeled S5,S6,S7,S8 -e 200 -lrd 0.98 -arc 3,3,3 --warmup 5 -b 64
```
This should give you an error around 65.2 mm. By contrast, if we only train supervised
```
python run.py -k gt --subjects-train S1 --subset 0.1 -e 200 -lrd 0.98 -arc 3,3,3 -b 64
```
we get around 80.7 mm, which is significantly higher.

### Visualization
If you have the original Human3.6M videos, you can generate nice visualizations of the model predictions. For instance:
```
python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin --render --viz-subject S11 --viz-action Walking --viz-camera 0 --viz-video "/path/to/videos/S11/Videos/Walking.54138969.mp4" --viz-output output.gif --viz-size 3 --viz-downsample 2 --viz-limit 60
```
The script can also export MP4 videos, and supports a variety of parameters (e.g. downsampling/FPS, size, bitrate). See [`DOCUMENTATION.md`](DOCUMENTATION.md) for more details.

## License
This work is licensed under CC BY-NC. See LICENSE for details. Third-party datasets are subject to their respective licenses.
If you use our code/models in your research, please cite our paper:
```
@article{pavllo:videopose3d:2018,
  title={3D human pose estimation in video with temporal convolutions and semi-supervised training},
  author={Pavllo, Dario and Feichtenhofer, Christoph and Grangier, David and Auli, Michael},
  journal={arXiv},
  volume={abs/1811.11742},
  year={2018}
}
```
