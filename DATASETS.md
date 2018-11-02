# Dataset setup

## Human3.6M
We provide two ways to set up the Human3.6M dataset on our pipeline. You can either use the [dataset preprocessed by Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) (fastest way) or convert the original dataset from scratch. The two methods produce the same result. After this step, you should end up with two files in the `data` directory: `data_3d_h36m.npz` for the 3D poses, and `data_2d_h36m_gt.npz` for the ground-truth 2D poses.

### Setup from preprocessed dataset
Download the [h36m.zip archive](https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip) (source: [3D pose baseline repository](https://github.com/una-dinosauria/3d-pose-baseline)) to the `data` directory, and run the conversion script from the same directory. This step does not require any additional dependency.

```sh
cd data
wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip
python prepare_data_h36m.py --from-archive h36m.zip
cd ..
```

### Setup from original source
Alternatively, you can download the dataset from the [Human3.6m website](http://vision.imar.ro/human3.6m/) and convert it from its original format. This is useful if the other link goes down, or if you want to be sure to use the original source. MATLAB is required for this step.

First, we need to convert the 3D poses from `.cdf` to `.mat`, so they can be loaded from Python scripts. To this end, we have provided the MATLAB script `convert_cdf_to_mat.m` in the `data` directory. Extract the archives named `Poses_D3_Positions_S*.tgz` (subjects 1, 5, 6, 7, 8, 9, 11) to a directory named `pose`, and set up your directory tree so that it looks like this:

```
/path/to/dataset/convert_cdf_to_mat.m
/path/to/dataset/pose/S1/MyPoseFeatures/D3_Positions/Directions 1.cdf
/path/to/dataset/pose/S1/MyPoseFeatures/D3_Positions/Directions.cdf
...
```
Then run `convert_cdf_to_mat.m` from MATLAB.

Finally, as before, run the Python conversion script specifying the dataset path:
```sh
cd data
python prepare_data_h36m.py --from-source /path/to/dataset/pose
cd ..
```

## 2D detections for Human3.6M
We provide support for the following 2D detections:

- `gt`: ground-truth 2D poses, extracted through the camera projection parameters.
- `sh_pt_mpii`: Stacked Hourglass detections, pretrained on MPII.
- `sh_ft_h36m`: Stacked Hourglass detections, fine-tuned on Human3.6M.
- `detectron_ft_h36m`: Detectron (Mask R-CNN) detections, fine-tuned on Human3.6M.
- `cpn_ft_h36m_dbb`: Cascaded Pyramid Network detections, fine-tuned on Human3.6M. Bounding boxes from `detectron_ft_h36m`.
- User-supplied (see below).

The 2D detection source is specified through the `--keypoints` parameter, which loads the file `data_2d_DATASET_DETECTION.npz` from the `data` directory, where `DATASET` is the dataset name (e.g. `h36m`) and `DETECTION` is the 2D detection source (e.g. `sh_pt_mpii`). Since all the files are encoded according to the same format, it is trivial to create a custom set of 2D detections.

Ground-truth poses (`gt`) have already been extracted by the previous step. The other detections must be downloaded manually (see instructions below). You only need to download the detections you want to use. For reference, our best results on Human3.6M are achieved by `cpn_ft_h36m_dbb`.

### Mask R-CNN and CPN detections
You can download these from AWS. You just have to put `data_2d_h36m_cpn_ft_h36m_dbb.npz` and `data_2d_h36m_detectron_ft_h36m.npz` in the `data` directory.

```sh
cd data
wget https://s3.amazonaws.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz
wget https://s3.amazonaws.com/video-pose-3d/data_2d_h36m_detectron_ft_h36m.npz
cd ..
```

These detections have been produced by models fine-tuned on Human3.6M. We adopted the usual protocol of fine-tuning on 5 subjects (S1, S5, S6, S7, and S8). We also included detections from the unlabeled subjects S2, S3, S4, which can be loaded by our framework for semi-supervised experimentation.

### Stacked Hourglass detections
These detections (both pretrained and fine-tuned) are provided by [Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) in their repository on 3D human pose estimation. The 2D poses produced by the pretrained model are in the same archive as the dataset ([h36m.zip](https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip)). The fine-tuned poses can be downloaded [here](https://drive.google.com/open?id=0BxWzojlLp259S2FuUXJ6aUNxZkE). Put the two archives in the `data` directory and run:

```sh
cd data
python prepare_data_2d_h36m_sh.py -pt h36m.zip
python prepare_data_2d_h36m_sh.py -ft stacked_hourglass_fined_tuned_240.tar.gz
cd ..
```

## HumanEva-I
For HumanEva, you need the original dataset and MATLAB. We provide a MATLAB script to extract the revelant parts of the dataset automatically.

1. Download the [HumanEva-I dataset](http://humaneva.is.tue.mpg.de/datasets_human_1) and extract it.
2. Download the official [source code v1.1 beta](http://humaneva.is.tue.mpg.de/main/download?file=Release_Code_v1_1_beta.zip) and extract it where you extracted the dataset.
3. Copy the contents of the directory `Release_Code_v1_1_beta\HumanEva_I` to the root of the source tree (`Release_Code_v1_1_beta/`).
4. Download the [critical dataset update](http://humaneva.is.tue.mpg.de/main/download?file=Critical_Update_OFS_files.zip) and apply it.
5. **Important:** for visualization purposes, the original code requires an old library named *dxAvi*, which is used for decoding XVID videos. A precompiled binary for 32-bit architectures is already included, but if you are running MATLAB on a 64-bit system, the code will not work. You can either recompile *dxAvi* library for x64, or bypass it entirely, since we are not using visualization features in our conversion script. To this end, you can patch `@sync_stream/sync_stream.m`, replacing line 202: `ImageStream(I) = image_stream(image_paths{I}, start_image_offset(I));` with `ImageStream(I) = 0;`
6. Now you can copy our script `ConvertHumanEva.m` (from `data/`) to `Release_Code_v1_1_beta/`, and run it. It will create a directory named `converted_15j`, which contains the converted 2D/3D ground-truth poses on a 15-joint skeleton.
7. **Optional:** if you want to experiment with a 20-joint skeleton, change `N_JOINTS` to 20 in `ConvertHumanEva.m`, and repeat the process. It will create a directory named `converted_20j`. Adapt next steps accordingly.

If you get warnings about mocap errors or dropped frames, this is normal. The HumanEva dataset contains some invalid frames due to occlusions, which are simply discarded. Since we work with videos (and not individual frames), we try to minimize the impact of this issue by grouping valid sequences into contiguous chunks.

Finally, run the Python script to produce the final files:
```
python prepare_data_humaneva.py -p /path/to/dataset/Release_Code_v1_1_beta/converted_15j --convert-3d
```
You should end up with two files in the `data` directory: `data_3d_humaneva15.npz` for the 3D poses, and `data_2d_humaneva15_gt.npz` for the ground-truth 2D poses.

### 2D detections for HumanEva-I
We provide support for the following 2D detections:

- `gt`: ground-truth 2D poses, extracted through camera projection.
- `detectron_pt_coco`: Detectron (Mask R-CNN) detections, pretrained on COCO.

Since HumanEva is very small, we do not fine-tune the pretrained models. As before, you can download Mask R-CNN detections from AWS (`data_2d_humaneva15_detectron_pt_coco.npz`, which must be copied to `data/`). As before, we have included detections for unlabeled subjects/actions. These begin with the prefix `Unlabeled/`. Chunks that correspond to corrupted motion capture streams are also marked as unlabeled.
```sh
cd data
wget https://s3.amazonaws.com/video-pose-3d/data_2d_humaneva15_detectron_pt_coco.npz
cd ..
```