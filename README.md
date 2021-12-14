# Dynamic View Synthesis from Dynamic Monocular Video

[![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2105.06468)

[Project Website](https://free-view-video.github.io/) | [Video](https://youtu.be/j8CUzIR0f8M) | [Paper](https://arxiv.org/abs/2105.06468)

> **Dynamic View Synthesis from Dynamic Monocular Video**<br>
> [Chen Gao](http://chengao.vision), [Ayush Saraf](#), [Johannes Kopf](https://johanneskopf.de/), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/) <br>
in ICCV 2021 <br>

## Setup
The code is test with
* Linux (tested on CentOS Linux release 7.4.1708)
* Anaconda 3
* Python 3.7.11
* CUDA 10.1
* 1 V100 GPU


To get started, please create the conda environment `dnerf` by running
```
conda create --name dnerf
conda activate dnerf
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
pip install imageio configargparse timm lpips
```
and install [COLMAP](https://colmap.github.io/install.html) manually. Then download MiDaS and RAFT weights
```
ROOT_PATH=/path/to/the/DynamicNeRF/folder
cd $ROOT_PATH
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/weights.zip
unzip weights.zip
rm weights.zip
```

## Dynamic Scene Dataset
The [Dynamic Scene Dataset](https://www-users.cse.umn.edu/~jsyoon/dynamic_synth/) is used to
quantitatively evaluate our method. Please download the pre-processed data by running:
```
cd $ROOT_PATH
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/data.zip
unzip data.zip
rm data.zip
```

### Training
You can train a model from scratch by running:
```
cd $ROOT_PATH/
python run_nerf.py --config configs/config_Balloon2.txt
```

Every 100k iterations, you should get videos like the following examples

The novel view-time synthesis results will be saved in `$ROOT_PATH/logs/Balloon2_H270_DyNeRF/novelviewtime`.
![novelviewtime](https://filebox.ece.vt.edu/~chengao/free-view-video/gif/novelviewtime_Balloon2.gif)
<!-- <img src="https://filebox.ece.vt.edu/~chengao/free-view-video/gif/novelviewtime.gif" height="270" /> -->

The reconstruction results will be saved in `$ROOT_PATH/logs/Balloon2_H270_DyNeRF/testset`.
![testset](https://filebox.ece.vt.edu/~chengao/free-view-video/gif/testset_Balloon2.gif)

The fix-view-change-time results will be saved in `$ROOT_PATH/logs/Balloon2_H270_DyNeRF/testset_view000`.
![testset_view000](https://filebox.ece.vt.edu/~chengao/free-view-video/gif/testset_view000_Balloon2.gif)

The fix-time-change-view results will be saved in `$ROOT_PATH/logs/Balloon2_H270_DyNeRF/testset_time000`.
![testset_time000](https://filebox.ece.vt.edu/~chengao/free-view-video/gif/testset_time000_Balloon2.gif)


### Rendering from pre-trained models
We also provide pre-trained models. You can download them by running:
```
cd $ROOT_PATH/
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/logs.zip
unzip logs.zip
rm logs.zip
```

Then you can render the results directly by running:
```
python run_nerf.py --config configs/config_Balloon2.txt --render_only --ft_path $ROOT_PATH/logs/Balloon2_H270_DyNeRF_pretrain/300000.tar
```

### Evaluating our method and others
Our goal is to make the evaluation as simple as possible for you. We have collected the fix-view-change-time results of the following methods:

`NeRF` \
`NeRF + t` \
`Yoon et al.` \
`Non-Rigid NeRF` \
`NSFF` \
`DynamicNeRF (ours)`

Please download the results by running:
```
cd $ROOT_PATH/
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/results.zip
unzip results.zip
rm results.zip
```

Then you can calculate the PSNR/SSIM/LPIPS by running:
```
cd $ROOT_PATH/utils
python evaluation.py
```

| PSNR / LPIPS |    Jumping    |    Skating    |     Truck     |    Umbrella   |    Balloon1   |    Balloon2   |   Playground  |    Average    |
|:-------------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| NeRF         | 20.99 / 0.305 | 23.67 / 0.311 | 22.73 / 0.229 | 21.29 / 0.440 | 19.82 / 0.205 | 24.37 / 0.098 | 21.07 / 0.165 | 21.99 / 0.250 |
| NeRF + t     | 18.04 / 0.455 | 20.32 / 0.512 | 18.33 / 0.382 | 17.69 / 0.728 | 18.54 / 0.275 | 20.69 / 0.216 | 14.68 / 0.421 | 18.33 / 0.427 |
| NR NeRF      | 20.09 / 0.287 | 23.95 / 0.227 | 19.33 / 0.446 | 19.63 / 0.421 | 17.39 / 0.348 | 22.41 / 0.213 | 15.06 / 0.317 | 19.69 / 0.323 |
| NSFF         | 24.65 / 0.151 | 29.29 / 0.129 | 25.96 / 0.167 | 22.97 / 0.295 | 21.96 / 0.215 | 24.27 / 0.222 | 21.22 / 0.212 | 24.33 / 0.199 |
| Ours         | 24.68 / 0.090 | 32.66 / 0.035 | 28.56 / 0.082 | 23.26 / 0.137 | 22.36 / 0.104 | 27.06 / 0.049 | 24.15 / 0.080 | 26.10 / 0.082 |


Please note:
1. The numbers reported in the paper are calculated using TF code. The numbers here are calculated using this improved Pytorch version.
2. In Yoon's results, the first frame and the last frame are missing. To compare with Yoon's results, we have to omit the first frame and the last frame. To do so, please uncomment line 72 and comment line 73 in `evaluation.py`.
3. We obtain the results of NSFF and NR NeRF using the official implementation with default parameters.


## Train a model on your sequence
0. Set some paths

```
ROOT_PATH=/path/to/the/DynamicNeRF/folder
DATASET_NAME=name_of_the_video_without_extension
DATASET_PATH=$ROOT_PATH/data/$DATASET_NAME
```

1. Prepare training images and background masks from a video.

```
cd $ROOT_PATH/utils
python generate_data.py --videopath /path/to/the/video
```

2. Use COLMAP to obtain camera poses.

```
colmap feature_extractor \
--database_path $DATASET_PATH/database.db \
--image_path $DATASET_PATH/images_colmap \
--ImageReader.mask_path $DATASET_PATH/background_mask \
--ImageReader.single_camera 1

colmap exhaustive_matcher \
--database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images_colmap \
    --output_path $DATASET_PATH/sparse \
    --Mapper.num_threads 16 \
    --Mapper.init_min_tri_angle 4 \
    --Mapper.multiple_models 0 \
    --Mapper.extract_colors 0
```

3. Save camera poses into the format that NeRF reads.

```
cd $ROOT_PATH/utils
python generate_pose.py --dataset_path $DATASET_PATH
```

4. Estimate monocular depth.

```
cd $ROOT_PATH/utils
python generate_depth.py --dataset_path $DATASET_PATH --model $ROOT_PATH/weights/midas_v21-f6b98070.pt
```

5. Predict optical flows.

```
cd $ROOT_PATH/utils
python generate_flow.py --dataset_path $DATASET_PATH --model $ROOT_PATH/weights/raft-things.pth
```

6. Obtain motion mask (code adapted from NSFF).

```
cd $ROOT_PATH/utils
python generate_motion_mask.py --dataset_path $DATASET_PATH
```

7. Train a model. Please change `expname` and `datadir` in `configs/config.txt`.

```
cd $ROOT_PATH/
python run_nerf.py --config configs/config.txt
```

Explanation of each parameter:

- `expname`: experiment name
- `basedir`: where to store ckpts and logs
- `datadir`: input data directory
- `factor`: downsample factor for the input images
- `N_rand`: number of random rays per gradient step
- `N_samples`: number of samples per ray
- `netwidth`: channels per layer
- `use_viewdirs`: whether enable view-dependency for StaticNeRF
- `use_viewdirsDyn`: whether enable view-dependency for DynamicNeRF
- `raw_noise_std`: std dev of noise added to regularize sigma_a output
- `no_ndc`: do not use normalized device coordinates
- `lindisp`: sampling linearly in disparity rather than depth
- `i_video`: frequency of novel view-time synthesis video saving
- `i_testset`: frequency of testset video saving
- `N_iters`: number of training iterations
- `i_img`: frequency of tensorboard image logging
- `DyNeRF_blending`: whether use DynamicNeRF to predict blending weight
- `pretrain`: whether pre-train StaticNeRF

## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details.

If you find this code useful for your research, please consider citing the following paper:

	@inproceedings{Gao-ICCV-DynNeRF,
	    author    = {Gao, Chen and Saraf, Ayush and Kopf, Johannes and Huang, Jia-Bin},
	    title     = {Dynamic View Synthesis from Dynamic Monocular Video},
	    booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
	    year      = {2021}
	}

## Acknowledgments
Our training code is build upon
[NeRF](https://github.com/bmild/nerf),
[NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch), and
[NSFF](https://github.com/zl548/Neural-Scene-Flow-Fields).
Our flow prediction code is modified from [RAFT](https://github.com/princeton-vl/RAFT).
Our depth prediction code is modified from [MiDaS](https://github.com/isl-org/MiDaS).
