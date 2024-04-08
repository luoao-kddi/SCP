# SCP

This is the official code for our paper "SCP: Spherical-Coordinate-based Learned Point Cloud Compression". [paper](https://ojs.aaai.org/index.php/AAAI/article/download/28188/28374)

## Installation

Directly run the `install.sh` script:
```
./install.sh
```

## Data

### Downloading

#### KITTI dataset

Please refer to the official [website of KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), Then download the 80GB dataset "odometry data set (velodyne laser data, 80 GB)".

After getting the KITTI dataset, unzip the file in `data/kitti/`.

#### Ford dataset

You need to be a member of MPEG first. The Ford dataset is provided on their [website](https://mpegfs.int-evry.fr/mpegcontent/ws-mpegcontent/MPEG-I).

### Training data preproc

It is better to use multiple processes to preprocess the data. It can be much longer time if you use only one process. We provide a `data_preproc/data_preprocess.py`, whose usage is:

```
python data_preproc/data_preprocess.py <proc_number> <command>
```
as used below.

Generating training set. **Note:** the generated dataset is big (KITTI dataset ~2TB, Ford is much smaller), take care of your storage.
```
# Spherical KITTI
python data_preproc/multi_data_preproc.py 128 python data_preproc/data_preprocess.py --type kitti --ori_dir data/kitti/sequences/\*/velodyne/\*.bin --out_dir data/kitti/spher --spher

# Spherical Ford
python data_preproc/multi_data_preproc.py 128 python data_preproc/data_preprocess.py --type ford --ori_dir data/ford/Ford_01_q_1mm/\*.ply --out_dir data/ford/spher --spher
```

You can remove `--spher` to generate Cartesian-coordinate-based data, remember to change the `--out_dir`.

### Testing data preproc

If you do not need D2 PSNR on KITTI dataset, you can skip this step.

Calculate normals for KITTI dataset to calculate D2 PSNR.
```
python data_preproc/multi_data_preproc.py 128 python data_preproc/gene_normals.py --ori_dir data/kitti/sequences/test/\*/velodyne/\*.bin --out_dir data/kitti/test_norm
```

Generate testing data with multi-level Octree.

You can change the lidar_level from **11 to 16 for KITTI** and **11 to 17 for Ford**. **Note:** the generated dataset is big, take care of your storage. You can add `--spher` or `--cylin` to generate spherical/cylindrical-coordinate-based data. You can **further** add `--mullevel` to generate multi-level octree.
```
python data_preproc/multi_data_preproc.py 128 python data_preproc/test_gene.py --type kitti --ori_dir data/kitti/test_norm/\*/\*.ply --out_dir data/kitti/spher_mullevel_16 --lidar_level 16

python data_preproc/multi_data_preproc.py 128 python data_preproc/test_gene.py --type ford --ori_dir data/ford/test/\*/\*.ply --out_dir data/ford/mullevel_17 --lidar_level 17
```

## Train

Train the EHEM model with generated Spherical-coordinate-based data.
```
# SCP-EHEM KITTI
python train.py --config-dir configs/ --config-name train_kitti_ehem.yaml data.batch_size=16 gpus=\[0,1\] data.root=data/kitti/spher/\*.npy

# SCP-EHEM Ford
python train.py --config-dir configs/ --config-name train_ford_ehem.yaml data.batch_size=16 gpus=\[0,1\] data.root=data/ford/spher/\*.npy
```

## Eval

Evaluate the SCP-EHEM model with generated testing data. Following are some examples:
```
CUDA_VISIBLE_DEVICES=0 python encode_mullevel.py --type kitti --lidar_level 16 --ckpt_path outputs/kitti/2023-07-05/15-21-55/ckpt/epoch=19-step=124840.ckpt --spher --preproc_path data/kitti/mullevel_16/ --test_files data/kitti/test_norm/17

CUDA_VISIBLE_DEVICES=0 python encode.py --type kitti --lidar_level 12 --ckpt_path outputs/kitti/2023-07-05/15-21-55/ckpt/epoch=19-step=124840.ckpt --spher --preproc_path data/kitti/samelevel_12/ --test_files data/kitti/test_norm/17

CUDA_VISIBLE_DEVICES=2 python encode.py --type kitti --lidar_level 14 --ckpt_path outputs/kitti/2023-07-04/17-21-18/ckpt/epoch=8-step=93303.ckpt --cylin --preproc_path data/kitti/cylinlevel_14/ --test_files data/kitti/test_norm/17
```

The `preproc_path` is generated in "Testing data preproc".

## Checkpoints

SCP-EHEM and SCP-OctAttention for KITTI dataset: [link](https://drive.google.com/drive/folders/1PIwSZGPIZwEMiHVf8br4XFRDsdxcdq2T?usp=sharing)

## Citation

If you found this work is helpful for you, please cite:

```
@inproceedings{luo2024scp,
  title={SCP: Spherical-Coordinate-Based Learned Point Cloud Compression},
  author={Luo, Ao and Song, Linxin and Nonaka, Keisuke and Unno, Kyohei and Sun, Heming and Goto, Masayuki and Katto, Jiro},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={3954--3962},
  year={2024}
}
```
