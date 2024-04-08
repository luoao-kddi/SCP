import os
import uuid
import math
import numpy as np
import torch.utils.data as data
from data_preproc.data_preprocess import proc_pc, MVUB_NAMES
from pathlib import Path
import data_preproc.pt as pointCloud
from utils import get_psnr


class EncodeEHEMDataset(data.Dataset):
    """ImageFolder can be used to load images where there are no labels."""

    def __init__(
        self,
        test_files,
        context_size,
        data_type,
        level_wise=True,
        lidar_level=12,
        cylin=False,
        spher=False,
        circle=False,
        extra_pos=False,
        preproc_path=''
    ):
        self.test_files = test_files
        self.context_size = context_size
        self.data_type = data_type
        self.level_wise = level_wise
        self.lidar_level = lidar_level
        self.cylin = cylin
        self.spher = spher
        self.extra_pos = extra_pos
        self.circle = circle
        self.preproc_path = preproc_path
        if not os.path.exists('temp'):
            os.mkdir('temp')

    def __getitem__(self, index):
        ori_file = self.test_files[index]
        if self.cylin:
            npy_path, pc, chamfer, bin_num, z_offset, psnr = self.preproc(ori_file)
        elif self.spher:
            npy_path, pc, chamfer, bin_num, psnr = self.preproc(ori_file)
            z_offset = 0
        else:
            npy_path, pc, chamfer, psnr = self.preproc(ori_file)
            z_offset = 0
            bin_num = 0
        oct_seq = np.load(npy_path + ".npy")

        oct_seq[:, :, 0] -= 1
        whole_ids = np.arange(len(oct_seq)).astype(np.int64)
        max_level = max(oct_seq[:, -1, 1])
        data = []
        poss = []
        pos_mm = []
        xyz_poss = []
        ids = []
        cur_level = 1 if self.level_wise else 100
        cur_level_start = 0
        for i in range(len(oct_seq)):
            if oct_seq[i, -1, 1] > cur_level:
                level_data = oct_seq[cur_level_start:i, :, :3]
                level_data = np.concatenate((level_data[:, :, 1:], level_data[:, :, :1]), axis=2)
                data.append(level_data)
                if self.spher or self.cylin:
                    cur_pos = oct_seq[cur_level_start:i, -1, 3:6]
                    pos_max, pos_min = cur_pos.max(), cur_pos.min()
                    poss.append(((cur_pos - pos_min) / (pos_max - pos_min + 1e-9)).astype(np.float32).transpose((1, 0)))
                    pos_mm.append((pos_min, pos_max))
                else:
                    poss.append((oct_seq[cur_level_start:i, :, 3:6] / (2**max_level)).astype(np.float32)[:, -1].transpose((1, 0)))
                if self.extra_pos:
                    xyz_pos = oct_seq[cur_level_start:i, -1, 6:9]
                    pos_max, pos_min = xyz_pos.max(), xyz_pos.min()
                    xyz_pos = ((xyz_pos-pos_min) / (pos_max-pos_min)).astype(np.float32).transpose((1, 0))
                    xyz_poss.append(xyz_pos)
                ids.append(whole_ids[cur_level_start:i] - cur_level_start)
                cur_level_start = i
                cur_level = oct_seq[i, -1, 1]

        level_data = oct_seq[cur_level_start:, :, :3]
        level_data[:, :, 1] = np.clip(level_data[:, :, 1], None, self.lidar_level)
        level_data = np.concatenate((level_data[:, :, 1:], level_data[:, :, :1]), axis=2)  # level, octant, occupancy
        data.append(level_data)
        if self.spher or self.cylin:
            cur_pos = oct_seq[cur_level_start:, -1, 3:6]
            pos_max, pos_min = cur_pos.max(), cur_pos.min()
            poss.append(((cur_pos - pos_min) / (pos_max - pos_min + 1e-9)).astype(np.float32).transpose((1, 0)))
            pos_mm.append((pos_min, pos_max))
        else:
            poss.append((oct_seq[cur_level_start:, :, 3:6] / (2**max_level)).astype(np.float32)[:, -1].transpose((1, 0)))
        if self.extra_pos:
            xyz_pos = oct_seq[cur_level_start:, -1, 6:9]
            pos_max, pos_min = xyz_pos.max(), xyz_pos.min()
            xyz_pos = ((xyz_pos-pos_min) / (pos_max-pos_min)).astype(np.float32).transpose((1, 0))
            xyz_poss.append(xyz_pos)
        ids.append(whole_ids[cur_level_start:] - cur_level_start)
        if self.extra_pos:
            return ids, poss, pos_mm, xyz_poss, data, oct_seq, len(pc), pc, bin_num, z_offset, chamfer, psnr
        else:
            return ids, poss, pos_mm, data, oct_seq, len(pc), pc, bin_num, z_offset, chamfer, psnr

    def preproc(self, ori_file):
        ori_path = Path(ori_file)
        out_file = ori_path.parent / ori_path.stem / Path(".npy")
        if out_file.exists():
            return str(out_file)

        tmp_test_file = "temp/pcerror_results" + str(uuid.uuid4()) + ".txt"
        if self.data_type == 'kitti':
            peak = '59.70'
        elif self.data_type == 'ford':
            peak = '30000'
        if self.data_type == "obj":
            rotate = False
            for mvub_name in MVUB_NAMES:
                if mvub_name in ori_file:
                    rotate = True
                    break
            return proc_pc(ori_file, ori_path.parent, ori_path.stem, rotation=rotate, test=True)[:2] + [0.0]
        elif self.cylin:
            if self.preproc_path:
                if self.data_type == 'kitti':
                    preproc_path = self.preproc_path + ori_file.split('/')[-2] + ori_path.stem
                else:
                    preproc_path = self.preproc_path + ori_path.stem
                whole_pc = pointCloud.loadply(ori_file)[0]
                bin_num, chamfer, z_offset = np.load(preproc_path + '_meta.npy')
                bin_num = int(bin_num)
                z_offset = int(z_offset)
                return preproc_path, whole_pc, chamfer, bin_num, z_offset, 0
            out_file, quantized_pc, pc, bin_num, offset = proc_pc(
                ori_file,
                ori_path.parent,
                ori_path.stem,
                normalize=False,
                qs=400 / (2**self.lidar_level - 1) if self.data_type == 'kitti' else 2**(18-self.lidar_level),
                test=True,
                cylin=self.cylin,
            )
            # TODO move this outside
            # pointCloud.pcerror(pc, quantized_pc, None, "-r " + peak, tmp_test_file)
            return (out_file, pc, pointCloud.distChamfer(pc, quantized_pc), bin_num, offset[0, 2], 0)
        elif self.spher:
            if self.preproc_path:
                if self.data_type == 'kitti':
                    preproc_path = self.preproc_path + ori_file.split('/')[-2] + ori_path.stem
                else:
                    preproc_path = self.preproc_path + ori_path.stem
                whole_pc = pointCloud.loadply(ori_file)[0]
                bin_num, chamfer = np.load(preproc_path + '_meta.npy')
                bin_num = int(bin_num)
                return preproc_path, whole_pc, chamfer, bin_num, 0
            else:
                out_file, quantized_pc, pc, bin_num = proc_pc(
                    ori_file,
                    ori_path.parent,
                    ori_path.stem,
                    normalize=False,
                    qs=400 / (2**self.lidar_level - 1) if self.data_type == 'kitti' else 2**(18-self.lidar_level),
                    test=True,
                    spher=self.spher,
                    xyz=self.extra_pos,
                    circle=self.circle
                )
                pointCloud.pcerror(pc, quantized_pc, None, "-r " + peak, tmp_test_file)
                return out_file, pc, pointCloud.distChamfer(pc, quantized_pc), bin_num, get_psnr(tmp_test_file)[0]
        else:
            out_file, quantized_pc, pc = proc_pc(
                ori_file,
                ori_path.parent,
                ori_path.stem,
                offset=-200,
                normalize=False,
                qs=400 / (2**self.lidar_level - 1) if self.data_type == 'kitti' else 2**(18-self.lidar_level),
                test=True,
            )
            pointCloud.pcerror(pc, quantized_pc, None, "-r " + peak, tmp_test_file)
            return out_file, pc, pointCloud.distChamfer(pc, quantized_pc), get_psnr(tmp_test_file)[0]

    def __len__(self):
        return len(self.test_files)
