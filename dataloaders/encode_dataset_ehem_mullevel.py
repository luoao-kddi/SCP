import os
import uuid
import math
import numpy as np
import torch.utils.data as data
from data_preproc.data_preprocess import mul_proc_pc
from pathlib import Path
import data_preproc.pt as pointCloud
from utils import get_psnr


class EncodeEHEMDataset(data.Dataset):
    """ImageFolder can be used to load images where there are no labels."""

    def __init__(self, test_files, context_size, data_type, level_wise=True, lidar_level=12, cylin=False, spher=False, preproc_path=''):
        self.test_files = test_files
        self.context_size = context_size
        self.data_type = data_type
        self.level_wise = level_wise
        self.lidar_level = lidar_level
        self.cylin = cylin
        self.spher = spher
        self.preproc_path = preproc_path
        if not os.path.exists('temp'):
            os.mkdir('temp')

    def __getitem__(self, index):
        ori_file = self.test_files[index]
        # if self.cylin:
        #     npy_paths, pcs, chamfer, bin_num, z_offset = self.preproc(ori_file)
        # else:
        npy_paths, whole_pc, chamfer, bin_num, z_offset, psnr = self.preproc(ori_file)
        # whole_pc = np.vstack(pcs)

        ids, pos, pos_mm, data, oct_seq = self.get_data(npy_paths[0])

        for path in npy_paths[1:]:
            cur_ids, cur_pos, cur_pos_mm, cur_data, cur_oct_seq = self.get_data(path)
            ids += cur_ids
            pos += cur_pos
            pos_mm += cur_pos_mm
            data += cur_data
            oct_seq = np.vstack((oct_seq, cur_oct_seq))

        return ids, pos, pos_mm, data, oct_seq, len(whole_pc), whole_pc, bin_num, z_offset, chamfer, psnr

    def get_data(self, npy_path):
        oct_seq = np.load(npy_path + '.npy')

        oct_seq[:, :, 0] -= 1
        whole_ids = np.arange(len(oct_seq)).astype(np.int64)
        data = []
        pos = []
        ids = []
        pos_mm = []
        cur_level = 1 if self.level_wise else 100
        cur_level_start = 0
        for i in range(len(oct_seq)):
            if oct_seq[i, -1, 1] > cur_level:
                level_data = oct_seq[cur_level_start:i, :, :3]
                level_data = np.concatenate((level_data[:, :, 1:], level_data[:, :, :1]), axis=2)
                data.append(level_data)

                cur_pos = oct_seq[cur_level_start:i, -1, 3:]
                pos_max, pos_min = cur_pos.max(), cur_pos.min()
                pos.append(((cur_pos-pos_min) / (pos_max-pos_min + 1e-9)).astype(np.float32).transpose((1, 0)))
                pos_mm.append((pos_min, pos_max))

                ids.append(whole_ids[cur_level_start:i] - cur_level_start)
                cur_level_start = i
                cur_level = oct_seq[i, -1, 1]

        level_data = oct_seq[cur_level_start:, :, :3]
        level_data[:, :, 1] = np.clip(level_data[:, :, 1], None, self.lidar_level)
        level_data = np.concatenate((level_data[:, :, 1:], level_data[:, :, :1]), axis=2) # level, octant, occupancy
        data.append(level_data)

        cur_pos = oct_seq[cur_level_start:, -1, 3:]
        pos_max, pos_min = cur_pos.max(), cur_pos.min()
        pos.append(((cur_pos-pos_min) / (pos_max-pos_min)).astype(np.float32).transpose((1, 0)))
        pos_mm.append((pos_min, pos_max))

        ids.append(whole_ids[cur_level_start:] - cur_level_start)

        return ids, pos, pos_mm, data, oct_seq

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
        if self.cylin:
            if self.preproc_path:
                if self.data_type == 'kitti':
                    preproc_path = self.preproc_path + ori_file.split('/')[-2] + ori_path.stem
                else:
                    preproc_path = self.preproc_path + ori_path.stem
                whole_pc = pointCloud.loadply(ori_file)[0]
                bin_num, chamfer = np.load(preproc_path + '_meta.npy')
                bin_num = int(bin_num)
                out_files = [preproc_path+'_0_0', preproc_path+'_0_1', preproc_path+'_1']
                return out_files, whole_pc, chamfer, bin_num, 0
            else:
                out_file, quantized_pc, pc, bin_num, z_offset = mul_proc_pc(
                    ori_file,
                    ori_path.parent,
                    ori_path.stem,
                    normalize=False,
                    qs=400 / (2**self.lidar_level - 1) if self.data_type == 'kitti' else 2**(18-self.lidar_level),
                    test=True,
                    cylin=self.cylin,
                    morton_path=[0, 0],
                    )
                out_file2, quantized_pc2, pc2, _, _ = mul_proc_pc(
                    ori_file,
                    ori_path.parent,
                    ori_path.stem,
                    normalize=False,
                    qs=400 / (2**(self.lidar_level + 1) - 1) if self.data_type == 'kitti' else 2**(17-self.lidar_level),
                    test=True,
                    cylin=self.cylin,
                    morton_path=[0, 1],
                    )
                out_file3, quantized_pc3, pc3, _, _ = mul_proc_pc(
                    ori_file,
                    ori_path.parent,
                    ori_path.stem,
                    normalize=False,
                    qs=400 / (2**(self.lidar_level + 2) - 1) if self.data_type == 'kitti' else 2**(16-self.lidar_level),
                    test=True,
                    cylin=self.cylin,
                    morton_path=[1],
                    )
                whole_pc = pc
                whole_q_pc = np.vstack((quantized_pc, quantized_pc2, quantized_pc3))
                pointCloud.pcerror(ori_file, whole_q_pc, None, '-r ' + peak, tmp_test_file)
                out_files = [out_file, out_file2, out_file3]
                return out_files, whole_pc, pointCloud.distChamfer(whole_pc, whole_q_pc), bin_num, z_offset, get_psnr(tmp_test_file)[0]
        elif self.spher:
            if self.preproc_path:
                if self.data_type == 'kitti':
                    preproc_path = self.preproc_path + ori_file.split('/')[-2] + ori_path.stem
                else:
                    preproc_path = self.preproc_path + ori_path.stem
                whole_pc = pointCloud.loadply(ori_file)[0]
                bin_num, chamfer = np.load(preproc_path + '_meta.npy')
                bin_num = int(bin_num)
                out_files = [preproc_path+'_0_0', preproc_path+'_0_1', preproc_path+'_1']
                return out_files, whole_pc, chamfer, bin_num, 0
            else:
                out_file, quantized_pc, pc, bin_num, z_offset = mul_proc_pc(
                    ori_file,
                    ori_path.parent,
                    ori_path.stem,
                    normalize=False,
                    qs=400 / (2**self.lidar_level - 1) if self.data_type == 'kitti' else 2**(18-self.lidar_level),
                    test=True,
                    spher=self.spher,
                    morton_path=[0, 0],
                    )
                out_file2, quantized_pc2, pc2, _, _ = mul_proc_pc(
                    ori_file,
                    ori_path.parent,
                    ori_path.stem,
                    normalize=False,
                    qs=400 / (2**(self.lidar_level + 1) - 1) if self.data_type == 'kitti' else 2**(17-self.lidar_level),
                    test=True,
                    spher=self.spher,
                    morton_path=[0, 1],
                    )
                out_file3, quantized_pc3, pc3, _, _ = mul_proc_pc(
                    ori_file,
                    ori_path.parent,
                    ori_path.stem,
                    normalize=False,
                    qs=400 / (2**(self.lidar_level + 2) - 1) if self.data_type == 'kitti' else 2**(16-self.lidar_level),
                    test=True,
                    spher=self.spher,
                    morton_path=[1],
                    )
                whole_pc = pc
                whole_q_pc = np.vstack((quantized_pc, quantized_pc2, quantized_pc3))
                pointCloud.pcerror(ori_file, whole_q_pc, None, '-r ' + peak, tmp_test_file)
                out_files = [out_file, out_file2, out_file3]
                return out_files, whole_pc, pointCloud.distChamfer(whole_pc, whole_q_pc), bin_num, z_offset, get_psnr(tmp_test_file)[0]

    def __len__(self):
        return len(self.test_files)
