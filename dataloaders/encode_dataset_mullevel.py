import os
import uuid
import numpy as np
import torch.utils.data as data
from data_preproc.data_preprocess import proc_pc, MVUB_NAMES
from pathlib import Path
import data_preproc.pt as pointCloud
from utils import get_psnr


class EncodeDataset(data.Dataset):
    """ImageFolder can be used to load images where there are no labels."""

    def __init__(self, test_files, context_size, data_type, level_wise=True, lidar_level=12, spher=False, preproc_path=''):
        self.test_files = test_files
        self.context_size = context_size
        self.data_type = data_type
        self.level_wise = level_wise
        self.lidar_level = lidar_level
        self.preproc_path = preproc_path
        if not self.preproc_path:
            raise Exception('no preproc_path!')
        self.spher = spher
        if not os.path.exists('temp'):
            os.mkdir('temp')

    def __getitem__(self, index):
        ori_file = self.test_files[index]
        if self.spher:
            npy_paths, pt, chamfer, bin_num, psnr = self.preproc(ori_file)
        else:
            npy_paths, pt, chamfer, psnr = self.preproc(ori_file)
            bin_num = 0

        ids, pos, data, oct_seq = self.get_data(npy_paths[0])

        for path in npy_paths[1:]:
            cur_ids, cur_pos, cur_data, cur_oct_seq = self.get_data(path)
            ids += cur_ids
            pos += cur_pos
            data += cur_data
            oct_seq = np.vstack((oct_seq, cur_oct_seq))
        return ids, pos, data, oct_seq, len(pt), bin_num, chamfer, psnr

    def get_data(self, npy_path):
        oct_seq = np.load(npy_path + ".npy")
        padding = np.zeros([self.context_size - 1, oct_seq.shape[1], oct_seq.shape[2]]).astype(np.int64)
        padding[:, :, 0] = 255
        ids_pad = np.ones([self.context_size - 1]).astype(np.int64) * -1

        oct_seq[:, :, 0] -= 1
        whole_ids = np.arange(len(oct_seq)).astype(np.int64)
        max_level = max(oct_seq[:, -1, 1])
        data = []
        pos = []
        ids = []
        cur_level = 1 if self.level_wise else 100
        cur_level_start = 0
        for i in range(len(oct_seq)):
            if oct_seq[i, -1, 1] > cur_level:
                data.append(np.vstack((padding[:, :, :3], oct_seq[cur_level_start:i, :, :3])))
                pos.append(np.vstack((padding[:, :, 3:].astype(np.float32), (oct_seq[cur_level_start:i, :, 3:] / (2**max_level)).astype(np.float32))))
                ids.append(np.hstack((ids_pad, whole_ids[cur_level_start:i] - cur_level_start)))
                cur_level_start = i
                cur_level = oct_seq[i, -1, 1]
        data.append(np.vstack((padding[:, :, :3], oct_seq[cur_level_start:, :, :3])))
        pos.append(np.vstack((padding[:, :, 3:].astype(np.float32), (oct_seq[cur_level_start:, :, 3:] / (2**max_level)).astype(np.float32))))
        ids.append(np.hstack((ids_pad, whole_ids[cur_level_start:] - cur_level_start)))
        return ids, pos, data, oct_seq
    
    def preproc(self, ori_file):
        ori_path = Path(ori_file)
        out_file = ori_path.parent / ori_path.stem / Path(".npy")
        if out_file.exists():
            return str(out_file)

        if self.data_type == 'kitti':
            peak = '59.70'
        elif self.data_type == 'ford':
            peak = '30000'

        tmp_test_file = "temp/pcerror_results" + str(uuid.uuid4()) + ".txt"
        if self.data_type == 'obj':
            rotate = False
            for mvub_name in MVUB_NAMES:
                if mvub_name in ori_file:
                    rotate = True
                    break
            return proc_pc(
                ori_file, ori_path.parent, ori_path.stem, rotation=rotate, test=True
            )[:2] + [0.]
        elif self.spher: # TODO: complete the testing code for OctAttention
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
                raise Exception('no preproc_path!')
        else:
            out_file, dq_pt, norm_pt = proc_pc(
                ori_file,
                ori_path.parent,
                ori_path.stem,
                offset=-200,
                normalize=False,
                qs=400/(2**self.lidar_level - 1),
                test=True,
                )
            pointCloud.pcerror(norm_pt, dq_pt, None, '-r ' + peak, tmp_test_file)
            return out_file, norm_pt, pointCloud.distChamfer(norm_pt, dq_pt), get_psnr(tmp_test_file)[0]

    def __len__(self):
        return len(self.test_files)
