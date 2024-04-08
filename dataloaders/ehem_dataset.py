import numpy as np
import glob
import torch.utils.data as data
import torch
import glob


class EHEMDataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.file_names = []
        self.total_point_num = 0
        for filename in sorted(glob.glob(cfg.root)):
            self.file_names.append('{}'.format(filename))
            self.total_point_num += int(filename.split('_')[-1].split('.')[0])
        self.root = cfg.root
        self.index = 0
        self.datalen = 0
        self.dataBuffer = []
        self.fileIndx = 0
        # self.tree_point_num = cfg.tree_size * cfg.context_size
        self.context_size = cfg.context_size
        assert self.file_names, 'no file found!'
        # self.max_time_each_file = self.total_point_num//(self.context_size*len(self.file_names))
        self.max_time_each_file = 0
        self.cur_times = self.max_time_each_file
        self.cur_max_level = 0

    def __getitem__(self, index):
        if self.cur_times >= self.max_time_each_file:
            file_idx = index % len(self.file_names) # randomly select file
            self.cur_data = np.load(self.file_names[file_idx])

            # this change the occupancy range from [1, 255] to [0, 254]
            self.cur_data[:, :, 0] -= 1
            self.cur_max_level = max(self.cur_data[:, -1, 1])

            self.cur_times = 0
            cur_file_len = int(self.file_names[file_idx].split('_')[-1].split('.')[0])
            self.max_time_each_file = cur_file_len // self.context_size
            self.order = torch.randperm(self.max_time_each_file)

        cur_idx = self.order[self.cur_times]
        data = np.copy(self.cur_data[cur_idx*self.context_size:cur_idx*self.context_size + self.context_size])
        pos = data[:, -1, 3:6]
        pos_max, pos_min = pos.max(), pos.min()
        pos = ((pos-pos_min) / (pos_max-pos_min)).astype(np.float32).transpose((1, 0))
        if self.cfg.extra_pos:
            xyz_pos = data[:, -1, 6:9]
            pos_max, pos_min = xyz_pos.max(), xyz_pos.min()
            xyz_pos = ((xyz_pos-pos_min) / (pos_max-pos_min)).astype(np.float32).transpose((1, 0))

        data = data[:, :, :3]
        data = np.concatenate((data[:, :, 1:], data[:, :, :1]), axis=2)
        label = np.copy(data[:, -1, 2])

        self.cur_times += 1

        if self.cfg.extra_pos:
            return data, pos, xyz_pos, label # data: level, octant, occ
        else:
            return data, pos, label # data: level, octant, occ

    def __len__(self):
        return self.total_point_num//self.context_size
