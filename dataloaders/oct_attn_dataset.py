import numpy as np
import glob
import torch.utils.data as data
import glob


class OctAttnDataset(data.Dataset):
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

        data = np.copy(self.cur_data[self.cur_times*self.context_size:self.cur_times*self.context_size + self.context_size])
        pos = (data[:, :, 3:] / (2**self.cur_max_level)).astype(np.float32)
        data = data[:, :, :3]
        label = np.copy(data[:, -1, 0])

        self.cur_times += 1

        return data, pos, label

    def __len__(self):
        return self.total_point_num//self.context_size
