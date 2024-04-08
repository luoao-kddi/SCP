'''
DGCNN is the feature extractor for point cloud positions in EHEM
'''

import torch
import torch.nn as nn
import math


def knn(x, k):
    # inner = -2*torch.matmul(x.transpose(2, 1), x)
    # xx = torch.sum(x**2, axis=1, keepdim=True)
    # pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    # return idx
    if x.shape[2] <= 8192:
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -inner - xx - xx.transpose(2, 1)
    
        # topk has strange bug if we directly calculate all the topk together, causing errors.
        idxs = []
        for i in range(math.ceil(pairwise_distance.shape[0] / 4)):
            idx = pairwise_distance[4*i:4*i+4].topk(k=k, dim=-1)[1]
            idxs.append(idx)
        idx = torch.concat(idxs, 0)
        return idx
    else:
        xx = torch.sum(x**2, dim=1, keepdim=True)
        interval = 4096
        all_idxs = []
        for i in range(math.ceil(x.shape[0] / 4)):
            idxs = []
            for j in range(math.ceil(x.shape[2] / interval)):
                inner = -2*torch.matmul(x.transpose(2, 1)[4*i:4*i+4, interval*j:interval*j+interval], x)
                pairwise_distance = -inner - xx[4*i:4*i+4] - xx.transpose(2, 1)[4*i:4*i+4, interval*j:interval*j+interval]
                idx = pairwise_distance.topk(k=k, dim=-1)[1]
                idxs.append(idx)
            sample_idxs = torch.concat(idxs, 1)
            all_idxs.append(sample_idxs)

        idx = torch.concat(all_idxs, 0)
        idx = idx.reshape(x.shape[0], x.shape[2], -1)
        return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class GeoFeatGenerator(nn.Module):
    def __init__(self, k=20, max_level=17):
        super(GeoFeatGenerator, self).__init__()
        self.k = k
        # TODO make sure whether bn should be used
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d((64+80)*2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d((128+64)*2, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.occ_enc = nn.Embedding(256, 16) # TODO former method is 128, temporarily set to 16 here
        self.level_enc = nn.Embedding(max_level, 4)
        self.octant_enc = nn.Embedding(9, 4)
        self.mlp2 = nn.Sequential(
            nn.Linear(80, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
        )
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(448, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x, pos):
        bsz, csz = x.shape[:2]
        occ = x[:, :, 2::3]
        level = x[:, :, ::3]
        octant = x[:, :, 1::3]
        occ_embed = self.occ_enc(occ).reshape(bsz, csz, -1)
        level_embed = self.level_enc(level).reshape(bsz, csz, -1)
        octant_embed = self.octant_enc(octant).reshape(bsz, csz, -1)
        x = torch.concat((occ_embed, level_embed, octant_embed), 2)

        k = min(self.k, pos.shape[2])
        pos = get_graph_feature(pos, k=k)
        pos = self.conv1(pos)
        pos1 = pos.max(dim=-1, keepdim=False)[0]

        pos = get_graph_feature(torch.concat((pos1, x.transpose(1, 2)), 1), k=k) # TODO check the dim of concat
        pos = self.conv2(pos)
        pos2 = pos.max(dim=-1, keepdim=False)[0]

        x = self.mlp2(x)

        pos = get_graph_feature(torch.concat((pos2, x.transpose(1, 2)), 1), k=k)
        pos = self.conv3(pos)
        pos3 = pos.max(dim=-1, keepdim=False)[0]

        x = self.mlp3(x)

        ec = self.edge_mlp1(torch.concat((pos1, pos2, pos3), 1).transpose(1, 2))
        ec = self.edge_mlp2(torch.concat((pos3.transpose(1, 2), ec), 2))

        return torch.concat((x, ec), 2) # output dim: 256

    def embed_occ(self, occ):
        return self.occ_enc(occ)

class GeoFeatGeneratorBYB(GeoFeatGenerator):
    def __init__(self, k=20):
        super(GeoFeatGeneratorBYB, self).__init__(k)
        self.occ_enc = nn.Embedding(34, 8)

    def forward(self, x, pos):
        bsz, csz = x.shape[:2]
        occ1 = x[:, :, 2::4]
        occ2 = x[:, :, 3::4]
        level = x[:, :, ::4]
        octant = x[:, :, 1::4]
        occ1_embed = self.occ_enc(occ1).reshape(bsz, csz, -1)
        occ2_embed = self.occ_enc(occ2).reshape(bsz, csz, -1)
        level_embed = self.level_enc(level).reshape(bsz, csz, -1)
        octant_embed = self.octant_enc(octant).reshape(bsz, csz, -1)
        x = torch.concat((occ1_embed, occ2_embed, level_embed, octant_embed), 2)

        pos = get_graph_feature(pos, k=self.k)
        pos = self.conv1(pos)
        pos1 = pos.max(dim=-1, keepdim=False)[0]

        pos = get_graph_feature(torch.concat((pos1, x.transpose(1, 2)), 1), k=self.k) # TODO check the dim of concat
        pos = self.conv2(pos)
        pos2 = pos.max(dim=-1, keepdim=False)[0]

        x = self.mlp2(x)

        pos = get_graph_feature(torch.concat((pos2, x.transpose(1, 2)), 1), k=self.k)
        pos = self.conv3(pos)
        pos3 = pos.max(dim=-1, keepdim=False)[0]

        x = self.mlp3(x)

        ec = self.edge_mlp1(torch.concat((pos1, pos2, pos3), 1).transpose(1, 2))
        ec = self.edge_mlp2(torch.concat((pos3.transpose(1, 2), ec), 2))

        return torch.concat((x, ec), 2) # output dim: 256


class GeoFeatGenerator2coord(GeoFeatGenerator):
    def __init__(self, k=20):
        super().__init__(k)
        self.conv12 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv22 = nn.Sequential(nn.Conv2d((64+80)*2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv32 = nn.Sequential(nn.Conv2d((128+64)*2, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.edge_mlp12 = nn.Sequential(
            nn.Linear(448, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )
        self.edge_mlp22 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x, pos, xyz_pos):
        bsz, csz = x.shape[:2]
        occ = x[:, :, 2::3]
        level = x[:, :, ::3]
        octant = x[:, :, 1::3]
        occ_embed = self.occ_enc(occ).reshape(bsz, csz, -1)
        level_embed = self.level_enc(level).reshape(bsz, csz, -1)
        octant_embed = self.octant_enc(octant).reshape(bsz, csz, -1)
        x = torch.concat((occ_embed, level_embed, octant_embed), 2)

        k = min(self.k, pos.shape[2])
        pos = get_graph_feature(pos, k=k)
        pos = self.conv1(pos)
        pos1 = pos.max(dim=-1, keepdim=False)[0]

        k = min(self.k, xyz_pos.shape[2])
        xyz_pos = get_graph_feature(xyz_pos, k=k)
        xyz_pos = self.conv12(xyz_pos)
        xyz_pos1 = xyz_pos.max(dim=-1, keepdim=False)[0]

        pos = get_graph_feature(torch.concat((pos1, x.transpose(1, 2)), 1), k=k) # TODO check the dim of concat
        pos = self.conv2(pos)
        pos2 = pos.max(dim=-1, keepdim=False)[0]

        xyz_pos = get_graph_feature(torch.concat((xyz_pos1, x.transpose(1, 2)), 1), k=k) # TODO check the dim of concat
        xyz_pos = self.conv22(xyz_pos)
        xyz_pos2 = xyz_pos.max(dim=-1, keepdim=False)[0]

        x = self.mlp2(x)

        pos = get_graph_feature(torch.concat((pos2, x.transpose(1, 2)), 1), k=k)
        pos = self.conv3(pos)
        pos3 = pos.max(dim=-1, keepdim=False)[0]

        xyz_pos = get_graph_feature(torch.concat((xyz_pos2, x.transpose(1, 2)), 1), k=k)
        xyz_pos = self.conv32(xyz_pos)
        xyz_pos3 = xyz_pos.max(dim=-1, keepdim=False)[0]

        x = self.mlp3(x)

        ec = self.edge_mlp1(torch.concat((pos1, pos2, pos3), 1).transpose(1, 2))
        ec = self.edge_mlp2(torch.concat((pos3.transpose(1, 2), ec), 2))

        xyz_ec = self.edge_mlp12(torch.concat((xyz_pos1, xyz_pos2, xyz_pos3), 1).transpose(1, 2))
        xyz_ec = self.edge_mlp22(torch.concat((xyz_pos3.transpose(1, 2), xyz_ec), 2))

        return torch.concat((x, ec, xyz_ec), 2) # output dim: 256
