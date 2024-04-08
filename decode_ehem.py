import time
import math
import os
import numpy as np
import torch
import argparse
from pathlib import Path
from hydra import initialize, compose
from tqdm import tqdm
from collections import deque

from data_preproc.Octree import DeOctree, dec2bin, dec2binAry, dec2bin_ary_torch
from data_preproc import pt
from data_preproc.data_preprocess import spher2cart, cylin2cart
from models import EHEM
import numpyAc


# TODO this is too stupid
def extract_info(file):
    spher = 'spher' in file
    cylin = 'cylin' in file
    rtn = list(map(lambda x: int(x), file.split('.')[0].split('_')[-3:]))
    if spher or cylin:
        pos_mm = torch.load(file+'.dat')
        return [spher, cylin, pos_mm[:, 1]] + rtn
    return [spher, cylin, []] + rtn


def cal_pos(parent_pos, i, cur_level, max_level):
    pos = torch.zeros_like(parent_pos)
    parent_pos = parent_pos * (2 ** max_level)
    parent_pos = torch.round(parent_pos).long()
    xyz = dec2bin(i, count=3)
    unit = 2 ** (max_level - cur_level + 1)
    for i in range(3):
        pos[i] = (xyz[i] * unit + parent_pos[i]) / (2 ** max_level)
    return pos


def cal_pos_ary(parent_pos, i, cur_level, max_level, pre_pos_max=-1, pos_max=-1):
    pos = torch.zeros_like(parent_pos)
    if pre_pos_max == -1:
        pre_pos_max = torch.Tensor([2 ** max_level - 1e-9]).to(parent_pos.device)
    parent_pos = parent_pos * (pre_pos_max.double() + 1e-9)
    parent_pos = torch.round(parent_pos).long()
    xyz = dec2bin_ary_torch(i-1, bits=3).reshape([-1, 3])
    unit = 2 ** (max_level - cur_level + 1)
    if pos_max == -1:
        pos_max = torch.Tensor([2 ** max_level - 1e-9]).to(parent_pos.device)
    for i in range(3):
        pos[:, i] = (xyz[:, i] * unit + parent_pos[:, i]) / (pos_max.double() + 1e-9)
    return pos


def decodeOct(binfile, oct_data_seq, model, context_size, anc_k):
    """
    description: decode bin file to occupancy code
    param {str;input bin file name} binfile
    param {N*1 array; occupancy code, only used for check} oct_data_seq
    param {model} model
    param {int; Context window length}context_size
    return {N*1,float}occupancy code,time
    """
    spher, cylin, pos_mm, max_level, bin_num, z_offset = extract_info(binfile)
    oct_len = len(oct_data_seq)
    dec = numpyAc.arithmeticDeCoding(None, oct_len, 255, binfile)

    model.eval()
    with torch.no_grad():

        pre_nodes = []
        pre_poses = []
        cur_nodes = []
        cur_poses = []
        oct_seq = []

        ipt = torch.zeros((1, anc_k, 3)).long().cuda()
        ipt[0, :, 2] = 255
        ipt[0, -1, :2] = 1
        ipt_pos = torch.zeros((3, 1)).cuda()

        elapsed = time.time()
        output = model.decode(ipt[True], ipt_pos[True])
        elapsed = time.time() - elapsed
        freqsinit = torch.softmax(output[0, 0], 0).cpu().numpy()

        root = dec.decode(freqsinit[True])

        # the node ids of 2-group points
        node_id = 1
        cur_level = 1

        ipt[-1, -1, 2] = root
        cur_nodes.append(ipt[-1, -(anc_k - 1):].clone())
        cur_poses.append(ipt_pos[:, 0].clone())
        oct_seq.append(root)  # decode the root
        rest = 0
        rest_ipt = torch.empty((0, ipt.shape[1], ipt.shape[2])).long().cuda()
        rest_pos = torch.empty((3, 0)).cuda()

        with tqdm(total=oct_len) as pbar:
            while True:
                if not rest and not pre_nodes:
                    pre_nodes = cur_nodes
                    pre_poses = cur_poses
                    cur_nodes = []
                    cur_poses = []
                    cur_level += 1
                if pre_nodes and rest < context_size:
                    # take all or context_size
                    # nodes
                    ancients = torch.stack(pre_nodes[:context_size-rest])
                    pre_nodes = pre_nodes[context_size-rest:]

                    child_occ = dec2bin_ary_torch(ancients[:, -1, 2] + 1)
                    child_occ = child_occ.reshape([-1, 8]).flip(1)
                    for i in range(1, 8):
                        child_occ[:, i] *= i + 1
                    child_occ = child_occ.reshape(-1)
                    child_idx = torch.where(child_occ != 0)[0]
                    child_occ = child_occ[child_idx]

                    ancients = ancients.repeat(8, 1, 1, 1).transpose(1, 0).reshape([-1, 3, 3])
                    ancients = ancients[child_idx]

                    parent_pos = torch.stack(pre_poses[:context_size])
                    pre_poses = pre_poses[context_size-rest:]
                    parent_pos = parent_pos.repeat(8, 1, 1).transpose(1, 0).reshape([-1, 3])
                    parent_pos = parent_pos[child_idx]

                    cur_feat = torch.zeros([child_occ.shape[0], 3]).cuda()
                    cur_feat[:, 0] = cur_level
                    cur_feat[:, 1] = child_occ
                    cur_feat[:, 2] = 255
                    if len(pos_mm) != 0:
                        cur_pos = cal_pos_ary(parent_pos, child_occ, cur_level, max_level, pos_mm[cur_level-2], pos_mm[cur_level-1])
                    else:
                        cur_pos = cal_pos_ary(parent_pos, child_occ, cur_level, max_level)

                    # add point into context
                    ipt = torch.hstack((ancients, cur_feat[:, True]))
                    ipt = torch.vstack((rest_ipt, ipt))
                    ipt_pos = cur_pos.transpose(1, 0)
                    ipt_pos = torch.hstack((rest_pos, ipt_pos))
                else:
                    ipt = rest_ipt
                    ipt_pos = rest_pos

                rest_ipt = ipt[context_size:]
                ipt = ipt[:context_size]
                rest_pos = ipt_pos[:, context_size:]
                ipt_pos = ipt_pos[:, :context_size]
                rest = rest_ipt.shape[0]

                start = time.time()
                prob1 = model.decode(ipt[True].long(), ipt_pos[True])
                elapsed += time.time() - start
                probabilities1 = torch.softmax(prob1[0], 1).cpu().numpy()
                nodes1 = dec.decode_ehem(probabilities1)
                nodes1 = torch.Tensor(nodes1).long().cuda()

                start = time.time()
                prob2 = model.decode(ipt[True].long(), ipt_pos[True], nodes1[True])
                elapsed += time.time() - start
                probabilities2 = torch.softmax(prob2[0], 1).cpu().numpy()
                nodes2 = dec.decode_ehem(probabilities2)
                nodes2 = torch.Tensor(nodes2).long().cuda()

                nodes12 = torch.zeros(ipt.shape[0]).cuda().long()

                nodes12[::2] = nodes1
                nodes12[1::2] = nodes2

                ipt[:, -1, -1] = nodes12
                cur_nodes += ipt[:, 1:].clone().long().reshape(-1, 3).chunk(ipt.shape[0])
                cur_poses += ipt_pos.transpose(1, 0).clone().reshape(-1).chunk(ipt_pos.shape[1])

                cur_label = oct_data_seq[node_id:node_id+ipt.shape[0], 0]

                node_id += ipt.shape[0]
                nodes12 = nodes12.detach().cpu().numpy()
                oct_seq += nodes12.tolist()
                assert (cur_label == nodes12 + 1).sum() == cur_label.shape[0]
                if node_id >= oct_len:
                    return oct_seq, bin_num, z_offset, elapsed, spher, cylin

                pbar.update(ipt.shape[0])


def main(args):
    root_path = args.ckpt_path.split("ckpt")[0]
    test_output_path = (
        root_path + "test_output" + args.ckpt_path.split("ckpt")[1][:-1] + "/"
    )
    cfg_path = Path(root_path, ".hydra")
    initialize(config_path=str(cfg_path))
    cfg = compose(config_name="config.yaml")

    model = EHEM.load_from_checkpoint(args.ckpt_path, cfg=cfg).cuda()
    if os.path.isdir(args.test_files[0]):
        args.test_files = list(filter(lambda x: x.endswith('.ply'), list(map(lambda x: args.test_files[0] + x, os.listdir(args.test_files[0])))))
    elapsed = 0
    for i, ori_file in enumerate(args.test_files):
        print(f'{i}/{len(args.test_files)}')
        ori_file = Path(ori_file)
        for file in os.listdir(test_output_path):
            if ori_file.stem in file and file.endswith('.bin'):
                binfile = test_output_path + file
                break

        # load ori data
        npy_path = str(ori_file).rsplit(".")[0]
        if args.preproc_path:
            npy_path = args.preproc_path + '/' + npy_path.split('/')[-1]
        oct_data_seq = np.load(npy_path + ".npy")[:, -1:, 0]
        lidar_level = int(binfile.split('/')[-1].split('_')[-3]) # TODO check for every exp

        code, bin_num, z_offset, t, spher, cylin = decodeOct(
            binfile, oct_data_seq, model, cfg.model.context_size, cfg.model.level_k
        )
        elapsed += t
        print("decode succeeded, time:", t)
        print("oct len:", len(code))
        print("avg dec time:", elapsed / (i+1))

        # DeOctree
        # TODO test this code
        pt_rec = DeOctree(np.array(code) + 1)

        if 'ford' in binfile:
            qs = 2**(18-lidar_level)
        else:
            qs = 400/(2**lidar_level-1)

        if spher:
            qs = np.array([qs, 2*math.pi / (bin_num-1), math.pi / (bin_num - 1)])[True]
            offset = [0, 0, 0]
        elif cylin:
            qs = np.array([qs, 2*math.pi / (bin_num-1), qs])[True]
            offset = [0, 0, z_offset]
        else:
            qs = np.array([qs])
            offset = -200
        pt_rec = pt_rec * qs
        pt_rec[:] += offset
        if spher:
            pt_rec = spher2cart(pt_rec)
        elif cylin:
            pt_rec = cylin2cart(pt_rec)
        pt.write_ply_data(test_output_path + ori_file.stem + ".ply", pt_rec)
        print(test_output_path + ori_file.stem + ".ply")
    print(elapsed / len(args.test_files))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="example: outputs/obj/2023-04-28/10-43-45/ckpt/epoch=7-step=64088.ckpt",
    )
    parser.add_argument(
        "--test_files",
        nargs="*",
        default=["data/obj/mpeg/8iVLSF_910bit/boxer_viewdep_vox9.ply"],
    )
    parser.add_argument("--preproc_path", type=str, default="")
    parser.add_argument("--sequential_enc", action="store_true")
    parser.add_argument("--level_wise", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
