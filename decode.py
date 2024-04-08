import time
import numpy as np
import torch
import argparse
from pathlib import Path
from hydra import initialize, compose
from tqdm import tqdm
from collections import deque

from data_preproc.Octree import DeOctree, dec2bin
from data_preproc import pt
from models import OctAttention
import numpyAc


def extract_max_level(file):
    if file[-5] == "9":
        return 9
    return int(file[-6:-4])


def cal_pos(parent_pos, i, cur_level, max_level):
    pos = torch.zeros_like(parent_pos)
    parent_pos = parent_pos * (2 ** max_level)
    parent_pos = torch.round(parent_pos).long()
    xyz = dec2bin(i, count=3)
    unit = 2 ** (max_level - cur_level + 1)
    for i in range(3):
        pos[i] = (xyz[i] * unit + parent_pos[i]) / (2 ** max_level)
    return pos


def decodeOct(binfile, oct_data_seq, model, context_size, level_k):
    """
    description: decode bin file to occupancy code
    param {str;input bin file name} binfile
    param {N*1 array; occupancy code, only used for check} oct_data_seq
    param {model} model
    param {int; Context window length}context_size
    return {N*1,float}occupancy code,time
    """
    model.eval()
    oct_data_seq -= 1
    max_level = extract_max_level(binfile)
    cur_level = 1

    with torch.no_grad():
        elapsed = time.time()

        nodeQ = deque()
        posQ = deque()
        oct_seq = []
        oct_len = len(oct_data_seq)

        ipt = torch.zeros((context_size, level_k, 3)).long().cuda()
        ipt[:, :, 0] = 255
        ipt[-1, -1, 1:3] = 1
        ipt_pos = torch.zeros((context_size, level_k, 3)).cuda()

        output = model(ipt[True], ipt_pos[True])
        freqsinit = torch.softmax(output[0, -1], 0).cpu().numpy()

        dec = numpyAc.arithmeticDeCoding(None, oct_len, 255, binfile)

        root = decodeNode(freqsinit, dec)
        node_id = 0

        ipt[-1, -1, 0] = root
        nodeQ.append(ipt[-1, -(level_k - 1):].clone())
        posQ.append(ipt_pos[-1, -(level_k - 1):].clone())
        oct_seq.append(root)  # decode the root

        with tqdm(total=oct_len) as pbar:
            while True:
                ancients = nodeQ.popleft()
                ancient_pos = posQ.popleft()
                parent_pos = ancient_pos[-1]

                childOcu = dec2bin(ancients[-1, 0] + 1)
                childOcu.reverse()
                cur_level = ancients[-1][1] + 1
                # TODO level变了要清空
                for i in range(8):
                    if childOcu[i]:
                        cur_feat = torch.vstack((ancients, torch.Tensor([[255, cur_level, i + 1]]).cuda()))
                        cur_pos = cal_pos(parent_pos, i, cur_level, max_level)
                        cur_pos = torch.vstack((ancient_pos.clone(), cur_pos))

                        # shift context_size window
                        ipt[:-1] = ipt[1:].clone()
                        ipt[-1] = cur_feat
                        ipt_pos[:-1] = ipt_pos[1:].clone()
                        ipt_pos[-1] = cur_pos

                        output = model(ipt[True], ipt_pos[True])
                        probabilities = torch.softmax(output[0, -1], 0).cpu().numpy()
                        root = decodeNode(probabilities, dec)

                        node_id += 1
                        pbar.update(1)

                        ipt[-1, -1, 0] = root
                        nodeQ.append(ipt[-1, 1:].clone())
                        posQ.append(ipt_pos[-1, 1:].clone())
                        if node_id == oct_len:
                            return oct_seq, time.time() - elapsed
                        oct_seq.append(root)
                        assert oct_data_seq[node_id] == root  # for check


def decodeNode(pro, dec):
    root = dec.decode(np.expand_dims(pro, 0))
    return root


def main(args):
    root_path = args.ckpt_path.split("ckpt")[0]
    test_output_path = (
        root_path + "test_output" + args.ckpt_path.split("ckpt")[1][:-1] + "/"
    )
    cfg_path = Path(root_path, ".hydra")
    initialize(config_path=str(cfg_path))
    cfg = compose(config_name="config.yaml")

    model = OctAttention.load_from_checkpoint(args.ckpt_path, cfg=cfg).cuda()
    for ori_file in args.test_files:
        ori_file = Path(ori_file)
        binfile = test_output_path + ori_file.stem + ".bin"

        # load ori data
        npy_path = str(ori_file).rsplit(".")[0]
        oct_data_seq = np.load(npy_path + ".npy")[:, -1:, 0]

        code, elapsed = decodeOct(
            binfile, oct_data_seq, model, cfg.model.context_size, cfg.model.level_k
        )
        print("decode succee,time:", elapsed)
        print("oct len:", len(code))

        # DeOctree
        pt_rec = DeOctree(code)
        # Dequantization
        # TODO fit for lidar dataset
        offset = 0
        qs = 1
        pt_rec = pt_rec * qs + offset
        pt.write_ply_data(test_output_path + ori_file.stem + ".ply", pt_rec)


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
    parser.add_argument("--sequential_enc", action="store_true")
    parser.add_argument("--level_wise", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
