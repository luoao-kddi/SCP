import os
import glob
import argparse
from pathlib import Path
import numpy as np
import time
import numpyAc
import tqdm
from hydra import initialize, compose
import torch
from torch.utils.data import DataLoader
from dataloaders.encode_dataset_mullevel import EncodeDataset
from dataloaders.encode_dataset_ehem_mullevel import EncodeEHEMDataset
from models import *


def encodeNode(pro, octvalue):
    assert octvalue <= 254 and octvalue >= 0
    pre = np.argmax(pro)
    return -np.log2(pro[octvalue] + 1e-07), int(octvalue == pre)


def compress(batch, outputfile, model, args):
    model.eval()

    context_size = model.cfg.model.context_size

    ids, pos, data, oct_seq, pt_num, bin_num = batch
    pt_num = int(pt_num)
    bin_num = int(bin_num)

    oct_seq = oct_seq[0, :, -1, 0].int()
    oct_len = len(oct_seq)

    elapsed = 0
    proBit = []
    with torch.no_grad():
        # set interval
        if args.sequential:
            interval = 1
        else:
            interval = context_size

        for l, (level_data, level_pos, level_ids) in enumerate(zip(data, pos, ids)):
            level_data, level_pos, level_ids = level_data.cuda(), level_pos.cuda(), level_ids.cuda()
            probabilities = torch.zeros((level_data.shape[1], model.cfg.model.token_num)).cuda()
            for i in tqdm.trange(0, level_data.shape[1], interval, desc=f"level {l} of {len(data)}"):
                ipt_data = level_data[:, i : i + context_size].long()
                ipt_pos = level_pos[:, i : i + context_size]
                node_id = level_ids[0, i : i + context_size]
                start_time = time.time()
                output = model(ipt_data, ipt_pos)
                elapsed = elapsed + time.time() - start_time

                p = torch.softmax(output, 2)
                if args.sequential:
                    probabilities[node_id[-1], :] = p[0, -1]
                else:
                    probabilities[node_id, :] = p[0]
            proBit.append(probabilities.detach().cpu().numpy()[:-1023])

    proBit = np.vstack(proBit)

    # entropy coding
    codec = numpyAc.arithmeticCoding()
    if args.spher:
        outputfile += '_spher'
    elif args.cylin:
        outputfile += '_cylin'
    outputfile += '_' + str(len(data)) + '_' + str(bin_num) + '_' + str(0) + '.bin'
    if not os.path.exists(os.path.dirname(outputfile)):
        os.makedirs(os.path.dirname(outputfile))
    _, real_rate = codec.encode(
        proBit[:oct_len, :], oct_seq.numpy().astype(np.int16), outputfile
    )

    np.set_printoptions(formatter={"float": "{: 0.4f}".format})
    print("outputfile                  :", outputfile)
    print("time(s)                     :", elapsed)
    print("pt num                      :", pt_num)
    print("oct num                     :", oct_len)
    print("total binsize               :", real_rate)
    print("bit per oct                 :", real_rate / oct_len)
    print("bit per pixel               :", real_rate / pt_num)
    return real_rate / pt_num, elapsed


def compress_ehem(batch, outputfile, model, args):
    model.eval()

    context_size = model.cfg.model.context_size

    ids, pos, pos_mm, data, oct_seq, pt_num, pc, bin_num, z_offset = batch
    pt_num = int(pt_num)
    oct_seq = oct_seq[0, :, -1, 0].int()
    oct_len = len(oct_seq)

    elapsed = 0
    proBit = []
    with torch.no_grad():
        # set interval
        interval = context_size
        coding_order = []
        coded_cnt = 0

        for l, (level_data, level_pos, level_ids) in enumerate(zip(data, pos, ids)):
            level_data, level_pos, level_ids = level_data.cuda(), level_pos.cuda(), level_ids.cuda()
            probabilities = torch.zeros((level_data.shape[1], model.cfg.model.token_num)).cuda()
            for i in tqdm.trange(0, level_data.shape[1], interval, desc=f"level {l} of {len(data)}"):
                ipt_data = level_data[:, i : i + context_size].long()
                ipt_pos = level_pos[:, :, i : i + context_size]
                node_id = level_ids[0, i : i + context_size]
                start_time = time.time()
                output1, output2 = model(ipt_data, ipt_pos, enc=True)
                elapsed = elapsed + time.time() - start_time

                if len(probabilities) == 1:
                    # level 0
                    probabilities[0] = torch.softmax(output1[:, -1], 1)
                    coding_order.append(node_id[-1:].detach().cpu().numpy() + coded_cnt)
                    continue

                p1 = torch.softmax(output1, 2)
                p2 = torch.softmax(output2, 2)
                probabilities[node_id[::2], :] = p1[0]
                probabilities[node_id[1::2], :] = p2[0]
                coding_order.append(node_id[::2].detach().cpu().numpy() + coded_cnt)
                coding_order.append(node_id[1::2].detach().cpu().numpy() + coded_cnt)
            coded_cnt += int(level_data.shape[1])
            proBit.append(probabilities.detach().cpu().numpy())

    proBit = np.vstack(proBit)
    coding_order = np.concatenate(coding_order)

    # entropy coding
    codec = numpyAc.arithmeticCoding()
    if args.spher:
        outputfile += '_spher'
    elif args.cylin:
        outputfile += '_cylin'
    outputfile += '_' + str(len(data)) + '_' + str(int(bin_num)) + '_' + str(int(z_offset)) + '.bin'
    if not os.path.exists(os.path.dirname(outputfile)):
        os.makedirs(os.path.dirname(outputfile))
    _, real_rate = codec.encode(
        proBit[coding_order], oct_seq.numpy().astype(np.int16)[coding_order], outputfile
    )
    torch.save(torch.Tensor(pos_mm), outputfile + '.dat')

    np.set_printoptions(formatter={"float": "{: 0.4f}".format})
    print("outputfile                  :", outputfile)
    print("time(s)                     :", elapsed)
    print("pt num                      :", pt_num)
    print("oct num                     :", oct_len)
    print("total binsize               :", real_rate)
    print("bit per oct                 :", real_rate / oct_len)
    print("bit per pixel               :", real_rate / pt_num)
    return real_rate / pt_num, elapsed


def main(args):
    # load ckpt config
    root_path = args.ckpt_path.split("ckpt")[0]
    test_output_path = (
        root_path + "test_output" + args.ckpt_path.split("ckpt")[1][:-1] + "/"
    )
    cfg_path = Path(root_path, ".hydra")
    initialize(config_path=str(cfg_path))
    cfg = compose(config_name="config.yaml")

    model_name = cfg.model.class_name
    if model_name == "OctAttention":
        model_class = OctAttention
    elif model_name == 'EHEM' or model_name == 'EHEMVoxel':
        model_class = EHEM
        args.level_wise = True # EHEM need to be level-wise enc/dec
    else:
        raise NotImplementedError("Not implemented model: ", model_name)
    model = model_class.load_from_checkpoint(args.ckpt_path, cfg=cfg).cuda()

    test_files = args.test_files
    combine_results = False
    if '*' in test_files[0]:
        test_files = glob.glob(test_files)
        # calculate averaged results if input is a directory
        combine_results = True

    if 'EHEM' in model_name:
        testset = EncodeEHEMDataset(test_files, model.cfg.model.context_size, args.type, args.level_wise, args.lidar_level, args.cylin, args.spher, args.preproc_path)
    else:
        testset = EncodeDataset(test_files, model.cfg.model.context_size, args.type, args.level_wise, args.lidar_level, args.spher, args.preproc_path)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    bpps = []
    times = []
    psnr = []
    chamfer = []
    print("Encoding with", model_name)
    for i, (cur_file, batch) in enumerate(zip(test_files, test_loader)):
        print("Encoding ", cur_file, i, '/', len(test_files))
        if 'EHEM' in model_name:
            if args.type == 'kitti':
                cur_file = cur_file.split('/')[-2] + cur_file.split('/')[-1].split('.')[0]
            else:
                cur_file = Path(cur_file).stem
            bpp, t = compress_ehem(batch[:-2], test_output_path + cur_file, model, args)
        else:
            bpp, t = compress(batch[:-2], test_output_path + Path(cur_file).stem, model, args)
        bpps.append(bpp)
        times.append(t)
        psnr.append(float(batch[-1]))
        chamfer.append(float(batch[-2]))
        print(psnr[-1], bpp, chamfer[-1], t)
        print(sum(psnr) / (i + 1), sum(bpps) / (i + 1), sum(chamfer) / (i + 1), sum(times) / (i + 1))

    if combine_results:
        print('sample number:', len(bpps))
        print('times:', float(np.array(times).mean()))
        print('bpp:', float(np.array(bpps).mean()))
        if args.type == 'kitti' or args.type == 'ford':
            print('chamfer_dist:', float(np.array(chamfer).mean()))
            print('PSNR:', sum(psnr) / len(psnr))
            out = f'mul {args.lidar_level} {args.test_files} {args.ckpt_path}\n' + \
                f'sample number: {len(bpps)}\ntimes: {float(np.array(times).mean())}\n' + \
                f'bpp: {float(np.array(bpps).mean())}\nchamfer_dist: {float(np.array(chamfer).mean())}\n' + \
                f'PSNR: {sum(psnr) / len(psnr)}\n\n'
            with open(f'test_results_mul_{args.type}_{args.lidar_level}.txt', 'a') as f:
                f.write(out)
    else:
        print('bpps:', bpps)
        if args.type == 'kitti' or args.type == 'ford':
            print('chamfer_dist:', chamfer)
            print('PSNR:', psnr)


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
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--type", type=str, default='obj', choices=['obj', 'kitti', 'ford'])
    parser.add_argument("--lidar_level", type=int, default=12)
    parser.add_argument("--level_wise", action="store_true")
    parser.add_argument("--cylin", action="store_true")
    parser.add_argument("--spher", action="store_true")
    parser.add_argument("--preproc_path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
