import argparse
import glob
import pt as pointCloud
import os
from data_preprocess import proc_pc, mul_proc_pc
from pt import write_ply_data, loadply, loadbin
import numpy as np
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="kitti", choices=["kitti", "ford"])
    parser.add_argument("--ori_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--parts", type=str, default="-1/-1")
    parser.add_argument("--lidar_level", type=int, default=16)
    parser.add_argument("--cylin", action="store_true", help="whether using cylindrical coordinate")
    parser.add_argument("--spher", action="store_true", help="whether using spherical coordinate")
    parser.add_argument("--mullevel", action="store_true", help="whether using more levels for distant area")
    return parser.parse_args()


def test_multi_level(ori_file):
    ori_path = Path(ori_file)
    out_dir = Path(args.out_dir)
    out_name = str(ori_path.parent).split('/')[-1] + ori_path.stem if args.type == 'kitti' else ori_path.stem
    out_file, quantized_pc, pc, bin_num, z_offset = mul_proc_pc(
        ori_file,
        out_dir,
        out_name,
        normalize=False,
        qs=400 / (2**args.lidar_level - 1) if args.type == 'kitti' else 2**(18-args.lidar_level),
        test=True,
        spher=args.spher,
        cylin=args.cylin,
        morton_path=[0, 0],
        )
    out_file2, quantized_pc2, pc2, _, _ = mul_proc_pc(
        ori_file,
        out_dir,
        out_name,
        normalize=False,
        qs=400 / (2**(args.lidar_level + 1) - 1) if args.type == 'kitti' else 2**(17-args.lidar_level),
        test=True,
        spher=args.spher,
        cylin=args.cylin,
        morton_path=[0, 1],
        )
    out_file3, quantized_pc3, pc3, _, _ = mul_proc_pc(
        ori_file,
        out_dir,
        out_name,
        normalize=False,
        qs=400 / (2**(args.lidar_level + 2) - 1) if args.type == 'kitti' else 2**(16-args.lidar_level),
        test=True,
        spher=args.spher,
        cylin=args.cylin,
        morton_path=[1],
        )

    whole_pc = pc
    whole_q_pc = np.vstack((quantized_pc, quantized_pc2, quantized_pc3))
    write_ply_data(out_dir / (out_name + "_quant.ply"), whole_q_pc)
    np.save(out_dir / (out_name + '_meta'), [bin_num, pointCloud.distChamfer(whole_pc, whole_q_pc), z_offset])


def test(ori_file):
    ori_path = Path(ori_file)
    out_dir = Path(args.out_dir)
    out_name = str(ori_path.parent).split('/')[-1] + ori_path.stem if args.type == 'kitti' else ori_path.stem
    out_file, quantized_pc, pc, bin_num = proc_pc(
        ori_file,
        out_dir,
        out_name,
        normalize=False,
        qs=400 / (2**args.lidar_level - 1) if args.type == 'kitti' else 2**(18-args.lidar_level),
        test=True,
        spher=args.spher,
        )

    if ori_file.endswith('.bin'):
        whole_pc = loadbin(ori_file)[0]
    else:
        whole_pc = loadply(ori_file)[0]
    write_ply_data(out_dir / (out_name + "_quant.ply"), quantized_pc)
    np.save(out_dir / (out_name + '_meta'), [bin_num, pointCloud.distChamfer(whole_pc, quantized_pc)])


def test_cylin(ori_file):
    ori_path = Path(ori_file)
    out_dir = Path(args.out_dir)
    out_name = str(ori_path.parent).split('/')[-1] + ori_path.stem if args.type == 'kitti' else ori_path.stem
    out_file, quantized_pc, pc, bin_num, offset = proc_pc(
        ori_file,
        out_dir,
        out_name,
        normalize=False,
        qs=400 / (2**args.lidar_level - 1) if args.type == 'kitti' else 2**(18-args.lidar_level),
        test=True,
        cylin=args.cylin
        )

    whole_pc = loadply(ori_file)[0]
    write_ply_data(out_dir / (out_name + "_quant.ply"), quantized_pc)
    np.save(out_dir / (out_name + '_meta'), [bin_num, pointCloud.distChamfer(whole_pc, quantized_pc), offset[0, 2]])


def main(args):
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    test_files = glob.glob(args.ori_dir)

    if not args.parts.startswith("-1"):
        part = int(args.parts.split("/")[0])
        total = int(args.parts.split("/")[1])
    else:
        part = 0
        total = 1
    start = len(test_files) * part // total
    end = len(test_files) * (part + 1) // total
    for i, ori_file in enumerate(test_files[start:end]):
        if args.mullevel:
            test_multi_level(ori_file)
        else:
            if args.cylin:
                test_cylin(ori_file)
            else:
                test(ori_file)
        print(f"part {part}/{total}: {i}/{end-start}")


if __name__ == '__main__':
    args = get_args()
    main(args)
