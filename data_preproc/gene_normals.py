# KITTI do not have normals, to calculate D2 PSNR, we need to generate normals first.
import glob
import open3d as o3d
from pt import loadbin
import argparse
import os
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--parts", type=str, default="-1/-1")
    return parser.parse_args()


def main(args):
    if args.out_dir[-1] != '/':
        args.out_dir += '/'
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
        print(f"part {part}/{total}: {i}/{end-start}")
        cur_out_dir = args.out_dir + ori_file.split('/')[-3]
        if not os.path.exists(cur_out_dir):
            os.mkdir(cur_out_dir)
        out_path = cur_out_dir + '/' + ori_file.split('/')[-1].split('.')[0] + '.ply'
        pc = loadbin(ori_file)[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
        o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
        with open(out_path, 'r+') as f:
            lines = f.readlines()
            for j in range(4, 10):
                lines[j] = lines[j].replace('double', 'float32')
            f.seek(0)
            f.writelines(lines)
            f.close()

if __name__ == '__main__':
    args = get_args()
    main(args)
