import argparse
import glob
import sys
sys.path.append('.')
from data_preproc.Octree import gen_K_parent_seq, gen_K_parent_seq_mullevel, mullevel_gen_octree, DeOctree
from data_preproc.OctreeCPP.Octreewarpper import gen_octree
import data_preproc.pt as pointCloud
import numpy as np
import math
import os


def proc_pc(
    inp_path,
    out_dir,
    out_name,
    qs=1,
    offset=0,
    qlevel=None,
    rotation=False,
    normalize=False,
    test=False,
    cylin=False,
    spher=False,
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    p = pointCloud.ptread(inp_path)

    ref_pt = p
    if normalize is True:  # normalize pc to [-1,1]^3
        p = p - np.mean(p, axis=0)
        p = p / abs(p).max()
        ref_pt = p

    if rotation:
        ref_pt = ref_pt[:, [0, 2, 1]]
        ref_pt[:, 2] = -ref_pt[:, 2]

    points = ref_pt
    if cylin:
        points = cart2cylin(ref_pt)
        bin_num = np.round(points[:, 0].max() / qs) + 1
        qs = np.array([qs, 2 * math.pi / (bin_num - 1), qs])[True]
        offset = np.array([0.0, 0.0, min(points[:, 2])])[True]
    elif spher:
        points = cart2spher(ref_pt)
        bin_num = np.round(points[:, 0].max() / qs) + 1
        qs = np.array([qs, 2 * math.pi / (bin_num - 1), math.pi / (bin_num - 1)])[True]

    if type(offset) == str and offset == "min":
        offset = np.min(points, 0)

    points = points - offset

    if qlevel is not None:
        if cylin:
            qs = ((np.array([points[:, 0].max(), 2 * math.pi, points[:, 0].max()])) / (2**qlevel - 1))[True]
            qs[:, 2] = qs[:, 0]  # rho and z should be quantized with the same stepsize
        elif spher:
            qs = ((np.array([points[:, 0].max(), 2 * math.pi, math.pi])) / (2**qlevel - 1))[True]
            qs[:, 2] = qs[:, 0]  # rho and z should be quantized with the same stepsize
        else:
            qs = (points.max() - points.min()) / (2**qlevel - 1)

    pt = np.round(points / qs)
    pt, _ = np.unique(pt, axis=0, return_index=True)
    pt = pt.astype(int)
    octree = gen_octree(pt)
    pc_struct = gen_K_parent_seq(octree, 4)

    out_pc = np.concatenate((pc_struct["Seq"][:, :, True], pc_struct["Level"], pc_struct["Pos"]), axis=2)

    if test:
        out_file = os.path.join(out_dir, out_name)
        np.save(out_file + "_loc", ref_pt)
    else:
        out_file = os.path.join(out_dir, out_name + "_" + str(out_pc.shape[0]))
    np.save(out_file, out_pc)

    if not test:
        return
    out_points = (pt * qs + offset).astype(np.float32)
    if cylin:
        out_points = cylin2cart(out_points)
        return [out_file, out_points, ref_pt, bin_num, offset]
    elif spher:
        out_points = spher2cart(out_points)
        return [out_file, out_points, ref_pt, bin_num]
    return [out_file, out_points, ref_pt]


def mul_proc_pc(inp_path, out_dir, out_name, qs=1, offset=0, qlevel=None, rotation=False, normalize=False, test=False, cylin=False, spher=False, morton_path=[0]):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    p = pointCloud.ptread(inp_path)

    ref_pt = p
    if normalize is True:  # normalize pc to [-1,1]^3
        p = p - np.mean(p, axis=0)
        p = p/abs(p).max()
        ref_pt = p

    if rotation:
        ref_pt = ref_pt[:, [0, 2, 1]]
        ref_pt[:, 2] = - ref_pt[:, 2]

    points = ref_pt
    if cylin:
        points = cart2cylin(ref_pt)
        bin_num = np.round(points[:, 0].max() / qs) + 1
        qs = np.array([qs, 2*math.pi / (bin_num - 1), qs])[True]
        offset = np.array([0., 0., min(points[:, 2])])[True]
    elif spher:
        points = cart2spher(ref_pt)
        bin_num = np.round(points[:, 0].max() / qs) + 1
        qs = np.array([qs, 2*math.pi / (bin_num - 1), math.pi / (bin_num - 1)])[True]

    if type(offset) == str and offset == 'min':
        offset = np.min(points, 0)

    points = points - offset

    if qlevel is not None:
        if cylin:
            qs = ((np.array([points[:, 0].max(), 2*math.pi, points[:, 0].max()]))/(2**qlevel-1))[True]
            qs[:, 2] = qs[:, 0] # rho and z should be quantized with the same stepsize
        elif spher:
            qs = ((np.array([points[:, 0].max(), 2*math.pi, math.pi]))/(2**qlevel-1))[True]
            qs[:, 2] = qs[:, 0] # rho and z should be quantized with the same stepsize
        else:
            qs = (points.max() - points.min())/(2**qlevel-1)

    pt = np.round(points/qs)
    _, pt_idx = np.unique(pt, axis=0, return_index=True)
    pt = pt[np.sort(pt_idx)]
    p = pt.astype(int)
    codes, octree, _, _ = mullevel_gen_octree(pt, morton_path=morton_path)
    pc_struct = gen_K_parent_seq_mullevel(octree, 4)

    out_pc = np.concatenate((
        pc_struct['Seq'][:, :, True],
        pc_struct['Level'],
        pc_struct['Pos']), axis=2)

    if test:
        for m in morton_path:
            out_name += '_' + str(m)
        out_file = os.path.join(out_dir, out_name)
        np.save(out_file + '_loc', ref_pt)
    else:
        out_file = os.path.join(out_dir, out_name + '_' + str(out_pc.shape[0]))
    np.save(out_file, out_pc)

    # for test
    if test:
        pt = DeOctree(codes)
    out_points = pt*qs+offset
    if cylin:
        out_points = cylin2cart(out_points)
        return [out_file, out_points, ref_pt, bin_num, offset[0, 2]]
    elif spher:
        out_points = spher2cart(out_points)
        return [out_file, out_points, ref_pt, bin_num, offset]


# cylindrical
def cart2cylin(points):
    if len(points.shape) == 2:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x + 1e-9)
        phi[np.where(phi < 0)[0]] += 2 * math.pi
        return np.vstack((rho, phi, z)).transpose(1, 0)
    elif len(points.shape) == 3:
        x, y, z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x + 1e-9)
        phi[np.where(phi < 0)[0]] += 2 * math.pi
        return np.stack((rho, phi, z), axis=1).transpose(0, 2, 1)


def cylin2cart(points):
    if len(points.shape) == 2:
        rho, phi, z = points[:, 0], points[:, 1], points[:, 2]
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return np.vstack((x, y, z)).transpose(1, 0)
    elif len(points.shape) == 3:
        rho, phi, z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return np.stack((x, y, z), axis=1).transpose(0, 2, 1)


# spherical
def cart2spher(points):
    if len(points.shape) == 2:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        rho = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x + 1e-9)
        phi[np.where(phi < 0)[0]] += 2 * math.pi
        theta = np.arccos(z / rho)
        return np.vstack((rho, phi, theta)).transpose(1, 0)
    elif len(points.shape) == 3:
        x, y, z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
        rho = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x + 1e-9)
        phi[np.where(phi < 0)[0]] += 2 * math.pi
        theta = np.arccos(z / rho)
        return np.stack((rho, phi, theta), axis=1).transpose(0, 2, 1)


def spher2cart(points):
    if len(points.shape) == 2:
        rho, phi, theta = points[:, 0], points[:, 1], points[:, 2]
        x = rho * np.sin(theta) * np.cos(phi)
        y = rho * np.sin(theta) * np.sin(phi)
        z = rho * np.cos(theta)
        return np.vstack((x, y, z)).transpose(1, 0)
    elif len(points.shape) == 3:
        rho, phi, theta = points[:, :, 0], points[:, :, 1], points[:, :, 2]
        x = rho * np.sin(theta) * np.cos(phi)
        y = rho * np.sin(theta) * np.sin(phi)
        z = rho * np.cos(theta)
        return np.stack((x, y, z), axis=1).transpose(0, 2, 1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="kitti", choices=["kitti", "ford"])
    parser.add_argument("--ori_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--parts", type=str, default="-1/-1")
    parser.add_argument("--cylin", action="store_true", help="whether using cylindrical coordinate")
    parser.add_argument("--spher", action="store_true", help="whether using spherical coordinate")
    return parser.parse_args()

MVUB_NAMES = ['andrew10', 'david10', 'phil10',
                'phil9', 'ricardo10', 'ricardo9', 'sarah10']

if __name__ == "__main__":
    args = get_args()

    data_type = args.type
    ori_dir = args.ori_dir
    out_dir = args.out_dir

    ori_files = glob.glob(ori_dir)
    existing_file_names  = list(map(lambda x: x.rsplit("_", 1)[0].split('/')[-1], glob.glob(out_dir + '/*.npy')))

    if not args.parts.startswith("-1"):
        part = int(args.parts.split("/")[0])
        total = int(args.parts.split("/")[1])
    else:
        part = 0
        total = 1
    start = len(ori_files) * part // total
    end = len(ori_files) * (part + 1) // total

    for i, ori_file in enumerate(ori_files[start:end]):
        print(f"part {part}/{total}: {i}/{end-start}")
        # numpy automatically adds .npy as suffix
        if args.type == "ford":
            out_name = os.path.basename(ori_file).split(".")[0]
        else:
            out_name = (ori_file.split("/")[-3] + os.path.basename(ori_file).split(".")[0])
        if out_name in existing_file_names:
            print(f"Already exists: {out_name}")
            continue

        if args.cylin: # cylindrical preprocessing
            proc_pc(
                ori_file,
                out_dir,
                out_name,
                normalize=False,
                qs=1 if args.type == 'ford' else 400 / (2**16 - 1),
                cylin=True,
            )
        elif args.spher: # spherical preprocessing
            proc_pc(
                ori_file,
                out_dir,
                out_name,
                normalize=False,
                qs=1 if args.type == 'ford' else 400 / (2**16 - 1),
                spher=True,
            )
        else: # cartesian preprocessing
            proc_pc(
                ori_file,
                out_dir,
                out_name,
                offset=-2**17 if args.type == 'ford' else -200,
                normalize=False,
                qs=1 if args.type == 'ford' else 400 / (2**16 - 1),
            )
