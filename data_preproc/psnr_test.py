import argparse
import glob
import data_preproc.pt as pointCloud
import uuid
from pathlib import Path
from tqdm import tqdm
from utils import get_psnr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="obj", choices=["obj", "kitti", "ford"])
    parser.add_argument("--ori_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--chamfer", action='store_true')
    parser.add_argument("--no_psnr", action='store_true')
    # parser.add_argument("--parts", type=str, default="-1/-1")
    # parser.add_argument("--lidar_level", type=int, default=16)
    # parser.add_argument("--cylin", action="store_true", help="whether using cylindrical coordinate")
    # parser.add_argument("--spher", action="store_true", help="whether using spherical coordinate")
    return parser.parse_args()


def main(args):
    if args.out_dir[-1] != '/':
        args.out_dir += '/'

    if not args.no_psnr:
        if args.type == 'kitti':
            peak = '59.70'
        elif args.type == 'ford':
            peak = '30000'
    psnrs_d1 = []
    psnrs_d2 = []
    chamfers = []
    test_files = glob.glob(args.ori_dir)
    tmp_test_file = "temp/pcerror_results" + str(uuid.uuid4()) + ".txt"
    with tqdm(total=len(test_files)) as t:
        for ori_file in test_files:
            ori_path = Path(ori_file)
            if args.type == 'kitti':
                preproc_path = args.out_dir + ori_file.split('/')[-2] + ori_path.stem
            else:
                preproc_path = args.out_dir + ori_file.split('/')[-1].split('.')[0]
            if not args.no_psnr:
                pointCloud.pcerror(ori_file, preproc_path + '_quant.ply', None, '-r ' + peak, tmp_test_file)
                psnr_d1, psnr_d2 = get_psnr(tmp_test_file)
                psnrs_d1.append(psnr_d1)
                psnrs_d2.append(psnr_d2)
            if args.chamfer:
                whole_pc = pointCloud.loadply(ori_file)[0]
                whole_quant_pc = pointCloud.loadply(preproc_path + '_quant.ply')[0]
                chamfer = pointCloud.distChamfer(whole_pc, whole_quant_pc)
                chamfers.append(chamfer)
            postfix = {}
            if not args.no_psnr:
                postfix['psnr d1'] = format(psnr_d1, '.3f')
                postfix['avg d1'] = format(sum(psnrs_d1) / len(psnrs_d1), '.3f')
                postfix['psnr d2'] = format(psnr_d2, '.3f')
                postfix['avg d2'] = format(sum(psnrs_d2) / len(psnrs_d2), '.3f')
            if args.chamfer:
                postfix['cd'] = format(chamfer, '.3f')
                postfix['avg cd'] = format(sum(chamfers) / len(chamfers), '.3f')
            t.set_postfix(postfix)
            t.update(1)
    if not args.no_psnr:
        print('avg psnr d1:', sum(psnrs_d1) / len(psnrs_d1))
        print('avg psnr d2:', sum(psnrs_d2) / len(psnrs_d2))
    if args.chamfer:
        print('avg cd:', sum(chamfers) / len(chamfers))
    print('total files:', max(len(psnrs_d1), len(chamfers)))


if __name__ == '__main__':
    args = get_args()
    main(args)
