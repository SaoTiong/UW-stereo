import sys
sys.path.append('core')

import argparse
import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image

DEVICE = 'cuda'


def load_image(imfile, size=None):
    img = Image.open(imfile)
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def list_images_from_dir(root, ext):
    return sorted(glob.glob(os.path.join(root, f"*.{ext}")))


def build_output_name(left_path, left_root=None):
    if left_root is None:
        return Path(left_path).stem
    rel = os.path.relpath(left_path, left_root)
    rel_no_ext = os.path.splitext(rel)[0]
    return rel_no_ext.replace(os.sep, "_")


def predict(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    disparity_directory = output_directory / "disparity"
    disparity_directory.mkdir(exist_ok=True, parents=True)
    covariance_directory = output_directory / "covariance"
    covariance_directory.mkdir(exist_ok=True, parents=True)

    if args.left_dir and args.right_dir:
        left_images = list_images_from_dir(args.left_dir, args.ext)
        right_images = list_images_from_dir(args.right_dir, args.ext)
    else:
        left_images = sorted(glob.glob(args.left_glob, recursive=True))
        right_images = sorted(glob.glob(args.right_glob, recursive=True))

    if len(left_images) != len(right_images):
        raise ValueError(f"Left/right count mismatch: {len(left_images)} vs {len(right_images)}")

    print(f"Found {len(left_images)} image pairs. Saving to {output_directory}/")

    with torch.no_grad():
        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            resize_hw = (args.resize_w, args.resize_h) if args.resize_h and args.resize_w else None
            image1 = load_image(imfile1, size=resize_hw)
            image2 = load_image(imfile2, size=resize_hw)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            preds = model(image1, image2, iters=args.valid_iters, test_mode=True)
            if isinstance(preds, list):
                flow_predictions = [padder.unpad(pred) for pred in preds]
                flow_up = flow_predictions[-1].squeeze()
                stacked = torch.stack([pred.squeeze(0) for pred in flow_predictions], dim=0)
                covariance = torch.var(stacked, dim=0, unbiased=False).squeeze(0)
            else:
                _, flow_up = preds
                flow_up = padder.unpad(flow_up).squeeze()
                covariance = None

            file_stem = build_output_name(imfile1, args.left_dir)
            if args.save_numpy:
                np.save(disparity_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            if args.save_covariance:
                if covariance is None:
                    raise RuntimeError("Covariance requested but test_mode did not return per-iteration predictions.")
                np.save(covariance_directory / f"{file_stem}.npy", covariance.cpu().numpy().squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', required=True)
    parser.add_argument('--save_numpy', action='store_true')
    parser.add_argument('--save_covariance', action='store_true')
    parser.add_argument('--left_dir')
    parser.add_argument('--right_dir')
    parser.add_argument('--left_glob')
    parser.add_argument('--right_glob')
    parser.add_argument('--ext', default='png')
    parser.add_argument('--output_directory', default='predict_output')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--valid_iters', type=int, default=32)
    parser.add_argument('--resize_h', type=int, default=600, help='resize height for inference')
    parser.add_argument('--resize_w', type=int, default=800, help='resize width for inference')

    # Architecture (same as demo.py)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument('--shared_backbone', action='store_true')
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'])
    parser.add_argument('--slow_fast_gru', action='store_true')
    parser.add_argument('--n_gru_layers', type=int, default=3)

    args = parser.parse_args()

    if not ((args.left_dir and args.right_dir) or (args.left_glob and args.right_glob)):
        raise ValueError("Provide either --left_dir/--right_dir or --left_glob/--right_glob.")

    predict(args)
