"""How strong is the edit of a network run below the frame's size, compared with the full-size one?

For each frame: the full-size network's output is the reference; the small run's edit (answer minus what it was
shown) is upsampled and added to the frame with a gain. Prints the least-squares gain and the PSNR at gain 1,
at that gain, and at the gain given with --fixed (a per-scale value fitted elsewhere).
"""
import argparse
import glob

import torch
import torch.nn.functional as F

from m4_upsample import to_np, to_t, up
from dlssnr.io import Controls, load_image
from dlssnr.metrics import psnr
from dlssnr.model import NRNet
from dlssnr.runtime import resolve_device
from dlssnr.weights import Weights

p = argparse.ArgumentParser()
p.add_argument("--frames", nargs="+", required=True)
p.add_argument("--scale", type=float, nargs="+", default=[0.75, 0.5, 0.375, 0.25])
p.add_argument("--fixed", type=float, nargs="*", default=[], help="gain per scale, same order as --scale")
p.add_argument("--weights", default="weights.safetensors")
args = p.parse_args()
device = resolve_device("auto")
weights = Weights.load(args.weights)
controls = Controls()
for path in args.frames:
    frame_np = load_image(path)
    height, width = frame_np.shape[:2]
    frame = to_t(frame_np, device)
    reference = to_t(NRNet(weights, width, height).to(device)(frame_np, controls), device)
    ref_np = to_np(reference)
    for k, scale in enumerate(args.scale):
        h, w = round(height * scale), round(width * scale)
        low_np = to_np(F.interpolate(frame, size=(h, w), mode="area"))
        low = to_t(low_np, device)
        edit = up(to_t(NRNet(weights, w, h).to(device)(low_np, controls), device) - low, (height, width))
        gain = ((edit * (reference - frame)).sum() / (edit * edit).sum()).item()
        row = f"{path.split('/')[-1]:<26} scale {scale:5.3f}  gain {gain:.2f}  1.00: {psnr(to_np(frame + edit), ref_np):6.2f}  fit: {psnr(to_np(frame + gain * edit), ref_np):6.2f}"
        if k < len(args.fixed):
            row += f"  fixed {args.fixed[k]:.2f}: {psnr(to_np(frame + args.fixed[k] * edit), ref_np):6.2f}"
        print(row, flush=True)
