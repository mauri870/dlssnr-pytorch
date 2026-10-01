"""PSNR of the network with its matrix products on an INT8 grid, per family and all together.

    python scripts/int8_sim.py --input frame.png --reference nvidia.png [--granularity token|tensor]
"""
import argparse

from dlssnr import int8
from dlssnr.io import Controls, load_image
from dlssnr.metrics import psnr, ssim
from dlssnr.model import NRNet
from dlssnr.runtime import resolve_device
from dlssnr.weights import Weights

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--weights", default="weights.safetensors")
args = p.parse_args()
device = resolve_device("auto")
frame, reference = load_image(args.input), load_image(args.reference)
net = NRNet(Weights.load(args.weights), frame.shape[1], frame.shape[0]).to(device)
controls = Controls()
full = net(frame, controls)
print(f"{'sites':<34}{'granularity':<8} vs full  vs ref   ssim")
print(f"{'none':<34}{'':<8} {psnr(full, full) if False else float('inf'):7.2f} {psnr(full, reference):7.2f} {ssim(full, reference):7.4f}")
for granularity in ("token", "tensor"):
    int8.granularity = granularity
    for sites in ({"swin"}, {"vit"}, {"split"}, {"decoder"}, {"vit", "split", "decoder"}, {"swin", "vit", "split", "decoder"}):
        int8.sites = set(sites)
        out = net(frame, controls)
        print(f"{'+'.join(sorted(sites)):<34}{granularity:<8} {psnr(out, full):7.2f} {psnr(out, reference):7.2f} {ssim(out, reference):7.4f}", flush=True)
