"""Which Swin blocks lose the picture when their matrix products run on INT8 (per-token activations)."""
import argparse

from dlssnr import int8
from dlssnr.io import Controls, load_image
from dlssnr.metrics import psnr
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
swin = sorted({s.block for s in net.plan if s.kind.startswith("Swin") or s.kind in ("PreBlock", "PostBlock")})
int8.sites = {"swin"}
for block in swin:
    int8.blocks = {block}
    out = net(frame, controls)
    kinds = next(s.kind for s in net.plan if s.block == block)
    print(f"block {block:2d} {kinds:<16} {psnr(out, reference):6.2f} dB vs ref", flush=True)
