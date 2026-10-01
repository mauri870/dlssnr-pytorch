"""Running the network below the frame's size: ways to carry its answer back onto the full-resolution frame.

The current runtime scales the frame down, runs the network, and adds the (bilinearly upsampled) difference
between the network's answer and what it was shown back onto the full-resolution frame. This script scores
that and alternatives against a reference output at full resolution.

    python scripts/m4_upsample.py --input 3840x2160.png --reference nvidia_3840x2160.png --scale 0.5
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dlssnr.io import Controls, load_image
from dlssnr.metrics import psnr, ssim
from dlssnr.model import NRNet
from dlssnr.runtime import resolve_device
from dlssnr.weights import Weights

LUMA = torch.tensor([0.2126, 0.7152, 0.0722])


def to_t(a, device):
    return torch.from_numpy(a).to(device).permute(2, 0, 1)[None].float() / 255


def to_np(t):
    return (t[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()


def luma(t):
    return (t * LUMA.to(t.device).view(1, 3, 1, 1)).sum(1, keepdim=True)


def box(x, r):
    k = 2 * r + 1
    x = F.pad(x, (r, r, r, r), mode="reflect")
    return F.avg_pool2d(x, k, stride=1)


def guided(guide, target, r, eps):
    """He et al.'s guided filter coefficients: target ~ a * guide + b over (2r+1)^2 windows."""
    mg, mt = box(guide, r), box(target, r)
    cov = box(guide * target, r) - mg * mt
    var = box(guide * guide, r) - mg * mg
    a = cov / (var + eps)
    b = mt - a * mg
    return box(a, r), box(b, r)


def up(x, size, mode="bilinear"):
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def methods(frame, low, answer, size, args):
    """name -> output at full size. ``frame`` full-res, ``low`` what the network saw, ``answer`` its output."""
    edit = answer - low
    out = {}
    out["bilinear edit (current)"] = frame + up(edit, size)
    out["bicubic edit"] = frame + up(edit, size, "bicubic")
    # Local-linear transfer: the network's answer as a per-window gain and offset of the luminance it was shown,
    # applied to the full-resolution luminance (the gain carries the network's local contrast boost onto the
    # detail the downscale threw away); chroma moves by the upsampled chroma edit.
    yl, ya, yf = luma(low), luma(answer), luma(frame)
    chroma_edit = (answer - ya) - (low - yl)
    for r in args.radii:
        for eps in args.eps:
            a, b = guided(yl, ya, r, eps)
            gain, offset = up(a, size), up(b, size)
            y_out = gain * yf + offset
            out[f"local-linear r={r} eps={eps}"] = (frame - yf) + up(chroma_edit, size) + y_out
    # Guided upsampling of the edit itself, guided by the full-resolution luminance.
    for r in args.radii:
        eps = args.eps[0]
        yf_small = F.interpolate(yf, size=yl.shape[-2:], mode="area")
        a, b = guided(yf_small, luma(edit), r, eps)
        out[f"guided edit r={r} eps={eps}"] = frame + up(chroma_edit, size) + up(a, size) * yf + up(b, size)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--weights", default="weights.safetensors")
    p.add_argument("--scale", type=float, nargs="+", default=[0.5])
    p.add_argument("--radii", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--save", help="write this method's picture to out/m4_<scale>.png")
    p.add_argument("--eps", type=float, nargs="+", default=[1e-3, 1e-2])
    args = p.parse_args()

    device = resolve_device("auto")
    frame_np, reference = load_image(args.input), load_image(args.reference)
    height, width = frame_np.shape[:2]
    weights = Weights.load(args.weights)
    frame = to_t(frame_np, device)
    print(f"{'method':<34}{'scale':>6} {'PSNR':>7} {'SSIM':>7}")
    print(f"{'unprocessed input':<34}{'':>6} {psnr(frame_np, reference):7.2f} {ssim(frame_np, reference):7.4f}")
    for scale in args.scale:
        w, h = round(width * scale), round(height * scale)
        low = F.interpolate(frame, size=(h, w), mode="area") if scale < 1 else frame
        low_np = to_np(low)
        net = NRNet(weights, w, h).to(device)
        answer = to_t(net(low_np, Controls()), device)
        low = to_t(low_np, device)
        del net
        for name, result in methods(frame, low, answer, (height, width), args).items():
            out = to_np(result)
            print(f"{name:<34}{scale:6.2f} {psnr(out, reference):7.2f} {ssim(out, reference):7.4f}", flush=True)
            if args.save and name == args.save:
                Image.fromarray(out).save(f"out/m4_{scale}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
