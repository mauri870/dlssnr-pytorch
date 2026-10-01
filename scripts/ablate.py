"""Which parts of the network carry the picture: leave out one unit at a time and score the result.

A unit is one plain Swin layer, one split-Swin block or one vision-transformer block (everything whose
output has the shape of its input). Each unit is skipped alone, then a few groups together. Scores are PSNR
of the variant against the full network's output (what the unit contributes) and against a reference
output (what it costs in fidelity), plus the unit's share of the network's multiply-adds.

    python scripts/ablate.py --input frame.png --reference nvidia.png --out out/ablation.md
"""
from __future__ import annotations

import argparse
import collections
import time

import torch

from dlssnr.io import Controls, load_image
from dlssnr.metrics import psnr, ssim
from dlssnr.model import NRNet
from dlssnr.runtime import resolve_device
from dlssnr.weights import Weights

SPLIT_SKIPPABLE = set(range(24, 30)) | set(range(40, 47))
VIT = set(range(31, 39))


def macs(spec) -> float:
    """Multiply-adds of a layer family per token (qkv, projection and a 4x feed-forward: 12 C^2)."""
    channels = 32 if spec.channels == 3 else spec.channels
    return 12.0 * channels * channels * spec.tokens


def units(plan):
    """name -> layer indices, for every skippable unit."""
    by_block = collections.defaultdict(list)
    for spec in plan:
        by_block[spec.block].append(spec)
    result = {}
    for block, specs in by_block.items():
        kind = specs[0].kind
        if kind.startswith("Swin") and len(specs) == 1 and specs[0].variant in (None, "", "-"):
            result[f"swin block {block} ({kind}, C={specs[0].channels})"] = [specs[0].index]
        elif block in SPLIT_SKIPPABLE or block in VIT:
            family = "vit" if block in VIT else "split"
            result[f"{family} block {block} (C={specs[0].channels})"] = [s.index for s in specs]
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--weights", default="weights.safetensors")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="out/ablation.md")
    args = p.parse_args()

    device = resolve_device(args.device)
    frame, reference = load_image(args.input), load_image(args.reference)
    net = NRNet(Weights.load(args.weights), frame.shape[1], frame.shape[0]).to(device)
    controls = Controls()
    total = sum(macs(s) for s in net.plan)
    full = net(frame, controls)
    print(f"full network vs reference: {psnr(full, reference):.2f} dB")

    candidates = units(net.plan)
    groups = {
        "all vit blocks 31-38": sorted(i for n, v in candidates.items() if n.startswith("vit") for i in v),
        "encoder split blocks 24-29": sorted(i for n, v in candidates.items()
                                              if n.startswith("split block") and int(n.split()[2]) < 30 for i in v),
        "decoder split blocks 40-46": sorted(i for n, v in candidates.items()
                                              if n.startswith("split block") and int(n.split()[2]) >= 40 for i in v),
        "all split blocks": sorted(i for n, v in candidates.items() if n.startswith("split") for i in v),
        "decoder swin layers (49-54, 57-60, 63-64, 67-68)": [
            i for n, v in candidates.items() if n.startswith("swin") and int(n.split()[2]) >= 49 for i in v],
        "encoder swin layers (2-3, 6-7, 10-13, 16-21)": [
            i for n, v in candidates.items() if n.startswith("swin") and int(n.split()[2]) < 49 for i in v],
    }
    rows = []
    for name, indices in {**candidates, **groups}.items():
        net.skip = set(indices)
        started = time.time()
        out = net(frame, controls)
        share = sum(macs(net.plan[i]) for i in indices) / total * 100
        rows.append((name, share, psnr(out, full), psnr(out, reference), ssim(out, reference)))
        print(f"{name:<55} {share:5.1f} % macs  {rows[-1][2]:6.2f} dB vs full  {rows[-1][3]:6.2f} dB vs ref  "
              f"({time.time() - started:.1f} s)", flush=True)
    net.skip = set()

    with open(args.out, "w") as handle:
        handle.write(f"Full network vs reference: {psnr(full, reference):.2f} dB, "
                     f"{frame.shape[1]}x{frame.shape[0]}\n\n")
        handle.write("| Left out | MACs saved | PSNR vs full network | PSNR vs reference | SSIM vs reference |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        for name, share, vs_full, vs_ref, structure in rows:
            handle.write(f"| {name} | {share:.1f} % | {vs_full:.2f} | {vs_ref:.2f} | {structure:.4f} |\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
