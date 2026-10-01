"""Command line: extract the weights, enhance an image, verify against a reference.

    dlssnr extract nvngx_dlssnr.dll -o weights.safetensors
    dlssnr run photo.png -o enhanced.png --weights weights.safetensors
    dlssnr verify --input photo.png --reference nvidia.png --weights weights.safetensors
    dlssnr plan 1920x1080
    dlssnr benchmark --size 1920x1080 --size 3840x2160 --layers --stages
"""
from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_WEIGHTS = os.environ.get("DLSSNR_WEIGHTS", "weights.safetensors")


def _controls(args):
    from .io import Controls

    return Controls(intensity=args.intensity, style=args.style, local_tone=args.local_tone,
                    local_structure=args.local_structure, skin_structure=args.skin_structure,
                    auto_mask=not args.no_auto_mask)


def _add_controls(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("controls (the defaults are the library's)")
    g.add_argument("--intensity", type=float, default=1.0, help="overall strength of the edit, 0..2")
    g.add_argument("--style", type=int, default=0)
    g.add_argument("--local-tone", type=float, default=1.0)
    g.add_argument("--local-structure", type=float, default=1.0)
    g.add_argument("--skin-structure", type=float, default=-1.0)
    g.add_argument("--no-auto-mask", action="store_true", help="turn the automatic skin mask off")


def _enhance(args, frame):
    import torch

    from .model import NRNet
    from .runtime import resolve_device
    from .weights import Weights

    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    label = torch.cuda.get_device_name(device) if device.type == "cuda" else "the CPU"
    height, width = frame.shape[:2]
    started = time.time()
    weights = Weights.load(args.weights)
    net = NRNet(weights, width, height).to(device)
    print(f"network built for {width}x{height} on {label} in {time.time() - started:.1f} s", file=sys.stderr)
    started = time.time()
    with torch.no_grad():
        out = net(frame, _controls(args))
    print(f"enhanced in {time.time() - started:.1f} s on {label}", file=sys.stderr)
    return out


def cmd_extract(args) -> int:
    from .extract import extract

    weights = extract(args.library, args.output)
    print(f"{len(weights.tensors)} tensors -> {args.output}")
    return 0


def cmd_run(args) -> int:
    from .io import load_image, save_image

    frame = load_image(args.image)
    out = _enhance(args, frame)
    output = args.output or os.path.splitext(args.image)[0] + "_dlssnr.png"
    if args.side_by_side:
        import numpy as np

        save_image(output, np.concatenate([frame, out], axis=1))
    else:
        save_image(output, out)
    print(output)
    return 0


def cmd_verify(args) -> int:
    from .io import load_image, save_image
    from .metrics import compare

    frame = load_image(args.input)
    reference = load_image(args.reference)
    if frame.shape != reference.shape:
        print(f"the input is {frame.shape[1]}x{frame.shape[0]} and the reference {reference.shape[1]}x{reference.shape[0]}",
              file=sys.stderr)
        return 2
    out = _enhance(args, frame)
    if args.output:
        save_image(args.output, out)
    stats = compare(out, reference, frame)
    print(f"PSNR vs reference      {stats['psnr_db']:.2f} dB   (the unprocessed input scores {stats['unprocessed_psnr_db']:.2f} dB)")
    print(f"SSIM vs reference      {stats['ssim']:.4f}")
    print(f"RMS error              {stats['rms_error']:.3f} / 255")
    print(f"mean difference (RGB)  {stats['mean_difference'][0]:+.2f} {stats['mean_difference'][1]:+.2f} {stats['mean_difference'][2]:+.2f}")
    print(f"within one 8-bit step  {stats['pixels_within_one_step_pct']:.1f} % of pixels")
    print(f"edit correlation       {stats['edit_correlation']:.4f}   (edit RMS {stats['edit_rms_output']:.2f} vs {stats['edit_rms_reference']:.2f})")
    if args.min_psnr is not None and stats["psnr_db"] < args.min_psnr:
        print(f"FAIL: PSNR below {args.min_psnr} dB", file=sys.stderr)
        return 1
    return 0


def cmd_benchmark(args) -> int:
    import torch

    from .benchmark import run, synthetic_frame
    from .io import load_image
    from .model import NRNet
    from .runtime import resolve_device
    from .weights import Weights

    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    label = torch.cuda.get_device_name(device) if device.type == "cuda" else f"the CPU ({torch.get_num_threads()} threads)"
    weights = Weights.load(args.weights)
    controls = _controls(args)
    for size in args.size or ["1920x1080"]:
        if args.image:
            frame = load_image(args.image)
        else:
            width, height = (int(v) for v in size.lower().split("x"))
            frame = synthetic_frame(width, height)
        net = NRNet(weights, frame.shape[1], frame.shape[0]).to(device)
        print(run(net, frame, controls, args.frames, args.layers or args.all, args.stages or args.all, label))
        print()
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


def cmd_plan(args) -> int:
    from .plan import build_plan

    width, height = (int(v) for v in args.size.lower().split("x"))
    specs = build_plan(width, height)
    print(f"{len(specs)} layers for a {width}x{height} frame")
    last = None
    for s in specs:
        key = (s.block, s.kind, s.channels, s.w, s.h)
        if key != last:
            print(f"block {s.block:2d}  {s.kind:<16} {s.variant or '-':<10} C={s.channels:<5} {s.w}x{s.h}  tokens={s.tokens}")
        last = key
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dlssnr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="read NVIDIA's nvngx_dlssnr.dll (or a zip holding it) and write the weights")
    e.add_argument("library")
    e.add_argument("-o", "--output", default=DEFAULT_WEIGHTS)
    e.set_defaults(func=cmd_extract)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--weights", default=DEFAULT_WEIGHTS)
        q.add_argument("--device", default="auto",
                       help="auto (the GPU if PyTorch sees one, else the CPU), cpu, cuda or cuda:N")
        q.add_argument("--threads", type=int, default=0, help="CPU threads (0: PyTorch's default)")
        _add_controls(q)

    r = sub.add_parser("run", help="enhance an image")
    r.add_argument("image")
    r.add_argument("-o", "--output")
    r.add_argument("--side-by-side", action="store_true", help="write the input and the result next to each other")
    common(r)
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("verify", help="enhance an image and score it against a reference output")
    v.add_argument("--input", required=True)
    v.add_argument("--reference", required=True)
    v.add_argument("-o", "--output")
    v.add_argument("--min-psnr", type=float)
    common(v)
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("benchmark", help="time a frame and show where the time goes")
    b.add_argument("--size", action="append", help="frame size such as 1920x1080 (repeatable; default 1920x1080)")
    b.add_argument("--image", help="time this image instead of a synthetic frame (its size is used)")
    b.add_argument("--frames", type=int, default=5, help="steady-state frames to time")
    b.add_argument("--layers", action="store_true", help="time per kind of layer")
    b.add_argument("--stages", action="store_true", help="time of the pieces around the layers")
    b.add_argument("--all", action="store_true", help="--layers and --stages")
    common(b)
    b.set_defaults(func=cmd_benchmark)

    pl = sub.add_parser("plan", help="print the layers of the network for a frame size, e.g. 1920x1080")
    pl.add_argument("size")
    pl.set_defaults(func=cmd_plan)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
