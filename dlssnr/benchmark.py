"""Timing: how long a frame takes and where the time goes.

``dlssnr benchmark`` builds the network for one or more frame sizes and reports the cold first frame (kernel
compilation and warm-up), the steady-state time per frame, and the peak GPU memory. ``--layers`` adds the
time per kind of layer and ``--stages`` the time of the pieces around the layers (frame conversion, the
pre-block's features and lift, the post-block's blend, projection and composition). Every measurement waits
for the device to finish, so the numbers are wall time, not launch time.
"""
from __future__ import annotations

import collections
import statistics
import time
from typing import Callable

import numpy as np
import torch

from .runtime import drain


def synthetic_frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A deterministic 8-bit test frame: smooth colour ramps with a little noise and some edges."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack([x / width, y / height, 0.5 + 0.5 * np.sin((x + y) / 97.0)], axis=-1)
    base[(x // 64 + y // 64) % 2 == 0] *= 0.7
    noisy = base * 255.0 + rng.normal(0.0, 4.0, base.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def time_call(function: Callable[[], object], device: torch.device, repeats: int = 5) -> float:
    """Mean milliseconds of ``function`` over ``repeats`` calls, after one untimed warm-up call."""
    function()
    drain(device)
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    drain(device)
    return (time.perf_counter() - started) / repeats * 1000.0


def frame_times(net, frame: np.ndarray, controls, repeats: int) -> tuple[float, list[float]]:
    """(milliseconds of the first frame, milliseconds of each of ``repeats`` steady-state frames)."""
    device = net.device
    started = time.perf_counter()
    net(frame, controls)
    drain(device)
    cold = (time.perf_counter() - started) * 1000.0
    steady = []
    for _ in range(repeats):
        started = time.perf_counter()
        net(frame, controls)
        drain(device)
        steady.append((time.perf_counter() - started) * 1000.0)
    return cold, steady


def layer_times(net, frame: np.ndarray, controls) -> dict[str, tuple[int, float]]:
    """Total milliseconds and layer count per kind of layer for one frame (after a warm-up frame)."""
    device = net.device
    net(frame, controls)
    drain(device)
    totals: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
    started: dict[int, float] = {}
    handles = []
    for spec, layer in zip(net.plan, net.layers):
        label = spec.kind + (f" ({spec.variant})" if spec.variant else "")

        def before(_module, _inputs, index=spec.index):
            drain(device)
            started[index] = time.perf_counter()

        def after(_module, _inputs, _output, index=spec.index, label=label):
            drain(device)
            entry = totals[label]
            entry[0] += 1
            entry[1] += (time.perf_counter() - started[index]) * 1000.0

        handles.append(layer.register_forward_pre_hook(before))
        handles.append(layer.register_forward_hook(after))
    try:
        net(frame, controls)
    finally:
        for handle in handles:
            handle.remove()
    return {label: (int(count), ms) for label, (count, ms) in totals.items()}


def stage_times(net, frame: np.ndarray, controls, repeats: int = 5) -> list[tuple[str, float]]:
    """Milliseconds of the pieces around the layers, measured on the real intermediate tensors of one frame."""
    from .io import frame_to_tensor, tensor_to_frame
    from .layers.pre_post import compose
    from .layers.swin import swin_core

    device = net.device
    image = frame_to_tensor(frame, device)
    outputs = net.run(image, controls, capture=True)
    pre, post = net.layers[0], net.layers[-1]
    last = len(net.plan) - 1
    main = outputs[last - 1][0] if isinstance(outputs[last - 1], tuple) else outputs[last - 1]
    skip = outputs[0][1]
    features = pre.features(image)
    blended = post.blend(main, skip)
    wide = swin_core(post.params, blended, post.spec.shift_x, post.spec.shift_y)
    residual = torch.einsum("rc,bchw->brhw", post.project, wide)
    result = compose(image, residual, controls.intensity)
    height, width = frame.shape[:2]
    steps: list[tuple[str, Callable[[], object]]] = [
        ("frame to tensor", lambda: frame_to_tensor(frame, device)),
        ("pre-block features (noise, colour, controls)", lambda: pre.features(image)),
        ("pre-block lift", lambda: torch.einsum("ck,bkhw->bchw", pre.lift, features)),
        ("post-block blend", lambda: post.blend(main, skip)),
        ("post-block Swin body", lambda: swin_core(post.params, blended, post.spec.shift_x, post.spec.shift_y)),
        ("post-block projection", lambda: torch.einsum("rc,bchw->brhw", post.project, wide)),
        ("post-block composition", lambda: compose(image, residual, controls.intensity)),
        ("tensor to frame", lambda: tensor_to_frame(result, width, height)),
    ]
    return [(name, time_call(function, device, repeats)) for name, function in steps]


def format_table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    """A left-aligned first column and right-aligned numbers, no dependencies."""
    table = [header, *rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = []
    for n, row in enumerate(table):
        lines.append("  ".join((cell.ljust(widths[0]) if i == 0 else cell.rjust(widths[i])) for i, cell in enumerate(row)))
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def report_layers(times: dict[str, tuple[int, float]]) -> str:
    total = sum(ms for _, ms in times.values())
    rows = [(label, str(count), f"{ms:.1f}", f"{ms / count:.2f}", f"{100 * ms / total:.1f} %")
            for label, (count, ms) in sorted(times.items(), key=lambda item: -item[1][1])]
    return format_table(rows, ("layer", "n", "total ms", "ms each", "share")) + f"\n(sum of the layers {total:.1f} ms)"


def report_stages(steps: list[tuple[str, float]]) -> str:
    return format_table([(name, f"{ms:.2f}") for name, ms in steps], ("stage", "ms"))


def run(net, frame: np.ndarray, controls, repeats: int, layers: bool, stages: bool, label: str) -> str:
    """The report for one frame size."""
    device = net.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cold, steady = frame_times(net, frame, controls, repeats)
    mean = statistics.fmean(steady)
    height, width = frame.shape[:2]
    lines = [f"{width}x{height} on {label}",
             f"  first frame   {cold / 1000:.2f} s   (includes kernel compilation and warm-up)",
             f"  steady state  {mean:.1f} ms   min {min(steady):.1f}, max {max(steady):.1f} over {repeats} frames"
             f"   = {1000 / mean:.2f} frames/s"]
    if device.type == "cuda":
        lines.append(f"  peak memory   {torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f} GiB")
    if layers:
        lines += ["", report_layers(layer_times(net, frame, controls))]
    if stages:
        lines += ["", report_stages(stage_times(net, frame, controls))]
    return "\n".join(lines)
