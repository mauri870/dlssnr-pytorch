"""Device selection and the numeric settings every device must share."""
from __future__ import annotations

import functools
import os
from types import ModuleType
from typing import Callable, NamedTuple

import torch
import torch._inductor.config

# Every matrix product accumulates in true float32: the e4m3 operands are exact in float32 and a reduced
# precision accumulator (TF32, bfloat16 splits, f16 reductions) would move values across rounding points.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# The compiler must round to float16 after every float16 operation, as eager execution does.
torch._inductor.config.emulate_precision_casts = True

_COMPILE = os.environ.get("DLSSNR_COMPILE", "1") != "0"
_TRITON = os.environ.get("DLSSNR_TRITON", "1") != "0"


class Kernels(NamedTuple):
    gemm: ModuleType
    attention: ModuleType
    swin_fused: ModuleType


def triton_kernels(tensor: torch.Tensor) -> Kernels | None:
    """The Triton kernel modules when they apply to ``tensor`` (on an AMD GPU, Triton installed), else ``None``.

    They run the float16 matrix units with a float32 accumulator and fuse the rounding steps, so sums are
    taken in a different order than the float32 reference path. ``DLSSNR_TRITON=0`` selects the reference path.
    """
    if not _TRITON or not tensor.is_cuda or torch.version.hip is None:
        return None
    try:
        from . import attention, gemm, swin_fused
    except ImportError:
        return None
    return Kernels(gemm, attention, swin_fused)


def clear_caches() -> None:
    """Drop the per-weight copies the kernels keep, which are keyed by address (see ``gemm.clear_cache``)."""
    import sys

    from . import int8

    int8.clear_cache()
    if "dlssnr.gemm" in sys.modules:
        sys.modules["dlssnr.gemm"].clear_cache()


def fuse(function: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Run an elementwise function as one fused GPU kernel instead of a launch per operation.

    The function runs compiled for tensors on a GPU and eagerly otherwise, and it must hold no matrix
    products and no float32 multiply feeding an add (a compiler may fuse that pair into a single rounding).
    The compiled code is checked to give the eager bits. ``DLSSNR_COMPILE=0`` turns compilation off.
    """
    compiled: Callable[..., torch.Tensor] | None = None

    @functools.wraps(function)
    def run(first: torch.Tensor, *rest):
        nonlocal compiled
        if not _COMPILE or not first.is_cuda:
            return function(first, *rest)
        if compiled is None:
            compiled = torch.compile(function, dynamic=True)
        return compiled(first, *rest)

    return run


def resolve_device(name: str = "auto") -> torch.device:
    """``"auto"`` is the GPU when PyTorch sees one (CUDA and ROCm builds both call it ``cuda``), else the CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def drain(device: torch.device) -> None:
    """Block until the GPU has finished everything queued so far; a no-op on the CPU.

    ROCm's runtime crashes (segmentation fault) once roughly ten thousand launches are queued ahead of the
    GPU, which the chunk loops of a 4K frame reach within one layer, so they drain once per chunk.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
