"""Experiment: the network's matrix products on an INT8 grid (W8A8), simulated in float32.

Weights get one scale per output channel, activations one scale per token (or one per tensor); both are
rounded to int8 and multiplied back, so the product is what an int8 matrix unit with an int32 accumulator
and a float rescale would give (up to float32 summation order). ``sites`` names the families that use it:
"swin" (Swin MLP and attention projections), "vit", "split" (split-Swin projections, qkv and feed-forward),
"decoder". Off by default: with ``sites`` empty ``mm`` is a plain matrix product.
"""
from __future__ import annotations

import torch

sites: set[str] = set()
granularity = "token"           # "token" or "tensor"
blocks: set[int] = set()        # only these blocks use int8 (empty: all)
current = -1                    # the block being run, set by NRNet.run
_cache: dict[tuple, torch.Tensor] = {}


def clear_cache() -> None:
    _cache.clear()


def _weight(b: torch.Tensor) -> torch.Tensor:
    key = (b.data_ptr(), tuple(b.shape), tuple(b.stride()), b.device)
    cached = _cache.get(key)
    if cached is None:
        scale = (b.abs().amax(dim=-2, keepdim=True) / 127.0).clamp_min(1e-30)
        cached = torch.round(b / scale).clamp_(-127, 127) * scale
        _cache[key] = cached
    return cached


def _activation(a: torch.Tensor) -> torch.Tensor:
    peak = a.abs().amax(dim=-1, keepdim=True) if granularity == "token" else a.abs().amax()
    scale = (peak / 127.0).clamp_min(1e-30)
    return torch.round(a / scale).clamp_(-127, 127) * scale


def mm(a: torch.Tensor, b: torch.Tensor, site: str) -> torch.Tensor:
    """``a @ b`` with ``a`` ``[..., T, K]`` and ``b`` ``[..., K, N]``."""
    if site in sites and (not blocks or current in blocks):
        return _activation(a) @ _weight(b)
    return a @ b
