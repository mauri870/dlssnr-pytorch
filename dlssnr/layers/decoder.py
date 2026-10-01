"""The decoder's input stage (block 39): the bottleneck's output meets the encoder's full-resolution skip."""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ..plan import LayerSpec
from ..quant import q8
from ..weights import Weights
from .vit import _Matmul, f16, to_tokens


class DecInputUpsample(nn.Module):
    """Project 1024 -> 512, double the resolution, add the gated skip.

    The projection is kept in float16; each of its values is repeated over a 2x2 block (nearest neighbour,
    cropped to the skip's extent), then ``f16(projection + f16(f16(skip) * gain))`` is rounded to e4m3.
    """

    def __init__(self, spec: LayerSpec, weights: Weights):
        super().__init__()
        self.matmul = _Matmul(weights.e4m3(spec.block, spec.layer, "weight", spec.c_out, spec.c_in), "decoder")
        self.register_buffer("gain", weights.f16(spec.block, spec.layer, "skip_weight", spec.c_out))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """``x``: bottleneck output ``[1, 1024, hv, wv]``; ``skip``: ``[1, 512, h, w]``; returns ``[1, 512, h, w]``."""
        height, width = skip.shape[-2:]
        low, wide = x.shape[-2:]
        projected = f16(self.matmul(to_tokens(x))).t().reshape(1, -1, low, wide)
        upsampled = projected.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)[..., :height, :width]
        gated = f16(f16(skip) * self.gain.reshape(1, -1, 1, 1))
        return q8(f16(upsampled + gated))


BUILDERS: dict[str, Callable[[LayerSpec, Weights], nn.Module]] = {"DecInputUpsample": DecInputUpsample}
