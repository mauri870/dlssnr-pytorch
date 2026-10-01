"""The vision-transformer bottleneck (blocks 31-38): 1024 channels, global attention over every token.

Each block runs five layers: FFN expand, FFN contract, QKV, attention, projection. The residual stream
``s`` flows through the contract and projection layers::

    h = expand(s_prev)             hidden, 4096 channels
    c = contract(h) + g1 * s_prev  first residual
    qkv = qkv(c)
    a = attention(qkv)
    s = projection(a) + g4 * c     second residual (its input is the *contract* output)

Nothing in the bottleneck depends on where a token sits, so a feature map ``[1, C, h, w]`` is simply read as
``h * w`` tokens in row-major order; the attention has no positional term.

Rounding follows the network's FP8 dataflow: matmul inputs are e4m3 and exact in float32, accumulation is
float32, and every layer output goes float32 -> float16 -> e4m3 (the float16 hop matters for the last bit).
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

from .. import int8
from ..plan import LayerSpec
from ..quant import q8
from ..runtime import drain, triton_kernels
from ..weights import Weights

HEAD_DIM = 32
KEY_GROUP = 64          # keys are summed in groups of 64 in float16, group by group
_EXP_SLOPE = 0.08953857421875
_EXP_OFFSET = 1.708984375
_EXP_LOW, _EXP_HIGH = 1.439453125, 1.9775390625
_DENOMINATOR_FLOOR = 6.198883056640625e-05
_NORM_FLOOR = 0.000062
_QUERY_CHUNK_ELEMENTS = 1 << 26     # logits kept in flight: heads * queries * keys


def to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    """``[1, C, h, w]`` -> ``[h * w, C]``."""
    return feature_map.flatten(2)[0].t()


def to_map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """``[h * w, C]`` -> ``[1, C, h, w]``."""
    return tokens.t().reshape(1, -1, height, width).contiguous()


def f16(x: torch.Tensor) -> torch.Tensor:
    """Round to float16 and keep going in float32."""
    return x.to(torch.float16).to(torch.float32)


class _Matmul(nn.Module):
    """``out = tokens @ W^T`` with an e4m3 weight ``[N, K]``, accumulated in float32."""

    def __init__(self, weight: torch.Tensor, site: str = "vit"):
        super().__init__()
        self.site = site
        self.register_buffer("weight", weight.contiguous())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return int8.mm(tokens, self.weight.t(), self.site)


def _skip_gain(weights: Weights, block: int, layer: int, channels: int) -> torch.Tensor:
    return weights.f16(block, layer, "skip_weight", channels)


class VitFfnExpand(nn.Module):
    """1024 -> 4096 with the cubic SiLU activation; no residual."""

    def __init__(self, spec: LayerSpec, weights: Weights):
        super().__init__()
        self.matmul = _Matmul(weights.e4m3(spec.block, spec.layer, "weight", spec.c_out, spec.c_in))

    @staticmethod
    def activation(x: torch.Tensor) -> torch.Tensor:
        clipped = x.clamp(-4.0, 4.0)
        return x * (-0.055908203125 * clipped.abs() * clipped + 0.447265625 * clipped + 0.89453125)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        height, width = x.shape[-2:]
        hidden = self.activation(self.matmul(to_tokens(x)))
        return to_map(q8(f16(hidden)), height, width)


class VitFfnContract(nn.Module):
    """4096 -> 1024 plus ``skip_weight * skip``, where ``skip`` is the block's input stream."""

    def __init__(self, spec: LayerSpec, weights: Weights):
        super().__init__()
        self.matmul = _Matmul(weights.e4m3(spec.block, spec.layer, "weight", spec.c_out, spec.c_in))
        self.register_buffer("gain", _skip_gain(weights, spec.block, spec.layer, spec.c_out))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        total = self.matmul(to_tokens(x)) + self.gain * to_tokens(skip)
        return to_map(q8(f16(total)), height, width)


class VitProjection(VitFfnContract):
    """1024 -> 1024 plus ``skip_weight * skip``, where ``skip`` is the contract layer's output."""


class VitQKV(nn.Module):
    """Query, key and value projections; q and k are normalised per head here.

    Per 32-channel head the arithmetic is float16: ``k = k / |k|`` and ``q = q * sqrt(32) / |q| * temperature``
    with a per-head temperature from the layer's header table. The squared norm is summed in the
    network's order (channel ``d`` with ``d + 16``, then a butterfly over distances 8, 4, 2, 1), and rounding
    of every step is float16. The result is ``[q | k | v]`` with 3 * 1024 channels, e4m3.
    """

    def __init__(self, spec: LayerSpec, weights: Weights):
        super().__init__()
        channels = spec.channels
        block, layer = spec.block, spec.layer
        matrices = [weights.e4m3(block, layer, name, channels, channels) for name in ("q", "k", "v")]
        self.matmul = _Matmul(torch.cat(matrices))
        self.channels = channels
        self.register_buffer("temperature", weights.f32(block, layer, "header", channels // HEAD_DIM))

    @staticmethod
    def _inverse_norm(values: torch.Tensor) -> torch.Tensor:
        """``1 / max(|x|, floor)`` over the head dimension of ``[tokens, heads, 32]`` float16 values."""
        squares = values * values                                  # float16
        total = squares[..., :16] + squares[..., 16:]
        for half in (8, 4, 2, 1):
            total = total[..., :half] + total[..., half:2 * half]
        floor = torch.tensor(_NORM_FLOOR, dtype=torch.float16)
        return torch.rsqrt(torch.maximum(total, floor).to(torch.float32)).to(torch.float16)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        height, width = x.shape[-2:]
        channels = self.channels
        heads = channels // HEAD_DIM
        projected = self.matmul(to_tokens(x)).to(torch.float16)
        tokens = projected.shape[0]
        query, key, value = (projected[:, i * channels:(i + 1) * channels].reshape(tokens, heads, HEAD_DIM)
                             for i in range(3))
        sqrt_head = torch.tensor(math.sqrt(HEAD_DIM), dtype=torch.float16)
        temperature = self.temperature.to(torch.float16).reshape(1, heads, 1)
        query = query * sqrt_head * self._inverse_norm(query) * temperature
        key = key * self._inverse_norm(key)
        out = torch.cat([query, key, value], dim=1).reshape(tokens, 3 * channels)
        return to_map(q8(out.to(torch.float32)), height, width)


def _approximate_exp(logits: torch.Tensor) -> torch.Tensor:
    """The network's exponential: a float16 fused multiply-add and a bit trick, float16 result.

    ``y = fma(x, slope, offset)`` is clamped to ``[1.4395, 1.9775]`` (logits of about -3..3); the 10 mantissa bits
    of ``y`` are then shifted up 4 places and read as the bits of a float16. That is a piecewise-linear
    ``2**(x * log2(e))`` times a constant, which the softmax normalisation absorbs.
    """
    x = logits.to(torch.float16).to(torch.float64)
    y = torch.round((x * _EXP_SLOPE + _EXP_OFFSET) * 1024.0).clamp(
        round(_EXP_LOW * 1024.0), round(_EXP_HIGH * 1024.0))
    mantissa = (y - 1024.0).to(torch.int64)
    exponent = (mantissa >> 6) - 15
    fraction = (mantissa & 63).to(torch.float32) / 64.0
    return torch.ldexp(1.0 + fraction, exponent)


def _group_sum(probabilities: torch.Tensor) -> torch.Tensor:
    """Sum of the float16 probabilities of one 64-key group, in the network's order.

    Within each 16-key block key ``c`` is added to key ``c + 8``; the four blocks accumulate one after the
    other; the eight partial sums are then added as ``((p0+p2)+p4)+p6`` and ``((p1+p3)+p5)+p7`` and the two
    results are added. Every addition is a float16 addition. Input ``[..., groups, 64]`` float16.
    """
    blocks = probabilities.reshape(*probabilities.shape[:-1], 4, 2, 8)
    pairs = blocks[..., 0, :] + blocks[..., 1, :]
    accumulated = pairs[..., 0, :]
    for block in range(1, 4):
        accumulated = accumulated + pairs[..., block, :]
    even = ((accumulated[..., 0] + accumulated[..., 2]) + accumulated[..., 4]) + accumulated[..., 6]
    odd = ((accumulated[..., 1] + accumulated[..., 3]) + accumulated[..., 5]) + accumulated[..., 7]
    return even + odd


class VitAttention(nn.Module):
    """Global multi-head attention over all tokens. The layer has no weights.

    q and k arrive normalised and scaled, so the logits are bounded and the softmax needs no running maximum:
    ``p = approximate_exp(q . k)`` in float16. The denominator is the float16 sum of ``p`` (keys grouped by
    64, padding keys counted as ``exp(0)`` and subtracted at the end), the numerator multiplies e4m3-rounded
    ``p`` with e4m3 ``v`` in float32, and the result is ``f16(f16(num) * f16(1 / max(den, floor)))`` rounded to e4m3.
    """

    def __init__(self, spec: LayerSpec, weights: Weights | None = None):
        super().__init__()
        self.channels = spec.channels

    @torch.no_grad()
    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        height, width = x.shape[-2:]
        channels = self.channels
        heads = channels // HEAD_DIM
        tokens = height * width
        padded = -(-tokens // KEY_GROUP) * KEY_GROUP
        kernels = triton_kernels(x)
        if kernels:
            padding = f16(torch.tensor((padded - tokens) * float(_approximate_exp(torch.zeros(1))[0]))).item()
            return kernels.attention.vit_attention(x, padding)
        qkv = to_tokens(x)

        def split(index: int, pad: int = 0) -> torch.Tensor:
            part = qkv[:, index * channels:(index + 1) * channels]
            part = part.reshape(tokens, heads, HEAD_DIM).permute(1, 0, 2)
            return torch.nn.functional.pad(part, (0, 0, 0, pad))

        query, key, value = split(0), split(1, padded - tokens), split(2, padded - tokens)
        padding_keys = padded - tokens
        padding = f16(torch.tensor(padding_keys * float(_approximate_exp(torch.zeros(1))[0])))
        floor = torch.tensor(_DENOMINATOR_FLOOR, dtype=torch.float32)

        step = max(1, _QUERY_CHUNK_ELEMENTS // (heads * padded))
        chunks = []
        for start in range(0, tokens, step):
            logits = torch.bmm(query[:, start:start + step], key.transpose(1, 2))      # [heads, q, keys]
            probabilities = _approximate_exp(logits)
            sums = _group_sum(probabilities.to(torch.float16).reshape(*logits.shape[:2], -1, KEY_GROUP))
            denominator = sums[..., 0]
            for group in range(1, sums.shape[-1]):
                denominator = denominator + sums[..., group]
            denominator = f16(denominator.to(torch.float32) - padding)
            numerator = torch.bmm(q8(probabilities), value)                              # [heads, q, 32]
            inverse = f16(1.0 / torch.maximum(denominator, floor))
            chunks.append(q8(f16(f16(numerator) * inverse.unsqueeze(-1))))
            drain(logits.device)
        out = torch.cat(chunks, dim=1).permute(1, 0, 2).reshape(tokens, channels)
        return to_map(out, height, width)


BUILDERS: dict[str, Callable[[LayerSpec, Weights], nn.Module]] = {
    "VitFfnExpand": VitFfnExpand,
    "VitFfnContract": VitFfnContract,
    "VitQKV": VitQKV,
    "VitAttention": VitAttention,
    "VitProjection": VitProjection,
}
