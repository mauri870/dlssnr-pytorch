"""The whole body of a single-head, 32-channel Swin block as one GPU kernel per attention window.

Everything ``swin.swin_block`` does for one window (the MLP, the e4m3 roundings between its stages, the
normalised attention and the output projection) happens in registers: the window's 64 tokens are gathered
from the feature map with the shifted window grid already applied, and the result is scattered back, so
the only memory traffic is one read and one write of the feature map. The rounding points and the f16
summation orders are the ones in ``swin.py`` and ``attention.py``; only the float32 sums inside the matrix
products are taken in the matrix unit's order.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .attention import (BLOCK, EXP_HIGH, EXP_LOW, EXP_SLOPE, HEAD_DIM, SUM_FLOOR, TOKENS, _norm_scale_tile,
                        _round16)
from .gemm import _activation, _round8, half_weight

CHANNELS = tl.constexpr(32)
HIDDEN = tl.constexpr(128)


@triton.jit
def _multiply(x, y):
    """A float32 product that the compiler cannot fuse into a following add."""
    return tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", "=v,v,v", [x, y], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _weight(pointer, rows: tl.constexpr, columns: tl.constexpr):
    """The transpose ``[columns, rows]`` of a row-major float16 ``[rows, columns]`` weight, as a matrix operand."""
    k = tl.arange(0, columns)
    n = tl.arange(0, rows)
    return tl.load(pointer + n[None, :] * columns + k[:, None])


@triton.jit
def _block_sum(numerators):
    """``[64 tokens, 64 keys]`` -> ``[64, 16]``: the four 16-key blocks added in order, each add rounded to f16."""
    by_block = tl.permute(tl.reshape(numerators, (TOKENS, 4, 16)), (0, 2, 1))
    blocks = tl.reshape(by_block, (TOKENS, 16, 2, 2))
    low_0, low_1 = tl.split(blocks)
    block_0, block_2 = tl.split(low_0)
    block_1, block_3 = tl.split(low_1)
    return _round16(_round16(_round16(block_0 + block_1) + block_2) + block_3)


@triton.jit
def _denominator(total):
    """``[64, 16]`` (pair, member, half) -> ``[64]``: pairs, then halves, then members, each add rounded to f16."""
    by_pair = tl.permute(tl.reshape(total, (TOKENS, 4, 2, 2)), (0, 2, 3, 1))
    pairs = tl.reshape(by_pair, (TOKENS, 2, 2, 2, 2))
    low_0, low_1 = tl.split(pairs)
    pair_0, pair_2 = tl.split(low_0)
    pair_1, pair_3 = tl.split(low_1)
    summed = _round16(_round16(_round16(pair_0 + pair_1) + pair_2) + pair_3)
    half_0, half_1 = tl.split(summed)
    member_0, member_1 = tl.split(_round16(half_0 + half_1))
    return _round16(member_0 + member_1)


@triton.jit
def _attend(query, key, value, bias_ptr):
    """The e4m3 context ``[64, 32]`` (float16) of one head's normalised ``query``, ``key`` and ``value``."""
    logits = tl.dot(query, tl.trans(key), out_dtype=tl.float32)
    token = tl.arange(0, TOKENS)
    bias = tl.load(bias_ptr + token[:, None] * TOKENS + token[None, :])
    ramp = _round16(tl.fma(logits, EXP_SLOPE, bias))
    ramp = tl.minimum(tl.maximum(ramp, EXP_LOW), EXP_HIGH).to(tl.float16)
    numerators = ((ramp.to(tl.int16, bitcast=True) & 0x3FF) << 5).to(tl.float16, bitcast=True).to(tl.float32)
    reciprocal = tl.math.div_rn(1.0, tl.maximum(_denominator(_block_sum(numerators)), SUM_FLOOR))
    probability = _round8(_round16(numerators * reciprocal[:, None])).to(tl.float16)
    context = tl.dot(probability, value, out_dtype=tl.float32)
    return _round8(_round16(context)).to(tl.float16)


@triton.jit
def _kernel(x_ptr, wide_ptr, accumulator_ptr, expand_ptr, contract_ptr, qkv_ptr, projection_ptr, bias_ptr,
            mlp_scale_ptr, attention_scale_ptr, head_scale_ptr,
            height, width, columns, front_h, front_w,
            WIDE_RESIDUAL: tl.constexpr, ACCUMULATOR: tl.constexpr):
    window = tl.program_id(0)
    window_row = window // columns
    window_column = window % columns
    token = tl.arange(0, TOKENS)
    channel = tl.arange(0, CHANNELS)
    row = window_row * 8 + (token // 32) * 4 + (token // 4) % 4 - front_h
    column = window_column * 8 + ((token // 16) % 2) * 4 + token % 4 - front_w
    inside = (row >= 0) & (row < height) & (column >= 0) & (column < width)
    pixel = row * width + column
    offsets = channel[None, :] * (height * width) + pixel[:, None]

    x = tl.load(x_ptr + offsets, mask=inside[:, None], other=0.0)
    quantised = _round8(x)

    hidden = tl.dot(quantised.to(tl.float16), _weight(expand_ptr, HIDDEN, CHANNELS), out_dtype=tl.float32)
    hidden = _round8(_activation(hidden)).to(tl.float16)
    residual = x if WIDE_RESIDUAL else quantised
    mlp_scale = tl.load(mlp_scale_ptr + channel).to(tl.float32)
    y = residual * mlp_scale[None, :] + tl.dot(hidden, _weight(contract_ptr, CHANNELS, HIDDEN), out_dtype=tl.float32)
    y_quantised = _round8(_round16(y)).to(tl.float16)

    query = tl.dot(y_quantised, _weight(qkv_ptr, CHANNELS, CHANNELS), out_dtype=tl.float32)
    key = tl.dot(y_quantised, _weight(qkv_ptr + CHANNELS * CHANNELS, CHANNELS, CHANNELS), out_dtype=tl.float32)
    value = tl.dot(y_quantised, _weight(qkv_ptr + 2 * CHANNELS * CHANNELS, CHANNELS, CHANNELS), out_dtype=tl.float32)
    query_scale = _norm_scale_tile(query, TOKENS) * tl.load(head_scale_ptr)
    key_scale = _norm_scale_tile(key, TOKENS)
    query = _round8(query * query_scale[:, None]).to(tl.float16)
    key = _round8(key * key_scale[:, None]).to(tl.float16)
    value = _round8(_round16(value)).to(tl.float16)

    context = _attend(query, key, value, bias_ptr)

    attention_scale = tl.load(attention_scale_ptr + channel).to(tl.float32)
    out = _multiply(y, attention_scale[None, :]) + tl.dot(context, _weight(projection_ptr, CHANNELS, CHANNELS),
                                                         out_dtype=tl.float32)
    tl.store(wide_ptr + offsets, out.to(tl.float16), mask=inside[:, None])
    if ACCUMULATOR:
        tl.store(accumulator_ptr + offsets, out, mask=inside[:, None])


def supported(params, x: torch.Tensor) -> bool:
    return params.channels == CHANNELS.value and params.heads == 1 and x.is_cuda


def swin_block_fused(params, x: torch.Tensor, shift_rows: int, shift_columns: int, wide_residual: bool,
                 accumulator: bool):
    """``swin.swin_block`` for a single-head, 32-channel block on ``x`` ``[1, 32, rows, columns]``.

    Returns ``wide`` as float16 (its f16-exact values), and with ``accumulator`` also the float32 ``acc``.
    """
    from .layers.swin import _window_grid

    _, _, height, width = x.shape
    window_rows, front_h = _window_grid(height, shift_rows)
    window_columns, front_w = _window_grid(width, shift_columns)
    x = x.contiguous()
    wide = torch.empty(x.shape, device=x.device, dtype=torch.float16)
    acc = torch.empty_like(x) if accumulator else wide
    _kernel[(window_rows * window_columns,)](
        x, wide, acc, half_weight(params.expand), half_weight(params.contract), half_weight(params.qkv),
        half_weight(params.out_proj), params.exp_bias, params.mlp_scale, params.attn_scale, params.head_scale,
        height, width, window_columns, front_h, front_w,
        WIDE_RESIDUAL=wide_residual, ACCUMULATOR=accumulator, num_warps=4)
    return (wide, acc) if accumulator else wide
