"""Windowed attention of a Swin block as one GPU kernel per (window, head).

The arithmetic is ``swin._attention``'s, with every rounding point and every f16 summation order the
network uses (the squared-norm chains, the softmax denominator); only the float32 sums inside the two
matrix products are taken in the matrix unit's order. Query, key and value are read straight from the
``[tokens, 3C]`` projection, and the context is written as float16 (e4m3 values) in the layout the output
projection reads, so no permuted copy of any operand exists.

A window has 64 tokens and a head 32 dimensions. The keys are handled as four blocks of 16 so that the
softmax denominator can be summed block by block, as the network adds it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .gemm import _round8

HEAD_DIM = tl.constexpr(32)
TOKENS = tl.constexpr(64)
BLOCK = tl.constexpr(16)
EXP_SLOPE = tl.constexpr(0.044921875)
EXP_LOW = tl.constexpr(1.03125)
EXP_HIGH = tl.constexpr(1.5693359375)
NORM_FLOOR = tl.constexpr(0.000062)
SUM_FLOOR = tl.constexpr(2.0 ** -14)


@triton.jit
def _round16(x):
    return x.to(tl.float16).to(tl.float32)


@triton.jit
def _norm_scale(base, rows, stride):
    """Reciprocal L2 norm of the 32-dimension rows at ``base + row * stride`` (``swin._norm_scale``).

    The squares go through four interleaved fused multiply-add chains over each of two halves (even and odd
    dimensions); the chains are added in a tree, the halves are narrowed to f16 and added in f16.
    """
    lanes = tl.arange(0, 8)
    total = tl.zeros((rows.shape[0], 8), dtype=tl.float32)
    for group in tl.static_range(2):
        for fragment in tl.static_range(2):
            v = tl.load(base + rows[:, None] * stride + fragment * 16 + group * 8 + lanes[None, :])
            total = tl.fma(v, v, total)
    return _norm_tail(total, rows.shape[0])


@triton.jit
def _norm_tail(total, count: tl.constexpr):
    """The reciprocal root of the chains ``total`` ``[count, 8]`` (lane = 4 * chain pair + 2 * chain + half)."""
    by_half = tl.reshape(total, (count, 2, 2, 2))
    half_0, half_1 = tl.split(by_half)
    half_0_a, half_0_b = tl.split(half_0)
    half_1_a, half_1_b = tl.split(half_1)
    sum_0_a, sum_0_b = tl.split(half_0_a + half_0_b)
    sum_1_a, sum_1_b = tl.split(half_1_a + half_1_b)
    narrow_0 = _round16(sum_0_a + sum_0_b)
    narrow_1 = _round16(sum_1_a + sum_1_b)
    squared = tl.maximum(_round16(narrow_0 + narrow_1), NORM_FLOOR)
    return tl.math.div_rn(1.0, tl.math.sqrt_rn(squared))


@triton.jit
def _norm_scale_tile(values, count: tl.constexpr):
    """``_norm_scale`` of the register tile ``values`` ``[count, 32]``."""
    by_dimension = tl.permute(tl.reshape(values, (count, 2, 2, 4, 2)), (0, 3, 4, 1, 2))   # row, chain, half, fragment, group
    group_0, group_1 = tl.split(by_dimension)
    fragment_00, fragment_10 = tl.split(group_0)
    fragment_01, fragment_11 = tl.split(group_1)
    total = tl.zeros((count, 4, 2), dtype=tl.float32)
    total = tl.fma(fragment_00, fragment_00, total)
    total = tl.fma(fragment_10, fragment_10, total)
    total = tl.fma(fragment_01, fragment_01, total)
    total = tl.fma(fragment_11, fragment_11, total)
    return _norm_tail(tl.reshape(total, (count, 8)), count)


@triton.jit
def _attention_kernel(qkv_ptr, scale_ptr, bias_ptr, out_ptr, heads, row_stride, out_stride):
    window = tl.program_id(0)
    head = tl.program_id(1)
    tokens = tl.arange(0, TOKENS)
    dims = tl.arange(0, HEAD_DIM)
    base = qkv_ptr + window * TOKENS * row_stride + head * 3 * HEAD_DIM

    query_scale = _norm_scale(base, tokens, row_stride) * tl.load(scale_ptr + head)
    query = tl.load(base + tokens[:, None] * row_stride + dims[None, :])
    query = _round8(query * query_scale[:, None]).to(tl.float16)

    bias_base = bias_ptr + head * TOKENS * TOKENS
    keys = tl.arange(0, BLOCK)
    numerators_0 = tl.zeros((TOKENS, BLOCK), dtype=tl.float32)
    numerators_1 = numerators_0
    numerators_2 = numerators_0
    numerators_3 = numerators_0
    total = numerators_0
    for block in tl.static_range(4):
        rows = block * BLOCK + keys
        key_base = base + HEAD_DIM
        key_scale = _norm_scale(key_base, rows, row_stride)
        key = tl.load(key_base + rows[:, None] * row_stride + dims[None, :])
        key = _round8(key * key_scale[:, None]).to(tl.float16)
        logits = tl.dot(query, tl.trans(key), out_dtype=tl.float32)
        bias = tl.load(bias_base + tokens[:, None] * TOKENS + rows[None, :])
        ramp = _round16(tl.fma(logits, EXP_SLOPE, bias))
        ramp = tl.minimum(tl.maximum(ramp, EXP_LOW), EXP_HIGH).to(tl.float16)
        bits = (ramp.to(tl.int16, bitcast=True) & 0x3FF) << 5
        numerators = bits.to(tl.float16, bitcast=True).to(tl.float32)
        if block == 0:
            numerators_0 = numerators
            total = numerators
        elif block == 1:
            numerators_1 = numerators
            total = _round16(total + numerators)
        elif block == 2:
            numerators_2 = numerators
            total = _round16(total + numerators)
        else:
            numerators_3 = numerators
            total = _round16(total + numerators)

    # total is [token, 16] = (pair, member, half); add the pairs, then the halves, then the members
    by_pair = tl.permute(tl.reshape(total, (TOKENS, 4, 2, 2)), (0, 2, 3, 1))    # token, member, half, pair
    pairs = tl.reshape(by_pair, (TOKENS, 2, 2, 2, 2))                           # ..., pair high, pair low
    low_0, low_1 = tl.split(pairs)                                              # pair low = 0, 1
    p0, p2 = tl.split(low_0)
    p1, p3 = tl.split(low_1)
    summed = _round16(_round16(_round16(p0 + p1) + p2) + p3)                    # [token, member, half]
    half_0, half_1 = tl.split(summed)
    member_0, member_1 = tl.split(_round16(half_0 + half_1))
    denominator = _round16(member_0 + member_1)
    reciprocal = tl.math.div_rn(1.0, tl.maximum(denominator, SUM_FLOOR))

    context = tl.zeros((TOKENS, HEAD_DIM), dtype=tl.float32)
    for block in tl.static_range(4):
        rows = block * BLOCK + keys
        value = tl.load(base + 2 * HEAD_DIM + rows[:, None] * row_stride + dims[None, :])
        value = _round8(_round16(value)).to(tl.float16)
        if block == 0:
            numerators = numerators_0
        elif block == 1:
            numerators = numerators_1
        elif block == 2:
            numerators = numerators_2
        else:
            numerators = numerators_3
        probability = _round8(_round16(numerators * reciprocal[:, None])).to(tl.float16)
        context = tl.dot(probability, value, context, out_dtype=tl.float32)
    context = _round8(_round16(context)).to(tl.float16)
    tl.store(out_ptr + (window * TOKENS + tokens[:, None]) * out_stride + head * HEAD_DIM + dims[None, :], context)


def window_attention(qkv: torch.Tensor, head_scale: torch.Tensor, exp_bias: torch.Tensor) -> torch.Tensor:
    """Attention context of ``qkv`` ``[windows * 64, heads * 96]`` (float32; per head q, k, v of 32).

    ``head_scale`` is ``[heads]`` float32 and ``exp_bias`` ``[heads, 64, 64]`` float32. Returns the e4m3
    context as float16 ``[windows * 64, heads * 32]``.
    """
    rows, width = qkv.shape
    heads = width // (3 * HEAD_DIM.value)
    out = torch.empty(rows, heads * HEAD_DIM.value, device=qkv.device, dtype=torch.float16)
    _attention_kernel[(rows // TOKENS.value, heads)](qkv, head_scale, exp_bias, out, heads, qkv.stride(0), out.stride(0),
                                     num_warps=4)
    return out


# ---- global attention of the vision-transformer blocks -----------------------------------------------------

VIT_SLOPE = tl.constexpr(0.08953857421875)
VIT_OFFSET = tl.constexpr(1.708984375)
VIT_LOW = tl.constexpr(1474.0)           # the ramp's clamp, in units of 1/1024
VIT_HIGH = tl.constexpr(2025.0)
VIT_DENOMINATOR_FLOOR = tl.constexpr(6.198883056640625e-05)
ROUND_MAGIC = tl.constexpr(6755399441055744.0)       # 1.5 * 2**52: adding and subtracting it rounds to even
QUERIES = tl.constexpr(64)


@triton.jit
def _vit_exponential(logits):
    """``vit._approximate_exp`` as float32 holding float16 values: the logit narrowed to f16, a float64 ramp
    rounded to 1/1024 and clamped, whose mantissa is shifted into a float16's exponent and fraction."""
    x = _round16(logits).to(tl.float64)
    ramp = (x * VIT_SLOPE + VIT_OFFSET) * 1024.0
    ramp = (ramp + ROUND_MAGIC) - ROUND_MAGIC
    ramp = tl.minimum(tl.maximum(ramp, VIT_LOW), VIT_HIGH)
    bits = ((ramp.to(tl.int32) - 1024) << 4).to(tl.int16)
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _vit_kernel(qkv_ptr, out_ptr, tokens, channels, padding):
    head = tl.program_id(0)
    queries = tl.program_id(1) * QUERIES + tl.arange(0, QUERIES)
    dims = tl.arange(0, HEAD_DIM)
    keys = tl.arange(0, BLOCK)
    channel = head * HEAD_DIM + dims
    query = tl.load(qkv_ptr + channel[None, :] * tokens + queries[:, None],
                    mask=queries[:, None] < tokens, other=0.0).to(tl.float16)
    key_base = qkv_ptr + channels * tokens
    value_base = qkv_ptr + 2 * channels * tokens

    numerator = tl.zeros((QUERIES, HEAD_DIM), dtype=tl.float32)
    denominator = tl.zeros((QUERIES,), dtype=tl.float32)
    for group in range(0, tl.cdiv(tokens, 64)):
        accumulated = tl.zeros((QUERIES, 8), dtype=tl.float32)
        for block in tl.static_range(4):
            rows = group * 64 + block * BLOCK + keys
            key = tl.load(key_base + channel[:, None] * tokens + rows[None, :],
                          mask=rows[None, :] < tokens, other=0.0).to(tl.float16)
            probability = _vit_exponential(tl.dot(query, key, out_dtype=tl.float32))
            # key c of a block is added to key c + 8, then the blocks one after the other
            by_half = tl.permute(tl.reshape(probability, (QUERIES, 2, 8)), (0, 2, 1))
            first, second = tl.split(by_half)
            pairs = _round16(first + second)
            if block == 0:
                accumulated = pairs
            else:
                accumulated = _round16(accumulated + pairs)
            value = tl.load(value_base + channel[None, :] * tokens + rows[:, None],
                            mask=rows[:, None] < tokens, other=0.0).to(tl.float16)
            numerator = tl.dot(_round8(probability).to(tl.float16), value, numerator, out_dtype=tl.float32)
        # the eight partial sums: ((p0+p2)+p4)+p6 and ((p1+p3)+p5)+p7, then their sum
        even, odd = tl.split(tl.reshape(accumulated, (QUERIES, 4, 2)))
        even_low, even_high = tl.split(tl.reshape(even, (QUERIES, 2, 2)))
        odd_low, odd_high = tl.split(tl.reshape(odd, (QUERIES, 2, 2)))
        e0, e2 = tl.split(even_low)
        e1, e3 = tl.split(even_high)
        o0, o2 = tl.split(odd_low)
        o1, o3 = tl.split(odd_high)
        even_sum = _round16(_round16(_round16(e0 + e1) + e2) + e3)
        odd_sum = _round16(_round16(_round16(o0 + o1) + o2) + o3)
        group_sum = _round16(even_sum + odd_sum)
        if group == 0:
            denominator = group_sum
        else:
            denominator = _round16(denominator + group_sum)

    denominator = _round16(denominator - padding)
    inverse = _round16(tl.math.div_rn(1.0, tl.maximum(denominator, VIT_DENOMINATOR_FLOOR)))
    result = _round8(_round16(_round16(numerator) * inverse[:, None]))
    tl.store(out_ptr + channel[None, :] * tokens + queries[:, None], result, mask=queries[:, None] < tokens)


def vit_attention(qkv: torch.Tensor, padding: float) -> torch.Tensor:
    """Global attention of ``qkv`` ``[1, 3C, h, w]`` (e4m3 values); returns float32 ``[1, C, h, w]``.

    ``padding`` is the float16 sum the padded keys of the last group add to every denominator.
    """
    channels = qkv.shape[1] // 3
    tokens = qkv.shape[2] * qkv.shape[3]
    qkv = qkv.contiguous()
    out = torch.empty(1, channels, *qkv.shape[2:], device=qkv.device, dtype=torch.float32)
    heads = channels // HEAD_DIM.value
    _vit_kernel[(heads, triton.cdiv(tokens, QUERIES.value))](qkv, out, tokens, channels, padding, num_warps=4)
    return out
