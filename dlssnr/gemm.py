"""Matrix product of e4m3-grid operands on the GPU's float16 matrix units, accumulated in float32.

Every e4m3 value is exactly a float16 value and an e4m3 product has at most 8 significant bits, so the
float16 multiply is exact and only the float32 summation order can differ from a float32 matrix product.
The activation is read as float32 (or float16) and narrowed to float16 inside the kernel, so no narrowed
copy of it is ever written; the result is stored as float32.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _round8(x):
    """Round to the nearest e4m3 value (ties to even), saturating at +-448; the result is float32."""
    return tl.minimum(tl.maximum(x, -448.0), 448.0).to(tl.float8e4nv).to(tl.float32)


@triton.jit
def _activation(x):
    """The network's cubic SiLU, with the two nested multiply-adds fused as in ``swin.activation``."""
    t = tl.minimum(tl.maximum(x, -4.0), 4.0)
    inner = tl.fma(tl.abs(t), -0.055908203125, 0.447265625)
    return x * tl.fma(t, inner, 0.89453125)


@triton.jit
def _gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_bn,
            ACTIVATE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        depth = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rows[:, None] * stride_am + depth[None, :],
                    mask=(rows[:, None] < M) & (depth[None, :] < K), other=0.0).to(tl.float16)
        b = tl.load(b_ptr + depth[:, None] * stride_bk + columns[None, :] * stride_bn,
                    mask=(depth[:, None] < K) & (columns[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
    if ACTIVATE:
        accumulator = _round8(_activation(accumulator)).to(tl.float16)
    tl.store(c_ptr + rows[:, None] * N + columns[None, :], accumulator,
             mask=(rows[:, None] < M) & (columns[None, :] < N))


def mm(a: torch.Tensor, b: torch.Tensor, activate: bool = False) -> torch.Tensor:
    """``a @ b`` for ``a`` ``[..., K]`` (float32 or float16, e4m3 values) and ``b`` ``[K, N]`` float16.

    Returns float32, or with ``activate`` the e4m3-rounded cubic SiLU of the product as float16.
    """
    lead = a.shape[:-1]
    a = a.reshape(-1, a.shape[-1])
    if a.stride(-1) != 1:
        a = a.contiguous()
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty(M, N, device=a.device, dtype=torch.float16 if activate else torch.float32)
    block_n = min(128, max(16, triton.next_power_of_2(N)))
    grid = (triton.cdiv(M, 128), triton.cdiv(N, block_n))
    _gemm_kernel[grid](a, b, out, M, N, K, a.stride(0), b.stride(0), b.stride(1),
                  ACTIVATE=activate, BLOCK_M=128, BLOCK_N=block_n, BLOCK_K=32, num_warps=4)
    return out.reshape(*lead, N)


_half_weights: dict[tuple, torch.Tensor] = {}


def clear_cache() -> None:
    """Forget the float16 weight copies; they are keyed by address, which a later network may reuse."""
    _half_weights.clear()


def half_weight(weight: torch.Tensor) -> torch.Tensor:
    """The float16 copy of an e4m3 ``weight``, made once."""
    key = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()), weight.device)
    half = _half_weights.get(key)
    if half is None:
        half = _half_weights[key] = weight.to(torch.float16).contiguous()
    return half


def clear_cache() -> None:
    """Forget the half copies. They are keyed by address, and the address of a freed network's weight is
    handed to the next network's weights, which would then be served the old network's copy."""
    _half_weights.clear()


def linear(x: torch.Tensor, weight: torch.Tensor, activate: bool = False) -> torch.Tensor:
    """``x @ weight.T`` for an e4m3 ``weight`` ``[N, K]``."""
    return mm(x, half_weight(weight).T, activate)
