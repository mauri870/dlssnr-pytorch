"""The 512-channel "split Swin" layers of the encoder bottom (blocks 23-30) and decoder start (blocks 40-47).

A block is four layers over a feature map ``[1, 512, rows, cols]`` that holds e4m3 values (``rows`` is
``spec.w`` and ``cols`` is ``spec.h``, the plan's naming):

1. ``SplitFfwd``      grouped feed-forward: eight groups, each 512 -> 64 -> 256 (cubic SiLU) -> 64 channels.
2. ``SplitFfwdProj``  1x1 projection of the feed-forward output plus a gained residual (the block input).
3. ``SplitQKVAttn``   window attention, 16 heads of 32 channels, 8x8 windows on an optionally shifted grid.
4. ``SplitProj``      1x1 projection of the attention output plus a gained residual (layer 2's output).

Block 30 ends with ``SplitProjPool`` (the projection plus a 2x2-mean copy at half resolution) and
``SplitFinalHead``, which projects that copy from 512 to 1024 channels for the transformer stage.

Rounding. Every matrix product accumulates in float32 and is then narrowed to float16 before it is rounded
to e4m3 (the network stores float16 results and converts them), so each stored value is rounded twice,
float32 -> float16 -> e4m3. The attention's normalisation, exponential and activation run in float16
arithmetic with a rounding after every operation, which is reproduced here with float16 tensors.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import int8
from ..plan import LayerSpec
from ..quant import q8
from ..weights import Weights

WINDOW = 8                  # pixels per window side
TILE = 4                    # pixels per tile side; shifts move the window grid by whole tiles
HEADS, HEAD_DIM = 16, 32
CHANNELS = 512
GROUPS, GROUP_IN, GROUP_HIDDEN = 8, 64, 256


def _half(x: torch.Tensor) -> torch.Tensor:
    """Round to float16 and back (float32 storage)."""
    return x.to(torch.float16).to(torch.float32)


def _store(x: torch.Tensor) -> torch.Tensor:
    """float32 -> float16 -> e4m3, the double rounding of every stored result."""
    return q8(_half(x))


def _fma_half(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``a * b + c`` with a single rounding to float16 (float16 inputs)."""
    wide = a.to(torch.float64) * b.to(torch.float64) + c.to(torch.float64)
    return wide.to(torch.float16)


def _pad4(n: int) -> int:
    return (n + 3) // 4 * 4


# --------------------------------------------------------------------------------------------------
# Weight unpacking
# --------------------------------------------------------------------------------------------------


def _bit(value: np.ndarray, index: int, to: int) -> np.ndarray:
    return ((value >> index) & 1) << to


def _slice_sources(rows: int, cols: int) -> np.ndarray:
    """Byte offset inside a packed matrix region of every element of the generic ``[rows, cols]`` unpacking.

    The packed matrix is a sequence of 512-byte slices, 16 output rows by 32 inputs, ordered
    input-group-major; inside a slice a lane holds eight bytes for each of two row halves.
    """
    groups = rows // 16
    sources = np.empty((rows, cols), dtype=np.int64)
    lane = np.arange(32)[:, None]
    byte = np.arange(8)[None, :]
    k_in_slice = 8 * (byte // 2) + 2 * (lane % 4) + byte % 2            # [32, 8]
    for slice_index in range(groups * (cols // 32)):
        row_group, col_group = slice_index % groups, slice_index // groups
        for half in range(2):
            row = row_group * 16 + half * 8 + lane // 4                  # [32, 1]
            offset = slice_index * 512 + lane * 16 + half * 8 + byte     # [32, 8]
            sources[np.broadcast_to(row, offset.shape), col_group * 32 + k_in_slice] = offset
    return sources


def _grouped_ffwd_matrices(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, ...]:
    """Recover the feed-forward's three grouped matrices from its two generically unpacked ``[512, 512]`` halves.

    The 524,288-byte record is not two matrices. It holds, bit-scattered, for each of eight groups
    ``A [64, 512]`` (first half), then ``Q0 [256, 64]`` and ``Q2 [64, 256]`` (second half). The generic
    unpacking is a byte permutation, so undoing it recovers the record and the scatter maps below
    then give the matrices.
    """
    sources = _slice_sources(512, 512)
    record = np.empty(2 * 512 * 512, dtype=np.uint8)
    record[sources.ravel()] = first.ravel()
    record[262144 + sources.ravel()] = second.ravel()

    g = np.arange(GROUPS)[:, None, None]
    row = np.arange(GROUP_IN)[None, :, None]
    k = np.arange(512)[None, None, :]
    a_offset = (_bit(k, 0, 0) | _bit(k, 1, 4) | _bit(k, 2, 5) | _bit(k, 3, 1) | _bit(k, 4, 2)
                | _bit(k, 5, 14) | _bit(k, 6, 15) | _bit(k, 7, 16) | _bit(k, 8, 17)
                | _bit(row, 0, 3) | _bit(row, 1, 6) | _bit(row, 2, 7) | _bit(row, 3, 8) | _bit(row, 4, 9)
                | _bit(row, 5, 10) | _bit(g, 0, 11) | _bit(g, 1, 12) | _bit(g, 2, 13))

    j = np.arange(GROUP_HIDDEN)[None, :, None]
    r = np.arange(GROUP_IN)[None, None, :]
    q0_offset = (262144 + g * 16384
                 + (_bit(r, 0, 1) | _bit(r, 1, 0) | _bit(r, 2, 4) | _bit(r, 3, 5) | _bit(r, 4, 2)
                    | _bit(r, 5, 13) | _bit(j, 0, 6) | _bit(j, 1, 3) | _bit(j, 2, 9) | _bit(j, 3, 7)
                    | _bit(j, 4, 8) | _bit(j, 5, 10) | _bit(j, 6, 11) | _bit(j, 7, 12)))

    n = np.arange(GROUP_IN)[None, :, None]
    j = np.arange(GROUP_HIDDEN)[None, None, :]
    q2_offset = (393216 + g * 16384
                 + (_bit(j, 0, 0) | _bit(j, 1, 1) | _bit(j, 2, 2) | _bit(j, 3, 4) | _bit(j, 4, 5)
                    | _bit(j, 5, 11) | _bit(j, 6, 12) | _bit(j, 7, 13)
                    | _bit(n, 0, 6) | _bit(n, 1, 7) | _bit(n, 2, 8) | _bit(n, 3, 3) | _bit(n, 4, 9)
                    | _bit(n, 5, 10)))
    return record[a_offset], record[q0_offset], record[q2_offset]


def _decode(raw: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(raw)).view(torch.float8_e4m3fn).to(torch.float32)


# --------------------------------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------------------------------


def _channels_last(x: torch.Tensor) -> torch.Tensor:
    """``[1, C, rows, cols]`` -> ``[rows * cols, C]``."""
    return x[0].flatten(1).t()


def _from_channels_last(x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return x.t().reshape(1, -1, rows, cols).contiguous()


class SplitFfwd(nn.Module):
    """Grouped feed-forward. Returns ``(out, x)``: the result and the unchanged input (the next layer's residual).

    Group ``g`` maps all 512 input channels through ``A_g`` to 64 channels, rounds them, expands to 256 with
    ``Q0_g``, applies the cubic SiLU and rounds, and contracts with ``Q2_g`` to the 64 output channels
    ``64 g .. 64 g + 63``.
    """

    def __init__(self, spec: LayerSpec, weights: Weights):
        super().__init__()
        first = weights.raw(spec.block, spec.layer, "ffwd_a").numpy().reshape(512, 512)
        second = weights.raw(spec.block, spec.layer, "ffwd_b").numpy().reshape(512, 512)
        a, q0, q2 = (_decode(m) for m in _grouped_ffwd_matrices(first, second))
        self.register_buffer("expand_in", a.reshape(GROUPS * GROUP_IN, 512).t().contiguous())   # [512, 8 * 64]
        self.register_buffer("expand_hidden", q0.transpose(1, 2).contiguous())                  # [8, 64, 256]
        self.register_buffer("contract", q2.transpose(1, 2).contiguous())                       # [8, 256, 64]

    @staticmethod
    def activation(x: torch.Tensor) -> torch.Tensor:
        """The cubic SiLU in float16 arithmetic: ``x * (t * (0.4473 - 0.0559 |t|) + 0.8945)``, ``t = clamp(x, +-4)``."""
        x = x.to(torch.float16)
        t = x.clamp(-4.0, 4.0)
        u = _fma_half(torch.tensor(-0.055908203125, dtype=torch.float16), t.abs(),
                      torch.tensor(0.447265625, dtype=torch.float16))
        v = _fma_half(t, u, torch.tensor(0.89453125, dtype=torch.float16))
        return (x * v).to(torch.float32)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, rows, cols = x.shape
        tokens = _channels_last(x)
        reduced = _store(int8.mm(tokens, self.expand_in, "split")).reshape(-1, GROUPS, GROUP_IN).transpose(0, 1)     # [8, T, 64]
        hidden = _store(self.activation(int8.mm(reduced, self.expand_hidden, "split")))                    # [8, T, 256]
        out = _store(int8.mm(hidden, self.contract, "split"))                                              # [8, T, 64]
        return _from_channels_last(out.transpose(0, 1).reshape(-1, CHANNELS), rows, cols), x


class _Projection(nn.Module):
    """``q(f16(W x + gain * residual))`` per channel, the form of every 1x1 projection in the family."""

    has_residual = True

    def __init__(self, spec: LayerSpec, weights: Weights):
        super().__init__()
        rows = spec.c_out
        self.register_buffer("weight", weights.e4m3(spec.block, spec.layer, "weight", rows, spec.c_in).t().contiguous())
        if self.has_residual:
            self.register_buffer("gain", weights.f16(spec.block, spec.layer, "skip_weight", rows))

    def project(self, x: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        """The float16 value of the projection, before the e4m3 rounding; ``[1, C_out, rows, cols]``."""
        _, _, rows, cols = x.shape
        total = int8.mm(_channels_last(x), self.weight, "split")
        if residual is not None:
            total = total + _channels_last(residual) * self.gain
        return _half(total).t().reshape(1, -1, rows, cols)


class SplitFfwdProj(_Projection):
    """Projection of the feed-forward output; the residual is the block input.

    ``forward`` takes the ``(out, block_input)`` tuple that ``SplitFfwd`` returns, or the feed-forward map as ``x``
    with the block input as ``skip``.
    """

    @torch.no_grad()
    def forward(self, x, skip: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(x, tuple):
            x, skip = x
        return q8(self.project(x, skip))


class SplitProj(_Projection):
    """Projection of the attention output; the residual is the output of ``SplitFfwdProj`` (``skip``)."""

    @torch.no_grad()
    def forward(self, x: torch.Tensor, skip) -> torch.Tensor:
        skip = skip[0] if isinstance(skip, tuple) else skip
        return q8(self.project(x, skip))


class SplitProjPool(_Projection):
    """``SplitProj`` plus a half-resolution copy. Returns ``(projection, pooled)``.

    The pooled map is the 2x2 mean of the float16 projection values (before they are rounded to e4m3),
    summed pairwise along columns then rows in float16, scaled by 0.25 and rounded to e4m3. Its extent is
    ``ceil(extent / 2)`` padded to a multiple of 4, the padding being zero.
    """

    @torch.no_grad()
    def forward(self, x: torch.Tensor, skip) -> tuple[torch.Tensor, torch.Tensor]:
        skip = skip[0] if isinstance(skip, tuple) else skip
        value = self.project(x, skip).to(torch.float16)
        horizontal = value[..., 0::2] + value[..., 1::2]
        pooled = (horizontal[..., 0::2, :] + horizontal[..., 1::2, :]) * torch.tensor(0.25, dtype=torch.float16)
        pooled = q8(pooled.to(torch.float32))
        rows, cols = pooled.shape[-2:]
        pooled = F.pad(pooled, (0, _pad4(cols) - cols, 0, _pad4(rows) - rows))
        return q8(value.to(torch.float32)), pooled


class SplitFinalHead(_Projection):
    """512 -> 1024 projection of the pooled map, without residual: the input of the transformer stage.

    ``skip`` is the ``(projection, pooled)`` tuple of ``SplitProjPool`` (the pooled map is read from it);
    without it ``x`` is taken to be the pooled map.
    """

    has_residual = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor, skip=None) -> torch.Tensor:
        pooled = skip[1] if isinstance(skip, tuple) else x
        return q8(self.project(pooled, None))


class SplitQKVAttn(nn.Module):
    """Window attention over 8x8 pixel windows of a grid shifted by ``spec.shift_x / shift_y`` tiles (0 or -1).

    ``shift_x`` moves the grid along the columns, ``shift_y`` along the rows. Per head: ``Q, K`` are normalised per token (float16 sum of squares), ``Q`` is scaled by the head's gain,
    both are rounded to e4m3; ``logits = Q K^T + bias``; the softmax uses the network's own float16
    exponential and rounds the probabilities to e4m3; the context ``P V`` is rounded to e4m3. Tiles of a
    shifted window that lie outside the map are zero tokens and take part as keys; their outputs are dropped.
    """

    EXP_SLOPE, EXP_OFFSET = 0.044921875, 1.30078125
    EXP_LOW, EXP_HIGH = 1.03125, 1.5693359375
    SUM_FLOOR = 6.103515625e-05
    NORM_FLOOR = 0.000062

    def __init__(self, spec: LayerSpec, weights: Weights):
        super().__init__()
        block, layer = spec.block, spec.layer
        self.shift_x, self.shift_y = spec.shift_x, spec.shift_y
        if self.shift_x not in (0, -1) or self.shift_y not in (0, -1):
            raise ValueError(f"unsupported window shift ({self.shift_x}, {self.shift_y})")
        qkv = weights.e4m3(block, layer, "qkv", HEADS * 3 * HEAD_DIM, CHANNELS)
        self.register_buffer("qkv", qkv.t().contiguous())                               # [512, 16 * 96]
        self.register_buffer("bias", self.unpack_bias(weights.f16(block, layer, "attn_pos_bias", HEADS, 4096)))
        self.register_buffer("scale", weights.f32(block, layer, "tail", HEADS))

    @staticmethod
    def unpack_bias(raw: torch.Tensor) -> torch.Tensor:
        """Per-head ``[64, 64]`` position biases from the order of the matrix unit's accumulator fragments."""
        i = torch.arange(64)[:, None]
        j = torch.arange(64)[None, :]
        lane = 4 * ((i % 16) % 8) + ((j % 16) % 8) // 2
        slot = (j % 2) | (((i % 16) // 8) << 1) | (((j % 16) // 8) << 2)
        index = (4 * (i // 16) + (j // 16)) * 256 + lane * 8 + slot
        return raw[:, index.reshape(-1)].reshape(HEADS, 64, 64).contiguous()

    @classmethod
    def exponential(cls, logits: torch.Tensor) -> torch.Tensor:
        """The network's float16 exponential: an affine map into the exponent field of a float16.

        ``y = clamp(f16(x) * 0.04492 + 1.3008, 1.03125, 1.5693)`` and the result is the float16 whose bit pattern is
        the ten mantissa bits of ``y`` shifted left by five.
        """
        y = _fma_half(logits.to(torch.float16), torch.tensor(cls.EXP_SLOPE, dtype=torch.float16),
                      torch.tensor(cls.EXP_OFFSET, dtype=torch.float16))
        y = y.clamp(cls.EXP_LOW, cls.EXP_HIGH)
        bits = (y.view(torch.int16) & 0x3FF) << 5
        return bits.view(torch.float16).to(torch.float32)

    @classmethod
    def normalise(cls, values: torch.Tensor) -> torch.Tensor:
        """Divide each 32-vector by its length with the network's float16 reduction order.

        ``values``: float16 ``[..., 32]``. Squares are summed as ``d`` with ``d + 16``, then with ``d + 8``, then the
        eight results pairwise as ``(0+4, 2+6)`` and ``(1+5, 3+7)``.
        """
        squares = values * values
        pairs = squares[..., :16] + squares[..., 16:]
        eights = pairs[..., :8] + pairs[..., 8:]
        quads = eights[..., :4] + eights[..., 4:]                    # (0+4, 1+5, 2+6, 3+7)
        total = (quads[..., 0] + quads[..., 2]) + (quads[..., 1] + quads[..., 3])
        floor = torch.tensor(cls.NORM_FLOOR, dtype=torch.float16)
        norm = torch.rsqrt(torch.maximum(total, floor).to(torch.float32)).to(torch.float16)
        return values * norm.unsqueeze(-1)

    def windows(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """``[1, C, rows, cols]`` -> ``[windows, 64, C]`` on the shifted grid, plus the geometry to undo it.

        A window's 64 tokens run over its four 4x4 tiles (top-left, top-right, bottom-left, bottom-right), each
        tile row-major: the order the position bias is stored in.
        """
        _, channels, height, width = x.shape
        top, left = -self.shift_y * TILE, -self.shift_x * TILE
        rows = -(-(height + top) // WINDOW)
        cols = -(-(width + left) // WINDOW)
        padded = F.pad(x, (left, cols * WINDOW - width - left, top, rows * WINDOW - height - top))
        tiles = padded.reshape(channels, rows, 2, TILE, cols, 2, TILE).permute(1, 4, 2, 5, 3, 6, 0)
        return tiles.reshape(rows * cols, WINDOW * WINDOW, channels), (rows, cols, top, left)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        _, channels, height, width = x.shape
        tokens, (rows, cols, top, left) = self.windows(x)
        count = tokens.shape[0]
        qkv = int8.mm(tokens, self.qkv, "split").reshape(count, 64, HEADS, 3, HEAD_DIM)
        query = self.normalise(qkv[..., 0, :].to(torch.float16))
        key = self.normalise(qkv[..., 1, :].to(torch.float16))
        query = q8((query * self.scale.to(torch.float16)[:, None]).to(torch.float32))
        key = q8(key.to(torch.float32))
        value = _store(qkv[..., 2, :])

        logits = torch.einsum("wihd,wjhd->whij", query, key) + self.bias
        exponent = self.exponential(logits)                                          # [windows, heads, query, key]
        partial = exponent.reshape(count, HEADS, 64, 4, 2, 8)                        # key = 16 tile + 8 half + 1 c
        columns = ((partial[..., 0, :, :] + partial[..., 1, :, :]) + partial[..., 2, :, :]) + partial[..., 3, :, :]
        pairs = columns[..., 0::2] + columns[..., 1::2]                              # (0+1, 2+3, 4+5, 6+7)
        halves = (pairs[..., 0] + pairs[..., 1]) + (pairs[..., 2] + pairs[..., 3])
        total = halves[..., 0] + halves[..., 1]
        inverse = 1.0 / total.clamp_min(self.SUM_FLOOR)
        probability = q8(_half(exponent * inverse.unsqueeze(-1)))
        context = _store(torch.einsum("whij,wjhd->wihd", probability, value))
        context = context.reshape(rows, cols, 2, 2, TILE, TILE, channels)
        out = context.permute(6, 0, 2, 4, 1, 3, 5).reshape(channels, rows * WINDOW, cols * WINDOW)
        return out[None, :, top:top + height, left:left + width].contiguous()


BUILDERS = {
    "SplitFfwd": SplitFfwd,
    "SplitFfwdProj": SplitFfwdProj,
    "SplitQKVAttn": SplitQKVAttn,
    "SplitProj": SplitProj,
    "SplitProjPool": SplitProjPool,
    "SplitFinalHead": SplitFinalHead,
}
