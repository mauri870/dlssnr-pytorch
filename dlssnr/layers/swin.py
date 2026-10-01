"""The windowed-attention blocks of the encoder and decoder (``Swin1H`` .. ``Swin8H``).

One block is an MLP followed by multi-head self-attention, each with a scaled residual::

    y   = rs  * x + Wc . q(mid(q(act(We . q(x)))))             MLP      (act = cubic SiLU)
    out = ars * y + Wo . q(attention(q(y)))                    attention

``q`` rounds onto the e4m3 grid. Every rounding point, the f16 intermediates between them and the order
the f16 sums are taken in are reproduced as the network does them, because a one-step difference at any
of them moves the picture. Attention runs inside 8x8 pixel windows (2x2 tiles of 4x4 pixels, 64 tokens).
Alternate layers move the window grid by half a window; pixels the shifted grid hangs over the frame edge
read as zero and are not written back.

Four variants wrap the body:

* ``ds``        - the body, then a 2x2 mean and a learned ``C -> 2C`` projection (the next level's input);
* ``upsample``  - a learned ``2C -> C`` projection of the coarser level, nearest 2x replication, a gain-scaled
                  blend with the encoder skip, then the body;
* ``inpview`` / ``outview`` only name the layout of the neighbouring layer's buffer and compute the plain body.

Feature maps are float32 ``[1, C, rows, columns]``; ``swin_core`` is the entry point the first and last
blocks of the network reuse, because they run the same body on f16 data.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import int8
from ..plan import LayerSpec
from ..quant import q8
from ..runtime import drain, fuse, triton_kernels
from ..weights import Weights

WINDOW = 8                   # pixels along each side of an attention window
TOKENS = WINDOW * WINDOW
TILE = 4                     # pixels along each side of a tile; a window is 2x2 tiles
HEAD_DIM = 32

EXP_SLOPE = 0.044921875
EXP_OFFSET = 1.30078125
EXP_LOW, EXP_HIGH = 1.03125, 1.5693359375   # clamp of the exponent ramp (both exact in f16)
NORM_FLOOR = float(torch.tensor(0.000062, dtype=torch.float16))   # lower bound of the squared norm (f16-exact)
SUM_FLOOR = 2.0 ** -14                      # lower bound of the softmax denominator
ACT_LIMIT = 4.0
ACT_CUBIC, ACT_QUADRATIC, ACT_LINEAR = -0.055908203125, 0.447265625, 0.89453125

# windows are processed in chunks so that the largest temporary (the attention logits) stays this size
_CHUNK_ELEMENTS = 1 << 22


def _f16(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest f16 value (ties to even); the result stays float32."""
    return x.to(torch.float16).to(torch.float32)


def _fma(a: torch.Tensor, b: float | torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """``a * b + c`` with a single rounding to float32 (the product is exact in float64)."""
    return (a.double() * torch.as_tensor(b, dtype=torch.float64) + torch.as_tensor(c, dtype=torch.float64)).float()


def _rsqrt(x: torch.Tensor) -> torch.Tensor:
    """float32 ``1 / sqrt(x)`` with the same bits on every device.

    A GPU's ``rsqrt`` is an approximation that differs from the CPU's in the last bit for a fifth of the
    inputs; the square root rounded to float32 (taken in float64, where it is exact on any device) followed
    by an exact division is the CPU's result.
    """
    return 1.0 / torch.sqrt(x.double()).float()


def activation(x: torch.Tensor) -> torch.Tensor:
    """The network's cubic SiLU, ``x * (0.894 + t * (0.447 - 0.056 |t|))`` with ``t = clamp(x, -4, 4)``.

    The two nested multiply-adds are fused (one rounding each) and the whole expression stays in float32.
    """
    t = x.clamp(-ACT_LIMIT, ACT_LIMIT)
    inner = _fma(t.abs(), ACT_CUBIC, ACT_QUADRATIC)
    return x * _fma(t, inner, ACT_LINEAR)


# ---- parameters -----------------------------------------------------------------------------------------

def _deswizzle_bias(raw: torch.Tensor) -> torch.Tensor:
    """One head's 64x64 position bias from the fragment order it is stored in to ``[query, key]``.

    The table is stored as sixteen 16x16 (query block, key block) tiles, each tile in the lane order of an
    accumulator fragment: 32 lanes of eight values.
    """
    i = np.arange(TOKENS)[:, None]
    j = np.arange(TOKENS)[None, :]
    lane = 4 * ((i % 16) % 8) + ((j % 16) % 8) // 2
    slot = (j % 2) | (((i % 16) // 8) << 1) | (((j % 16) // 8) << 2)
    index = (4 * (i // 16) + (j // 16)) * 256 + lane * 8 + slot
    return raw[torch.from_numpy(index.reshape(-1))].reshape(TOKENS, TOKENS)


def _residual_table(weights: Weights, block: int, layer: int) -> tuple[torch.Tensor, int]:
    """The ``residual_scale`` table and where the MLP scales start in it.

    The table is zero padded in front (eight entries) except on the wide upsample layers, where it is the
    C scales followed by sixteen entries that belong to the layer's skip gain.
    """
    table = weights.f16(block, layer, "residual_scale", -1)
    padded = table.numel() >= 8 and bool((table[:8] == 0).all())
    return table, 8 if padded else 0


class SwinParams(nn.Module):
    """Everything one block reads, decoded into the form the arithmetic wants and held as buffers.

    Buffers (shapes, all float32): ``expand`` ``[4C, C]``, ``mid`` ``[C, 4C/heads]`` (block diagonal over
    heads, ``None`` for a single head), ``contract`` ``[C, 4C]`` or ``[C, C]``, ``qkv`` ``[3C, C]`` (per head:
    q rows, k rows, v rows), ``out_proj`` ``[C, C]`` (all e4m3 values); ``exp_bias`` ``[heads, 64, 64]``, the
    position bias already carried through the exponent ramp; ``head_scale`` ``[heads]`` as stored; and the
    f16-exact residual scales ``mlp_scale`` and ``attn_scale``, ``[C]``.
    """

    def __init__(self, channels: int, heads: int, **buffers: torch.Tensor | None):
        super().__init__()
        self.channels, self.heads = channels, heads
        for name, value in buffers.items():
            self.register_buffer(name, value)

    @classmethod
    def from_weights(cls, weights: Weights, block: int, layer: int, channels: int, heads: int) -> "SwinParams":
        hidden = 4 * channels
        mid = weights.e4m3(block, layer, "mlp_mid", channels, hidden // heads) if heads > 1 else None
        contract_width = channels if heads > 1 else hidden
        bias = weights.f16(block, layer, "attn_pos_bias", heads * TOKENS * TOKENS).reshape(heads, -1)
        table, start = _residual_table(weights, block, layer)
        return cls(
            channels, heads,
            expand=weights.e4m3(block, layer, "mlp_expand", hidden, channels), mid=mid,
            contract=weights.e4m3(block, layer, "mlp_contract", channels, contract_width),
            qkv=weights.e4m3(block, layer, "qkv", 3 * channels, channels),
            out_proj=weights.e4m3(block, layer, "attn_out_proj", channels, channels),
            exp_bias=_fma(torch.stack([_deswizzle_bias(row) for row in bias]), EXP_SLOPE, EXP_OFFSET),
            head_scale=weights.f32(block, layer, "scalars_b", -1)[:heads].contiguous(),
            mlp_scale=table[start:start + channels].contiguous(),
            attn_scale=weights.f16(block, layer, "attn_residual_scale", -1)[:channels].contiguous(),
        )


# ---- window partition -----------------------------------------------------------------------------------

def _window_grid(extent: int, shift: int) -> tuple[int, int]:
    """(windows along one axis, zero pixels in front) for a grid moved by ``shift`` tiles."""
    if shift not in (-1, 0):
        raise ValueError(f"window shift of {shift} tiles is not supported")
    front = -TILE * shift
    return -(-(extent + front) // WINDOW), front


def _to_windows(x: torch.Tensor, shift_rows: int, shift_cols: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """``[1, C, rows, columns]`` -> ``[windows, 64, C]`` with tokens in tile order (tile row, tile column, y, x)."""
    _, channels, height, width = x.shape
    rows, front_h = _window_grid(height, shift_rows)
    cols, front_w = _window_grid(width, shift_cols)
    padded = F.pad(x, (front_w, cols * WINDOW - width - front_w, front_h, rows * WINDOW - height - front_h))
    tiles = padded.reshape(channels, rows, 2, TILE, cols, 2, TILE)           # c, wy, qy, y, wx, qx, x
    tokens = tiles.permute(1, 4, 2, 5, 3, 6, 0).reshape(rows * cols, TOKENS, channels)
    return tokens, (rows, cols, front_h, front_w)


def _from_windows(tokens: torch.Tensor, grid: tuple[int, int, int, int], height: int, width: int) -> torch.Tensor:
    rows, cols, front_h, front_w = grid
    channels = tokens.shape[-1]
    tiles = tokens.reshape(rows, cols, 2, 2, TILE, TILE, channels).permute(6, 0, 2, 4, 1, 3, 5)
    padded = tiles.reshape(1, channels, rows * WINDOW, cols * WINDOW)
    return padded[:, :, front_h:front_h + height, front_w:front_w + width]


# ---- the block body, on one chunk of windows ------------------------------------------------------------

@fuse
def _activate(x: torch.Tensor) -> torch.Tensor:
    """``q8(activation(x))`` as one kernel."""
    return q8(activation(x))


round8 = fuse(q8)


@fuse
def store(x: torch.Tensor) -> torch.Tensor:
    """float32 -> f16 -> e4m3, the double rounding of a stored result, as one kernel."""
    return q8(_f16(x))


def _linear(x: torch.Tensor, weight: torch.Tensor, activate: bool = False) -> torch.Tensor:
    """``x @ weight.T`` for e4m3 operands; with ``activate`` the e4m3-rounded cubic SiLU of it.

    On a GPU this is one float16 matrix kernel with the float32 accumulator rounded in its epilogue.
    """
    kernels = None if "swin" in int8.sites else triton_kernels(x)
    if kernels:
        return kernels.gemm.linear(x, weight, activate)
    product = int8.mm(x, weight.T, "swin")
    return _activate(product) if activate else product


def _mlp(params: SwinParams, quantised: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """``rs * x + Wc . hidden`` for ``[n, 64, C]`` tokens, still in float32 (rounded by the caller).

    The activation goes straight from the float32 accumulator to e4m3; the grouped middle projection is
    narrowed to f16 first.
    """
    hidden = _linear(quantised, params.expand, activate=True)
    if params.mid is not None:
        groups = hidden.float().reshape(*hidden.shape[:-1], params.heads, -1)
        mid = params.mid.reshape(params.heads, HEAD_DIM, -1)
        hidden = store(torch.einsum("ntgj,gdj->ntgd", groups, mid).reshape(*hidden.shape[:-1], -1))
    return residual * params.mlp_scale + _linear(hidden, params.contract)


def _norm_scale(values: torch.Tensor) -> torch.Tensor:
    """Reciprocal L2 norm over the 32 head dimensions (``[..., 32] -> [...]``), float32.

    The squares are accumulated by fused multiply-add in float32, in four interleaved chains over each of
    the two halves of the dimension set (even and odd dimensions); the halves are narrowed to f16 and added in
    f16. A sum that overflows f16 gives a scale of zero. The reciprocal root itself is not narrowed.
    """
    squares = values.reshape(*values.shape[:-1], 2, 2, 4, 2).double()        # fragment, component group, chain, half
    squares = squares * squares
    total = torch.zeros_like(squares[..., 0, 0, :, :])                       # [..., chain, half]
    for group in range(2):
        for fragment in range(2):
            total = (squares[..., fragment, group, :, :] + total).float().double()
    chains = total.float()
    halves = ((chains[..., 0, :] + chains[..., 1, :]) + (chains[..., 2, :] + chains[..., 3, :])).to(torch.float16)
    total = (halves[..., 0] + halves[..., 1]).clamp(min=NORM_FLOOR)
    return _rsqrt(total.float())


def _softmax_numerators(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """exp(logit + bias) as the network builds it: an affine ramp rounded to f16 and clamped, whose bit
    pattern is shifted into the exponent field. ``[n, heads, q, k]`` -> f16 tensor of the same shape."""
    ramp = _fma(logits, EXP_SLOPE, bias).to(torch.float16).clamp(EXP_LOW, EXP_HIGH)
    return ((ramp.view(torch.int16) & 0x3FF) << 5).view(torch.float16)


def _denominator(numerators: torch.Tensor) -> torch.Tensor:
    """Sum over the 64 keys in the order the original adds them, in f16.

    The keys of a query sit in two lane halves, each holding eight of every 16-key tile block as four
    pairs: key = 16 block + 4 pair + 2 member + half. Blocks are added first, then pairs, then the two
    halves, then the two members of a pair.
    """
    keys = numerators.reshape(*numerators.shape[:-1], 4, 4, 2, 2)           # block, pair, member, half
    blocks = ((keys[..., 0, :, :, :] + keys[..., 1, :, :, :]) + keys[..., 2, :, :, :]) + keys[..., 3, :, :, :]
    pairs = ((blocks[..., 0, :, :] + blocks[..., 1, :, :]) + blocks[..., 2, :, :]) + blocks[..., 3, :, :]
    halves = pairs[..., 0] + pairs[..., 1]
    return halves[..., 0] + halves[..., 1]


@fuse
def _normalise_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, head_scale: torch.Tensor):
    return (q8(query * (_norm_scale(query) * head_scale).unsqueeze(-1)),
            q8(key * _norm_scale(key).unsqueeze(-1)),
            q8(_f16(value)))


@fuse
def _softmax(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """e4m3 probabilities from ``[n, heads, q, k]`` logits and the exponent-ramp bias."""
    numerators = _softmax_numerators(logits, bias)
    reciprocal = 1.0 / _denominator(numerators).float().clamp(min=SUM_FLOOR)
    return q8(_f16(numerators.float() * reciprocal.unsqueeze(-1)))


def _attention(params: SwinParams, skip: torch.Tensor, quantised: torch.Tensor) -> torch.Tensor:
    """``ars * skip + Wo . context`` for the quantised MLP output ``quantised`` (``[n, 64, C]``).

    Q and K are scaled straight from the float32 accumulator (Q also by the head's scale); V, the
    probabilities and the context are narrowed to f16 before they reach the e4m3 grid.
    """
    heads, count = params.heads, quantised.shape[0]
    kernels = None if "swin" in int8.sites else triton_kernels(quantised)
    if kernels:
        qkv = _linear(quantised, params.qkv).reshape(count * TOKENS, -1)
        context = kernels.attention.window_attention(qkv, params.head_scale, params.exp_bias)
        return skip * params.attn_scale + _linear(context, params.out_proj).reshape(count, TOKENS, -1)
    qkv = _linear(quantised, params.qkv).reshape(count, TOKENS, heads, 3, HEAD_DIM)
    query, key, value = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
    query, key, value = _normalise_qkv(query, key, value, params.head_scale)
    logits = torch.einsum("ntgd,nsgd->ngts", query, key)                     # [n, head, query, key]
    probability = _softmax(logits, params.exp_bias)
    context = store(torch.einsum("ngts,nsgd->ntgd", probability, value).reshape(count, TOKENS, -1))
    return skip * params.attn_scale + _linear(context, params.out_proj)


def swin_block(params: SwinParams, x: torch.Tensor, shift_x: int, shift_y: int,
               x_wide: torch.Tensor | None = None, accumulator: bool = False):
    """Run one block on ``[1, C, rows, columns]``.

    Returns ``wide``, the f16-exact output before its final e4m3 rounding (the layer's output is
    ``q8(wide)``); with ``accumulator=True`` returns ``(wide, acc)`` where ``acc`` is the same output
    before the f16 rounding, which the 2x2 pooling of a downsample reads.

    ``x`` is what the matrix products read (rounded to e4m3); ``x_wide``, when given, is the f16 value
    the MLP residual uses instead (the first and last blocks feed f16 data). ``shift_x`` / ``shift_y`` are
    the plan's window shifts in tiles: ``shift_x`` moves the grid along the columns of the map (its last
    dimension), ``shift_y`` along the rows.
    """
    _, channels, height, width = x.shape
    kernels = None if "swin" in int8.sites else triton_kernels(x)
    if kernels and kernels.swin_fused.supported(params, x) and (x_wide is None or x_wide is x):
        fused = kernels.swin_fused.swin_block_fused(params, x, shift_y, shift_x, x_wide is not None, accumulator)
        return (fused[0].float(), fused[1]) if accumulator else fused.float()
    quantised, grid = _to_windows(round8(x), shift_y, shift_x)
    residual = quantised if x_wide is None else _to_windows(x_wide, shift_y, shift_x)[0]
    per_window = max(1, _CHUNK_ELEMENTS // (params.heads * TOKENS * TOKENS))
    out = torch.empty_like(quantised)
    for start in range(0, quantised.shape[0], per_window):
        chunk = slice(start, start + per_window)
        y = _mlp(params, quantised[chunk], residual[chunk])
        y_quantised = store(y)
        # one head keeps the unrounded MLP output as the attention skip; several heads use its e4m3 form
        skip = y if params.heads == 1 else y_quantised
        out[chunk] = _attention(params, skip, y_quantised)
        drain(out.device)
    acc = _from_windows(out, grid, height, width)
    wide = _f16(acc)
    return (wide, acc) if accumulator else wide


def swin_core(params: SwinParams, x: torch.Tensor, shift_x: int, shift_y: int, accumulator: bool = False):
    """``swin_block`` on f16-exact data: ``x`` doubles as the MLP residual."""
    return swin_block(params, x, shift_x, shift_y, x_wide=x, accumulator=accumulator)


# ---- resampling -----------------------------------------------------------------------------------------

@fuse
def _pool(accumulator: torch.Tensor) -> torch.Tensor:
    pairs = (accumulator[..., 0::2] + accumulator[..., 1::2]).to(torch.float16)
    total = pairs[..., 0::2, :] + pairs[..., 1::2, :]
    return q8(total.float() * 0.25)


def pool2x2(accumulator: torch.Tensor) -> torch.Tensor:
    """2x2 mean of a block's unrounded output, ``[1, C, 2a, 2b] -> [1, C, a, b]`` on the e4m3 grid.

    Pixel pairs along the columns are added in float32 and narrowed to f16, the two row sums are added in
    f16, and the quarter is exact.
    """
    return _pool(accumulator)


@fuse
def narrow(x: torch.Tensor) -> torch.Tensor:
    return _f16(x)


def _project(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """1x1 convolution ``[N, K] . [1, K, h, w]`` in float32, narrowed to f16 (still float32)."""
    return narrow(torch.einsum("nk,bkhw->bnhw", weight, x))


def _pad_to(x: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    return F.pad(x, (0, columns - x.shape[-1], 0, rows - x.shape[-2]))


def downsample(params: SwinParams, resample: torch.Tensor, x: torch.Tensor, shift_x: int, shift_y: int,
               rows: int, columns: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A block followed by pooling and the learned ``C -> 2C`` projection.

    Returns ``(next, body)``: the projected pooled map zero padded to ``rows x columns`` and the block's own
    e4m3 output.
    """
    wide, acc = swin_block(params, x, shift_x, shift_y, accumulator=True)
    pooled = store(torch.einsum("nk,bkhw->bnhw", resample, pool2x2(acc)))
    return _pad_to(pooled, rows, columns), round8(wide)


@fuse
def _gated_sum(skip: torch.Tensor, gain: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """``f16(skip * gain + other)`` with the product and sum taken in float64 (one rounding to f16)."""
    return (skip.double() * gain.double() + other.double()).to(torch.float16).float()


def upsample_blend(project: torch.Tensor, gain: torch.Tensor, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """``f16(skip * gain + P(x))`` on the skip's grid: ``P`` is the learned ``2C -> C`` projection of the
    coarser level, replicated 2x in both directions. Returns the f16-exact blend ``[1, C, rows, columns]``."""
    projected = _project(project, x)
    rows, columns = skip.shape[-2:]
    index_r = (torch.arange(rows, device=x.device) // 2).clamp(max=projected.shape[-2] - 1)
    index_c = (torch.arange(columns, device=x.device) // 2).clamp(max=projected.shape[-1] - 1)
    replicated = projected[:, :, index_r][:, :, :, index_c]
    return _gated_sum(skip, gain.reshape(1, -1, 1, 1), replicated)


# ---- layer modules --------------------------------------------------------------------------------------

class SwinLayer(nn.Module):
    """One ``CCTinlayoutFusedSwin*`` layer: forward(x, skip=None) as documented per variant below.

    * plain, ``inpview``, ``outview``: ``x`` ``[1, C, rows, columns]`` -> the block's e4m3 output.
    * ``ds``: -> ``(next, body)``; ``next`` is ``[1, 2C, out rows, out columns]``, ``body`` the full-resolution
      output (what the matching upsample layer reads as its skip).
    * ``upsample``: ``x`` ``[1, 2C, ...]`` is the previous level, ``skip`` the ``body`` of the matching ``ds``
      layer (a ``(next, body)`` tuple is accepted); -> ``[1, C, rows, columns]`` of the skip's size.
    """

    def __init__(self, spec: LayerSpec, weights: Weights):
        super().__init__()
        self.spec = spec
        channels = spec.channels
        self.params = SwinParams.from_weights(weights, spec.block, spec.layer, channels, spec.heads)
        resample = gain = None
        if spec.variant == "ds":
            resample = weights.e4m3(spec.block, spec.layer, "resample", 2 * channels, channels)
        elif spec.variant == "upsample":
            resample = weights.e4m3(spec.block, spec.layer, "resample", channels, 2 * channels)
            gain = weights.f16(spec.block, spec.layer, "upsample_gain", -1)
            if gain.numel() < channels:
                table, _ = _residual_table(weights, spec.block, spec.layer)
                gain = torch.cat([table[table.numel() - (channels - gain.numel()):], gain])
            gain = gain[:channels].contiguous()
        self.register_buffer("resample", resample)
        self.register_buffer("gain", gain)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, skip=None):
        spec = self.spec
        if spec.variant == "ds":
            return downsample(self.params, self.resample, x, spec.shift_x, spec.shift_y, spec.out_w, spec.out_h)
        if spec.variant == "upsample":
            skip = skip[1] if isinstance(skip, tuple) else skip
            blend = upsample_blend(self.resample, self.gain, x, skip)
            if spec.channels == 32:
                return round8(swin_block(self.params, blend, spec.shift_x, spec.shift_y, x_wide=blend))
            return round8(swin_block(self.params, round8(blend), spec.shift_x, spec.shift_y))
        return round8(swin_block(self.params, x, spec.shift_x, spec.shift_y))


BUILDERS: dict[str, Callable[[LayerSpec, Weights], nn.Module]] = {
    kind: SwinLayer for kind in ("Swin1H", "Swin2H", "Swin4H", "Swin8H")
}
