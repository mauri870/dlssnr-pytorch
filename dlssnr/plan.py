"""The network as a list of layers: what each one is, how big its tensors are, and what feeds it.

``LayerSpec`` describes one layer. ``build_plan(width, height)`` returns the 152 specs for a frame of that
size (the geometry of every level depends on the frame size).

Axis convention: the frame keeps its natural orientation. A layer's ``w`` is the number of image rows
(padded) and ``h`` the number of image columns (padded); the names are historical. Feature maps are float32
tensors ``[1, C, w, h]`` = ``[1, C, rows, columns]`` whose values lie on the e4m3 grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class LayerSpec:
    index: int            # position in the plan
    block: int            # block number (0..70)
    layer: int            # layer number inside the block
    kind: str             # layer type, e.g. "TFSwin1H", "SSFfwd", "VAttention" (see ``KINDS``)
    variant: str          # "", "inpview", "ds", "upsample", "outview", "proj_pool", "final_head", ...
    w: int                # number of rows of this layer's grid
    h: int                # number of columns of this layer's grid
    tokens: int           # tokens the layer processes (padded)
    channels: int         # channel count of the layer's body
    heads: int            # attention heads
    c_in: int             # input channels
    c_out: int            # output channels
    shifted: bool         # the window grid of this layer is shifted
    shift_x: int          # window shift, -1 / 0 / 1 steps along x, as in the plan
    shift_y: int
    input_block: int      # block whose output feeds this layer
    skip_block: int       # block whose output is the skip input (-1: none)
    name: str             # the full layer name
    out_w: int = 0        # rows of the tensor this layer produces (differs from ``w`` across resamplings)
    out_h: int = 0        # columns of the tensor this layer produces
    grid_x: int = 0       # windows (tiles) across the columns for this layer's window grid, shift included
    grid_y: int = 0       # windows (tiles) down the rows


KINDS = {
    "CCTinlayoutFusedPreBlockSwin1H": "PreBlock",
    "CCTinlayoutFusedSwin1H": "Swin1H", "CCTinlayoutFusedSwin2H": "Swin2H",
    "CCTinlayoutFusedSwin4H": "Swin4H", "CCTinlayoutFusedSwin8H": "Swin8H",
    "CCSplitSwin16HFfwd": "SplitFfwd", "CCSplitSwin16HFfwdProj": "SplitFfwdProj",
    "CCSplitSwin16HQKVAttn": "SplitQKVAttn", "CCSplitSwin16HProj": "SplitProj",
    "CCSplitSwin16HProjPool": "SplitProjPool", "CCSplitSwin16HFinalHead": "SplitFinalHead",
    "CCVit1DFfnExpand": "VitFfnExpand", "CCVit1DFfnContract": "VitFfnContract", "CCVit1DQKV": "VitQKV",
    "CCVit1DAttention": "VitAttention", "CCVit1DProjection": "VitProjection",
    "CCDecInputUpsample": "DecInputUpsample",
    "CCTinlayoutFusedPostBlockSwin1H": "PostBlock",
}


# ---- the layer table ---------------------------------------------------------------------------------

_VARIANTS = ("inpview", "ds", "upsample", "outview", "proj_pool", "final_head")
_SHIFT_CYCLE = ((0, 0), (-1, -1), (-1, 0), (0, -1))

PLAIN, PRE, POST, DOWN, SWIN_UP, DECODER_UP, FINAL_HEAD = range(7)


class _Row(NamedTuple):
    block: int
    layer: int
    cls: str
    name: str
    channels: int
    heads: int
    c_in: int
    c_out: int
    shift: tuple[int, int]
    role: int                       # how the layer changes the extent (PLAIN, PRE, ...)
    skip: tuple[int, int] | None    # (block, layer) of the layer whose extent the output takes over


def _rows() -> list[_Row]:
    rows: list[_Row] = []

    def swin(block, heads, channels, tag, shift, c_in=None, c_out=None, role=PLAIN, skip=None):
        stem = f"cc_tinlayout_fused_swin_{heads}h_{channels}_{heads}"
        rows.append(_Row(block, 0, f"CCTinlayoutFusedSwin{heads}H", f"{stem}_{tag}fp8", channels, heads,
                         c_in or channels, c_out or channels, shift, role, skip))

    rows.append(_Row(0, 0, "CCTinlayoutFusedPreBlockSwin1H", "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8",
                     3, 0, 3, 32, (0, 0), PRE, None))

    # Encoder: four levels of Swin blocks, each ending in a downsample that doubles the channels.
    block = 1
    for heads, channels, count in ((1, 32, 4), (2, 64, 4), (4, 128, 6), (8, 256, 8)):
        for i in range(count):
            first, last = i == 0, i == count - 1
            tag = "inpview_" if first else "ds_" if last else ""
            swin(block, heads, channels, tag, _SHIFT_CYCLE[i % 4], c_out=2 * channels if last else None,
                 role=DOWN if last else PLAIN)
            block += 1

    def split_block(block, shift, first=False, last=False, pool=False):
        pre = "cc_split_swin_16h_"
        views = "_inpview" if first else ""
        rows.extend([
            _Row(block, 0, "CCSplitSwin16HFfwd", f"{pre}ffwd{views}_512_fp8", 512, 4, 512, 512, (0, 0), PLAIN, None),
            _Row(block, 1, "CCSplitSwin16HFfwdProj", f"{pre}ffwd_proj{views}_512_fp8", 512, 4, 512, 512,
                 (0, 0), PLAIN, None),
            _Row(block, 2, "CCSplitSwin16HQKVAttn", f"{pre}qkv_512_fp8", 512, 4, 512, 512, shift, PLAIN, None),
        ])
        if pool:
            rows.append(_Row(block, 3, "CCSplitSwin16HProjPool", f"{pre}proj_pool_512_fp8", 512, 4, 512, 512,
                             (0, 0), PLAIN, None))
            rows.append(_Row(block, 4, "CCSplitSwin16HFinalHead", f"{pre}final_head_512_fp8", 512, 8, 512, 1024,
                             (0, 0), FINAL_HEAD, None))
        elif last:
            rows.append(_Row(block, 3, "CCSplitSwin16HProj", f"{pre}proj_512_outview_fp8", 512, 4, 512, 512,
                             (0, 0), PLAIN, None))
        else:
            rows.append(_Row(block, 3, "CCSplitSwin16HProj", f"{pre}proj_512_fp8", 512, 8, 512, 512,
                             (0, 0), PLAIN, None))

    for i in range(8):
        split_block(23 + i, _SHIFT_CYCLE[i % 4], first=i == 0, pool=i == 7)

    # Bottleneck: eight transformer blocks over the flattened tokens.
    vit = (("ffn_expand", "FfnExpand", 1024, 4096), ("ffn_contract", "FfnContract", 4096, 1024),
           ("qkv", "QKV", 1024, 1024), ("attention", "Attention", 1024, 1024),
           ("projection", "Projection", 1024, 1024))
    for block in range(31, 39):
        for layer, (stem, cls, c_in, c_out) in enumerate(vit):
            rows.append(_Row(block, layer, f"CCVit1D{cls}", f"cc_vit_1d_{stem}_fp8", 1024, 4, c_in, c_out,
                             (0, 0), PLAIN, None))

    # Decoder: the bottleneck output is upsampled onto the final-head extent, then split blocks, then one
    # Swin level per encoder level, each opened by an upsample that takes the matching encoder skip.
    rows.append(_Row(39, 0, "CCDecInputUpsample", "cc_dec_input_upsample_1024_512_fp8", 512, 1, 1024, 512,
                     (0, 0), DECODER_UP, (30, 3)))
    for i in range(8):
        split_block(40 + i, _SHIFT_CYCLE[i % 4], last=i == 7)

    block = 48
    for heads, channels, count, start, skip_block in ((8, 256, 8, 0, 22), (4, 128, 6, 2, 14),
                                                      (2, 64, 4, 0, 8), (1, 32, 4, 0, 4)):
        for i in range(count):
            first, last = i == 0, i == count - 1
            tag = "upsample_" if first else "outview_" if last else ""
            swin(block, heads, channels, tag, _SHIFT_CYCLE[(start + i) % 4],
                 c_in=2 * channels if first else None, role=SWIN_UP if first else PLAIN,
                 skip=(skip_block, 0) if first else None)
            block += 1
    rows.append(_Row(70, 0, "CCTinlayoutFusedPostBlockSwin1H", "cc_tinlayout_fused_post_block_swin_1h_32_fp8",
                     3, 0, 32, 3, (-1, -1), POST, None))
    return rows


def _variant(name: str) -> str:
    for variant in _VARIANTS:
        if f"_{variant}_" in name:
            return variant
    return ""


def _up(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


class _Shape(NamedTuple):
    w: int
    h: int


def _walk(frame_w: int, frame_h: int, rows: list[_Row]) -> tuple[list[tuple[_Shape, _Shape]], int, int]:
    """Extent before and after every layer for a frame of ``frame_w`` x ``frame_h``, plus the number of
    times each axis is halved along the way.

    Extents are (rows, columns), so the walk starts from (frame_h, frame_w).
    """
    current = previous = _Shape(frame_h, frame_w)
    shapes: list[tuple[_Shape, _Shape]] = []
    index = {(row.block, row.layer): i for i, row in enumerate(rows)}
    halvings_x = halvings_y = 0
    last_block = -1
    for row in rows:
        if row.block != last_block:
            halvings_x += current.h < previous.h
            halvings_y += current.w < previous.w
            previous, last_block = current, row.block
        out = current
        if row.role in (PRE, DOWN, FINAL_HEAD):
            out = _Shape(_up((current.w + 1) // 2, 4), _up((current.h + 1) // 2, 4))
        elif row.role in (SWIN_UP, DECODER_UP):
            before, after = shapes[index[row.skip]]
            # An encoder skip is taken at the extent it had before its downsample; the bottleneck skip at
            # the extent the final head produced.
            out = before if row.role == SWIN_UP else after
            if row.role == SWIN_UP and row.c_out == 32:
                out = _Shape(_up(out.w, 8), _up(out.h, 8))
        elif row.role == POST:
            out = _Shape(frame_h, frame_w)
        shapes.append((current, out))
        current = out
    return shapes, halvings_x, halvings_y


class Padding(NamedTuple):
    """Working extent of the frame and how much the network pads it by (in frame pixels)."""
    width: int
    height: int
    pad_x: int
    pad_y: int


def padding(width: int, height: int) -> Padding:
    """The extent the frame is padded to before entering the network, and the padding added.

    Every halving of an axis needs its extent divisible by two, so each axis is rounded up to a multiple
    of 2**halvings, with a floor of 320. A frame already aligned to full 4-pixel cells at every level gets
    one extra unit so that the shifted windows have room.
    """
    if not (0 < width <= 16384 and 0 < height <= 16384):
        raise ValueError(f"unsupported frame size {width}x{height}")
    _, halvings_x, halvings_y = _walk(width, height, _ROWS)
    align_x, align_y = 1 << halvings_x, 1 << halvings_y
    padded_w = max(320, _up(width, align_x))
    padded_h = max(320, _up(height, align_y))
    if padded_h % (align_y * 4) == 0 and padded_w % (align_x * 4) == 0:
        padded_w += align_x
    return Padding(padded_w, padded_h, padded_w - width, padded_h - height)


def working_extent(width: int, height: int) -> tuple[int, int]:
    """(width, height) in frame pixels of the padded frame the network processes."""
    padded = padding(width, height)
    return padded.width, padded.height


def build_plan(width: int, height: int) -> list[LayerSpec]:
    """The 152 layer specs for a ``width`` x ``height`` frame."""
    padded = padding(width, height)
    shapes, _, _ = _walk(padded.width, padded.height, _ROWS)
    specs: list[LayerSpec] = []
    for index, (row, (current, out)) in enumerate(zip(_ROWS, shapes)):
        plan = current
        if row.role == FINAL_HEAD:
            plan = out
        elif row.role == DECODER_UP:
            plan = _Shape(_up(out.w, 4), _up(out.h, 4))
        elif row.role == POST:
            plan = _Shape(padded.height // 2, padded.width // 2)
        tokens = plan.w * plan.h if 31 <= row.block <= 38 else _up(plan.w, 8) * _up(plan.h, 8)
        grid = out if row.role in (POST, SWIN_UP) else plan
        sx, sy = row.shift
        skip_block = row.skip[0] if row.skip else 0 if row.role == POST else -1
        specs.append(LayerSpec(
            index=index, block=row.block, layer=row.layer, kind=KINDS[row.cls], variant=_variant(row.name),
            w=plan.w, h=plan.h, tokens=tokens, channels=row.channels, heads=row.heads,
            c_in=row.c_in, c_out=row.c_out, shifted=bool(sx or sy) and row.role != POST,
            shift_x=sx, shift_y=sy, input_block=row.block - 1 if row.layer == 0 else row.block,
            skip_block=skip_block, name=row.name, out_w=out.w, out_h=out.h,
            grid_x=(grid.h - 4 * sx + 7) // 8, grid_y=(grid.w - 4 * sy + 7) // 8))
    return specs


_ROWS = _rows()
