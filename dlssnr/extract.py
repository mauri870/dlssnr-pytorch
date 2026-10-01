"""Extract the network's weights from NVIDIA's ``nvngx_dlssnr.dll``.

The library is only ever read as bytes, never loaded or executed. The weights live in a PE resource
named ``WEIGHTS_HT``: a flat list of records, one per layer (``block{B}.layer{L}.layer``) plus a couple
of scalar records. Each record is an opaque payload whose internal layout depends on the layer family.
This module slices every payload into its named tensors and writes them, as exact bytes, to a
``dlssnr.weights.Weights`` file (safetensors) keyed ``block{B}.layer{L}.{name}``, flat uint8.

Envelope (little endian)
------------------------
Resource: ``u64 total_size`` (equals the resource size), then records until the end:

    u64 name_length, name (utf-8)
    u64 body_length
    body: u64 body_length, u64 payload_length, u32 device (0 or 1), payload,
          u32 type_id, u32 layout_id, u64 ndim, u32 dims[ndim]

Only the payload is used. The layer family is recognised from the payload size, which is unique per
family and width; a size nothing matches means the library holds different weights than the layouts
below describe, and extraction stops with an error instead of guessing.

Matrix packing
--------------
e4m3 matrices are stored as 512-byte slices of 16 output channels by 32 inputs, laid out the way a
tensor-core B operand is held per lane: lane ``4*n + j`` carries output ``n`` (0..7) and the byte
``2*q + p`` of its 8-byte half carries input ``8*q + 2*j + p``; the two halves of a lane's 16 bytes are
output rows ``n`` and ``n + 8`` of the slice. Slices are ordered with the output tile fastest inside a
group of ``inner`` tiles, then the input group, then the next group of tiles. The unpacked tensors are
plain row-major ``[N, K]`` (N outputs, K inputs).

Tensors per layer family
------------------------
All matrices are e4m3 ``[N, K]`` row-major. "f16" / "f32" tensors are little-endian floats. Sizes in
bytes. ``C`` is the working width of a Swin layer (the smaller of its in and out channels), ``H = 4C``,
``heads = C / 32``.

Fused Swin layers (``CCTinlayoutFusedSwin{1,2,4,8}H``), equal width, downsample (``C -> 2C``) or
upsample (``2C -> C``):

    mlp_expand            e4m3 [H, C]
    mlp_mid               e4m3 [C, H/heads]        only when heads > 1 (grouped middle matrix)
    mlp_contract          e4m3 [C, H]              heads == 1
                          e4m3 [C, C]              heads > 1
    resample              e4m3 [c_out, c_in]       upsample layers only, stored before residual_scale
    residual_scale        f16 [C + 16]             MLP skip scale (2C + 32 bytes), see below
    upsample_gain         f16 [C] (C = 32) or [C - 16] (C > 32)   upsample layers only
    qkv                   e4m3 [3C, C]             rows: q, k, v blocks of C
    attn_pos_bias         f16 [heads * 4096]       per head a 64x64 bias, in accumulator-fragment order
    scalars_b             f32 [max(4, heads)]      per-head q/k scale, first ``heads`` entries used
    attn_out_proj         e4m3 [C, C]
    attn_residual_scale   f16 [C + 8]              attention skip scale (2C + 16 bytes)
                          f16 [C]                  downsample layers
    resample              e4m3 [c_out, c_in]       downsample layers only, stored after attn_residual_scale
    tail                  16 raw bytes             downsample layers with C = 32 only

``residual_scale`` and ``attn_residual_scale`` hold C scales padded with zeros (eight leading and
trailing halves at C = 32); on upsample layers wider than 32 the first sixteen halves of the gain
overlap the end of ``residual_scale``, so the layer must read the region by its content.
``attn_pos_bias`` for head h, query i, key j (64 tokens of a window) is element
``(4*(i//16) + j//16)*256 + 8*(4*((i%16)%8) + ((j%16)%8)//2) + ((j%2) | (((i%16)//8) << 1) | (((j%16)//8) << 2))``
of that head's 4096 halves.

Pre-block image adapter (block 0, ``CCTinlayoutFusedPreBlockSwin1H``, C = 32, heads = 1): the C = 32
equal-width tensors above (``mlp_expand``, ``mlp_contract``, ``residual_scale`` (f16 [40]), ``qkv``,
``attn_pos_bias``, ``scalars_b``, ``attn_out_proj``, ``attn_residual_scale``) plus

    input_lift            f16 [32, 16]             row-major, sixteen colour features to 32 channels

Post-block image head (block 70, ``CCTinlayoutFusedPostBlockSwin1H``, C = 32, heads = 1): the same
equal-width tensors (``residual_scale`` is f16 [32]) plus

    main_gain             f16 [32]                 scale of the upsampled main input
    skip_gain             f16 [32]                 scale of the encoder skip input
    out_project           f16 [16, 32]             row-major, 32 channels to 16 output features
    blend_scale           f16 [1]                  record ``block70.layer0.blend_scale``

Split Swin layers (512 channels, 16 heads):

    CCSplitSwin16HFfwd         ffwd_a e4m3 [512, 512];  ffwd_b e4m3 [512, 512]
    CCSplitSwin16HFfwdProj     weight e4m3 [512, 512];  skip_weight f16 [512]
    CCSplitSwin16HProj         weight e4m3 [512, 512];  skip_weight f16 [512]
    CCSplitSwin16HProjPool     weight e4m3 [512, 512];  skip_weight f16 [512]
    CCSplitSwin16HQKVAttn      qkv e4m3 [1536, 512];  attn_pos_bias f16 [16 * 4096];  tail f32 [16]
                               (the tail record is 64 bytes: sixteen per-head f32 scales)
    CCSplitSwin16HFinalHead    weight e4m3 [1024, 512];  tail raw 16 bytes
    CCDecInputUpsample         weight e4m3 [512, 1024];  skip_weight f16 [512]

Transformer layers on the 1024-channel latent:

    CCVit1DQKV                 header raw 128 bytes;  q, k, v e4m3 [1024, 1024] each
    CCVit1DFfnExpand           weight e4m3 [4096, 1024];  tail raw 16 bytes
    CCVit1DFfnContract         weight e4m3 [1024, 4096];  skip_weight f16 [1024]
    CCVit1DProjection          weight e4m3 [1024, 1024];  skip_weight f16 [1024]
    CCVit1DAttention           scale f16 [1]

A ``skip_weight`` is the weight of the residual input in the layer's variance-preserving sum.
"""
from __future__ import annotations

import argparse
import hashlib
import mmap
import re
import struct
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np
import torch

from .weights import Weights

FORMAT = "dlssnr-weights-1"
LIBRARY_NAME = "nvngx_dlssnr.dll"
RESOURCE_NAME = "WEIGHTS_HT"

SLICE_BYTES = 512  # 16 outputs x 32 inputs of e4m3


class ExtractError(ValueError):
    """The input is not a library this extractor understands."""


# ---------------------------------------------------------------------------------------------------
# Tensor layouts
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Part:
    """One tensor inside a record's payload."""

    name: str
    start: int
    stop: int
    kind: str            # "e4m3" matrix, "raw" bytes, "qkv3" (three packed matrices), "proj16"
    shape: tuple[int, ...] = ()
    inner: int = 0       # output tiles per group for packed e4m3 matrices


class _Layout:
    """Builds the parts of one record, front to back."""

    def __init__(self) -> None:
        self.parts: list[Part] = []
        self.cursor = 0

    def skip(self, size: int) -> "_Layout":
        self.cursor += size
        return self

    def at(self, offset: int) -> "_Layout":
        self.cursor = offset
        return self

    def raw(self, name: str, size: int) -> "_Layout":
        self.parts.append(Part(name, self.cursor, self.cursor + size, "raw"))
        self.cursor += size
        return self

    def matrix(self, name: str, rows: int, cols: int, inner: int) -> "_Layout":
        size = rows * cols
        self.parts.append(Part(name, self.cursor, self.cursor + size, "e4m3", (rows, cols), inner))
        self.cursor += size
        return self

    def special(self, name: str, size: int, kind: str) -> "_Layout":
        self.parts.append(Part(name, self.cursor, self.cursor + size, kind))
        self.cursor += size
        return self

    @property
    def size(self) -> int:
        return self.parts[-1].stop


def _swin_layout(width: int, c_in: int | None = None, c_out: int | None = None) -> _Layout:
    """A fused Swin layer of working width ``width``; ``c_in != c_out`` makes it a resampling layer."""
    heads = width // 32
    hidden = 4 * width
    resample = c_in * c_out if c_in != c_out else 0
    upsample = bool(resample) and c_out < c_in
    layout = _Layout()
    if heads == 1:
        layout.matrix("mlp_expand", hidden, width, hidden // 16)
        layout.matrix("mlp_contract", width, hidden, width // 16)
    else:
        layout.matrix("mlp_expand", hidden, width, hidden // heads // 16)
        layout.matrix("mlp_mid", width, hidden // heads, width // heads // 16)
        layout.matrix("mlp_contract", width, width, width // 16)
    if upsample:
        layout.matrix("resample", c_out, c_in, c_out // 16)
    layout.raw("residual_scale", 2 * width + 32)
    if upsample:
        layout.raw("upsample_gain", 2 * width if width == 32 else 2 * width - 32)
    layout.matrix("qkv", 3 * width, width, 3 * width // 16)
    layout.raw("attn_pos_bias", heads * 8192)
    layout.raw("scalars_b", max(16, 4 * heads))
    layout.matrix("attn_out_proj", width, width, width // 16)
    if resample and not upsample:
        layout.raw("attn_residual_scale", 2 * width)
        layout.matrix("resample", c_out, c_in, c_out // 16)
        if width == 32:
            layout.raw("tail", 16)
    else:
        layout.raw("attn_residual_scale", 2 * width + 16)
    return layout


def _body_32(layout: _Layout) -> None:
    """The C = 32 attention-plus-MLP body shared by the pre-block and post-block records."""
    layout.matrix("qkv", 96, 32, 6)
    layout.raw("attn_pos_bias", 8192)
    layout.raw("scalars_b", 16)
    layout.matrix("attn_out_proj", 32, 32, 2)
    layout.raw("attn_residual_scale", 80)


def _pre_block_layout() -> _Layout:
    layout = _Layout()
    layout.matrix("mlp_expand", 128, 32, 8).matrix("mlp_contract", 32, 128, 2).skip(16)
    layout.raw("input_lift", 1024).raw("residual_scale", 80)
    _body_32(layout)
    return layout


def _post_block_layout() -> _Layout:
    layout = _Layout()
    layout.matrix("mlp_expand", 128, 32, 8).matrix("mlp_contract", 32, 128, 2).skip(16)
    layout.raw("residual_scale", 64).raw("main_gain", 64).raw("skip_gain", 64)
    _body_32(layout)
    layout.special("out_project", 1024, "proj16")
    return layout


def _split_layouts() -> dict[str, _Layout]:
    def square(*extra: tuple[str, int]) -> _Layout:
        layout = _Layout().matrix("weight", 512, 512, 32)
        for name, size in extra:
            layout.raw(name, size)
        return layout

    return {
        "CCSplitSwin16HFfwd": _Layout().matrix("ffwd_a", 512, 512, 32).matrix("ffwd_b", 512, 512, 32),
        "CCSplitSwin16HFfwdProj": square(("skip_weight", 1024)),
        "CCSplitSwin16HQKVAttn": _Layout().matrix("qkv", 1536, 512, 96)
        .raw("attn_pos_bias", 16 * 8192).raw("tail", 64),
        "CCSplitSwin16HFinalHead": _Layout().matrix("weight", 1024, 512, 64).raw("tail", 16),
        "CCDecInputUpsample": _Layout().matrix("weight", 512, 1024, 32).raw("skip_weight", 1024),
    }


def _vit_layouts() -> dict[str, _Layout]:
    return {
        "CCVit1DQKV": _Layout().raw("header", 128).special("qkv", 3 * 1024 * 1024, "qkv3"),
        "CCVit1DFfnExpand": _Layout().matrix("weight", 4096, 1024, 256).raw("tail", 16),
        "CCVit1DFfnContract": _Layout().matrix("weight", 1024, 4096, 64).raw("skip_weight", 2048),
        "CCVit1DProjection": _Layout().matrix("weight", 1024, 1024, 64).raw("skip_weight", 2048),
    }


def _layouts_by_size() -> dict[int, tuple[str, _Layout]]:
    """Every known layer layout, keyed by payload size (unique across families)."""
    table: dict[int, tuple[str, _Layout]] = {}

    def register(family: str, layout: _Layout) -> None:
        if layout.size in table:
            raise AssertionError(f"layout size {layout.size} is not unique ({family})")
        table[layout.size] = (family, layout)

    register("CCTinlayoutFusedPreBlockSwin1H", _pre_block_layout())
    register("CCTinlayoutFusedPostBlockSwin1H", _post_block_layout())
    for width in (32, 64, 128, 256):
        register(f"fused Swin C={width}", _swin_layout(width, width, width))
        register(f"fused Swin C={width} downsample", _swin_layout(width, width, 2 * width))
        register(f"fused Swin C={width} upsample", _swin_layout(width, 2 * width, width))
    for family, layout in {**_split_layouts(), **_vit_layouts()}.items():
        register(family, layout)
    return table


# ProjPool and Proj share the FfwdProj layout, whose size is therefore registered once above.
LAYOUTS = _layouts_by_size()
ATTENTION_SCALE_SIZE = 2


# ---------------------------------------------------------------------------------------------------
# Unpacking
# ---------------------------------------------------------------------------------------------------


def unswizzle_matrix(payload: np.ndarray, start: int, rows: int, cols: int, inner: int) -> np.ndarray:
    """Packed 512-byte slices to a row-major ``[rows, cols]`` uint8 matrix."""
    tiles, groups = rows // 16, cols // 32
    packed = payload[start:start + rows * cols].reshape(tiles // inner, groups, inner, 8, 4, 2, 4, 2)
    # axes: outer, group, tile, n, j, half, q, p -> rows (outer, tile, half, n), cols (group, q, j, p)
    return packed.transpose(0, 2, 5, 3, 1, 6, 4, 7).reshape(rows, cols)


def unpack_qkv3(payload: np.ndarray, start: int) -> np.ndarray:
    """The 1024-wide transformer QKV: three matrices interleaved in 1024-byte tiles of 32x32 inputs."""
    rows = np.arange(1024)[:, None]
    cols = np.arange(1024)[None, :]
    within = ((cols & 1) | ((cols >> 1 & 1) << 4) | ((cols >> 2 & 1) << 5) | ((cols >> 3 & 1) << 1)
              | ((cols >> 4 & 1) << 2) | ((rows & 1) << 6) | ((rows >> 1 & 1) << 7)
              | ((rows >> 2 & 1) << 8) | ((rows >> 3 & 1) << 3) | ((rows >> 4 & 1) << 9))
    tile = (rows >> 5) | ((cols >> 5) << 5)
    return np.stack([payload[start + (3 * tile + which) * 1024 + within] for which in range(3)])


def unpack_proj16(raw: np.ndarray) -> np.ndarray:
    """The 16x32 half-precision output projection (bytes) from its two 16-input operand blocks."""
    halves = raw.view(np.uint16)
    n = np.arange(16)[:, None]
    k = np.arange(32)[None, :]
    index = ((k // 16) * 256 + 32 * (n % 8) + 8 * ((k % 8) // 2) + 4 * (n // 8) + k % 2
             + 2 * ((k % 16) // 8))
    return halves[index].view(np.uint8).reshape(-1)


def _is_nan_free(matrix: np.ndarray) -> bool:
    """A trained e4m3 matrix has no NaN encodings (0x7f, 0xff); an offset error produces them fast."""
    return not bool(((matrix & 0x7F) == 0x7F).any())


def unpack_record(block: int, layer: int, payload: np.ndarray) -> dict[str, np.ndarray]:
    """Split one ``block{B}.layer{L}.layer`` payload into its named tensors (flat uint8)."""
    where = f"block{block}.layer{layer}"
    if payload.size == ATTENTION_SCALE_SIZE:
        return {"scale": payload.copy()}
    if payload.size not in LAYOUTS:
        raise ExtractError(
            f"{where}: a {payload.size}-byte payload matches no layer layout this extractor knows; "
            "the library holds different weights than the ones it was written for")
    family, layout = LAYOUTS[payload.size]
    tensors: dict[str, np.ndarray] = {}
    for part in layout.parts:
        if part.kind == "e4m3":
            rows, cols = part.shape
            matrix = unswizzle_matrix(payload, part.start, rows, cols, part.inner)
            if not _is_nan_free(matrix):
                raise ExtractError(f"{where}.{part.name} ({family}): decoded matrix contains NaN "
                                   "codes; the record layout does not match")
            tensors[part.name] = matrix.reshape(-1).copy()
        elif part.kind == "qkv3":
            for name, matrix in zip("qkv", unpack_qkv3(payload, part.start)):
                if not _is_nan_free(matrix):
                    raise ExtractError(f"{where}.{name} ({family}): decoded matrix contains NaN codes")
                tensors[name] = matrix.reshape(-1).copy()
        elif part.kind == "proj16":
            tensors[part.name] = unpack_proj16(payload[part.start:part.stop])
        else:
            tensors[part.name] = payload[part.start:part.stop].copy()
    return tensors


# ---------------------------------------------------------------------------------------------------
# The library file
# ---------------------------------------------------------------------------------------------------


def _sections(image: mmap.mmap) -> tuple[list[tuple[int, int, int]], tuple[int, int]]:
    """Section table as (rva, file offset, file size) and the resource directory (rva, size)."""
    if image[:2] != b"MZ":
        raise ExtractError("not a Windows PE file (no MZ header)")
    pe_offset, = struct.unpack_from("<I", image, 0x3C)
    if image[pe_offset:pe_offset + 4] != b"PE\0\0":
        raise ExtractError("not a Windows PE file (no PE signature)")
    section_count, = struct.unpack_from("<H", image, pe_offset + 6)
    optional_size, = struct.unpack_from("<H", image, pe_offset + 20)
    optional = pe_offset + 24
    magic, = struct.unpack_from("<H", image, optional)
    if magic != 0x20B:
        raise ExtractError("expected a 64-bit PE file")
    resource = struct.unpack_from("<II", image, optional + 128)
    table = optional + optional_size
    sections = []
    for i in range(section_count):
        _, _, rva, size, offset = struct.unpack_from("<8sIIII", image, table + 40 * i)
        sections.append((rva, offset, size))
    return sections, resource


def _find_weights_resource(image: mmap.mmap) -> tuple[int, int]:
    """File offset and size of the ``WEIGHTS_HT`` resource."""
    sections, (resource_rva, resource_size) = _sections(image)

    def file_offset(rva: int, size: int) -> int:
        for start, offset, available in sections:
            if start <= rva and rva + size <= start + available:
                return offset + rva - start
        raise ExtractError(f"resource data at RVA {rva:#x} is not backed by the file")

    if not resource_rva:
        raise ExtractError(f"no resource section: this is not an NVIDIA DLSS-NR library ({RESOURCE_NAME} missing)")
    root = file_offset(resource_rva, resource_size)

    def walk(directory: int, path: tuple[str, ...], depth: int):
        if depth > 6:
            raise ExtractError("resource tree is too deep")
        named, numbered = struct.unpack_from("<HH", image, root + directory + 12)
        for i in range(named + numbered):
            name, target = struct.unpack_from("<II", image, root + directory + 16 + 8 * i)
            label = str(name)
            if name & 0x80000000:
                at = root + (name & 0x7FFFFFFF)
                length, = struct.unpack_from("<H", image, at)
                label = image[at + 2:at + 2 + 2 * length].decode("utf-16le")
            if target & 0x80000000:
                yield from walk(target & 0x7FFFFFFF, path + (label,), depth + 1)
            else:
                rva, size = struct.unpack_from("<II", image, root + target)
                yield path + (label,), rva, size

    for path, rva, size in walk(0, (), 0):
        if RESOURCE_NAME in path:
            return file_offset(rva, size), size
    raise ExtractError(f"no {RESOURCE_NAME} resource: this is not an NVIDIA DLSS-NR library")


_RECORD_NAME = re.compile(r"block(\d+)\.layer(\d+)\.(\w+)")


def iter_records(image: mmap.mmap, offset: int, size: int):
    """Yield ``(record name, payload offset, payload size)`` for every record of the resource."""
    total, = struct.unpack_from("<Q", image, offset)
    if total != size:
        raise ExtractError(f"weight resource declares {total} bytes but is {size}")
    position = offset + 8
    end = offset + size
    while position < end:
        name_length, = struct.unpack_from("<Q", image, position)
        if not 0 < name_length <= 4096:
            raise ExtractError("corrupt weight record (name length)")
        name = bytes(image[position + 8:position + 8 + name_length]).decode("utf-8")
        position += 8 + name_length
        body_length, = struct.unpack_from("<Q", image, position)
        position += 8
        inner_length, payload_length, device = struct.unpack_from("<QQI", image, position)
        if inner_length != body_length or device not in (0, 1) or not 0 < payload_length <= body_length:
            raise ExtractError(f"corrupt weight record {name!r}")
        yield name, position + 20, payload_length
        position += body_length
    if position != end:
        raise ExtractError("weight records do not end at the resource boundary")


def _open_library(path: Path, scratch: Path) -> tuple[BinaryIO, str, int]:
    """Open the library (or the one inside a zip), returning it with its sha256 and size.

    A zip member is inflated to a scratch file in one streaming pass so the image can be memory mapped
    instead of held in memory.
    """
    digest = hashlib.sha256()
    size = 0
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.infolist() if Path(m.filename).name.lower() == LIBRARY_NAME]
            if len(members) != 1:
                raise ExtractError(f"expected exactly one {LIBRARY_NAME} in {path.name}, found {len(members)}")
            target = scratch / LIBRARY_NAME
            with archive.open(members[0]) as source, open(target, "wb") as sink:
                while chunk := source.read(1 << 22):
                    digest.update(chunk)
                    sink.write(chunk)
                    size += len(chunk)
        return open(target, "rb"), digest.hexdigest(), size
    handle = open(path, "rb")
    while chunk := handle.read(1 << 22):
        digest.update(chunk)
        size += len(chunk)
    return handle, digest.hexdigest(), size


def extract(path: str, out_path: str) -> Weights:
    """Read the library (or a zip holding it) at ``path`` and save its weights to ``out_path``."""
    source = Path(path)
    if not source.is_file():
        raise ExtractError(f"no such file: {path}")
    with tempfile.TemporaryDirectory(prefix="dlssnr-") as scratch:
        handle, sha256, size = _open_library(source, Path(scratch))
        failure = ""
        if size == 0:
            handle.close()
            raise ExtractError(f"{path} is empty")
        with handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as image:
            try:
                offset, resource_size = _find_weights_resource(image)
                records = list(iter_records(image, offset, resource_size))
                if not records:
                    raise ExtractError("the weight resource is empty")
                tensors = _unpack_all(image, records)
            except (ExtractError, struct.error, UnicodeDecodeError) as error:
                # Re-raised outside the mapping: the traceback would otherwise keep views of it alive.
                failure = str(error) if isinstance(error, ExtractError) else f"truncated or corrupt file ({error})"
        if failure:
            raise ExtractError(failure)
    metadata = {"format": FORMAT, "source": LIBRARY_NAME, "dll_sha256": sha256, "dll_size": str(size),
                "records": str(len(records))}
    weights = Weights(tensors, metadata)
    weights.save(out_path)
    return weights


def _unpack_all(image: mmap.mmap, records: Sequence[tuple[str, int, int]]) -> dict[str, torch.Tensor]:
    layer_records = [r for r in records if _RECORD_NAME.fullmatch(r[0])]
    if len(layer_records) != len(records):
        unknown = next(r[0] for r in records if not _RECORD_NAME.fullmatch(r[0]))
        raise ExtractError(f"unexpected weight record name {unknown!r}")
    blocks = sorted({int(_RECORD_NAME.fullmatch(name)[1]) for name, _, _ in records})
    tensors: dict[str, torch.Tensor] = {}
    for block in blocks:
        count = nbytes = 0
        for name, start, length in records:
            match = _RECORD_NAME.fullmatch(name)
            if int(match[1]) != block:
                continue
            layer, tail = int(match[2]), match[3]
            payload = np.frombuffer(image, dtype=np.uint8, count=length, offset=start)
            if tail == "layer":
                parts = unpack_record(block, layer, payload)
            else:
                parts = {tail: payload.copy()}
            for part_name, data in parts.items():
                tensors[Weights.key(block, layer, part_name)] = torch.from_numpy(np.ascontiguousarray(data))
                count += 1
                nbytes += data.size
        print(f"block {block:2d}/{blocks[-1]}: {count} tensors, {nbytes / 2**20:.2f} MiB", file=sys.stderr)
    return tensors


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry: ``extract <dll-or-zip> -o weights.safetensors``."""
    parser = argparse.ArgumentParser(prog="dlssnr extract", description="Extract the network weights from nvngx_dlssnr.dll.")
    parser.add_argument("source", help="nvngx_dlssnr.dll, or a zip that contains it")
    parser.add_argument("-o", "--output", default="weights.safetensors", help="output file (default: %(default)s)")
    args = parser.parse_args(argv)
    try:
        weights = extract(args.source, args.output)
    except ExtractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    megabytes = sum(t.numel() for t in weights.tensors.values()) / 2**20
    print(f"wrote {args.output}: {len(weights.tensors)} tensors, {megabytes:.1f} MiB, "
          f"dll sha256 {weights.metadata['dll_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
