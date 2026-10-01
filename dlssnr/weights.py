"""The network's weights: raw tensors keyed by ``block{B}.layer{L}.{name}``.

Extraction stores every tensor of a layer as the exact bytes found in NVIDIA's library (a flat uint8
tensor), so nothing is lost or reinterpreted on the way. A layer reads the names it owns through the
typed accessors below, which view the bytes as the type and shape that layer defines.
"""
from __future__ import annotations

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .quant import e4m3_decode


class Weights:
    def __init__(self, tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None = None):
        self.tensors = tensors
        self.metadata = metadata or {}

    # ---- storage ---------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "Weights":
        with safe_open(path, framework="pt") as file:
            return cls({k: file.get_tensor(k) for k in file.keys()}, file.metadata())

    def save(self, path: str) -> None:
        save_file({k: v.contiguous() for k, v in self.tensors.items()}, path, metadata=self.metadata)

    # ---- access ----------------------------------------------------------------------------------
    @staticmethod
    def key(block: int, layer: int, name: str) -> str:
        return f"block{block}.layer{layer}.{name}"

    def has(self, block: int, layer: int, name: str) -> bool:
        return self.key(block, layer, name) in self.tensors

    def names(self, block: int, layer: int) -> list[str]:
        prefix = f"block{block}.layer{layer}."
        return sorted(k[len(prefix):] for k in self.tensors if k.startswith(prefix))

    def raw(self, block: int, layer: int, name: str) -> torch.Tensor:
        """The bytes of one tensor, flat uint8."""
        return self.tensors[self.key(block, layer, name)]

    def e4m3(self, block: int, layer: int, name: str, *shape: int) -> torch.Tensor:
        """An e4m3 tensor decoded exactly to float32, reshaped to ``shape``."""
        return e4m3_decode(self.raw(block, layer, name)).reshape(*shape)

    def f16(self, block: int, layer: int, name: str, *shape: int) -> torch.Tensor:
        return self.raw(block, layer, name).view(torch.float16).to(torch.float32).reshape(*shape)

    def f32(self, block: int, layer: int, name: str, *shape: int) -> torch.Tensor:
        return self.raw(block, layer, name).view(torch.float32).reshape(*shape)
