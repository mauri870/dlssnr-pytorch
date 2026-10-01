"""The whole network: one module per layer, run in plan order with the skip connections the plan names.

``NRNet`` pads the frame to the working extent, runs the 152 layers and crops the result. Which tensor a
layer reads is decided here, from the plan, so the layer modules stay free of graph knowledge:

* every layer reads the main output of the layer before it (the pre block reads the padded frame);
* an upsample layer and the post block also read the full-resolution output that the matching
  downsample (or the pre block) set aside before pooling;
* the decoder entry reads the projection of the last split block of the encoder, the final head its pooled
  copy, and a split block's last projection the output of that block's feed-forward projection.

A layer that produces more than one tensor returns a tuple; element 0 is what the next layer reads.
"""
from __future__ import annotations

import importlib
from typing import Callable

import numpy as np
import torch
from torch import nn

from . import int8
from .plan import LayerSpec, build_plan, padding
from .runtime import clear_caches
from .weights import Weights

_FAMILIES = ("pre_post", "swin", "split_swin", "vit", "decoder")


def _registry() -> dict[str, Callable[[LayerSpec, Weights], nn.Module]]:
    """Builders of every layer family that is available, keyed by ``LayerSpec.kind``."""
    builders: dict[str, Callable[[LayerSpec, Weights], nn.Module]] = {}
    for family in _FAMILIES:
        try:
            module = importlib.import_module(f"{__package__}.layers.{family}")
        except ModuleNotFoundError as error:
            if error.name != f"{__package__}.layers.{family}":
                raise
            continue
        builders.update(module.BUILDERS)
    return builders


def _main(value):
    return value[0] if isinstance(value, tuple) else value


class NRNet(nn.Module):
    """The enhancement network for frames of one size."""

    def __init__(self, weights: Weights, width: int, height: int,
                 builders: dict[str, Callable[[LayerSpec, Weights], nn.Module]] | None = None):
        super().__init__()
        # Cached per-weight copies are keyed by address; a network built after another was freed may reuse it.
        clear_caches()
        self.width, self.height = width, height
        self.plan = build_plan(width, height)
        self.padding = padding(width, height)
        builders = builders if builders is not None else _registry()
        missing = sorted({s.kind for s in self.plan} - builders.keys())
        if missing:
            raise NotImplementedError(f"no layer implementation available for: {', '.join(missing)}")
        self.layers = nn.ModuleList(builders[s.kind](s, weights) for s in self.plan)
        self.skip_source = self._skip_sources()
        self.last_use = self._last_uses()
        # Layers to leave out (their output is their input); only shape-preserving layers can be skipped.
        # This is an experiment hook for ablation studies; an empty set is the real network.
        self.skip: set[int] = set()

    @property
    def device(self) -> torch.device:
        """The device the layers live on (move them with ``.to(device)``)."""
        return next(self.buffers()).device

    def _skip_sources(self) -> list[int | None]:
        """For every layer, the index of the layer whose output it takes as its skip input."""
        index = {(s.block, s.layer): s.index for s in self.plan}
        sources: list[int | None] = []
        for s in self.plan:
            source = None
            if s.kind in ("SplitProj", "SplitProjPool", "SplitFinalHead"):
                # The projections add the feed-forward projection back; the final head reads the pooled copy.
                source = index[s.block, 1 if s.kind != "SplitFinalHead" else 3]
            elif s.kind == "VitFfnContract":
                source = index[s.block - 1, 4]
            elif s.kind == "VitProjection":
                source = index[s.block, 1]
            elif s.skip_block >= 0:
                source = index[s.skip_block, 3 if s.skip_block == 30 else 0]
            sources.append(source)
        return sources

    def _last_uses(self) -> list[int]:
        """Index of the last layer that reads each layer's output, so tensors can be freed early."""
        last = list(range(len(self.plan)))
        for i, source in enumerate(self.skip_source):
            if source is not None:
                last[source] = max(last[source], i)
        for i in range(1, len(self.plan)):
            last[i - 1] = max(last[i - 1], i)
        return last

    @torch.no_grad()
    def forward(self, frame: np.ndarray, controls=None, capture: bool = False):
        """Enhance ``frame`` (uint8 ``[height, width, 3]``, a numpy array on the host).

        Every layer runs on ``self.device``; only the input and the output frame cross the host boundary.

        Returns the enhanced uint8 frame; with ``capture=True`` returns ``(frame, outputs)`` where
        ``outputs[i]`` is what layer ``i`` returned.
        """
        from .io import Controls, frame_to_tensor, tensor_to_frame

        controls = controls if controls is not None else Controls()
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(f"network built for {self.width}x{self.height}, got {frame.shape[1]}x{frame.shape[0]}")
        image = frame_to_tensor(frame, self.device)
        outputs = self.run(image, controls, capture)
        result = tensor_to_frame(_main(outputs[len(self.plan) - 1]), self.width, self.height)
        return (result, outputs) if capture else result

    def run(self, image: torch.Tensor, controls=None, capture: bool = False) -> dict[int, object]:
        """Run every layer on the padded ``image`` tensor and return the kept outputs."""
        outputs: dict[int, object] = {}
        for spec, layer in zip(self.plan, self.layers):
            if hasattr(layer, "controls") and controls is not None:
                layer.controls = controls
            i = spec.index
            int8.current = spec.block
            skip = self.skip_source[i]
            skip_value = None if skip is None else outputs[skip]
            previous = outputs[i - 1] if i else None
            if i in self.skip:
                outputs[i] = previous
            elif spec.kind == "PreBlock":
                outputs[i] = layer(image)
            elif spec.kind == "PostBlock":
                outputs[i] = layer(_main(previous), skip_value[1], image)
            elif spec.kind == "SplitFfwdProj":
                outputs[i] = layer(previous)
            else:
                if spec.kind == "DecInputUpsample":
                    skip_value = _main(skip_value)
                outputs[i] = layer(_main(previous), skip_value)
            if not capture:
                for j in [j for j in outputs if self.last_use[j] <= i and j != i]:
                    del outputs[j]
        return outputs

