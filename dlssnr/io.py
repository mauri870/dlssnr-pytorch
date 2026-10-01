"""Frames in, frames out: image files, the network's input tensor and the controls of the library.

The network's tensors are ``[1, C, rows, columns]``: the frame in its natural orientation, padded to the
working extent of ``plan.padding``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .plan import padding


@dataclass
class Controls:
    """The controls the library takes, with the defaults the reference frames were made with.

    Only the single-frame path exists so far: ``depth``, ``motion``, ``history`` and ``reset`` are
    accepted so a caller can already pass what the temporal mode will need, and are ignored.
    """
    intensity: float = 1.0           # blend between the input frame (0) and the network's output (1); 0..2
    style: int = 0                   # 0..2
    local_tone: float = 1.0          # 0..2, strength of the low-frequency (tone) edit
    local_structure: float = 1.0     # 0..2, strength of the fine-detail edit
    skin_structure: float = -1.0     # -1: follow ``local_structure``; otherwise the detail strength on skin
    auto_mask: bool = True           # derive the skin / other-region split automatically
    seed: int = 0                    # noise seed; the library counts frames, so a single frame uses 0
    depth: np.ndarray | None = None      # not used by the single-frame path
    motion: np.ndarray | None = None     # not used by the single-frame path
    history: np.ndarray | None = None    # not used by the single-frame path
    reset: bool = False                  # not used by the single-frame path

    def __post_init__(self) -> None:
        if not 0.0 <= self.intensity <= 2.0:
            raise ValueError("intensity must be within 0..2")
        if self.style not in (0, 1, 2):
            raise ValueError("style must be 0, 1 or 2")
        if not 0.0 <= self.local_tone <= 2.0 or not 0.0 <= self.local_structure <= 2.0:
            raise ValueError("local_tone and local_structure must be within 0..2")
        if not -1.0 <= self.skin_structure <= 2.0:
            raise ValueError("skin_structure must be within -1..2")

    def conditioning(self) -> tuple[float, float, float, float, float]:
        """The five conditioning features (style, tone, structure, skin, other) the pre-block receives.

        With the automatic mask the structure feature is fixed at 1 and the mask itself carries the
        strengths: skin takes ``skin_structure`` (or ``local_structure`` when that is -1), everything else
        takes ``local_structure``. Without the mask both mask features are -1 and the structure feature
        carries ``local_structure``.
        """
        if not self.auto_mask:
            return self.style / 128.0, self.local_tone, self.local_structure, -1.0, -1.0
        skin = self.local_structure if self.skin_structure < 0 else self.skin_structure
        return self.style / 128.0, self.local_tone, 1.0, skin, self.local_structure


def load_image(path: str) -> np.ndarray:
    """An 8-bit RGB image as ``uint8 [height, width, 3]`` (alpha is dropped)."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def save_image(path: str, frame: np.ndarray) -> None:
    """Write ``uint8 [height, width, 3]`` as an RGB PNG."""
    Image.fromarray(np.ascontiguousarray(frame), "RGB").save(path)


def _mirror(index: torch.Tensor, extent: int) -> torch.Tensor:
    """Reflect an out-of-range index once about the last pixel, then clamp what is still outside."""
    reflected = torch.where(index < extent, index, 2 * extent - index - 2)
    return reflected.clamp(0, extent - 1)


def frame_to_tensor(frame: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
    """``uint8 [height, width, 3]`` -> float32 ``[1, 3, rows, columns]``, values k/255, padded to the working extent.

    The padding rows and columns repeat the frame mirrored about its last row/column, as the network
    expects them. The frame is moved to ``device`` as bytes and converted there.
    """
    height, width, _ = frame.shape
    work = padding(width, height)
    # a table rather than ``/ 255``: a GPU divides by a constant as a multiplication by its reciprocal
    levels = (torch.arange(256, dtype=torch.float32) / 255.0).to(device)
    pixels = levels[torch.from_numpy(frame).to(device).long()].permute(2, 0, 1)             # [3, height, width]
    columns = _mirror(torch.arange(work.width, device=device), width)
    rows = _mirror(torch.arange(work.height, device=device), height)
    return pixels[:, rows][:, :, columns].unsqueeze(0).contiguous()


def tensor_to_frame(tensor: torch.Tensor, width: int, height: int) -> np.ndarray:
    """float32 ``[1, 3, rows, columns]`` in [0, 1] -> ``uint8 [height, width, 3]``, cropped and rounded to 8 bits.

    Values round half up (``floor(255 x + 0.5)``) after clamping, which is how the library stores 8-bit
    frames.
    """
    scaled = tensor[0, :, :height, :width].clamp(0.0, 1.0) * 255.0
    whole = scaled.floor()
    codes = whole + (scaled - whole >= 0.5).to(scaled.dtype)
    return codes.permute(1, 2, 0).to(torch.uint8).cpu().numpy().copy()
