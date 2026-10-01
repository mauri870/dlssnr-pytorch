"""The two ends of the network: the pre-block that turns a colour frame into features and the post-block
that turns features back into a frame.

Both are a C=32 Swin block with something around it. Their bodies run in f16 rather than on the e4m3 grid:
the pre-block's input is the lifted image and the post-block's input is an f16 blend of two levels, and the
post-block's output goes straight to the RGB projection without being rounded to e4m3.

Layout: images are ``[1, 3, rows, columns]`` float32 in natural orientation, padded to the working extent
(see ``dlssnr.io``). Feature maps are ``[1, 32, rows, columns]``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn

from ..io import Controls
from ..plan import LayerSpec
from ..weights import Weights
from ..runtime import fuse
from .swin import SwinParams, narrow, pool2x2, round8, swin_core

CHANNELS = 32
NOISE_GAIN = 1.0      # gain of the three noise features
CONSTANT_FEATURE = 1.0
COLOUR_SCALE = 0.125  # image colour is centred on 0.5 and scaled into the lift's working range
OUTPUT_SCALE = 0.25   # weight of the network's residual in the output colour


def _f16(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest f16 value; the result stays float32."""
    return x.to(torch.float16).to(torch.float32)


# ---- noise ---------------------------------------------------------------------------------------------

def _pcg(v: np.ndarray) -> np.ndarray:
    v = (v >> ((v >> np.uint32(28)) + np.uint32(4))) ^ v
    return v * np.uint32(0x108EF2D9)


def _unit(stream: np.ndarray) -> np.ndarray:
    """Uniform number in (0, 1] from a 32-bit stream value."""
    t = _pcg(stream)
    return (((t >> np.uint32(30)) ^ (t >> np.uint32(8))) + np.uint32(1)).astype(np.float32) * np.float32(2.0 ** -24)


def gaussian_noise(extent_x: int, extent_y: int, seed: int) -> torch.Tensor:
    """Three standard-normal fields ``[3, extent_y, extent_x]`` from a counter-based generator.

    Each pixel hashes its coordinates and the seed with a PCG-style mixer into four uniform numbers that
    feed two Box-Muller transforms; the third field is the cosine half of the second transform. It is a
    pure function of (x, y, seed), so it needs no state and any region can be generated independently.
    """
    with np.errstate(over="ignore"):
        x = np.arange(extent_x, dtype=np.uint32)[None, :]
        y = np.arange(extent_y, dtype=np.uint32)[:, None]
        base = (x * np.uint32(0x8DA6B343)) ^ (np.uint32(seed) * np.uint32(0x9E3779B9)) \
            ^ (y * np.uint32(0xD8163841)) ^ np.uint32(0x243F6A88)
        t = _pcg(base)
        h = (t >> np.uint32(22)) ^ t

        def stream(multiplier: int, offset: int) -> np.ndarray:
            return _unit(h * np.uint32(multiplier) + np.uint32(offset))

        u_a, u_b = stream(0x2C9277B5, 0xAC564B05), stream(0xFA6DC5F9, 0x4712A88E)
        u_c, u_d = stream(0xCAA5B80D, 0x21DD796B), stream(0x83232C31, 0x3463E0AC)
        two_pi = np.float32(6.28318530718)
        radius_a = np.sqrt(np.float32(-2.0) * np.log(u_a))
        radius_c = np.sqrt(np.float32(-2.0) * np.log(u_c))
        angle_a, angle_c = u_b * two_pi, u_d * two_pi
        fields = np.stack([radius_a * np.cos(angle_a), radius_a * np.sin(angle_a), radius_c * np.cos(angle_c)])
    return torch.from_numpy(fields.astype(np.float32))


# ---- pre-block -----------------------------------------------------------------------------------------

@fuse
def _centred_colour(image: torch.Tensor) -> torch.Tensor:
    """The image centred on 0.5 and scaled into the lift's range, on the f16 grid."""
    return _f16(_f16(_f16(image) - 0.5) * COLOUR_SCALE)


def _lift_matrix(weights: Weights, block: int, layer: int) -> torch.Tensor:
    """The learned 16 -> 32 input lift as ``[32, 16]`` float32 (f16 values).

    The stored record is arranged for the matrix hardware: rows in groups of 16, inside a group the
    channel's low three bits select a 32-element stride, and the 16 inputs are split into halves and
    pairs. This undoes that arrangement.
    """
    stored = weights.f16(block, layer, "input_lift", 512)
    channel = torch.arange(32)[:, None]
    k = torch.arange(16)[None, :]
    offset = (channel // 16) * 256 + 32 * (channel % 8) + 8 * ((k % 8) // 2) + 4 * ((channel // 8) % 2) \
        + (k % 2) + 2 * (k // 8)
    return stored[offset]


class PreBlock(nn.Module):
    """Block 0: image -> 32 features, the block's Swin body, and the first 2x down-sampling.

    The 16 input features of a pixel are three noise values, a constant, the centred colour twice (the
    second copy stands where the previous frame's colour goes in temporal mode; with no history it is the
    current colour) and five conditioning values (style, tone, structure, skin, other), one zero pad. The
    lift maps them to 32 channels in f16. The down-sample is a plain 2x2 mean taken in f16 before
    rounding to e4m3.

    ``forward(image)`` returns ``(pooled, skip)``: ``pooled`` is ``[1, 32, h/2, w/2]`` and feeds block 1,
    ``skip`` is ``[1, 32, h, w]``, the block's e4m3 output before pooling, which the post-block consumes.
    """

    def __init__(self, spec: LayerSpec, weights: Weights):
        super().__init__()
        self.spec = spec
        self.controls = Controls()
        self.params = SwinParams.from_weights(weights, spec.block, spec.layer, CHANNELS, 1)
        self.register_buffer("lift", _lift_matrix(weights, spec.block, spec.layer))
        self._noise: tuple[tuple[int, int, int, torch.device], torch.Tensor] | None = None

    def noise(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """The f16 noise features ``[1, 3, h, w]``, generated on the host once per (size, seed) and kept on ``device``.

        The field comes from numpy rather than torch so every device sees the same bits: the exact
        logarithm, square root and trigonometric functions differ from one backend to the next.
        """
        key = (h, w, self.controls.seed, device)
        if self._noise is None or self._noise[0] != key:
            field = _f16(NOISE_GAIN * gaussian_noise(w, h, self.controls.seed))[None]
            self._noise = (key, field.to(device))
        return self._noise[1]

    def features(self, image: torch.Tensor) -> torch.Tensor:
        """The 16 input features, ``[1, 16, h, w]``, on the f16 grid."""
        _, _, h, w = image.shape
        colour = _centred_colour(image)
        constants = torch.tensor([CONSTANT_FEATURE], device=image.device)
        conditioning = torch.tensor(self.controls.conditioning(), device=image.device).to(torch.float16).float()

        def spread(values: torch.Tensor) -> torch.Tensor:
            return values.view(1, -1, 1, 1).expand(1, -1, h, w)

        return torch.cat([self.noise(h, w, image.device), spread(constants), colour, colour,
                          spread(conditioning), spread(torch.zeros(1, device=image.device))], dim=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lifted = narrow(torch.einsum("ck,bkhw->bchw", self.lift, self.features(image)))
        wide, accumulator = swin_core(self.params, lifted, self.spec.shift_x, self.spec.shift_y, accumulator=True)
        # The pooling reads the block's output before its f16 rounding, as the Swin layers' own pooling does.
        return pool2x2(accumulator), round8(wide)


# ---- post-block ----------------------------------------------------------------------------------------

@fuse
def _mix(up: torch.Tensor, skip: torch.Tensor, main_gain: torch.Tensor, skip_gain: torch.Tensor) -> torch.Tensor:
    """``f16(skip * skip_gain + f16(up * main_gain))``, the skip product fused into the add (float64 holds it exactly)."""
    main = _f16(up * main_gain)
    return _f16(skip.to(torch.float64) * skip_gain.to(torch.float64) + main.to(torch.float64)).to(torch.float32)


class PostBlock(nn.Module):
    """Block 70: decoder output + encoder skip -> Swin body -> RGB frame.

    The input at half resolution is replicated 2x and blended with the full-resolution skip using one gain
    per channel for each (f16 arithmetic: ``main * main_gain`` rounded, then the skip product fused into the
    add). The body runs on that blend. Its 32 f16 output channels are projected to three colour residuals
    with the learned ``out_project`` (accumulated in f32); the residual is added to the input frame with
    weight 1/4, clamped to [0, 1], and blended toward the input frame by the Intensity control.

    ``forward(x, skip, frame)`` returns ``[1, 3, h, w]`` float32 in [0, 1], the library's own float output.
    """

    def __init__(self, spec: LayerSpec, weights: Weights):
        super().__init__()
        self.spec = spec
        self.controls = Controls()
        self.params = SwinParams.from_weights(weights, spec.block, spec.layer, CHANNELS, 1)
        b, l = spec.block, spec.layer
        self.register_buffer("main_gain", weights.f16(b, l, "main_gain", CHANNELS).view(1, CHANNELS, 1, 1))
        self.register_buffer("skip_gain", weights.f16(b, l, "skip_gain", CHANNELS).view(1, CHANNELS, 1, 1))
        self.register_buffer("project", weights.f16(b, l, "out_project", 16, CHANNELS)[:3].contiguous())

    def blend(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Replicate ``x`` 2x (repeating the last row/column if the skip is larger) and mix in the skip."""
        _, channels, rows, columns = x.shape
        up = x[:, :, :, None, :, None].expand(1, channels, rows, 2, columns, 2).reshape(1, channels, 2 * rows, 2 * columns)
        h, w = skip.shape[2:]
        up = nn.functional.pad(up, (0, max(0, w - up.shape[3]), 0, max(0, h - up.shape[2])), mode="replicate")
        up = up[:, :, :h, :w]
        return _mix(up, skip, self.main_gain, self.skip_gain)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        wide = swin_core(self.params, self.blend(x, skip), self.spec.shift_x, self.spec.shift_y)
        residual = torch.einsum("rc,bchw->brhw", self.project, wide)
        return compose(frame, residual, self.controls.intensity)


def compose(frame: torch.Tensor, residual: torch.Tensor, intensity: float) -> torch.Tensor:
    """The output colour from the input frame and the projected residual.

    ``nr = clamp(frame + residual / 4)``; for intensity up to 1 the output is ``frame + intensity * (nr - frame)``.
    Past 1 a plain extrapolation would push the colour channels apart faster than the brightness moves, so
    the luminance ratio ``nr / frame`` is raised to the intensity instead and the colour is rescaled to
    that brightness, which keeps the hue; the amplification is bounded to a factor of 2 either way.
    """
    network = (frame + residual * OUTPUT_SCALE).clamp(0.0, 1.0)
    out = (frame + min(intensity, 1.0) * (network - frame)).clamp(0.0, 1.0)
    if intensity > 1.0:
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=frame.device).view(1, 3, 1, 1)
        floor, guard = 1.0 / 512.0, 2.0
        ratio = ((out * weights).sum(1, keepdim=True) + floor) / ((frame * weights).sum(1, keepdim=True) + floor)
        amplification = ratio.clamp_min(1e-6).pow(intensity).clamp(1.0 / guard, guard)
        out = (out * amplification / ratio.clamp_min(1e-6)).clamp(0.0, 1.0)
    return out


BUILDERS: dict[str, Callable[[LayerSpec, Weights], nn.Module]] = {
    "PreBlock": PreBlock,
    "PostBlock": PostBlock,
}
