"""Picture comparison: how close one 8-bit frame is to another.

Frames are uint8 arrays ``[H, W, 3]``. SSIM is the mean over R, G and B of the 11x11 Gaussian-window
(sigma 1.5) structural similarity with the usual constants for an 8-bit range.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB for an 8-bit range (infinity for identical frames)."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity over the three colour channels."""
    k = _gaussian_kernel().to(torch.float64)
    pad = k.numel() // 2
    kh, kw = k.view(1, 1, -1, 1), k.view(1, 1, 1, -1)

    def blur(z: torch.Tensor) -> torch.Tensor:                      # z: [N, 1, H, W], reflect padding
        z = F.pad(z, (0, 0, pad, pad), mode="reflect")
        z = F.conv2d(z, kh)
        z = F.pad(z, (pad, pad, 0, 0), mode="reflect")
        return F.conv2d(z, kw)

    x = torch.from_numpy(a).to(torch.float64).permute(2, 0, 1).unsqueeze(1)
    y = torch.from_numpy(b).to(torch.float64).permute(2, 0, 1).unsqueeze(1)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    score = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(score.mean())


def compare(output: np.ndarray, reference: np.ndarray, source: np.ndarray | None = None) -> dict:
    """The numbers used to judge a frame against a reference.

    With ``source`` (the frame that was enhanced) it also reports how well the *change* the network made
    matches the reference's change: the correlation of the two edits and their RMS sizes.
    """
    diff = output.astype(np.float64) - reference.astype(np.float64)
    result = {
        "psnr_db": psnr(output, reference),
        "ssim": ssim(output, reference),
        "rms_error": float(np.sqrt(np.mean(diff ** 2))),
        "mean_difference": [float(v) for v in diff.reshape(-1, 3).mean(0)],
        "pixels_within_one_step_pct": 100.0 * float(np.mean(np.abs(diff).max(axis=2) <= 1)),
    }
    if source is not None:
        edit_out = output.astype(np.float64) - source
        edit_ref = reference.astype(np.float64) - source
        result["edit_correlation"] = float(np.corrcoef(edit_out.ravel(), edit_ref.ravel())[0, 1])
        result["edit_rms_output"] = float(np.sqrt(np.mean(edit_out ** 2)))
        result["edit_rms_reference"] = float(np.sqrt(np.mean(edit_ref ** 2)))
        result["unprocessed_psnr_db"] = psnr(source, reference)
    return result
