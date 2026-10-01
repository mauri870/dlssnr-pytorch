import math

import numpy as np

from dlssnr.metrics import compare, psnr, ssim


def _frame(seed: int = 0, size=(48, 64)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(*size, 3), dtype=np.uint8)


def test_identical_frames():
    a = _frame()
    assert math.isinf(psnr(a, a))
    assert abs(ssim(a, a) - 1.0) < 1e-9


def test_psnr_matches_the_definition():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.full((8, 8, 3), 5, dtype=np.uint8)
    assert abs(psnr(a, b) - 10 * math.log10(255 ** 2 / 25)) < 1e-9


def test_noise_lowers_ssim():
    ramp = np.linspace(0, 255, 64, dtype=np.float64)
    a = np.repeat(np.tile(ramp, (48, 1))[..., None], 3, axis=2).astype(np.uint8)
    noisy = np.clip(a.astype(int) + np.random.default_rng(2).integers(-20, 21, a.shape), 0, 255).astype(np.uint8)
    assert ssim(a, noisy) < 0.9


def test_compare_edit_statistics():
    source = _frame(3)
    reference = np.clip(source.astype(int) + 10, 0, 255).astype(np.uint8)
    stats = compare(reference, reference, source)
    assert stats["pixels_within_one_step_pct"] == 100.0
    assert abs(stats["edit_correlation"] - 1.0) < 1e-9
