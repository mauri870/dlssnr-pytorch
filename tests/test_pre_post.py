"""The outer edge of the network: noise, down-sampling, composition and the lift's layout."""
import numpy as np
import pytest
import torch

from dlssnr.layers import pre_post
from dlssnr.layers.pre_post import _lift_matrix, compose, gaussian_noise
from dlssnr.layers.swin import pool2x2
from dlssnr.quant import q8
from dlssnr.weights import Weights


def test_noise_is_deterministic_and_seeded():
    a = gaussian_noise(64, 48, seed=0)
    assert a.shape == (3, 48, 64)
    assert torch.equal(a, gaussian_noise(64, 48, seed=0))
    assert not torch.equal(a, gaussian_noise(64, 48, seed=1))


def test_noise_is_a_function_of_the_pixel_only():
    assert torch.equal(gaussian_noise(64, 48, 3)[:, :20, :30], gaussian_noise(30, 20, 3))


def test_noise_is_standard_normal():
    noise = gaussian_noise(256, 256, seed=0)
    assert noise.mean().abs() < 0.02 and noise.std() == pytest.approx(1.0, abs=0.02)
    assert abs(float(np.corrcoef(noise[0].flatten(), noise[1].flatten())[0, 1])) < 0.02


def test_pool_is_a_mean_on_the_e4m3_grid():
    wide = torch.randn(1, 4, 8, 8).to(torch.float16).to(torch.float32)
    pooled = pool2x2(wide)
    assert pooled.shape == (1, 4, 4, 4)
    assert torch.equal(pooled, q8(pooled))
    exact = wide.reshape(1, 4, 4, 2, 4, 2).mean((3, 5))
    assert (pooled - exact).abs().max() < 0.07


def test_compose_intensity_zero_returns_the_frame_and_one_the_network():
    frame = torch.rand(1, 3, 6, 6)
    residual = torch.randn(1, 3, 6, 6)
    assert torch.equal(compose(frame, residual, 0.0), frame)
    full = compose(frame, residual, 1.0)
    assert torch.allclose(full, (frame + residual * 0.25).clamp(0, 1))
    half = compose(frame, residual, 0.5)
    assert torch.allclose(half, frame + 0.5 * (full - frame), atol=1e-6)


def test_compose_stays_in_range_above_one():
    frame = torch.rand(1, 3, 6, 6)
    out = compose(frame, torch.randn(1, 3, 6, 6) * 4, 2.0)
    assert out.min() >= 0 and out.max() <= 1


def test_lift_layout_is_a_permutation_of_the_stored_record():
    stored = torch.arange(512, dtype=torch.float16)
    weights = Weights({"block0.layer0.input_lift": stored.view(torch.uint8)})
    lift = _lift_matrix(weights, 0, 0)
    assert lift.shape == (32, 16)
    assert sorted(lift.flatten().tolist()) == list(range(512))


def test_builders_cover_both_kinds():
    assert set(pre_post.BUILDERS) == {"PreBlock", "PostBlock"}
