import numpy as np
import pytest
import torch
from torch.nn import functional as F

from dlssnr.layers import split_swin as split
from dlssnr.plan import build_plan
from dlssnr.quant import q8
from dlssnr.weights import Weights

SPECS = {(s.block, s.layer): s for s in build_plan(1920, 1080)}


def random_e4m3(generator, *shape):
    """Small random e4m3 codes (no NaN, magnitudes up to 2)."""
    sign = torch.randint(0, 2, shape, generator=generator, dtype=torch.uint8) << 7
    exponent = torch.randint(0, 9, shape, generator=generator, dtype=torch.uint8) << 3
    mantissa = torch.randint(0, 8, shape, generator=generator, dtype=torch.uint8)
    return sign | exponent | mantissa


def f16_bytes(values):
    return torch.as_tensor(values, dtype=torch.float16).view(torch.uint8).reshape(-1)


def make_weights(block, layer, generator):
    name = SPECS[block, layer].kind
    tensors = {}
    if name == "SplitFfwd":
        tensors["ffwd_a"] = random_e4m3(generator, 512 * 512)
        tensors["ffwd_b"] = random_e4m3(generator, 512 * 512)
    elif name == "SplitQKVAttn":
        tensors["qkv"] = random_e4m3(generator, 1536 * 512)
        tensors["attn_pos_bias"] = f16_bytes(torch.randn(16 * 4096, generator=generator) * 2 - 3)
        tensors["tail"] = torch.full((16,), 8.0).view(torch.uint8).reshape(-1)
    elif name == "SplitFinalHead":
        tensors["weight"] = random_e4m3(generator, 1024 * 512)
    else:
        tensors["weight"] = random_e4m3(generator, 512 * 512)
        tensors["skip_weight"] = f16_bytes(torch.full((512,), 0.5))
    return Weights({Weights.key(block, layer, key): value for key, value in tensors.items()})


def random_map(generator, rows, cols, channels=512):
    return q8(torch.randn(1, channels, rows, cols, generator=generator))


def on_grid(x):
    return bool(torch.equal(q8(x), x))


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(7)


def test_builders_cover_every_split_kind():
    kinds = {s.kind for s in SPECS.values() if s.block in range(23, 31) or s.block in range(40, 48)}
    assert kinds == set(split.BUILDERS)


def test_ffwd_unpacking_uses_every_byte_once(generator):
    first = random_e4m3(generator, 512, 512).numpy()
    second = random_e4m3(generator, 512, 512).numpy()
    matrices = split._grouped_ffwd_matrices(first, second)
    assert [m.shape for m in matrices] == [(8, 64, 512), (8, 256, 64), (8, 64, 256)]
    unpacked = np.concatenate([m.ravel() for m in matrices])
    assert np.array_equal(np.sort(unpacked), np.sort(np.concatenate([first.ravel(), second.ravel()])))


def test_ffwd_output_is_on_grid_and_passes_input_through(generator):
    layer = split.SplitFfwd(SPECS[24, 0], make_weights(24, 0, generator))
    x = random_map(generator, 8, 12)
    out, passed = layer(x)
    assert out.shape == x.shape and out.dtype == torch.float32 and on_grid(out)
    assert passed is x


def test_ffwd_groups_are_independent_in_the_output_channels(generator):
    layer = split.SplitFfwd(SPECS[24, 0], make_weights(24, 0, generator))
    x = random_map(generator, 4, 4)
    base, _ = layer(x)
    layer.expand_hidden[3].zero_()
    changed, _ = layer(x)
    different = (base != changed).any(dim=(0, 2, 3))
    assert different[3 * 64:4 * 64].any() and not different[:3 * 64].any() and not different[4 * 64:].any()


def test_activation_values():
    act = split.SplitFfwd.activation(torch.tensor([0.0, 1.0, -1.0, 8.0, -8.0]))
    assert act[0] == 0.0
    assert act[1] == pytest.approx(1.2861, abs=1e-3)
    assert act[2] == pytest.approx(-0.5032, abs=1e-3)
    assert act[3] == pytest.approx(8 * (4 * (0.447265625 - 0.055908203125 * 4) + 0.89453125), rel=2e-3)
    assert act[4] == 0.0


def test_projection_adds_gained_residual(generator):
    layer = split.SplitFfwdProj(SPECS[24, 1], make_weights(24, 1, generator))
    x = random_map(generator, 4, 8)
    residual = random_map(generator, 4, 8)
    with_residual = layer(x, residual)
    assert with_residual.shape == x.shape and on_grid(with_residual)
    assert torch.equal(layer((x, residual)), with_residual)
    assert not torch.equal(layer(x, torch.zeros_like(residual)), with_residual)


def test_proj_pool_extents_and_mean(generator):
    layer = split.SplitProjPool(SPECS[30, 3], make_weights(30, 3, generator))
    x = random_map(generator, 12, 20)
    projection, pooled = layer(x, random_map(generator, 12, 20))
    assert projection.shape == (1, 512, 12, 20)
    assert pooled.shape == (1, 512, 8, 12) and on_grid(pooled)
    assert not pooled[..., 6:, :].any() and not pooled[..., 10:].any()
    assert torch.equal(layer(torch.zeros_like(x), torch.zeros_like(x))[1], torch.zeros(1, 512, 8, 12))


def test_final_head_reads_the_pooled_copy(generator):
    layer = split.SplitFinalHead(SPECS[30, 4], make_weights(30, 4, generator))
    pooled = random_map(generator, 8, 12)
    out = layer(None, (random_map(generator, 16, 24), pooled))
    assert out.shape == (1, 1024, 8, 12) and on_grid(out)
    assert torch.equal(out, layer(pooled))


def test_exponential_range():
    x = torch.tensor([-1e4, -6.0, 0.0, 6.0, 1e4])
    e = split.SplitQKVAttn.exponential(x)
    assert e[0] == e[1] == pytest.approx(2.0 ** -14)
    assert torch.all(e[1:] >= e[:-1]) and e[-1] == e[-2] and e[-1] < 10


def test_attention_is_local_to_its_window(generator):
    layer = split.SplitQKVAttn(SPECS[23, 2], make_weights(23, 2, generator))
    x = random_map(generator, 16, 16)
    out = layer(x)
    assert out.shape == x.shape and on_grid(out)
    other = x.clone()
    other[..., 8:, 8:] = random_map(generator, 8, 8)
    changed = layer(other)
    assert torch.equal(changed[..., :8, :], out[..., :8, :]) and torch.equal(changed[..., :, :8], out[..., :, :8])
    assert not torch.equal(changed[..., 8:, 8:], out[..., 8:, 8:])


def test_shifted_windows_are_the_unshifted_grid_on_a_padded_map(generator):
    weights = make_weights(24, 2, generator)
    shifted = split.SplitQKVAttn(SPECS[24, 2], weights)
    assert (shifted.shift_x, shifted.shift_y) == (-1, -1)
    plain = split.SplitQKVAttn(SPECS[23, 2], Weights({k.replace("block24", "block23"): v
                                                      for k, v in weights.tensors.items()}))
    x = random_map(generator, 12, 20)
    expected = plain(F.pad(x, (4, 8, 4, 8)))[..., 4:16, 4:24]
    assert torch.equal(shifted(x), expected)
