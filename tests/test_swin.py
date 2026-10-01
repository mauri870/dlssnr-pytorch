import pytest
import torch

from dlssnr.layers import swin
from dlssnr.plan import build_plan
from dlssnr.quant import q8
from dlssnr.weights import Weights

SPECS = {(s.block, s.layer): s for s in build_plan(1920, 1080)}


def e4m3_bytes(generator, *shape, scale=0.5):
    return q8(torch.randn(*shape, generator=generator) * scale).to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)


def f16_bytes(values):
    return torch.as_tensor(values, dtype=torch.float16).view(torch.uint8).reshape(-1)


def make_weights(block, channels, heads, variant="", generator=None):
    """Random weights with the stems and sizes a fused Swin layer stores."""
    g = generator or torch.Generator().manual_seed(3)
    hidden = 4 * channels
    t = {
        "mlp_expand": e4m3_bytes(g, hidden, channels),
        "mlp_contract": e4m3_bytes(g, channels, channels if heads > 1 else hidden),
        "qkv": e4m3_bytes(g, 3 * channels, channels),
        "attn_out_proj": e4m3_bytes(g, channels, channels),
        "attn_pos_bias": f16_bytes(torch.randn(heads * 4096, generator=g) - 2),
        "scalars_b": torch.full((max(4, heads),), 2.0).view(torch.uint8),
        "residual_scale": f16_bytes(torch.cat([torch.zeros(8), torch.full((channels,), 0.9), torch.zeros(8)])),
        "attn_residual_scale": f16_bytes(torch.cat([torch.full((channels,), 0.8), torch.zeros(8)])),
    }
    if heads > 1:
        t["mlp_mid"] = e4m3_bytes(g, channels, hidden // heads)
    if variant == "ds":
        t["resample"] = e4m3_bytes(g, 2 * channels, channels)
    if variant == "upsample":
        t["resample"] = e4m3_bytes(g, channels, 2 * channels)
        t["upsample_gain"] = f16_bytes(torch.full((channels,), 0.5))
    return Weights({Weights.key(block, 0, name): value for name, value in t.items()})


def random_map(channels, rows, cols, seed=1):
    return q8(torch.randn(1, channels, rows, cols, generator=torch.Generator().manual_seed(seed)))


def on_grid(x):
    return bool(torch.equal(q8(x), x))


@pytest.mark.parametrize("shift", [(0, 0), (-1, 0), (0, -1), (-1, -1)])
def test_window_partition_round_trip(shift):
    x = random_map(8, 20, 28)
    tokens, grid = swin._to_windows(x, *shift)
    assert tokens.shape[1:] == (64, 8)
    assert torch.equal(swin._from_windows(tokens, grid, 20, 28), x)


def test_shifted_grid_has_an_extra_window():
    rows, front = swin._window_grid(64, -1)
    assert (rows, front) == (9, 4)
    assert swin._window_grid(64, 0) == (8, 0)
    with pytest.raises(ValueError):
        swin._window_grid(64, 1)


def test_window_tokens_are_tile_ordered():
    x = torch.arange(8 * 8, dtype=torch.float32).reshape(1, 1, 8, 8)
    tokens, _ = swin._to_windows(x, 0, 0)
    # token 0 is pixel (0, 0); token 1 is the next pixel in the row; token 16 starts the second tile (x = 4)
    assert tokens[0, :4, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert tokens[0, 4, 0].item() == 8.0
    assert tokens[0, 16, 0].item() == 4.0
    assert tokens[0, 32, 0].item() == 32.0


def test_bias_deswizzle_is_a_permutation():
    out = swin._deswizzle_bias(torch.arange(4096, dtype=torch.float32))
    assert sorted(out.reshape(-1).tolist()) == list(range(4096))


def test_activation_values():
    x = torch.tensor([0.0, 1.0, -1.0, 8.0, -8.0])
    expected = torch.tensor([0.0, 1.285888671875, -0.503173828125, 14.3125, 0.0])
    assert torch.allclose(swin.activation(x), expected, atol=1e-5)


def test_numerators_are_positive_f16_ramp_values():
    logits = torch.linspace(-30, 30, 1000).reshape(1, 1, 10, 100)
    bias = torch.zeros(1, 10, 100)
    values = swin._softmax_numerators(logits, bias)
    assert values.dtype == torch.float16
    assert bool((values > 0).all())
    flat = values.reshape(-1).float()
    assert bool((flat[1:] >= flat[:-1]).all())          # monotone in the logit


def test_denominator_matches_a_plain_sum_closely():
    numerators = torch.rand(3, 2, 64, 64, generator=torch.Generator().manual_seed(2)).half() * 8
    reference = numerators.double().sum(-1)
    assert torch.allclose(swin._denominator(numerators).double(), reference, rtol=2e-2)


def test_norm_scale_of_unit_vectors_is_one():
    v = torch.zeros(5, 32)
    v[:, 3] = 1.0
    assert torch.allclose(swin._norm_scale(v), torch.ones(5), rtol=1e-3)
    assert swin._norm_scale(torch.zeros(1, 32)).item() == pytest.approx(1 / 0.0079, rel=0.01)   # clamped floor


def test_pool_of_constant_and_grid():
    x = torch.full((1, 3, 8, 8), 1.1)
    pooled = swin.pool2x2(x)
    assert pooled.shape == (1, 3, 4, 4)
    assert on_grid(pooled)
    assert torch.equal(pooled, q8(torch.full((1, 3, 4, 4), 1.1)))


@pytest.mark.parametrize("channels,heads,shift", [(32, 1, (0, 0)), (32, 1, (-1, -1)), (64, 2, (-1, 0))])
def test_block_output_is_on_grid_and_windows_are_independent(channels, heads, shift):
    params = swin.SwinParams.from_weights(make_weights(5, channels, heads), 5, 0, channels, heads)
    x = random_map(channels, 24, 32)
    wide = swin.swin_block(params, x, *shift)
    assert wide.shape == x.shape
    assert on_grid(q8(wide))
    assert bool((wide == wide.to(torch.float16).float()).all())       # f16-exact
    y = x.clone()
    y[:, :, 20, 28] = -y[:, :, 20, 28] + 1                              # lies in a window away from the top-left one
    other = swin.swin_block(params, y, *shift)
    assert torch.equal(wide[:, :, :8, :8], other[:, :, :8, :8])
    assert not torch.equal(wide, other)


def test_zero_input_stays_zero():
    params = swin.SwinParams.from_weights(make_weights(2, 32, 1), 2, 0, 32, 1)
    assert not swin.swin_block(params, torch.zeros(1, 32, 16, 16), -1, -1).any()


def spec_of(block, **overrides):
    s = SPECS[block, 0]
    return type(s)(**{**s.__dict__, **overrides})


def test_plain_layer_module():
    s = spec_of(2, w=16, h=24, out_w=16, out_h=24)
    layer = swin.BUILDERS["Swin1H"](s, make_weights(2, 32, 1))
    y = layer(random_map(32, 16, 24))
    assert y.shape == (1, 32, 16, 24) and on_grid(y)


def test_downsample_returns_pooled_projection_and_body():
    s = spec_of(4, w=16, h=24, out_w=12, out_h=12)
    layer = swin.BUILDERS["Swin1H"](s, make_weights(4, 32, 1, "ds"))
    pooled, body = layer(random_map(32, 16, 24))
    assert pooled.shape == (1, 64, 12, 12) and body.shape == (1, 32, 16, 24)
    assert on_grid(pooled) and on_grid(body)
    assert not pooled[:, :, 8:].any()                                  # rows past the pooled extent are padding
    assert pooled[:, :, :8].any()


def test_upsample_uses_skip_tuple_or_map():
    s = spec_of(66, w=8, h=12, out_w=16, out_h=24)
    layer = swin.BUILDERS["Swin1H"](s, make_weights(66, 32, 1, "upsample"))
    x, skip = random_map(64, 8, 12), random_map(32, 16, 24, seed=2)
    a = layer(x, skip)
    b = layer(x, (random_map(64, 8, 12), skip))
    assert a.shape == (1, 32, 16, 24) and on_grid(a)
    assert torch.equal(a, b)


def test_wide_upsample_runs_on_the_grid():
    s = spec_of(62, w=8, h=12, out_w=16, out_h=24, channels=64, heads=2, c_in=128, c_out=64)
    layer = swin.BUILDERS["Swin2H"](s, make_weights(62, 64, 2, "upsample"))
    y = layer(random_map(128, 8, 12), random_map(64, 16, 24, seed=2))
    assert y.shape == (1, 64, 16, 24) and on_grid(y)


def test_builders_cover_the_fused_swin_kinds():
    kinds = {s.kind for s in SPECS.values() if s.kind.startswith("Swin")}
    assert kinds == set(swin.BUILDERS)
