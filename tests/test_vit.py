"""Vision-transformer layers and the decoder input stage: shapes, e4m3 grid, and attention properties."""
import torch

from dlssnr.layers import decoder, vit
from dlssnr.plan import build_plan
from dlssnr.quant import q8
from dlssnr.weights import Weights

PLAN = {(s.block, s.layer): s for s in build_plan(1920, 1080)}
GRID = (4, 6)


def _e4m3_bytes(generator, count, scale):
    values = q8(torch.randn(count, generator=generator) * scale)
    return values.to(torch.float8_e4m3fn).view(torch.uint8)


def _half_bytes(values):
    return torch.as_tensor(values, dtype=torch.float16).view(torch.uint8)


def _weights():
    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for layer, (rows, columns) in {0: (4096, 1024), 1: (1024, 4096), 4: (1024, 1024)}.items():
        tensors[f"block31.layer{layer}.weight"] = _e4m3_bytes(generator, rows * columns, 0.03)
        tensors[f"block31.layer{layer}.skip_weight"] = _half_bytes(torch.rand(1024, generator=generator))
    for name in "qkv":
        tensors[f"block31.layer2.{name}"] = _e4m3_bytes(generator, 1024 * 1024, 0.05)
    tensors["block31.layer2.header"] = torch.full((32,), 1.5).view(torch.uint8)
    tensors["block39.layer0.weight"] = _e4m3_bytes(generator, 512 * 1024, 0.05)
    tensors["block39.layer0.skip_weight"] = _half_bytes(torch.rand(512, generator=generator))
    return Weights(tensors)


def _feature_map(channels, generator, shape=GRID):
    return q8(torch.randn(1, channels, *shape, generator=generator) * 2)


def _on_grid(tensor):
    return torch.equal(q8(tensor), tensor)


def test_block_shapes_and_grid():
    weights = _weights()
    generator = torch.Generator().manual_seed(1)
    stream = _feature_map(1024, generator)
    hidden = vit.BUILDERS["VitFfnExpand"](PLAN[31, 0], weights)(stream)
    contracted = vit.BUILDERS["VitFfnContract"](PLAN[31, 1], weights)(hidden, stream)
    qkv = vit.BUILDERS["VitQKV"](PLAN[31, 2], weights)(contracted)
    attended = vit.BUILDERS["VitAttention"](PLAN[31, 3], weights)(qkv)
    out = vit.BUILDERS["VitProjection"](PLAN[31, 4], weights)(attended, contracted)
    for tensor, channels in ((hidden, 4096), (contracted, 1024), (qkv, 3072), (attended, 1024), (out, 1024)):
        assert tensor.shape == (1, channels, *GRID)
        assert tensor.dtype == torch.float32 and _on_grid(tensor)


def test_activation_values():
    x = torch.tensor([0.0, 1.0, 4.0, 8.0])
    y = vit.VitFfnExpand.activation(x)
    assert y[0] == 0.0
    assert abs(y[1].item() - 1.286) < 1e-3
    assert abs(y[3].item() - 8 * y[2].item() / 4) < 1e-4       # beyond |x| = 4 the cubic term is frozen


def test_qkv_heads_are_unit_scaled():
    weights = _weights()
    qkv = vit.BUILDERS["VitQKV"](PLAN[31, 2], weights)(_feature_map(1024, torch.Generator().manual_seed(2)))
    key = qkv[:, 1024:2048].reshape(1, 32, 32, *GRID)
    norms = key.pow(2).sum(dim=2).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=0.1)
    query = qkv[:, :1024].reshape(1, 32, 32, *GRID)
    expected = 1.5 * 32 ** 0.5
    assert torch.allclose(query.pow(2).sum(dim=2).sqrt(), torch.full_like(norms, expected), rtol=0.1)


def _exact_attention(qkv, heads=32):
    tokens = qkv.shape[2] * qkv.shape[3]
    q, k, v = (vit.to_tokens(qkv)[:, i * 1024:(i + 1) * 1024].reshape(tokens, heads, 32) for i in range(3))
    weights = torch.softmax(torch.einsum("qhd,khd->hqk", q, k).double(), dim=-1)
    return torch.einsum("hqk,khd->qhd", weights, v.double()).reshape(tokens, 1024)


def test_attention_approximates_softmax_for_any_token_count():
    generator = torch.Generator().manual_seed(3)
    layer = vit.VitAttention(PLAN[31, 3])
    for shape in ((4, 16), (5, 7)):                    # 64 tokens, and 35 tokens that need key padding
        qkv = torch.randn(1, 3072, *shape, generator=generator)
        qkv[:, :2048] = qkv[:, :2048].reshape(1, 64, 32, *shape).div(
            qkv[:, :2048].reshape(1, 64, 32, *shape).norm(dim=2, keepdim=True)).reshape(1, 2048, *shape) * 2.0
        qkv = q8(qkv)
        out = vit.to_tokens(layer(qkv))
        reference = _exact_attention(qkv)
        assert (out - reference).abs().mean() < 0.05 * reference.abs().mean() + 0.02


def test_attention_ignores_token_order():
    generator = torch.Generator().manual_seed(4)
    layer = vit.VitAttention(PLAN[31, 3])
    qkv = q8(torch.randn(1, 3072, 4, 16, generator=generator))
    order = torch.randperm(64, generator=generator)
    shuffled = vit.to_map(vit.to_tokens(qkv)[order], 4, 16)
    # the float16 denominator is summed in key order, so equality holds up to the odd last-bit flip
    same = vit.to_tokens(layer(shuffled)) == vit.to_tokens(layer(qkv))[order]
    assert same.float().mean() > 0.99


def test_approximate_exp_tracks_exponential():
    x = torch.linspace(-3, 3, 241)
    ratio = vit._approximate_exp(x).double() / torch.exp(x.double())
    assert ratio.max() / ratio.min() < 1.15


def test_decoder_input_upsample_repeats_the_projection():
    weights = _weights()
    layer = decoder.DecInputUpsample(PLAN[39, 0], weights)
    generator = torch.Generator().manual_seed(5)
    low = _feature_map(1024, generator, (3, 4))
    skip = _feature_map(512, generator, (5, 7))
    out = layer(low, skip)
    assert out.shape == (1, 512, 5, 7) and _on_grid(out)
    layer.gain.zero_()
    plain = layer(low, skip)
    assert torch.equal(plain[..., 0:4:2, 0:6:2], plain[..., 1:4:2, 1:6:2])
