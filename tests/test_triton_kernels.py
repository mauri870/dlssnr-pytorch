"""The Triton kernels against the reference arithmetic. They sum float32 products in the matrix unit's
order, so a few values land one e4m3 step away; everything else must agree exactly."""
import pytest
import torch

from dlssnr.layers import swin, vit
from dlssnr.plan import build_plan
from dlssnr.quant import q8

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from dlssnr import attention, gemm  # noqa: E402

DEVICE = "cuda"


def mismatches(a, b):
    return (a.float() != b.float()).float().mean().item()


def test_matmul_matches_float32():
    a = q8(torch.randn(3000, 96, device=DEVICE))
    weight = q8(torch.randn(160, 96, device=DEVICE))
    assert torch.allclose(gemm.linear(a, weight), a @ weight.T, rtol=1e-5, atol=1e-5)


def test_activation_epilogue_is_exact():
    a = q8(torch.randn(3000, 32, device=DEVICE) * 2)
    weight = q8(torch.randn(128, 32, device=DEVICE))
    product = gemm.linear(a, weight)
    assert torch.equal(gemm.linear(a, weight, activate=True).float(), q8(swin.activation(product)))


@pytest.mark.parametrize("heads", [1, 4])
def test_window_attention(heads):
    count = 64
    qkv = torch.randn(count * 64, heads * 96, device=DEVICE)
    head_scale = 2.0 + torch.rand(heads, device=DEVICE)
    bias = torch.randn(heads, 64, 64, device=DEVICE) * 0.5
    split = qkv.reshape(count, 64, heads, 3, 32)
    query, key, value = swin._normalise_qkv(split[..., 0, :], split[..., 1, :], split[..., 2, :], head_scale)
    probability = swin._softmax(torch.einsum("ntgd,nsgd->ngts", query, key), bias)
    expected = swin.store(torch.einsum("ngts,nsgd->ntgd", probability, value).reshape(count * 64, -1))
    assert mismatches(attention.window_attention(qkv, head_scale, bias), expected) < 1e-3


@pytest.mark.parametrize("shape", [(5, 7), (20, 32)])
def test_vit_attention(shape):
    layer = vit.VitAttention(build_plan(1920, 1080)[0])
    layer.channels = 1024
    generator = torch.Generator().manual_seed(1)
    qkv = torch.randn(1, 3072, *shape, generator=generator)
    heads = qkv[:, :2048].reshape(1, 64, 32, *shape)
    qkv[:, :2048] = (heads / heads.norm(dim=2, keepdim=True) * 2).reshape(1, 2048, *shape)
    qkv = q8(qkv).to(DEVICE)
    fast = layer(qkv)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("dlssnr.runtime._TRITON", False)
        reference = layer(qkv)
    assert mismatches(fast, reference) < 2e-3
