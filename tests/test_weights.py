"""Weights container: storage round trip and typed accessors."""
import numpy as np
import torch

from dlssnr.quant import e4m3_decode
from dlssnr.weights import Weights


def _sample() -> Weights:
    matrix = torch.tensor([0x00, 0x38, 0xB8, 0x7E, 0x01, 0x30], dtype=torch.uint8)
    halves = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float16).view(torch.uint8)
    floats = torch.tensor([0.5, -3.0], dtype=torch.float32).view(torch.uint8)
    return Weights({
        "block1.layer0.matrix": matrix,
        "block1.layer0.halves": halves,
        "block1.layer0.floats": floats,
        "block2.layer3.matrix": matrix.clone(),
    }, {"source": "test"})


def test_round_trip_keeps_bytes_and_metadata(tmp_path):
    original = _sample()
    path = str(tmp_path / "w.safetensors")
    original.save(path)
    loaded = Weights.load(path)
    assert loaded.metadata == {"source": "test"}
    assert loaded.tensors.keys() == original.tensors.keys()
    for key, tensor in original.tensors.items():
        assert loaded.tensors[key].dtype == torch.uint8
        assert torch.equal(loaded.tensors[key], tensor)


def test_key_and_lookup():
    weights = _sample()
    assert Weights.key(1, 0, "matrix") == "block1.layer0.matrix"
    assert weights.has(1, 0, "matrix")
    assert not weights.has(1, 1, "matrix")
    assert weights.names(1, 0) == ["floats", "halves", "matrix"]
    assert weights.names(2, 3) == ["matrix"]
    assert weights.names(9, 9) == []


def test_typed_accessors():
    weights = _sample()
    assert weights.raw(1, 0, "matrix").shape == (6,)
    assert torch.equal(weights.e4m3(1, 0, "matrix", 2, 3), e4m3_decode(weights.raw(1, 0, "matrix")).reshape(2, 3))
    assert weights.e4m3(1, 0, "matrix", 2, 3)[0, 1] == 1.0
    assert weights.f16(1, 0, "halves", 3).tolist() == [1.5, -2.0, 0.25]
    assert weights.f32(1, 0, "floats", 2, 1).tolist() == [[0.5], [-3.0]]
    assert weights.f16(1, 0, "halves", 3).dtype == torch.float32
