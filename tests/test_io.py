"""Image conversion and controls: shapes, padding, rounding, validation."""
import numpy as np
import pytest
import torch

from dlssnr.io import Controls, frame_to_tensor, load_image, save_image, tensor_to_frame
from dlssnr.plan import padding


def _frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_defaults_are_the_reference_controls():
    c = Controls()
    assert (c.intensity, c.style, c.local_tone, c.local_structure, c.skin_structure, c.auto_mask) == \
        (1.0, 0, 1.0, 1.0, -1.0, True)
    assert c.depth is None and c.motion is None and c.history is None and c.reset is False


def test_conditioning_with_and_without_the_mask():
    assert Controls().conditioning() == (0.0, 1.0, 1.0, 1.0, 1.0)
    assert Controls(style=2, local_structure=0.5, skin_structure=1.5).conditioning() == (2 / 128, 1.0, 1.0, 1.5, 0.5)
    assert Controls(auto_mask=False, local_structure=0.5).conditioning() == (0.0, 1.0, 0.5, -1.0, -1.0)


@pytest.mark.parametrize("bad", [dict(intensity=2.5), dict(style=3), dict(local_tone=-0.1), dict(skin_structure=-2)])
def test_rejects_out_of_range_controls(bad):
    with pytest.raises(ValueError):
        Controls(**bad)


def test_tensor_is_padded_to_the_working_extent():
    frame = _frame(70, 40)
    tensor = frame_to_tensor(frame)
    work = padding(70, 40)
    assert tensor.shape == (1, 3, work.height, work.width) and tensor.dtype == torch.float32
    assert tensor[0, 1, 7, 5].item() == pytest.approx(frame[7, 5, 1] / 255)


def test_padding_mirrors_about_the_last_pixel():
    frame = _frame(70, 40)
    tensor = frame_to_tensor(frame)
    assert torch.equal(tensor[0, :, :40, 70], tensor[0, :, :40, 68])
    assert torch.equal(tensor[0, :, 41, :70], tensor[0, :, 37, :70])


def test_frame_round_trip_is_exact():
    frame = _frame(333, 111, seed=3)
    assert np.array_equal(tensor_to_frame(frame_to_tensor(frame), 333, 111), frame)


def test_rounding_is_half_up_and_clamped():
    tensor = torch.zeros(1, 3, 4, 4)
    tensor[0, 0, 0, 0] = 0.5 / 255
    tensor[0, 0, 0, 1] = 1.5 / 255 - 1e-4
    tensor[0, 1, 0, 0] = -0.3
    tensor[0, 2, 0, 0] = 7.0
    frame = tensor_to_frame(tensor, 4, 4)
    assert frame[0, 0].tolist() == [1, 0, 255]
    assert frame[0, 1, 0] == 1


def test_png_round_trip(tmp_path):
    frame = _frame(33, 21)
    save_image(str(tmp_path / "a.png"), frame)
    assert np.array_equal(load_image(str(tmp_path / "a.png")), frame)
