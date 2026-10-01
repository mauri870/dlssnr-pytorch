import torch

from dlssnr.benchmark import format_table, report_layers, synthetic_frame, time_call


def test_synthetic_frame_is_deterministic_and_8_bit():
    a, b = synthetic_frame(96, 64), synthetic_frame(96, 64)
    assert a.shape == (64, 96, 3) and a.dtype.name == "uint8"
    assert (a == b).all()
    assert (synthetic_frame(96, 64, seed=1) != a).any()


def test_time_call_runs_warmup_plus_repeats():
    calls = []
    ms = time_call(lambda: calls.append(1), torch.device("cpu"), repeats=4)
    assert len(calls) == 5 and ms >= 0.0


def test_tables_align_and_share_sums_to_a_hundred():
    table = format_table([("a", "1"), ("longer", "22")], ("name", "n"))
    assert [len(line) for line in table.splitlines()] == [len("longer") + 2 + 2] * 4
    text = report_layers({"X": (2, 30.0), "Y (ds)": (1, 10.0)})
    assert "75.0 %" in text and "25.0 %" in text and "sum of the layers 40.0 ms" in text
