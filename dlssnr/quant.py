"""The 8-bit floating point format the network computes in.

The network is FP8 e4m3 at every operation boundary: a value is rounded to the nearest e4m3 number
(ties to even), saturating at +-448, and the next operation reads it back exactly. Every e4m3 value is
exactly representable in float32, so a layer can hold its tensors as float32 that happen to lie on the
e4m3 grid, multiply them exactly (an e4m3 product has at most 8 significant bits) and accumulate in
float32.
"""
import torch

E4M3_MAX = 448.0

# Experiment hook: the blocks in ``skip_blocks`` keep their values at float16 instead of rounding them to e4m3 (the
# network was calibrated with the rounding, so this changes the picture). ``current_block`` is set by NRNet.run.
# The compiled and Triton paths bake the rounding in: set DLSS5_COMPILE=0 and DLSS5_TRITON=0 when using it.
skip_blocks: set[int] = set()
current_block = -1


def q8(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e4m3 value, saturating at +-448; NaN is not preserved.

    Round to nearest even onto the 3-bit-mantissa grid, with the fixed 2**-9 step below 2**-6 (the
    e4m3 subnormals). Returns float32.
    """
    x = x.to(torch.float32).clamp(-E4M3_MAX, E4M3_MAX)
    if skip_blocks and current_block in skip_blocks:
        return x.to(torch.float16).to(torch.float32)
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def e4m3_decode(raw: torch.Tensor) -> torch.Tensor:
    """uint8 e4m3 codes -> float32 values (0x7F and 0xFF decode to NaN)."""
    return raw.view(torch.float8_e4m3fn).to(torch.float32)
