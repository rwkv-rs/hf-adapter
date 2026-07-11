#!/usr/bin/env python3
# coding=utf-8
"""CPU correctness coverage for projection-axis packed W4 R/K/V GEMV."""
from __future__ import annotations

import torch

from rwkv7_hf.fused_time_mix import fused_attn_shift_mix_stacked_rkv
from rwkv7_hf.native_quant import (
    int4_fused_rkv_gemv,
    int4_stacked_rkv_gemv,
    quantize_int4_rowwise,
)


def run_case(hidden: int, batch: int) -> None:
    weights = [torch.randn(hidden, hidden) for _ in range(3)]
    activations = [torch.randn(batch, hidden) for _ in range(3)]
    packed = [quantize_int4_rowwise(weight) for weight in weights]
    q_stacked = torch.stack([item[0] for item in packed], dim=0)
    scales_stacked = torch.stack([item[1] for item in packed], dim=0)
    x_stacked = torch.stack(activations, dim=1)

    current = int4_fused_rkv_gemv(
        activations[0],
        activations[1],
        activations[2],
        packed[0][0],
        packed[1][0],
        packed[2][0],
        packed[0][1],
        packed[1][1],
        packed[2][1],
        force_fallback=True,
    )
    candidate = int4_stacked_rkv_gemv(
        x_stacked,
        q_stacked,
        scales_stacked,
        force_fallback=True,
    )
    assert candidate.shape == (batch, 3, hidden)
    for projection in range(3):
        assert torch.equal(candidate[:, projection, :], current[projection])


def main() -> int:
    torch.manual_seed(1234)
    run_case(hidden=32, batch=1)
    run_case(hidden=33, batch=2)

    hidden = 32
    batch = 2
    x = torch.randn(batch, hidden)
    previous = torch.randn_like(x)
    mixes = [torch.randn(hidden) for _ in range(6)]
    rkv, xw, xa, xg = fused_attn_shift_mix_stacked_rkv(
        x,
        previous,
        *mixes,
        force_fallback=True,
    )
    delta = previous - x
    expected = [x + delta * mix for mix in mixes]
    assert torch.allclose(rkv[:, 0, :], expected[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(rkv[:, 1, :], expected[2], atol=1e-6, rtol=1e-6)
    assert torch.allclose(rkv[:, 2, :], expected[3], atol=1e-6, rtol=1e-6)
    assert torch.allclose(xw, expected[1], atol=1e-6, rtol=1e-6)
    assert torch.allclose(xa, expected[4], atol=1e-6, rtol=1e-6)
    assert torch.allclose(xg, expected[5], atol=1e-6, rtol=1e-6)

    x = torch.randn(1, 2, 32)
    q = torch.zeros(3, 32, 16, dtype=torch.uint8)
    scales = torch.ones(3, 32)
    try:
        int4_stacked_rkv_gemv(x, q, scales, force_fallback=True)
    except ValueError as error:
        assert "[batch, 3, hidden]" in str(error)
    else:
        raise AssertionError("invalid projection dimension should fail")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
