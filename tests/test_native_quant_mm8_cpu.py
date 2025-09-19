#!/usr/bin/env python3
# coding=utf-8
"""CPU/no-CUDA regression coverage for native mm8 quantization.

This keeps the optional official-RWKV-style mm8 path safe for CI and CPU-only
contributors: quantized modules may request ``fused=True`` by default, but a
CPU forward must fall back to the portable dequant+matmul path instead of trying
to launch a Triton kernel without an active CUDA driver.
"""
from __future__ import annotations

import torch

from rwkv7_hf.native_quant_mm8 import (
    MM8Linear,
    mm8_gemv_available,
    mm8_matmul,
    mm8_matmul_triton,
    quantize_mm8,
    quantize_model_mm8,
)


def main() -> int:
    torch.manual_seed(1234)
    assert not torch.cuda.is_available(), "run this smoke with CUDA_VISIBLE_DEVICES=''"
    assert not mm8_gemv_available(), "no-CUDA CI should not advertise fused Triton availability"

    lin = torch.nn.Linear(32, 16)
    q = MM8Linear(lin, fused=True)
    x = torch.randn(6, 32)
    out = q(x)
    assert out.shape == (6, 16)
    assert torch.isfinite(out).all()

    # Direct matmul_triton wrapper should also fall back for CPU or large 2D
    # inputs, matching the reference shape without launching Triton.
    wu8, mx, rx, my, ry = quantize_mm8(lin.weight.detach().t().contiguous())
    ref = mm8_matmul(x, wu8, mx, rx, my, ry)
    got = mm8_matmul_triton(x, wu8, mx, rx, my, ry)
    assert torch.allclose(got, ref, atol=0.0, rtol=0.0)

    model = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8))
    replaced = quantize_model_mm8(model, min_params=1, fused=False)
    assert replaced == 2
    assert all(isinstance(m, MM8Linear) and not m.fused for m in (model[0], model[2]))
    y = model(torch.randn(2, 32))
    assert y.shape == (2, 8)
    assert torch.isfinite(y).all()

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
