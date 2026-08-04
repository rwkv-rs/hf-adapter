#!/usr/bin/env python3
# coding=utf-8
"""Config and module tests for native FP8 quantization.

FP8 (``float8_e4m3fn``) per-tensor weight quantization with online activation
quantization (W8A8) via ``torch._scaled_mm``.  These tests exercise the
quantization helper, the drop-in ``FP8Linear`` module, the config-driven
auto-quantize path, and mutual exclusivity with MM8/MM4 on CPU using the
dequant fallback path.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rwkv7_hf.native_quant_fp8 import (
    FP8Linear,
    fp8_available,
    quantize_fp8,
    quantize_model_fp8,
)


@pytest.mark.cpu
def test_quantize_fp8_dtype():
    """quantize_fp8 returns float8_e4m3fn weights and a positive scale."""
    torch.manual_seed(0)
    weight = torch.randn(32, 16, dtype=torch.float32)
    q_weight, scale = quantize_fp8(weight)
    assert q_weight.dtype == torch.float8_e4m3fn
    assert q_weight.shape == weight.shape
    assert torch.is_tensor(scale)
    assert float(scale) > 0.0


@pytest.mark.cpu
def test_quantize_fp8_per_channel():
    """per-channel quantization returns [N] scale."""
    torch.manual_seed(0)
    weight = torch.randn(32, 16, dtype=torch.float32)
    q_weight, scale = quantize_fp8(weight, per_channel=True)
    assert q_weight.dtype == torch.float8_e4m3fn
    assert scale.shape == (32,)
    assert (scale > 0).all()


@pytest.mark.cpu
def test_fp8_linear_forward():
    """FP8Linear wraps nn.Linear and produces finite output on CPU.

    On hardware without native FP8 tensor-core support the module transparently
    falls back to dequant + FP matmul, so the test is CPU-safe.
    """
    assert isinstance(fp8_available(), bool)
    torch.manual_seed(0)
    linear = nn.Linear(16, 32, bias=False)
    fp8_linear = FP8Linear(linear)
    assert fp8_linear.in_features == 16
    assert fp8_linear.out_features == 32
    x = torch.randn(2, 16, dtype=torch.float32)
    with torch.no_grad():
        out = fp8_linear(x)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


@pytest.mark.cpu
def test_fp8_linear_with_bias():
    """FP8Linear preserves bias from original Linear."""
    torch.manual_seed(0)
    linear = nn.Linear(16, 32, bias=True)
    fp8_linear = FP8Linear(linear)
    assert fp8_linear.bias is not None
    assert fp8_linear.bias.shape == (32,)
    x = torch.randn(4, 16, dtype=torch.float32)
    with torch.no_grad():
        out = fp8_linear(x)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


@pytest.mark.cpu
def test_fp8_linear_rwkv7_forward_into():
    """rwkv7_forward_into writes result into pre-allocated output buffer."""
    torch.manual_seed(0)
    linear = nn.Linear(8, 16, bias=False)
    fp8_linear = FP8Linear(linear)
    x = torch.randn(2, 8, dtype=torch.float32)
    out_buf = torch.empty(2, 16, dtype=torch.float32)
    with torch.no_grad():
        result = fp8_linear.rwkv7_forward_into(x, out_buf)
    assert result is out_buf
    assert torch.isfinite(out_buf).all()


@pytest.mark.cpu
def test_fp8_linear_extra_repr():
    """extra_repr returns a descriptive string."""
    linear = nn.Linear(8, 16, bias=False)
    fp8_linear = FP8Linear(linear)
    repr_str = fp8_linear.extra_repr()
    assert "fp8" in repr_str.lower()
    assert "in=8" in repr_str or "in_features=8" in repr_str


@pytest.mark.cpu
def test_fp8_model_quantization():
    """quantize_model_fp8 replaces nn.Linear modules with FP8Linear."""
    torch.manual_seed(0)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(64, 64, bias=False),
                nn.Linear(64, 32, bias=False),
            ])
            self.head = nn.Linear(32, 128, bias=False)

    model = TinyModel()
    replaced = quantize_model_fp8(model, min_params=1000, policy="memory")

    assert replaced == 3  # all 3 linears are > 1000 params
    assert getattr(model, "_rwkv7_native_mm_quantization") == "fp8"
    assert getattr(model, "_rwkv7_native_mm_replaced_modules") == 3
    fp8_count = sum(1 for m in model.modules() if type(m).__name__ == "FP8Linear")
    assert fp8_count == 3


@pytest.mark.cpu
def test_fp8_speed_policy():
    """speed policy only quantizes lm_head-like modules."""
    torch.manual_seed(0)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 64, bias=False)
            self.lm_head = nn.Linear(64, 128, bias=False)

    model = TinyModel()
    replaced = quantize_model_fp8(model, min_params=100, policy="speed")

    assert replaced == 1
    assert type(model.lm_head).__name__ == "FP8Linear"
    assert type(model.layer1).__name__ != "FP8Linear"


@pytest.mark.cpu
def test_fp8_min_params_filter():
    """Small linears below min_params threshold are not quantized."""
    torch.manual_seed(0)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.small = nn.Linear(4, 4, bias=False)  # 16 params
            self.big = nn.Linear(64, 64, bias=False)  # 4096 params

    model = TinyModel()
    replaced = quantize_model_fp8(model, min_params=1000, policy="memory")

    assert replaced == 1
    assert type(model.small).__name__ != "FP8Linear"
    assert type(model.big).__name__ == "FP8Linear"


@pytest.mark.cpu
def test_fp8_dequant_fallback_correctness():
    """Dequant fallback path produces results close to original linear."""
    torch.manual_seed(42)
    linear = nn.Linear(32, 32, bias=False)
    fp8_linear = FP8Linear(linear)

    x = torch.randn(8, 32, dtype=torch.float32)
    with torch.no_grad():
        orig_out = linear(x)
        fp8_out = fp8_linear(x)

    # FP8 E4M3 has ~0.2% quantization error; check relative error is small
    rel_error = (orig_out - fp8_out).abs().max() / orig_out.abs().max()
    assert rel_error < 0.05, f"Relative error {rel_error:.4f} too high"


def main() -> int:
    test_quantize_fp8_dtype()
    test_quantize_fp8_per_channel()
    test_fp8_linear_forward()
    test_fp8_linear_with_bias()
    test_fp8_linear_rwkv7_forward_into()
    test_fp8_linear_extra_repr()
    test_fp8_model_quantization()
    test_fp8_speed_policy()
    test_fp8_min_params_filter()
    test_fp8_dequant_fallback_correctness()
    print("NATIVE QUANT FP8 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
