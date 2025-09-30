#!/usr/bin/env python3
# coding=utf-8
"""Tests for the optional MLX packed W8/W4 quantized projection path."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_mlx_quant_import_safe():
    import rwkv7_hf.mlx_quant as mq

    assert hasattr(mq, "MLXQuantizedLinear")
    assert hasattr(mq, "quantize_mlx_mm8")
    assert hasattr(mq, "quantize_mlx_mm4")


def test_mlx_quant_formula_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_quant import (
        MLXQuantizedLinear,
        metal_quant_available,
        mm4_matmul_mlx,
        mm8_matmul_mlx,
        quantize_mlx_mm4,
        quantize_mlx_mm8,
    )

    mx.random.seed(20260704)
    x = mx.random.normal((2, 5)).astype(mx.float16)
    weight = mx.random.normal((5, 7)).astype(mx.float16)  # W used as x @ W

    q8 = quantize_mlx_mm8(weight)
    y8_ref = mm8_matmul_mlx(x, q8, backend="reference")
    y8_affine = mm8_matmul_mlx(x, q8, backend="affine")
    mx.eval(y8_ref, y8_affine)
    assert float(mx.max(mx.abs(y8_ref - y8_affine))) < 1e-2
    if metal_quant_available():
        q8_metal = quantize_mlx_mm8(weight, layout="metal")
        y8_metal = mm8_matmul_mlx(x, q8_metal, backend="metal")
        mx.eval(y8_metal)
        assert float(mx.max(mx.abs(y8_ref - y8_metal))) < 5e-2

    q4 = quantize_mlx_mm4(weight)
    y4_ref = mm4_matmul_mlx(x, q4, backend="reference")
    y4_affine = mm4_matmul_mlx(x, q4, backend="affine")
    mx.eval(y4_ref, y4_affine)
    assert float(mx.max(mx.abs(y4_ref - y4_affine))) < 1e-2
    if metal_quant_available():
        q4_metal = quantize_mlx_mm4(weight, layout="metal")
        y4_metal = mm4_matmul_mlx(x, q4_metal, backend="metal")
        mx.eval(y4_metal)
        assert float(mx.max(mx.abs(y4_ref - y4_metal))) < 5e-2

    # Linear weights are stored [out, in]; the helper quantizes weight.T.
    linear_weight = weight.T
    lin = MLXQuantizedLinear.from_linear_weight(linear_weight, bits=8, backend="affine")
    y = lin(x)
    mx.eval(y)
    assert tuple(int(v) for v in y.shape) == (2, 7)
    assert lin.telemetry()["bits"] == 8
    assert lin.telemetry()["backend_counts"]["affine"] == 1

    lin8_auto = MLXQuantizedLinear.from_linear_weight(linear_weight, bits=8, backend="auto")
    y8_auto = lin8_auto(x)
    mx.eval(y8_auto)
    assert tuple(int(v) for v in y8_auto.shape) == (2, 7)
    assert lin8_auto.telemetry()["last_backend"] == "affine"

    lin4_auto = MLXQuantizedLinear.from_linear_weight(linear_weight, bits=4, backend="auto")
    y4_auto = lin4_auto(x)
    mx.eval(y4_auto)
    assert tuple(int(v) for v in y4_auto.shape) == (2, 7)
    expected_auto_backend = "metal" if metal_quant_available() else "affine"
    assert lin4_auto.telemetry()["last_backend"] == expected_auto_backend
    if metal_quant_available():
        assert lin4_auto.telemetry()["auto_metal_max_rows"] == 4096

    old_limit = os.environ.get("RWKV7_MLX_QUANT_AUTO_W4_METAL_MAX_ROWS")
    os.environ["RWKV7_MLX_QUANT_AUTO_W4_METAL_MAX_ROWS"] = "1"
    try:
        lin4_auto_limited = MLXQuantizedLinear.from_linear_weight(linear_weight, bits=4, backend="auto")
        y4_auto_limited = lin4_auto_limited(x)
        mx.eval(y4_auto_limited)
        assert lin4_auto_limited.telemetry()["last_backend"] == "affine"
    finally:
        if old_limit is None:
            os.environ.pop("RWKV7_MLX_QUANT_AUTO_W4_METAL_MAX_ROWS", None)
        else:
            os.environ["RWKV7_MLX_QUANT_AUTO_W4_METAL_MAX_ROWS"] = old_limit


def test_mlx_model_quantized_linear_hook_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    _, model, _ = tiny_torch_model_to_mlx()
    dense_logits, _ = model.forward([[1, 2, 3]], collect_all=False)
    mx.eval(dense_logits)

    replaced = model.quantize_linears("mm8", min_params=1, backend="affine")
    assert replaced > 0
    assert "model.embeddings.weight" in model.arrays
    assert "lm_head.weight" in model.quantized_linears
    telemetry = model.telemetry()
    assert telemetry["quantized_linear_count"] == replaced
    assert telemetry["quantized_linear_bits"] == 8
    assert telemetry["quantized_linear_backend"] == "affine"
    assert telemetry["quantized_linear_last_backend_counts"] == {"reference": 0, "affine": 0, "metal": 0}
    assert telemetry["quantized_linear_bytes"] > 0
    assert telemetry["quantized_dense_equivalent_bytes"] > 0

    q_logits, q_state = model.forward([[1, 2, 3]], collect_all=False)
    mx.eval(q_logits)
    assert tuple(int(v) for v in q_logits.shape) == tuple(int(v) for v in dense_logits.shape)
    assert int(q_state.seen_tokens) == 3
    assert model.telemetry()["quantized_linear_last_backend_counts"]["affine"] > 0


def test_mlx_model_metal_quantized_linear_hook_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_quant import metal_quant_available
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    if not metal_quant_available():
        return

    _, model, _ = tiny_torch_model_to_mlx()
    replaced = model.quantize_linears("mm8", min_params=1, backend="metal")
    assert replaced > 0
    telemetry = model.telemetry()
    assert telemetry["quantized_linear_backend"] == "metal"
    logits, state = model.forward([[1, 2, 3]], collect_all=False)
    mx.eval(logits)
    assert tuple(int(v) for v in logits.shape[:2]) == (1, 1)
    assert int(state.seen_tokens) == 3


def test_mlx_model_auto_quantized_linear_hook_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_quant import metal_quant_available
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    _, model, _ = tiny_torch_model_to_mlx()
    replaced = model.quantize_linears("mm4", min_params=1, backend="auto")
    assert replaced > 0
    logits, state = model.forward([[1, 2, 3]], collect_all=False)
    mx.eval(logits)
    telemetry = model.telemetry()
    assert telemetry["quantized_linear_backend"] == "auto"
    if metal_quant_available():
        assert next(iter(model.quantized_linears.values())).telemetry()["auto_metal_max_rows"] == 4096
    expected = "metal" if metal_quant_available() else "affine"
    assert telemetry["quantized_linear_last_backend_counts"][expected] > 0
    assert int(state.seen_tokens) == 3


if __name__ == "__main__":
    test_mlx_quant_import_safe()
    test_mlx_quant_formula_if_available()
    test_mlx_model_quantized_linear_hook_if_available()
    test_mlx_model_metal_quantized_linear_hook_if_available()
    test_mlx_model_auto_quantized_linear_hook_if_available()
    print("MLX QUANT TESTS PASS")
