#!/usr/bin/env python3
from __future__ import annotations

import importlib.util


def test_mlx_norm_import_safe():
    import rwkv7_hf.mlx_norm as module

    assert hasattr(module, "add_layer_norm_metal_fp16")


def test_mlx_add_layer_norm_matches_fast_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_norm import add_layer_norm_metal_fp16

    mx.random.seed(20260714)
    residual = mx.random.normal((2, 5, 2048)).astype(mx.float16)
    update = mx.random.normal((2, 5, 2048)).astype(mx.float16)
    weight = mx.random.normal((2048,)).astype(mx.float16)
    bias = mx.random.normal((2048,)).astype(mx.float16)
    expected_residual = residual + update
    expected_norm = mx.fast.layer_norm(expected_residual, weight, bias, 1e-5)
    actual_residual, actual_norm = add_layer_norm_metal_fp16(
        residual, update, weight, bias, 1e-5
    )
    mx.eval(expected_residual, expected_norm, actual_residual, actual_norm)
    residual_diff = float(
        mx.max(mx.abs(expected_residual.astype(mx.float32) - actual_residual.astype(mx.float32)))
    )
    norm_diff = float(mx.max(mx.abs(expected_norm.astype(mx.float32) - actual_norm.astype(mx.float32))))
    assert residual_diff == 0.0
    assert norm_diff <= 0.04
