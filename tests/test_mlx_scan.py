#!/usr/bin/env python3
# coding=utf-8
"""Tests for the optional MLX/Metal multi-token WKV scan seam."""
from __future__ import annotations

import importlib.util


def test_mlx_scan_import_safe():
    import rwkv7_hf.mlx_scan as ms

    assert hasattr(ms, "wkv_scan")
    assert hasattr(ms, "metal_wkv_scan_available")


def test_mlx_wkv_scan_formula_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_scan import metal_wkv_scan_available, wkv_scan

    mx.random.seed(20260707)
    B, T, H, N = 1, 5, 2, 4
    state = mx.random.normal((B, H, N, N)).astype(mx.float32) * 0.1
    r = mx.random.normal((B, T, H, N)).astype(mx.float16)
    w = mx.sigmoid(mx.random.normal((B, T, H, N)).astype(mx.float32)).astype(mx.float16)
    v = mx.random.normal((B, T, H, N)).astype(mx.float16)
    k = mx.random.normal((B, T, H, N)).astype(mx.float16)
    kk_raw = mx.random.normal((B, T, H, N)).astype(mx.float32)
    kk = (kk_raw / mx.sqrt(mx.maximum(mx.sum(kk_raw * kk_raw, axis=-1, keepdims=True), 1e-12))).astype(mx.float16)
    a = mx.sigmoid(mx.random.normal((B, T, H, N)).astype(mx.float32)).astype(mx.float16)

    ref_out, ref_state, ref_backend = wkv_scan(state, w, v, k, kk, a, r, backend="reference")
    auto_out, auto_state, auto_backend = wkv_scan(state, w, v, k, kk, a, r, backend="auto")
    mx.eval(ref_out, ref_state, auto_out, auto_state)
    expected_backend = "metal" if metal_wkv_scan_available() else "reference"
    assert auto_backend == expected_backend
    assert ref_backend == "reference"
    assert tuple(int(x) for x in auto_out.shape) == (B, T, H, N)
    assert tuple(int(x) for x in auto_state.shape) == (B, H, N, N)
    assert float(mx.max(mx.abs(auto_out.astype(mx.float32) - ref_out.astype(mx.float32)))) < 5e-3
    assert float(mx.max(mx.abs(auto_state.astype(mx.float32) - ref_state.astype(mx.float32)))) < 5e-3
