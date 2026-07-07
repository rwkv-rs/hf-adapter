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


def test_mlx_model_scan_prefill_matches_token_path_if_available(monkeypatch):
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    monkeypatch.setenv("RWKV7_MLX_WKV_SCAN_PREFILL", "1")
    _, token_model, cfg = tiny_torch_model_to_mlx()
    scan_model = MLXRWKV7Model.from_arrays(cfg, dict(token_model.arrays), wkv_backend="reference")
    ids = [[1, 2, 3, 4], [4, 3, 2, 1]]
    token_logits, token_state = token_model.forward(ids, collect_all=True)
    scan_logits, scan_state = scan_model.forward(ids, collect_all=True)
    mx.eval(token_logits, scan_logits, *token_state.recurrent_state, *scan_state.recurrent_state)
    assert tuple(int(x) for x in scan_logits.shape) == tuple(int(x) for x in token_logits.shape)
    assert float(mx.max(mx.abs(token_logits - scan_logits))) < 1e-4
    assert int(token_state.seen_tokens) == int(scan_state.seen_tokens) == 4
    telemetry = scan_model.telemetry()
    assert telemetry["wkv_scan_prefill"] is True
    assert telemetry["wkv_scan_prefill_counts"]["reference"] == int(cfg["num_hidden_layers"])


def test_mlx_model_scan_prefill_state_only_if_available(monkeypatch):
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    monkeypatch.setenv("RWKV7_MLX_WKV_SCAN_PREFILL", "1")
    _, token_model, cfg = tiny_torch_model_to_mlx()
    scan_model = MLXRWKV7Model.from_arrays(cfg, dict(token_model.arrays), wkv_backend="reference")
    ids = [[1, 2, 3, 4]]
    token_state = token_model.prefill_state_only(ids)
    scan_state = scan_model.prefill_state_only(ids)
    mx.eval(*token_state.recurrent_state, *scan_state.recurrent_state)
    assert int(token_state.seen_tokens) == int(scan_state.seen_tokens) == 4
    for a, b in zip(token_state.recurrent_state, scan_state.recurrent_state, strict=True):
        assert float(mx.max(mx.abs(a - b))) < 1e-4
    telemetry = scan_model.telemetry()
    assert telemetry["state_only_prefill_calls"] == 1
    assert telemetry["state_only_prefill_tokens"] == 4
