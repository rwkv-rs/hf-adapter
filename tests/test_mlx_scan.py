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


def test_mlx_wkv_scan_serving_batch_local_state_if_available():
    """Exercise the B8/N64 geometry used by the thread-local scan path."""

    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_scan import metal_wkv_scan_available, wkv_scan

    if not metal_wkv_scan_available():
        return
    mx.random.seed(20260714)
    B, T, H, N = 8, 17, 4, 64
    state = mx.random.normal((B, H, N, N)).astype(mx.float32) * 0.02
    r = mx.random.normal((B, T, H, N)).astype(mx.float16)
    w = mx.sigmoid(mx.random.normal((B, T, H, N))).astype(mx.float16)
    v = mx.random.normal((B, T, H, N)).astype(mx.float16)
    k = mx.random.normal((B, T, H, N)).astype(mx.float16)
    kk_raw = mx.random.normal((B, T, H, N)).astype(mx.float32)
    kk = (kk_raw / mx.sqrt(mx.maximum(mx.sum(kk_raw * kk_raw, axis=-1, keepdims=True), 1e-12))).astype(
        mx.float16
    )
    a = mx.sigmoid(mx.random.normal((B, T, H, N))).astype(mx.float16)
    reference_out, reference_state, _ = wkv_scan(state, w, v, k, kk, a, r, backend="reference")
    actual_out, actual_state, backend = wkv_scan(state, w, v, k, kk, a, r, backend="metal")
    mx.eval(reference_out, reference_state, actual_out, actual_state)
    assert backend == "metal"
    assert float(mx.max(mx.abs(actual_out.astype(mx.float32) - reference_out.astype(mx.float32)))) <= 0.125
    assert float(mx.max(mx.abs(actual_state - reference_state))) <= 0.02


def test_mlx_wkv_scan_fused_post_matches_generic_fp16_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_scan import (
        metal_wkv_scan_available,
        wkv_scan,
        wkv_scan_post_metal_fp16,
    )

    if not metal_wkv_scan_available():
        return
    mx.random.seed(20260714)
    B, T, H, N = 2, 5, 3, 4
    state = mx.random.normal((B, H, N, N)).astype(mx.float32) * 0.02
    r = mx.random.normal((B, T, H, N)).astype(mx.float16)
    w = mx.sigmoid(mx.random.normal((B, T, H, N))).astype(mx.float16)
    v = mx.random.normal((B, T, H, N)).astype(mx.float16)
    k = mx.random.normal((B, T, H, N)).astype(mx.float16)
    kk_raw = mx.random.normal((B, T, H, N)).astype(mx.float32)
    kk = (kk_raw / mx.sqrt(mx.maximum(mx.sum(kk_raw * kk_raw, axis=-1, keepdims=True), 1e-12))).astype(
        mx.float16
    )
    a = mx.sigmoid(mx.random.normal((B, T, H, N))).astype(mx.float16)
    g = mx.sigmoid(mx.random.normal((B, T, H, N))).astype(mx.float16)
    norm_weight = mx.random.normal((H * N,)).astype(mx.float16)
    norm_bias = mx.random.normal((H * N,)).astype(mx.float16)
    r_k = mx.random.normal((H, N)).astype(mx.float16)

    out, generic_state, _ = wkv_scan(state, w, v, k, kk, a, r, backend="metal")
    xf = out.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    variance = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
    generic = ((xf - mean) * mx.rsqrt(variance + N * 1e-5)).astype(mx.float16)
    generic = generic * norm_weight.reshape(1, 1, H, N) + norm_bias.reshape(1, 1, H, N)
    sk = (r * k * r_k.reshape(1, 1, H, N)).sum(axis=-1, keepdims=True)
    generic = (generic + sk * v) * g
    fused, fused_state = wkv_scan_post_metal_fp16(
        state,
        w,
        v,
        k,
        kk,
        a,
        r,
        norm_weight,
        norm_bias,
        r_k,
        g,
    )
    mx.eval(generic, fused, generic_state, fused_state)
    assert float(mx.max(mx.abs(fused.astype(mx.float32) - generic.astype(mx.float32)))) <= 0.0078125
    assert float(mx.max(mx.abs(fused_state - generic_state))) == 0.0


def test_mlx_wkv_scan_fused_prep_post_matches_split_fp16_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_scan import metal_wkv_scan_available, wkv_scan_post_metal_fp16

    if not metal_wkv_scan_available():
        return
    mx.random.seed(20260715)
    batch, tokens, heads, head_dim = 2, 7, 3, 4
    shape = (batch, tokens, heads, head_dim)
    state = mx.random.normal((batch, heads, head_dim, head_dim)).astype(mx.float32) * 0.02
    raw_w = mx.random.normal(shape).astype(mx.float16)
    raw_k = mx.random.normal(shape).astype(mx.float16)
    raw_a = mx.random.normal(shape).astype(mx.float16)
    raw_v = mx.random.normal(shape).astype(mx.float16)
    v_first = mx.random.normal(shape).astype(mx.float16)
    raw_v_mix = mx.random.normal(shape).astype(mx.float16)
    r = mx.random.normal(shape).astype(mx.float16)
    g = mx.random.normal(shape).astype(mx.float16)
    k_k = mx.random.normal((heads, head_dim)).astype(mx.float16)
    k_a = mx.random.normal((heads, head_dim)).astype(mx.float16)
    norm_weight = mx.random.normal((heads * head_dim,)).astype(mx.float16)
    norm_bias = mx.random.normal((heads * head_dim,)).astype(mx.float16)
    r_k = mx.random.normal((heads, head_dim)).astype(mx.float16)

    a = mx.sigmoid(raw_a)
    kk_pre = raw_k * k_k.reshape(1, 1, heads, head_dim)
    kk_float = kk_pre.astype(mx.float32)
    kk = (
        kk_float
        / mx.sqrt(mx.maximum(mx.sum(kk_float * kk_float, axis=-1, keepdims=True), 1e-12))
    ).astype(mx.float16)
    k = raw_k * (1 + (a - 1) * k_a.reshape(1, 1, heads, head_dim))
    w = mx.exp(-0.606531 * mx.sigmoid(raw_w.astype(mx.float32)))
    v_mix = mx.sigmoid(raw_v_mix)
    v = raw_v + (v_first - raw_v) * v_mix
    split_out, split_state = wkv_scan_post_metal_fp16(
        state, w, v, k, kk, a, r, norm_weight, norm_bias, r_k, g
    )
    fused_out, fused_state = wkv_scan_post_metal_fp16(
        state,
        raw_w,
        raw_v,
        raw_k,
        raw_k,
        raw_a,
        r,
        norm_weight,
        norm_bias,
        r_k,
        g,
        preprocess=True,
        k_k=k_k,
        k_a=k_a,
        v_first=v_first,
        v_mix=raw_v_mix,
    )
    mx.eval(split_out, split_state, fused_out, fused_state)
    assert float(mx.max(mx.abs(fused_out.astype(mx.float32) - split_out.astype(mx.float32)))) <= 0.25
    assert float(mx.max(mx.abs(fused_state - split_state))) <= 0.1


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


def test_mlx_model_scan_prefill_auto_threshold_if_available(monkeypatch):
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    _, _token_model, cfg = tiny_torch_model_to_mlx()
    ids = [[1, 2, 3, 4]]

    monkeypatch.setenv("RWKV7_MLX_WKV_SCAN_PREFILL", "auto")
    monkeypatch.setenv("RWKV7_MLX_WKV_SCAN_PREFILL_MIN_TOKENS", "4")
    scan_model = MLXRWKV7Model.from_arrays(cfg, dict(_token_model.arrays), wkv_backend="reference")
    logits, state = scan_model.forward(ids, collect_all=True)
    mx.eval(logits)
    telemetry = scan_model.telemetry()
    assert int(state.seen_tokens) == 4
    assert telemetry["wkv_scan_prefill_mode"] == "auto"
    assert telemetry["wkv_scan_prefill_min_tokens"] == 4
    assert telemetry["wkv_scan_prefill_counts"]["reference"] == int(cfg["num_hidden_layers"])
    assert telemetry["wkv_scan_prefill_reason_counts"]["auto"] == 1

    monkeypatch.setenv("RWKV7_MLX_WKV_SCAN_PREFILL_MIN_TOKENS", "8")
    token_model = MLXRWKV7Model.from_arrays(cfg, dict(_token_model.arrays), wkv_backend="reference")
    logits, state = token_model.forward(ids, collect_all=True)
    mx.eval(logits)
    telemetry = token_model.telemetry()
    assert int(state.seen_tokens) == 4
    assert telemetry["wkv_scan_prefill_counts"]["reference"] == 0
    assert telemetry["wkv_scan_prefill_reason_counts"]["below_min_tokens"] == 1


def test_mlx_model_compiled_scan_prefill_is_shape_gated_if_available(monkeypatch):
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from rwkv7_hf.mlx_scan import metal_wkv_scan_available
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    if not callable(getattr(mx, "compile", None)) or not metal_wkv_scan_available():
        return
    monkeypatch.setenv("RWKV7_MLX_WKV_SCAN_PREFILL", "1")
    monkeypatch.setenv("RWKV7_MLX_COMPILED_SCAN_PREFILL", "auto")
    _, source_model, cfg = tiny_torch_model_to_mlx()
    model = MLXRWKV7Model.from_arrays(cfg, dict(source_model.arrays), wkv_backend="metal")
    ids = [[1, 2, 3, 4], [4, 3, 2, 1]]

    eager_logits, eager_state = model.prefill(ids)
    assert model.compiled_scan_prefill_backend_last == "eager"
    validation = model.validate_compiled_scan_prefill(ids)
    assert validation["status"] == "pass"
    assert validation["next_token_match"] is True
    assert validation["logits_max_abs"] == 0.0
    assert validation["state_max_abs"] == 0.0

    compiled_logits, compiled_state = model.prefill(ids)
    mx.eval(
        eager_logits,
        compiled_logits,
        *eager_state.recurrent_state,
        *compiled_state.recurrent_state,
    )
    assert float(mx.max(mx.abs(eager_logits - compiled_logits))) == 0.0
    assert int(eager_state.seen_tokens) == int(compiled_state.seen_tokens) == 4
    telemetry = model.telemetry()
    assert telemetry["compiled_scan_prefill_mode"] == "auto"
    assert telemetry["compiled_scan_prefill_backend_last"] == "compiled"
    assert telemetry["compiled_scan_prefill_validated_shapes"] == ["b2_t4_last"]
    assert telemetry["compiled_scan_prefill_validation"]["b2_t4_last"]["status"] == "pass"


def test_mlx_fused_lora_down_evicts_replaced_sources_if_available(monkeypatch):
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    monkeypatch.setenv("RWKV7_MLX_FUSED_LORA_DOWN", "0")
    _, reference, cfg = tiny_torch_model_to_mlx()
    arrays = dict(reference.arrays)
    monkeypatch.setenv("RWKV7_MLX_FUSED_LORA_DOWN", "1")
    monkeypatch.setenv("RWKV7_MLX_FUSED_LORA_DOWN_INCLUDE_G", "0")
    monkeypatch.setenv("RWKV7_MLX_FUSED_LORA_DOWN_INCLUDE_V", "0")
    monkeypatch.setenv("RWKV7_MLX_FUSED_LORA_DOWN_EVICT_SOURCE", "1")
    fused = MLXRWKV7Model.from_arrays(cfg, dict(arrays), wkv_backend="reference")

    replaced = [
        key
        for key in arrays
        if key.endswith((".w_lora.lora.0.weight", ".a_lora.lora.0.weight"))
    ]
    assert replaced
    assert all(key not in fused.arrays for key in replaced)
    telemetry = fused.telemetry()
    assert telemetry["fused_lora_down_source_bytes_released"] > 0
    assert telemetry["fused_lora_down_cache_bytes"] == 2 * telemetry[
        "fused_lora_down_source_bytes_released"
    ]

    ids = [[1, 2, 3, 4], [4, 3, 2, 1]]
    reference_logits, reference_state = reference.forward(ids, collect_all=True)
    fused_logits, fused_state = fused.forward(ids, collect_all=True)
    mx.eval(
        reference_logits,
        fused_logits,
        *reference_state.recurrent_state,
        *fused_state.recurrent_state,
    )
    assert mx.argmax(reference_logits, axis=-1).tolist() == mx.argmax(fused_logits, axis=-1).tolist()
    assert float(mx.max(mx.abs(reference_logits - fused_logits))) < 1e-4
    for actual, expected in zip(
        fused_state.recurrent_state,
        reference_state.recurrent_state,
        strict=True,
    ):
        assert float(mx.max(mx.abs(actual - expected))) < 1e-4
