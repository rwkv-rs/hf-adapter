#!/usr/bin/env python3
# coding=utf-8
"""Tests for the optional MLX/Metal RWKV-7 WKV update seam."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_mlx_wkv_import_safe():
    import rwkv7_hf.mlx_wkv as mw

    assert hasattr(mw, "wkv_update")
    assert hasattr(mw, "wkv_update_reference")
    assert hasattr(mw, "wkv_update_metal")
    assert isinstance(mw.metal_wkv_available(), bool)


def test_mlx_wkv_formula_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_wkv import metal_wkv_available, wkv_update_metal, wkv_update_reference

    mx.random.seed(20260704)
    b, h, n = 2, 3, 8
    state = mx.random.normal((b, h, n, n)).astype(mx.float32)
    w = mx.sigmoid(mx.random.normal((b, h, n))).astype(mx.float32)
    v = mx.random.normal((b, h, n)).astype(mx.float16)
    k = mx.random.normal((b, h, n)).astype(mx.float16)
    kk = mx.random.normal((b, h, n)).astype(mx.float16)
    a = mx.sigmoid(mx.random.normal((b, h, n))).astype(mx.float16)
    r = mx.random.normal((b, h, n)).astype(mx.float16)

    out_ref, state_ref = wkv_update_reference(state, w, v, k, kk, a, r)
    vk = v.reshape(b, h, n, 1) @ k.reshape(b, h, 1, n)
    ab = (-kk).reshape(b, h, n, 1) @ (kk * a).reshape(b, h, 1, n)
    state_orig = state * w.reshape(b, h, 1, n) + state @ ab.astype(mx.float32) + vk.astype(mx.float32)
    out_orig = (state_orig.astype(r.dtype) @ r.reshape(b, h, n, 1)).reshape(b, h, n)
    mx.eval(out_ref, state_ref, out_orig, state_orig)
    assert float(mx.max(mx.abs(state_ref - state_orig))) < 1e-5
    assert float(mx.max(mx.abs(out_ref.astype(mx.float32) - out_orig.astype(mx.float32)))) < 2e-2

    if metal_wkv_available():
        out_metal, state_metal = wkv_update_metal(state, w, v, k, kk, a, r)
        mx.eval(out_metal, state_metal)
        # The Metal path uses the algebraically fused update and skips the
        # materialized fp16 `ab`/`vk` matrices, so tiny fp16-order drift is
        # expected. Keep this tight enough to catch layout/indexing bugs.
        assert float(mx.max(mx.abs(state_ref - state_metal))) < 1e-2
        assert float(mx.max(mx.abs(out_ref.astype(mx.float32) - out_metal.astype(mx.float32)))) < 5e-2


def test_mlx_model_metal_wkv_hook_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from rwkv7_hf.mlx_wkv import metal_wkv_available
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    if not metal_wkv_available():
        return

    _, ref_model, cfg = tiny_torch_model_to_mlx()
    metal_model = MLXRWKV7Model.from_arrays(cfg, dict(ref_model.arrays), wkv_backend="metal")
    ids = [[1, 2, 3, 4], [4, 3, 2, 1]]
    ref_logits, ref_state = ref_model.forward(ids, collect_all=True)
    metal_logits, metal_state = metal_model.forward(ids, collect_all=True)
    mx.eval(ref_logits, metal_logits)
    assert float(mx.max(mx.abs(ref_logits - metal_logits))) < 2e-1
    assert int(ref_state.seen_tokens) == int(metal_state.seen_tokens) == 4
    telemetry = metal_model.telemetry()
    assert telemetry["wkv_backend"] == "metal"
    assert telemetry["wkv_backend_last"] == "metal"
    assert telemetry["wkv_backend_counts"]["metal"] > 0


def test_mlx_prefill_eval_interval_parity_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    _, eager_model, cfg = tiny_torch_model_to_mlx()
    batched_model = MLXRWKV7Model.from_arrays(cfg, dict(eager_model.arrays))
    eager_model.prefill_eval_interval = 1
    batched_model.prefill_eval_interval = 4
    ids = [[1, 2, 3, 4], [4, 3, 2, 1]]
    eager_logits, eager_state = eager_model.prefill(ids)
    batched_logits, batched_state = batched_model.prefill(ids)
    mx.eval(
        eager_logits,
        batched_logits,
        eager_state.v_first,
        batched_state.v_first,
        *eager_state.recurrent_state,
        *batched_state.recurrent_state,
        *eager_state.attn_x_prev,
        *batched_state.attn_x_prev,
        *eager_state.ffn_x_prev,
        *batched_state.ffn_x_prev,
    )
    assert float(mx.max(mx.abs(eager_logits.astype(mx.float32) - batched_logits.astype(mx.float32)))) < 1e-5
    assert mx.argmax(eager_logits[:, -1, :], axis=-1).tolist() == mx.argmax(
        batched_logits[:, -1, :], axis=-1
    ).tolist()
    assert float(mx.max(mx.abs(eager_state.v_first - batched_state.v_first))) < 1e-5
    for eager_arrays, batched_arrays in (
        (eager_state.recurrent_state, batched_state.recurrent_state),
        (eager_state.attn_x_prev, batched_state.attn_x_prev),
        (eager_state.ffn_x_prev, batched_state.ffn_x_prev),
    ):
        for eager_layer, batched_layer in zip(eager_arrays, batched_arrays, strict=True):
            assert float(mx.max(mx.abs(eager_layer - batched_layer))) < 1e-5
    assert int(eager_state.seen_tokens) == int(batched_state.seen_tokens) == 4
    assert batched_model.telemetry()["prefill_eval_interval"] == 4


if __name__ == "__main__":
    test_mlx_wkv_import_safe()
    test_mlx_wkv_formula_if_available()
    test_mlx_model_metal_wkv_hook_if_available()
    test_mlx_prefill_eval_interval_parity_if_available()
    print("MLX WKV TESTS PASS")


def test_mlx_model_decode_step_matches_forward_t1_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    _, base_model, cfg = tiny_torch_model_to_mlx()
    old_model = MLXRWKV7Model.from_arrays(cfg, dict(base_model.arrays), wkv_backend="reference")
    new_model = MLXRWKV7Model.from_arrays(cfg, dict(base_model.arrays), wkv_backend="reference")
    _, old_state = old_model.prefill([[1, 2, 3]])
    _, new_state = new_model.prefill([[1, 2, 3]])
    old_logits, old_state = old_model.forward([[4]], state=old_state, collect_all=False)
    new_logits, new_state = new_model.decode_step([4], new_state)
    mx.eval(old_logits, new_logits)
    assert float(mx.max(mx.abs(old_logits - new_logits))) < 1e-5
    assert int(old_state.seen_tokens) == int(new_state.seen_tokens) == 4


def test_mlx_model_fast_group_norm_matches_manual_if_available(monkeypatch):
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    _, base_model, cfg = tiny_torch_model_to_mlx()
    ids = [[1, 2, 3, 4], [4, 3, 2, 1]]
    monkeypatch.setenv("RWKV7_MLX_FAST_GROUP_NORM", "0")
    manual = MLXRWKV7Model.from_arrays(cfg, dict(base_model.arrays), wkv_backend="reference")
    monkeypatch.setenv("RWKV7_MLX_FAST_GROUP_NORM", "1")
    fast = MLXRWKV7Model.from_arrays(cfg, dict(base_model.arrays), wkv_backend="reference")
    manual_logits, _ = manual.forward(ids, collect_all=True)
    fast_logits, _ = fast.forward(ids, collect_all=True)
    mx.eval(manual_logits, fast_logits)
    assert float(mx.max(mx.abs(manual_logits - fast_logits))) < 1e-5
    assert fast.telemetry()["fast_group_norm"] is True
