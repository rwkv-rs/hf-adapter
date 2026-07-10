#!/usr/bin/env python3
# coding=utf-8
"""Tests for the optional MLX/Metal RWKV-7 WKV update seam."""
from __future__ import annotations

import importlib.util
import json
import subprocess
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


def test_mlx_compiled_decode_matches_eager_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from rwkv7_hf.mlx_wkv import metal_wkv_available
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    if not metal_wkv_available() or not callable(getattr(mx, "compile", None)):
        return
    _, source_model, cfg = tiny_torch_model_to_mlx()
    eager_model = MLXRWKV7Model.from_arrays(cfg, dict(source_model.arrays), wkv_backend="metal")
    compiled_model = MLXRWKV7Model.from_arrays(cfg, dict(source_model.arrays), wkv_backend="metal")
    eager_model.decode_backend = "eager"
    compiled_model.decode_backend = "auto"
    compile_s = compiled_model.prepare_compiled_decode(batch_size=2)
    assert compile_s >= 0.0

    ids = [[1, 2, 3, 4], [4, 3, 2, 1]]
    eager_logits, eager_state = eager_model.prefill(ids)
    compiled_logits, compiled_state = compiled_model.prefill(ids)
    # Merely compiling a graph is not enough to promote it in auto mode.
    # Model-dependent fusion drift is parity-gated below.
    probe_token = mx.argmax(compiled_logits[:, -1, :], axis=-1).astype(mx.int32)
    _, probe_state = compiled_model.decode_step(probe_token, compiled_state.clone())
    assert compiled_model.decode_backend_last == "eager"
    assert compiled_model.decode_backend_counts["eager"] == 1
    assert int(probe_state.seen_tokens) == 5
    validation = compiled_model.validate_compiled_decode(
        compiled_logits,
        compiled_state,
        steps=4,
    )
    assert validation["status"] == "pass"
    eager_tokens: list[list[int]] = []
    compiled_tokens: list[list[int]] = []
    for _ in range(4):
        eager_token = mx.argmax(eager_logits[:, -1, :], axis=-1).astype(mx.int32)
        compiled_token = mx.argmax(compiled_logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(eager_token, compiled_token)
        eager_tokens.append([int(value) for value in eager_token.tolist()])
        compiled_tokens.append([int(value) for value in compiled_token.tolist()])
        eager_logits, eager_state = eager_model.decode_step(eager_token, eager_state)
        compiled_logits, compiled_state = compiled_model.decode_step(compiled_token, compiled_state)
        mx.eval(
            eager_logits,
            compiled_logits,
            *eager_state.recurrent_state,
            *compiled_state.recurrent_state,
        )

    assert eager_tokens == compiled_tokens
    assert float(mx.max(mx.abs(eager_logits.astype(mx.float32) - compiled_logits.astype(mx.float32)))) < 1e-5
    for eager_layer, compiled_layer in zip(
        eager_state.recurrent_state,
        compiled_state.recurrent_state,
        strict=True,
    ):
        assert float(mx.max(mx.abs(eager_layer - compiled_layer))) < 1e-5
    assert int(eager_state.seen_tokens) == int(compiled_state.seen_tokens) == 8
    telemetry = compiled_model.telemetry()
    assert telemetry["decode_backend"] == "auto"
    assert telemetry["decode_backend_last"] == "compiled"
    assert telemetry["decode_backend_counts"]["eager"] == 1
    assert telemetry["decode_backend_counts"]["compiled"] == 4
    assert telemetry["decode_compiled_batches"] == [2]
    assert telemetry["decode_compiled_validated_batches"] == [2]
    assert telemetry["decode_compiled_rejected_batches"] == []
    assert telemetry["decode_compiled_validation_by_batch"][2]["status"] == "pass"
    assert telemetry["decode_compile_s_by_batch"][2] >= 0.0


def test_mlx_decode_compile_bench_dry_run(tmp_path: Path):
    output = tmp_path / "mlx_decode_compile_plan.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/mlx_decode_compile_bench.py",
            "--models",
            "/tmp/rwkv-a,/tmp/rwkv-b",
            "--decode-tokens",
            "16",
            "--results",
            str(output),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["axis"] == "mlx_decode_compile_env"
    assert row["models"] == ["/tmp/rwkv-a", "/tmp/rwkv-b"]
    assert row["decode_tokens"] == 16


if __name__ == "__main__":
    test_mlx_wkv_import_safe()
    test_mlx_wkv_formula_if_available()
    test_mlx_model_metal_wkv_hook_if_available()
    test_mlx_prefill_eval_interval_parity_if_available()
    test_mlx_compiled_decode_matches_eager_if_available()
    print("MLX WKV TESTS PASS")
