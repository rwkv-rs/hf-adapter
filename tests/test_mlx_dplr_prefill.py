#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mlx_dplr_prefill_import_safe() -> None:
    import rwkv7_hf.mlx_dplr_prefill as dplr

    assert callable(dplr.mlx_compact_wy_chunk_summary)
    assert callable(dplr.mlx_compact_wy_chunk_summary_metal)
    assert callable(dplr.mlx_compact_wy_prefix_combine)
    assert callable(dplr.mlx_compact_wy_chunk_apply)
    assert callable(dplr.mlx_compact_wy_chunk_apply_metal)
    assert callable(dplr.mlx_compact_wy_three_stage)
    assert callable(dplr.mlx_compact_wy_three_stage_metal)


def test_mlx_dplr_three_stage_matches_recurrent_scan_if_available() -> None:
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_dplr_prefill import (
        mlx_compact_wy_chunk_summary,
        mlx_compact_wy_chunk_summary_metal,
        mlx_compact_wy_chunk_apply_metal,
        mlx_compact_wy_prefix_combine,
        mlx_compact_wy_summary_to_dense,
        mlx_compact_wy_three_stage,
        mlx_compact_wy_three_stage_metal,
        mlx_dplr_recurrent_scan_reference,
        mlx_dplr_metal_available,
    )

    mx.random.seed(7007)
    batch, tokens, heads, head_dim = 1, 8, 2, 4
    shape = (batch, tokens, heads, head_dim)
    r = mx.random.normal(shape).astype(mx.float32) * 0.2
    w = mx.sigmoid(mx.random.normal(shape)).astype(mx.float32)
    k = mx.random.normal(shape).astype(mx.float32) * 0.2
    v = mx.random.normal(shape).astype(mx.float32) * 0.2
    kk = mx.random.normal(shape).astype(mx.float32) * 0.2
    a = mx.random.normal(shape).astype(mx.float32) * 0.2
    state = mx.random.normal((batch, heads, head_dim, head_dim)).astype(mx.float32) * 0.2

    ref_out, ref_state = mlx_dplr_recurrent_scan_reference(r, w, k, v, kk, a, state)
    summary = mlx_compact_wy_chunk_summary(w, k, v, kk, a, chunk_size=4)
    dense = mlx_compact_wy_summary_to_dense(summary)
    start_states, prefix_state = mlx_compact_wy_prefix_combine(state, summary)
    got_out, got_state, telemetry = mlx_compact_wy_three_stage(
        r, w, k, v, kk, a, state, chunk_size=4
    )
    mx.eval(
        ref_out,
        ref_state,
        dense["transition"],
        dense["additive"],
        start_states,
        prefix_state,
        got_out,
        got_state,
        telemetry["chunk_ends"],
    )

    assert summary["transition_diag"].shape == (1, 2, 2, 4)
    assert summary["transition_left"].shape == (1, 2, 2, 4, 4)
    assert summary["transition_right"].shape == (1, 2, 2, 4, 4)
    assert summary["additive_left"].shape == (1, 2, 2, 4, 4)
    assert summary["additive_right"].shape == (1, 2, 2, 4, 4)
    assert start_states.shape == (1, 2, 2, 4, 4)
    # Compact factor application groups fp32 operations differently from the
    # token recurrence. The CUDA compact-WY oracle uses the same tolerance
    # class; this remains tight enough to catch layout/sign errors.
    assert float(mx.max(mx.abs(prefix_state - ref_state))) < 5e-5
    assert float(mx.max(mx.abs(got_state - ref_state))) < 5e-5
    assert float(mx.max(mx.abs(got_out - ref_out))) < 5e-5
    assert float(mx.max(mx.abs(telemetry["chunk_ends"][:, -1] - ref_state))) < 5e-5

    if mlx_dplr_metal_available():
        metal_summary = mlx_compact_wy_chunk_summary_metal(w, k, v, kk, a, chunk_size=4)
        metal_starts, metal_state = mlx_compact_wy_prefix_combine(state, metal_summary)
        mx.eval(*[metal_summary[key] for key in (
            "transition_diag",
            "transition_left",
            "transition_right",
            "additive_left",
            "additive_right",
        )], metal_starts, metal_state)
        for key in (
            "transition_diag",
            "transition_left",
            "transition_right",
            "additive_left",
            "additive_right",
        ):
            assert float(mx.max(mx.abs(metal_summary[key] - summary[key]))) < 5e-5, key
        assert float(mx.max(mx.abs(metal_starts - start_states))) < 5e-5
        assert float(mx.max(mx.abs(metal_state - ref_state))) < 5e-5
        metal_out, metal_ends = mlx_compact_wy_chunk_apply_metal(
            r, w, k, v, kk, a, metal_starts, chunk_size=4
        )
        full_metal_out, full_metal_state, full_metal_telemetry = mlx_compact_wy_three_stage_metal(
            r, w, k, v, kk, a, state, chunk_size=4
        )
        mx.eval(
            metal_out,
            metal_ends,
            full_metal_out,
            full_metal_state,
            full_metal_telemetry["chunk_ends"],
        )
        # Metal uses fused multiply-add ordering inside each state row.
        assert float(mx.max(mx.abs(metal_out - ref_out))) < 2e-4
        assert float(mx.max(mx.abs(metal_ends[:, -1] - ref_state))) < 2e-4
        assert float(mx.max(mx.abs(full_metal_out - ref_out))) < 2e-4
        assert float(mx.max(mx.abs(full_metal_state - ref_state))) < 2e-4


def test_mlx_dplr_summary_rejects_partial_chunk_if_available() -> None:
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx
    import pytest

    from rwkv7_hf.mlx_dplr_prefill import mlx_compact_wy_chunk_summary

    x = mx.ones((1, 5, 1, 4), dtype=mx.float32)
    with pytest.raises(ValueError, match="must be divisible"):
        mlx_compact_wy_chunk_summary(x, x, x, x, x, chunk_size=4)


def test_mlx_dplr_prefill_bench_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "mlx_dplr_plan.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/mlx_dplr_prefill_bench.py",
            "--tokens",
            "32",
            "--chunk-size",
            "8",
            "--results",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["axis"] == "mlx_dplr_prefill_stage_env"
    assert row["status"] == "plan"
    assert row["tokens"] == 32
    assert row["chunk_size"] == 8


if __name__ == "__main__":
    test_mlx_dplr_prefill_import_safe()
    test_mlx_dplr_three_stage_matches_recurrent_scan_if_available()
    test_mlx_dplr_summary_rejects_partial_chunk_if_available()
    print("MLX DPLR PREFILL TESTS PASS")
