#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.audit_qwen35_apple_goal import AUDIT_AXIS, SUMMARY_AXIS, Shape, Tier, run_audit
from bench.run_qwen35_apple_baseline import AXIS as BASELINE_AXIS
from bench.compare_qwen35_apple_baseline import COMPARISON_AXIS
from bench.score_qwen35_quality import COMPARISON_AXIS as QUALITY_COMPARISON_AXIS


def passing_rows() -> list[dict[str, object]]:
    return [
        {
            "axis": BASELINE_AXIS,
            "status": "pass",
            "engine": "mlx_vlm",
            "runtime": "mlx_vlm_token_only",
            "model": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            "prompt_case": "chars64",
            "prompt_target_chars": 64,
            "requested_generated_tokens": 8,
            "generated_tokens": 8,
            "decode_tok_s": 20.0,
            "prefill_tok_s": 100.0,
            "ttft_s": 0.2,
            "mlx_peak_memory_bytes": 1_000_000_000,
        },
        {
            "axis": BASELINE_AXIS,
            "status": "pass",
            "engine": "rwkv7_hf",
            "runtime": "mlx",
            "model": "rwkv7-g1d-0.4b-hf",
            "prompt_case": "chars64",
            "prompt_target_chars": 64,
            "requested_generated_tokens": 8,
            "generated_tokens": 8,
            "decode_tok_s": 25.0,
            "prefill_tok_s": 120.0,
            "ttft_s": 0.18,
            "mlx_peak_memory_bytes": 500_000_000,
            "quantization": "mm4",
            "chunked_prefill_max_abs": 0.0,
            "seen_tokens_after_generate": 72,
            "expected_seen_tokens": 72,
        },
        {
            "axis": COMPARISON_AXIS,
            "status": "pass",
            "qwen_model": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            "rwkv_model": "rwkv7-g1d-0.4b-hf",
            "prompt_case": "chars64",
            "requested_generated_tokens": 8,
        },
        {
            "axis": QUALITY_COMPARISON_AXIS,
            "status": "pass",
            "qwen_model": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            "rwkv_model": "rwkv7-g1d-0.4b-hf",
            "prompt_case": "chars64",
            "requested_generated_tokens": 8,
        },
        {
            "axis": BASELINE_AXIS,
            "status": "pass",
            "engine": "rwkv7_hf",
            "runtime": "coreml",
            "model": "rwkv7-g1d-0.4b-hf",
            "prompt_case": "chars64",
            "prompt_target_chars": 64,
            "requested_generated_tokens": 8,
            "decode_tok_s": 30.0,
            "prefill_tok_s": 120.0,
            "ttft_s": 0.15,
        },
    ]


def test_goal_audit_can_pass_complete_tiny_matrix() -> None:
    audits = run_audit(
        passing_rows(),
        tiers=[Tier(("qwen3.5:0.8b-mlx", "mlx-community/Qwen3.5-0.8B-MLX-4bit"), ("rwkv7-g1d-0.4b-hf",))],
        shapes=[Shape("chars64", 8)],
        state_tolerance=1e-4,
        long_context_chars=64,
        require_quality=True,
        require_coreml=True,
    )
    assert [row["axis"] for row in audits] == [AUDIT_AXIS, AUDIT_AXIS, SUMMARY_AXIS]
    assert {row["status"] for row in audits} == {"pass"}
    shape_row = audits[0]
    assert shape_row["checks"]["quant"]["status"] == "pass"
    assert shape_row["checks"]["state_cache"]["status"] == "pass"
    assert shape_row["checks"]["quality"]["status"] == "pass"
    assert audits[-1]["action_counts"] == {}


def test_goal_audit_reports_missing_rwkv_and_comparison_actions() -> None:
    rows = [
        {
            "axis": BASELINE_AXIS,
            "status": "pass",
            "model": "qwen3.5:2b-mlx",
            "prompt_case": "chars64",
            "prompt_target_chars": 64,
            "requested_generated_tokens": 8,
            "decode_tok_s": 40.0,
            "prefill_tok_s": 100.0,
            "ttft_s": 0.2,
            "peak_memory_bytes": 2_000_000_000,
        }
    ]
    audits = run_audit(
        rows,
        tiers=[Tier(("qwen3.5:2b-mlx",), ("rwkv7-g1g-1.5b-hf",))],
        shapes=[Shape("chars64", 8)],
        state_tolerance=1e-4,
        long_context_chars=64,
        require_quality=True,
        require_coreml=True,
    )
    shape_row = audits[0]
    assert shape_row["status"] == "missing"
    actions = {action["action"] for action in shape_row["actions"]}
    assert "collect_rwkv_mlx_rows" in actions
    assert "run_comparison_gates" in actions
    assert "collect_quality_rows_with_store_responses" in actions
    tier_row = audits[1]
    assert tier_row["checks"]["coreml_stateful_runtime"]["status"] == "missing"
    assert audits[-1]["status"] == "missing"
    assert audits[-1]["action_counts"]["collect_rwkv_mlx_rows"] == 1


def test_goal_audit_cli_appends_and_fail_on_gate(tmp_path: Path) -> None:
    source = tmp_path / "evidence.jsonl"
    local_qwen_row = dict(passing_rows()[0])
    local_qwen_row["model"] = "/tmp/qwen35-0.8b-mlx-4bit"
    source.write_text(json.dumps(local_qwen_row) + "\n", encoding="utf-8")
    appended = tmp_path / "audit.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "bench/audit_qwen35_apple_goal.py",
            "--results",
            str(source),
            "--tier",
            "qwen3.5:0.8b-mlx|mlx-community/Qwen3.5-0.8B-MLX-4bit=rwkv7-g1d-0.4b-hf",
            "--required-shape",
            "chars64:8",
            "--append",
            str(appended),
            "--fail-on-gate",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    rows = [json.loads(line) for line in appended.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["axis"] == AUDIT_AXIS
    assert rows[-1]["axis"] == SUMMARY_AXIS
    assert rows[-1]["status"] == "missing"
