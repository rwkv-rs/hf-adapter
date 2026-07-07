#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.run_qwen35_apple_baseline import (
    AXIS,
    PromptCase,
    build_prompt_cases,
    make_prompt,
    ollama_row_from_chunks,
    summarize_rows,
    tok_s,
)
from bench.compare_qwen35_apple_baseline import (
    COMPARISON_AXIS,
    DIAGNOSTIC_AXIS,
    SUMMARY_AXIS,
    Pair,
    compare_rows,
    comparison_gap_actions,
    gap_diagnostic_rows,
    summarize_comparisons,
)
from bench.score_qwen35_quality import (
    COMPARISON_AXIS as QUALITY_COMPARISON_AXIS,
    QUALITY_AXIS,
    SUMMARY_AXIS as QUALITY_SUMMARY_AXIS,
    Pair as QualityPair,
    compare_quality,
    score_rows,
    summarize as summarize_quality,
)
from scripts.ollama_pull_with_timeout import AXIS as PULL_AXIS
from scripts.ollama_pull_with_timeout import PullProgress, format_event


def test_make_prompt_hits_target_chars() -> None:
    prompt = make_prompt("abc", 8)
    assert prompt == "abcabcab"
    cases = build_prompt_cases([5, 7], "xy")
    assert [case.name for case in cases] == ["chars5", "chars7"]
    assert [len(case.prompt) for case in cases] == [5, 7]


def test_ollama_stream_row_uses_final_duration_metrics() -> None:
    case = PromptCase(name="chars16", target_chars=16, prompt="0123456789abcdef")
    chunks = [
        {"response": "hel", "done": False},
        {"response": "lo", "done": False},
        {
            "response": "",
            "done": True,
            "prompt_eval_count": 8,
            "prompt_eval_duration": 2_000_000_000,
            "eval_count": 4,
            "eval_duration": 1_000_000_000,
            "total_duration": 3_100_000_000,
            "load_duration": 100_000_000,
        },
    ]
    row = ollama_row_from_chunks(
        model="qwen3.5:0.8b-mlx",
        prompt_case=case,
        max_new_tokens=4,
        chunks=chunks,
        elapsed_s=3.2,
    )
    assert row["axis"] == AXIS
    assert row["status"] == "pass"
    assert row["engine"] == "ollama"
    assert row["prompt_eval_tokens"] == 8
    assert row["generated_tokens"] == 4
    assert row["prefill_tok_s"] == 4.0
    assert row["decode_tok_s"] == 4.0
    assert row["first_response_chunk_index"] == 0
    assert row["public_package_gb"] == 1.2


def test_ollama_row_can_store_full_response_for_quality() -> None:
    case = PromptCase(name="chars16", target_chars=16, prompt="0123456789abcdef")
    row = ollama_row_from_chunks(
        model="qwen3.5:0.8b-mlx",
        prompt_case=case,
        max_new_tokens=4,
        chunks=[
            {"response": "answer ", "done": False},
            {"response": "throughput", "done": False},
            {"response": "", "done": True, "eval_count": 2, "eval_duration": 1_000_000_000},
        ],
        elapsed_s=1.2,
        store_response=True,
    )
    assert row["response_preview"] == "answer throughput"
    assert row["response_text"] == "answer throughput"
    assert row["response_chars"] == len("answer throughput")


def test_ollama_pull_progress_times_out_repeated_non_progress() -> None:
    progress = PullProgress(model="qwen3.5:0.8b-mlx", timeout_s=120.0, idle_timeout_s=10.0, start_s=0.0)
    assert progress.observe({"status": "pulling manifest"}, now_s=1.0) is None
    assert progress.observe({"status": "pulling model", "digest": "sha256:model", "total": 100}, now_s=2.0) is None
    row = progress.observe({"status": "pulling model", "digest": "sha256:model", "total": 100}, now_s=13.5)
    assert row is not None
    assert row["axis"] == PULL_AXIS
    assert row["status"] == "fail"
    assert "no byte/status progress" in row["reason"]
    assert row["last_pull_event"]["digest"] == "sha256:model"


def test_ollama_pull_progress_accepts_completed_bytes_and_success() -> None:
    progress = PullProgress(model="qwen3.5:0.8b-mlx", timeout_s=120.0, idle_timeout_s=10.0, start_s=0.0)
    assert progress.observe({"status": "pulling model", "digest": "sha256:abc", "total": 100, "completed": 10}, now_s=1.0) is None
    assert progress.observe({"status": "pulling model", "digest": "sha256:abc", "total": 100, "completed": 20}, now_s=11.0) is None
    row = progress.observe({"status": "success"}, now_s=12.0)
    assert row is not None
    assert row["axis"] == PULL_AXIS
    assert row["status"] == "pass"
    assert row["last_completed"] == 20
    assert format_event({"status": "pulling model", "total": 100, "completed": 25}) == "pulling model 25.0% 25/100"


def test_quality_scoring_and_pair_comparison() -> None:
    rows = [
        {
            "axis": AXIS,
            "status": "pass",
            "engine": "ollama",
            "runtime": "ollama_mlx",
            "model": "qwen3.5:0.8b-mlx",
            "prompt_case": "chars64",
            "requested_generated_tokens": 8,
            "response_text": "throughput memory cache",
        },
        {
            "axis": AXIS,
            "status": "pass",
            "engine": "rwkv7_hf",
            "runtime": "mlx",
            "model": "rwkv7-g1d-0.4b-hf",
            "prompt_case": "chars64",
            "requested_generated_tokens": 8,
            "response_text": "throughput memory cache",
        },
    ]
    rubric = {
        "tasks": [
            {
                "id": "apple-metrics",
                "prompt_case": "chars64",
                "requested_generated_tokens": 8,
                "required_substrings": ["throughput", "memory", "cache"],
                "min_response_chars": 8,
            }
        ]
    }
    scored = score_rows(rows, rubric)
    assert [row["axis"] for row in scored] == [QUALITY_AXIS, QUALITY_AXIS]
    assert {row["status"] for row in scored} == {"pass"}
    assert {row["score"] for row in scored} == {1.0}
    comparisons = compare_quality(scored, [QualityPair("qwen3.5:0.8b-mlx", "rwkv7-g1d-0.4b-hf")])
    assert comparisons[0]["axis"] == QUALITY_COMPARISON_AXIS
    assert comparisons[0]["status"] == "pass"
    summary = summarize_quality(scored, comparisons)
    assert summary["axis"] == QUALITY_SUMMARY_AXIS
    assert summary["status"] == "pass"


def test_quality_cli_keeps_missing_response_unknown(tmp_path: Path) -> None:
    source = tmp_path / "baseline.jsonl"
    output = tmp_path / "quality.jsonl"
    source.write_text(
        json.dumps(
            {
                "axis": AXIS,
                "status": "pass",
                "model": "rwkv7-g1d-0.4b-hf",
                "prompt_case": "chars64",
                "requested_generated_tokens": 8,
                "response_preview": "truncated",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "bench/score_qwen35_quality.py",
            "--results",
            str(source),
            "--append",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["axis"] == QUALITY_AXIS
    assert rows[0]["status"] == "unknown"
    assert "missing response_text" in rows[0]["reasons"][0]
    assert rows[-1]["axis"] == QUALITY_SUMMARY_AXIS
    assert rows[-1]["status"] == "unknown"


def test_summary_groups_engines_and_min_decode() -> None:
    rows = [
        {
            "axis": AXIS,
            "status": "pass",
            "engine": "ollama",
            "runtime": "ollama_mlx",
            "decode_tok_s": 50.0,
            "prefill_tok_s": 100.0,
        },
        {
            "axis": AXIS,
            "status": "pass",
            "engine": "ollama",
            "runtime": "ollama_mlx",
            "decode_tok_s": 40.0,
            "prefill_tok_s": 90.0,
        },
        {
            "axis": AXIS,
            "status": "pass",
            "engine": "rwkv7_hf",
            "runtime": "mlx",
            "decode_tok_s": 70.0,
            "prefill_tok_s": 80.0,
            "ttft_s": 0.7,
        },
        {"axis": AXIS, "status": "skip", "engine": "rwkv7_hf", "runtime": "mlx"},
    ]
    summary = summarize_rows(rows)
    assert summary["axis"] == AXIS + "_summary"
    assert summary["pass_rows"] == 3
    assert summary["ollama_ollama_mlx_min_decode_tok_s"] == 40.0
    assert summary["rwkv7_hf_mlx_min_decode_tok_s"] == 70.0
    assert summary["rwkv7_hf_mlx_max_ttft_s"] == 0.7


def test_tok_s_rejects_empty_duration() -> None:
    assert tok_s(10, 0) is None
    assert tok_s(None, 10) is None
    assert tok_s(10, 2_000_000_000) == 5.0


def test_dry_run_cli_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "qwen35_plan.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "bench/run_qwen35_apple_baseline.py",
            "--dry-run",
            "--results",
            str(out),
            "--prompt-target-chars",
            "32,64",
            "--decode-lengths",
            "4,8",
            "--warmup-repeats",
            "2",
            "--qwen-models",
            "qwen3.5:0.8b-mlx",
            "--qwen-mlx-vlm-models",
            "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            "--qwen-mlx-vlm-token-only",
            "--rwkv-mlx-models",
            "/tmp/rwkv-a,/tmp/rwkv-b",
            "--rwkv-quant-min-params",
            "4000000",
            "--rwkv-quant-rkv-min-params",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["axis"] == AXIS + "_env"
    assert rows[1]["axis"] == AXIS + "_plan"
    assert rows[0]["warmup_repeats"] == 2
    assert rows[1]["qwen_jobs"] == 4
    assert rows[1]["qwen_mlx_vlm_jobs"] == 4
    assert rows[1]["warmup_repeats"] == 2
    assert rows[1]["warmup_jobs"] == 32
    assert rows[1]["qwen_mlx_vlm_models"] == ["mlx-community/Qwen3.5-0.8B-MLX-4bit"]
    assert rows[1]["qwen_mlx_vlm_token_only"] is True
    assert rows[1]["rwkv_quant_min_params"] == 4_000_000
    assert rows[1]["rwkv_quant_rkv_min_params"] == 0
    assert rows[1]["rwkv_mlx_jobs"] == 8


def test_acceptance_wrapper_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "acceptance.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RUN_QWEN": "0",
            "QWEN_MLX_VLM_MODELS": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            "QWEN_MLX_VLM_TOKEN_ONLY": "1",
            "RWKV_MLX_MODELS": "/tmp/rwkv-a,/tmp/rwkv-b",
            "RWKV_QUANT_RKV_MIN_PARAMS": "0",
            "PROMPT_TARGET_CHARS": "16",
            "DECODE_LENGTHS": "4",
            "WARMUP_REPEATS": "1",
            "RESULTS": str(out),
            "PYTHON_BIN": sys.executable,
            "SKIP_COMPARE": "1",
        }
    )
    result = subprocess.run(
        ["bash", "scripts/run_qwen35_apple_acceptance.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["axis"] == AXIS + "_env"
    assert rows[1]["axis"] == AXIS + "_plan"
    assert rows[0]["warmup_repeats"] == 1
    assert rows[1]["qwen_jobs"] == 0
    assert rows[1]["qwen_mlx_vlm_jobs"] == 1
    assert rows[1]["rwkv_mlx_jobs"] == 2
    assert rows[1]["warmup_repeats"] == 1
    assert rows[1]["warmup_jobs"] == 3
    assert rows[1]["qwen_mlx_vlm_models"] == ["mlx-community/Qwen3.5-0.8B-MLX-4bit"]
    assert rows[1]["qwen_mlx_vlm_token_only"] is True
    assert rows[1]["rwkv_quant_rkv_min_params"] == 0
    assert rows[1]["rwkv_mlx_models"] == ["/tmp/rwkv-a", "/tmp/rwkv-b"]


def test_compare_rows_reports_decode_and_memory_pass() -> None:
    rows = [
        {
            "axis": AXIS,
            "status": "pass",
            "engine": "ollama",
            "runtime": "ollama_mlx",
            "model": "qwen3.5:2b-mlx",
            "prompt_case": "chars1024",
            "requested_generated_tokens": 128,
            "decode_tok_s": 30.0,
            "prefill_tok_s": 100.0,
            "ttft_s": 1.0,
            "peak_memory_bytes": 4_000_000_000,
        },
        {
            "axis": AXIS,
            "status": "pass",
            "engine": "rwkv7_hf",
            "runtime": "mlx",
            "model": "rwkv7-g1g-1.5b-hf",
            "prompt_case": "chars1024",
            "requested_generated_tokens": 128,
            "decode_tok_s": 45.0,
            "prefill_tok_s": 110.0,
            "ttft_s": 0.9,
            "mlx_peak_memory_bytes": 2_000_000_000,
        },
    ]
    comparisons = compare_rows(
        rows,
        pairs=[Pair("qwen3.5:2b-mlx", "rwkv7-g1g-1.5b-hf")],
        require_prefill=True,
        require_ttft=True,
        require_memory=True,
    )
    assert len(comparisons) == 1
    row = comparisons[0]
    assert row["axis"] == COMPARISON_AXIS
    assert row["status"] == "pass"
    assert row["decode_ratio_rwkv_over_qwen"] == 1.5
    assert row["prefill_ratio_rwkv_over_qwen"] == 1.1
    assert row["ttft_ratio_rwkv_over_qwen"] == 0.9
    assert row["memory_ratio_rwkv_over_qwen"] == 0.5
    assert summarize_comparisons(comparisons)["status"] == "pass"


def test_compare_rows_keeps_missing_memory_unknown() -> None:
    rows = [
        {
            "axis": AXIS,
            "status": "pass",
            "engine": "ollama",
            "runtime": "ollama_mlx",
            "model": "qwen3.5:0.8b-mlx",
            "prompt_case": "chars1024",
            "requested_generated_tokens": 128,
            "decode_tok_s": 40.0,
        },
        {
            "axis": AXIS,
            "status": "pass",
            "engine": "rwkv7_hf",
            "runtime": "mlx",
            "model": "rwkv7-g1d-0.4b-hf",
            "prompt_case": "chars1024",
            "requested_generated_tokens": 128,
            "decode_tok_s": 50.0,
            "mlx_peak_memory_bytes": 1_000_000_000,
        },
    ]
    comparisons = compare_rows(
        rows,
        pairs=[Pair("qwen3.5:0.8b-mlx", "rwkv7-g1d-0.4b-hf")],
        require_memory=True,
    )
    assert comparisons[0]["status"] == "unknown"
    assert comparisons[0]["memory_ratio_rwkv_over_qwen"] is None
    assert summarize_comparisons(comparisons)["status"] == "unknown"


def test_compare_rows_emit_gap_diagnostics_for_failed_gates() -> None:
    rows = [
        {
            "axis": AXIS,
            "status": "pass",
            "engine": "ollama",
            "runtime": "ollama_mlx",
            "model": "qwen3.5:2b-mlx",
            "prompt_case": "chars4096",
            "requested_generated_tokens": 256,
            "decode_tok_s": 100.0,
            "prefill_tok_s": 200.0,
            "ttft_s": 1.0,
            "peak_memory_bytes": 2_000_000_000,
        },
        {
            "axis": AXIS,
            "status": "pass",
            "engine": "rwkv7_hf",
            "runtime": "mlx",
            "model": "rwkv7-g1g-1.5b-hf",
            "prompt_case": "chars4096",
            "requested_generated_tokens": 256,
            "decode_tok_s": 70.0,
            "prefill_tok_s": 150.0,
            "ttft_s": 1.5,
            "mlx_peak_memory_bytes": 3_000_000_000,
        },
    ]
    comparisons = compare_rows(
        rows,
        pairs=[Pair("qwen3.5:2b-mlx", "rwkv7-g1g-1.5b-hf")],
        require_prefill=True,
        require_ttft=True,
        require_memory=True,
    )
    comparison = comparisons[0]
    assert comparison["status"] == "fail"
    actions = comparison_gap_actions(comparison)
    assert {action["action"] for action in actions} == {
        "optimize_decode_kernel_or_batching",
        "optimize_prefill_or_chunked_prefill",
        "reduce_ttft_load_prefill_or_first_token",
        "reduce_peak_memory_or_quantize_more",
    }
    decode_action = next(action for action in actions if action["metric"] == "decode")
    assert decode_action["current"] == 70.0
    assert decode_action["target"] == 100.0
    assert decode_action["needed_speedup_over_current"] == 1.428571
    diagnostics = gap_diagnostic_rows(comparisons)
    assert diagnostics[0]["axis"] == DIAGNOSTIC_AXIS
    assert diagnostics[0]["action_count"] == 4
    summary = summarize_comparisons(comparisons)
    assert summary["gap_action_counts"]["optimize_decode_kernel_or_batching"] == 1
    assert summary["top_gap_actions"][0]["count"] == 1


def test_compare_cli_can_append_gap_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "baseline.jsonl"
    compared = tmp_path / "compared.jsonl"
    source.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {
                    "axis": AXIS,
                    "status": "pass",
                    "model": "qwen3.5:0.8b-mlx",
                    "prompt_case": "chars64",
                    "requested_generated_tokens": 8,
                    "decode_tok_s": 30.0,
                },
                {
                    "axis": AXIS,
                    "status": "pass",
                    "model": "rwkv7-g1d-0.4b-hf",
                    "prompt_case": "chars64",
                    "requested_generated_tokens": 8,
                    "decode_tok_s": 10.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "bench/compare_qwen35_apple_baseline.py",
            "--results",
            str(source),
            "--pair",
            "qwen3.5:0.8b-mlx=rwkv7-g1d-0.4b-hf",
            "--append",
            str(compared),
            "--diagnostics",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in compared.read_text(encoding="utf-8").splitlines()]
    assert [row["axis"] for row in rows] == [COMPARISON_AXIS, DIAGNOSTIC_AXIS, SUMMARY_AXIS]
    assert rows[1]["actions"][0]["action"] == "optimize_decode_kernel_or_batching"
    assert rows[-1]["gap_action_counts"] == {"optimize_decode_kernel_or_batching": 1}


def test_compare_cli_writes_comparison_rows(tmp_path: Path) -> None:
    source = tmp_path / "baseline.jsonl"
    compared = tmp_path / "compared.jsonl"
    baseline_rows = [
        {
            "axis": AXIS,
            "status": "pass",
            "model": "qwen3.5:0.8b-mlx",
            "prompt_case": "chars64",
            "requested_generated_tokens": 8,
            "decode_tok_s": 10.0,
        },
        {
            "axis": AXIS,
            "status": "pass",
            "model": "rwkv7-g1d-0.4b-hf",
            "prompt_case": "chars64",
            "requested_generated_tokens": 8,
            "decode_tok_s": 20.0,
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in baseline_rows),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "bench/compare_qwen35_apple_baseline.py",
            "--results",
            str(source),
            "--pair",
            "qwen3.5:0.8b-mlx=rwkv7-g1d-0.4b-hf",
            "--append",
            str(compared),
            "--fail-on-gate",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output_rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert output_rows[0]["axis"] == COMPARISON_AXIS
    assert output_rows[0]["status"] == "pass"
    assert output_rows[-1]["axis"] == SUMMARY_AXIS
    assert output_rows[-1]["status"] == "pass"
    appended_rows = [json.loads(line) for line in compared.read_text(encoding="utf-8").splitlines()]
    assert [row["axis"] for row in appended_rows] == [COMPARISON_AXIS, SUMMARY_AXIS]


def test_mlx_model_reset_telemetry_counters_without_mlx_runtime() -> None:
    from rwkv7_hf.mlx_model import MLXRWKV7Model

    class DummyQLinear:
        def __init__(self) -> None:
            self.last_backend = "metal"
            self.backend_counts = {"reference": 1, "affine": 2, "metal": 3}

    model = object.__new__(MLXRWKV7Model)
    model.wkv_backend_last = "metal"
    model.wkv_backend_counts = {"reference": 4, "metal": 5}
    model.fused_ffn_key_relu2_counts = {"metal": 8, "fallback": 9}
    model.group_rkv_quant_projection_counts = {"metal": 6, "fallback": 7}
    qlinear = DummyQLinear()
    model.quantized_linears = {"x.weight": qlinear}

    model.reset_telemetry_counters()

    assert model.wkv_backend_last is None
    assert model.wkv_backend_counts == {"reference": 0, "metal": 0}
    assert model.fused_ffn_key_relu2_counts == {"metal": 0, "fallback": 0}
    assert model.group_rkv_quant_projection_counts == {"metal": 0, "fallback": 0}
    assert qlinear.last_backend is None
    assert qlinear.backend_counts == {"reference": 0, "affine": 0, "metal": 0}
