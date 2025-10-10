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
    SUMMARY_AXIS,
    Pair,
    compare_rows,
    summarize_comparisons,
)


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
            "--qwen-models",
            "qwen3.5:0.8b-mlx",
            "--rwkv-mlx-models",
            "/tmp/rwkv-a,/tmp/rwkv-b",
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
    assert rows[1]["qwen_jobs"] == 4
    assert rows[1]["rwkv_mlx_jobs"] == 8


def test_acceptance_wrapper_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "acceptance.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RUN_QWEN": "0",
            "RWKV_MLX_MODELS": "/tmp/rwkv-a,/tmp/rwkv-b",
            "PROMPT_TARGET_CHARS": "16",
            "DECODE_LENGTHS": "4",
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
    assert rows[1]["qwen_jobs"] == 0
    assert rows[1]["rwkv_mlx_jobs"] == 2
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
