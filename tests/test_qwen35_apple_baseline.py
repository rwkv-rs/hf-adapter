#!/usr/bin/env python3
from __future__ import annotations

import json
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
