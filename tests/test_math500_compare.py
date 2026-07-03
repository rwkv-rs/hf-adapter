#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def summaries(tmp_path: Path) -> tuple[Path, Path, Path]:
    hf = {
        "num_tasks": 500,
        "rollout": 64,
        "total_generations": 32000,
        "correct_generations": 4489,
        "rollout_accuracy": 0.14028125,
        "pass_at_rollout_accuracy": 0.38,
        "token_per_sec": 10426.943,
        "sample_per_sec": 16.89,
        "elapsed_sec": 1964.5,
        "speed_timing": "generation",
        "wall_token_per_sec": 10053.618,
        "generation_token_per_sec": 10426.943,
        "decode_sec": 1704.369,
        "decoded_token_events": 19750537,
    }
    alb = {
        "num_tasks": 500,
        "rollout": 64,
        "total_generations": 32000,
        "correct_generations": 4670,
        "rollout_accuracy": 0.1459375,
        "pass_at_rollout_accuracy": 0.37,
        "token_per_sec": 3903.633,
        "sample_per_sec": 6.36,
        "elapsed_sec": 5030.2,
    }
    hf_path = tmp_path / "hf.json"
    alb_path = tmp_path / "alb.json"
    log_path = tmp_path / "alb.log"
    write_json(hf_path, hf)
    write_json(alb_path, alb)
    log_path.write_text("dynamic done B=64 rows=32000 decode_s=4945.952 tokens=19636096\n", encoding="utf-8")
    return hf_path, alb_path, log_path


def run_compare(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    hf_path, alb_path, log_path = summaries(tmp_path)
    return subprocess.run(
        [
            sys.executable,
            "bench/compare_math500_summaries.py",
            "--hf-summary",
            str(hf_path),
            "--albatross-summary",
            str(alb_path),
            "--albatross-log",
            str(log_path),
            *extra,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_math500_compare_acceptance_gates_pass(tmp_path: Path) -> None:
    json_out = tmp_path / "comparison.json"
    text_out = tmp_path / "comparison.txt"
    proc = run_compare(
        tmp_path,
        "--require-compatible-shape",
        "--min-pass-at-rollout",
        "0.370",
        "--min-summary-speed-ratio",
        "2.0",
        "--min-decode-speed-ratio",
        "2.0",
        "--json-output",
        str(json_out),
        "--text-output",
        str(text_out),
        "--fail-on-gate",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["gates"]["overall_pass"] is True
    assert data["gates"]["pass_at_rollout"]["passed"] is True
    assert data["gates"]["summary_speed_ratio"]["passed"] is True
    assert data["gates"]["decode_speed_ratio"]["passed"] is True
    assert "overall: PASS" in text_out.read_text(encoding="utf-8")


def test_math500_compare_acceptance_gates_fail(tmp_path: Path) -> None:
    proc = run_compare(
        tmp_path,
        "--require-compatible-shape",
        "--min-pass-at-rollout",
        "0.390",
        "--fail-on-gate",
    )
    assert proc.returncode == 1
    assert "pass_at_rollout: FAIL" in proc.stdout


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_math500_compare_acceptance_gates_pass(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_math500_compare_acceptance_gates_fail(Path(tmp))
    print("MATH500 COMPARE TESTS PASS")
