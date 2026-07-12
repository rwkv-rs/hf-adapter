from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from bench.run_mlx_quant_quality import common_prefix_length, quality_gate, token_agreement


ROOT = Path(__file__).resolve().parents[1]


def test_quant_quality_agreement_helpers():
    assert common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix_length([], []) == 0
    assert token_agreement([1, 2, 3], [1, 9, 3]) == 2 / 3
    assert token_agreement([], []) == 0.0
    assert token_agreement([1], [1, 2]) == 0.0


def test_quant_quality_gate():
    thresholds = {
        "max_nll_delta": 0.08,
        "max_perplexity_ratio": 1.09,
        "min_teacher_top1_agreement": 0.80,
    }
    status, reasons = quality_gate(
        nll_delta=0.03,
        perplexity_ratio=1.04,
        teacher_top1_agreement=0.85,
        thresholds=thresholds,
    )
    assert status == "pass"
    assert reasons == []

    status, reasons = quality_gate(
        nll_delta=0.10,
        perplexity_ratio=1.11,
        teacher_top1_agreement=0.70,
        thresholds=thresholds,
    )
    assert status == "fail"
    assert reasons == ["nll_delta", "perplexity_ratio", "teacher_top1_agreement"]


def test_quant_quality_dry_run(tmp_path: Path):
    out = tmp_path / "quality.jsonl"
    command = [
        sys.executable,
        str(ROOT / "bench" / "run_mlx_quant_quality.py"),
        "--models",
        "/tmp/rwkv-model",
        "--results",
        str(out),
        "--dry-run",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    row = json.loads(completed.stdout.strip())
    assert row["axis"] == "mlx_quant_quality_env"
    assert row["status"] == "plan"
    assert row["w4_profile"] == "q4_k_m"
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "plan"
