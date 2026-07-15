from __future__ import annotations

import json
import shutil
from pathlib import Path

from bench.summarize_v100_acceptance import (
    FULL_FLA_DIR,
    PRODUCTION_DIR,
    TORCH_MATRIX_DIR,
    render_markdown,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_v100_evidence_passes_current_gates() -> None:
    report = summarize(ROOT)
    assert report["validation_status"] == "pass"
    assert report["project_status"] == "partial_gpu_followups_required"
    assert report["errors"] == []

    full_fla = report["lanes"]["qwen_full_fla"]
    assert full_fla["coverage"] == {
        "expected_cells": 2,
        "joined_cells": 2,
        "complete": True,
    }
    assert full_fla["reference_backend"]["required"] == "fla"
    assert full_fla["memory"]["peak_vram_not_larger_cells"] == 1

    historical = report["lanes"]["qwen_torch_fallback"]
    assert historical["coverage"]["joined_cells"] == 216
    assert historical["reference_backend"] == {
        "required": "torch",
        "matching_cells": 216,
        "total_cells": 216,
        "complete": True,
    }
    assert historical["memory"]["model_footprint_not_larger_cells"] == 213


def test_v100_summary_fails_closed_on_backend_drift(tmp_path: Path) -> None:
    for relative in (PRODUCTION_DIR, FULL_FLA_DIR, TORCH_MATRIX_DIR):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    path = tmp_path / TORCH_MATRIX_DIR / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    reference = next(row for row in rows if row.get("model_role") == "reference")
    reference["effective_backend"] = "qwen_fla_gated_delta_rule_fla_triton_conv"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = summarize(tmp_path)
    assert report["validation_status"] == "fail"
    assert any("another backend" in error for error in report["errors"])


def test_v100_markdown_keeps_acceptance_boundaries_visible() -> None:
    text = render_markdown(summarize(ROOT))
    assert "Not an optimized-Qwen comparison" in text
    assert "Full-FLA B1 peak VRAM ratio" in text
    assert "Full-memory native MM8/MM4 remains open" in text
