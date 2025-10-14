#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.profile_mlx_components import ComponentRecorder, component_name, make_prompt


def test_component_recorder_summary_orders_by_total() -> None:
    rec = ComponentRecorder()
    rec.add("ffn_step", 0.1)
    rec.add("attention_step", 0.2)
    rec.add("ffn_step", 0.3)
    summary = rec.summary(limit=2)
    assert summary["total_profiled_s"] == 0.6
    assert summary["top_components"][0]["name"] == "ffn_step"
    assert summary["components"]["ffn_step"]["count"] == 2
    assert summary["components"]["ffn_step"]["avg_ms"] == 200.0


def test_component_name_groups_layer_norm_prefixes() -> None:
    assert component_name("_layer_norm", (None, "model.norm")) == "layer_norm:final"
    assert component_name("_layer_norm", (None, "model.layers.0.attn_norm")) == "layer_norm:attn_norm"
    assert component_name("_layer_norm", (None, "model.layers.0.ffn_norm")) == "layer_norm:ffn_norm"
    assert component_name("_attn_step", ()) == "attention_step"


def test_profile_mlx_components_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "plan.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "bench/profile_mlx_components.py",
            "--model-dir",
            "/tmp/model",
            "--prompt-target-chars",
            "128",
            "--decode-length",
            "8",
            "--results",
            str(out),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["axis"] == "mlx_component_profile_plan"
    assert row["status"] == "plan"
    assert row["decode_length"] == 8


def test_make_prompt_reaches_requested_chars() -> None:
    assert make_prompt("abc", 5) == "abcab"
