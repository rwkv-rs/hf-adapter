from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.mlx_serving_load_bench import latency_summary, percentile, prompts


ROOT = Path(__file__).resolve().parents[1]


def test_serving_load_helpers():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([], 0.95) is None
    assert latency_summary([1.0, 2.0, 3.0])["p95"] == 2.9
    values = prompts(64)
    assert len(values) == len(set(values)) == 64
    assert len({len(value) for value in values}) > 4


def test_serving_load_dry_run(tmp_path: Path):
    out = tmp_path / "load.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "mlx_serving_load_bench.py"),
            "--model",
            "/tmp/model",
            "--requests",
            "10000",
            "--concurrency",
            "32",
            "--results",
            str(out),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    row = json.loads(completed.stdout)
    assert row["axis"] == "mlx_serving_load_env"
    assert row["status"] == "plan"
    assert row["requests"] == 10000
    assert row["concurrency"] == 32
