from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mlx_safe_defaults_bench_dry_run(tmp_path: Path):
    results = tmp_path / "safe.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "mlx_safe_defaults_bench.py"),
            "--models",
            "/tmp/model-a,/tmp/model-b",
            "--results",
            str(results),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    row = json.loads(completed.stdout)
    assert row["axis"] == "mlx_safe_defaults_env"
    assert row["status"] == "plan"
    assert row["models"] == ["/tmp/model-a", "/tmp/model-b"]
