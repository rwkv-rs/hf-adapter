from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dynamic_serving_runtime_is_shipped_with_remote_adapter() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "mlx_cache.py" in ADAPTER_FILES
    assert "mlx_scheduler.py" in ADAPTER_FILES


def test_dynamic_serving_bench_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "dynamic.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/mlx_dynamic_serving_bench.py",
            "--models",
            "/tmp/a,/tmp/b",
            "--results",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["axis"] == "mlx_dynamic_serving_env"
    assert row["status"] == "plan"
    assert row["models"] == ["/tmp/a", "/tmp/b"]
    assert row["max_batch_size"] == 4
