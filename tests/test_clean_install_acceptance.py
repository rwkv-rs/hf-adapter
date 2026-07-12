from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from bench.run_clean_install_acceptance import evidence_row, parse_test_counts


ROOT = Path(__file__).resolve().parents[1]


def test_parse_clean_install_counts():
    output = """
No broken requirements found.
installed rwkv7-hf-adapter=0.5.0 from /tmp/venv/site-packages/rwkv7_hf/__init__.py
193 tests collected in 1.23s
186 passed, 6 skipped in 9.87s
SKIPPED Apple executable profile on Darwin arm64
"""
    assert parse_test_counts(output) == {
        "collected": 193,
        "passed": 186,
        "failed": 0,
        "skipped": 6,
        "errors": 0,
    }
    row = evidence_row(
        profile="full",
        command=["runner", "full"],
        returncode=0,
        output=output,
        elapsed_s=10.0,
        timed_out=False,
        log_path="bench/logs/test.log",
    )
    assert row["status"] == "pass"
    assert row["clean_wheel_import_pass"] is True
    assert row["pytest_collection_pass"] is True


def test_clean_install_evidence_dry_run(tmp_path: Path):
    results = tmp_path / "results.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "run_clean_install_acceptance.py"),
            "--profile",
            "smoke",
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
    assert row["status"] == "plan"
    assert row["profile"] == "smoke"
    assert json.loads(results.read_text(encoding="utf-8"))["axis"] == "apple_clean_install_acceptance"
