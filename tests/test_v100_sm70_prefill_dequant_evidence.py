#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = (
    ROOT
    / "bench"
    / "v100_sm70_prefill_dequant_20260723"
    / "check_results.py"
)


def test_v100_sm70_prefill_dequant_evidence_passes_fail_closed_gate() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4/4 group256 speed cells" in result.stdout
