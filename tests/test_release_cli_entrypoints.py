from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "evaluation/build_backend_v2_device_validation.py",
    "evaluation/preflight_cuda_toolchain.py",
    "scripts/audit_release_wheels.py",
    "scripts/build_release_provenance.py",
    "scripts/verify_release_assets.py",
)


def test_release_tools_run_directly_from_repository_root():
    for entrypoint in ENTRYPOINTS:
        completed = subprocess.run(
            [sys.executable, entrypoint, "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (entrypoint, completed.stderr)
        assert "usage:" in completed.stdout
