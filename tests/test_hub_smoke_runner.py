from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_hub_release_smokes", ROOT / "scripts" / "run_hub_release_smokes.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_smoke_runner_uses_separate_hub_and_module_caches(tmp_path: Path):
    command, report = MODULE.smoke_command(
        verifier=Path("/verify.py"),
        repo="owner/model",
        revision="v1.0.0",
        device="cuda",
        output_root=tmp_path,
    )
    assert "--require-package-free" in command
    assert "--require-empty-cache" in command
    hub = Path(command[command.index("--cache-dir") + 1])
    modules = Path(command[command.index("--modules-cache-dir") + 1])
    assert hub != modules
    assert report == tmp_path / "model.json"


def test_smoke_runner_rejects_nonempty_output(tmp_path: Path):
    (tmp_path / "stale").write_text("stale")
    with pytest.raises(ValueError, match="not empty"):
        MODULE.prepare_output(tmp_path)
