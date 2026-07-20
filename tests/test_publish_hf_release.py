from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publish_hf_release", ROOT / "scripts" / "publish_hf_release.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_staged_file_hashes_are_verified(tmp_path: Path):
    target = tmp_path / "repo"
    target.mkdir()
    staged = target / "modeling_rwkv7.py"
    staged.write_text("canonical\n")
    expected = {staged.name: MODULE.sha256(staged)}
    MODULE.verify_staged_files(target, [staged.name], expected)
    staged.write_text("changed\n")
    with pytest.raises(SystemExit, match="identity differs"):
        MODULE.verify_staged_files(target, [staged.name], expected)


def test_publish_weight_inventory_preserves_lfs_identities():
    class Sibling:
        rfilename = "model.safetensors"
        lfs = {"size": 456, "sha256": "a" * 64}

    assert MODULE.weight_rows([Sibling()]) == {
        "model.safetensors": {"size": 456, "sha256": "a" * 64}
    }
