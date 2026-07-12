from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_hf_release", ROOT / "scripts" / "verify_hf_release.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_hub_smoke_cache_must_start_empty(tmp_path: Path):
    cache = tmp_path / "hub-cache"
    cache.mkdir()
    (cache / "stale").write_text("cached")
    with pytest.raises(ValueError, match="cache is not empty"):
        MODULE.prepare_cache_dir(cache, True)


def test_hub_smoke_cache_records_fresh_directory(tmp_path: Path):
    cache = tmp_path / "hub-cache"
    assert MODULE.prepare_cache_dir(cache, True) == {
        "path": str(cache.resolve()),
        "was_empty": True,
    }
    assert cache.is_dir()


def test_package_free_inventory_has_both_distributions(monkeypatch):
    def version(name):
        if name == "rwkv7-hf":
            return "1.0.0"
        raise MODULE.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(MODULE.importlib.metadata, "version", version)
    assert MODULE.installed_rwkv_distributions() == {
        "rwkv7-hf": "1.0.0",
        "rwkv7-kernels": None,
    }


def test_package_free_inventory_detects_source_checkout(monkeypatch):
    class Spec:
        origin = "/checkout/rwkv7_hf/__init__.py"

    monkeypatch.setattr(
        MODULE.importlib.util,
        "find_spec",
        lambda name: Spec() if name == "rwkv7_hf" else None,
    )
    assert MODULE.local_rwkv_import_origins() == {
        "rwkv7_hf": "/checkout/rwkv7_hf/__init__.py",
        "rwkv7_kernels": None,
    }
