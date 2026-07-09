from __future__ import annotations

import importlib.util
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]


def test_core_package_contains_only_canonical_model_modules():
    modules = {path.name for path in (ROOT / "rwkv7_hf").glob("*.py")}
    assert modules == {
        "__init__.py",
        "cache_rwkv7.py",
        "configuration_rwkv7.py",
        "modeling_rwkv7.py",
        "ops_rwkv7.py",
        "tokenization_rwkv7.py",
    }


def test_tools_are_a_sibling_package_and_core_does_not_import_them():
    modules = {path.name for path in (ROOT / "rwkv7_hf_tools").glob("*.py")}
    assert modules == {
        "__init__.py",
        "cli.py",
        "converter.py",
        "manifest.py",
        "smoke.py",
    }
    for source in (ROOT / "rwkv7_hf").glob("*.py"):
        assert "rwkv7_hf_tools" not in source.read_text(encoding="utf-8")


def test_one_console_entrypoint_dispatches_all_tools():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {
        "rwkv7-hf": "rwkv7_hf_tools.cli:cli"
    }


def test_legacy_python_module_paths_are_removed():
    for module in (
        "rwkv7_hf.model_cache",
        "rwkv7_hf.model_config",
        "rwkv7_hf.native_model",
        "rwkv7_hf.cli",
        "rwkv7_hf.converter",
        "rwkv7_hf.smoke",
        "rwkv7_hf.adapter_manifest",
    ):
        assert importlib.util.find_spec(module) is None
