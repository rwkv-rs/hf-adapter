from __future__ import annotations

import importlib.machinery
from pathlib import Path
import re

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


def test_optional_kernels_are_outside_the_hf_model_package():
    kernel_root = ROOT / "kernels" / "rwkv7_kernels"
    assert (kernel_root / "protocol.py").is_file()
    assert (kernel_root / "dispatcher.py").is_file()
    assert (kernel_root / "recurrent" / "graph.py").is_file()
    assert (kernel_root / "recurrent" / "triton.py").is_file()

    modeling = (ROOT / "rwkv7_hf" / "modeling_rwkv7.py").read_text(
        encoding="utf-8"
    )
    config = (ROOT / "rwkv7_hf" / "configuration_rwkv7.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("rwkv7_kernels", "RWKV7_KERNEL_IMPL", "torch.cuda"):
        assert forbidden not in modeling
        assert forbidden not in config


def test_base_distribution_does_not_depend_on_kernel_wheel():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert all(
        not dependency.startswith("rwkv7-kernels")
        for dependency in project["project"]["dependencies"]
    )


def test_kernel_distribution_declares_direct_runtime_dependencies():
    project = tomllib.loads(
        (ROOT / "kernels" / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = {
        dependency.split(";", 1)[0]
        .split("[", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .strip()
        .lower()
        for dependency in project["project"]["dependencies"]
    }
    assert dependencies == {"torch", "numpy", "packaging", "ninja"}


def test_one_console_entrypoint_dispatches_all_tools():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {
        "rwkv7-hf": "rwkv7_hf_tools.cli:cli"
    }


def test_distribution_package_discovery_is_explicit():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "rwkv7_hf",
        "rwkv7_hf_tools",
    ]


def test_release_version_metadata_stays_in_lockstep():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    kernel_project = tomllib.loads(
        (ROOT / "kernels" / "pyproject.toml").read_text(encoding="utf-8")
    )
    version = project["project"]["version"]
    assert kernel_project["project"]["version"] == version
    assert project["build-system"]["requires"] == kernel_project["build-system"][
        "requires"
    ]
    assert all("==" in requirement for requirement in project["build-system"]["requires"])
    assert project["project"]["optional-dependencies"]["kernels"] == [
        f"rwkv7-kernels=={version}"
    ]

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_version = re.search(r"^version:\s*([^\s#]+)\s*$", citation, re.MULTILINE)
    assert citation_version is not None
    assert citation_version.group(1) == version

    for package in (ROOT / "rwkv7_hf", ROOT / "kernels" / "rwkv7_kernels"):
        init_source = (package / "__init__.py").read_text(encoding="utf-8")
        fallback = re.search(
            r'^\s*__version__\s*=\s*["\']([^"\']+)["\']\s*$',
            init_source,
            re.MULTILINE,
        )
        assert fallback is not None
        assert fallback.group(1) == version


def test_legacy_python_module_paths_are_removed():
    package_path = [str(ROOT / "rwkv7_hf")]
    for module in (
        "rwkv7_hf.model_cache",
        "rwkv7_hf.model_config",
        "rwkv7_hf.native_model",
        "rwkv7_hf.cli",
        "rwkv7_hf.converter",
        "rwkv7_hf.smoke",
        "rwkv7_hf.adapter_manifest",
    ):
        # Restrict discovery to this checkout. A developer environment may have
        # an unrelated legacy editable install whose meta-path finder still
        # advertises one of these historical module names.
        assert importlib.machinery.PathFinder.find_spec(module, package_path) is None
