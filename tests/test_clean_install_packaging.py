from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CLEAN_INSTALL = ROOT / "scripts" / "run_clean_install_tests.sh"


def _requirements(extra: str) -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"][extra]


def _requirement_names(requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for requirement in requirements:
        head = requirement.split(";", 1)[0].strip()
        for separator in ("<", ">", "=", "!", "~", "["):
            head = head.split(separator, 1)[0]
        names.add(head.strip().lower())
    return names


def test_test_extra_covers_collection_dependencies() -> None:
    requirements = _requirements("test")
    names = _requirement_names(requirements)
    assert {"pytest", "numpy", "peft", "accelerate", "datasets", "trl"} <= names
    assert any(
        requirement.startswith("mlx;")
        and "platform_system == 'Darwin'" in requirement
        and "platform_machine == 'arm64'" in requirement
        for requirement in requirements
    )


def test_pep517_build_backend_is_explicit() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert any(item.startswith("setuptools") for item in data["build-system"]["requires"])


def test_clean_install_runner_contract() -> None:
    if os.name == "nt":
        indexed = subprocess.run(
            ["git", "ls-files", "--stage", "--", str(CLEAN_INSTALL.relative_to(ROOT))],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        assert indexed.startswith("100755 "), indexed
    else:
        mode = CLEAN_INSTALL.stat().st_mode
        assert mode & stat.S_IXUSR
    text = CLEAN_INSTALL.read_text(encoding="utf-8")
    assert 'PROFILE="${1:-smoke}"' in text
    assert '"${ROOT}[test]"' in text
    assert "--collect-only" in text
    assert "RWKV7_REQUIRE_APPLE" in text
    assert "RWKV7_CONSTRAINTS_FILE" in text
    assert "smoke|full|apple|compat" in text
    assert "RWKV7_TEST_MODEL" in text
    shell_path = CLEAN_INSTALL.relative_to(ROOT).as_posix() if os.name == "nt" else str(CLEAN_INSTALL)
    subprocess.run(["bash", "-n", shell_path], cwd=ROOT, check=True)


def test_ci_profiles_use_the_clean_install_runner() -> None:
    smoke = (ROOT / ".github" / "workflows" / "cpu-smoke.yml").read_text(encoding="utf-8")
    full = (ROOT / ".github" / "workflows" / "full-tests.yml").read_text(encoding="utf-8")
    assert "scripts/run_clean_install_tests.sh smoke" in smoke
    assert "scripts/run_clean_install_tests.sh full" in full
    assert "scripts/run_clean_install_tests.sh apple" in full
    assert "runs-on: macos-26" in full
    assert "RWKV7_REQUIRE_APPLE: \"1\"" in full

    compat = (ROOT / ".github" / "workflows" / "hf-ecosystem-compat.yml").read_text(
        encoding="utf-8"
    )
    assert "scripts/run_clean_install_tests.sh compat" in compat
    assert "configs/ci/hf-minimum.txt" in compat
    assert "lane: current" in compat


def test_hf_ecosystem_dependency_bounds_match_the_minimum_lane() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert "transformers>=5.12.1,<6" in data["project"]["dependencies"]
    test_requirements = data["project"]["optional-dependencies"]["test"]
    assert "peft>=0.19.1,<1" in test_requirements
    assert "trl>=1.7,<2" in test_requirements

    constraints = (ROOT / "configs" / "ci" / "hf-minimum.txt").read_text(
        encoding="utf-8"
    )
    assert "transformers==5.12.1" in constraints
    assert "peft==0.19.1" in constraints
    assert "trl==1.7.0" in constraints


def test_clean_runner_does_not_inherit_pythonpath() -> None:
    text = CLEAN_INSTALL.read_text(encoding="utf-8")
    assert 'export PYTHONPATH=""' in text
    assert "PYTHONNOUSERSITE=1" in text
    assert "clean-install import leaked into source tree" in text
