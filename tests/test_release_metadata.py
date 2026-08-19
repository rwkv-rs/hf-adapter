from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility lane.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_v070_distribution_metadata_is_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["name"] == "rwkv7-hf"
    assert project["version"] == "0.7.0"
    assert project["scripts"]["rwkv7-hf-doctor"] == "rwkv7_hf.doctor:cli"

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "version: 0.7.0" in citation
    assert "date-released: 2026-08-12" in citation

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [v0.7.0]" in changelog
    assert "Published the first PyPI distribution as `rwkv7-hf`" in changelog


def test_pypi_workflow_uses_trusted_publishing() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert "types: [published]" in workflow
    assert "name: pypi" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "password:" not in workflow
