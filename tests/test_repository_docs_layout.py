from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _line_count(relative: str) -> int:
    return len(_read(relative).splitlines())


def test_root_entry_documents_stay_navigable() -> None:
    assert _line_count("README.md") <= 300
    assert _line_count("AGENTS.md") <= 300
    assert _line_count("CONTRIBUTING.md") <= 400


def test_split_documents_exist_and_are_linked() -> None:
    split_documents = (
        "docs/architecture/REPOSITORY_LAYOUT.md",
        "docs/contributing/APPLE_VALIDATION.md",
        "docs/RESULTS_INDEX.md",
    )
    for relative in split_documents:
        assert (ROOT / relative).is_file(), relative

    readme = _read("README.md")
    agents = _read("AGENTS.md")
    contributing = _read("CONTRIBUTING.md")
    docs_index = _read("docs/README.md")

    assert "docs/architecture/REPOSITORY_LAYOUT.md" in readme
    assert "docs/RESULTS_INDEX.md" in agents
    assert "docs/contributing/APPLE_VALIDATION.md" in contributing
    for relative in split_documents:
        assert Path(relative).name in docs_index


def test_stable_remote_code_contract_is_documented() -> None:
    layout = _read("docs/architecture/REPOSITORY_LAYOUT.md")
    for required in (
        "scripts/adapter_manifest.py",
        "native_model.py",
        "auto_map",
        "compatibility shim",
    ):
        assert required in layout


def test_project_review_documents_are_linked() -> None:
    readme = _read("README.md")
    readme_zh = _read("README_ZH.md")
    docs_index = _read("docs/README.md")
    for relative in (
        "CHANGELOG.md",
        "docs/PROJECT_SUMMARY.md",
        "docs/RESULTS_INDEX.md",
    ):
        assert (ROOT / relative).is_file(), relative
        assert relative in readme, relative

    assert "docs/PROJECT_SUMMARY.md" in readme_zh
    assert "docs/RESULTS_INDEX.md" in readme_zh
    assert "PROJECT_SUMMARY.md" in docs_index
    assert "RESULTS_INDEX.md" in docs_index


def main() -> int:
    test_root_entry_documents_stay_navigable()
    test_split_documents_exist_and_are_linked()
    test_stable_remote_code_contract_is_documented()
    test_project_review_documents_are_linked()
    print("REPOSITORY DOCS LAYOUT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
