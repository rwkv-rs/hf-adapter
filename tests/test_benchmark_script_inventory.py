from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
MANIFEST = BENCH / "BENCH_SCRIPTS.json"
CATEGORY_KINDS = {
    "runners": "runner",
    "probes": "probe",
    "validators": "validator",
    "analyzers": "analyzer",
    "tools": "tool",
}


def _document() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _actual_scripts() -> set[str]:
    scripts = {path.relative_to(ROOT).as_posix() for path in BENCH.glob("*.sh")}
    for category in CATEGORY_KINDS:
        scripts.update(
            path.relative_to(ROOT).as_posix()
            for path in (BENCH / category).iterdir()
            if path.is_file()
            and path.suffix in {".py", ".sh"}
            and path.name != "__init__.py"
        )
    return scripts


def test_benchmark_script_manifest_is_complete() -> None:
    document = _document()
    assert document["schema_version"] == 1
    scripts = document["scripts"]
    assert isinstance(scripts, list) and scripts
    paths = [item["path"] for item in scripts]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert set(paths) == _actual_scripts()


def test_benchmark_script_manifest_has_fail_closed_categories() -> None:
    allowed_statuses = {"current", "probe", "support", "deprecated"}
    for item in _document()["scripts"]:
        assert set(item) == {
            "path",
            "kind",
            "status",
            "scope",
            "hardware",
            "replacement",
        }
        path = ROOT / item["path"]
        assert path.is_file()
        assert item["status"] in allowed_statuses
        assert isinstance(item["scope"], str) and item["scope"]
        assert isinstance(item["hardware"], list) and item["hardware"]
        if path.parent == BENCH:
            assert item["kind"] == "runner"
            assert item["status"] == "current"
        else:
            assert item["kind"] == CATEGORY_KINDS[path.parent.name]


def test_benchmark_root_contains_only_stable_shell_entrypoints() -> None:
    root_scripts = [
        path
        for path in BENCH.iterdir()
        if path.is_file()
        and path.suffix in {".py", ".sh"}
        and path.name != "__init__.py"
    ]
    assert len(root_scripts) <= 20
    assert all(path.suffix == ".sh" for path in root_scripts)
    assert not list(BENCH.glob("*.jsonl"))
    assert not list(BENCH.glob("*.log"))


def test_moved_python_scripts_support_direct_execution() -> None:
    marker = "Support direct ``python bench/<category>/<script>.py`` execution."
    for category in CATEGORY_KINDS:
        for path in (BENCH / category).glob("*.py"):
            if path.name == "__init__.py":
                continue
            assert marker in path.read_text(encoding="utf-8"), path


def test_default_result_stream_is_not_in_benchmark_root() -> None:
    forbidden = 'Path(__file__).parent / "results.jsonl"'
    for category in CATEGORY_KINDS:
        for path in (BENCH / category).glob("*.py"):
            assert forbidden not in path.read_text(encoding="utf-8"), path
