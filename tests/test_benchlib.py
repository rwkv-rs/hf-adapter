from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bench.benchlib.gpu_guard import matches_gpu_product
from bench.benchlib.paths import BENCH_ROOT, DEFAULT_RESULTS_PATH, REPO_ROOT
from bench.benchlib.results import append_jsonl, read_jsonl, write_json
from bench.benchlib.timing import median, temporary_environ


def test_benchmark_paths_are_stable() -> None:
    assert BENCH_ROOT == ROOT / "bench"
    assert REPO_ROOT == ROOT
    assert DEFAULT_RESULTS_PATH == BENCH_ROOT / "_runs" / "results.jsonl"


ROOT = Path(__file__).resolve().parents[1]


def test_json_helpers_round_trip(tmp_path: Path) -> None:
    jsonl = tmp_path / "nested" / "rows.jsonl"
    append_jsonl(jsonl, {"value": 1, "text": "中文"})
    append_jsonl(jsonl, {"value": 2})
    assert read_jsonl(jsonl) == [
        {"text": "中文", "value": 1},
        {"value": 2},
    ]

    document = tmp_path / "document.json"
    write_json(document, {"status": "pass"})
    assert json.loads(document.read_text(encoding="utf-8")) == {"status": "pass"}


def test_temporary_environment_restores_missing_and_present_values() -> None:
    os.environ["RWKV7_BENCHLIB_PRESENT"] = "before"
    os.environ.pop("RWKV7_BENCHLIB_MISSING", None)
    with temporary_environ(
        RWKV7_BENCHLIB_PRESENT="during",
        RWKV7_BENCHLIB_MISSING="created",
    ):
        assert os.environ["RWKV7_BENCHLIB_PRESENT"] == "during"
        assert os.environ["RWKV7_BENCHLIB_MISSING"] == "created"
    assert os.environ["RWKV7_BENCHLIB_PRESENT"] == "before"
    assert "RWKV7_BENCHLIB_MISSING" not in os.environ


def test_median_uses_the_statistical_even_case() -> None:
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5
    with pytest.raises(ValueError):
        median([])


def test_gpu_guard_is_fail_closed() -> None:
    assert matches_gpu_product(
        "NVIDIA GeForce RTX 4080", rtx_model="4080"
    )
    assert not matches_gpu_product(
        "NVIDIA GeForce RTX 4080 SUPER", rtx_model="4080"
    )
    assert matches_gpu_product(
        "Tesla V100-PCIE-32GB", exact_name="Tesla V100-PCIE-32GB"
    )
    with pytest.raises(ValueError):
        matches_gpu_product("anything")
