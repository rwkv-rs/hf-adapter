from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
MANIFEST = BENCH / "CURRENT_ARTIFACTS.json"
DATED_DIRECTORY = re.compile(r".+_20\d{6}$")


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_current_artifact_manifest_is_well_formed() -> None:
    document = _manifest()
    assert document["schema_version"] == 1
    artifacts = document["artifacts"]
    assert isinstance(artifacts, list) and artifacts

    ids = [item["id"] for item in artifacts]
    paths = [item["path"] for item in artifacts]
    assert len(ids) == len(set(ids))
    assert len(paths) == len(set(paths))
    assert paths == sorted(paths)

    for item in artifacts:
        assert set(item) == {"id", "platform", "line", "path"}
        assert all(isinstance(item[key], str) and item[key] for key in item)
        artifact = BENCH / item["path"]
        assert artifact.is_dir(), item["path"]
        assert (artifact / "README.md").is_file(), item["path"]


def test_every_dated_benchmark_directory_is_current() -> None:
    expected = {item["path"] for item in _manifest()["artifacts"]}
    actual = {
        path.name
        for path in BENCH.iterdir()
        if path.is_dir() and DATED_DIRECTORY.fullmatch(path.name)
    }
    assert actual == expected, {
        "unlisted": sorted(actual - expected),
        "missing": sorted(expected - actual),
    }


def test_standalone_artifacts_are_explicit_and_present() -> None:
    standalone = _manifest()["standalone_artifacts"]
    assert standalone == sorted(standalone)
    assert len(standalone) == len(set(standalone))
    for relative in standalone:
        assert isinstance(relative, str) and "/" not in relative
        assert (BENCH / relative).is_file(), relative


def test_current_native_and_qwen_entrypoints_are_retained() -> None:
    current_entrypoints = {
        "run_3090_qwen35_best_optimized_hf.sh",
        "run_3090_rwkv_paired_pd_v2.sh",
        "run_4080_qwen35_best_optimized_hf_v1.sh",
        "run_4080_qwen35_paired_pd_v1.sh",
        "run_4080_rwkv_paired_pd_v1.sh",
        "run_4090_rwkv_paired_pd_v2.sh",
        "run_5090_qwen35_best_optimized_hf.sh",
        "run_5090_rwkv_paired_decode_v1.sh",
        "run_v100_qwen35_best_optimized_hf_v1.sh",
        "run_v100_qwen35_paired_pd_v1.sh",
        "run_v100_rwkv_paired_pd_v1.sh",
    }
    for name in current_entrypoints:
        assert (BENCH / name).is_file(), name


def test_superseded_full_matrix_entrypoints_are_removed() -> None:
    obsolete_entrypoints = {
        "run_3090_qwen35_speed_matrix.sh",
        "run_4080_qwen35_pair_acceptance.sh",
        "run_5090_qwen35_correctness.sh",
        "run_5090_qwen35_full_matrix.sh",
        "run_5090_qwen35_pair_acceptance.sh",
        "run_v100_qwen35_speed_matrix.sh",
    }
    for name in obsolete_entrypoints:
        assert not (BENCH / name).exists(), name


def test_fla_compatibility_and_correctness_oracles_are_retained() -> None:
    for relative in (
        "bench/compare_rwkv_prefill_probe.py",
        "bench/qwen35_fla_triton_conv.py",
        "rwkv7_hf/modeling_rwkv7.py",
    ):
        assert (ROOT / relative).is_file(), relative
