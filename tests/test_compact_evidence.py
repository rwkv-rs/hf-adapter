from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest

from evaluation.build_backend_v2_compact_bundle import build_bundle, validate_bundle


def args(source: Path, output: Path) -> Namespace:
    return Namespace(
        input_dir=source,
        output_dir=output,
        device="test-gpu",
        harness_sha="a" * 40,
        max_file_mib=1.0,
    )


def test_compact_bundle_keeps_evidence_and_excludes_raw_payloads(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "inference.json").write_text('{"status":"passed"}\n')
    (source / "manifest.jsonl").write_text('{"unit":"0.1b-b1-piqa"}\n')
    (source / "metrics.jsonl").write_text('{"loss":1.0}\n')
    logs = source / "logs"
    logs.mkdir()
    (logs / "stage.command.txt").write_text("python validate.py\n")
    (logs / "stage.exit-code.txt").write_text("0\n")
    (logs / "stage.stdout.log").write_text("large log\n")
    (source / "samples_piqa.jsonl").write_text("raw sample\n")
    (source / "results_2026.json").write_text("{}\n")
    (source / "model.safetensors").write_bytes(b"weight")
    checkpoint = source / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}\n")

    output = build_bundle(args(source, tmp_path / "bundle"))
    validate_bundle(output)
    assert (output / "inference.json").is_file()
    assert (output / "manifest.jsonl").is_file()
    assert (output / "metrics.jsonl").is_file()
    assert (output / "logs/stage.command.txt").is_file()
    assert (output / "logs/stage.exit-code.txt").is_file()
    assert not (output / "logs/stage.stdout.log").exists()
    assert not (output / "samples_piqa.jsonl").exists()
    assert not (output / "results_2026.json").exists()
    assert not (output / "model.safetensors").exists()
    assert not (output / "checkpoint-1").exists()
    metadata = json.loads((output / "BUNDLE.json").read_text())
    assert metadata["evidence_file_count"] == 5
    assert metadata["excluded_file_counts"]["raw-samples"] == 1
    assert metadata["excluded_file_counts"]["raw-lm-eval-result"] == 1


def test_compact_bundle_rejects_secrets(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "command.txt").write_text("HF_TOKEN=hf_" + "a" * 30 + "\n")
    with pytest.raises(ValueError, match="token"):
        build_bundle(args(source, tmp_path / "bundle"))


def test_compact_bundle_rejects_output_inside_input(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    with pytest.raises(ValueError, match="must not be inside"):
        build_bundle(args(source, source / "bundle"))


def test_compact_bundle_rejects_symbolic_links(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    target = tmp_path / "outside.json"
    target.write_text("{}\n")
    (source / "linked.json").symlink_to(target)
    with pytest.raises(ValueError, match="symbolic links"):
        build_bundle(args(source, tmp_path / "bundle"))
