#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.bench_cross_model_speed import build_exact_prompt, failure_row, model_metadata, validate_args
from bench.run_qwen35_speed_matrix import (
    MatrixConfig,
    build_run_environment,
    build_run_specs,
    existing_keys,
    parse_pair_spec,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def row(
    role: str,
    *,
    prompt: int,
    prefill: float,
    decode: float,
    status: str = "pass",
) -> dict:
    return {
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": "qwen35_v100_hf",
        "model_pair": "rwkv-1.5b__qwen3.5-2b",
        "model_role": role,
        "model_kind": "rwkv" if role == "candidate" else "qwen35",
        "status": status,
        "dtype": "fp16",
        "quantization": "none",
        "prompt_tokens": prompt,
        "decode_tokens": 128,
        "batch_size": 1,
        "prefill_tokps_total": prefill,
        "decode_tokps_total": decode,
    }


def run_compare(tmp: Path, rows: list[dict], *extra: str) -> subprocess.CompletedProcess[str]:
    results = tmp / "results.jsonl"
    write_rows(results, rows)
    return subprocess.run(
        [
            sys.executable,
            "bench/compare_qwen35_speed_matrix.py",
            "--results",
            str(results),
            "--json-output",
            str(tmp / "summary.json"),
            "--markdown-output",
            str(tmp / "summary.md"),
            *extra,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_comparator_passes_complete_matrix(tmp_path: Path) -> None:
    rows = [
        row("candidate", prompt=128, prefill=120.0, decode=220.0),
        row("reference", prompt=128, prefill=100.0, decode=200.0),
        row("candidate", prompt=512, prefill=210.0, decode=330.0),
        row("reference", prompt=512, prefill=200.0, decode=300.0),
    ]
    proc = run_compare(
        tmp_path,
        rows,
        "--expected-cells",
        "2",
        "--min-prefill-speedup",
        "1.05",
        "--min-decode-speedup",
        "1.05",
        "--fail-on-gate",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"] == {"expected_cells": 2, "joined_cells": 2, "complete": True}
    assert summary["speed"]["min_prefill_speedup"] == 1.05
    assert summary["speed"]["min_decode_speedup"] == 1.1
    assert summary["gates"]["overall_pass"] is True
    assert "Overall: PASS" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_comparator_reports_missing_and_slow_cells(tmp_path: Path) -> None:
    rows = [
        row("candidate", prompt=128, prefill=90.0, decode=180.0),
        row("reference", prompt=128, prefill=100.0, decode=200.0),
        row("candidate", prompt=512, prefill=210.0, decode=330.0),
    ]
    proc = run_compare(
        tmp_path,
        rows,
        "--expected-cells",
        "2",
        "--min-prefill-speedup",
        "1.0",
        "--min-decode-speedup",
        "1.0",
        "--fail-on-gate",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"]["joined_cells"] == 1
    assert summary["coverage"]["complete"] is False
    assert len(summary["missing"]["reference"]) == 1
    assert len(summary["red_cells"]) == 1
    assert summary["gates"]["overall_pass"] is False


class FakeTokenizer:
    def __call__(self, _text: str, **_kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long))


def worker_args(**updates) -> Namespace:
    values = {
        "model": "/models/rwkv7-g1g-1.5b-hf",
        "model_kind": "rwkv",
        "model_role": "candidate",
        "model_pair": "rwkv-1.5b__qwen3.5-2b",
        "model_size_label": "1.5b",
        "benchmark_matrix": "qwen35_test",
        "dtype": "fp16",
        "quantization": "none",
        "device": "cpu",
        "batch_size": 2,
        "prompt_tokens": 8,
        "decode_tokens": 4,
        "warmup": 1,
        "runs": 1,
        "rwkv_code_source": "repo",
        "qwen_backend": "auto",
    }
    values.update(updates)
    return Namespace(**values)


def test_worker_helpers_build_exact_shape_and_metadata() -> None:
    args = worker_args()
    validate_args(args)
    ids = build_exact_prompt(FakeTokenizer(), args.prompt_tokens, args.batch_size, "cpu")
    assert ids.tolist() == [[5, 6, 7, 5, 6, 7, 5, 6]] * 2

    config = SimpleNamespace(
        model_type="rwkv7",
        hidden_size=2048,
        num_hidden_layers=24,
        vocab_size=65536,
    )
    metadata = model_metadata(args, SimpleNamespace(config=config))
    assert metadata["model_name"] == "rwkv7-g1g-1.5b-hf"
    assert metadata["model_type"] == "rwkv7"
    assert metadata["hidden_size"] == 2048


def test_worker_helpers_validate_and_emit_failure() -> None:
    args = worker_args(prompt_tokens=0)
    try:
        validate_args(args)
    except ValueError as exc:
        assert "prompt-tokens" in str(exc)
    else:
        raise AssertionError("validate_args should reject prompt_tokens=0")

    args = worker_args()
    result = failure_row(args, RuntimeError("synthetic failure"))
    assert result["axis"] == "qwen35_cross_model_speed"
    assert result["status"] == "fail"
    assert result["model_role"] == "candidate"
    assert "synthetic failure" in result["error"]


def test_orchestrator_expands_432_raw_rows() -> None:
    pairs = [
        parse_pair_spec("rwkv-1.5b__qwen3.5-2b=/rwkv/1.5b::Qwen/Qwen3.5-2B"),
        parse_pair_spec("rwkv-2.9b__qwen3.5-4b=/rwkv/2.9b::Qwen/Qwen3.5-4B"),
        parse_pair_spec("rwkv-7.2b__qwen3.5-9b=/rwkv/7.2b::Qwen/Qwen3.5-9B"),
    ]
    config = MatrixConfig(
        pairs=pairs,
        prompts=[128, 512, 2048],
        decodes=[128, 512],
        batch_sizes=[1, 2, 4, 8],
        quantizations=["none", "bnb8", "bnb4"],
        dtype="fp16",
    )
    specs = build_run_specs(config)
    assert len(specs) == 432
    assert len({spec.cell_key for spec in specs}) == 216
    assert {spec.model_role for spec in specs} == {"candidate", "reference"}
    assert specs[0].model_kind == "rwkv"
    assert specs[1].model_kind == "qwen35"

    candidate_specs = [spec for spec in specs if spec.model_role == "candidate"]
    assert len(candidate_specs) == 216
    assert len({spec.cell_key for spec in candidate_specs}) == 216


def test_orchestrator_existing_keys_are_resumable(tmp_path: Path) -> None:
    result_path = tmp_path / "results.jsonl"
    rows = [
        row("candidate", prompt=128, prefill=120.0, decode=220.0),
        row("reference", prompt=128, prefill=100.0, decode=200.0, status="fail"),
    ]
    write_rows(result_path, rows)
    keys = existing_keys(result_path)
    assert len(keys) == 2
    assert any(key[-1] == "candidate" for key in keys)
    assert any(key[-1] == "reference" for key in keys)


def test_orchestrator_forces_production_rwkv_wrapper() -> None:
    args = Namespace(rwkv_fast_token_backend="native_graph")
    env = build_run_environment(args, {"RWKV7_NATIVE_MODEL": "1", "PYTHONPATH": "/existing"})
    assert env["RWKV7_NATIVE_MODEL"] == "0"
    assert env["RWKV7_FAST_TOKEN_BACKEND"] == "native_graph"
    assert env["PYTHONPATH"].endswith(f"{os.pathsep}/existing")


def test_3090_entrypoint_requires_optimized_qwen_path() -> None:
    for name in ("run_3090_qwen35_pair.sh", "run_3090_qwen35_speed_matrix.sh"):
        script = (ROOT / "bench" / name).read_text(encoding="utf-8")
        assert "--require-qwen-fast-path" in script


def test_hardware_entrypoints_are_fail_closed() -> None:
    for name in ("run_v100_qwen35_speed_matrix.sh", "run_3090_qwen35_speed_matrix.sh"):
        script = (ROOT / "bench" / name).read_text(encoding="utf-8")
        assert "--expected-cells 216" in script
        assert "--min-prefill-speedup 1.05" in script
        assert "--min-decode-speedup 1.05" in script
        assert "--fail-on-gate" in script

    pair_script = (ROOT / "bench" / "run_3090_qwen35_pair.sh").read_text(encoding="utf-8")
    assert "--expected-cells 72" in pair_script
    assert "--min-prefill-speedup 1.05" in pair_script
    assert "--min-decode-speedup 1.05" in pair_script
    assert "--fail-on-gate" in pair_script
    assert 'QWEN_BACKEND="${QWEN_BACKEND:-auto}"' in pair_script
    assert '--qwen-backend "${QWEN_BACKEND}"' in pair_script
    assert '--model-roles "${MODEL_ROLE_ARGS[@]}"' in pair_script
    assert 'COMPARE_AFTER="${COMPARE_AFTER:-1}"' in pair_script


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        test_comparator_passes_complete_matrix(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_comparator_reports_missing_and_slow_cells(Path(td))
    test_worker_helpers_build_exact_shape_and_metadata()
    test_worker_helpers_validate_and_emit_failure()
    test_orchestrator_expands_432_raw_rows()
    with tempfile.TemporaryDirectory() as td:
        test_orchestrator_existing_keys_are_resumable(Path(td))
    test_orchestrator_forces_production_rwkv_wrapper()
    test_3090_entrypoint_requires_optimized_qwen_path()
    test_hardware_entrypoints_are_fail_closed()
    print("QWEN35 SPEED MATRIX TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
