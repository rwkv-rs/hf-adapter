#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.bench_cross_model_speed import (  # noqa: E402
    build_exact_prompt,
    effective_quantization_metadata,
    enforce_qwen_backend,
    failure_row,
    forward_prefill,
    last_rwkv_prefill_backend,
    model_metadata,
    model_parameter_metadata,
    prepare_rwkv_model_dir,
    qwen35_fast_path_bindings,
    qwen_effective_backend,
    qwen_fla_operator_contract,
    validate_loaded_model,
    validate_args,
)
from bench.qwen35_fla_triton_conv import (  # noqa: E402
    bind_qwen35_fla_triton_conv,
    qwen35_fla_triton_causal_conv1d,
    qwen35_fla_triton_causal_conv1d_update,
)
from bench.bench_cross_model_speed_resident import cell_args, resolve_sweep_cells, resolve_sweep_shapes
from bench.compare_qwen35_speed_matrix import quantization_family
from bench.compare_qwen35_backend_probe import compare as compare_backend_probe  # noqa: E402
from bench.compare_rwkv_prefill_probe import compare as compare_rwkv_prefill_probe  # noqa: E402
from bench.run_qwen35_speed_matrix import (  # noqa: E402
    MatrixConfig,
    RunSpec,
    append_orchestrator_failure,
    build_run_environment,
    build_run_specs,
    build_worker_environment,
    existing_keys,
    parse_pair_spec,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_resident_exact_cells_avoid_cartesian_reruns() -> None:
    args = Namespace(
        cells=["8x512x512", "2x2048x512", "8x512x512"],
        shapes=None,
        batch_sizes=[1],
        prompt_tokens=[128],
        decode_tokens=[128, 512],
    )
    assert resolve_sweep_cells(args) == [(8, 512, 512), (2, 2048, 512)]


def test_resident_exact_cells_reject_shapes_mix() -> None:
    args = Namespace(cells=["1x128x128"], shapes=["1x128"])
    try:
        resolve_sweep_cells(args)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected --cells/--shapes conflict")


def test_effective_bnb_metadata_reports_loaded_policy(monkeypatch) -> None:
    for name in (
        "RWKV7_NATIVE_BNB8_DIRECT",
        "RWKV7_NATIVE_BNB8_RELU_QUANT",
        "RWKV7_NATIVE_BNB8_RKV_MIX_QUANT",
        "RWKV7_NATIVE_BNB8_FFN_MIX_QUANT",
    ):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK", "1024")
    monkeypatch.setenv("RWKV7_NATIVE_BNB8_FFN_MIX_BLOCK", "1024")
    model = SimpleNamespace(
        hf_quantizer=SimpleNamespace(
            quantization_config=SimpleNamespace(llm_int8_threshold=0.0)
        ),
        _rwkv7_bnb_skip_policy="memory",
    )
    metadata = effective_quantization_metadata(
        model,
        Namespace(quantization="bnb8", model_kind="rwkv"),
    )
    assert metadata == {
        "bnb_int8_threshold": 0.0,
        "rwkv_bnb_skip_policy": "memory",
        "rwkv_bnb_prefill_value_stride": 8,
        "rwkv_native_bnb8_direct_effective": False,
        "rwkv_native_bnb8_relu_quant_effective": False,
        "rwkv_native_bnb8_rkv_mix_quant_effective": False,
        "rwkv_native_bnb8_ffn_mix_quant_effective": False,
        "rwkv_native_bnb8_attn_mix_block_effective": 1024,
        "rwkv_native_bnb8_ffn_mix_block_effective": 1024,
        "quantization_backend": "bitsandbytes",
        "quantized_modules": None,
        "native_quant_block_modules": None,
        "a8w8_gemv_max_rows": None,
        "a8w8_gemv_block_k": None,
        "a8w8_gemv_block_n": None,
        "a8w8_gemv_warps": None,
        "mm4_fused_max_rows": None,
        "mm4_gemv_block_pairs": None,
        "mm4_gemv_block_n": None,
        "mm4_dot_min_rows": None,
        "mm4_dot_block_b": None,
        "mm4_dot_block_pairs": None,
        "mm4_dot_block_n": None,
        "mm4_dot_warps": None,
        "native_quant_kernel_active": False,
    }


def test_effective_native_quant_metadata_reports_hybrid_backend(monkeypatch) -> None:
    # Keep this metadata unit test independent of the host GPU policy. Exact
    # card defaults are covered by test_kernel_policy.py and hardware rows.
    monkeypatch.setenv("RWKV7_A8W8_GEMV_MAX_ROWS", "1")
    model = SimpleNamespace(_rwkv7_cross_model_quant_replaced_modules=1)
    metadata = effective_quantization_metadata(
        model,
        Namespace(quantization="a8w8", model_kind="rwkv"),
    )
    assert metadata == {
        "bnb_int8_threshold": None,
        "rwkv_bnb_skip_policy": None,
        "rwkv_bnb_prefill_value_stride": None,
        "rwkv_native_bnb8_direct_effective": None,
        "rwkv_native_bnb8_relu_quant_effective": None,
        "rwkv_native_bnb8_rkv_mix_quant_effective": None,
        "rwkv_native_bnb8_ffn_mix_quant_effective": None,
        "rwkv_native_bnb8_attn_mix_block_effective": None,
        "rwkv_native_bnb8_ffn_mix_block_effective": None,
        "quantization_backend": "rwkv_native",
        "quantized_modules": 1,
        "native_quant_block_modules": None,
        "a8w8_gemv_max_rows": 1,
        "a8w8_gemv_block_k": 256,
        "a8w8_gemv_block_n": 64,
        "a8w8_gemv_warps": 1,
        "mm4_fused_max_rows": None,
        "mm4_gemv_block_pairs": None,
        "mm4_gemv_block_n": None,
        "mm4_dot_min_rows": None,
        "mm4_dot_block_b": None,
        "mm4_dot_block_pairs": None,
        "mm4_dot_block_n": None,
        "mm4_dot_warps": None,
        "native_quant_kernel_active": True,
    }


def test_effective_bnb_metadata_resolves_hardware_policy_defaults(monkeypatch) -> None:
    flags = {
        "native_bnb8_direct": True,
        "native_bnb8_relu_quant": True,
        "native_bnb8_rkv_mix_quant": True,
        "native_bnb8_ffn_mix_quant": True,
    }
    blocks = {
        "native_bnb8_attn_mix_block": 4096,
        "native_bnb8_ffn_mix_block": 2048,
    }
    fake_native_jit = SimpleNamespace(
        _native_bnb8_policy_flag=lambda _env, policy: flags[policy],
        _native_bnb8_policy_block=lambda _env, policy, _fallback: blocks[policy],
    )
    imported_prefill = lambda: None
    imported_prefill.__module__ = "test_dynamic_native_jit"
    monkeypatch.setitem(sys.modules, imported_prefill.__module__, fake_native_jit)
    fake_prefill = FunctionType(
        (lambda: None).__code__, {"_native_jit_prefill": imported_prefill}
    )
    model = SimpleNamespace(
        rwkv7_prefill_native=fake_prefill,
        hf_quantizer=SimpleNamespace(
            quantization_config=SimpleNamespace(llm_int8_threshold=0.0)
        ),
        _rwkv7_bnb_skip_policy="memory",
    )

    metadata = effective_quantization_metadata(
        model,
        Namespace(quantization="bnb8", model_kind="rwkv"),
    )

    assert metadata["rwkv_native_bnb8_direct_effective"] is True
    assert metadata["rwkv_native_bnb8_relu_quant_effective"] is True
    assert metadata["rwkv_native_bnb8_rkv_mix_quant_effective"] is True
    assert metadata["rwkv_native_bnb8_ffn_mix_quant_effective"] is True
    assert metadata["rwkv_native_bnb8_attn_mix_block_effective"] == 4096
    assert metadata["rwkv_native_bnb8_ffn_mix_block_effective"] == 2048


def test_effective_mm4_metadata_is_safe_without_model_parameters(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_MM4_GEMV_BLOCK_PAIRS", "32")
    monkeypatch.setenv("RWKV7_MM4_GEMV_BLOCK_N", "16")
    monkeypatch.setenv("RWKV7_MM4_DOT_MIN_ROWS", "3")
    metadata = effective_quantization_metadata(
        SimpleNamespace(_rwkv7_cross_model_quant_replaced_modules=1),
        Namespace(quantization="mm4", model_kind="rwkv"),
    )
    assert metadata["quantization_backend"] == "rwkv_native"
    assert metadata["quantized_modules"] == 1
    assert metadata["mm4_gemv_block_pairs"] == 32
    assert metadata["mm4_gemv_block_n"] == 16
    assert metadata["mm4_dot_min_rows"] == 3
    assert metadata["native_quant_kernel_active"] is True


def test_hybrid_bnb8_a8w8_head_metadata_and_family(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_A8W8_GEMV_MAX_ROWS", "8")
    model = SimpleNamespace(
        hf_quantizer=SimpleNamespace(
            quantization_config=SimpleNamespace(llm_int8_threshold=0.0)
        ),
        _rwkv7_bnb_skip_policy="memory",
        _rwkv7_cross_model_quant_replaced_modules=1,
    )
    metadata = effective_quantization_metadata(
        model,
        Namespace(quantization="bnb8_a8w8_head", model_kind="rwkv"),
    )
    assert quantization_family("bnb8_a8w8_head") == "w8"
    assert metadata["quantization_backend"] == "bitsandbytes+rwkv_native"
    assert metadata["bnb_int8_threshold"] == 0.0
    assert metadata["a8w8_gemv_max_rows"] == 8
    assert metadata["quantized_modules"] == 1
    assert metadata["native_quant_kernel_active"] is True


def row(
    role: str,
    *,
    prompt: int,
    prefill: float,
    decode: float,
    status: str = "pass",
    quantization: str = "none",
    footprint: float | None = None,
    qwen_backend: str = "fla",
) -> dict:
    candidate = role == "candidate"
    result = {
        "axis": "qwen35_cross_model_speed",
        "model_pair": "rwkv-1.5b__qwen3.5-2b",
        "benchmark_matrix": "qwen35_test_hf",
        "model_role": role,
        "model_kind": "rwkv" if role == "candidate" else "qwen35",
        "status": status,
        "dtype": "fp16",
        "quantization": quantization,
        "prompt_tokens": prompt,
        "decode_tokens": 128,
        "batch_size": 1,
        "prefill_tokps_total": prefill,
        "decode_tokps_total": decode,
        "prefill_sec_median": prompt / prefill,
        "decode_sec_median": 128 / decode,
        "prefill_effective_backend": "native_prefill" if candidate else "module_call",
        "effective_backend": "native_graph" if candidate else "fla+causal_conv1d",
        "qwen_fast_path_verified": None if candidate else True,
        "model_footprint_mb": footprint if footprint is not None else (100.0 if candidate else 120.0),
        "peak_vram_mb": footprint if footprint is not None else (100.0 if candidate else 120.0),
    }
    if role == "reference":
        result.update(
            {
                "qwen_backend_requested": qwen_backend,
                "qwen_operator_contract_pass": qwen_backend == "fla",
                "qwen_force_torch": qwen_backend == "torch",
                "effective_backend": (
                    "qwen_fla_gated_delta_rule"
                    if qwen_backend == "fla"
                    else "transformers_torch_fallback"
                ),
            }
        )
    else:
        result.update({"qwen_backend_requested": qwen_backend, "effective_backend": "native_graph"})
    return result


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


def test_resident_worker_direct_entrypoint_imports_sibling_worker() -> None:
    proc = subprocess.run(
        [sys.executable, "bench/bench_cross_model_speed_resident.py", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Single-load RWKV/Qwen speed sweep" in proc.stdout
    assert "--shapes" in proc.stdout
    assert "--probe-output" in proc.stdout
    assert "{auto,fla,torch}" in proc.stdout


def test_resident_worker_forwards_probe_defaults_to_shared_worker() -> None:
    args = Namespace(
        model="/models/rwkv",
        model_kind="rwkv",
        model_role="candidate",
        model_pair="rwkv-1.5b__qwen3.5-2b",
        model_size_label="1.5b",
        benchmark_matrix="qwen35_test_hf",
        dtype="fp16",
        quantization="none",
        native_quant_min_params=1_000_000,
        native_quant_policy="memory",
        torchao_group_size=128,
        device="cuda",
        prefill_chunk_size=0,
        warmup=1,
        runs=3,
        rwkv_attn_mode="fused_recurrent",
        rwkv_code_source="repo",
        qwen_backend="auto",
        qwen_conv_backend="fla_triton",
        require_qwen_fast_path=False,
        probe_output="",
        probe_tokens=8,
        results="results.jsonl",
    )
    forwarded = cell_args(args, 8, 128, 128)
    assert forwarded.probe_output == ""
    assert forwarded.probe_tokens == 8
    assert forwarded.qwen_conv_backend == "fla_triton"
    validate_args(forwarded)


def test_resident_worker_accepts_exact_non_cartesian_shapes() -> None:
    args = Namespace(
        shapes=["2x512", "8X128", "2x512"],
        batch_sizes=[1, 2],
        prompt_tokens=[128, 512],
    )
    assert resolve_sweep_shapes(args) == [(2, 512), (8, 128)]


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


def test_comparator_strict_backend_and_quant_memory_gates(tmp_path: Path) -> None:
    rows = [
        row("candidate", prompt=128, prefill=120.0, decode=220.0, footprint=200.0),
        row("reference", prompt=128, prefill=100.0, decode=200.0),
        row(
            "candidate",
            prompt=128,
            prefill=130.0,
            decode=230.0,
            quantization="bnb8",
            footprint=100.0,
        ),
        row(
            "reference",
            prompt=128,
            prefill=100.0,
            decode=200.0,
            quantization="bnb8",
        ),
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
        "--min-quant-prefill-speedup",
        "1.0",
        "--min-quant-decode-speedup",
        "1.0",
        "--require-native-candidate",
        "--require-qwen-fast-path",
        "--require-quant-memory-reduction",
        "--require-quant-not-slower-than-dense",
        "--fail-on-gate",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["gates"]["backend_pass"] is True
    assert summary["gates"]["quant_memory_pass"] is True
    quant = next(cell for cell in summary["cells"] if cell["quantization"] == "w8")
    assert quant["quant_memory_ratio_vs_dense"] == 0.5
    assert quant["quant_peak_memory_ratio_vs_dense"] == 0.5
    assert abs(quant["quant_prefill_speedup_vs_dense"] - 130.0 / 120.0) < 1e-6
    assert abs(quant["quant_decode_speedup_vs_dense"] - 230.0 / 220.0) < 1e-6
    assert quant["quant_dense_speed_pass"] is True
    assert quant["quant_dense_prefill_mode_pass"] is True
    family = summary["speed_by_quantization"]["w8"]
    assert family["cells"] == 1
    assert family["min_prefill_speedup"] == 1.3
    assert family["min_decode_speedup"] == 1.15
    assert abs(family["min_prefill_speedup_vs_dense"] - 130.0 / 120.0) < 1e-6
    assert abs(family["min_decode_speedup_vs_dense"] - 230.0 / 220.0) < 1e-6
    assert family["max_footprint_ratio_vs_dense"] == 0.5


def test_comparator_joins_different_w8_implementations(tmp_path: Path) -> None:
    candidate = row(
        "candidate",
        prompt=128,
        prefill=130.0,
        decode=230.0,
        quantization="torchao_w8",
    )
    candidate["quantization_backend"] = "torchao"
    reference = row(
        "reference",
        prompt=128,
        prefill=100.0,
        decode=200.0,
        quantization="bnb8",
    )
    reference["quantization_backend"] = "bitsandbytes"
    proc = run_compare(
        tmp_path,
        [candidate, reference],
        "--expected-cells",
        "1",
        "--fail-on-gate",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"]["joined_cells"] == 1
    assert summary["cells"][0]["quantization"] == "w8"
    assert summary["cells"][0]["candidate_quantization_backend"] == "torchao"
    assert summary["cells"][0]["reference_quantization_backend"] == "bitsandbytes"


def test_comparator_fails_mismatched_prefill_chunking(tmp_path: Path) -> None:
    candidate = row("candidate", prompt=128, prefill=120.0, decode=220.0)
    reference = row("reference", prompt=128, prefill=100.0, decode=200.0)
    candidate["prefill_chunk_size"] = 512
    reference["prefill_chunk_size"] = 0
    proc = run_compare(
        tmp_path,
        [candidate, reference],
        "--expected-cells",
        "1",
        "--require-prefill-mode-match",
        "--fail-on-gate",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["gates"]["prefill_mode_pass"] is False
    assert summary["red_cells"][0]["prefill_mode_pass"] is False


def test_comparator_fails_quant_vs_dense_chunk_mismatch(tmp_path: Path) -> None:
    dense_candidate = row("candidate", prompt=128, prefill=120.0, decode=220.0, footprint=200.0)
    dense_reference = row("reference", prompt=128, prefill=100.0, decode=200.0)
    quant_candidate = row(
        "candidate",
        prompt=128,
        prefill=130.0,
        decode=230.0,
        quantization="bnb8",
        footprint=100.0,
    )
    quant_reference = row(
        "reference",
        prompt=128,
        prefill=100.0,
        decode=200.0,
        quantization="bnb8",
    )
    dense_candidate["prefill_chunk_size"] = 512
    dense_reference["prefill_chunk_size"] = 512
    quant_candidate["prefill_chunk_size"] = 0
    quant_reference["prefill_chunk_size"] = 0
    proc = run_compare(
        tmp_path,
        [dense_candidate, dense_reference, quant_candidate, quant_reference],
        "--expected-cells",
        "2",
        "--require-prefill-mode-match",
        "--require-quant-not-slower-than-dense",
        "--fail-on-gate",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    quant = next(cell for cell in summary["red_cells"] if cell["quantization"] == "w8")
    assert quant["prefill_mode_pass"] is True
    assert quant["quant_dense_prefill_mode_pass"] is False
    assert quant["quant_dense_speed_pass"] is False


def test_comparator_can_gate_quant_on_exact_cell_total_latency(tmp_path: Path) -> None:
    dense = row("candidate", prompt=128, prefill=100.0, decode=100.0, footprint=200.0)
    reference = row("reference", prompt=128, prefill=80.0, decode=80.0)
    quant = row(
        "candidate",
        prompt=128,
        prefill=98.0,
        decode=104.0,
        quantization="mm4",
        footprint=100.0,
    )
    quant_reference = row(
        "reference", prompt=128, prefill=80.0, decode=80.0, quantization="bnb4"
    )
    proc = run_compare(
        tmp_path,
        [dense, reference, quant, quant_reference],
        "--expected-cells",
        "2",
        "--min-prefill-speedup",
        "1.0",
        "--min-decode-speedup",
        "1.0",
        "--min-quant-prefill-speedup",
        "0.0",
        "--min-quant-decode-speedup",
        "0.0",
        "--require-quant-memory-reduction",
        "--require-quant-not-slower-than-dense",
        "--allow-quant-total-not-slower-than-dense",
        "--fail-on-gate",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    cell = next(item for item in summary["cells"] if item["quantization"] == "w4")
    assert cell["quant_prefill_speedup_vs_dense"] == 0.98
    assert cell["quant_total_speedup_vs_dense"] > 1.0
    assert cell["quant_dense_speed_pass"] is True
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Quant/fp16 total min" in markdown


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


def test_comparator_supports_strict_nonnegative_quant_gate(tmp_path: Path) -> None:
    rows = [
        row("candidate", prompt=128, prefill=101.0, decode=202.0, quantization="bnb4"),
        row("reference", prompt=128, prefill=100.0, decode=200.0, quantization="bnb4"),
    ]
    proc = run_compare(
        tmp_path,
        rows,
        "--expected-cells",
        "1",
        "--min-prefill-speedup",
        "1.05",
        "--min-decode-speedup",
        "1.05",
        "--min-quant-prefill-speedup",
        "1.0",
        "--min-quant-decode-speedup",
        "1.0",
        "--fail-on-gate",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_red_candidate_rerunner_builds_append_only_command(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    write_rows(
        results,
        [
            {
                **row("candidate", prompt=128, prefill=90.0, decode=180.0),
                "model_id_or_path": "/models/rwkv",
                "model_size_label": "1.5b",
                "qwen_backend_requested": "auto",
            },
            row("reference", prompt=128, prefill=100.0, decode=200.0),
        ],
    )
    proc = subprocess.run(
        [
            sys.executable,
            "bench/rerun_qwen35_red_candidates.py",
            "--results",
            str(results),
            "--expected-cells",
            "1",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 1
    assert "[1/1]" in proc.stdout
    assert "--model /models/rwkv" in proc.stdout


def test_red_candidate_rerunner_resolves_normalized_quant_family(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    candidate = {
        **row("candidate", prompt=128, prefill=90.0, decode=220.0, quantization="torchao_w8"),
        "model_id_or_path": "/models/rwkv",
        "model_size_label": "1.5b",
        "qwen_backend_requested": "auto",
        "native_quant_policy_requested": "speed",
        "native_quant_min_params_requested": 8_000_000,
        "torchao_group_size_requested": 128,
    }
    reference = row(
        "reference",
        prompt=128,
        prefill=100.0,
        decode=200.0,
        quantization="bnb8",
    )
    write_rows(results, [candidate, reference])
    proc = subprocess.run(
        [
            sys.executable,
            "bench/rerun_qwen35_red_candidates.py",
            "--results",
            str(results),
            "--expected-cells",
            "1",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 1
    assert "--quantization torchao_w8" in proc.stdout
    assert "--native-quant-policy speed" in proc.stdout
    assert "--native-quant-min-params 8000000" in proc.stdout


def test_comparator_can_gate_memory_per_cell(tmp_path: Path) -> None:
    candidate = row("candidate", prompt=128, prefill=120.0, decode=220.0)
    reference = row("reference", prompt=128, prefill=100.0, decode=200.0)
    candidate.update({"model_footprint_mb": 90.0, "peak_vram_mb": 110.0})
    reference.update({"model_footprint_mb": 100.0, "peak_vram_mb": 100.0})
    proc = run_compare(
        tmp_path,
        [candidate, reference],
        "--expected-cells",
        "1",
        "--require-memory-not-larger",
        "--fail-on-gate",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["gates"]["memory_pass"] is False
    assert summary["red_cells"][0]["memory_pass"] is False


def test_comparator_requires_full_fla_conv_and_active_parameter_work(tmp_path: Path) -> None:
    candidate = row("candidate", prompt=128, prefill=130.0, decode=130.0, footprint=90.0)
    reference = row("reference", prompt=128, prefill=100.0, decode=100.0, footprint=100.0)
    candidate.update(
        {
            "runtime_working_set_mb": 20.0,
            "active_parameter_count": 80,
            "prefill_active_parameter_tops": 10.4,
            "decode_active_parameter_tops": 10.4,
        }
    )
    reference.update(
        {
            "runtime_working_set_mb": 25.0,
            "active_parameter_count": 100,
            "prefill_active_parameter_tops": 10.0,
            "decode_active_parameter_tops": 10.0,
            "qwen_full_fused_contract_pass": False,
            "qwen_conv_backend_effective": "fallback",
            "effective_backend": "qwen_fla_gated_delta_rule_torch_conv",
        }
    )
    failed = run_compare(
        tmp_path,
        [candidate, reference],
        "--expected-cells",
        "1",
        "--require-qwen-full-fused",
        "--min-active-parameter-throughput-ratio",
        "1.0",
        "--fail-on-gate",
    )
    assert failed.returncode == 1, failed.stdout + failed.stderr
    failed_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert failed_summary["gates"]["backend_pass"] is False
    assert failed_summary["cells"][0]["qwen_full_fused_pass"] is False

    reference.update(
        {
            "qwen_full_fused_contract_pass": True,
            "qwen_conv_backend_effective": "fla_triton",
            "effective_backend": "qwen_fla_gated_delta_rule_fla_triton_conv",
        }
    )
    passed = run_compare(
        tmp_path,
        [candidate, reference],
        "--expected-cells",
        "1",
        "--require-qwen-full-fused",
        "--min-active-parameter-throughput-ratio",
        "1.0",
        "--fail-on-gate",
    )
    assert passed.returncode == 0, passed.stdout + passed.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["gates"]["backend_pass"] is True
    assert summary["gates"]["active_parameter_pass"] is True
    assert summary["active_parameter_work"]["min_prefill_throughput_ratio"] == 1.04
    assert summary["active_parameter_work"]["min_decode_throughput_ratio"] == 1.04


def test_comparator_active_parameter_work_does_not_reward_smaller_model(tmp_path: Path) -> None:
    candidate = row("candidate", prompt=128, prefill=120.0, decode=120.0)
    reference = row("reference", prompt=128, prefill=100.0, decode=100.0)
    candidate.update(
        {
            "active_parameter_count": 80,
            "prefill_active_parameter_tops": 9.6,
            "decode_active_parameter_tops": 9.6,
        }
    )
    reference.update(
        {
            "active_parameter_count": 100,
            "prefill_active_parameter_tops": 10.0,
            "decode_active_parameter_tops": 10.0,
        }
    )
    proc = run_compare(
        tmp_path,
        [candidate, reference],
        "--expected-cells",
        "1",
        "--min-active-parameter-throughput-ratio",
        "1.0",
        "--fail-on-gate",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["cells"][0]["prefill_speedup"] == 1.2
    assert summary["cells"][0]["active_parameter_ratio"] == 0.8
    assert summary["cells"][0]["active_parameter_pass"] is False
    assert summary["gates"]["active_parameter_pass"] is False


def test_comparator_can_gate_active_parameter_work_on_decode_only(tmp_path: Path) -> None:
    candidate = row("candidate", prompt=128, prefill=120.0, decode=130.0)
    reference = row("reference", prompt=128, prefill=100.0, decode=100.0)
    candidate.update(
        {
            "active_parameter_count": 80,
            "prefill_active_parameter_tops": 9.6,
            "decode_active_parameter_tops": 10.4,
        }
    )
    reference.update(
        {
            "active_parameter_count": 100,
            "prefill_active_parameter_tops": 10.0,
            "decode_active_parameter_tops": 10.0,
        }
    )
    proc = run_compare(
        tmp_path,
        [candidate, reference],
        "--expected-cells",
        "1",
        "--min-decode-active-parameter-throughput-ratio",
        "1.0",
        "--fail-on-gate",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    cell = summary["cells"][0]
    assert cell["prefill_active_parameter_throughput_ratio"] == 0.96
    assert cell["prefill_active_parameter_work_pass"] is True
    assert cell["decode_active_parameter_throughput_ratio"] == 1.04
    assert cell["decode_active_parameter_work_pass"] is True
    assert summary["active_parameter_work"]["prefill_gate"] is None
    assert summary["active_parameter_work"]["decode_gate"] == 1.0
    assert summary["gates"]["active_parameter_work_pass"] is True


def test_comparator_active_parameter_efficiency_normalizes_smaller_model(tmp_path: Path) -> None:
    candidate = row("candidate", prompt=128, prefill=120.0, decode=120.0)
    reference = row("reference", prompt=128, prefill=100.0, decode=100.0)
    candidate.update({"active_parameter_count": 80})
    reference.update({"active_parameter_count": 100})
    proc = run_compare(
        tmp_path,
        [candidate, reference],
        "--expected-cells",
        "1",
        "--min-active-parameter-efficiency-ratio",
        "1.0",
        "--fail-on-gate",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    cell = summary["cells"][0]
    assert cell["prefill_active_parameter_efficiency_ratio"] == 1.5
    assert cell["decode_active_parameter_efficiency_ratio"] == 1.5
    assert cell["active_parameter_efficiency_pass"] is True
    assert summary["gates"]["active_parameter_efficiency_pass"] is True


def test_rwkv_prefill_probe_requires_greedy_and_logits_alignment() -> None:
    reference = {
        "input_ids": torch.tensor([[1, 2]]),
        "greedy_tokens": torch.tensor([3, 4]),
        "prompt_logits": torch.tensor([[1.0, 2.0]]),
        "final_logits": torch.tensor([[2.0, 3.0]]),
    }
    native = {key: value.clone() for key, value in reference.items()}
    assert compare_rwkv_prefill_probe(reference, native, 0.9999)["status"] == "pass"
    native["greedy_tokens"][1] = 5
    assert compare_rwkv_prefill_probe(reference, native, 0.9999)["status"] == "fail"


def test_comparator_rejects_torch_qwen_reference(tmp_path: Path) -> None:
    tmp = tmp_path
    rows = [
        row("candidate", prompt=128, prefill=120.0, decode=220.0),
        row("reference", prompt=128, prefill=100.0, decode=200.0, qwen_backend="torch"),
    ]
    proc = run_compare(tmp, rows, "--expected-cells", "1", "--fail-on-gate")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    summary = json.loads((tmp / "summary.json").read_text(encoding="utf-8"))
    assert summary["reference_backend"]["required"] == "fla"
    assert summary["reference_backend"]["matching_cells"] == 0
    assert summary["gates"]["reference_backend_pass"] is False


def test_comparator_accepts_fla_core_with_torch_conv(tmp_path: Path) -> None:
    reference = row("reference", prompt=128, prefill=100.0, decode=200.0)
    reference.update(
        {
            "effective_backend": "qwen_fla_gated_delta_rule_torch_conv",
            "qwen_fla_core_contract_pass": True,
            "qwen_causal_conv1d_contract_pass": False,
            "qwen_full_fused_contract_pass": False,
        }
    )
    proc = run_compare(
        tmp_path,
        [row("candidate", prompt=128, prefill=120.0, decode=220.0), reference],
        "--expected-cells",
        "1",
        "--fail-on-gate",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["reference_backend"]["matching_cells"] == 1
    assert summary["gates"]["reference_backend_pass"] is True


def fake_operator(origin: str):
    module, name = origin.rsplit(".", 1)

    def operator(*_args, **_kwargs):
        return None

    operator.__module__ = module
    operator.__name__ = name
    operator.__qualname__ = name
    return operator


def fake_qwen_model(*, accelerated: bool, fused_conv: bool = True):
    if accelerated:
        prefill = fake_operator("fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule")
        decode = fake_operator("fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule")
        conv = (
            fake_operator("causal_conv1d.causal_conv1d_interface.causal_conv1d_fn")
            if fused_conv
            else None
        )
        conv_update = (
            fake_operator("causal_conv1d.causal_conv1d_interface.causal_conv1d_update")
            if fused_conv
            else fake_operator("transformers.models.qwen3_5.modeling_qwen3_5.torch_causal_conv1d_update")
        )
        norm_type = type("FusedRMSNormGated", (), {})
        norm_type.__module__ = "fla.modules"
    else:
        prefill = fake_operator("transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule")
        decode = fake_operator("transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule")
        conv = None
        conv_update = fake_operator("transformers.models.qwen3_5.modeling_qwen3_5.torch_causal_conv1d_update")
        norm_type = type("Qwen3_5RMSNormGated", (), {})
        norm_type.__module__ = "transformers.models.qwen3_5.modeling_qwen3_5"
    layer = SimpleNamespace(
        chunk_gated_delta_rule=prefill,
        recurrent_gated_delta_rule=decode,
        causal_conv1d_fn=conv,
        causal_conv1d_update=conv_update,
        norm=norm_type(),
    )
    return SimpleNamespace(named_modules=lambda: [("model.layers.0.linear_attn", layer)])


def test_qwen_fla_operator_contract_checks_bound_operators() -> None:
    model = fake_qwen_model(accelerated=True)
    contract = qwen_fla_operator_contract(model)
    assert contract["qwen_operator_contract_pass"] is True
    assert contract["qwen_linear_attention_layers"] == 1
    assert contract["qwen_fla_prefill_layers"] == 1
    assert contract["qwen_fla_decode_layers"] == 1
    assert contract["qwen_causal_conv1d_prefill_layers"] == 1
    assert contract["qwen_causal_conv1d_update_layers"] == 1
    assert contract["qwen_fla_norm_layers"] == 1
    assert contract["qwen_fla_core_contract_pass"] is True
    assert contract["qwen_causal_conv1d_contract_pass"] is True
    assert contract["qwen_full_fused_contract_pass"] is True
    assert enforce_qwen_backend(model, worker_args(model_kind="qwen35", qwen_backend="fla")) == contract

    windows_model = fake_qwen_model(accelerated=True, fused_conv=False)
    windows_contract = qwen_fla_operator_contract(windows_model)
    assert windows_contract["qwen_operator_contract_pass"] is True
    assert windows_contract["qwen_fla_core_contract_pass"] is True
    assert windows_contract["qwen_causal_conv1d_contract_pass"] is False
    assert windows_contract["qwen_full_fused_contract_pass"] is False
    assert enforce_qwen_backend(
        windows_model, worker_args(model_kind="qwen35", qwen_backend="fla")
    ) == windows_contract

    fallback = fake_qwen_model(accelerated=False)
    fallback_contract = qwen_fla_operator_contract(fallback)
    assert fallback_contract["qwen_operator_contract_pass"] is False
    try:
        enforce_qwen_backend(fallback, worker_args(model_kind="qwen35", qwen_backend="fla"))
    except RuntimeError as exc:
        assert "FLA backend was required" in str(exc)
        assert "chunk_gated_delta_rule" in str(exc)
    else:
        raise AssertionError("required Qwen FLA backend must reject bound torch fallback operators")

    partial_layer = fake_qwen_model(accelerated=True).named_modules()[0][1]
    del partial_layer.recurrent_gated_delta_rule
    partial = SimpleNamespace(named_modules=lambda: [("model.layers.0.linear_attn", partial_layer)])
    partial_contract = qwen_fla_operator_contract(partial)
    assert partial_contract["qwen_linear_attention_layers"] == 1
    assert partial_contract["qwen_fla_decode_layers"] == 0
    assert partial_contract["qwen_operator_contract_pass"] is False


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
        "benchmark_matrix": "qwen35_test_hf",
        "dtype": "fp16",
        "quantization": "none",
        "device": "cpu",
        "batch_size": 2,
        "prompt_tokens": 8,
        "decode_tokens": 4,
        "warmup": 1,
        "runs": 1,
        "rwkv_code_source": "repo",
        "qwen_backend": "fla",
        "probe_output": "",
        "probe_tokens": 8,
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
    assert last_rwkv_prefill_backend(SimpleNamespace(_rwkv7_last_fast_prefill_backend="native_prefill")) == "native_prefill"


def test_worker_chunked_prefill_carries_hf_cache() -> None:
    calls: list[tuple[list[int], int | None]] = []

    class FakeQwen:
        def __call__(self, ids, *, past_key_values=None, **_kwargs):
            seen = None if past_key_values is None else int(past_key_values)
            calls.append((ids.flatten().tolist(), seen))
            total = (seen or 0) + int(ids.shape[1])
            return SimpleNamespace(
                logits=torch.zeros((ids.shape[0], 1, 4)),
                past_key_values=total,
            )

    args = worker_args(model_kind="qwen35", prefill_chunk_size=3)
    ids = torch.arange(8).reshape(1, 8)
    out = forward_prefill(args, FakeQwen(), ids)
    assert calls == [([0, 1, 2], None), ([3, 4, 5], 3), ([6, 7], 6)]
    assert out.past_key_values == 8

    class FakeRWKV:
        def rwkv7_prefill_chunks(self, ids, *, chunk_size, logits_to_keep):
            return (tuple(ids.shape), chunk_size, logits_to_keep)

    rwkv_args = worker_args(prefill_chunk_size=4)
    assert forward_prefill(rwkv_args, FakeRWKV(), ids) == ((1, 8), 4, 1)


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

    qwen_torchao = worker_args(model_kind="qwen35", quantization="torchao_w8")
    try:
        validate_args(qwen_torchao)
    except ValueError as exc:
        assert "RWKV candidate backend" in str(exc)
    else:
        raise AssertionError("Qwen reference must not be mislabeled as TorchAO-quantized")

    qwen_native = worker_args(model_kind="qwen35", quantization="a8w8")
    try:
        validate_args(qwen_native)
    except ValueError as exc:
        assert "RWKV candidate backend" in str(exc)
    else:
        raise AssertionError("Qwen reference must not be mislabeled as native-quantized")

    qwen_hybrid = worker_args(model_kind="qwen35", quantization="bnb8_a8w8_head")
    try:
        validate_args(qwen_hybrid)
    except ValueError as exc:
        assert "RWKV candidate backend" in str(exc)
    else:
        raise AssertionError("Qwen reference must not be mislabeled as hybrid-quantized")


def _fake_operator(module_name: str):
    def op(*_args, **_kwargs):
        return None

    op.__module__ = module_name
    return op


class FakeQwenModel:
    def __init__(self, *, fast: bool) -> None:
        origin = {
            "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface",
            "causal_conv1d_update": "causal_conv1d.causal_conv1d_interface",
            "chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk",
            "recurrent_gated_delta_rule": "fla.ops.gated_delta_rule.fused_recurrent",
        }
        if not fast:
            origin["chunk_gated_delta_rule"] = "transformers.models.qwen3_5.modeling_qwen3_5"
        self.layer = SimpleNamespace(**{name: _fake_operator(module) for name, module in origin.items()})

    def modules(self):
        return [self, self.layer]


def test_qwen_fast_path_binding_verification_is_fail_closed() -> None:
    fast = qwen35_fast_path_bindings(FakeQwenModel(fast=True))
    assert fast["verified"] is True
    assert fast["layer_count"] == 1
    assert fast["bindings"]["chunk_gated_delta_rule"].startswith("fla.")

    fallback = qwen35_fast_path_bindings(FakeQwenModel(fast=False))
    assert fallback["verified"] is False
    assert fallback["layer_count"] == 1


def test_qwen_fla_triton_bridge_preserves_qwen_layout_and_cache(monkeypatch) -> None:
    convolution = pytest.importorskip("fla.modules.convolution")

    seen: dict[str, tuple[int, ...] | str] = {}

    def fake_prefill(x, *, weight, bias, activation, backend):
        seen["prefill_shape"] = tuple(x.shape)
        seen["prefill_backend"] = backend
        return x + 1, torch.zeros_like(x[:, -2:, :])

    def fake_update(x, state, *, weight, bias, activation):
        seen["update_shape"] = tuple(x.shape)
        state.add_(2)
        return x + 3, state

    monkeypatch.setattr(convolution, "causal_conv1d", fake_prefill)
    monkeypatch.setattr(convolution, "causal_conv1d_update", fake_update)
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    state = torch.zeros((2, 2, 3), dtype=torch.float32)
    prefill = qwen35_fla_triton_causal_conv1d(
        x=x,
        weight=torch.ones((3, 2)),
        activation="silu",
    )
    update = qwen35_fla_triton_causal_conv1d_update(
        x[:, :, :1],
        state,
        torch.ones((3, 2)),
        activation="silu",
    )
    assert seen == {
        "prefill_shape": (2, 4, 3),
        "prefill_backend": "triton",
        "update_shape": (2, 1, 3),
    }
    assert prefill.shape == x.shape
    assert torch.equal(prefill, x + 1)
    assert update.shape == (2, 3, 1)
    assert torch.equal(update, x[:, :, :1] + 3)
    assert torch.equal(state, torch.full_like(state, 2))

    def fake_update_copy(x, state, *, weight, bias, activation):
        return x + 4, state + 5

    monkeypatch.setattr(convolution, "causal_conv1d_update", fake_update_copy)
    copied_state = torch.zeros((2, 2, 3), dtype=torch.float32)
    copied = qwen35_fla_triton_causal_conv1d_update(
        x[:, :, :1],
        copied_state,
        torch.ones((3, 2)),
        activation="silu",
    )
    assert torch.equal(copied, x[:, :, :1] + 4)
    assert torch.equal(copied_state, torch.full_like(copied_state, 5))


def test_qwen_fla_triton_binding_satisfies_live_full_fused_contract() -> None:
    layer_type = type("Qwen3_5GatedDeltaNet", (), {})
    layer = layer_type()
    layer.chunk_gated_delta_rule = _fake_operator("fla.ops.gated_delta_rule.chunk")
    layer.recurrent_gated_delta_rule = _fake_operator(
        "fla.ops.gated_delta_rule.fused_recurrent"
    )
    layer.norm = _fake_operator("fla.modules.fused_norm_gate")
    model = SimpleNamespace(
        modules=lambda: [model, layer],
        named_modules=lambda: [("model.layers.0.linear_attn", layer)],
    )
    assert bind_qwen35_fla_triton_conv(model) == 1
    contract = qwen_fla_operator_contract(model)
    assert contract["qwen_full_fused_contract_pass"] is True
    assert contract["qwen_conv_backend_effective"] == "fla_triton"
    args = worker_args(
        model_kind="qwen35",
        qwen_backend="fla",
        qwen_conv_backend="fla_triton",
        require_qwen_fast_path=True,
    )
    validate_loaded_model(args, model)
    assert qwen_effective_backend(args, contract) == "qwen_fla_gated_delta_rule_fla_triton_conv"


def test_model_parameter_metadata_counts_logical_and_active_work() -> None:
    class FakeParameter:
        def __init__(self, numel: int, logical_shape=None) -> None:
            self._numel = numel
            self.quant_state = (
                SimpleNamespace(shape=logical_shape) if logical_shape is not None else None
            )

        def numel(self) -> int:
            return self._numel

    shared = FakeParameter(10)
    packed_expert = FakeParameter(8, logical_shape=(4, 8))
    model = SimpleNamespace(
        config=SimpleNamespace(num_experts=4, num_experts_per_tok=1),
        named_parameters=lambda: [
            ("shared.weight", shared),
            ("tied.weight", shared),
            ("block.experts.weight", packed_expert),
        ],
    )
    metadata = model_parameter_metadata(
        model,
        Namespace(batch_size=8, prompt_tokens=128, decode_tokens=512),
    )
    assert metadata["logical_parameter_count"] == 42
    assert metadata["active_parameter_count"] == 18
    assert metadata["active_parameter_method"] == "moe_topk_logical"
    assert metadata["prefill_active_parameter_applications"] == 18 * 8 * 128
    assert metadata["decode_active_parameter_applications"] == 18 * 8 * 512


def test_repo_code_staging_works_without_symlink_privilege(tmp_path: Path) -> None:
    source = tmp_path / "rwkv-model"
    source.mkdir()
    weight = source / "model.safetensors"
    weight.write_bytes(b"weights")
    (source / "stale_modeling.py").write_text("STALE = True\n", encoding="utf-8")

    staged_path, temporary = prepare_rwkv_model_dir(str(source), "repo")
    assert temporary is not None
    staged = Path(staged_path)
    assert (staged / "model.safetensors").read_bytes() == b"weights"
    assert not (staged / "stale_modeling.py").exists()
    assert (staged / "modeling_rwkv7.py").exists()
    temporary.cleanup()
    assert not staged.exists()


def test_backend_probe_comparator_checks_logits_and_greedy() -> None:
    common = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "prompt_logits": torch.tensor([[0.1, 0.2, 0.3]]),
        "final_logits": torch.tensor([[0.4, 0.5, 0.6]]),
        "greedy_tokens": torch.tensor([3, 4, 5]),
    }
    result = compare_backend_probe(
        {**common, "qwen_backend_requested": "fla"},
        {**common, "qwen_backend_requested": "torch"},
        0.999,
    )
    assert result["status"] == "pass"
    assert result["greedy_tokens_match"] is True

    mismatch = compare_backend_probe(
        {**common, "greedy_tokens": torch.tensor([3, 4, 6])},
        common,
        0.999,
    )
    assert mismatch["status"] == "fail"


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
    assert any(key[-2] == "candidate" for key in keys)
    assert any(key[-2] == "reference" for key in keys)
    assert {key[-1] for key in keys} == {"fla"}


def test_orchestrator_failure_row_does_not_depend_on_main_scope(tmp_path: Path) -> None:
    result_path = tmp_path / "failed.jsonl"
    spec = RunSpec(
        model_pair="rwkv-1.5b__qwen3.5-2b",
        model_role="candidate",
        model_kind="rwkv",
        model_size_label="1.5b",
        model="/models/rwkv",
        prompt_tokens=128,
        decode_tokens=128,
        batch_size=1,
        dtype="fp16",
        quantization="bnb8",
    )
    proc = subprocess.CompletedProcess(["python", "worker.py"], 7, stdout="", stderr="boom")
    append_orchestrator_failure(
        result_path,
        spec,
        ["python", "worker.py"],
        proc,
        benchmark_matrix="qwen35_test",
    )
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved["benchmark_matrix"] == "qwen35_test"
    assert saved["returncode"] == 7
    assert saved["error"] == "boom"


def test_orchestrator_forces_production_rwkv_wrapper() -> None:
    args = Namespace(rwkv_fast_token_backend="native_graph")
    env = build_run_environment(args, {"RWKV7_NATIVE_MODEL": "1", "PYTHONPATH": "/existing"})
    assert env["RWKV7_NATIVE_MODEL"] == "0"
    assert env["RWKV7_FAST_TOKEN_BACKEND"] == "native_graph"
    assert env["PYTHONPATH"].endswith(f"{os.pathsep}/existing")


def test_3090_entrypoint_requires_optimized_qwen_path() -> None:
    for name in (
        "run_3090_qwen35_pair.sh",
        "run_3090_qwen35_pair_resident.sh",
        "run_3090_qwen35_pair_acceptance.sh",
        "run_3090_qwen35_speed_matrix.sh",
    ):
        script = (ROOT / "bench" / name).read_text(encoding="utf-8")
        assert "--require-qwen-fast-path" in script

    acceptance = (ROOT / "bench" / "run_3090_qwen35_pair_acceptance.sh").read_text(
        encoding="utf-8"
    )
    assert 'qwen_backend="fla"' in acceptance
    assert 'DENSE_PREFILL_GATE="${DENSE_PREFILL_GATE:-1.00}"' in acceptance
    assert "DENSE_DECODE_GATE=" in acceptance


def test_4090_acceptance_entrypoint_is_exact_card_and_chunk_safe() -> None:
    script = (ROOT / "bench" / "run_4090_qwen35_pair_acceptance.sh").read_text(
        encoding="utf-8"
    )
    assert 'PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-512}"' in script
    assert 'REQUIRED_GPU_SUBSTRING="${REQUIRED_GPU_SUBSTRING:-RTX 4090}"' in script
    assert '"${gpu_name}" != *"${REQUIRED_GPU_SUBSTRING}"*' in script
    assert 'BENCHMARK_MATRIX="${BENCHMARK_MATRIX:-qwen35_4090_hf_final}"' in script
    assert 'QWEN_CONV_BACKEND="${QWEN_CONV_BACKEND:-auto}"' in script
    assert 'REQUIRE_QWEN_FULL_FUSED="${REQUIRE_QWEN_FULL_FUSED:-0}"' in script
    assert '--qwen-conv-backend "${QWEN_CONV_BACKEND}"' in script
    assert 'common_compare+=(--require-qwen-full-fused)' in script
    assert 'if [[ "${RUN_NATIVE_MM8:-0}" == "1" ]]' in script
    assert "native_speed_mm8" in script
    assert "--require-qwen-fast-path" in script
    assert 'qwen_backend="fla"' in script
    assert 'DENSE_PREFILL_GATE="${DENSE_PREFILL_GATE:-1.00}"' in script
    assert 'ACTIVE_DECODE_WORK_GATE="${ACTIVE_DECODE_WORK_GATE:-1.00}"' in script
    assert "--min-decode-active-parameter-throughput-ratio" in script
    assert "RWKV7_NATIVE_GRAPH_EXTERNAL_QUANT=1" in script
    assert "RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT_GRAPH=1" in script
    assert "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MIN_TOKENS=128" in script
    assert "--allow-quant-total-not-slower-than-dense" in script
    assert 'printf \'%s\\n\' "${pipeline_rc}" > "${OUT_DIR}/pipeline_exit_code.txt"' in script


def test_4080_acceptance_entrypoint_is_full_prompt_and_fail_closed() -> None:
    script = (ROOT / "bench" / "run_4080_qwen35_pair_acceptance.sh").read_text(
        encoding="utf-8"
    )
    assert 'REQUIRED_GPU_SUBSTRING="${REQUIRED_GPU_SUBSTRING:-RTX 4080}"' in script
    assert '--benchmark-matrix qwen35_4080_hf_final' in script
    assert 'rwkv-0.4b__qwen3.5-0.8b)' in script
    assert 'rwkv-1.5b__qwen3.5-2b)' in script
    assert 'rwkv-2.9b__qwen3.5-4b)' in script
    assert 'BATCH_SIZE="${BATCH_SIZE:-8}"' in script
    assert '--batch-sizes "${BATCH_SIZE}" --prompt-tokens 128 512 2048' in script
    assert '--decode-tokens 128 512 --prefill-chunk-size "${PREFILL_CHUNK_SIZE}"' in script
    assert 'if [[ "${BATCH_SIZE}" == "1" ]]' in script
    assert 'DENSE_DECODE_GATE="${DENSE_DECODE_GATE:-1.00}"' in script
    assert 'default_active_work_gate="1.75"' in script
    assert '--min-active-work-decode "${ACTIVE_WORK_DECODE_GATE}"' in script
    assert '--model-pair "${PAIR_LABEL}"' in script
    assert '--batch-size "${BATCH_SIZE}"' in script
    assert '--qwen-backend fla' in script
    assert '--require-qwen-fast-path' in script
    assert '--paired-baseline' in script
    assert 'for quant in a8w8 torchao_w4' in script
    assert 'for quant in bnb8 bnb4' in script
    assert 'summarize_4080_qwen35_acceptance.py' in script
    assert 'printf \'%s\\n\' "${pipeline_rc}" > "${OUT_DIR}/pipeline_exit_code.txt"' in script


def test_5090_acceptance_reuses_contract_without_4090_policy_defaults() -> None:
    script = (ROOT / "bench" / "run_5090_qwen35_pair_acceptance.sh").read_text(
        encoding="utf-8"
    )
    assert 'REQUIRED_GPU_SUBSTRING="RTX 5090"' in script
    assert 'BENCHMARK_MATRIX="${BENCHMARK_MATRIX:-qwen35_5090_hf_final}"' in script
    assert 'QWEN_CONV_BACKEND="fla_triton"' in script
    assert "export RUN_NATIVE_MM8=1" in script
    assert "export REQUIRE_QWEN_FULL_FUSED=1" in script
    assert "unset ALLOW_NON_4090" in script
    assert "RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M" not in script
    assert "export RWKV7_FAST_PREFILL=1" in script
    assert "export RWKV7_FAST_PREFILL_QUANT=1" in script
    assert "export RWKV7_NATIVE_PREFILL_FUSED_SCAN=1" in script
    assert "export RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX=1" in script
    assert "export RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP=1" in script
    assert "export RWKV7_NATIVE_PREFILL_FUSED_OUTPUT=1" in script
    assert "export RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN=0" in script
    assert 'run_4090_qwen35_pair_acceptance.sh" "$@"' in script


def test_5090_correctness_entrypoint_checks_full_fla_and_native_prefill() -> None:
    script = (ROOT / "bench" / "run_5090_qwen35_correctness.sh").read_text(
        encoding="utf-8"
    )
    assert 'REQUIRED_GPU_SUBSTRING="RTX 5090"' in script
    assert 'CORRECTNESS_PROMPT_TOKENS="${CORRECTNESS_PROMPT_TOKENS:-512}"' in script
    assert 'CORRECTNESS_BATCH_SIZE="${CORRECTNESS_BATCH_SIZE:-8}"' in script
    assert 'QWEN_CORRECTNESS_PROMPT_TOKENS="${QWEN_CORRECTNESS_PROMPT_TOKENS:-}"' in script
    assert '[[ "${qwen_size}" == "9b" ]]' in script
    assert "QWEN_CORRECTNESS_PROMPT_TOKENS=512" in script
    assert '--prompt-tokens "${QWEN_CORRECTNESS_PROMPT_TOKENS}"' in script
    assert '--batch-size "${CORRECTNESS_BATCH_SIZE}"' in script
    assert "--qwen-conv-backend fla_triton --require-qwen-fast-path" in script
    assert "compare_qwen35_backend_probe.py" in script
    assert "--min-cosine 0.999" in script
    assert "for quantization in none bnb8 bnb4" in script
    assert "compare_rwkv_prefill_probe.py" in script
    assert "--min-cosine 0.9999" in script
    assert "RWKV7_NATIVE_PREFILL_FUSED_SCAN=1" in script
    assert "RWKV7_NATIVE_PREFILL_FUSED_OUTPUT=1" in script
    assert '[[ ${failures} -eq 0 ]]' in script


def test_5090_full_matrix_entrypoint_runs_all_exact_pairs() -> None:
    script = (ROOT / "bench" / "run_5090_qwen35_full_matrix.sh").read_text(
        encoding="utf-8"
    )
    for pair in (
        "rwkv-0.4b__qwen3.5-0.8b",
        "rwkv-1.5b__qwen3.5-2b",
        "rwkv-2.9b__qwen3.5-4b",
        "rwkv-7.2b__qwen3.5-9b",
    ):
        assert pair in script
    assert "run_5090_qwen35_correctness.sh" in script
    assert "run_5090_qwen35_pair_acceptance.sh" in script
    assert 'ACCEPTANCE_BATCH_SIZES="${ACCEPTANCE_BATCH_SIZES:-1 8}"' in script
    assert 'for batch_size in ${ACCEPTANCE_BATCH_SIZES}' in script
    assert 'out_dir="${OUT_ROOT}/b${batch_size}/${out_name}"' in script
    assert 'CORRECTNESS_BATCH_SIZE="${batch_size}"' in script
    assert 'BATCH_SIZES="${batch_size}"' in script
    assert "summarize_5090_qwen35_acceptance.py" in script
    assert 'printf \'%s\\n\' "${summary_rc}" > "${OUT_ROOT}/summary-exit-code.txt"' in script
    assert 'printf \'%s\\n\' "${pipeline_rc}" > "${OUT_ROOT}/pipeline-exit-code.txt"' in script


def test_5090_g1h_13b_entrypoint_is_fail_closed() -> None:
    script = (ROOT / "bench" / "run_5090_g1h_13b_acceptance.sh").read_text(
        encoding="utf-8"
    )
    assert 'REQUIRED_GPU_SUBSTRING="RTX 5090"' in script
    assert "bench_larger_model_smoke.py" in script
    assert "--fast-token-backend native_jit" in script
    assert "--quantizations none mm8 mm4" in script
    assert "--paired-baseline --fail-fast" in script
    assert "--gate --expected-rows 3 --min-speed-ratio 0.98" in script
    assert "smoke_rc" in script and "quant_rc" in script and "gate_rc" in script
    assert 'printf \'%s\\n\' "${pipeline_rc}" > "${OUT_DIR}/pipeline-exit-code.txt"' in script


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
    assert "--min-quant-prefill-speedup 1.00" in pair_script
    assert "--min-quant-decode-speedup 1.00" in pair_script
    assert "--fail-on-gate" in pair_script
    assert 'QWEN_BACKEND="${QWEN_BACKEND:-auto}"' in pair_script
    assert '--qwen-backend "${QWEN_BACKEND}"' in pair_script
    assert '--model-roles "${MODEL_ROLE_ARGS[@]}"' in pair_script
    assert 'COMPARE_AFTER="${COMPARE_AFTER:-1}"' in pair_script
    for name in (
        "run_3090_qwen35_pair.sh",
        "run_3090_qwen35_pair_resident.sh",
        "run_3090_qwen35_speed_matrix.sh",
    ):
        script = (ROOT / "bench" / name).read_text(encoding="utf-8")
        assert "--require-native-candidate" in script
        assert "--require-qwen-fast-path" in script
        assert "--require-quant-memory-reduction" in script
        assert "--require-prefill-mode-match" in script
        assert "--require-quant-not-slower-than-dense" in script


def test_orchestrator_isolates_qwen_import_backend() -> None:
    pair = parse_pair_spec("rwkv-1.5b__qwen3.5-2b=/rwkv/1.5b::Qwen/Qwen3.5-2B")
    specs = build_run_specs(
        MatrixConfig(
            pairs=[pair],
            prompts=[128],
            decodes=[8],
            batch_sizes=[1],
            quantizations=["none"],
            dtype="fp16",
        )
    )
    qwen_spec = next(spec for spec in specs if spec.model_kind == "qwen35")
    base = {"RWKV7_QWEN35_FORCE_TORCH": "1"}
    assert "RWKV7_QWEN35_FORCE_TORCH" not in build_worker_environment(base, qwen_spec, "fla")
    assert build_worker_environment({}, qwen_spec, "torch")["RWKV7_QWEN35_FORCE_TORCH"] == "1"
    rwkv_bnb8 = next(spec for spec in specs if spec.model_kind == "rwkv")
    rwkv_bnb8 = type(rwkv_bnb8)(**{**rwkv_bnb8.__dict__, "quantization": "bnb8"})
    env = build_worker_environment({}, rwkv_bnb8, "fla", "decode_rk")
    assert env["RWKV7_BNB_SKIP_POLICY"] == "decode_rk"


def test_5070_qwen_fla_evidence_is_complete() -> None:
    evidence = ROOT / "bench" / "5070_qwen35_fla_matrix_20260713"
    rows = [json.loads(line) for line in (evidence / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 144
    assert all(row["status"] == "pass" for row in rows)

    qwen_rows = [row for row in rows if row["model_role"] == "reference"]
    assert len(qwen_rows) == 72
    assert all(row["qwen_fla_core_contract_pass"] is True for row in qwen_rows)
    assert all(
        row["effective_backend"] == "qwen_fla_gated_delta_rule_torch_conv"
        for row in qwen_rows
    )

    summary = json.loads((evidence / "summary.json").read_text(encoding="utf-8-sig"))
    assert summary["coverage"]["joined_cells"] == 72
    assert summary["reference_backend"]["matching_cells"] == 72
    assert summary["speed"]["strict_gate_cells"] == 35
    assert summary["speed"]["decode_at_least_equal_cells"] == 72
    assert summary["memory"]["model_footprint_not_larger_cells"] == 72
    assert summary["memory"]["peak_vram_not_larger_cells"] == 72

    probe = json.loads((evidence / "fla-vs-torch-probe.json").read_text(encoding="utf-8-sig"))
    assert probe["status"] == "pass"
    assert probe["greedy_tokens_match"] is True
    assert min(probe["prompt_logits_cosine"], probe["final_logits_cosine"]) >= 0.999

    exit_codes = json.loads((evidence / "exit-codes.json").read_text(encoding="utf-8-sig"))
    assert exit_codes
    assert all(code == 0 for code in exit_codes.values())


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        test_comparator_passes_complete_matrix(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_comparator_reports_missing_and_slow_cells(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_comparator_supports_strict_nonnegative_quant_gate(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_red_candidate_rerunner_builds_append_only_command(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_comparator_rejects_torch_qwen_reference(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_comparator_accepts_fla_core_with_torch_conv(Path(td))
    test_qwen_fla_operator_contract_checks_bound_operators()
    test_worker_helpers_build_exact_shape_and_metadata()
    test_worker_helpers_validate_and_emit_failure()
    with tempfile.TemporaryDirectory() as td:
        test_repo_code_staging_works_without_symlink_privilege(Path(td))
    test_backend_probe_comparator_checks_logits_and_greedy()
    test_orchestrator_expands_432_raw_rows()
    with tempfile.TemporaryDirectory() as td:
        test_orchestrator_existing_keys_are_resumable(Path(td))
    test_orchestrator_forces_production_rwkv_wrapper()
    test_3090_entrypoint_requires_optimized_qwen_path()
    test_hardware_entrypoints_are_fail_closed()
    test_orchestrator_isolates_qwen_import_backend()
    test_5070_qwen_fla_evidence_is_complete()
    print("QWEN35 SPEED MATRIX TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
