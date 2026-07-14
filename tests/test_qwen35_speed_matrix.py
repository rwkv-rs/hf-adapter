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

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.bench_cross_model_speed import (
    build_exact_prompt,
    effective_quantization_metadata,
    failure_row,
    forward_prefill,
    last_rwkv_prefill_backend,
    model_metadata,
    qwen35_fast_path_bindings,
    validate_args,
)
from bench.bench_cross_model_speed_resident import resolve_sweep_cells, resolve_sweep_shapes
from bench.compare_qwen35_speed_matrix import quantization_family
from bench.run_qwen35_speed_matrix import (
    MatrixConfig,
    RunSpec,
    append_orchestrator_failure,
    build_run_environment,
    build_run_specs,
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
) -> dict:
    candidate = role == "candidate"
    return {
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": "qwen35_v100_hf",
        "model_pair": "rwkv-1.5b__qwen3.5-2b",
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
        "prefill_effective_backend": "native_prefill" if candidate else "module_call",
        "effective_backend": "native_graph" if candidate else "fla+causal_conv1d",
        "qwen_fast_path_verified": None if candidate else True,
        "model_footprint_mb": footprint if footprint is not None else (100.0 if candidate else 120.0),
        "peak_vram_mb": footprint if footprint is not None else (100.0 if candidate else 120.0),
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
        "run_3090_qwen35_speed_matrix.sh",
    ):
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


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        test_comparator_passes_complete_matrix(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_comparator_reports_missing_and_slow_cells(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_comparator_supports_strict_nonnegative_quant_gate(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_red_candidate_rerunner_builds_append_only_command(Path(td))
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
