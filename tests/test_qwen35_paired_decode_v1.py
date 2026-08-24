from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
import torch

from bench.analyzers.compare_rwkv_prefill_probe import compare as compare_rwkv_probes
from bench.validators.validate_qwen35_paired_decode_v1 import (
    CANDIDATE_LANE,
    CANDIDATE_MATRIX,
    DECODE_CORRECTNESS_PROTOCOL,
    FROZEN_REFERENCE_SHA256,
    PARAMETERS,
    PAIR_RANK,
    RWKV_PAIR_SIZES,
    main,
    render_markdown,
    validate_files,
    validate_paired_decode,
)
from tests.test_qwen35_best_optimized_hf_v1 import (
    complete_rows as complete_qwen_rows,
)


DEVICE = "NVIDIA GeForce RTX 5090"
FIXTURE_REFERENCE_SHA256 = "a" * 64
ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_COMMIT = "c" * 40
REFERENCE_COMMIT = "b" * 40


def reference_rows() -> list[dict]:
    rows = deepcopy(complete_qwen_rows())
    for line_number, row in enumerate(rows, 1):
        row["active_parameter_count"] = PARAMETERS[row["model_pair"]][1]
        row["benchmark_repository_commit"] = REFERENCE_COMMIT
        row["_source"] = f"reference.jsonl:{line_number}"
        row["_line_number"] = line_number
    return rows


def candidate_rows(
    references: list[dict], *, adjusted_ratio: float = 1.1
) -> list[dict]:
    rows: list[dict] = []
    for line_number, reference in enumerate(references, 1):
        pair = reference["model_pair"]
        candidate_parameters, reference_parameters = PARAMETERS[pair]
        decode_rate = (
            float(reference["decode_tokps_total_raw"])
            * adjusted_ratio
            * reference_parameters
            / candidate_parameters
        )
        batch = int(reference["batch_size"])
        decode = int(reference["decode_tokens"])
        decode_s = batch * decode / decode_rate
        requires_sm120_route = batch == 8 and pair in {
            "rwkv-0.4b__qwen3.5-0.8b",
            "rwkv-1.5b__qwen3.5-2b",
        }
        sm120_layers = list(range(24)) if requires_sm120_route else []
        rows.append(
            {
                "_source": f"candidate.jsonl:{line_number}",
                "_line_number": line_number,
                "axis": "qwen35_cross_model_speed",
                "benchmark_matrix": CANDIDATE_MATRIX,
                "benchmark_repository_commit": CANDIDATE_COMMIT,
                "optimization_lane": CANDIDATE_LANE,
                "model_pair": pair,
                "model_size_label": RWKV_PAIR_SIZES[pair],
                "model_id_or_path": f"/models/{RWKV_PAIR_SIZES[pair]}",
                "model_role": "candidate",
                "model_kind": "rwkv",
                "rwkv_implementation_requested": "auto",
                "rwkv_implementation_effective": "native_model",
                "dtype": "fp16",
                "quantization": "none",
                "quantization_backend": "dense",
                "native_quant_kernel_active": False,
                "batch_size": batch,
                "prompt_tokens": int(reference["prompt_tokens"]),
                "decode_tokens": decode,
                "prefill_chunk_size": 512,
                "warmup": 3,
                "runs": 7,
                "timing_statistic": "median",
                "mtp_enabled": False,
                "speculative_decoding_enabled": False,
                "resident_sweep": True,
                "status": "pass",
                "logits_finite": True,
                "device": DEVICE,
                "gpu_arch": "sm_120",
                "gpu_compute_capability": [12, 0],
                "torch_version": reference["torch_version"],
                "torch_cuda_version": reference["torch_cuda_version"],
                "triton_version": reference["triton_version"],
                "transformers_version": reference["transformers_version"],
                "fla_version": reference["fla_version"],
                "causal_conv1d_version": reference["causal_conv1d_version"],
                "rwkv_fast_token_backend_requested": "native_graph",
                "rwkv_native_model_backend_requested": "native_graph",
                "effective_backend": "native_graph",
                "step_backend": "rwkv_fast_token",
                "cache_type": "NativeRWKV7Cache",
                "rwkv_native_graph_ada_wagv_bmm_requested": requires_sm120_route,
                "rwkv_native_graph_ada_wagv_bmm_selected": requires_sm120_route,
                "rwkv_native_graph_ada_wagv_bmm_effective": requires_sm120_route,
                "rwkv_native_graph_ada_wagv_bmm_effective_layer_count": (
                    24 if requires_sm120_route else 0
                ),
                "rwkv_native_graph_ada_wagv_bmm_full_model_effective": requires_sm120_route,
                "rwkv_native_graph_sm120_wagv_bmm_g_requested": requires_sm120_route,
                "rwkv_native_graph_sm120_wagv_bmm_g_selected": requires_sm120_route,
                "rwkv_native_graph_sm120_wagv_bmm_g_effective": requires_sm120_route,
                "rwkv_native_graph_sm120_wagv_bmm_g_selected_layers": sm120_layers,
                "rwkv_native_graph_sm120_wagv_bmm_g_effective_layers": sm120_layers,
                "rwkv_native_graph_sm120_wagv_bmm_g_effective_layer_count": len(
                    sm120_layers
                ),
                "rwkv_native_graph_sm120_wagv_bmm_g_full_model_effective": requires_sm120_route,
                "rwkv_native_graph_sm120_compiled_ffn_requested": requires_sm120_route,
                "rwkv_native_graph_sm120_compiled_ffn_selected": requires_sm120_route,
                "rwkv_native_graph_sm120_compiled_ffn_effective": requires_sm120_route,
                "rwkv_native_graph_sm120_compiled_ffn_selected_layers": sm120_layers,
                "rwkv_native_graph_sm120_compiled_ffn_effective_layers": sm120_layers,
                "rwkv_native_graph_sm120_compiled_ffn_effective_layer_count": len(
                    sm120_layers
                ),
                "rwkv_native_graph_sm120_compiled_ffn_full_model_effective": requires_sm120_route,
                "rwkv_native_graph_sm120_compiled_ffn_compile_effective": (
                    True if requires_sm120_route else None
                ),
                "rwkv_native_graph_sm120_compiled_ffn_compile_reused": (
                    True if requires_sm120_route else None
                ),
                "rwkv_native_graph_sm120_compiled_ffn_unique_graphs": (
                    1 if requires_sm120_route else None
                ),
                "rwkv_native_graph_sm120_compiled_ffn_graph_breaks": (
                    0 if requires_sm120_route else None
                ),
                "rwkv_native_graph_sm120_compiled_ffn_compile_mode": (
                    "max-autotune-no-cudagraphs" if requires_sm120_route else None
                ),
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_all_finite": (
                    True if requires_sm120_route else None
                ),
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine": (
                    0.99999 if requires_sm120_route else None
                ),
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_argmax_all_equal": (
                    True if requires_sm120_route else None
                ),
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_max_abs_diff": (
                    0.015625 if requires_sm120_route else None
                ),
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_indices": (
                    sm120_layers if requires_sm120_route else None
                ),
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_count": (
                    24 if requires_sm120_route else None
                ),
                "active_parameter_count": candidate_parameters,
                "decode_sec_samples": [decode_s] * 7,
                "decode_sec_median": round(decode_s, 6),
                "decode_sec_median_raw": decode_s,
                "decode_tokps_total": round(decode_rate, 3),
                "decode_tokps_total_raw": decode_rate,
            }
        )
    return rows


def validate(
    candidates: list[dict],
    references: list[dict],
    *,
    before: str = FIXTURE_REFERENCE_SHA256,
    after: str = FIXTURE_REFERENCE_SHA256,
) -> dict:
    return validate_paired_decode(
        candidates,
        references,
        expected_device=DEVICE,
        expected_reference_sha256=FIXTURE_REFERENCE_SHA256,
        reference_sha256_before=before,
        reference_sha256_after=after,
        candidate_sha256="b" * 64,
    )


def set_adjusted_ratio(candidate: dict, reference: dict, adjusted_ratio: float) -> None:
    candidate_parameters, reference_parameters = PARAMETERS[candidate["model_pair"]]
    rate = (
        float(reference["decode_tokps_total_raw"])
        * adjusted_ratio
        * reference_parameters
        / candidate_parameters
    )
    decode_s = int(candidate["batch_size"]) * int(candidate["decode_tokens"]) / rate
    candidate["decode_tokps_total_raw"] = rate
    candidate["decode_sec_samples"] = [decode_s] * 7
    candidate["decode_sec_median_raw"] = decode_s
    candidate["decode_sec_median"] = round(decode_s, 6)
    candidate["decode_tokps_total"] = round(rate, 3)


def clean_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"_source", "_line_number"}
        }
        for row in rows
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in clean_rows(rows)),
        encoding="utf-8",
    )


def write_ab_evidence(tmp_path: Path, candidates: list[dict]) -> tuple[Path, Path]:
    model_hashes = tmp_path / "model_hashes.sha256"
    model_hashes.write_text(
        "".join(
            f"[/models/{size}]\n"
            + "0" * 64
            + "  config.json\n"
            + "1" * 64
            + "  model.safetensors\n"
            + "2" * 64
            + "  tokenizer.json\n"
            for size in ("0.4b", "1.5b", "2.9b", "7.2b")
        )
    )
    entries = []
    for tag, pair, size in (
        ("0p4", "rwkv-0.4b__qwen3.5-0.8b", "0.4b"),
        ("1p5", "rwkv-1.5b__qwen3.5-2b", "1.5b"),
    ):
        source = next(
            row
            for row in candidates
            if row["model_pair"] == pair
            and row["batch_size"] == 8
            and row["prompt_tokens"] == 2048
            and row["decode_tokens"] == 512
        )
        lane_artifacts = {}
        probes = {}
        for lane in ("baseline", "candidate"):
            row = deepcopy(source)
            row["benchmark_matrix"] = "sm120_b8_decode_ab_v1"
            row["optimization_lane"] = lane
            row["probe_tokens"] = 512
            row["probe_batch_size"] = 8
            row["probe_distinct_batch_prompts"] = True
            row["probe_decode_logits_all_finite"] = True
            row["probe_decode_logits_finite_by_batch"] = [True] * 8
            probe_path = tmp_path / f"sm120_{tag}_{lane}.pt"
            row["probe_output"] = str(probe_path)
            if lane == "baseline":
                baseline_rate = float(source["decode_tokps_total_raw"]) / 1.05
                baseline_s = 8 * 512 / baseline_rate
                row["decode_tokps_total_raw"] = baseline_rate
                row["decode_tokps_total"] = round(baseline_rate, 3)
                row["decode_sec_samples"] = [baseline_s] * 7
                row["decode_sec_median_raw"] = baseline_s
                row["decode_sec_median"] = round(baseline_s, 6)
                for field in (
                    "rwkv_native_graph_ada_wagv_bmm_requested",
                    "rwkv_native_graph_ada_wagv_bmm_selected",
                    "rwkv_native_graph_ada_wagv_bmm_effective",
                    "rwkv_native_graph_ada_wagv_bmm_full_model_effective",
                ):
                    row[field] = False
                row["rwkv_native_graph_ada_wagv_bmm_effective_layer_count"] = 0
                for route in ("sm120_wagv_bmm_g", "sm120_compiled_ffn"):
                    row[f"rwkv_native_graph_{route}_requested"] = False
                    row[f"rwkv_native_graph_{route}_selected"] = False
                    row[f"rwkv_native_graph_{route}_effective"] = False
                    row[f"rwkv_native_graph_{route}_selected_layers"] = []
                    row[f"rwkv_native_graph_{route}_effective_layers"] = []
                    row[f"rwkv_native_graph_{route}_effective_layer_count"] = 0
                    row[f"rwkv_native_graph_{route}_full_model_effective"] = False
                for field in (
                    "rwkv_native_graph_sm120_compiled_ffn_compile_effective",
                    "rwkv_native_graph_sm120_compiled_ffn_compile_reused",
                    "rwkv_native_graph_sm120_compiled_ffn_unique_graphs",
                    "rwkv_native_graph_sm120_compiled_ffn_graph_breaks",
                    "rwkv_native_graph_sm120_compiled_ffn_compile_mode",
                    "rwkv_native_graph_sm120_compiled_ffn_prewarm_all_finite",
                    "rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine",
                    "rwkv_native_graph_sm120_compiled_ffn_prewarm_argmax_all_equal",
                    "rwkv_native_graph_sm120_compiled_ffn_prewarm_max_abs_diff",
                    "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_indices",
                    "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_count",
                ):
                    row[field] = None
            row_path = tmp_path / f"sm120_{tag}_{lane}.jsonl"
            write_jsonl(row_path, [row])
            input_ids = torch.arange(8 * 2048).reshape(8, 2048)
            probe = {
                "probe_schema_version": 2,
                "benchmark_repository_commit": CANDIDATE_COMMIT,
                "model_pair": pair,
                "model_size_label": size,
                "model_id_or_path": f"/models/{size}",
                "probe_output": str(probe_path),
                "input_ids": input_ids,
                "prompt_logits": torch.arange(32, dtype=torch.float32).reshape(8, 4),
                "final_logits": torch.arange(32, dtype=torch.float32).reshape(8, 4),
                "greedy_tokens": torch.zeros(512, 8, dtype=torch.int64),
                "decode_logits_finite_by_batch": torch.ones(8, dtype=torch.bool),
                "decode_logits_all_finite": True,
            }
            torch.save(probe, probe_path)
            probes[lane] = probe
            lane_artifacts[lane] = {
                "row": {
                    "path": row_path.name,
                    "sha256": sha256(row_path.read_bytes()).hexdigest(),
                },
                "probe": {
                    "path": probe_path.name,
                    "sha256": sha256(probe_path.read_bytes()).hexdigest(),
                },
            }
        comparison = compare_rwkv_probes(
            probes["baseline"], probes["candidate"], 0.9999
        )
        comparison.update(
            {
                "contract_errors": [],
                "probe_batch_size": 8,
                "probe_tokens": 512,
                "distinct_batch_prompts": True,
            }
        )
        comparison_path = tmp_path / f"sm120_{tag}_compare.json"
        comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
        entries.append(
            {
                "model_pair": pair,
                "model_size_label": size,
                **lane_artifacts,
                "comparison": {
                    "path": comparison_path.name,
                    "sha256": sha256(comparison_path.read_bytes()).hexdigest(),
                },
            }
        )
    manifest = tmp_path / "rwkv_sm120_b8_ab.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "sm120_b8_decode_ab_v1",
                "benchmark_repository_commit": CANDIDATE_COMMIT,
                "model_hashes_sha256": sha256(model_hashes.read_bytes()).hexdigest(),
                "cell": {
                    "batch_size": 8,
                    "prompt_tokens": 2048,
                    "decode_tokens": 512,
                    "probe_tokens": 512,
                    "distinct_batch_prompts": True,
                },
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest, model_hashes


def write_decode_correctness_evidence(
    tmp_path: Path,
    candidates: list[dict],
    model_hashes: Path,
) -> Path:
    runtime_lock = tmp_path / "runtime-lock.json"
    runtime_lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "qwen35_paired_decode_v1",
                "repository_commit": CANDIDATE_COMMIT,
                "runtime": {
                    "python": "3.10.12",
                    "torch": "2.8.0+cu128",
                    "torch_cuda": "12.8",
                    "triton": "3.4.0",
                    "transformers": "5.12.1",
                    "fla": "0.5.1",
                    "causal_conv1d": "1.6.2.post1",
                },
                "pip_freeze_sha256": (
                    "f5bf8ef181f2c1b29b79d6fae5c8019fa85008df120569b9e18646bd09eee5cf"
                ),
                "torch_cuda_arch_list": "12.0",
            }
        ),
        encoding="utf-8",
    )

    def artifact(path: Path) -> dict[str, str]:
        return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}

    entries = []
    order = {
        (128, 128): 1,
        (128, 512): 2,
        (512, 128): 3,
        (512, 512): 4,
        (2048, 128): 5,
        (2048, 512): 6,
    }
    for tag, pair, size in (
        ("0p4", "rwkv-0.4b__qwen3.5-0.8b", "0.4b"),
        ("1p5", "rwkv-1.5b__qwen3.5-2b", "1.5b"),
        ("2p9", "rwkv-2.9b__qwen3.5-4b", "2.9b"),
        ("7p2", "rwkv-7.2b__qwen3.5-9b", "7.2b"),
    ):
        for batch in (1, 8):
            source_rows = [
                row
                for row in candidates
                if row["model_pair"] == pair and row["batch_size"] == batch
            ]
            source_rows.sort(
                key=lambda row: order[(row["prompt_tokens"], row["decode_tokens"])]
            )
            native_probe_path = (
                tmp_path / f"decode_correctness_{tag}_b{batch}_native_candidate.pt"
            )
            for row in source_rows:
                cell = (row["prompt_tokens"], row["decode_tokens"])
                row["resident_cell_index"] = order[cell]
                row["resident_cells_total"] = 6
                row["load_amortized"] = order[cell] > 1
                row["resident_probe_cell"] = [batch, 2048, 512]
                row["resident_probe_cell_selected"] = cell == (2048, 512)
            native_row = next(
                row
                for row in source_rows
                if row["prompt_tokens"] == 2048 and row["decode_tokens"] == 512
            )

            input_ids = torch.arange(batch * 2048, dtype=torch.int64).reshape(
                batch, 2048
            )
            logits = torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 4)
            greedy_shape = (512,) if batch == 1 else (512, batch)
            greedy = torch.zeros(greedy_shape, dtype=torch.int64)

            def make_probe(path: Path) -> dict:
                probe = {
                    "probe_schema_version": 2,
                    "benchmark_repository_commit": CANDIDATE_COMMIT,
                    "model_pair": pair,
                    "model_size_label": size,
                    "model_id_or_path": f"/models/{size}",
                    "probe_output": str(path.resolve()),
                    "input_ids": input_ids,
                    "prompt_logits": logits,
                    "final_logits": logits,
                    "greedy_tokens": greedy,
                    "decode_logits_finite_by_batch": torch.ones(
                        batch, dtype=torch.bool
                    ),
                    "decode_logits_all_finite": True,
                    "qwen_backend_requested": "auto",
                }
                torch.save(probe, path)
                return probe

            native_probe = make_probe(native_probe_path)
            native_row.update(
                {
                    "probe_output": str(native_probe_path.resolve()),
                    "probe_tokens": 512,
                    "probe_batch_size": batch,
                    "probe_distinct_batch_prompts": batch == 8,
                    "probe_decode_logits_all_finite": True,
                    "probe_decode_logits_finite_by_batch": [True] * batch,
                    "probe_greedy_tokens": greedy.tolist(),
                }
            )
            source_lane_path = tmp_path / f"rwkv_{tag}_b{batch}.jsonl"
            write_jsonl(source_lane_path, source_rows)
            native_row_path = (
                tmp_path / f"decode_correctness_{tag}_b{batch}_native_candidate.jsonl"
            )
            write_jsonl(native_row_path, [native_row])

            fla_probe_path = (
                tmp_path / f"decode_correctness_{tag}_b{batch}_fla_reference.pt"
            )
            fla_probe = make_probe(fla_probe_path)
            fla_row = deepcopy(native_row)
            fla_row.update(
                {
                    "benchmark_matrix": DECODE_CORRECTNESS_PROTOCOL,
                    "optimization_lane": "fla_reference",
                    "warmup": 1,
                    "runs": 1,
                    "decode_sec_samples": [native_row["decode_sec_median_raw"]],
                    "resident_cell_index": 1,
                    "resident_cells_total": 1,
                    "load_amortized": False,
                    "rwkv_fast_token_backend_requested": "fla",
                    "rwkv_native_model_backend_requested": "eager",
                    "rwkv_implementation_requested": "wrapper_repo",
                    "rwkv_implementation_effective": "wrapper_repo",
                    "rwkv_fast_prefill_requested": "0",
                    "rwkv_prefill_graph_requested": "0",
                    "effective_backend": "fla",
                    "prefill_backend_effective": None,
                    "cache_type": "RWKV7StateCache",
                    "probe_output": str(fla_probe_path.resolve()),
                }
            )
            for field in (
                "rwkv_native_graph_ada_wagv_bmm_requested",
                "rwkv_native_graph_ada_wagv_bmm_selected",
                "rwkv_native_graph_ada_wagv_bmm_effective",
                "rwkv_native_graph_ada_wagv_bmm_full_model_effective",
            ):
                fla_row[field] = None
            fla_row["rwkv_native_graph_ada_wagv_bmm_effective_layer_count"] = None
            for route in ("sm120_wagv_bmm_g", "sm120_compiled_ffn"):
                fla_row[f"rwkv_native_graph_{route}_requested"] = None
                fla_row[f"rwkv_native_graph_{route}_selected"] = None
                fla_row[f"rwkv_native_graph_{route}_effective"] = None
                fla_row[f"rwkv_native_graph_{route}_selected_layers"] = None
                fla_row[f"rwkv_native_graph_{route}_effective_layers"] = None
                fla_row[f"rwkv_native_graph_{route}_effective_layer_count"] = None
                fla_row[f"rwkv_native_graph_{route}_full_model_effective"] = None
            for field in (
                "rwkv_native_graph_sm120_compiled_ffn_compile_effective",
                "rwkv_native_graph_sm120_compiled_ffn_compile_reused",
                "rwkv_native_graph_sm120_compiled_ffn_unique_graphs",
                "rwkv_native_graph_sm120_compiled_ffn_graph_breaks",
                "rwkv_native_graph_sm120_compiled_ffn_compile_mode",
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_all_finite",
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine",
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_argmax_all_equal",
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_max_abs_diff",
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_indices",
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_count",
            ):
                fla_row[field] = None
            fla_row_path = (
                tmp_path / f"decode_correctness_{tag}_b{batch}_fla_reference.jsonl"
            )
            write_jsonl(fla_row_path, [fla_row])

            comparison = compare_rwkv_probes(fla_probe, native_probe, 0.9999)
            comparison["contract_errors"] = []
            comparison_path = (
                tmp_path / f"decode_correctness_{tag}_b{batch}_compare.json"
            )
            comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
            entries.append(
                {
                    "model_pair": pair,
                    "model_size_label": size,
                    "model_path": f"/models/{size}",
                    "batch_size": batch,
                    "prompt_tokens": 2048,
                    "decode_tokens": 512,
                    "probe_tokens": 512,
                    "fla_reference": {
                        "row": artifact(fla_row_path),
                        "probe": artifact(fla_probe_path),
                    },
                    "native_candidate": {
                        "row": artifact(native_row_path),
                        "probe": artifact(native_probe_path),
                        "source_lane": artifact(source_lane_path),
                        "source_cell": {
                            "batch_size": batch,
                            "prompt_tokens": 2048,
                            "decode_tokens": 512,
                        },
                    },
                    "comparison": artifact(comparison_path),
                }
            )

    manifest = tmp_path / "rwkv_native_graph_fla_correctness.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": DECODE_CORRECTNESS_PROTOCOL,
                "benchmark_repository_commit": CANDIDATE_COMMIT,
                "model_hashes_sha256": sha256(model_hashes.read_bytes()).hexdigest(),
                "runtime": artifact(runtime_lock),
                "coverage": {
                    "models": 4,
                    "batch_sizes": [1, 8],
                    "entries": 8,
                    "baseline_fresh_gpu_processes": 8,
                    "candidate_additional_gpu_processes": 0,
                    "candidate_formal_lane_processes": 8,
                    "prompt_tokens": 2048,
                    "decode_tokens": 512,
                    "probe_tokens": 512,
                },
                "reference_contract": {
                    "rwkv_implementation": "wrapper_repo",
                    "RWKV7_FAST_TOKEN_BACKEND": "fla",
                    "RWKV7_NATIVE_MODEL_BACKEND": "eager",
                    "RWKV7_FAST_PREFILL": 0,
                    "RWKV7_NATIVE_PREFILL_GRAPH": 0,
                },
                "candidate_contract": {
                    "rwkv_implementation": "auto",
                    "RWKV7_FAST_TOKEN_BACKEND": "native_graph",
                    "RWKV7_NATIVE_MODEL_BACKEND": "native_graph",
                    "RWKV7_FAST_PREFILL": "unset_exact_card_policy",
                    "RWKV7_NATIVE_PREFILL_GRAPH": "unset_exact_card_policy",
                    "small_model_b8_promoted_bundle": True,
                },
                "gates": {
                    "greedy_tokens": "exact_all_512",
                    "prompt_logits_min_row_cosine": 0.9999,
                    "final_logits_min_row_cosine": 0.9999,
                    "decode_logits_all_finite": True,
                    "b8_distinct_prompts": True,
                },
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def write_file_evidence(
    tmp_path: Path, candidates: list[dict]
) -> tuple[Path, Path, Path, Path, Path]:
    ab_manifest, model_hashes = write_ab_evidence(tmp_path, candidates)
    correctness_manifest = write_decode_correctness_evidence(
        tmp_path, candidates, model_hashes
    )
    candidate_path = tmp_path / "rwkv_candidate.jsonl"
    write_jsonl(candidate_path, candidates)
    candidate_digest = sha256(candidate_path.read_bytes()).hexdigest()
    sidecar = tmp_path / "rwkv_candidate.sha256"
    sidecar.write_text(f"{candidate_digest}  {candidate_path.name}\n", encoding="utf-8")
    model_hashes_after = tmp_path / "model_hashes.after.sha256"
    model_hashes_after.write_bytes(model_hashes.read_bytes())
    pip_freeze = tmp_path / "pip-freeze.txt"
    pip_freeze.write_bytes(
        (
            ROOT
            / "bench"
            / "5090_qwen35_best_optimized_hf_v1_20260813"
            / "pip-freeze.txt"
        ).read_bytes()
    )
    system_identity = tmp_path / "system.csv"
    system_identity.write_text(
        "name, uuid, pci.bus_id, compute_cap, driver_version, "
        "memory.total [MiB], power.limit [W]\n"
        "NVIDIA GeForce RTX 5090, GPU-fixture, 00000000:01:00.0, "
        "12.0, 595.58.03, 32607 MiB, 575.00 W\n",
        encoding="utf-8",
    )
    runtime_lock = tmp_path / "runtime-lock.json"

    def artifact(path: Path) -> dict[str, str]:
        return {
            "path": f"/evidence/{path.name}",
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }

    lanes = []
    for pair in PARAMETERS:
        size = RWKV_PAIR_SIZES[pair]
        for batch in (1, 8):
            promoted = batch == 8 and size in {"0.4b", "1.5b"}
            lanes.append(
                {
                    "model_pair": pair,
                    "model_size_label": size,
                    "model_path": f"/models/{size}",
                    "batch_size": batch,
                    "cells": 6,
                    "fresh_process": True,
                    "rwkv_implementation_requested": "auto",
                    "rwkv_implementation_effective": "native_model",
                    "RWKV7_NATIVE_PREFILL_GRAPH": "exact_card_policy",
                    "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM": int(promoted),
                    "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G": int(promoted),
                    "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN": int(promoted),
                    "RWKV7_NATIVE_GRAPH_RKV_POLICY": (
                        "vkwr_auto" if promoted else None
                    ),
                    "RWKV7_BLACKWELL_TORCH_COMPILE": 1 if promoted else None,
                    "compile_cache": ("fresh_unique_directory" if promoted else None),
                }
            )
    route_manifest = tmp_path / "rwkv_candidate_routes.json"
    repository_root = "/repo/rwkv7-hf-adapter"
    route_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "qwen35_paired_decode_v1",
                "benchmark_repository_commit": CANDIDATE_COMMIT,
                "repository_root": repository_root,
                "repository_clean_pre_and_post": True,
                "candidate_rows": 48,
                "qwen_rerun": False,
                "rwkv_implementation_requested": "auto",
                "rwkv_implementation_effective": "native_model",
                "forced_environment": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "PYTHONPATH": repository_root,
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "TORCH_CUDA_ARCH_LIST": "12.0",
                    "TORCHDYNAMO_DISABLE": "0",
                    "TORCH_COMPILE_DISABLE": "0",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "RWKV7_FAST_TOKEN_BACKEND": "native_graph",
                    "RWKV7_NATIVE_MODEL_BACKEND": "native_graph",
                    "RWKV7_NATIVE_PREFILL_GRAPH": "unset_exact_card_policy",
                },
                "candidate_result": artifact(candidate_path),
                "candidate_sha256_sidecar": artifact(sidecar),
                "model_hash_contract": {
                    "algorithm": "sha256",
                    "scope": "every recursive regular file",
                    "before": artifact(model_hashes),
                    "after": artifact(model_hashes_after),
                    "byte_identical": True,
                },
                "sm120_b8_ab_manifest": artifact(ab_manifest),
                "native_graph_fla_correctness_manifest": artifact(correctness_manifest),
                "runtime_lock": artifact(runtime_lock),
                "pip_freeze": artifact(pip_freeze),
                "system_identity": artifact(system_identity),
                "lanes": lanes,
            }
        ),
        encoding="utf-8",
    )
    return (
        ab_manifest,
        correctness_manifest,
        route_manifest,
        runtime_lock,
        model_hashes,
    )


def prepared_file_bundle(tmp_path: Path) -> dict[str, object]:
    references = reference_rows()
    candidates = candidate_rows(references)
    reference = tmp_path / "qwen_reference.jsonl"
    candidate = tmp_path / "rwkv_candidate.jsonl"
    write_jsonl(reference, references)
    ab_manifest, correctness_manifest, route_manifest, runtime_lock, model_hashes = (
        write_file_evidence(tmp_path, candidates)
    )
    return {
        "references": references,
        "candidates": candidates,
        "reference": reference,
        "candidate": candidate,
        "ab_manifest": ab_manifest,
        "correctness_manifest": correctness_manifest,
        "route_manifest": route_manifest,
        "runtime_lock": runtime_lock,
        "model_hashes": model_hashes,
    }


def validate_bundle(bundle: dict[str, object]) -> dict:
    reference = bundle["reference"]
    assert isinstance(reference, Path)
    return validate_files(
        qwen_reference=reference,
        rwkv_candidate=bundle["candidate"],
        sm120_ab_manifest=bundle["ab_manifest"],
        decode_correctness_manifest=bundle["correctness_manifest"],
        candidate_route_manifest=bundle["route_manifest"],
        runtime_lock=bundle["runtime_lock"],
        model_hashes=bundle["model_hashes"],
        expected_reference_sha256=sha256(reference.read_bytes()).hexdigest(),
        expected_device=DEVICE,
    )


def refresh_route_artifact(
    route_manifest: Path, field: str, artifact_path: Path
) -> None:
    route = json.loads(route_manifest.read_text(encoding="utf-8"))
    route[field]["sha256"] = sha256(artifact_path.read_bytes()).hexdigest()
    route_manifest.write_text(json.dumps(route), encoding="utf-8")


def test_complete_matrix_passes_and_is_stably_sorted() -> None:
    references = list(reversed(reference_rows()))
    candidates = list(reversed(candidate_rows(references)))

    summary = validate(candidates, references)

    assert summary["status"] == "pass"
    assert summary["paired_decode_table_eligible"] is True
    assert summary["continuous_e2e_eligible"] is False
    assert summary["gate"]["passing_cells"] == 48
    assert summary["coverage"] == {
        "candidate_rows": 48,
        "reference_rows": 48,
        "joined_cells": 48,
        "expected_cells": 48,
    }
    assert summary["repository_commits"] == {
        "candidate": [CANDIDATE_COMMIT],
        "reference": [REFERENCE_COMMIT],
    }
    keys = [
        (
            PAIR_RANK[cell["model_pair"]],
            cell["device"],
            cell["batch_size"],
            cell["prompt_tokens"],
            cell["decode_tokens"],
        )
        for cell in summary["cells"]
    ]
    assert keys == sorted(keys)
    assert all(cell["adjusted_decode_ratio"] > 1.0 for cell in summary["cells"])


@pytest.mark.parametrize(
    "field",
    (
        "rwkv_native_graph_sm120_wagv_bmm_g_requested",
        "rwkv_native_graph_sm120_wagv_bmm_g_selected",
        "rwkv_native_graph_sm120_wagv_bmm_g_effective",
        "rwkv_native_graph_sm120_wagv_bmm_g_full_model_effective",
        "rwkv_native_graph_sm120_compiled_ffn_requested",
        "rwkv_native_graph_sm120_compiled_ffn_selected",
        "rwkv_native_graph_sm120_compiled_ffn_effective",
        "rwkv_native_graph_sm120_compiled_ffn_full_model_effective",
    ),
)
def test_small_model_b8_requires_exact_sm120_route(field: str) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    row = next(
        item
        for item in candidates
        if item["model_pair"] == "rwkv-0.4b__qwen3.5-0.8b" and item["batch_size"] == 8
    )
    row[field] = False

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any(field in error for error in summary["errors"])


def test_small_model_b8_requires_all_24_sm120_layers() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    row = next(
        item
        for item in candidates
        if item["model_pair"] == "rwkv-1.5b__qwen3.5-2b" and item["batch_size"] == 8
    )
    row["rwkv_native_graph_sm120_wagv_bmm_g_effective_layers"] = list(range(23))
    row["rwkv_native_graph_sm120_wagv_bmm_g_effective_layer_count"] = 23

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any("effective_layers" in error for error in summary["errors"])


def test_small_model_b8_requires_all_24_compiled_ffn_layers() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    row = next(
        item
        for item in candidates
        if item["model_pair"] == "rwkv-0.4b__qwen3.5-0.8b" and item["batch_size"] == 8
    )
    row["rwkv_native_graph_sm120_compiled_ffn_selected_layers"] = list(range(23))
    row["rwkv_native_graph_sm120_compiled_ffn_effective_layer_count"] = 23

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any(
        "sm120_compiled_ffn_selected_layers" in error for error in summary["errors"]
    )
    assert any(
        "sm120_compiled_ffn_effective_layer_count" in error
        for error in summary["errors"]
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("rwkv_native_graph_sm120_compiled_ffn_compile_effective", False),
        ("rwkv_native_graph_sm120_compiled_ffn_compile_reused", False),
        ("rwkv_native_graph_sm120_compiled_ffn_unique_graphs", 2),
        ("rwkv_native_graph_sm120_compiled_ffn_graph_breaks", 1),
        ("rwkv_native_graph_sm120_compiled_ffn_prewarm_all_finite", False),
        ("rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine", 0.9998),
        (
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_argmax_all_equal",
            False,
        ),
    ),
)
def test_small_model_b8_requires_compiled_ffn_proof(
    field: str, bad_value: object
) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    row = next(
        item
        for item in candidates
        if item["model_pair"] == "rwkv-1.5b__qwen3.5-2b" and item["batch_size"] == 8
    )
    row[field] = bad_value

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any(field in error for error in summary["errors"])


def test_non_target_cells_reject_compiled_ffn_evidence_pollution() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    row = next(
        item
        for item in candidates
        if item["model_pair"] == "rwkv-7.2b__qwen3.5-9b" and item["batch_size"] == 8
    )
    row["rwkv_native_graph_sm120_compiled_ffn_compile_effective"] = False

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any("compile_effective" in error for error in summary["errors"])


def test_sm120_route_rejects_bool_as_integer_evidence() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    small_b8 = next(
        item
        for item in candidates
        if item["model_pair"] == "rwkv-0.4b__qwen3.5-0.8b" and item["batch_size"] == 8
    )
    small_b8["rwkv_native_graph_sm120_wagv_bmm_g_effective_layers"][0] = False
    non_target = next(
        item
        for item in candidates
        if item["model_pair"] == "rwkv-2.9b__qwen3.5-4b" and item["batch_size"] == 1
    )
    non_target["rwkv_native_graph_sm120_wagv_bmm_g_effective_layer_count"] = False
    non_target["rwkv_native_graph_sm120_compiled_ffn_effective_layer_count"] = False

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any("effective_layers" in error for error in summary["errors"])
    assert any("effective_layer_count" in error for error in summary["errors"])
    assert any(
        "sm120_compiled_ffn_effective_layer_count" in error
        for error in summary["errors"]
    )


def test_versioned_frozen_reference_digest_matches_checked_in_artifact() -> None:
    reference = (
        ROOT
        / "bench"
        / "5090_qwen35_best_optimized_hf_v1_20260813"
        / "qwen_reference.jsonl"
    )

    assert sha256(reference.read_bytes()).hexdigest() == FROZEN_REFERENCE_SHA256


def test_unrounded_adjusted_ratio_equal_to_one_fails_strict_gate() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    set_adjusted_ratio(candidates[0], references[0], 1.0)
    candidates[0]["decode_tokps_total"] = 999_999.0
    candidates[0]["decode_active_parameter_tops"] = 999_999.0

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert summary["gate"]["passing_cells"] == 47
    assert len(summary["red_cells"]) == 1
    assert summary["red_cells"][0]["adjusted_decode_ratio"] <= 1.0
    assert summary["red_cells"][0]["strict_pass"] is False
    assert summary["red_cells"][0]["candidate_decode_tokps_total_raw"] != 999_999.0


def test_one_raw_cell_below_gate_reports_required_rate_and_gap() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    set_adjusted_ratio(candidates[-1], references[-1], 0.999999)

    summary = validate(candidates, references)
    red = summary["red_cells"][0]

    assert summary["gate"]["passing_cells"] == 47
    assert red["adjusted_decode_ratio"] == pytest.approx(0.999999)
    assert (
        red["required_candidate_decode_tokps"] > red["candidate_decode_tokps_total_raw"]
    )
    assert red["candidate_margin_tokps"] < 0
    assert red["candidate_margin_percent"] == pytest.approx(-0.0001)


@pytest.mark.parametrize(
    ("field", "value", "error_fragment"),
    [
        ("torch_version", "different", "runtime signature"),
        ("device", "NVIDIA GeForce RTX 4090", "device="),
        ("effective_backend", "native_jit", "effective_backend"),
        ("cache_type", "RWKV7StateCache", "cache_type"),
        ("active_parameter_count", 1, "active_parameter_count"),
        ("status", "fail", "status="),
        ("logits_finite", False, "logits_finite"),
    ],
)
def test_candidate_contract_drift_fails(
    field: str, value: object, error_fragment: str
) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    candidates[0][field] = value

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any(error_fragment in error for error in summary["errors"])


@pytest.mark.parametrize("invalid", [0.0, float("nan"), float("inf"), True])
def test_non_positive_or_non_finite_raw_throughput_fails(invalid: object) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    candidates[0]["decode_tokps_total_raw"] = invalid

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any("decode_tokps_total_raw" in error for error in summary["errors"])


def test_missing_d512_duplicate_and_extra_cells_never_get_overwritten() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    without_d512 = [row for row in candidates if row["decode_tokens"] != 512]
    duplicate = deepcopy(without_d512[0])
    duplicate["_source"] = "duplicate.jsonl:1"
    extra = deepcopy(without_d512[1])
    extra["prompt_tokens"] = 4096
    extra["_source"] = "extra.jsonl:1"

    summary = validate([*without_d512, duplicate, extra], references)

    assert summary["status"] == "fail"
    assert summary["coverage"]["joined_cells"] < 48
    assert any("candidate duplicate cells" in error for error in summary["errors"])
    assert any("candidate missing cells" in error for error in summary["errors"])
    assert any("candidate extra cells" in error for error in summary["errors"])


def test_reference_hash_mismatch_or_mid_validation_change_fails() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)

    wrong = validate(candidates, references, before="c" * 64, after="c" * 64)
    changed = validate(candidates, references, after="d" * 64)

    assert wrong["paired_decode_table_eligible"] is False
    assert any("SHA256 mismatch" in error for error in wrong["errors"])
    assert changed["paired_decode_table_eligible"] is False
    assert any("changed while validation" in error for error in changed["errors"])


def test_candidate_repository_commit_must_be_internally_consistent() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    candidates[0]["benchmark_repository_commit"] = "d" * 40

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any("exactly one repository commit" in error for error in summary["errors"])


@pytest.mark.parametrize("role", ["candidate", "reference"])
def test_repository_commit_must_be_exact_40_hex(role: str) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    rows = candidates if role == "candidate" else references
    rows[0]["benchmark_repository_commit"] = "not-a-commit"

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any("40 hexadecimal" in error for error in summary["errors"])


def test_non_small_b8_rejects_base_bmm_route_pollution() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    row = next(
        item
        for item in candidates
        if item["model_pair"] == "rwkv-2.9b__qwen3.5-4b" and item["batch_size"] == 8
    )
    for field, value in (
        ("rwkv_native_graph_ada_wagv_bmm_requested", True),
        ("rwkv_native_graph_ada_wagv_bmm_selected", True),
        ("rwkv_native_graph_ada_wagv_bmm_effective", True),
        ("rwkv_native_graph_ada_wagv_bmm_effective_layer_count", 32),
        ("rwkv_native_graph_ada_wagv_bmm_full_model_effective", True),
    ):
        row[field] = value

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any("rwkv_native_graph_ada_wagv_bmm" in error for error in summary["errors"])


def test_malformed_runtime_type_fails_without_exception() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    candidates[0]["torch_version"] = ["2.8.0+cu128"]

    summary = validate(candidates, references)

    assert summary["status"] == "fail"
    assert any("torch_version" in error for error in summary["errors"])


def test_file_validation_uses_exact_frozen_bytes_and_raw_display_rules(
    tmp_path: Path,
) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    reference = tmp_path / "qwen_reference.jsonl"
    candidate = tmp_path / "rwkv_candidate.jsonl"
    write_jsonl(reference, references)
    ab_manifest, correctness_manifest, route_manifest, runtime_lock, model_hashes = (
        write_file_evidence(tmp_path, candidates)
    )
    write_jsonl(candidate, candidates)
    frozen_hash = sha256(reference.read_bytes()).hexdigest()

    summary = validate_files(
        qwen_reference=reference,
        rwkv_candidate=candidate,
        sm120_ab_manifest=ab_manifest,
        decode_correctness_manifest=correctness_manifest,
        candidate_route_manifest=route_manifest,
        runtime_lock=runtime_lock,
        model_hashes=model_hashes,
        expected_reference_sha256=frozen_hash,
        expected_device=DEVICE,
    )
    markdown = render_markdown(summary)

    assert summary["status"] == "pass"
    assert summary["frozen_reference"]["sha256_before"] == frozen_hash
    assert summary["frozen_reference"]["sha256_after"] == frozen_hash
    assert markdown.count("| PASS |") == 48
    assert "1.100000x" in markdown
    assert "B8 Decode is aggregate throughput across eight sequences" in markdown


def test_file_validation_requires_external_candidate_commit(
    tmp_path: Path,
) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    reference = tmp_path / "qwen_reference.jsonl"
    candidate = tmp_path / "rwkv_candidate.jsonl"
    write_jsonl(reference, references)
    ab_manifest, correctness_manifest, route_manifest, runtime_lock, model_hashes = (
        write_file_evidence(tmp_path, candidates)
    )
    write_jsonl(candidate, candidates)

    summary = validate_files(
        qwen_reference=reference,
        rwkv_candidate=candidate,
        sm120_ab_manifest=ab_manifest,
        decode_correctness_manifest=correctness_manifest,
        candidate_route_manifest=route_manifest,
        runtime_lock=runtime_lock,
        model_hashes=model_hashes,
        expected_candidate_commit="d" * 40,
        expected_reference_sha256=sha256(reference.read_bytes()).hexdigest(),
        expected_device=DEVICE,
    )

    assert summary["status"] == "fail"
    assert any("externally expected" in error for error in summary["errors"])


def test_file_validation_binds_ab_models_and_probes_to_promoted_rows(
    tmp_path: Path,
) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    reference = tmp_path / "qwen_reference.jsonl"
    candidate = tmp_path / "rwkv_candidate.jsonl"
    write_jsonl(reference, references)
    ab_manifest, correctness_manifest, route_manifest, runtime_lock, model_hashes = (
        write_file_evidence(tmp_path, candidates)
    )
    write_jsonl(candidate, candidates)
    manifest = json.loads(ab_manifest.read_text(encoding="utf-8"))
    manifest["entries"][0]["candidate"]["probe"] = manifest["entries"][1]["candidate"][
        "probe"
    ]
    ab_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    summary = validate_files(
        qwen_reference=reference,
        rwkv_candidate=candidate,
        sm120_ab_manifest=ab_manifest,
        decode_correctness_manifest=correctness_manifest,
        candidate_route_manifest=route_manifest,
        runtime_lock=runtime_lock,
        model_hashes=model_hashes,
        expected_reference_sha256=sha256(reference.read_bytes()).hexdigest(),
        expected_device=DEVICE,
    )

    assert summary["status"] == "fail"
    assert any("model_pair" in error for error in summary["errors"])


def test_file_validation_rejects_non_dictionary_probe_payload(
    tmp_path: Path,
) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    reference = tmp_path / "qwen_reference.jsonl"
    candidate = tmp_path / "rwkv_candidate.jsonl"
    write_jsonl(reference, references)
    ab_manifest, correctness_manifest, route_manifest, runtime_lock, model_hashes = (
        write_file_evidence(tmp_path, candidates)
    )
    write_jsonl(candidate, candidates)
    manifest = json.loads(ab_manifest.read_text(encoding="utf-8"))
    probe_entry = manifest["entries"][0]["candidate"]["probe"]
    probe_path = tmp_path / probe_entry["path"]
    torch.save(True, probe_path)
    probe_entry["sha256"] = sha256(probe_path.read_bytes()).hexdigest()
    ab_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    summary = validate_files(
        qwen_reference=reference,
        rwkv_candidate=candidate,
        sm120_ab_manifest=ab_manifest,
        decode_correctness_manifest=correctness_manifest,
        candidate_route_manifest=route_manifest,
        runtime_lock=runtime_lock,
        model_hashes=model_hashes,
        expected_reference_sha256=sha256(reference.read_bytes()).hexdigest(),
        expected_device=DEVICE,
    )

    assert summary["status"] == "fail"
    assert any(
        "probe payload must be a dictionary" in error for error in summary["errors"]
    )


def test_file_validation_requires_ab_candidate_to_be_strictly_faster(
    tmp_path: Path,
) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    reference = tmp_path / "qwen_reference.jsonl"
    candidate = tmp_path / "rwkv_candidate.jsonl"
    write_jsonl(reference, references)
    ab_manifest, correctness_manifest, route_manifest, runtime_lock, model_hashes = (
        write_file_evidence(tmp_path, candidates)
    )
    write_jsonl(candidate, candidates)
    manifest = json.loads(ab_manifest.read_text(encoding="utf-8"))
    entry = manifest["entries"][0]
    baseline_path = tmp_path / entry["baseline"]["row"]["path"]
    candidate_path = tmp_path / entry["candidate"]["row"]["path"]
    baseline_row = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_row = json.loads(candidate_path.read_text(encoding="utf-8"))
    slower_rate = float(baseline_row["decode_tokps_total_raw"]) * 0.99
    slower_seconds = 8 * 512 / slower_rate
    candidate_row["decode_tokps_total_raw"] = slower_rate
    candidate_row["decode_tokps_total"] = round(slower_rate, 3)
    candidate_row["decode_sec_samples"] = [slower_seconds] * 7
    candidate_row["decode_sec_median_raw"] = slower_seconds
    candidate_row["decode_sec_median"] = round(slower_seconds, 6)
    write_jsonl(candidate_path, [candidate_row])
    entry["candidate"]["row"]["sha256"] = sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    ab_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    summary = validate_files(
        qwen_reference=reference,
        rwkv_candidate=candidate,
        sm120_ab_manifest=ab_manifest,
        decode_correctness_manifest=correctness_manifest,
        candidate_route_manifest=route_manifest,
        runtime_lock=runtime_lock,
        model_hashes=model_hashes,
        expected_reference_sha256=sha256(reference.read_bytes()).hexdigest(),
        expected_device=DEVICE,
    )

    assert summary["status"] == "fail"
    assert any("strictly faster" in error for error in summary["errors"])


def test_decode_correctness_requires_all_eight_pair_batch_entries(
    tmp_path: Path,
) -> None:
    bundle = prepared_file_bundle(tmp_path)
    correctness = bundle["correctness_manifest"]
    route = bundle["route_manifest"]
    assert isinstance(correctness, Path) and isinstance(route, Path)
    document = json.loads(correctness.read_text(encoding="utf-8"))
    document["entries"].pop()
    correctness.write_text(json.dumps(document), encoding="utf-8")
    refresh_route_artifact(route, "native_graph_fla_correctness_manifest", correctness)

    summary = validate_bundle(bundle)

    assert summary["status"] == "fail"
    assert any("exactly one entry" in error for error in summary["errors"])


def test_decode_correctness_recomputes_full_512_step_greedy_oracle(
    tmp_path: Path,
) -> None:
    bundle = prepared_file_bundle(tmp_path)
    correctness = bundle["correctness_manifest"]
    route = bundle["route_manifest"]
    assert isinstance(correctness, Path) and isinstance(route, Path)
    document = json.loads(correctness.read_text(encoding="utf-8"))
    probe_evidence = document["entries"][0]["native_candidate"]["probe"]
    probe_path = tmp_path / Path(probe_evidence["path"]).name
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    probe["greedy_tokens"][0] = 1
    torch.save(probe, probe_path)
    probe_evidence["sha256"] = sha256(probe_path.read_bytes()).hexdigest()
    correctness.write_text(json.dumps(document), encoding="utf-8")
    refresh_route_artifact(route, "native_graph_fla_correctness_manifest", correctness)

    summary = validate_bundle(bundle)

    assert summary["status"] == "fail"
    assert any(
        "recomputed native_graph-vs-FLA correctness comparison failed" in error
        for error in summary["errors"]
    )


def test_decode_correctness_rejects_non_distinct_b8_prompts(
    tmp_path: Path,
) -> None:
    bundle = prepared_file_bundle(tmp_path)
    correctness = bundle["correctness_manifest"]
    route = bundle["route_manifest"]
    assert isinstance(correctness, Path) and isinstance(route, Path)
    document = json.loads(correctness.read_text(encoding="utf-8"))
    entry = next(item for item in document["entries"] if item["batch_size"] == 8)
    probe_evidence = entry["native_candidate"]["probe"]
    probe_path = tmp_path / Path(probe_evidence["path"]).name
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    probe["input_ids"][1:] = probe["input_ids"][0]
    torch.save(probe, probe_path)
    probe_evidence["sha256"] = sha256(probe_path.read_bytes()).hexdigest()
    correctness.write_text(json.dumps(document), encoding="utf-8")
    refresh_route_artifact(route, "native_graph_fla_correctness_manifest", correctness)

    summary = validate_bundle(bundle)

    assert summary["status"] == "fail"
    assert any(
        "B8 prompt rows must be distinct" in error for error in summary["errors"]
    )


def test_decode_correctness_native_row_must_bind_production_source_lane(
    tmp_path: Path,
) -> None:
    bundle = prepared_file_bundle(tmp_path)
    correctness = bundle["correctness_manifest"]
    route = bundle["route_manifest"]
    assert isinstance(correctness, Path) and isinstance(route, Path)
    document = json.loads(correctness.read_text(encoding="utf-8"))
    row_evidence = document["entries"][0]["native_candidate"]["row"]
    row_path = tmp_path / Path(row_evidence["path"]).name
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["decode_tokps_total_raw"] *= 1.01
    row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    row_evidence["sha256"] = sha256(row_path.read_bytes()).hexdigest()
    correctness.write_text(json.dumps(document), encoding="utf-8")
    refresh_route_artifact(route, "native_graph_fla_correctness_manifest", correctness)

    summary = validate_bundle(bundle)

    assert summary["status"] == "fail"
    assert any("does not bind" in error for error in summary["errors"])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("dirty", "repository_clean_pre_and_post"),
        ("environment", "forced_environment"),
        ("lane", "exact route contract"),
        ("model_after", "model hash snapshots are not byte-identical"),
        ("sidecar", "exact digest and basename"),
        ("system_gpu", "system identity name"),
        ("system_driver", "system identity driver_version"),
    ],
)
def test_candidate_route_manifest_is_fail_closed(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    bundle = prepared_file_bundle(tmp_path)
    route_path = bundle["route_manifest"]
    assert isinstance(route_path, Path)
    route = json.loads(route_path.read_text(encoding="utf-8"))
    if mutation == "dirty":
        route["repository_clean_pre_and_post"] = False
    elif mutation == "environment":
        route["forced_environment"]["HF_HUB_OFFLINE"] = "0"
    elif mutation == "lane":
        route["lanes"][0]["fresh_process"] = False
    elif mutation == "model_after":
        after_path = tmp_path / Path(route["model_hash_contract"]["after"]["path"]).name
        after_path.write_text("mutated\n", encoding="utf-8")
        route["model_hash_contract"]["after"]["sha256"] = sha256(
            after_path.read_bytes()
        ).hexdigest()
    elif mutation == "sidecar":
        sidecar_path = tmp_path / Path(route["candidate_sha256_sidecar"]["path"]).name
        sidecar_path.write_text("0" * 64 + "  wrong.jsonl\n", encoding="utf-8")
        route["candidate_sha256_sidecar"]["sha256"] = sha256(
            sidecar_path.read_bytes()
        ).hexdigest()
    elif mutation in {"system_gpu", "system_driver"}:
        system_path = tmp_path / Path(route["system_identity"]["path"]).name
        system_text = system_path.read_text(encoding="utf-8")
        if mutation == "system_gpu":
            system_text = system_text.replace(
                "NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 4090"
            )
        else:
            system_text = system_text.replace("595.58.03", "000.00")
        system_path.write_text(system_text, encoding="utf-8")
        route["system_identity"]["sha256"] = sha256(
            system_path.read_bytes()
        ).hexdigest()
    route_path.write_text(json.dumps(route), encoding="utf-8")

    summary = validate_bundle(bundle)

    assert summary["status"] == "fail"
    assert any(expected_error in error for error in summary["errors"])


def test_failed_cli_does_not_emit_promotable_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    set_adjusted_ratio(candidates[0], references[0], 0.99)
    reference = tmp_path / "qwen_reference.jsonl"
    candidate = tmp_path / "rwkv_candidate.jsonl"
    validation = tmp_path / "validation.json"
    paired = tmp_path / "paired.jsonl"
    write_jsonl(reference, references)
    ab_manifest, correctness_manifest, route_manifest, runtime_lock, model_hashes = (
        write_file_evidence(tmp_path, candidates)
    )
    write_jsonl(candidate, candidates)
    paired.write_text("stale-pass\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_qwen35_paired_decode_v1.py",
            "--qwen-reference",
            str(reference),
            "--rwkv-candidate",
            str(candidate),
            "--sm120-ab-manifest",
            str(ab_manifest),
            "--decode-correctness-manifest",
            str(correctness_manifest),
            "--candidate-route-manifest",
            str(route_manifest),
            "--runtime-lock",
            str(runtime_lock),
            "--model-hashes",
            str(model_hashes),
            "--expected-candidate-commit",
            CANDIDATE_COMMIT,
            "--validation",
            str(validation),
            "--paired-table",
            str(paired),
        ],
    )

    assert main() == 1
    assert validation.exists()
    assert not paired.exists()
    report = json.loads(validation.read_text(encoding="utf-8"))
    assert report["gate"]["passing_cells"] == 47


def test_cli_rejects_input_output_overlap_without_touching_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    reference = tmp_path / "qwen_reference.jsonl"
    candidate = tmp_path / "rwkv_candidate.jsonl"
    write_jsonl(reference, references)
    write_jsonl(candidate, candidates)
    ab_manifest = tmp_path / "ab.json"
    correctness_manifest = tmp_path / "correctness.json"
    route_manifest = tmp_path / "routes.json"
    runtime_lock = tmp_path / "runtime-lock.json"
    model_hashes = tmp_path / "model_hashes.sha256"
    original = reference.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_qwen35_paired_decode_v1.py",
            "--qwen-reference",
            str(reference),
            "--rwkv-candidate",
            str(candidate),
            "--sm120-ab-manifest",
            str(ab_manifest),
            "--decode-correctness-manifest",
            str(correctness_manifest),
            "--candidate-route-manifest",
            str(route_manifest),
            "--runtime-lock",
            str(runtime_lock),
            "--model-hashes",
            str(model_hashes),
            "--expected-candidate-commit",
            CANDIDATE_COMMIT,
            "--validation",
            str(reference),
            "--paired-table",
            str(tmp_path / "paired.jsonl"),
        ],
    )

    assert main() == 1
    assert reference.read_bytes() == original


def test_cli_output_collision_removes_stale_promotable_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    reference = tmp_path / "qwen_reference.jsonl"
    candidate = tmp_path / "rwkv_candidate.jsonl"
    paired = tmp_path / "paired.jsonl"
    write_jsonl(reference, references)
    write_jsonl(candidate, candidates)
    ab_manifest = tmp_path / "ab.json"
    correctness_manifest = tmp_path / "correctness.json"
    route_manifest = tmp_path / "routes.json"
    runtime_lock = tmp_path / "runtime-lock.json"
    model_hashes = tmp_path / "model_hashes.sha256"
    paired.write_text("stale-pass\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_qwen35_paired_decode_v1.py",
            "--qwen-reference",
            str(reference),
            "--rwkv-candidate",
            str(candidate),
            "--sm120-ab-manifest",
            str(ab_manifest),
            "--decode-correctness-manifest",
            str(correctness_manifest),
            "--candidate-route-manifest",
            str(route_manifest),
            "--runtime-lock",
            str(runtime_lock),
            "--model-hashes",
            str(model_hashes),
            "--expected-candidate-commit",
            CANDIDATE_COMMIT,
            "--validation",
            str(paired),
            "--paired-table",
            str(paired),
        ],
    )

    assert main() == 1
    assert not paired.exists()


def test_rendering_keeps_sub_100_tokps_at_one_decimal() -> None:
    references = reference_rows()
    candidates = candidate_rows(references)
    reference = references[-1]
    candidate = candidates[-1]
    reference_rate = 99.94
    reference_s = (
        int(reference["batch_size"]) * int(reference["decode_tokens"]) / reference_rate
    )
    reference["decode_sec_samples"] = [reference_s] * 7
    reference["decode_sec_median"] = round(reference_s, 6)
    reference["decode_sec_median_raw"] = reference_s
    reference["decode_tokps_total"] = round(reference_rate, 3)
    reference["decode_tokps_total_raw"] = reference_rate
    set_adjusted_ratio(candidate, reference, 1.1)

    summary = validate(candidates, references)
    markdown = render_markdown(summary)

    assert summary["status"] == "pass"
    assert "| 99.9 |" in markdown
    assert all(
        math.isfinite(cell["adjusted_decode_ratio"]) for cell in summary["cells"]
    )
