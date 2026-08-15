#!/usr/bin/env python3
"""Validate strict RTX 4090 RWKV/Qwen raw and parameter-adjusted P/D parity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch

try:
    from bench.compare_rwkv_prefill_probe import compare as compare_probes
    from bench.validate_qwen35_best_optimized_hf_v1 import (
        EXPECTED_SHAPES,
        RUNTIME_FIELDS,
        validate_matrix as validate_qwen,
    )
except ModuleNotFoundError:
    from compare_rwkv_prefill_probe import compare as compare_probes
    from validate_qwen35_best_optimized_hf_v1 import (
        EXPECTED_SHAPES,
        RUNTIME_FIELDS,
        validate_matrix as validate_qwen,
    )


PROTOCOL = "qwen35_4090_paired_pd_v2"
EXPECTED_DEVICE = "NVIDIA GeForce RTX 4090"
REFERENCE_SHA256 = "7274b4ba3c549320740a4ea3bf7d72ce4dcafb1a671e6ab01e4fa1c1ba1db24f"
CORRECTNESS_PROTOCOL = "rwkv_native_graph_fla_correctness_4090_v2"
QWEN_CONTRACT = "official_fla_causal_conv1d_static_cache_cudagraph_same_cache_4090_v2"
QWEN_CROSS_CACHE_FULL_GREEDY_POLICY = "strict"
EXPECTED_RUNTIME = {
    "python": "3.12.8",
    "torch": "2.7.1+cu126",
    "torch_cuda": "12.6",
    "triton": "3.3.1",
    "transformers": "5.12.1",
    "fla": "0.5.1",
    "causal_conv1d": "1.6.2.post1",
}
EXPECTED_ARCH = "8.9"
EXPECTED_DRIVER = "550.142"
EXPECTED_MEMORY = "24564 MiB"
SPECIAL_SMALL_B8_BUNDLE = True
BASE_ADA_WAGV_BMM_EXPECTED: bool | None = None
REPORT_TITLE = "RTX 4090 strict RWKV/Qwen Prefill+Decode v2"
PAIRS = (
    "rwkv-0.4b__qwen3.5-0.8b",
    "rwkv-1.5b__qwen3.5-2b",
    "rwkv-2.9b__qwen3.5-4b",
    "rwkv-7.2b__qwen3.5-9b",
)
PAIR_RANK = {pair: index for index, pair in enumerate(PAIRS)}
PARAMETERS = {
    PAIRS[0]: (450_767_872, 752_393_024),
    PAIRS[1]: (1_527_404_544, 1_881_825_088),
    PAIRS[2]: (2_947_735_040, 4_205_751_296),
    PAIRS[3]: (7_199_141_888, 8_953_803_264),
}
SIZES = {PAIRS[0]: "0.4b", PAIRS[1]: "1.5b", PAIRS[2]: "2.9b", PAIRS[3]: "7.2b"}
LAYERS = {PAIRS[0]: 24, PAIRS[1]: 24, PAIRS[2]: 32, PAIRS[3]: 32}
QWEN_ROUTES = {
    PAIRS[0]: "static_cache_inductor_cudagraph",
    PAIRS[1]: "static_cache_inductor_cudagraph",
    PAIRS[2]: "static_cache_raw_cudagraph",
    PAIRS[3]: "static_cache_raw_cudagraph",
}
EXPECTED_KEYS = {
    (pair, batch, prompt, decode)
    for pair in PAIRS
    for batch, prompt, decode in EXPECTED_SHAPES
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _strict(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _require(row: dict[str, Any], field: str, expected: Any, errors: list[str]) -> None:
    if not _strict(row.get(field), expected):
        errors.append(
            f"{row.get('_source', '<row>')}: {field}={row.get(field)!r}, expected {expected!r}"
        )


def _read_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    data = path.read_bytes()
    rows = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        row["_source"] = f"{path}:{line_number}"
        rows.append(row)
    return data, rows


def _key(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        row.get("model_pair"),
        row.get("batch_size"),
        row.get("prompt_tokens"),
        row.get("decode_tokens"),
    )


def _validate_samples(
    row: dict[str, Any], axis: str, tokens: int, errors: list[str]
) -> None:
    samples = row.get(f"{axis}_sec_samples")
    if (
        type(samples) is not list
        or len(samples) != 7
        or not all(_finite(x) and x > 0 for x in samples)
    ):
        errors.append(
            f"{row.get('_source')}: {axis} must contain seven positive finite samples"
        )
        return
    median = statistics.median(samples)
    if row.get(f"{axis}_sec_median_raw") != median:
        errors.append(f"{row.get('_source')}: {axis} raw median mismatch")
    tokps = row.get(f"{axis}_tokps_total_raw")
    expected = tokens / median
    if not _finite(tokps) or not math.isclose(
        float(tokps), expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        errors.append(f"{row.get('_source')}: {axis} raw tok/s arithmetic mismatch")


def _validate_candidate(row: dict[str, Any], errors: list[str]) -> None:
    for field, expected in (
        ("axis", "qwen35_cross_model_speed"),
        ("benchmark_matrix", PROTOCOL),
        ("optimization_lane", "best_optimized_hf"),
        ("model_role", "candidate"),
        ("model_kind", "rwkv"),
        ("rwkv_implementation_requested", "auto"),
        ("rwkv_implementation_effective", "native_model"),
        ("dtype", "fp16"),
        ("quantization", "none"),
        ("quantization_backend", "dense"),
        ("native_quant_kernel_active", False),
        ("device", EXPECTED_DEVICE),
        ("gpu_arch", "sm_89"),
        ("gpu_compute_capability", [8, 9]),
        ("prefill_chunk_size", 512),
        ("warmup", 3),
        ("runs", 7),
        ("timing_statistic", "median"),
        ("resident_sweep", True),
        ("status", "pass"),
        ("logits_finite", True),
        ("rwkv_fast_token_backend_requested", "native_graph"),
        ("rwkv_native_model_backend_requested", "native_graph"),
        ("effective_backend", "native_graph"),
        ("step_backend", "rwkv_fast_token"),
        ("cache_type", "NativeRWKV7Cache"),
    ):
        _require(row, field, expected, errors)
    pair = row.get("model_pair")
    if pair not in PAIRS:
        errors.append(f"{row.get('_source')}: unexpected model pair")
        return
    _require(row, "model_size_label", SIZES[pair], errors)
    _require(row, "active_parameter_count", PARAMETERS[pair][0], errors)
    batch = row.get("batch_size")
    small_b8 = SPECIAL_SMALL_B8_BUNDLE and pair in PAIRS[:2] and batch == 8
    layers = list(range(LAYERS[pair])) if small_b8 else []
    if BASE_ADA_WAGV_BMM_EXPECTED is not None:
        for suffix, expected in (
            ("requested", BASE_ADA_WAGV_BMM_EXPECTED),
            ("selected", BASE_ADA_WAGV_BMM_EXPECTED),
            ("effective", BASE_ADA_WAGV_BMM_EXPECTED),
            ("selected_layers", []),
            ("effective_layers", []),
            ("effective_layer_count", 0),
            ("full_model_effective", False),
        ):
            _require(row, f"rwkv_native_graph_ada_wagv_bmm_{suffix}", expected, errors)
    for route in ("sm120_wagv_bmm_g", "sm120_compiled_ffn"):
        for suffix, expected in (
            ("requested", small_b8),
            ("selected", small_b8),
            ("effective", small_b8),
            ("selected_layers", layers),
            ("effective_layers", layers),
            ("effective_layer_count", len(layers)),
            ("full_model_effective", small_b8),
        ):
            _require(row, f"rwkv_native_graph_{route}_{suffix}", expected, errors)
    if small_b8:
        for field, expected in (
            ("rwkv_native_graph_rkv_policy", "vkwr_auto"),
            ("rwkv_native_graph_sm120_compiled_ffn_compile_effective", True),
            ("rwkv_native_graph_sm120_compiled_ffn_compile_reused", True),
            ("rwkv_native_graph_sm120_compiled_ffn_unique_graphs", 1),
            ("rwkv_native_graph_sm120_compiled_ffn_graph_breaks", 0),
            (
                "rwkv_native_graph_sm120_compiled_ffn_compile_mode",
                "max-autotune-no-cudagraphs",
            ),
            ("rwkv_native_graph_sm120_compiled_ffn_prewarm_all_finite", True),
            ("rwkv_native_graph_sm120_compiled_ffn_prewarm_argmax_all_equal", True),
        ):
            _require(row, field, expected, errors)
        cosine = row.get("rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine")
        if not _finite(cosine) or cosine < 0.9999:
            errors.append(f"{row.get('_source')}: compiled FFN prewarm cosine failed")
    shape = (row.get("batch_size"), row.get("prompt_tokens"), row.get("decode_tokens"))
    if shape not in EXPECTED_SHAPES:
        errors.append(f"{row.get('_source')}: unexpected shape {shape!r}")
        return
    _validate_samples(row, "prefill", int(shape[0]) * int(shape[1]), errors)
    _validate_samples(row, "decode", int(shape[0]) * int(shape[2]), errors)
    commit = row.get("benchmark_repository_commit")
    if (
        type(commit) is not str
        or len(commit) != 40
        or any(c not in "0123456789abcdef" for c in commit.lower())
    ):
        errors.append(f"{row.get('_source')}: invalid repository commit")


def _index(
    rows: list[dict[str, Any]], label: str, errors: list[str]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    keys = [_key(row) for row in rows]
    counts = Counter(keys)
    if len(rows) != 48:
        errors.append(f"{label} has {len(rows)} rows, expected 48")
    if any(count != 1 for count in counts.values()) or set(keys) != EXPECTED_KEYS:
        errors.append(f"{label} does not contain exactly the 48 expected unique cells")
    return {
        key: row
        for key, row in zip(keys, rows, strict=True)
        if counts[key] == 1 and key in EXPECTED_KEYS
    }


def _validate_correctness(
    path: Path, candidate_commit: str, errors: list[str]
) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        errors.append("correctness manifest must be an object")
        return {"status": "fail"}
    if (
        doc.get("protocol") != CORRECTNESS_PROTOCOL
        or doc.get("benchmark_repository_commit") != candidate_commit
    ):
        errors.append("correctness manifest protocol/commit mismatch")
    entries = doc.get("entries")
    if type(entries) is not list or len(entries) != 8:
        errors.append("correctness manifest must contain eight entries")
        return {"status": "fail"}
    root = path.parent
    seen = set()
    min_cosine = 1.0
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("correctness entry must be an object")
            continue
        key = (entry.get("model_pair"), entry.get("batch_size"))
        seen.add(key)
        if (
            entry.get("status") != "pass"
            or entry.get("prompt_tokens") != 2048
            or entry.get("decode_tokens") != 512
            or entry.get("probe_tokens") != 512
        ):
            errors.append(f"correctness entry {key!r} has invalid scope/status")
        loaded = {}
        for name in ("fla_probe", "native_probe", "comparison"):
            evidence = entry.get(name)
            if (
                not isinstance(evidence, dict)
                or type(evidence.get("path")) is not str
                or type(evidence.get("sha256")) is not str
            ):
                errors.append(f"correctness entry {key!r} has invalid {name} evidence")
                continue
            artifact = root / Path(evidence["path"]).name
            if (
                not artifact.is_file()
                or _sha256(artifact.read_bytes()) != evidence["sha256"]
            ):
                errors.append(f"correctness entry {key!r} {name} hash mismatch")
                continue
            loaded[name] = artifact
        if set(loaded) != {"fla_probe", "native_probe", "comparison"}:
            continue
        reference = torch.load(
            loaded["fla_probe"], map_location="cpu", weights_only=True
        )
        native = torch.load(
            loaded["native_probe"], map_location="cpu", weights_only=True
        )
        if not isinstance(reference, dict) or not isinstance(native, dict):
            errors.append(f"correctness entry {key!r} probe is not a dictionary")
            continue
        recomputed = compare_probes(reference, native, 0.9999)
        recorded = json.loads(loaded["comparison"].read_text(encoding="utf-8"))
        for result, label in ((recomputed, "recomputed"), (recorded, "recorded")):
            if (
                result.get("status") != "pass"
                or result.get("greedy_tokens_match") is not True
                or result.get("input_ids_match") is not True
            ):
                errors.append(f"correctness entry {key!r} {label} comparison failed")
            for axis in ("prompt_logits", "final_logits"):
                cosine = result.get(f"{axis}_cosine")
                if not _finite(cosine) or cosine < 0.9999:
                    errors.append(
                        f"correctness entry {key!r} {label} {axis} cosine failed"
                    )
                elif label == "recomputed":
                    min_cosine = min(min_cosine, float(cosine))
                if (
                    result.get(f"{axis}_shape_match") is not True
                    or result.get(f"{axis}_finite") is not True
                ):
                    errors.append(
                        f"correctness entry {key!r} {label} {axis} shape/finite failed"
                    )
        for probe, label in ((reference, "FLA"), (native, "native")):
            if probe.get("decode_logits_all_finite") is not True:
                errors.append(
                    f"correctness entry {key!r} {label} decode was not all finite"
                )
            inputs = probe.get("input_ids")
            greedy = probe.get("greedy_tokens")
            batch = int(key[1])
            if not isinstance(inputs, torch.Tensor) or tuple(inputs.shape) != (
                batch,
                2048,
            ):
                errors.append(f"correctness entry {key!r} {label} input shape mismatch")
            if not isinstance(greedy, torch.Tensor) or greedy.numel() != batch * 512:
                errors.append(
                    f"correctness entry {key!r} {label} greedy horizon mismatch"
                )
            if (
                batch == 8
                and isinstance(inputs, torch.Tensor)
                and torch.unique(inputs, dim=0).shape[0] != 8
            ):
                errors.append(
                    f"correctness entry {key!r} {label} prompts are not distinct"
                )
    if seen != {(pair, batch) for pair in PAIRS for batch in (1, 8)}:
        errors.append("correctness manifest coverage mismatch")
    return {
        "status": "pass" if not errors else "fail",
        "entries": len(entries),
        "min_logits_cosine": min_cosine,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    candidate_bytes, candidate_rows = _read_jsonl(args.candidate)
    reference_bytes, reference_rows = _read_jsonl(args.reference)
    reference_sha = _sha256(reference_bytes)
    if reference_sha != REFERENCE_SHA256:
        errors.append(f"reference sha256={reference_sha}, expected {REFERENCE_SHA256}")
    qwen_result = validate_qwen(
        reference_rows,
        expected_device=EXPECTED_DEVICE,
        expected_pairs=PAIRS,
        expected_routes_by_pair=QWEN_ROUTES,
        expected_matrix="qwen35_best_optimized_hf_v1",
        expected_lane="qwen_best_optimized_hf",
        expected_conv_backend="causal_conv1d",
        expected_causal_conv1d_importable=True,
        expected_fast_path_available=True,
        qwen_contract=QWEN_CONTRACT,
        expected_cross_cache_full_greedy_policy=QWEN_CROSS_CACHE_FULL_GREEDY_POLICY,
    )
    if qwen_result.get("status") != "pass":
        errors.extend(f"Qwen reference: {x}" for x in qwen_result.get("errors", []))
    for row in candidate_rows:
        _validate_candidate(row, errors)
    candidate_index = _index(candidate_rows, "candidate", errors)
    reference_index = _index(reference_rows, "reference", errors)
    candidate_runtime = {
        tuple(row.get(field) for field in RUNTIME_FIELDS) for row in candidate_rows
    }
    reference_runtime = {
        tuple(row.get(field) for field in RUNTIME_FIELDS) for row in reference_rows
    }
    if len(candidate_runtime) != 1 or candidate_runtime != reference_runtime:
        errors.append("candidate/reference runtime signatures are not one exact match")
    commits = {row.get("benchmark_repository_commit") for row in candidate_rows}
    candidate_commit = next(iter(commits)) if len(commits) == 1 else ""
    if len(commits) != 1:
        errors.append("candidate rows do not have one repository commit")
    cells = []
    for key in sorted(EXPECTED_KEYS, key=lambda x: (PAIR_RANK[x[0]], x[1], x[2], x[3])):
        candidate = candidate_index.get(key)
        reference = reference_index.get(key)
        if candidate is None or reference is None:
            continue
        pair, batch, prompt, decode = key
        parameter_ratio = PARAMETERS[pair][0] / PARAMETERS[pair][1]
        values = {}
        gates = []
        for axis in ("prefill", "decode"):
            c = candidate.get(f"{axis}_tokps_total_raw")
            r = reference.get(f"{axis}_tokps_total_raw")
            if not (_finite(c) and c > 0 and _finite(r) and r > 0):
                errors.append(f"cell {key!r}: invalid {axis} throughput")
                continue
            raw = float(c) / float(r)
            adjusted = raw * parameter_ratio
            values.update(
                {
                    f"candidate_{axis}_tokps_total_raw": float(c),
                    f"reference_{axis}_tokps_total_raw": float(r),
                    f"raw_{axis}_ratio": raw,
                    f"adjusted_{axis}_ratio": adjusted,
                    f"required_candidate_{axis}_tokps": float(r) / parameter_ratio,
                }
            )
            gates.extend((raw > 1.0, adjusted > 1.0))
        cells.append(
            {
                "model_pair": pair,
                "device": EXPECTED_DEVICE,
                "batch_size": batch,
                "prompt_tokens": prompt,
                "decode_tokens": decode,
                "active_parameter_ratio": parameter_ratio,
                **values,
                "strict_pass": len(gates) == 4 and all(gates),
            }
        )
    if len(cells) != 48 or not all(cell["strict_pass"] for cell in cells):
        errors.append(
            "all 48 cells must strictly pass raw and adjusted Prefill and Decode"
        )
    runtime_doc = json.loads(args.runtime_lock.read_text(encoding="utf-8"))
    if (
        runtime_doc.get("runtime") != EXPECTED_RUNTIME
        or runtime_doc.get("repository_commit") != candidate_commit
        or runtime_doc.get("torch_cuda_arch_list") != EXPECTED_ARCH
    ):
        errors.append("runtime lock mismatch")
    if args.model_hashes.read_bytes() != args.model_hashes_after.read_bytes():
        errors.append("model hashes changed during capture")
    with args.system.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    if (
        len(rows) != 1
        or rows[0].get("name") != EXPECTED_DEVICE
        or rows[0].get("compute_cap") != EXPECTED_ARCH
        or rows[0].get("driver_version") != EXPECTED_DRIVER
        or rows[0].get("memory.total [MiB]") != EXPECTED_MEMORY
    ):
        errors.append("system identity is not the frozen RTX 4090 host")
    correctness_errors: list[str] = []
    correctness = _validate_correctness(
        args.correctness_manifest, str(candidate_commit), correctness_errors
    )
    errors.extend(f"correctness: {x}" for x in correctness_errors)
    metrics = {}
    for field in (
        "raw_prefill_ratio",
        "adjusted_prefill_ratio",
        "raw_decode_ratio",
        "adjusted_decode_ratio",
    ):
        vals = [cell[field] for cell in cells if field in cell]
        metrics[field] = (
            {
                "min": min(vals),
                "median": statistics.median(vals),
                "max": max(vals),
                "passed": sum(v > 1.0 for v in vals),
            }
            if vals
            else None
        )
    eligible = not errors and len(cells) == 48
    return {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "status": "pass" if eligible else "fail",
        "paired_pd_table_eligible": eligible,
        "candidate_sha256": _sha256(candidate_bytes),
        "reference_sha256": reference_sha,
        "repository_commits": {
            "candidate": candidate_commit,
            "reference": sorted(
                {row.get("benchmark_repository_commit") for row in reference_rows}
            ),
        },
        "metrics": metrics,
        "correctness": correctness,
        "cells": cells,
        "errors": errors,
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        f"# {REPORT_TITLE}",
        "",
        f"Status: **{summary['status'].upper()}**",
        "",
    ]
    for name, value in summary.get("metrics", {}).items():
        if value:
            lines.append(
                f"- {name}: min `{value['min']:.6f}x`, median `{value['median']:.6f}x`, pass `{value['passed']}/48`"
            )
    lines.extend(
        ["", "All gates use unrounded raw throughput and require strict `> 1.0`.", ""]
    )
    if summary.get("errors"):
        lines.extend(["## Errors", ""] + [f"- {x}" for x in summary["errors"]] + [""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--correctness-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--model-hashes", type=Path, required=True)
    parser.add_argument("--model-hashes-after", type=Path, required=True)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paired-table", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate(args)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(summary), encoding="utf-8")
    if summary["paired_pd_table_eligible"]:
        args.paired_table.write_text(
            "".join(
                json.dumps(cell, separators=(",", ":")) + "\n"
                for cell in summary["cells"]
            ),
            encoding="utf-8",
        )
    elif args.paired_table.exists():
        args.paired_table.unlink()
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
