#!/usr/bin/env python3
"""Authenticate the complete RTX 4080 strict paired Prefill/Decode bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import torch

try:
    from bench.compare_rwkv_prefill_probe import compare
    from bench.validate_qwen35_paired_pd_v1 import (
        EXPECTED_DEVICE,
        PAIRS,
        PROTOCOL,
        QWEN_ROUTES,
        RWKV_SIZES,
        _read_jsonl_bytes,
        render_markdown,
        validate_paired_pd,
    )
except ModuleNotFoundError:
    from compare_rwkv_prefill_probe import compare
    from validate_qwen35_paired_pd_v1 import (
        EXPECTED_DEVICE,
        PAIRS,
        PROTOCOL,
        QWEN_ROUTES,
        RWKV_SIZES,
        _read_jsonl_bytes,
        render_markdown,
        validate_paired_pd,
    )


CORRECTNESS_PROTOCOL = "rwkv_native_graph_fla_correctness_4080_v1"
QWEN_ROUTE_PROTOCOL = "qwen35_best_optimized_hf_4080_v1"
EXPECTED_RUNTIME = {
    "python": "3.12.2",
    "torch": "2.11.0+cu130",
    "torch_cuda": "13.0",
    "triton": "3.6.0",
    "transformers": "5.12.1",
    "fla": "0.5.1",
    "causal_conv1d": "1.6.2.post1",
}
ROW_RUNTIME = {
    "torch_version": EXPECTED_RUNTIME["torch"],
    "torch_cuda_version": EXPECTED_RUNTIME["torch_cuda"],
    "triton_version": EXPECTED_RUNTIME["triton"],
    "transformers_version": EXPECTED_RUNTIME["transformers"],
    "fla_version": EXPECTED_RUNTIME["fla"],
    "causal_conv1d_version": EXPECTED_RUNTIME["causal_conv1d"],
}
EXPECTED_QWEN_ROUTES = {
    PAIRS[0]: ("0.8b", "static_cache_inductor_cudagraph", "max-autotune"),
    PAIRS[1]: ("2b", "static_cache_inductor_cudagraph", "max-autotune"),
    PAIRS[2]: ("4b", "module_call_dynamic", None),
}
assert {pair: route for pair, (_, route, _) in EXPECTED_QWEN_ROUTES.items()} == QWEN_ROUTES
HEX_DIGITS = frozenset("0123456789abcdef")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"_source", "_line_number"}
    }


def _recorded_path(value: Any) -> str | None:
    if type(value) is not str or not value.strip():
        return None
    return value.replace("\\", "/").rstrip("/")


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(right, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _strict_equal(left[key], value) for key, value in right.items()
        )
    return bool(left == right)


def _expect(source: str, actual: Any, expected: Any, errors: list[str]) -> None:
    if not _strict_equal(actual, expected):
        errors.append(f"{source}: observed {actual!r}, expected {expected!r}")


def _load_json(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{label} could not be read: {exc}")
        return {}
    if type(value) is not dict:
        errors.append(f"{label} must be a JSON object")
        return {}
    return value


def _artifact(
    manifest: Path,
    value: Any,
    label: str,
    errors: list[str],
    *,
    expected: Path | None = None,
) -> Path | None:
    if type(value) is not dict:
        errors.append(f"{label} must be an artifact object")
        return None
    recorded = _recorded_path(value.get("path"))
    digest = value.get("sha256")
    if recorded is None or not recorded.startswith("/") or "/../" in recorded:
        errors.append(f"{label}.path must be one absolute normalized POSIX path")
        return None
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in HEX_DIGITS for character in digest.lower())
    ):
        errors.append(f"{label}.sha256 must be 64 hexadecimal characters")
        return None
    basename = recorded.rsplit("/", 1)[-1]
    path = expected if expected is not None else manifest.parent / basename
    if path.name != basename:
        errors.append(f"{label} basename {basename!r} does not bind {path.name!r}")
    try:
        observed = _sha256(path)
    except OSError as exc:
        errors.append(f"{label} could not be read: {exc}")
        return None
    if observed != digest.lower():
        errors.append(f"{label} SHA256 mismatch: {observed} != {digest.lower()}")
    return path


def _parse_hashes(path: Path, errors: list[str]) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current: str | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"model hash manifest could not be read: {exc}")
        return sections
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = _recorded_path(line[1:-1])
            if current is None or not current.startswith("/") or current in sections:
                errors.append(f"{path}:{line_number}: invalid model section")
                current = None
            else:
                sections[current] = {}
            continue
        if current is None:
            errors.append(f"{path}:{line_number}: hash outside model section")
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"{path}:{line_number}: malformed hash entry")
            continue
        digest, filename = parts
        filename = filename.strip().replace("\\", "/")
        relative = Path(filename)
        if (
            len(digest) != 64
            or any(character not in HEX_DIGITS for character in digest.lower())
            or not filename
            or relative.is_absolute()
            or ".." in relative.parts
            or filename in sections[current]
        ):
            errors.append(f"{path}:{line_number}: invalid hash entry")
            continue
        sections[current][filename] = digest.lower()
    for model, files in sections.items():
        if "config.json" not in files or not any(
            name.endswith(".safetensors") for name in files
        ):
            errors.append(f"{path}: {model!r} lacks config/weights hashes")
        if not any("tokenizer" in name or name.endswith("vocab.txt") for name in files):
            errors.append(f"{path}: {model!r} lacks tokenizer evidence")
    return sections


def _load_probe(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"{path}: invalid probe: {exc}")
        return None
    if type(value) is not dict:
        errors.append(f"{path}: probe must be a dictionary")
        return None
    return value


def _validate_probe(
    probe: dict[str, Any],
    path: Path,
    *,
    pair: str,
    size: str,
    model_path: str,
    commit: str,
    batch: int,
    errors: list[str],
) -> None:
    for field, expected in (
        ("probe_schema_version", 2),
        ("benchmark_repository_commit", commit),
        ("model_pair", pair),
        ("model_size_label", size),
        ("model_id_or_path", model_path),
        ("decode_logits_all_finite", True),
    ):
        _expect(f"{path}:{field}", probe.get(field), expected, errors)
    recorded = _recorded_path(probe.get("probe_output"))
    if recorded is None or recorded.rsplit("/", 1)[-1] != path.name:
        errors.append(f"{path}: probe_output does not bind the probe")
    input_ids = probe.get("input_ids")
    greedy = probe.get("greedy_tokens")
    finite = probe.get("decode_logits_finite_by_batch")
    greedy_shape = (512,) if batch == 1 else (512, batch)
    if not isinstance(input_ids, torch.Tensor) or tuple(input_ids.shape) != (
        batch,
        2048,
    ):
        errors.append(f"{path}: input_ids must be [{batch}, 2048]")
    elif batch == 8 and int(torch.unique(input_ids, dim=0).shape[0]) != 8:
        errors.append(f"{path}: B8 prompts are not distinct")
    if not isinstance(greedy, torch.Tensor) or tuple(greedy.shape) != greedy_shape:
        errors.append(f"{path}: greedy_tokens must be {greedy_shape}")
    if (
        not isinstance(finite, torch.Tensor)
        or tuple(finite.shape) != (batch,)
        or not bool(finite.bool().all())
    ):
        errors.append(f"{path}: all 512 Decode steps must be finite")
    for label in ("prompt_logits", "final_logits"):
        logits = probe.get(label)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.dim() != 2
            or logits.shape[0] != batch
            or logits.numel() == 0
            or not bool(torch.isfinite(logits).all())
        ):
            errors.append(f"{path}: {label} must be finite [{batch}, vocab]")


def _validate_row_probe_binding(
    row: dict[str, Any],
    probe: dict[str, Any],
    path: Path,
    batch: int,
    errors: list[str],
) -> None:
    recorded = _recorded_path(row.get("probe_output"))
    if recorded is None or recorded.rsplit("/", 1)[-1] != path.name:
        errors.append(f"{path}: result row probe_output does not bind the probe")
    for field, expected in (
        ("probe_tokens", 512),
        ("probe_batch_size", batch),
        ("probe_distinct_batch_prompts", batch == 8),
        ("probe_decode_logits_all_finite", True),
        ("probe_decode_logits_finite_by_batch", [True] * batch),
    ):
        _expect(f"{path}:row {field}", row.get(field), expected, errors)
    greedy = probe.get("greedy_tokens")
    expected_greedy = greedy.tolist() if isinstance(greedy, torch.Tensor) else None
    _expect(
        f"{path}:row probe_greedy_tokens",
        row.get("probe_greedy_tokens"),
        expected_greedy,
        errors,
    )


def _validate_runtime_and_system(
    runtime_path: Path,
    system_path: Path,
    pip_path: Path,
    commit: str,
    errors: list[str],
) -> None:
    runtime = _load_json(runtime_path, "runtime lock", errors)
    for field, expected in (
        ("schema_version", 1),
        ("protocol", PROTOCOL),
        ("repository_commit", commit),
        ("runtime", EXPECTED_RUNTIME),
        ("torch_cuda_arch_list", "8.9"),
        ("pip_freeze_sha256", _sha256(pip_path)),
    ):
        _expect(f"runtime-lock:{field}", runtime.get(field), expected, errors)
    try:
        with system_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        errors.append(f"system.csv could not be read: {exc}")
        return
    if len(rows) != 1:
        errors.append("system.csv must contain exactly one GPU")
        return
    row = {str(key).strip(): str(value).strip() for key, value in rows[0].items()}
    for field, expected in (
        ("name", EXPECTED_DEVICE),
        ("compute_cap", "8.9"),
        ("driver_version", "595.71.05"),
        ("memory.total [MiB]", "16376 MiB"),
    ):
        _expect(f"system.csv:{field}", row.get(field), expected, errors)
    for field in ("uuid", "pci.bus_id"):
        if not row.get(field):
            errors.append(f"system.csv:{field} must be non-empty")


def _validate_qwen_routes(
    manifests: list[Path], reference_rows: list[dict[str, Any]], errors: list[str]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if len(manifests) != 3:
        errors.append("exactly three Qwen route manifests are required")
        return summaries
    reference_by_pair = {
        pair: [
            _row_payload(row) for row in reference_rows if row.get("model_pair") == pair
        ]
        for pair in PAIRS
    }
    seen_pairs: set[str] = set()
    manifest_pairs: list[str] = []
    cache_roots: set[str] = set()
    for manifest_path in manifests:
        doc = _load_json(manifest_path, "Qwen route manifest", errors)
        pair = doc.get("model_pair")
        if pair not in EXPECTED_QWEN_ROUTES or pair in seen_pairs:
            errors.append(f"{manifest_path}: unexpected/duplicate model_pair={pair!r}")
            continue
        seen_pairs.add(pair)
        manifest_pairs.append(pair)
        size, route, compile_mode = EXPECTED_QWEN_ROUTES[pair]
        commits = {
            row.get("benchmark_repository_commit")
            for row in reference_rows
            if row.get("model_pair") == pair
        }
        commit = next(iter(commits)) if len(commits) == 1 else ""
        for field, expected in (
            ("schema_version", 1),
            ("protocol", QWEN_ROUTE_PROTOCOL),
            ("benchmark_repository_commit", commit),
            ("repository_clean_pre_and_post", True),
            ("model_size_label", size),
            ("decode_route", route),
            ("compile_mode", compile_mode),
        ):
            _expect(f"{manifest_path}:{field}", doc.get(field), expected, errors)
        model_path = _recorded_path(doc.get("model_path"))
        row_paths = {
            _recorded_path(row.get("model_id_or_path"))
            for row in reference_rows
            if row.get("model_pair") == pair
        }
        _expect(f"{manifest_path}:model_path", row_paths, {model_path}, errors)
        result = _artifact(manifest_path, doc.get("result"), "Qwen result", errors)
        if result is not None:
            try:
                rows = _read_jsonl_bytes(result.read_bytes(), result)
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                errors.append(f"{result}: invalid Qwen rows: {exc}")
            else:
                _expect(
                    f"{result}:rows",
                    [_row_payload(row) for row in rows],
                    reference_by_pair[pair],
                    errors,
                )
        hashes = doc.get("model_hash_contract")
        if type(hashes) is not dict:
            errors.append(f"{manifest_path}: model_hash_contract must be an object")
            hashes = {}
        before = _artifact(
            manifest_path, hashes.get("before"), "Qwen hashes before", errors
        )
        after = _artifact(
            manifest_path, hashes.get("after"), "Qwen hashes after", errors
        )
        if before is not None and after is not None:
            _expect(
                f"{manifest_path}:hash bytes",
                before.read_bytes(),
                after.read_bytes(),
                errors,
            )
            sections = _parse_hashes(before, errors)
            _expect(f"{manifest_path}:hash model", set(sections), {model_path}, errors)
        _expect(
            f"{manifest_path}:hash algorithm", hashes.get("algorithm"), "sha256", errors
        )
        _expect(
            f"{manifest_path}:hash scope",
            hashes.get("scope"),
            "every recursive regular file",
            errors,
        )
        _expect(
            f"{manifest_path}:hash stable", hashes.get("byte_identical"), True, errors
        )
        forced = doc.get("forced_environment")
        if type(forced) is not dict:
            errors.append(f"{manifest_path}: forced_environment must be an object")
            forced = {}
        expected_forced = {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONPATH": forced.get("PYTHONPATH"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CACHE_ROOT": forced.get("CACHE_ROOT"),
        }
        _expect(f"{manifest_path}:forced_environment", forced, expected_forced, errors)
        for name in ("PYTHONPATH", "CACHE_ROOT"):
            value = _recorded_path(forced.get(name))
            if value is None or not value.startswith("/"):
                errors.append(f"{manifest_path}: {name} must be absolute")
        cache = str(forced.get("CACHE_ROOT"))
        if cache in cache_roots:
            errors.append("Qwen CACHE_ROOT values must be unique per model")
        cache_roots.add(cache)
        summaries.append(
            {"model_pair": pair, "route": route, "compile_mode": compile_mode}
        )
    _expect("Qwen route coverage", seen_pairs, set(PAIRS), errors)
    _expect("Qwen route order", manifest_pairs, list(PAIRS), errors)
    return summaries


def _validate_correctness(
    manifest_path: Path,
    model_hashes_path: Path,
    runtime_path: Path,
    candidate_rows: list[dict[str, Any]],
    errors: list[str],
) -> list[dict[str, Any]]:
    doc = _load_json(manifest_path, "correctness manifest", errors)
    commits = {row.get("benchmark_repository_commit") for row in candidate_rows}
    commit = next(iter(commits)) if len(commits) == 1 else ""
    for field, expected in (
        ("schema_version", 1),
        ("protocol", CORRECTNESS_PROTOCOL),
        ("benchmark_repository_commit", commit),
        ("model_hashes_sha256", _sha256(model_hashes_path)),
        (
            "coverage",
            {
                "models": 3,
                "batch_sizes": [1, 8],
                "entries": 6,
                "baseline_fresh_gpu_processes": 6,
                "candidate_additional_gpu_processes": 0,
                "candidate_formal_lane_processes": 6,
                "prompt_tokens": 2048,
                "decode_tokens": 512,
                "probe_tokens": 512,
            },
        ),
        (
            "reference_contract",
            {
                "rwkv_implementation": "wrapper_repo",
                "RWKV7_FAST_TOKEN_BACKEND": "fla",
                "RWKV7_NATIVE_MODEL_BACKEND": "eager",
                "RWKV7_FAST_PREFILL": "0",
                "RWKV7_NATIVE_PREFILL_GRAPH": "0",
                "TORCHDYNAMO_DISABLE": "1",
                "TORCH_COMPILE_DISABLE": "1",
                "performance_role": False,
            },
        ),
        (
            "gates",
            {
                "greedy_tokens": "exact_all_512",
                "prompt_logits_min_row_cosine": 0.9999,
                "final_logits_min_row_cosine": 0.9999,
                "decode_logits_all_finite": True,
                "b8_distinct_prompts": True,
            },
        ),
    ):
        _expect(f"{manifest_path}:{field}", doc.get(field), expected, errors)
    _artifact(
        manifest_path,
        doc.get("runtime"),
        "correctness runtime",
        errors,
        expected=runtime_path,
    )
    model_sections = _parse_hashes(model_hashes_path, errors)
    candidate_index = {
        (
            row.get("model_pair"),
            row.get("batch_size"),
            row.get("prompt_tokens"),
            row.get("decode_tokens"),
        ): row
        for row in candidate_rows
    }
    expected_keys = [(pair, batch) for pair in PAIRS for batch in (1, 8)]
    entries = doc.get("entries")
    if type(entries) is not list:
        errors.append("correctness entries must be a list")
        entries = []
    actual_keys = [
        (entry.get("model_pair"), entry.get("batch_size"))
        for entry in entries
        if type(entry) is dict
    ]
    _expect("correctness entry order", actual_keys, expected_keys, errors)
    summaries: list[dict[str, Any]] = []
    for index, (pair, batch) in enumerate(expected_keys):
        if index >= len(entries) or type(entries[index]) is not dict:
            continue
        entry = entries[index]
        size, _layers = RWKV_SIZES[pair]
        main = candidate_index.get((pair, batch, 2048, 512))
        if main is None:
            errors.append(f"missing promoted correctness cell {pair} B{batch}")
            continue
        model_path = _recorded_path(main.get("model_id_or_path"))
        for field, expected in (
            ("model_pair", pair),
            ("model_size_label", size),
            ("model_path", model_path),
            ("batch_size", batch),
            ("prompt_tokens", 2048),
            ("decode_tokens", 512),
            ("probe_tokens", 512),
        ):
            _expect(
                f"correctness {pair} B{batch}:{field}",
                entry.get(field),
                expected,
                errors,
            )
        if model_path not in model_sections:
            errors.append(
                f"correctness {pair} B{batch}: model path has no hash section"
            )
        fla = (
            entry.get("fla_reference")
            if type(entry.get("fla_reference")) is dict
            else {}
        )
        native = (
            entry.get("native_candidate")
            if type(entry.get("native_candidate")) is dict
            else {}
        )
        _expect(
            f"correctness {pair} B{batch}:source_cell",
            native.get("source_cell"),
            {"batch_size": batch, "prompt_tokens": 2048, "decode_tokens": 512},
            errors,
        )
        fla_row_path = _artifact(manifest_path, fla.get("row"), "FLA row", errors)
        fla_probe_path = _artifact(manifest_path, fla.get("probe"), "FLA probe", errors)
        native_row_path = _artifact(
            manifest_path, native.get("row"), "native row", errors
        )
        native_probe_path = _artifact(
            manifest_path, native.get("probe"), "native probe", errors
        )
        lane_path = _artifact(
            manifest_path, native.get("source_lane"), "native source lane", errors
        )
        comparison_path = _artifact(
            manifest_path, entry.get("comparison"), "comparison", errors
        )
        native_row: dict[str, Any] | None = None
        if native_row_path is not None:
            rows = _read_jsonl_bytes(native_row_path.read_bytes(), native_row_path)
            if len(rows) != 1:
                errors.append(f"{native_row_path}: expected one row")
            else:
                native_row = rows[0]
                _expect(
                    f"{native_row_path}:main binding",
                    _row_payload(native_row),
                    _row_payload(main),
                    errors,
                )
        if lane_path is not None:
            lane_rows = _read_jsonl_bytes(lane_path.read_bytes(), lane_path)
            _expect(f"{lane_path}:row count", len(lane_rows), 6, errors)
            matches = [
                row
                for row in lane_rows
                if row.get("prompt_tokens") == 2048 and row.get("decode_tokens") == 512
            ]
            if (
                len(matches) != 1
                or native_row is None
                or _row_payload(matches[0]) != _row_payload(native_row)
            ):
                errors.append(f"{lane_path}: promoted row binding failed")
        fla_row: dict[str, Any] | None = None
        if fla_row_path is not None:
            rows = _read_jsonl_bytes(fla_row_path.read_bytes(), fla_row_path)
            if len(rows) != 1:
                errors.append(f"{fla_row_path}: expected one row")
            else:
                fla_row = rows[0]
                for field, expected in (
                    ("benchmark_matrix", CORRECTNESS_PROTOCOL),
                    ("optimization_lane", "fla_reference"),
                    ("benchmark_repository_commit", commit),
                    ("model_pair", pair),
                    ("model_size_label", size),
                    ("model_id_or_path", model_path),
                    ("batch_size", batch),
                    ("prompt_tokens", 2048),
                    ("decode_tokens", 512),
                    ("warmup", 1),
                    ("runs", 1),
                    ("rwkv_implementation_requested", "wrapper_repo"),
                    ("rwkv_implementation_effective", "wrapper_repo"),
                    ("effective_backend", "fla"),
                    ("cache_type", "RWKV7StateCache"),
                    ("device", EXPECTED_DEVICE),
                    ("gpu_arch", "sm_89"),
                    ("gpu_compute_capability", [8, 9]),
                    ("status", "pass"),
                    ("logits_finite", True),
                ):
                    _expect(
                        f"{fla_row_path}:{field}", fla_row.get(field), expected, errors
                    )
                for field, expected in ROW_RUNTIME.items():
                    _expect(
                        f"{fla_row_path}:{field}", fla_row.get(field), expected, errors
                    )
        reference_probe = (
            _load_probe(fla_probe_path, errors) if fla_probe_path else None
        )
        candidate_probe = (
            _load_probe(native_probe_path, errors) if native_probe_path else None
        )
        if reference_probe is not None and candidate_probe is not None:
            for probe, path in (
                (reference_probe, fla_probe_path),
                (candidate_probe, native_probe_path),
            ):
                _validate_probe(
                    probe,
                    path,
                    pair=pair,
                    size=size,
                    model_path=model_path or "",
                    commit=commit,
                    batch=batch,
                    errors=errors,
                )
            if fla_row is not None:
                _validate_row_probe_binding(
                    fla_row, reference_probe, fla_probe_path, batch, errors
                )
            if native_row is not None:
                _validate_row_probe_binding(
                    native_row, candidate_probe, native_probe_path, batch, errors
                )
            recomputed = compare(reference_probe, candidate_probe, 0.9999)
            recomputed["contract_errors"] = []
            if batch == 8 and recomputed.get("distinct_batch_prompts") is not True:
                recomputed["contract_errors"].append(
                    "batch prompts are not all distinct"
                )
                recomputed["status"] = "fail"
            _expect(
                f"correctness {pair} B{batch}:status",
                recomputed.get("status"),
                "pass",
                errors,
            )
            if comparison_path is not None:
                recorded = _load_json(comparison_path, "comparison", errors)
                _expect(f"{comparison_path}:recomputed", recorded, recomputed, errors)
            summaries.append(
                {
                    "model_pair": pair,
                    "batch_size": batch,
                    "prompt_min_row_cosine": recomputed.get("prompt_logits_cosine"),
                    "final_min_row_cosine": recomputed.get("final_logits_cosine"),
                    "greedy_tokens_match": recomputed.get("greedy_tokens_match"),
                }
            )
    return summaries


def validate_bundle(
    *,
    candidate: Path,
    reference: Path,
    candidate_route_manifest: Path,
    correctness_manifest: Path,
    runtime_lock: Path,
    candidate_model_hashes: Path,
    qwen_route_manifests: list[Path],
    expected_candidate_commit: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    candidate_bytes = candidate.read_bytes()
    reference_bytes = reference.read_bytes()
    candidate_rows = _read_jsonl_bytes(candidate_bytes, candidate)
    reference_rows = _read_jsonl_bytes(reference_bytes, reference)
    summary = validate_paired_pd(
        candidate_rows,
        reference_rows,
        candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
        reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
    )
    errors.extend(summary.get("errors", []))
    commits = {
        str(row.get("benchmark_repository_commit", "")).lower()
        for row in candidate_rows
    }
    commit = next(iter(commits)) if len(commits) == 1 else ""
    if expected_candidate_commit is not None:
        normalized = expected_candidate_commit.lower()
        if len(normalized) != 40 or any(
            value not in HEX_DIGITS for value in normalized
        ):
            errors.append(
                "expected candidate commit must be exactly 40 hexadecimal characters"
            )
        elif commit != normalized:
            errors.append("candidate rows do not match externally expected commit")
    route = _load_json(candidate_route_manifest, "candidate route manifest", errors)
    for field, expected in (
        ("schema_version", 1),
        ("protocol", PROTOCOL),
        ("benchmark_repository_commit", commit),
        ("repository_clean_pre_and_post", True),
        ("candidate_rows", 36),
    ):
        _expect(f"candidate route:{field}", route.get(field), expected, errors)
    candidate_artifact = _artifact(
        candidate_route_manifest,
        route.get("candidate_result"),
        "candidate result",
        errors,
        expected=candidate,
    )
    sidecar = _artifact(
        candidate_route_manifest,
        route.get("candidate_sha256_sidecar"),
        "candidate sidecar",
        errors,
    )
    if candidate_artifact is not None and sidecar is not None:
        expected_line = f"{_sha256(candidate)}  {candidate.name}\n"
        try:
            _expect(
                "candidate sidecar content",
                sidecar.read_text(encoding="utf-8"),
                expected_line,
                errors,
            )
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"candidate sidecar could not be read: {exc}")
    model_contract = (
        route.get("model_hash_contract")
        if type(route.get("model_hash_contract")) is dict
        else {}
    )
    before = _artifact(
        candidate_route_manifest,
        model_contract.get("before"),
        "candidate hashes before",
        errors,
        expected=candidate_model_hashes,
    )
    after = _artifact(
        candidate_route_manifest,
        model_contract.get("after"),
        "candidate hashes after",
        errors,
    )
    for field, expected in (
        ("algorithm", "sha256"),
        ("scope", "every recursive regular file"),
        ("byte_identical", True),
    ):
        _expect(
            f"candidate hashes:{field}", model_contract.get(field), expected, errors
        )
    if before is not None and after is not None:
        _expect(
            "candidate model hashes before/after",
            before.read_bytes(),
            after.read_bytes(),
            errors,
        )
    sections = _parse_hashes(candidate_model_hashes, errors)
    row_model_paths = {
        _recorded_path(row.get("model_id_or_path")) for row in candidate_rows
    }
    _expect("candidate model hash coverage", set(sections), row_model_paths, errors)
    correctness_bound = _artifact(
        candidate_route_manifest,
        route.get("native_graph_fla_correctness_manifest"),
        "correctness manifest",
        errors,
        expected=correctness_manifest,
    )
    runtime_bound = _artifact(
        candidate_route_manifest,
        route.get("runtime_lock"),
        "runtime lock",
        errors,
        expected=runtime_lock,
    )
    pip_path = _artifact(
        candidate_route_manifest, route.get("pip_freeze"), "pip freeze", errors
    )
    system_path = _artifact(
        candidate_route_manifest,
        route.get("system_identity"),
        "system identity",
        errors,
    )
    if runtime_bound is not None and pip_path is not None and system_path is not None:
        _validate_runtime_and_system(
            runtime_bound, system_path, pip_path, commit, errors
        )
    forced = route.get("forced_environment")
    if type(forced) is not dict:
        errors.append("candidate forced_environment must be an object")
        forced = {}
    expected_forced = {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONPATH": forced.get("PYTHONPATH"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "CPATH": forced.get("CPATH"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "RWKV7_FAST_TOKEN_BACKEND": "native_graph",
        "RWKV7_NATIVE_MODEL_BACKEND": "native_graph",
        "RWKV7_NATIVE_PREFILL_GRAPH": "unset_exact_card_policy",
        "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION": "1_for_B1_0_for_B8",
        "RWKV7_NATIVE_GRAPH_RKV_POLICY": "vkwr_auto_except_2p9_B8_manual",
        "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM": "1",
        "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G": "0",
        "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN": "0",
        "CACHE_ROOT": forced.get("CACHE_ROOT"),
    }
    _expect("candidate forced_environment", forced, expected_forced, errors)
    for field in ("PYTHONPATH", "CPATH", "CACHE_ROOT"):
        value = _recorded_path(forced.get(field))
        if value is None or not value.startswith("/"):
            errors.append(f"candidate {field} must be absolute")
    lanes = route.get("lanes")
    expected_lane_keys = [(pair, batch) for pair in PAIRS for batch in (1, 8)]
    if type(lanes) is not list:
        errors.append("candidate lanes must be a list")
        lanes = []
    _expect(
        "candidate lane order",
        [
            (lane.get("model_pair"), lane.get("batch_size"))
            for lane in lanes
            if type(lane) is dict
        ],
        expected_lane_keys,
        errors,
    )
    lane_rows: list[dict[str, Any]] = []
    for index, key in enumerate(expected_lane_keys):
        if index >= len(lanes) or type(lanes[index]) is not dict:
            continue
        lane = lanes[index]
        _expect(f"candidate lane {key}:rows", lane.get("rows"), 6, errors)
        _expect(
            f"candidate lane {key}:probe",
            lane.get("probe_cell"),
            [key[1], 2048, 512],
            errors,
        )
        _expect(
            f"candidate lane {key}:extension",
            lane.get("ada_wagv_lora_require_extension"),
            key[1] == 1,
            errors,
        )
        expected_rkv = "manual" if key == (PAIRS[2], 8) else "vkwr_auto"
        _expect(
            f"candidate lane {key}:rkv_policy",
            lane.get("rkv_policy"),
            expected_rkv,
            errors,
        )
        path = _artifact(
            candidate_route_manifest,
            lane.get("artifact"),
            f"candidate lane {key}",
            errors,
        )
        if path is not None:
            rows = _read_jsonl_bytes(path.read_bytes(), path)
            lane_rows.extend(_row_payload(row) for row in rows)
    _expect(
        "candidate lane concatenation",
        lane_rows,
        [_row_payload(row) for row in candidate_rows],
        errors,
    )
    correctness_summary = _validate_correctness(
        correctness_bound or correctness_manifest,
        candidate_model_hashes,
        runtime_bound or runtime_lock,
        candidate_rows,
        errors,
    )
    qwen_summary = _validate_qwen_routes(qwen_route_manifests, reference_rows, errors)
    summary["core_errors"] = summary.get("errors", [])
    summary["errors"] = errors
    summary["evidence"] = {
        "candidate_route_manifest": str(candidate_route_manifest),
        "correctness_manifest": str(correctness_manifest),
        "correctness_entries": correctness_summary,
        "qwen_routes": qwen_summary,
        "runtime_lock": str(runtime_lock),
        "candidate_model_hashes": str(candidate_model_hashes),
    }
    eligible = bool(summary.get("paired_pd_table_eligible") and not errors)
    summary["status"] = "pass" if eligible else "fail"
    summary["paired_pd_table_eligible"] = eligible
    summary["bundle_authenticated"] = eligible
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-route-manifest", type=Path, required=True)
    parser.add_argument("--correctness-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--candidate-model-hashes", type=Path, required=True)
    parser.add_argument(
        "--qwen-route-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--expected-candidate-commit")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paired-table", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = {
        path.resolve(strict=False)
        for path in (
            args.candidate,
            args.reference,
            args.candidate_route_manifest,
            args.correctness_manifest,
            args.runtime_lock,
            args.candidate_model_hashes,
            *args.qwen_route_manifest,
        )
    }
    output_paths = [
        args.summary.resolve(strict=False),
        args.paired_table.resolve(strict=False),
        args.markdown.resolve(strict=False),
    ]
    collision_errors: list[str] = []
    if len(set(output_paths)) != len(output_paths):
        collision_errors.append("summary, paired table, and markdown paths must differ")
    if any(path in input_paths for path in output_paths):
        collision_errors.append("output paths must not overwrite any evidence input")
    if collision_errors:
        if args.summary.resolve(strict=False) not in input_paths:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol": PROTOCOL,
                        "status": "fail",
                        "paired_pd_table_eligible": False,
                        "bundle_authenticated": False,
                        "cells": [],
                        "errors": collision_errors,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if (
            args.paired_table.resolve(strict=False) not in input_paths
            and args.paired_table.exists()
        ):
            args.paired_table.unlink()
        return 1
    try:
        summary = validate_bundle(
            candidate=args.candidate,
            reference=args.reference,
            candidate_route_manifest=args.candidate_route_manifest,
            correctness_manifest=args.correctness_manifest,
            runtime_lock=args.runtime_lock,
            candidate_model_hashes=args.candidate_model_hashes,
            qwen_route_manifests=args.qwen_route_manifest,
            expected_candidate_commit=args.expected_candidate_commit,
        )
    except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError) as exc:
        summary = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "fail",
            "paired_pd_table_eligible": False,
            "bundle_authenticated": False,
            "cells": [],
            "errors": [f"bundle validation failed closed: {exc}"],
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(summary), encoding="utf-8")
    if summary.get("paired_pd_table_eligible") is True:
        args.paired_table.write_text(
            "".join(json.dumps(cell) + "\n" for cell in summary["cells"]),
            encoding="utf-8",
        )
    elif args.paired_table.exists():
        args.paired_table.unlink()
    return 0 if summary.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
