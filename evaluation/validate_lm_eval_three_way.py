#!/usr/bin/env python3
"""Validate reference/optimized/FLA formal lm_eval bundles as one matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LANES = ("reference", "optimized", "fla")
ALLOWED_OPTIMIZED_ROUTES = {
    "native-triton-rank1-scan-v1",
    "torch-cuda-graph-reference-v1",
}
ALLOWED_MODEL_ROUTE_PREFIXES = (
    "native-nvidia-prefill-v2[",
    "native-nvidia-fused-decode-v2[",
)
EXPECTED_FLA_COMMIT = "80e494f6c588e091fc8316b612870df29375c5b8"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for lane in LANES:
        parser.add_argument(f"--{lane}-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-model-routes",
        action="store_true",
        help="require backend-v2 whole-model execution in every optimized unit",
    )
    return parser.parse_args()


def latest_manifest(root: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["unit"]] = row
    return latest


def artifact_provenance(root: Path) -> dict[str, Any]:
    path = root / "artifacts.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact provenance: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def result_json(unit_dir: Path) -> Path:
    rows = [path for path in unit_dir.rglob("results_*.json") if path.is_file()]
    if not rows:
        raise FileNotFoundError(f"missing result JSON in {unit_dir}")
    return max(rows, key=lambda path: path.stat().st_mtime)


def sample_jsonl(unit_dir: Path, task: str) -> Path:
    rows = list(unit_dir.rglob(f"samples_{task}_*.jsonl"))
    if not rows:
        raise FileNotFoundError(f"missing samples for {unit_dir.name}/{task}")
    return max(rows, key=lambda path: path.stat().st_mtime)


def numeric_metrics(payload: dict[str, Any], task: str) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in payload["results"][task].items()
        if isinstance(value, (int, float)) and not key.endswith("_stderr")
    }


def _response_pair(value: Any) -> tuple[float, bool]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"expected a loglikelihood response pair, got {value!r}")
    score = float(value[0])
    raw_greedy = value[1]
    if isinstance(raw_greedy, str):
        greedy = raw_greedy.strip().lower() in {"1", "true", "yes"}
    else:
        greedy = bool(raw_greedy)
    return score, greedy


def _choice_lengths(sample: dict[str, Any], count: int) -> list[int]:
    arguments = sample.get("arguments", {})
    rows = []
    for index in range(count):
        value = arguments.get(f"gen_args_{index}", {}).get("arg_1", "")
        # lm_eval's multiple-choice normalization uses the visible choice
        # length. Requests commonly prepend one formatting space, so remove
        # only leading whitespace introduced at the request boundary.
        rows.append(max(1, len(str(value).lstrip())))
    return rows


def sample_outcomes(path: Path, task: str) -> dict[str, tuple[Any, ...]]:
    """Extract predictions, not merely per-sample correctness booleans."""

    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            if task == "wikitext":
                filtered = sample.get("filtered_resps", ())
                if not filtered:
                    raise ValueError("wikitext sample has no rolling loglikelihood")
                score = float(filtered[0])
                word_metric = sample.get("word_perplexity", (score, 0))
                byte_metric = sample.get("byte_perplexity", (score, 0))
                outcome = (
                    "rolling",
                    -score,
                    int(word_metric[1]),
                    int(byte_metric[1]),
                )
            else:
                filtered = sample.get("filtered_resps", ())
                if not filtered:
                    raise ValueError(f"{task} sample has no filtered responses")
                pairs = [_response_pair(value) for value in filtered]
                if len(pairs) == 1:
                    # LAMBADA has one continuation; its discrete prediction is
                    # whether every continuation token was greedily selected.
                    outcome = ("loglikelihood", pairs[0][1])
                else:
                    scores = [value[0] for value in pairs]
                    lengths = _choice_lengths(sample, len(scores))
                    raw_choice = max(range(len(scores)), key=scores.__getitem__)
                    normalized = [
                        score / length for score, length in zip(scores, lengths)
                    ]
                    normalized_choice = max(
                        range(len(normalized)), key=normalized.__getitem__
                    )
                    outcome = (
                        "multiple_choice",
                        int(raw_choice),
                        int(normalized_choice),
                    )
            rows[sample["doc_hash"]] = outcome
    return rows


def compare_sample_outcomes(
    reference: dict[str, tuple[Any, ...]],
    candidate: dict[str, tuple[Any, ...]],
    task: str,
) -> dict[str, Any]:
    common = reference.keys() & candidate.keys()
    missing_docs = len(reference.keys() ^ candidate.keys())
    if task != "wikitext":
        mismatches = sum(reference[key] != candidate[key] for key in common)
        return {
            "prediction_mismatches": int(mismatches),
            "continuous_mismatches": 0,
            "max_relative_nll_difference": 0.0,
            "missing_docs": int(missing_docs),
        }
    continuous_mismatches = 0
    maximum = 0.0
    for key in common:
        left = reference[key]
        right = candidate[key]
        if left[0] != "rolling" or right[0] != "rolling":
            continuous_mismatches += 1
            continue
        if left[2:] != right[2:]:
            continuous_mismatches += 1
        difference = relative_difference(float(left[1]), float(right[1]))
        maximum = max(maximum, difference)
        if difference > 0.001:
            continuous_mismatches += 1
    return {
        "prediction_mismatches": 0,
        "continuous_mismatches": int(continuous_mismatches),
        "max_relative_nll_difference": float(maximum),
        "missing_docs": int(missing_docs),
    }


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), 1e-12)


def main() -> int:
    args = arguments()
    roots = {lane: getattr(args, f"{lane}_dir") for lane in LANES}
    manifests = {lane: latest_manifest(root) for lane, root in roots.items()}
    failures: list[dict[str, Any]] = []
    artifacts = {lane: artifact_provenance(root) for lane, root in roots.items()}
    wheel_hashes = {
        lane: {
            name: row.get("sha256") for name, row in payload.get("wheels", {}).items()
        }
        for lane, payload in artifacts.items()
    }
    expected_wheels = wheel_hashes["reference"]
    if set(expected_wheels) != {"rwkv7_hf", "rwkv7_kernels"} or not all(
        expected_wheels.values()
    ):
        failures.append({"reason": "reference lane lacks both immutable wheel hashes"})
    for lane, hashes in wheel_hashes.items():
        if hashes != expected_wheels:
            failures.append(
                {
                    "lane": lane,
                    "reason": "wheel hashes differ from reference",
                    "expected": expected_wheels,
                    "actual": hashes,
                }
            )
        fla = artifacts[lane].get("fla") or {}
        if fla.get("commit") != EXPECTED_FLA_COMMIT:
            failures.append(
                {
                    "lane": lane,
                    "reason": f"FLA commit is not pinned to {EXPECTED_FLA_COMMIT}",
                    "actual": fla.get("commit"),
                }
            )

    for lane, rows in manifests.items():
        if len(rows) != 48:
            failures.append(
                {"lane": lane, "reason": f"expected 48 units, got {len(rows)}"}
            )
        for unit, row in rows.items():
            if row.get("exit_code") != 0:
                failures.append({"lane": lane, "unit": unit, "reason": "nonzero exit"})
            if not row.get("formal"):
                failures.append({"lane": lane, "unit": unit, "reason": "limited run"})
            if row.get("lane", lane) != lane:
                failures.append({"lane": lane, "unit": unit, "reason": "lane mismatch"})
            if not row.get("task_provenance", {}).get("dataset_fingerprint"):
                failures.append(
                    {"lane": lane, "unit": unit, "reason": "missing provenance"}
                )
            if lane == "optimized":
                trace = row.get("kernel_route_trace", {})
                counts = trace.get("actual_recurrent_calls", {})
                unknown = set(counts) - ALLOWED_OPTIMIZED_ROUTES
                if unknown:
                    failures.append(
                        {
                            "lane": lane,
                            "unit": unit,
                            "reason": f"unknown routes {sorted(unknown)}",
                        }
                    )
                model_counts = trace.get("actual_model_calls", {})
                unknown_model = [
                    name
                    for name in model_counts
                    if not str(name).startswith(ALLOWED_MODEL_ROUTE_PREFIXES)
                ]
                if unknown_model:
                    failures.append(
                        {
                            "lane": lane,
                            "unit": unit,
                            "reason": f"unknown model routes {sorted(unknown_model)}",
                        }
                    )
                if args.require_model_routes and not model_counts:
                    failures.append(
                        {
                            "lane": lane,
                            "unit": unit,
                            "reason": "missing backend-v2 actual model route",
                        }
                    )
                elif not model_counts and not counts:
                    failures.append(
                        {"lane": lane, "unit": unit, "reason": "missing actual route"}
                    )

    common_units = set.intersection(*(set(rows) for rows in manifests.values()))
    if len(common_units) != 48:
        failures.append({"reason": f"three lanes share only {len(common_units)} units"})

    aggregates: dict[tuple[str, str], dict[str, float]] = {}
    samples: dict[tuple[str, str], dict[str, tuple[Any, ...]]] = {}
    for lane in LANES:
        root = roots[lane]
        for unit, row in manifests[lane].items():
            if row.get("exit_code") != 0:
                continue
            payload = json.loads(result_json(root / unit).read_text(encoding="utf-8"))
            metrics = numeric_metrics(payload, row["task"])
            if any(not math.isfinite(value) for value in metrics.values()):
                failures.append(
                    {"lane": lane, "unit": unit, "reason": "non-finite metric"}
                )
            aggregates[(lane, unit)] = metrics
            samples[(lane, unit)] = sample_outcomes(
                sample_jsonl(root / unit, row["task"]), row["task"]
            )

    comparisons = []
    for unit in sorted(common_units):
        task = manifests["reference"][unit]["task"]
        baseline_metrics = aggregates.get(("reference", unit), {})
        baseline_samples = samples.get(("reference", unit), {})
        for lane in ("optimized", "fla"):
            candidate_metrics = aggregates.get((lane, unit), {})
            metric_failures = []
            for name in baseline_metrics.keys() & candidate_metrics.keys():
                left, right = baseline_metrics[name], candidate_metrics[name]
                if task == "wikitext":
                    if relative_difference(left, right) > 0.001:
                        metric_failures.append((name, left, right))
                elif left != right:
                    metric_failures.append((name, left, right))
            candidate_samples = samples.get((lane, unit), {})
            sample_comparison = compare_sample_outcomes(
                baseline_samples, candidate_samples, task
            )
            row = {
                "unit": unit,
                "candidate_lane": lane,
                "task": task,
                "metric_failures": metric_failures,
                **sample_comparison,
            }
            comparisons.append(row)
            if (
                metric_failures
                or sample_comparison["prediction_mismatches"]
                or sample_comparison["continuous_mismatches"]
                or sample_comparison["missing_docs"]
            ):
                failures.append(row)

    # Batch-size invariance inside each lane.
    for lane in LANES:
        rows = manifests[lane]
        grouped: dict[tuple[str, str], dict[int, str]] = {}
        for unit, row in rows.items():
            grouped.setdefault((row["model_label"], row["task"]), {})[
                int(row["batch_size"])
            ] = unit
        for (model, task), units in grouped.items():
            if set(units) != {1, 8}:
                failures.append(
                    {
                        "lane": lane,
                        "model": model,
                        "task": task,
                        "reason": "missing batch",
                    }
                )
                continue
            left = aggregates.get((lane, units[1]), {})
            right = aggregates.get((lane, units[8]), {})
            for name in left.keys() & right.keys():
                if task == "wikitext":
                    failed = relative_difference(left[name], right[name]) > 0.001
                else:
                    failed = abs(left[name] - right[name]) > 0.001
                if failed:
                    failures.append(
                        {
                            "lane": lane,
                            "model": model,
                            "task": task,
                            "metric": name,
                            "batch_1": left[name],
                            "batch_8": right[name],
                        }
                    )
            sample_comparison = compare_sample_outcomes(
                samples.get((lane, units[1]), {}),
                samples.get((lane, units[8]), {}),
                task,
            )
            if (
                sample_comparison["prediction_mismatches"]
                or sample_comparison["continuous_mismatches"]
                or sample_comparison["missing_docs"]
            ):
                failures.append(
                    {
                        "lane": lane,
                        "model": model,
                        "task": task,
                        "reason": "batch sample outcomes differ",
                        **sample_comparison,
                    }
                )

    report = {
        "schema": "rwkv7-lm-eval-three-way-validation-v1",
        "units": sum(len(rows) for rows in manifests.values()),
        "status": "passed" if not failures else "failed",
        "require_model_routes": bool(args.require_model_routes),
        "artifacts": artifacts,
        "failures": failures,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "failures": len(failures),
            }
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
