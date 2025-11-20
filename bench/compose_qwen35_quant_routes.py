#!/usr/bin/env python3
"""Compose measured dense and quant deployment profiles without hiding gaps.

Each selected W8/W4 row must independently beat the matching Qwen row, avoid
regressing the measured RWKV fp16 row, reduce both model footprint and peak
VRAM, and use the declared native candidate/Qwen fast backends.  The original
quantization implementation is retained in every row and a manifest records
all eligible alternatives and the deterministic winner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare_qwen35_speed_matrix import quantization_family

BASE_FIELDS = ("model_pair", "prompt_tokens", "decode_tokens", "batch_size", "dtype")


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("axis") != "qwen35_cross_model_speed":
                continue
            row = dict(row)
            row["_route_source_file"] = str(path)
            row["_route_source_line"] = lineno
            rows.append(row)
    return rows


def base_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in BASE_FIELDS)


def divided(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def latest_by(rows: list[dict[str, Any]], key_fn) -> dict[Any, dict[str, Any]]:
    selected = {}
    for row in rows:
        selected[key_fn(row)] = row
    return selected


def evaluate(
    candidate: dict[str, Any],
    dense: dict[str, Any],
    reference: dict[str, Any],
    *,
    min_qwen_prefill: float,
    min_qwen_decode: float,
    min_dense_prefill: float,
    min_dense_decode: float,
    require_qwen_gate: bool = True,
) -> dict[str, Any]:
    metrics = {
        "qwen_prefill_speedup": divided(
            candidate.get("prefill_tokps_total"), reference.get("prefill_tokps_total")
        ),
        "qwen_decode_speedup": divided(
            candidate.get("decode_tokps_total"), reference.get("decode_tokps_total")
        ),
        "dense_prefill_speedup": divided(
            candidate.get("prefill_tokps_total"), dense.get("prefill_tokps_total")
        ),
        "dense_decode_speedup": divided(
            candidate.get("decode_tokps_total"), dense.get("decode_tokps_total")
        ),
        "footprint_ratio": divided(
            candidate.get("model_footprint_mb"), dense.get("model_footprint_mb")
        ),
        "peak_vram_ratio": divided(
            candidate.get("peak_vram_mb"), dense.get("peak_vram_mb")
        ),
    }
    checks = {
        "status": candidate.get("status") == "pass" and reference.get("status") == "pass",
        "finite": candidate.get("logits_finite") is True,
        "native_candidate": (
            candidate.get("prefill_effective_backend") in {"native_prefill", "native_prefill_graph"}
            and candidate.get("effective_backend") == "native_graph"
        ),
        "qwen_fast_path": (
            not require_qwen_gate
            or
            reference.get("effective_backend") == "fla+causal_conv1d"
            and reference.get("qwen_fast_path_verified") is True
        ),
        "prefill_mode": int(candidate.get("prefill_chunk_size") or 0)
        == int(reference.get("prefill_chunk_size") or 0)
        == int(dense.get("prefill_chunk_size") or 0),
        "qwen_prefill": not require_qwen_gate
        or (
            metrics["qwen_prefill_speedup"] is not None
            and metrics["qwen_prefill_speedup"] >= min_qwen_prefill
        ),
        "qwen_decode": not require_qwen_gate
        or (
            metrics["qwen_decode_speedup"] is not None
            and metrics["qwen_decode_speedup"] >= min_qwen_decode
        ),
        "dense_prefill": metrics["dense_prefill_speedup"] is not None
        and metrics["dense_prefill_speedup"] >= min_dense_prefill,
        "dense_decode": metrics["dense_decode_speedup"] is not None
        and metrics["dense_decode_speedup"] >= min_dense_decode,
        "footprint": metrics["footprint_ratio"] is not None and metrics["footprint_ratio"] < 1.0,
        "peak_vram": metrics["peak_vram_ratio"] is not None and metrics["peak_vram_ratio"] < 1.0,
    }
    margins = [metrics["dense_prefill_speedup"], metrics["dense_decode_speedup"]]
    if require_qwen_gate:
        margins = [
            metrics["qwen_prefill_speedup"],
            metrics["qwen_decode_speedup"],
            *margins,
        ]
    available_margins = [float(value) for value in margins if value is not None]
    return {
        "implementation": candidate.get("quantization"),
        "source_file": candidate.get("_route_source_file"),
        "source_line": candidate.get("_route_source_line"),
        "metrics": metrics,
        "checks": checks,
        "eligible": all(checks.values()),
        "score": min(available_margins) if available_margins else float("-inf"),
    }


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_route_")}


def compose(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_rows([Path(path) for path in args.results])
    dense_candidates = latest_by(
        [r for r in rows if r.get("model_role") == "candidate" and quantization_family(r.get("quantization")) == "none"],
        base_key,
    )
    dense_references = latest_by(
        [r for r in rows if r.get("model_role") == "reference" and quantization_family(r.get("quantization")) == "none"],
        base_key,
    )
    quant_references = latest_by(
        [
            r
            for r in rows
            if r.get("model_role") == "reference"
            and quantization_family(r.get("quantization")) in {"w8", "w4"}
        ],
        lambda row: (quantization_family(row.get("quantization")), *base_key(row)),
    )
    # Keep every measured variant. Exact-card tuning legitimately emits
    # several rows with the same public implementation name (for example
    # BnB8 at different sequence chunk sizes). Collapsing those rows here
    # would make input order, rather than the acceptance gates, pick a result.
    quant_candidates = [
        r
        for r in rows
        if r.get("model_role") == "candidate"
        and quantization_family(r.get("quantization")) in {"w8", "w4"}
    ]

    output = []
    decisions = []
    failures = []
    base_keys = sorted(dense_candidates, key=lambda key: tuple(str(value) for value in key))
    for key in base_keys:
        dense_candidate = dense_candidates[key]
        dense_reference = dense_references.get(key)
        if dense_reference is None:
            failures.append({"cell": dict(zip(BASE_FIELDS, key)), "reason": "missing dense reference"})
            continue
        output.extend([clean_row(dense_candidate), clean_row(dense_reference)])
        for family in ("w8", "w4"):
            measured_quant_reference = quant_references.get((family, *key))
            # Quantized RWKV is accepted against the matching RWKV dense row.
            # A quantized Qwen row is optional reporting context, never a
            # requirement when --no-quant-qwen-gate is selected.
            reference = (
                measured_quant_reference
                if args.quant_qwen_gate
                else dense_reference
            )
            variants = [
                row
                for row in quant_candidates
                if base_key(row) == key
                and quantization_family(row.get("quantization")) == family
            ]
            if (args.quant_qwen_gate and reference is None) or not variants:
                failures.append(
                    {
                        "cell": {**dict(zip(BASE_FIELDS, key)), "quantization": family},
                        "reason": "missing quant reference or candidate variant",
                    }
                )
                continue
            evaluations = [
                evaluate(
                    variant,
                    dense_candidate,
                    reference,
                    min_qwen_prefill=args.min_qwen_prefill,
                    min_qwen_decode=args.min_qwen_decode,
                    min_dense_prefill=args.min_dense_prefill,
                    min_dense_decode=args.min_dense_decode,
                    require_qwen_gate=args.quant_qwen_gate,
                )
                for variant in variants
            ]
            eligible = [item for item in evaluations if item["eligible"]]
            eligible.sort(
                key=lambda item: (
                    item["score"],
                    -float(item["metrics"]["footprint_ratio"]),
                    str(item["implementation"]),
                    str(item["source_file"]),
                    int(item["source_line"] or 0),
                ),
                reverse=True,
            )
            decision = {
                "cell": {**dict(zip(BASE_FIELDS, key)), "quantization": family},
                "alternatives": evaluations,
                "selected": eligible[0] if eligible else None,
            }
            decisions.append(decision)
            if not eligible:
                failures.append({"cell": decision["cell"], "reason": "no candidate passed all gates"})
                continue
            chosen_eval = eligible[0]
            chosen = next(
                row
                for row in variants
                if row.get("quantization") == chosen_eval["implementation"]
                and row.get("_route_source_file") == chosen_eval["source_file"]
                and row.get("_route_source_line") == chosen_eval["source_line"]
            )
            chosen = clean_row(chosen)
            chosen["quantization_route"] = "measured_profile_auto"
            chosen["quantization_route_source"] = chosen_eval["implementation"]
            chosen["quantization_route_metrics"] = chosen_eval["metrics"]
            reference_output = clean_row(reference)
            if not args.quant_qwen_gate:
                reference_output = dict(reference_output)
                reference_output["quantization"] = family
                reference_output["quantization_reference_mode"] = "dense_qwen_non_gating"
                reference_output["quantization_reference_non_gating"] = True
            output.extend([chosen, reference_output])

    manifest = {
        "axis": "qwen35_quant_route_composition",
        "status": "pass" if not failures else "fail",
        "inputs": args.results,
        "thresholds": {
            "min_qwen_prefill": args.min_qwen_prefill,
            "min_qwen_decode": args.min_qwen_decode,
            "min_dense_prefill": args.min_dense_prefill,
            "min_dense_decode": args.min_dense_decode,
            "require_footprint_and_peak_reduction": True,
            "require_native_candidate": True,
            "require_qwen_fast_path": args.quant_qwen_gate,
            "quant_qwen_gate": args.quant_qwen_gate,
        },
        "output_rows": len(output),
        "decisions": decisions,
        "failures": failures,
    }
    return output, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--min-qwen-prefill", type=float, default=1.0)
    parser.add_argument("--min-qwen-decode", type=float, default=1.0)
    parser.add_argument("--min-dense-prefill", type=float, default=1.0)
    parser.add_argument("--min-dense-decode", type=float, default=1.0)
    parser.add_argument(
        "--quant-qwen-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Gate quantized RWKV against matching quantized Qwen rows. "
            "Use --no-quant-qwen-gate to accept W8/W4 only against RWKV dense "
            "speed/memory and reuse dense Qwen rows as explicitly non-gating report references."
        ),
    )
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output, manifest = compose(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_rows": manifest["output_rows"],
                "decisions": len(manifest["decisions"]),
                "failures": len(manifest["failures"]),
            }
        )
    )
    return 1 if args.fail_on_gate and manifest["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
