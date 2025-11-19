#!/usr/bin/env python3
"""Join RWKV/Qwen3.5 speed rows and enforce declared matrix gates."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

CELL_FIELDS = (
    "model_pair",
    "prompt_tokens",
    "decode_tokens",
    "batch_size",
    "dtype",
    "quantization",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{lineno}: {exc}") from exc
        if row.get("axis") == "qwen35_cross_model_speed":
            row["_lineno"] = lineno
            rows.append(row)
    return rows


def cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in CELL_FIELDS)


def key_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(CELL_FIELDS, key))


def ratio(candidate: Any, reference: Any) -> float | None:
    if candidate is None or reference in (None, 0):
        return None
    return round(float(candidate) / float(reference), 6)


def median_or_none(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 6) if values else None


def reference_backend_matches(row: dict[str, Any], required: str) -> bool:
    if required == "any":
        return True
    requested = row.get("qwen_backend_requested")
    effective = row.get("effective_backend")
    if required == "fla":
        return bool(
            requested == "fla"
            and effective
            in {
                "qwen_fla_gated_delta_rule",
                "qwen_fla_gated_delta_rule_torch_conv",
            }
            and row.get("qwen_operator_contract_pass") is True
        )
    return bool(
        requested == "torch"
        and effective == "transformers_torch_fallback"
        and row.get("qwen_force_torch") is True
    )


def compare(
    rows: list[dict[str, Any]],
    *,
    expected_cells: int,
    min_prefill_speedup: float,
    min_decode_speedup: float,
    required_reference_backend: str,
    require_memory_not_larger: bool = False,
) -> dict[str, Any]:
    indexed: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {"candidate": {}, "reference": {}}
    for row in rows:
        role = str(row.get("model_role") or "")
        if role in indexed:
            indexed[role][cell_key(row)] = row

    candidate_keys = set(indexed["candidate"])
    reference_keys = set(indexed["reference"])
    joined_keys = sorted(candidate_keys & reference_keys, key=lambda key: tuple(str(x) for x in key))
    missing_candidate = sorted(reference_keys - candidate_keys, key=lambda key: tuple(str(x) for x in key))
    missing_reference = sorted(candidate_keys - reference_keys, key=lambda key: tuple(str(x) for x in key))

    cells: list[dict[str, Any]] = []
    red_cells: list[dict[str, Any]] = []
    prefill_ratios: list[float] = []
    decode_ratios: list[float] = []
    footprint_ratios: list[float] = []
    peak_vram_ratios: list[float] = []
    backend_matches = 0
    for key in joined_keys:
        candidate = indexed["candidate"][key]
        reference = indexed["reference"][key]
        both_pass = candidate.get("status") == "pass" and reference.get("status") == "pass"
        reference_backend_pass = reference_backend_matches(reference, required_reference_backend)
        backend_matches += int(reference_backend_pass)
        prefill_speedup = ratio(candidate.get("prefill_tokps_total"), reference.get("prefill_tokps_total"))
        decode_speedup = ratio(candidate.get("decode_tokps_total"), reference.get("decode_tokps_total"))
        footprint_ratio = ratio(candidate.get("model_footprint_mb"), reference.get("model_footprint_mb"))
        peak_vram_ratio = ratio(candidate.get("peak_vram_mb"), reference.get("peak_vram_mb"))
        memory_pass = bool(
            not require_memory_not_larger
            or (
                footprint_ratio is not None
                and peak_vram_ratio is not None
                and footprint_ratio <= 1.0
                and peak_vram_ratio <= 1.0
            )
        )
        if prefill_speedup is not None:
            prefill_ratios.append(prefill_speedup)
        if decode_speedup is not None:
            decode_ratios.append(decode_speedup)
        if footprint_ratio is not None:
            footprint_ratios.append(footprint_ratio)
        if peak_vram_ratio is not None:
            peak_vram_ratios.append(peak_vram_ratio)
        passed = bool(
            both_pass
            and reference_backend_pass
            and prefill_speedup is not None
            and decode_speedup is not None
            and prefill_speedup >= min_prefill_speedup
            and decode_speedup >= min_decode_speedup
            and memory_pass
        )
        cell = {
            **key_dict(key),
            "candidate_status": candidate.get("status"),
            "reference_status": reference.get("status"),
            "reference_backend_requested": reference.get("qwen_backend_requested"),
            "reference_effective_backend": reference.get("effective_backend"),
            "reference_backend_pass": reference_backend_pass,
            "candidate_prefill_tokps_total": candidate.get("prefill_tokps_total"),
            "reference_prefill_tokps_total": reference.get("prefill_tokps_total"),
            "prefill_speedup": prefill_speedup,
            "candidate_decode_tokps_total": candidate.get("decode_tokps_total"),
            "reference_decode_tokps_total": reference.get("decode_tokps_total"),
            "decode_speedup": decode_speedup,
            "candidate_model_footprint_mb": candidate.get("model_footprint_mb"),
            "reference_model_footprint_mb": reference.get("model_footprint_mb"),
            "model_footprint_ratio": footprint_ratio,
            "candidate_peak_vram_mb": candidate.get("peak_vram_mb"),
            "reference_peak_vram_mb": reference.get("peak_vram_mb"),
            "peak_vram_ratio": peak_vram_ratio,
            "memory_pass": memory_pass,
            "passed": passed,
        }
        cells.append(cell)
        if not passed:
            red_cells.append(cell)

    expected = expected_cells if expected_cells > 0 else len(candidate_keys | reference_keys)
    coverage_complete = bool(
        len(joined_keys) == expected and not missing_candidate and not missing_reference
    )
    reference_backend_complete = bool(cells) and backend_matches == len(cells)
    cells_pass = not red_cells and bool(cells)
    overall_pass = coverage_complete and reference_backend_complete and cells_pass
    return {
        "axis": "qwen35_cross_model_speed_comparison",
        "coverage": {
            "expected_cells": expected,
            "joined_cells": len(joined_keys),
            "complete": coverage_complete,
        },
        "thresholds": {
            "min_prefill_speedup": min_prefill_speedup,
            "min_decode_speedup": min_decode_speedup,
            "required_reference_backend": required_reference_backend,
            "require_memory_not_larger": require_memory_not_larger,
        },
        "reference_backend": {
            "required": required_reference_backend,
            "matching_cells": backend_matches,
            "total_cells": len(cells),
            "complete": reference_backend_complete,
        },
        "speed": {
            "min_prefill_speedup": min(prefill_ratios) if prefill_ratios else None,
            "median_prefill_speedup": median_or_none(prefill_ratios),
            "max_prefill_speedup": max(prefill_ratios) if prefill_ratios else None,
            "prefill_at_least_equal_cells": sum(value >= 1.0 for value in prefill_ratios),
            "prefill_gate_cells": sum(value >= min_prefill_speedup for value in prefill_ratios),
            "min_decode_speedup": min(decode_ratios) if decode_ratios else None,
            "median_decode_speedup": median_or_none(decode_ratios),
            "max_decode_speedup": max(decode_ratios) if decode_ratios else None,
            "decode_at_least_equal_cells": sum(value >= 1.0 for value in decode_ratios),
            "decode_gate_cells": sum(value >= min_decode_speedup for value in decode_ratios),
            "strict_gate_cells": len(cells) - len(red_cells),
            "total_cells": len(cells),
        },
        "memory": {
            "min_model_footprint_ratio": min(footprint_ratios) if footprint_ratios else None,
            "median_model_footprint_ratio": median_or_none(footprint_ratios),
            "max_model_footprint_ratio": max(footprint_ratios) if footprint_ratios else None,
            "model_footprint_not_larger_cells": sum(value <= 1.0 for value in footprint_ratios),
            "min_peak_vram_ratio": min(peak_vram_ratios) if peak_vram_ratios else None,
            "median_peak_vram_ratio": median_or_none(peak_vram_ratios),
            "max_peak_vram_ratio": max(peak_vram_ratios) if peak_vram_ratios else None,
            "peak_vram_not_larger_cells": sum(value <= 1.0 for value in peak_vram_ratios),
            "total_cells": len(cells),
        },
        "missing": {
            "candidate": [key_dict(key) for key in missing_candidate],
            "reference": [key_dict(key) for key in missing_reference],
        },
        "red_cells": red_cells,
        "cells": cells,
        "gates": {
            "coverage_pass": coverage_complete,
            "reference_backend_pass": reference_backend_complete,
            "memory_pass": all(cell["memory_pass"] for cell in cells) and bool(cells),
            "speed_pass": cells_pass,
            "overall_pass": overall_pass,
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    backend = summary["reference_backend"]
    speed = summary["speed"]
    memory = summary["memory"]
    overall = "PASS" if summary["gates"]["overall_pass"] else "FAIL"
    lines = [
        "# RWKV-7 vs Qwen3.5 HF speed matrix",
        "",
        f"Overall: {overall}",
        "",
        f"Coverage: `{coverage['joined_cells']}/{coverage['expected_cells']}` cells.",
        "",
        f"Required Qwen backend: `{backend['required']}`; verified: "
        f"`{backend['matching_cells']}/{backend['total_cells']}` cells.",
        "",
        "| Metric | Minimum | Median | Maximum | Passing cells |",
        "|---|---:|---:|---:|---:|",
        f"| Prefill RWKV/Qwen | {fmt(speed['min_prefill_speedup'])}x | {fmt(speed['median_prefill_speedup'])}x | "
        f"{fmt(speed['max_prefill_speedup'])}x | {speed['prefill_gate_cells']}/{speed['total_cells']} |",
        f"| Decode RWKV/Qwen | {fmt(speed['min_decode_speedup'])}x | {fmt(speed['median_decode_speedup'])}x | "
        f"{fmt(speed['max_decode_speedup'])}x | {speed['decode_gate_cells']}/{speed['total_cells']} |",
        f"| Model footprint RWKV/Qwen | {fmt(memory['min_model_footprint_ratio'])}x | "
        f"{fmt(memory['median_model_footprint_ratio'])}x | {fmt(memory['max_model_footprint_ratio'])}x | "
        f"{memory['model_footprint_not_larger_cells']}/{memory['total_cells']} |",
        f"| Peak VRAM RWKV/Qwen | {fmt(memory['min_peak_vram_ratio'])}x | "
        f"{fmt(memory['median_peak_vram_ratio'])}x | {fmt(memory['max_peak_vram_ratio'])}x | "
        f"{memory['peak_vram_not_larger_cells']}/{memory['total_cells']} |",
        "",
        f"Strict speed cells: `{speed['strict_gate_cells']}/{speed['total_cells']}`.",
        "",
        "## Red cells",
        "",
    ]
    if not summary["red_cells"]:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Pair | Prompt | Decode | Bsz | Quant | Prefill | Decode | Qwen backend | Candidate | Reference |",
                "|---|---:|---:|---:|---|---:|---:|---|---|---|",
            ]
        )
        for cell in summary["red_cells"]:
            lines.append(
                "| {model_pair} | {prompt_tokens} | {decode_tokens} | {batch_size} | {quantization} | "
                "{prefill}x | {decode}x | {backend} | {candidate_status} | {reference_status} |".format(
                    **cell,
                    prefill=fmt(cell["prefill_speedup"]),
                    decode=fmt(cell["decode_speedup"]),
                    backend=cell["reference_effective_backend"] or "-",
                )
            )
    lines.extend(
        [
            "",
            f"Missing candidate rows: `{len(summary['missing']['candidate'])}`.",
            f"Missing reference rows: `{len(summary['missing']['reference'])}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--expected-cells", type=int, default=216)
    ap.add_argument("--min-prefill-speedup", type=float, default=1.05)
    ap.add_argument("--min-decode-speedup", type=float, default=1.05)
    ap.add_argument("--required-reference-backend", choices=["fla", "torch", "any"], default="fla")
    ap.add_argument("--require-memory-not-larger", action="store_true")
    ap.add_argument("--json-output", default="")
    ap.add_argument("--markdown-output", default="")
    ap.add_argument("--fail-on-gate", action="store_true")
    args = ap.parse_args()

    summary = compare(
        load_rows(Path(args.results)),
        expected_cells=args.expected_cells,
        min_prefill_speedup=args.min_prefill_speedup,
        min_decode_speedup=args.min_decode_speedup,
        required_reference_backend=args.required_reference_backend,
        require_memory_not_larger=args.require_memory_not_larger,
    )
    markdown = render_markdown(summary)
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        out = Path(args.markdown_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 1 if args.fail_on_gate and not summary["gates"]["overall_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
