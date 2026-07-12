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


def compare(
    rows: list[dict[str, Any]],
    *,
    expected_cells: int,
    min_prefill_speedup: float,
    min_decode_speedup: float,
    min_quant_prefill_speedup: float | None = None,
    min_quant_decode_speedup: float | None = None,
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
    for key in joined_keys:
        candidate = indexed["candidate"][key]
        reference = indexed["reference"][key]
        both_pass = candidate.get("status") == "pass" and reference.get("status") == "pass"
        prefill_speedup = ratio(candidate.get("prefill_tokps_total"), reference.get("prefill_tokps_total"))
        decode_speedup = ratio(candidate.get("decode_tokps_total"), reference.get("decode_tokps_total"))
        if prefill_speedup is not None:
            prefill_ratios.append(prefill_speedup)
        if decode_speedup is not None:
            decode_ratios.append(decode_speedup)
        quantized = key_dict(key).get("quantization") != "none"
        prefill_gate = (
            min_quant_prefill_speedup
            if quantized and min_quant_prefill_speedup is not None
            else min_prefill_speedup
        )
        decode_gate = (
            min_quant_decode_speedup
            if quantized and min_quant_decode_speedup is not None
            else min_decode_speedup
        )
        passed = bool(
            both_pass
            and prefill_speedup is not None
            and decode_speedup is not None
            and prefill_speedup >= prefill_gate
            and decode_speedup >= decode_gate
        )
        cell = {
            **key_dict(key),
            "candidate_status": candidate.get("status"),
            "reference_status": reference.get("status"),
            "candidate_prefill_tokps_total": candidate.get("prefill_tokps_total"),
            "reference_prefill_tokps_total": reference.get("prefill_tokps_total"),
            "prefill_speedup": prefill_speedup,
            "candidate_decode_tokps_total": candidate.get("decode_tokps_total"),
            "reference_decode_tokps_total": reference.get("decode_tokps_total"),
            "decode_speedup": decode_speedup,
            "candidate_peak_vram_mb": candidate.get("peak_vram_mb"),
            "reference_peak_vram_mb": reference.get("peak_vram_mb"),
            "passed": passed,
        }
        cells.append(cell)
        if not passed:
            red_cells.append(cell)

    expected = expected_cells if expected_cells > 0 else len(candidate_keys | reference_keys)
    coverage_complete = bool(
        len(joined_keys) == expected and not missing_candidate and not missing_reference
    )
    ratios_pass = not red_cells and bool(cells)
    overall_pass = coverage_complete and ratios_pass
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
            "min_quant_prefill_speedup": min_quant_prefill_speedup,
            "min_quant_decode_speedup": min_quant_decode_speedup,
        },
        "speed": {
            "min_prefill_speedup": min(prefill_ratios) if prefill_ratios else None,
            "median_prefill_speedup": median_or_none(prefill_ratios),
            "min_decode_speedup": min(decode_ratios) if decode_ratios else None,
            "median_decode_speedup": median_or_none(decode_ratios),
        },
        "missing": {
            "candidate": [key_dict(key) for key in missing_candidate],
            "reference": [key_dict(key) for key in missing_reference],
        },
        "red_cells": red_cells,
        "cells": cells,
        "gates": {
            "coverage_pass": coverage_complete,
            "speed_pass": ratios_pass,
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
    speed = summary["speed"]
    overall = "PASS" if summary["gates"]["overall_pass"] else "FAIL"
    lines = [
        "# RWKV-7 vs Qwen3.5 HF speed matrix",
        "",
        f"Overall: {overall}",
        "",
        f"Coverage: `{coverage['joined_cells']}/{coverage['expected_cells']}` cells.",
        "",
        "| Metric | Minimum | Median |",
        "|---|---:|---:|",
        f"| Prefill RWKV/Qwen | {fmt(speed['min_prefill_speedup'])}x | {fmt(speed['median_prefill_speedup'])}x |",
        f"| Decode RWKV/Qwen | {fmt(speed['min_decode_speedup'])}x | {fmt(speed['median_decode_speedup'])}x |",
        "",
        "## Red cells",
        "",
    ]
    if not summary["red_cells"]:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Pair | Prompt | Decode | Bsz | Quant | Prefill | Decode | Candidate | Reference |",
                "|---|---:|---:|---:|---|---:|---:|---|---|",
            ]
        )
        for cell in summary["red_cells"]:
            lines.append(
                "| {model_pair} | {prompt_tokens} | {decode_tokens} | {batch_size} | {quantization} | "
                "{prefill}x | {decode}x | {candidate_status} | {reference_status} |".format(
                    **cell,
                    prefill=fmt(cell["prefill_speedup"]),
                    decode=fmt(cell["decode_speedup"]),
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
    ap.add_argument("--min-quant-prefill-speedup", type=float, default=None)
    ap.add_argument("--min-quant-decode-speedup", type=float, default=None)
    ap.add_argument("--json-output", default="")
    ap.add_argument("--markdown-output", default="")
    ap.add_argument("--fail-on-gate", action="store_true")
    args = ap.parse_args()

    summary = compare(
        load_rows(Path(args.results)),
        expected_cells=args.expected_cells,
        min_prefill_speedup=args.min_prefill_speedup,
        min_decode_speedup=args.min_decode_speedup,
        min_quant_prefill_speedup=args.min_quant_prefill_speedup,
        min_quant_decode_speedup=args.min_quant_decode_speedup,
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
