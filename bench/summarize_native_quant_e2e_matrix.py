#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def fusion_mode(row: dict) -> str:
    if row.get("fused_quant_ffn_down_add", False):
        return "deep"
    if row.get("fused_quant_ffn", False):
        return "up"
    return "off"


def cell_key(row: dict) -> tuple:
    return (
        str(row.get("model_size_label", "")),
        int(row.get("batch_size", 0)),
        int(row.get("prompt_tokens", 0)),
        int(row.get("decode_tokens", 0)),
    )


def row_key(row: dict) -> tuple:
    return (*cell_key(row), str(row.get("quantization", "")), fusion_mode(row))


def failure_key(row: dict) -> tuple:
    return (
        str(row.get("model_label", "")),
        int(row.get("batch_size", 0)),
        int(row.get("prompt_tokens", 0)),
        int(row.get("decode_tokens", 0)),
        str(row.get("quantization", "")),
        str(row.get("fusion_mode", "off")),
    )


def load_rows(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    successful: dict[tuple, dict] = {}
    failures: list[dict] = []
    for path in paths:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("axis") == "native_quant_e2e_decode" and row.get("status") == "pass":
                successful[row_key(row)] = row
            elif row.get("status") == "fail":
                failures.append(row)
    return list(successful.values()), failures


def metric_summary(rows: list[dict]) -> dict:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]

    speed = values("decode_speed_ratio_vs_fp16")
    footprint = values("footprint_ratio_vs_fp16")
    prompt_cos = values("prompt_logits_cos_vs_fp16")
    final_cos = values("final_logits_cos_vs_fp16")
    peak = values("peak_vram_mb")
    return {
        "count": len(rows),
        "speed_pass_count": sum(
            row.get("decode_speed_ratio_vs_fp16") is not None
            and float(row["decode_speed_ratio_vs_fp16"]) >= 1.0
            for row in rows
        ),
        "footprint_pass_count": sum(
            row.get("footprint_ratio_vs_fp16") is not None
            and float(row["footprint_ratio_vs_fp16"]) < 1.0
            for row in rows
        ),
        "decode_ratio_median": statistics.median(speed) if speed else None,
        "decode_ratio_min": min(speed) if speed else None,
        "decode_ratio_max": max(speed) if speed else None,
        "footprint_ratio_median": statistics.median(footprint) if footprint else None,
        "peak_vram_mb_max": max(peak) if peak else None,
        "prompt_cosine_min": min(prompt_cos) if prompt_cos else None,
        "final_cosine_min": min(final_cos) if final_cos else None,
        "same_token_count": sum(row.get("same_next_token_as_fp16") is True for row in rows),
    }


def paired_summary(rows_by_key: dict[tuple, dict], quantization: str, left: str, right: str) -> dict:
    ratios = []
    cells = {key[:4] for key in rows_by_key if key[4] == quantization}
    for cell in cells:
        left_row = rows_by_key.get((*cell, quantization, left))
        right_row = rows_by_key.get((*cell, quantization, right))
        if left_row is None or right_row is None:
            continue
        denominator = float(left_row["decode_tokps_total"])
        if denominator > 0:
            ratios.append(float(right_row["decode_tokps_total"]) / denominator)
    return {
        "left": left,
        "right": right,
        "paired_cells": len(ratios),
        "right_wins": sum(ratio > 1.0 for ratio in ratios),
        "right_non_regressions": sum(ratio >= 0.99 for ratio in ratios),
        "ratio_median": statistics.median(ratios) if ratios else None,
        "ratio_min": min(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
    }


def summarize(rows: list[dict], failures: list[dict], expected_models: int) -> dict:
    rows_by_key = {row_key(row): row for row in rows}
    unresolved_failures = [row for row in failures if failure_key(row) not in rows_by_key]
    expected_base_cells = expected_models * 7
    expected_rows = expected_base_cells * 6
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("model_size_label")), str(row.get("quantization")), fusion_mode(row))].append(row)
    aggregates = {
        "/".join(key): metric_summary(group_rows)
        for key, group_rows in sorted(grouped.items())
    }
    acceptance = {}
    for name, item in aggregates.items():
        if "/none/" in name:
            continue
        speed = item["speed_pass_count"] == item["count"]
        footprint = item["footprint_pass_count"] == item["count"]
        greedy = item["same_token_count"] == item["count"]
        acceptance[name] = {
            "cells": item["count"],
            "speed_pass_cells": item["speed_pass_count"],
            "footprint_pass_cells": item["footprint_pass_count"],
            "greedy_pass_cells": item["same_token_count"],
            "speed_gate_pass": speed,
            "footprint_gate_pass": footprint,
            "greedy_gate_pass": greedy,
            "acceptance_pass": speed and footprint and greedy,
        }
    return {
        "profile": "expanded",
        "expected_models": expected_models,
        "expected_base_cells": expected_base_cells,
        "expected_rows": expected_rows,
        "completed_rows": len(rows),
        "failure_attempts": len(failures),
        "unresolved_failures": len(unresolved_failures),
        "complete": len(rows) == expected_rows and not unresolved_failures,
        "all_quant_paths_accepted": bool(acceptance)
        and all(item["acceptance_pass"] for item in acceptance.values()),
        "aggregates": aggregates,
        "acceptance": acceptance,
        "paired": {
            "mm4_up_vs_off": paired_summary(rows_by_key, "mm4", "off", "up"),
            "mm8_up_vs_off": paired_summary(rows_by_key, "mm8", "off", "up"),
            "mm8_deep_vs_up": paired_summary(rows_by_key, "mm8", "up", "deep"),
        },
    }


def fmt(value, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def markdown(summary: dict) -> str:
    lines = [
        "# Native quant end-to-end matrix summary",
        "",
        f"- Completion: `{summary['completed_rows']}/{summary['expected_rows']}` rows",
        f"- Failed attempts: `{summary['failure_attempts']}`",
        f"- Unresolved failures: `{summary['unresolved_failures']}`",
        f"- Execution complete: `{'yes' if summary['complete'] else 'no'}`",
        f"- All quant paths accepted: `{'yes' if summary['all_quant_paths_accepted'] else 'no'}`",
        "",
        "| Model / quant / fusion | Rows | Speed >=fp16 | Decode/fp16 median | Min | Max | Footprint median | Min final cosine | Greedy match | Accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["aggregates"].items():
        lines.append(
            f"| {name} | {item['count']} | {item['speed_pass_count']}/{item['count']} | "
            f"{fmt(item['decode_ratio_median'])} | "
            f"{fmt(item['decode_ratio_min'])} | {fmt(item['decode_ratio_max'])} | "
            f"{fmt(item['footprint_ratio_median'])} | {fmt(item['final_cosine_min'], 8)} | "
            f"{item['same_token_count']}/{item['count']} | "
            f"{'n/a' if '/none/' in name else ('yes' if summary['acceptance'][name]['acceptance_pass'] else 'no')} |"
        )
    lines.extend(
        [
            "",
            "| Paired comparison | Cells | Right wins | >=0.99x | Median | Min | Max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary["paired"].items():
        lines.append(
            f"| {name} | {item['paired_cells']} | {item['right_wins']} | "
            f"{item['right_non_regressions']} | {fmt(item['ratio_median'])} | "
            f"{fmt(item['ratio_min'])} | {fmt(item['ratio_max'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", type=Path, required=True)
    ap.add_argument("--expected-models", type=int, default=3)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()
    rows, failures = load_rows(args.input)
    summary = summarize(rows, failures, args.expected_models)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("completed_rows", "expected_rows", "failure_attempts", "unresolved_failures", "complete")}, sort_keys=True))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
