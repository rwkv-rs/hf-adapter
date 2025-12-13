#!/usr/bin/env python3
"""Build a fail-closed Native-vs-official fixed-shape decode summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def greedy_trace_sha256(greedy_tokens: list[list[int]]) -> str:
    payload = json.dumps(greedy_tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def summarize(rows: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
    repetitions = int(reference["native_repetitions"])
    expected_batches = {int(value) for value in reference["batch_sizes"]}
    by_batch: dict[int, list[dict[str, Any]]] = {value: [] for value in expected_batches}
    for row in rows:
        batch_size = int(row["batch_size"])
        if batch_size not in by_batch:
            raise ValueError(f"unexpected batch size {batch_size}")
        if int(row["decode_steps"]) != int(reference["native_decode_steps"]):
            raise ValueError("native decode step count does not match reference contract")
        if row["device"] != reference["device"] or row["dtype"] != reference["dtype"]:
            raise ValueError("native device/dtype does not match reference contract")
        by_batch[batch_size].append(row)

    shape_rows: list[dict[str, Any]] = []
    passed = True
    for batch_size in sorted(by_batch):
        items = by_batch[batch_size]
        if len(items) != repetitions:
            raise ValueError(
                f"batch {batch_size} requires {repetitions} repetitions; got {len(items)}"
            )
        speeds = [float(item["decode_tokps"]) for item in items]
        hashes: list[str] = []
        batch_consistent = True
        extensions_active = True
        for item in items:
            traces = item.get("greedy_tokens") or []
            batch_consistent &= bool(
                not traces or all(trace == traces[0] for trace in traces)
            )
            hashes.append(item.get("greedy_trace_sha256") or greedy_trace_sha256(traces))
            extensions = item.get("requested_extensions") or {}
            extensions_active &= bool(
                extensions
                and all(
                    not status.get("requested") or status.get("active")
                    for status in extensions.values()
                )
            )
        native_median = statistics.median(speeds)
        official = float(reference["batch_sizes"][str(batch_size)]["decode_tokps"])
        ratio = native_median / official
        repeat_consistent = len(set(hashes)) == 1
        shape_pass = bool(
            batch_consistent and repeat_consistent and extensions_active and ratio >= 1.0
        )
        passed &= shape_pass
        shape_rows.append(
            {
                "batch_size": batch_size,
                "native_decode_tokps": speeds,
                "native_median_tokps": native_median,
                "official_p50_tokps": official,
                "matched_shape_ratio": ratio,
                "greedy_trace_sha256": hashes,
                "batch_traces_equal": batch_consistent,
                "repeat_traces_equal": repeat_consistent,
                "requested_extensions_active": extensions_active,
                "status": "pass" if shape_pass else "fail",
            }
        )

    return {
        "axis": "native_official_decode_acceptance",
        "status": "pass" if passed else "fail",
        "device": reference["device"],
        "checkpoint": reference["checkpoint"],
        "precision_mode": reference["precision_mode"],
        "official_engine": reference["official_engine"],
        "official_commit": reference["official_commit"],
        "official_iterations": reference["official_iterations"],
        "native_repetitions": repetitions,
        "native_decode_steps": reference["native_decode_steps"],
        "rows": shape_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", nargs="+", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    report = summarize(load_rows(args.native), reference)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
