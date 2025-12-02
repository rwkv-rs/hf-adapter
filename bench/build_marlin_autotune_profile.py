#!/usr/bin/env python3
"""Build an exact-device Marlin runtime profile from a measured JSONL sweep."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, default=1.02)
    parser.add_argument("--min-cosine", type=float, default=0.99999)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    passed = [row for row in rows if row.get("status") == "pass"]
    if not passed:
        raise SystemExit("input contains no passing rows")
    identity_keys = ("device", "compute_capability", "torch_version", "cuda_version")
    identity = {key: passed[0].get(key) for key in identity_keys}
    if any(any(row.get(key) != value for key, value in identity.items()) for row in passed):
        raise SystemExit("input mixes runtime identities")

    groups = defaultdict(list)
    for row in passed:
        key = (
            int(row["k"]),
            int(row["n"]),
            int(row.get("group_size", 128)),
            int(row["rows"]),
        )
        groups[key].append(row)

    entries = []
    for (k, n, group_size, logical_rows), candidates in sorted(groups.items()):
        manual = [
            row
            for row in candidates
            if row.get("schedule") != "auto"
            and float(row.get("cosine_vs_auto", 0.0)) >= args.min_cosine
            and float(row.get("speedup_vs_auto_median_of_pairs", 0.0)) >= args.min_speedup
            and min(float(value) for value in row.get("speedup_vs_auto_samples", (0.0,)))
            >= 1.0
        ]
        if not manual:
            continue
        best = min(manual, key=lambda row: float(row["candidate_ms"]))
        entries.append(
            {
                "k": k,
                "n": n,
                "group_size": group_size,
                "rows": logical_rows,
                "schedule": [
                    int(best["tile_k"]),
                    int(best["block_n"]),
                    int(best["num_threads"]),
                    int(best.get("sms", -1)),
                    int(best.get("stages", -1)),
                ],
                "candidate_ms": float(best["candidate_ms"]),
                "auto_ms": float(best["auto_ms"]),
                "speedup_vs_auto": float(best["speedup_vs_auto_median_of_pairs"]),
                "cosine_vs_auto": float(best["cosine_vs_auto"]),
            }
        )

    payload = {
        "schema_version": 1,
        **identity,
        "source": str(args.input),
        "min_speedup": float(args.min_speedup),
        "min_cosine": float(args.min_cosine),
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "entries": len(entries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
