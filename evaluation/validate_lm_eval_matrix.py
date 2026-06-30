#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the formal lm_eval matrix")
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def find_result(unit_dir: Path) -> Path:
    candidates = [
        path
        for path in unit_dir.rglob("*.json")
        if "samples_" not in path.name and path.name != "manifest.json"
    ]
    if not candidates:
        raise FileNotFoundError(f"no result JSON in {unit_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def numeric_metrics(result: dict, task: str) -> dict[str, float]:
    row = result["results"][task]
    return {
        key: float(value)
        for key, value in row.items()
        if isinstance(value, (int, float)) and not key.endswith("_stderr")
    }


def main():
    args = parse_args()
    latest = {}
    for line in (args.result_dir / "manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["unit"]] = row
    rows = list(latest.values())
    if len(rows) != 48:
        raise SystemExit(f"expected 48 unique units, found {len(rows)}")
    if any(row["exit_code"] != 0 for row in rows):
        raise SystemExit("at least one lm_eval unit failed")
    if any(not row.get("formal") for row in rows):
        raise SystemExit("formal validation refuses results produced with --limit")
    if any(not row.get("code_sha") for row in rows):
        raise SystemExit("every unit must record its source code SHA")
    if any(not row.get("task_provenance", {}).get("dataset_fingerprint") for row in rows):
        raise SystemExit("every unit must record task configuration and dataset fingerprint")

    indexed = {}
    for row in rows:
        result = json.loads(find_result(args.result_dir / row["unit"]).read_text())
        metrics = numeric_metrics(result, row["task"])
        if any(not math.isfinite(value) for value in metrics.values()):
            raise SystemExit(f"non-finite metric in {row['unit']}")
        indexed[(row["model"], row["task"], row["batch_size"])] = metrics

    failures = []
    for model, task, _ in {(m, t, b) for m, t, b in indexed}:
        left = indexed[(model, task, 1)]
        right = indexed[(model, task, 8)]
        for key in left.keys() & right.keys():
            absolute = abs(left[key] - right[key])
            if task == "wikitext" and "perplexity" in key:
                relative = absolute / max(abs(left[key]), 1e-12)
                if relative > 0.001:
                    failures.append((model, task, key, left[key], right[key]))
            elif absolute > 0.001:
                failures.append((model, task, key, left[key], right[key]))
    report = {"units": 48, "status": "passed" if not failures else "failed", "failures": failures}
    (args.result_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
