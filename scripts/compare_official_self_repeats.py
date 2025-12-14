#!/usr/bin/env python3
"""Measure pinned official FP16 inference run-to-run variation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from scripts.compare_official_native_inference import (
    ALIGNMENT_THRESHOLDS,
    metrics_pass,
    sha256_file,
    tensor_metrics,
)


def compare_repeats(reference: dict[str, Any], repeats: list[dict[str, Any]]) -> dict[str, Any]:
    if reference.get("engine") != "official_v3a":
        raise ValueError("reference must be an official_v3a capture")
    rows: list[dict[str, Any]] = []
    for repeat_index, candidate in enumerate(repeats, start=1):
        if candidate.get("engine") != "official_v3a":
            raise ValueError("repeat must be an official_v3a capture")
        for key in (
            "source_revision",
            "precision",
            "prompt_tokens",
            "decode_steps",
            "batch_sizes",
        ):
            if candidate.get(key) != reference.get(key):
                raise ValueError(f"official repeat metadata mismatch for {key}")
        if not torch.equal(candidate["prompt_ids"], reference["prompt_ids"]):
            raise ValueError("official repeat prompt token IDs do not match")
        for batch_size in reference["batch_sizes"]:
            label = str(batch_size)
            current = candidate["captures"][label]
            baseline = reference["captures"][label]
            metrics: dict[str, Any] = {}
            logits = tensor_metrics(
                current["logits"],
                baseline["logits"],
                absolute_threshold=ALIGNMENT_THRESHOLDS["logits"]["max_abs"],
            )
            logits["threshold_pass"] = metrics_pass(logits, "logits")
            metrics["logits"] = logits
            for phase in ("prefill", "final"):
                metrics[phase] = {}
                for name in ("state", "xpa", "xpf"):
                    item = tensor_metrics(
                        current[phase][name],
                        baseline[phase][name],
                        absolute_threshold=ALIGNMENT_THRESHOLDS[name]["max_abs"],
                    )
                    item["threshold_pass"] = metrics_pass(item, name)
                    metrics[phase][name] = item
            rows.append(
                {
                    "repeat": repeat_index,
                    "batch_size": int(batch_size),
                    "greedy_exact": bool(
                        torch.equal(current["greedy_tokens"], baseline["greedy_tokens"])
                    ),
                    "metrics": metrics,
                }
            )

    envelope: dict[str, Any] = {}
    for path in (
        ("logits",),
        ("prefill", "state"),
        ("prefill", "xpa"),
        ("prefill", "xpf"),
        ("final", "state"),
        ("final", "xpa"),
        ("final", "xpf"),
    ):
        values = [row["metrics"][path[0]] if len(path) == 1 else row["metrics"][path[0]][path[1]] for row in rows]
        envelope[".".join(path)] = {
            "max_abs": max(float(item["max_abs"]) for item in values),
            "max_fraction_over_abs_threshold": max(
                float(item["fraction_over_abs_threshold"]) for item in values
            ),
            "min_cosine": min(float(item["cosine"]) for item in values),
            "threshold_passes": sum(bool(item["threshold_pass"]) for item in values),
            "total": len(values),
        }
    observational_pass = all(row["greedy_exact"] for row in rows) and all(
        item["finite"]
        for row in rows
        for phase in ("logits", "prefill", "final")
        for item in (
            [row["metrics"]["logits"]]
            if phase == "logits"
            else list(row["metrics"][phase].values())
        )
    )
    return {
        "axis": "official_fp16_self_repeat",
        "status": "pass" if observational_pass else "fail",
        "official_commit": reference["source_revision"],
        "precision": reference["precision"],
        "prompt_tokens": reference["prompt_tokens"],
        "decode_steps": reference["decode_steps"],
        "batch_sizes": reference["batch_sizes"],
        "thresholds": ALIGNMENT_THRESHOLDS,
        "envelope": envelope,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--repeats", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    repeats = [torch.load(path, map_location="cpu", weights_only=False) for path in args.repeats]
    report = compare_repeats(reference, repeats)
    report["reference_sha256"] = sha256_file(args.reference)
    report["repeat_sha256"] = [sha256_file(path) for path in args.repeats]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
