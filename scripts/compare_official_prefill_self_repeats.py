#!/usr/bin/env python3
"""Measure pinned official FP16 sequence-prefill run-to-run variation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_official_native_inference import sha256_file, tensor_metrics


METRIC_NAMES = ("logits", "first_decode_logits", "layer_outputs")
STATE_NAMES = ("state", "xpa", "xpf")


def compare_repeats(
    reference: dict[str, Any], repeats: list[dict[str, Any]]
) -> dict[str, Any]:
    if reference.get("engine") != "official_v3a_sequence":
        raise ValueError("reference must be an official_v3a_sequence capture")
    rows: list[dict[str, Any]] = []
    for repeat_index, candidate in enumerate(repeats, start=1):
        if candidate.get("engine") != "official_v3a_sequence":
            raise ValueError("repeat must be an official_v3a_sequence capture")
        for key in ("source_revision", "precision", "batch_size", "prompt_tokens"):
            if candidate.get(key) != reference.get(key):
                raise ValueError(f"official prefill repeat metadata mismatch for {key}")
        if not torch.equal(candidate["prompt_ids"], reference["prompt_ids"]):
            raise ValueError("official prefill repeat prompt token IDs do not match")
        metrics = {
            name: tensor_metrics(candidate[name], reference[name])
            for name in METRIC_NAMES
        }
        metrics.update(
            {
                name: tensor_metrics(
                    candidate["prefill"][name], reference["prefill"][name]
                )
                for name in STATE_NAMES
            }
        )
        rows.append(
            {
                "repeat": repeat_index,
                "first_token_exact": bool(
                    torch.equal(candidate["first_token"], reference["first_token"])
                ),
                "first_decode_token_exact": bool(
                    torch.equal(
                        candidate["first_decode_token"],
                        reference["first_decode_token"],
                    )
                ),
                "metrics": metrics,
            }
        )

    envelope = {}
    for name in (*METRIC_NAMES, *STATE_NAMES):
        values = [row["metrics"][name] for row in rows]
        envelope[name] = {
            "max_abs": max(float(item["max_abs"]) for item in values),
            "max_fraction_over_abs_threshold": max(
                float(item["fraction_over_abs_threshold"]) for item in values
            ),
            "min_cosine": min(float(item["cosine"]) for item in values),
            "total": len(values),
        }
    passed = all(
        row["first_token_exact"]
        and row["first_decode_token_exact"]
        and all(item["finite"] for item in row["metrics"].values())
        for row in rows
    )
    return {
        "axis": "official_prefill_self_repeat",
        "status": "pass" if passed else "fail",
        "official_commit": reference["source_revision"],
        "precision": reference["precision"],
        "batch_size": reference["batch_size"],
        "prompt_tokens": reference["prompt_tokens"],
        "repetitions": len(repeats) + 1,
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
    repeats = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.repeats
    ]
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
