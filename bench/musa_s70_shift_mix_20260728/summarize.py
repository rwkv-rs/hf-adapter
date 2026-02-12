#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (RAW / name).read_text(encoding="utf-8").splitlines()]


def main() -> int:
    rows = []
    for prompt in (128, 512):
        for round_id in (1, 2):
            off = load(f"p{prompt}_r{round_id}_off.jsonl")
            on = load(f"p{prompt}_r{round_id}_on.jsonl")
            for batch in (1, 2, 4, 8):
                baseline = next(
                    row for row in off
                    if row["batch_size"] == batch
                    and row["decode_api"] == "rwkv7_forward_token"
                )
                candidate = next(
                    row for row in on
                    if row["batch_size"] == batch
                    and row["decode_api"] == "rwkv7_forward_token"
                )
                rows.append({
                    "prompt_tokens": prompt,
                    "round": round_id,
                    "batch_size": batch,
                    "prefill_off_tokps": baseline["prefill_tokps_total"],
                    "prefill_on_tokps": candidate["prefill_tokps_total"],
                    "prefill_ratio": candidate["prefill_tokps_total"] / baseline["prefill_tokps_total"],
                    "decode_off_tokps": baseline["decode_tokps_total"],
                    "decode_on_tokps": candidate["decode_tokps_total"],
                    "decode_ratio": candidate["decode_tokps_total"] / baseline["decode_tokps_total"],
                    "peak_off_mb": baseline["peak_vram_mb"],
                    "peak_on_mb": candidate["peak_vram_mb"],
                    "route_calls_off": baseline["musa_attn_shift_mix_calls"],
                    "route_calls_on": candidate["musa_attn_shift_mix_calls"],
                })

    state = json.loads((ROOT / "state-compare.json").read_text(encoding="utf-8"))
    e2e_off = json.loads((ROOT / "e2e-off.json").read_text(encoding="utf-8"))
    e2e_on = json.loads((ROOT / "e2e-on.json").read_text(encoding="utf-8"))
    summary = {
        "scope": {
            "device": "MTT S70",
            "model": "RWKV-7 G1D 0.1B",
            "dtype": "fp16",
            "attention_mode": "chunk",
            "fusion_default": "off",
            "fusion_opt_in": "RWKV7_MUSA_ATTN_SHIFT_MIX=1",
        },
        "correctness": {
            "generated_ids_equal": e2e_off["generated_ids"] == e2e_on["generated_ids"],
            "generated_length": e2e_off["generated_length"],
            "state_compare_passed": state["passed"],
            "logits_equal": state["logits_equal"],
            "all_state_groups_equal": all(group["equal"] for group in state["groups"].values()),
        },
        "paired_cells": rows,
        "aggregate": {
            "cell_count": len(rows),
            "prefill_ratio_median": statistics.median(row["prefill_ratio"] for row in rows),
            "prefill_ratio_min": min(row["prefill_ratio"] for row in rows),
            "prefill_ratio_max": max(row["prefill_ratio"] for row in rows),
            "decode_ratio_median": statistics.median(row["decode_ratio"] for row in rows),
            "decode_ratio_min": min(row["decode_ratio"] for row in rows),
            "decode_ratio_max": max(row["decode_ratio"] for row in rows),
            "peak_memory_equal": all(row["peak_off_mb"] == row["peak_on_mb"] for row in rows),
            "route_valid": all(row["route_calls_off"] == 0 and row["route_calls_on"] > 0 for row in rows),
        },
        "decision": "retain as exact-MTT-S70 fp16 inference-only opt-in experiment; do not enable by default",
    }
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
