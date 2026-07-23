#!/usr/bin/env python3
"""Fail-closed checks for the exact-sm70 W4 prefill artifact."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict:
    rows = [
        json.loads(line)
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1, f"{name}: expected one row, got {len(rows)}"
    return rows[0]


def common(row: dict, *, batch: int) -> None:
    assert row["status"] == "pass"
    assert row["device"] == "Tesla V100-PCIE-32GB"
    assert row["dtype"] == "fp16"
    assert row["model_size_label"] == "1.5b"
    assert row["batch_size"] == batch
    assert row["prompt_tokens"] == 128
    assert row["decode_tokens"] == 128
    assert row["paired_baseline"] is True
    assert row["timing_repeats"] >= 5
    assert row["same_next_token_as_fp16"] is True
    assert row["same_greedy_tokens_as_fp16"] is True
    assert row["greedy_repeat_deterministic"] is True


def main() -> int:
    for batch in (1, 8):
        old = load(f"dp4a_b{batch}.jsonl")
        new = load(f"auto_b{batch}.jsonl")
        common(old, batch=batch)
        common(new, batch=batch)
        for row in (old, new):
            assert row["native_mm_policy"] == "memory"
            assert row["native_mm4_group_size"] == 128
            assert row["native_mm4_group_policy"] == "lm_head"
            assert row["replaced_modules"] == 49
            assert row["footprint_ratio_vs_fp16"] < 1.0
            assert row["decode_speed_ratio_vs_fp16"] >= 1.0
            assert row["final_logits_cos_vs_fp16"] >= 0.998
        assert new["prefill_tokps_total"] > old["prefill_tokps_total"] * 2.5
        assert new["prefill_speed_ratio_vs_fp16"] >= (0.78 if batch == 1 else 0.90)

    speed_files = {
        1: "speed_group256_b1.jsonl",
        2: "speed_group256_b2.jsonl",
        4: "speed_group256_b4.jsonl",
        8: "speed_b8_group256_true.jsonl",
    }
    for batch, name in speed_files.items():
        row = load(name)
        common(row, batch=batch)
        assert row["native_mm_policy"] == "speed"
        assert row["native_mm4_group_size"] == 256
        assert row["native_mm4_group_policy"] == "lm_head"
        assert row["native_mm_kernel"] == "sm7x_dp4a_w4_group256"
        assert row["replaced_modules"] == 1
        assert row["footprint_ratio_vs_fp16"] < 1.0
        assert row["prefill_speed_ratio_vs_fp16"] >= 1.0
        assert row["decode_speed_ratio_vs_fp16"] >= 1.0
        assert row["prompt_logits_cos_vs_fp16"] >= 0.9995
        assert row["final_logits_cos_vs_fp16"] >= 0.9995

    print("PASS: 2/2 memory-prefill improvements and 4/4 group256 speed cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
