#!/usr/bin/env python3
from __future__ import annotations

from bench.summarize_blackwell_quant_matrix import acceptance_failures


def row(quant: str, **overrides):
    value = {
        "status": "pass",
        "quantization": quant,
        "decode_speed_ratio_vs_fp16": 1.0,
        "footprint_ratio_vs_fp16": 0.95,
        "prompt_logits_cos_vs_fp16": 0.999,
        "final_logits_cos_vs_fp16": 0.999,
        "same_next_token_as_fp16": True,
    }
    value.update(overrides)
    return value


def test_gate_passes_complete_equivalent_rows() -> None:
    rows = [row("none"), row("mm8"), row("mm4")]
    assert acceptance_failures(rows, expected_rows=3) == []


def test_gate_is_fail_closed() -> None:
    rows = [
        row("none"),
        row("mm8", decode_speed_ratio_vs_fp16=0.98),
        row("mm4", same_next_token_as_fp16=False, footprint_ratio_vs_fp16=None),
    ]
    failures = acceptance_failures(rows, expected_rows=4)
    assert any(item.startswith("row_count=") for item in failures)
    assert any(item.startswith("speed_fail=") for item in failures)
    assert any(item.startswith("footprint_fail=") for item in failures)
    assert any(item.startswith("same_next_fail=") for item in failures)
