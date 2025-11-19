#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.bench_logit_compression_alignment import evaluate_quality_gates


def test_quality_gate_passes_ratio_and_coverage() -> None:
    gates = evaluate_quality_gates(
        {"candidate_over_reference_bits_ratio": 1.0015, "tokens": 950},
        max_bits_ratio=1.01,
        min_scored_tokens=900,
    )
    assert gates["overall_pass"] is True
    assert gates["max_candidate_over_reference_bits_ratio"]["passed"] is True
    assert gates["minimum_scored_tokens"]["passed"] is True


def test_quality_gate_fails_closed() -> None:
    missing = evaluate_quality_gates(
        {"candidate_over_reference_bits_ratio": None, "tokens": 950},
        max_bits_ratio=None,
        min_scored_tokens=1,
    )
    assert missing["overall_pass"] is False

    degraded = evaluate_quality_gates(
        {"candidate_over_reference_bits_ratio": 1.02, "tokens": 8},
        max_bits_ratio=1.01,
        min_scored_tokens=900,
    )
    assert degraded["overall_pass"] is False
    assert degraded["max_candidate_over_reference_bits_ratio"]["passed"] is False
    assert degraded["minimum_scored_tokens"]["passed"] is False
