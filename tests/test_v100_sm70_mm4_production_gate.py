from __future__ import annotations

import json
from pathlib import Path

from bench.run_v100_sm70_mm4_production_matrix import (
    CELLS,
    acceptance_failures,
    cell_policy,
    row_matches_configuration,
)


EVIDENCE_ROOT = (
    Path(__file__).parents[1] / "bench" / "v100_sm70_mm4_bntn_20260716"
)
PROMOTED_EVIDENCE = {
    "production_1p5b_memory_fused.jsonl": ("1.5b", "memory", 128, True),
    "production_2p9b_group256_speed.jsonl": ("2.9b", "speed", 256, False),
    "production_7p2b_memory.jsonl": ("7.2b", "memory", 128, False),
}


def passing_row() -> dict[str, object]:
    return {
        "decode_speed_ratio_vs_fp16": 1.0,
        "footprint_ratio_vs_fp16": 0.54,
        "final_logits_cos_vs_fp16": 0.998,
        "same_greedy_tokens_as_fp16": True,
        "greedy_repeat_deterministic": True,
    }


def test_v100_sm70_mm4_production_gate_passes_only_complete_row() -> None:
    assert acceptance_failures(passing_row()) == []


def test_v100_sm70_mm4_production_gate_is_fail_closed() -> None:
    checks = {
        "decode_speed_ratio_vs_fp16": (0.9999, "decode"),
        "footprint_ratio_vs_fp16": (1.0, "footprint"),
        "final_logits_cos_vs_fp16": (0.9979, "logits"),
        "same_greedy_tokens_as_fp16": (False, "greedy"),
        "greedy_repeat_deterministic": (False, "repeat_determinism"),
    }
    for field, (value, expected) in checks.items():
        row = passing_row()
        row[field] = value
        assert expected in acceptance_failures(row)


def test_v100_sm70_mm4_policy_is_exact_cell_scoped() -> None:
    assert cell_policy("2.9b", 4, 128, 128) == "speed"
    assert cell_policy("2.9b", 2, 128, 128) == "memory"
    assert cell_policy("1.5b", 4, 128, 128) == "memory"


def test_v100_sm70_mm4_does_not_reuse_wrong_policy_row() -> None:
    row = {
        "native_mm_policy": "speed",
        "native_mm4_group_size": 256,
        "native_mm4_group_policy": "lm_head",
    }
    row["sm70_w4_fused_epilogue"] = False
    assert row_matches_configuration(row, "speed", 256, "lm_head", False)
    assert not row_matches_configuration(row, "memory", 256, "lm_head", False)
    assert not row_matches_configuration(row, "speed", 128, "lm_head", False)
    assert not row_matches_configuration(row, "speed", 256, "all", False)
    assert not row_matches_configuration(row, "speed", 256, "lm_head", True)
    assert not row_matches_configuration({}, "memory", 128, "lm_head", False)


def test_promoted_v100_sm70_mm4_evidence_is_complete_and_fail_closed() -> None:
    expected_cells = set(CELLS)
    for filename, (label, policy, group_size, fused_epilogue) in (
        PROMOTED_EVIDENCE.items()
    ):
        rows = [
            json.loads(line)
            for line in (EVIDENCE_ROOT / filename)
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        assert len(rows) == len(CELLS), filename
        assert {
            (
                int(row["batch_size"]),
                int(row["prompt_tokens"]),
                int(row["decode_tokens"]),
            )
            for row in rows
        } == expected_cells
        for row in rows:
            assert row["status"] == "pass"
            assert row["model_size_label"] == label
            assert row_matches_configuration(
                row, policy, group_size, "lm_head", fused_epilogue
            )
            assert acceptance_failures(row) == []
