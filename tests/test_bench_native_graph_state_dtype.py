from __future__ import annotations

import os

from bench.bench_native_graph_state_dtype import (
    _set_mode,
    aggregate_mode_results,
    balanced_mode_order,
)


def test_balanced_mode_order_gives_each_mode_both_positions() -> None:
    assert balanced_mode_order(2, candidate_first=False) == (
        False,
        True,
        True,
        False,
    )
    assert balanced_mode_order(2, candidate_first=True) == (
        True,
        False,
        False,
        True,
    )


def test_aggregate_mode_results_uses_balanced_median_and_max_memory() -> None:
    rows = [
        {
            "first_logits": "first",
            "greedy": [1],
            "ms_per_step": 4.0,
            "tokps_total": 250.0,
            "peak_vram_mb": 100.0,
            "route": {"state_dtype": "torch.float16"},
            "cache_stats": {"requests": 1},
        },
        {
            "first_logits": "second",
            "greedy": [2],
            "ms_per_step": 6.0,
            "tokps_total": 166.0,
            "peak_vram_mb": 120.0,
            "route": {"state_dtype": "torch.float16"},
            "cache_stats": {"requests": 2},
        },
    ]
    result = aggregate_mode_results(rows)
    assert result["first_logits"] == "first"
    assert result["ms_per_step"] == 5.0
    assert result["tokps_total"] == 208.0
    assert result["peak_vram_mb"] == 120.0
    assert result["cache_stats"] == {"requests": 2}
    assert result["timing_samples_ms"] == [4.0, 6.0]


def test_set_mode_toggles_decode_feature_without_changing_state(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_STATE_DTYPE", "fp32")
    _set_mode(
        candidate=True,
        force_candidate=False,
        candidate_feature="fused-norm-mix",
    )
    assert os.environ["RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX"] == "1"
    assert os.environ["RWKV7_NATIVE_GRAPH_STATE_DTYPE"] == "fp32"

    _set_mode(
        candidate=False,
        force_candidate=False,
        candidate_feature="fused-norm-mix",
    )
    assert os.environ["RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX"] == "0"
