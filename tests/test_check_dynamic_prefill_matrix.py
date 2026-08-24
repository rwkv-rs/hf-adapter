from __future__ import annotations

from bench.validators.check_dynamic_prefill_matrix import analyze, parse_int_set


def _row(batch: int, prompt: int, *, ms: float, tokps: float) -> dict:
    return {
        "status": "pass",
        "batch_size": batch,
        "prompt_tokens": prompt,
        "native_prefill_ms": ms,
        "native_prefill_tokps_total": tokps,
        "fused_scan_effective": True,
        "prefill_graph_effective": batch in {1, 2, 4, 8} and prompt == 128,
        "greedy_match": True,
        "decode_after_prefill_greedy_match": True,
        "prefill_fused_shift_mix_effective": True,
        "prefill_fused_state_prep_effective": True,
        "prefill_fused_output_effective": True,
    }


def test_parse_int_set_accepts_ranges_and_deduplicates() -> None:
    assert parse_int_set("1-3,3,5") == [1, 2, 3, 5]


def test_dynamic_prefill_matrix_passes_continuous_shapes() -> None:
    rows = [
        _row(batch, prompt, ms=10.0 + batch, tokps=20_000.0 + prompt)
        for batch in (1, 2, 3, 4)
        for prompt in (127, 128, 129)
    ]
    summary = analyze(
        rows,
        batches=(1, 2, 3, 4),
        prompts=(127, 128, 129),
        require_safe_fusions=True,
    )
    assert summary["status"] == "pass"
    assert summary["observed_shape_count"] == 12
    assert [3, 128] in summary["dynamic_shapes"]
    assert [4, 128] in summary["graph_shapes"]


def test_dynamic_prefill_matrix_rejects_missing_route_and_padding_cliff() -> None:
    rows = [
        _row(1, 128, ms=10.0, tokps=10_000.0),
        _row(2, 128, ms=12.0, tokps=18_000.0),
        _row(3, 128, ms=300.0, tokps=1_000.0),
        _row(4, 128, ms=20.0, tokps=22_000.0),
    ]
    rows[2]["fused_scan_effective"] = False
    rows[2]["prefill_fused_state_prep_effective"] = False
    summary = analyze(
        rows,
        batches=(1, 2, 3, 4),
        prompts=(128,),
        require_safe_fusions=True,
    )
    assert summary["status"] == "fail"
    assert {item["kind"] for item in summary["failures"]} >= {
        "route",
        "padding_cliff",
    }
    assert any(
        item.get("field") == "prefill_fused_state_prep_effective"
        for item in summary["failures"]
    )


def test_dynamic_prefill_matrix_rejects_prompt_boundary_cliff_and_missing_shape() -> None:
    rows = [
        _row(1, 127, ms=10.0, tokps=20_000.0),
        _row(1, 128, ms=10.0, tokps=1_000.0),
    ]
    summary = analyze(rows, batches=(1,), prompts=(127, 128, 129))
    assert summary["status"] == "fail"
    assert {item["kind"] for item in summary["failures"]} >= {
        "missing_shape",
        "cross_route_boundary_cliff",
    }


def test_dynamic_prefill_matrix_allows_bounded_exact_graph_bonus() -> None:
    rows = [
        _row(1, 127, ms=24.0, tokps=5_000.0),
        _row(1, 128, ms=9.0, tokps=13_500.0),
        _row(1, 129, ms=24.5, tokps=5_100.0),
    ]
    summary = analyze(rows, batches=(1,), prompts=(127, 128, 129))

    assert summary["status"] == "pass"
    assert summary["observed_worst_boundary_throughput_ratio"] < 1.35
    assert 2.6 < summary["observed_worst_cross_route_boundary_ratio"] < 3.0


def test_dynamic_prefill_matrix_rejects_severe_cross_route_cliff() -> None:
    rows = [
        _row(1, 127, ms=40.0, tokps=4_000.0),
        _row(1, 128, ms=8.0, tokps=16_000.0),
        _row(1, 129, ms=41.0, tokps=4_100.0),
    ]
    summary = analyze(rows, batches=(1,), prompts=(127, 128, 129))

    assert summary["status"] == "fail"
    assert any(
        item["kind"] == "cross_route_boundary_cliff"
        for item in summary["failures"]
    )


def test_dynamic_prefill_matrix_uses_latest_native_scan_rerun() -> None:
    failed = _row(8, 2049, ms=6000.0, tokps=2_700.0)
    failed["fused_scan_effective"] = False
    fixed = _row(8, 2049, ms=620.0, tokps=26_400.0)

    summary = analyze(
        [failed, fixed],
        batches=(8,),
        prompts=(2049,),
        require_safe_fusions=True,
    )

    assert summary["status"] == "pass"
