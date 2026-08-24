from __future__ import annotations

import math

import pytest

from bench.probes.probe_sm120_b8_dense_ffn_compile import (
    choose_fastest_passing_candidate,
    correctness_gate_pass,
    is_exact_rtx5090,
    parse_args,
    probe_status,
    strict_saving_pass,
    summarize_us,
    validate_shape,
)


def test_exact_device_requires_name_and_sm120() -> None:
    assert is_exact_rtx5090("NVIDIA GeForce RTX 5090", (12, 0))
    assert not is_exact_rtx5090("NVIDIA GeForce RTX 5070 Laptop GPU", (12, 0))
    assert not is_exact_rtx5090("NVIDIA GeForce RTX 5090", (8, 9))


def test_shape_contract_is_exact_b8_and_supported_hidden() -> None:
    validate_shape(8, 1024, 24)
    validate_shape(8, 2048, 24)
    with pytest.raises(ValueError, match="batch size 8"):
        validate_shape(1, 1024, 24)
    with pytest.raises(ValueError, match="hidden"):
        validate_shape(8, 4096, 24)
    with pytest.raises(ValueError, match="layers"):
        validate_shape(8, 1024, 0)


def test_strict_saving_gate_rejects_equality_and_nonfinite() -> None:
    assert not strict_saving_pass(50.0, 50.0)
    assert strict_saving_pass(math.nextafter(50.0, math.inf), 50.0)
    assert not strict_saving_pass(None, 50.0)
    assert not strict_saving_pass(float("nan"), 50.0)


def test_timing_summary_uses_median_and_rejects_bad_samples() -> None:
    summary = summarize_us([12.0, 10.0, 11.0])
    assert summary["median_us_per_step"] == 11.0
    assert summary["min_us_per_step"] == 10.0
    assert summary["max_us_per_step"] == 12.0
    for samples in ([], [0.0], [float("inf")], [float("nan")]):
        with pytest.raises(ValueError, match="timing samples"):
            summarize_us(samples)


def _passing_correctness() -> dict[str, object]:
    return {
        "vectors_compared": 192,
        "min_cosine": 0.99991,
        "max_abs_diff": 0.003,
        "all_finite": True,
        "argmax_all_equal": True,
    }


def test_correctness_gate_requires_every_semantic_field() -> None:
    assert correctness_gate_pass(_passing_correctness(), min_cosine=0.9999)
    for key, value in (
        ("vectors_compared", 0),
        ("min_cosine", 0.99989),
        ("min_cosine", float("nan")),
        ("all_finite", False),
        ("argmax_all_equal", False),
    ):
        row = _passing_correctness()
        row[key] = value
        assert not correctness_gate_pass(row, min_cosine=0.9999)
    assert not correctness_gate_pass({}, min_cosine=0.9999)


def test_fastest_candidate_must_compile_reuse_and_pass_correctness() -> None:
    rows = [
        {
            "label": "fast_but_wrong",
            "median_us_per_step": 100.0,
            "compile_effective": True,
            "compile_reused": True,
            "correctness_pass": False,
        },
        {
            "label": "passing_slow",
            "median_us_per_step": 120.0,
            "compile_effective": True,
            "compile_reused": True,
            "correctness_pass": True,
        },
        {
            "label": "passing_fast",
            "median_us_per_step": 110.0,
            "compile_effective": True,
            "compile_reused": True,
            "correctness_pass": True,
        },
    ]
    assert choose_fastest_passing_candidate(rows)["label"] == "passing_fast"
    assert choose_fastest_passing_candidate(rows[:1]) is None


def test_proxy_can_never_be_reported_as_5090_pass() -> None:
    assert probe_status(
        exact_rtx5090=False,
        compile_effective=True,
        compile_reused=True,
        correctness_passed=True,
        saving_passed=True,
    ) == ("diagnostic_only", "proxy_device_no_5090_claim")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"compile_effective": False}, "fullgraph_compile_not_effective"),
        ({"compile_reused": False}, "compiled_callable_recompiled_for_layer_weights"),
        ({"correctness_passed": False}, "correctness_gate_failed"),
        ({"saving_passed": False}, "strict_step_saving_not_met"),
    ],
)
def test_status_fails_closed_in_gate_order(
    kwargs: dict[str, bool], expected: str
) -> None:
    values = {
        "exact_rtx5090": True,
        "compile_effective": True,
        "compile_reused": True,
        "correctness_passed": True,
        "saving_passed": True,
    }
    values.update(kwargs)
    status, conclusion = probe_status(**values)
    assert conclusion == expected
    assert status in {"fail", "diagnostic_miss"}


def test_cli_rejects_non_b8_and_duplicate_shapes() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--batch-size", "1"])
    with pytest.raises(SystemExit):
        parse_args(["--hidden-sizes", "1024", "1024"])
