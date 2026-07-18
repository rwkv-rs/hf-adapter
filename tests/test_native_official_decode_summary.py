from __future__ import annotations

from bench.summarize_native_official_decode import summarize


def make_row(batch_size: int, speed: float) -> dict:
    return {
        "device": "NVIDIA GeForce RTX 5090",
        "dtype": "fp16",
        "batch_size": batch_size,
        "decode_steps": 512,
        "decode_tokps": speed,
        "greedy_tokens": [[1, 2] for _ in range(batch_size)],
        "requested_extensions": {
            "native_wkv_fp16": {"requested": True, "active": True}
        },
    }


def reference() -> dict:
    return {
        "device": "NVIDIA GeForce RTX 5090",
        "dtype": "fp16",
        "checkpoint": "g1h-7.2B",
        "precision_mode": "fp16_state_fp16_io",
        "official_engine": "rwkv7_fast_v3a.py",
        "official_commit": "abc",
        "official_iterations": 20,
        "native_repetitions": 3,
        "native_decode_steps": 512,
        "batch_sizes": {"1": {"decode_tokps": 100.0}, "8": {"decode_tokps": 800.0}},
    }


def test_summary_requires_every_exact_shape_and_reports_ratios() -> None:
    rows = [make_row(1, speed) for speed in (101, 102, 103)]
    rows += [make_row(8, speed) for speed in (801, 802, 803)]
    report = summarize(rows, reference())
    assert report["status"] == "pass"
    assert report["official_commit"] == "abc"
    assert report["rows"][0]["matched_shape_ratio"] == 1.02
    assert len(report["rows"][1]["greedy_trace_sha256"]) == 3
    assert report["rows"][1]["repeat_traces_equal"] is True


def test_summary_fails_a_slow_shape_or_inactive_extension() -> None:
    rows = [make_row(1, speed) for speed in (99, 99, 99)]
    rows += [make_row(8, speed) for speed in (801, 802, 803)]
    rows[-1]["requested_extensions"]["native_wkv_fp16"]["active"] = False
    report = summarize(rows, reference())
    assert report["status"] == "fail"
    assert report["rows"][0]["status"] == "fail"
    assert report["rows"][1]["requested_extensions_active"] is False


def test_summary_fails_when_repetitions_change_greedy_trace() -> None:
    rows = [make_row(1, speed) for speed in (101, 102, 103)]
    rows += [make_row(8, speed) for speed in (801, 802, 803)]
    rows[1]["greedy_tokens"] = [[1, 3]]

    report = summarize(rows, reference())

    assert report["status"] == "fail"
    assert report["rows"][0]["repeat_traces_equal"] is False
