from __future__ import annotations

import torch

from scripts.compare_official_native_inference import (
    build_parser,
    compare_captures,
    metrics_pass,
    metrics_pass_official_envelope,
    tensor_metrics,
    verify_official_source,
)
from scripts.compare_official_self_repeats import compare_repeats
from scripts.compare_official_native_prefill import (
    metric_pass as prefill_metric_pass,
    parser as prefill_parser,
)


OFFICIAL_COMMIT = "cc57df475465c6cacd42ecd4f2f05a588ee5473b"


def test_official_capture_exposes_low_memory_runtime_options() -> None:
    args = build_parser().parse_args(
        [
            "capture-official",
            "--hf-dir",
            "hf",
            "--output",
            "capture.pt",
            "--official-emb",
            "cpu",
            "--official-lowrank-weight",
            "transpose",
            "--official-orig-linear-groups",
            "none",
        ]
    )
    assert args.official_emb == "cpu"
    assert args.official_lowrank_weight == "transpose"
    assert args.official_orig_linear_groups == "none"

    prefill_args = prefill_parser().parse_args(
        [
            "--mode",
            "capture-official",
            "--official-emb",
            "cpu",
            "--official-lowrank-weight",
            "transpose",
            "--official-orig-linear-groups",
            "none",
        ]
    )
    assert prefill_args.official_emb == "cpu"
    assert prefill_args.official_lowrank_weight == "transpose"
    assert prefill_args.official_orig_linear_groups == "none"

    prefill_compare = prefill_parser().parse_args(
        [
            "--mode",
            "compare",
            "--official-self-envelope",
            "envelope.json",
            "--official-envelope-multiplier",
            "1.5",
        ]
    )
    assert prefill_compare.official_self_envelope == "envelope.json"
    assert prefill_compare.official_envelope_multiplier == 1.5


def test_official_envelope_gate_is_bounded_by_explicit_multiplier() -> None:
    metrics = {
        "finite": True,
        "max_abs": 0.3,
        "fraction_over_abs_threshold": 2.0e-4,
        "cosine": 0.9999997,
    }
    envelope = {
        "max_abs": 0.25,
        "max_fraction_over_abs_threshold": 1.7e-4,
        "min_cosine": 0.99999975,
    }
    assert metrics_pass_official_envelope(metrics, envelope, multiplier=1.25)
    assert not metrics_pass_official_envelope(metrics, envelope, multiplier=1.0)


def test_tensor_metrics_reports_exact_and_close_values() -> None:
    left = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    exact = tensor_metrics(left, left.clone())
    assert exact["finite"] is True
    assert exact["max_abs"] == 0.0
    assert exact["cosine"] == 1.0

    close = tensor_metrics(left, torch.tensor([[1.0, 2.125]], dtype=torch.float16))
    assert close["max_abs"] == 0.125
    assert close["cosine"] > 0.999
    assert close["count_over_abs_threshold"] == 0


def test_fp16_ulp_tail_gate_is_explicit_and_bounded() -> None:
    official = torch.full((100_000,), -70.0, dtype=torch.float16)
    native = official.clone()
    native[17] = torch.nextafter(
        torch.nextafter(
            torch.nextafter(official[17], torch.tensor(float("inf"), dtype=torch.float16)),
            torch.tensor(float("inf"), dtype=torch.float16),
        ),
        torch.tensor(float("inf"), dtype=torch.float16),
    )
    metrics = tensor_metrics(native, official, absolute_threshold=0.125)
    assert metrics["max_abs"] == 0.1875
    assert metrics["max_abs_ulps_at_max"] == 3.0
    assert metrics_pass(metrics, "logits") is True


def test_prefill_first_decode_reuses_the_bounded_decode_tail_gate() -> None:
    official = torch.full((100_000,), -70.0, dtype=torch.float16)
    native = official.clone()
    native[17] = torch.nextafter(
        torch.nextafter(
            torch.nextafter(official[17], torch.tensor(float("inf"), dtype=torch.float16)),
            torch.tensor(float("inf"), dtype=torch.float16),
        ),
        torch.tensor(float("inf"), dtype=torch.float16),
    )
    metrics = tensor_metrics(native, official, absolute_threshold=0.125)

    assert prefill_metric_pass(metrics, "first_decode_logits") is True
    assert metrics["fp16_tail_pass"] is True
    assert metrics["fixed_abs_pass"] is False
    assert metrics["fp16_tail_pass"] is True

    official = torch.full((100_000,), 4.0, dtype=torch.float16)
    native = official.clone()
    native[17] = 4.15625
    metrics = tensor_metrics(native, official, absolute_threshold=0.125)
    assert metrics["max_abs_ulps_at_max"] > 4.0
    assert metrics_pass(metrics, "logits") is True
    assert metrics["fixed_abs_pass"] is False
    assert metrics["fp16_tail_pass"] is True

    native[:10] = native[17]
    metrics = tensor_metrics(native, official, absolute_threshold=0.125)
    assert metrics_pass(metrics, "logits") is False
    assert metrics["fp16_tail_pass"] is False


def make_capture(engine: str, revision: str) -> dict:
    state = torch.zeros(1, 1, 1, 2, 2, dtype=torch.float16)
    shift = torch.zeros(1, 1, 2, dtype=torch.float16)
    elapsed = (
        torch.ones(1, 1, 1, dtype=torch.int32)
        if engine == "native_hf"
        else torch.ones(1, dtype=torch.int32)
    )
    snapshot = {"state": state, "xpa": shift, "xpf": shift, "elapsed": elapsed}
    return {
        "engine": engine,
        "source_revision": revision,
        "precision": "fp16_state_fp16_io",
        "prompt_tokens": 1,
        "prompt_ids": torch.tensor([[7]]),
        "decode_steps": 1,
        "batch_sizes": [1],
        "captures": {
            "1": {
                "logits": torch.tensor([[[0.0, 2.0]], [[1.0, 3.0]]]),
                "greedy_tokens": torch.tensor([[1]]),
                "prefill": snapshot,
                "final": snapshot,
            }
        },
    }


def test_compare_captures_requires_pin_and_exact_tokens() -> None:
    native = make_capture("native_hf", "native")
    official = make_capture("official_v3a", OFFICIAL_COMMIT)
    report = compare_captures(
        native,
        official,
        expected_official_commit=OFFICIAL_COMMIT,
    )
    assert report["status"] == "pass"
    assert report["rows"][0]["greedy_exact"] is True
    assert report["rows"][0]["logits"]["top1_match_rate"] == 1.0

    official["source_revision"] = "wrong"
    try:
        compare_captures(native, official, expected_official_commit=OFFICIAL_COMMIT)
    except ValueError as exc:
        assert "commit" in str(exc)
    else:
        raise AssertionError("an unpinned official capture must fail")


def test_compare_captures_enforces_numeric_thresholds_even_when_top1_matches() -> None:
    native = make_capture("native_hf", "native")
    official = make_capture("official_v3a", OFFICIAL_COMMIT)
    native["captures"]["1"]["logits"][0, 0] = torch.tensor([-100.0, 2.0])
    report = compare_captures(
        native,
        official,
        expected_official_commit=OFFICIAL_COMMIT,
    )
    assert report["rows"][0]["logits"]["top1_match_rate"] == 1.0
    assert report["rows"][0]["logits"]["threshold_pass"] is False
    assert report["status"] == "fail"


def test_official_self_repeat_reports_numeric_envelope() -> None:
    reference = make_capture("official_v3a", OFFICIAL_COMMIT)
    repeat = make_capture("official_v3a", OFFICIAL_COMMIT)
    repeat["captures"]["1"]["logits"][0, 0, 0] = 0.125

    report = compare_repeats(reference, [repeat])

    assert report["status"] == "pass"
    assert report["envelope"]["logits"]["max_abs"] == 0.125
    assert report["envelope"]["logits"]["total"] == 1
    assert report["rows"][0]["greedy_exact"] is True


def test_official_source_manifest_is_fail_closed(tmp_path) -> None:
    source = tmp_path / "official"
    source.mkdir()
    (source / "kernel.cu").write_text("official", encoding="utf-8")
    digest = __import__("hashlib").sha256(b"official").hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        __import__("json").dumps(
            {"commit": OFFICIAL_COMMIT, "files": {"kernel.cu": digest}}
        ),
        encoding="utf-8",
    )

    verified = verify_official_source(
        source,
        expected_commit=OFFICIAL_COMMIT,
        manifest_path=manifest,
    )
    assert verified["method"] == "sha256_manifest"

    (source / "kernel.cu").write_text("modified", encoding="utf-8")
    try:
        verify_official_source(
            source,
            expected_commit=OFFICIAL_COMMIT,
            manifest_path=manifest,
        )
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("modified official source must fail verification")
