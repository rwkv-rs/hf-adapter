from __future__ import annotations

import torch

from scripts.compare_official_native_inference import (
    compare_captures,
    tensor_metrics,
    verify_official_source,
)


OFFICIAL_COMMIT = "cc57df475465c6cacd42ecd4f2f05a588ee5473b"


def test_tensor_metrics_reports_exact_and_close_values() -> None:
    left = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    exact = tensor_metrics(left, left.clone())
    assert exact["finite"] is True
    assert exact["max_abs"] == 0.0
    assert exact["cosine"] == 1.0

    close = tensor_metrics(left, torch.tensor([[1.0, 2.125]], dtype=torch.float16))
    assert close["max_abs"] == 0.125
    assert close["cosine"] > 0.999


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
