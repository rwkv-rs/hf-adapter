from __future__ import annotations

import torch

from scripts.compare_official_native_inference import (
    build_parser,
    compare_captures,
    metrics_pass,
    metrics_pass_fp16_trajectory,
    metrics_pass_official_envelope,
    snapshot_native,
    tensor_metrics,
    verify_official_source,
)
from scripts.compare_official_native_prefill import (
    metric_pass_official_self_envelope,
)
from scripts.compare_official_self_repeats import compare_repeats
from scripts.compare_official_native_prefill import (
    compare as compare_prefill,
    metric_pass as prefill_metric_pass,
    native_runtime_environment,
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
    assert args.official_wkv == "fp32io16"
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


def test_prefill_envelope_only_applies_to_first_decode_logits() -> None:
    metrics = {
        "finite": True,
        "max_abs": 0.15,
        "fraction_over_abs_threshold": 0.00006,
        "cosine": 0.9999999,
    }
    envelope = {
        "max_abs": 0.20,
        "max_fraction_over_abs_threshold": 0.0001,
        "min_cosine": 0.9999998,
    }

    assert metric_pass_official_self_envelope(
        "first_decode_logits", metrics, envelope, multiplier=1.25
    )
    assert not metric_pass_official_self_envelope(
        "logits", metrics, envelope, multiplier=1.25
    )
    assert not metric_pass_official_self_envelope(
        "layer_outputs", metrics, envelope, multiplier=1.25
    )


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


def test_fp16_trajectory_gate_has_independent_mean_cosine_and_max_bounds() -> None:
    official = torch.full((4096,), 32.0, dtype=torch.float16)
    native = official.clone()
    native[:128] += 0.25
    metrics = tensor_metrics(native, official, absolute_threshold=0.125)

    assert metrics_pass(metrics, "logits") is False
    assert metrics_pass_fp16_trajectory(metrics, "logits") is True

    native[0] = 34.0
    metrics = tensor_metrics(native, official, absolute_threshold=0.125)
    assert metrics_pass_fp16_trajectory(metrics, "logits") is False

    fp32_metrics = tensor_metrics(native.float(), official.float(), absolute_threshold=0.125)
    assert metrics_pass_fp16_trajectory(fp32_metrics, "logits") is False


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


def test_prefill_shift_states_reuse_the_bounded_decode_tail_gate() -> None:
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

    assert prefill_metric_pass(metrics, "xpf") is True
    assert metrics["fixed_abs_pass"] is False
    assert metrics["fp16_tail_pass"] is True


def test_prefill_capture_records_only_explicit_native_runtime_controls(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_GRAPH", "1")
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_STATE_DTYPE", "fp16")
    monkeypatch.setenv("RWKV7_UNRELATED", "ignored")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/tmp/extensions")

    runtime = native_runtime_environment()
    assert runtime["RWKV7_NATIVE_GRAPH_STATE_DTYPE"] == "fp16"
    assert runtime["RWKV7_NATIVE_PREFILL_GRAPH"] == "1"
    assert "RWKV7_UNRELATED" not in runtime
    assert "TORCH_EXTENSIONS_DIR" not in runtime


def test_prefill_report_records_capture_hashes_and_source_metadata(tmp_path) -> None:
    prompt = torch.tensor([[7]])
    state = torch.zeros(1, 1, 1, 2, 2, dtype=torch.float16)
    shift = torch.zeros(1, 1, 2, dtype=torch.float16)

    def capture(engine: str, revision: str) -> dict:
        return {
            "engine": engine,
            "source_revision": revision,
            "source_verification": {"method": "sha256_manifest"},
            "runtime": {"emb": "cpu"},
            "precision": "fp16_state_fp16_io",
            "batch_size": 1,
            "prompt_tokens": 1,
            "prompt_ids": prompt,
            "seen_tokens": 1,
            "logits": torch.zeros(1, 1, 2, dtype=torch.float16),
            "first_decode_logits": torch.zeros(1, 1, 2, dtype=torch.float16),
            "first_token": torch.zeros(1, 1, dtype=torch.long),
            "first_decode_token": torch.zeros(1, 1, dtype=torch.long),
            "prefill": {"state": state, "xpa": shift, "xpf": shift},
            "layer_outputs": torch.zeros(1, 1, 2, dtype=torch.float16),
            "timing": {"aggregate_tokps": 1.0},
            "peak_vram_mb": 1.0,
            "fp16_recurrent_effective": True,
            "prefill_state_dtype": "torch.float16",
            "first_decode_state_dtype": "torch.float16",
            "prefill_backend": "native_graph",
            "first_decode_backend": "native_graph",
            "stacked_rkv_effective": True,
            "wavg_lora_effective": True,
            "sequence_ffn_effective": True,
            "fp16_accum_ffn_key_effective": True,
            "runtime_env": {"RWKV7_NATIVE_PREFILL_GRAPH": "1"},
        }

    native_path = tmp_path / "native.pt"
    official_path = tmp_path / "official.pt"
    torch.save(capture("native_hf", "native-revision"), native_path)
    torch.save(capture("official_v3a", OFFICIAL_COMMIT), official_path)
    args = prefill_parser().parse_args(
        [
            "--mode",
            "compare",
            "--native-capture",
            str(native_path),
            "--official-capture",
            str(official_path),
        ]
    )

    report = compare_prefill(args)

    assert len(report["native_capture_sha256"]) == 64
    assert len(report["official_capture_sha256"]) == 64
    assert report["native"]["source_revision"] == "native-revision"
    assert report["official"]["source_revision"] == OFFICIAL_COMMIT
    assert report["official"]["runtime"] == {"emb": "cpu"}
    assert report["official"]["source_verification"] == {
        "method": "sha256_manifest"
    }


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


def test_compare_captures_accepts_bounded_fp16_drift_only_with_exact_trajectory() -> None:
    native = make_capture("native_hf", "native")
    official = make_capture("official_v3a", OFFICIAL_COMMIT)
    for capture in (native, official):
        capture["captures"]["1"]["logits"] = capture["captures"]["1"][
            "logits"
        ].repeat(1, 1, 64).half()
        for phase in ("prefill", "final"):
            capture["captures"]["1"][phase]["xpa"] = capture["captures"]["1"][
                phase
            ]["xpa"].half()
            capture["captures"]["1"][phase]["xpf"] = capture["captures"]["1"][
                phase
            ]["xpf"].half()
    native["captures"]["1"]["logits"][0, 0, 0] = 0.1875

    report = compare_captures(
        native,
        official,
        expected_official_commit=OFFICIAL_COMMIT,
    )

    assert report["rows"][0]["standard_quality_pass"] is False
    assert report["rows"][0]["fp16_trajectory_quality_pass"] is True
    assert report["status"] == "pass"

    native["captures"]["1"]["greedy_tokens"][0, 0] = 0
    report = compare_captures(
        native,
        official,
        expected_official_commit=OFFICIAL_COMMIT,
    )
    assert report["rows"][0]["fp16_trajectory_quality_pass"] is True
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


def test_official_source_accepts_a_clean_git_subdirectory(tmp_path) -> None:
    import subprocess

    root = tmp_path / "official"
    source = root / "engine"
    source.mkdir(parents=True)
    (source / "kernel.cu").write_text("official", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "engine/kernel.cu"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    verified = verify_official_source(
        source,
        expected_commit=revision,
        manifest_path=None,
    )

    assert verified["method"] == "git"
    assert verified["subdirectory"] == "engine"


def test_native_snapshot_uses_cache_position_without_a_graph_runner() -> None:
    class Config:
        num_heads = 2
        num_hidden_layers = 3

    class Model:
        config = Config()

    class Cache:
        _state = [torch.zeros(1, 2, 2, 2) for _ in range(3)]
        _xpa = [torch.zeros(1, 4) for _ in range(3)]
        _xpf = [torch.zeros(1, 4) for _ in range(3)]

        def get_batch_size(self):
            return 1

        def get_seq_length(self):
            return 17

    snapshot = snapshot_native(Model(), Cache())
    assert snapshot["elapsed"].shape == (3, 1, 2)
    assert torch.equal(
        snapshot["elapsed"],
        torch.full((3, 1, 2), 17, dtype=torch.int32),
    )
