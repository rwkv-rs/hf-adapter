import os

import torch

from bench.bench_native_model_decode import (
    greedy_trace_sha256,
    requested_extension_status,
    summarize_iteration_times,
)
from bench.bench_native_model_decode_alignment import (
    candidate_environment,
    compare_traces,
    decode_environment,
)
from rwkv7_hf import blackwell_norm_mix, native_wkv_fp16


def test_decode_environment_restores_managed_values(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_RKV_POLICY", "original")
    monkeypatch.delenv("RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN", raising=False)
    with decode_environment(
        {
            "RWKV7_NATIVE_GRAPH_RKV_POLICY": "manual",
            "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN": "1",
        }
    ):
        assert os.environ["RWKV7_NATIVE_GRAPH_RKV_POLICY"] == "manual"
        assert os.environ["RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN"] == "1"
    assert os.environ["RWKV7_NATIVE_GRAPH_RKV_POLICY"] == "original"
    assert "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN" not in os.environ


def test_candidate_environment_exposes_boundary_tuning() -> None:
    values = candidate_environment(
        sparse_ffn=True,
        wag_lora=False,
        wagv_lora=True,
        ada_linear=False,
        fp32_accum=False,
        official_boundary=True,
        deterministic_splits=4,
        num_warps=4,
        precompute_embedding=True,
        rkv_policy="vkwr_auto",
    )
    assert values["RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_FP32_ACCUM"] == "0"
    assert values["RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_OFFICIAL_BOUNDARY"] == "1"
    assert values["RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_DETERMINISTIC_SPLITS"] == "4"
    assert values["RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS"] == "4"
    assert values["RWKV7_NATIVE_GRAPH_PRECOMPUTE_EMB_LN0"] == "1"
    assert values["RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN"] == "1"
    assert values["RWKV7_NATIVE_GRAPH_ADA_WAG_LORA"] == "0"
    assert values["RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA"] == "1"
    assert values["RWKV7_NATIVE_GRAPH_ADA_LINEAR"] == "0"
    assert values["RWKV7_NATIVE_GRAPH_RKV_POLICY"] == "vkwr_auto"


def test_compare_traces_reports_cosine_and_top1_contract() -> None:
    baseline = [torch.tensor([[4.0, 1.0], [1.0, 5.0]])]
    candidate = [torch.tensor([[3.9, 1.1], [1.1, 4.9]])]
    metrics = compare_traces(baseline, candidate)
    assert metrics["logits_finite"] is True
    assert metrics["top1_matches"] == 2
    assert metrics["top1_total"] == 2
    assert metrics["top1_match_rate"] == 1.0
    assert metrics["min_logits_cosine"] > 0.999
    assert metrics["max_logits_abs_diff"] < 0.11


def test_fp16_state_extension_is_reported_and_built(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_STATE_DTYPE", "fp16")
    monkeypatch.setattr(
        native_wkv_fp16,
        "native_fp16_recurrent_available",
        lambda *, build=False: calls.append(build) or True,
    )
    monkeypatch.setattr(
        native_wkv_fp16,
        "native_fp16_recurrent_build_error",
        lambda: None,
    )

    status = requested_extension_status("cuda")

    assert calls == [True]
    assert status["native_wkv_fp16"] == {
        "requested": True,
        "active": True,
        "error": None,
    }


def test_blackwell_norm_extension_is_reported_and_built(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX", "1")
    monkeypatch.setattr(
        blackwell_norm_mix,
        "blackwell_norm_mix_available",
        lambda *, build=False: calls.append(build) or True,
    )
    monkeypatch.setattr(
        blackwell_norm_mix,
        "blackwell_norm_mix_build_error",
        lambda: None,
    )

    status = requested_extension_status("cuda")

    assert calls == [True]
    assert status["blackwell_norm_mix"] == {
        "requested": True,
        "active": True,
        "error": None,
    }


def test_graph_replay_summary_matches_official_quantile_method() -> None:
    summary = summarize_iteration_times([8.0, 9.0, 10.0], batch_size=8)
    assert summary["p10_ms"] == 8.2
    assert summary["p50_ms"] == 9.0
    assert summary["p90_ms"] == 9.8
    assert summary["decode_tokps"] == 888.89


def test_greedy_trace_hash_is_canonical_and_batch_sensitive() -> None:
    first = greedy_trace_sha256([[1, 2], [1, 2]])
    assert first == greedy_trace_sha256([[1, 2], [1, 2]])
    assert first != greedy_trace_sha256([[1, 2], [1, 3]])
