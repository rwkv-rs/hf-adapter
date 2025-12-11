import os

import torch

from bench.bench_native_model_decode_alignment import compare_traces, decode_environment


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
