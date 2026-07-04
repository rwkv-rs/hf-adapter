from __future__ import annotations

import sys
from pathlib import Path

import torch


FLA_BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks" / "fla"
sys.path.insert(0, str(FLA_BENCHMARK))

from compare import (  # noqa: E402
    metrics,
    reference_backend_context,
    state_metrics,
    thresholds,
)


def test_reference_backend_context_is_shared_by_reference_and_optional_lines(
    monkeypatch,
):
    try:
        from rwkv7_hf.kernel_bridge import current_backend_mode
    except ModuleNotFoundError:
        with reference_backend_context():
            pass
        return

    monkeypatch.setenv("RWKV7_BACKEND", "optimized")
    assert current_backend_mode() == "optimized"
    with reference_backend_context():
        assert current_backend_mode() == "reference"
    assert current_backend_mode() == "optimized"


def test_low_precision_thresholds_match_release_contract():
    comparison = {
        "finite": True,
        "cosine": 0.99995,
        "max_abs": 1.0,
        "fp32_allclose": False,
    }
    assert thresholds("bf16", comparison, logits=True)
    assert thresholds("fp16", comparison, logits=False)
    assert not thresholds("fp16", comparison, logits=True)


def test_fp32_threshold_uses_allclose_result():
    comparison = {
        "finite": True,
        "cosine": 1.0,
        "max_abs": 2e-4,
        "fp32_allclose": True,
    }
    assert thresholds("fp32", comparison, logits=True)
    comparison["fp32_allclose"] = False
    assert not thresholds("fp32", comparison, logits=False)


def test_state_metrics_records_full_metrics_and_layout():
    reference = [torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)]
    candidate = [reference[0].transpose(-1, -2).contiguous()]
    row = state_metrics(reference, candidate)[0]
    assert row["layout"] == "fla_transposed"
    assert row["max_abs"] == 0.0
    assert row["cosine"] == 1.0
    assert row["finite"]
    assert row["fp32_allclose"]


def test_cosine_is_numerically_bounded():
    value = torch.linspace(-10, 10, 1_000_000)
    row = metrics(value, value.clone())
    assert 0.999999999 <= row["cosine"] <= 1.0
