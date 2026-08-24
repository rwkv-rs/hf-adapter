from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from bench.probes.bench_native_graph_ada_wagv_lora import (
    greedy_match_summary,
    logits_pair_metrics,
    model_metadata,
    wagv_bmm_route_pass,
    wagv_bmm_route_status,
    wagv_extension_status,
    wagv_mode_flags,
)


def test_model_metadata_identifies_checkpoint_shape() -> None:
    args = SimpleNamespace(hf_dir="../models/rwkv7-g1g-1.5b-hf")
    model = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=24,
            head_dim=64,
            num_heads=32,
        )
    )

    assert model_metadata(args, model) == {
        "model_name": "rwkv7-g1g-1.5b-hf",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 24,
        "head_dim": 64,
        "num_heads": 32,
    }


def test_extension_status_reports_build_and_error(monkeypatch) -> None:
    package = "fake_wagv_bench_package"
    model_type = type("FakeModel", (), {"__module__": package + ".native_model"})
    module = types.ModuleType(package + ".ada_lora")
    module.ada_wagv_lora_available = lambda device, build: device == "cuda" and build
    module.ada_wagv_lora_build_error = lambda device: None
    monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    monkeypatch.setitem(sys.modules, package + ".ada_lora", module)

    assert wagv_extension_status(model_type(), "cuda") == {
        "wagv_extension_available": True,
        "wagv_extension_error": None,
    }


@pytest.mark.parametrize(
    ("axis", "enabled", "expected"),
    [
        ("ada_wagv_lora", False, (False, False, False)),
        ("ada_wagv_lora", True, (True, False, False)),
        ("ada_wagv_bmm", False, (True, False, False)),
        ("ada_wagv_bmm", True, (True, True, False)),
        ("ada_wagv_bmm_from_default", False, (False, False, False)),
        ("ada_wagv_bmm_from_default", True, (True, True, False)),
        ("sm120_wagv_bmm_g", False, (True, True, False)),
        ("sm120_wagv_bmm_g", True, (True, True, True)),
    ],
)
def test_wagv_mode_flags(
    axis: str, enabled: bool, expected: tuple[bool, bool, bool]
) -> None:
    assert wagv_mode_flags(axis, enabled) == expected


def test_wagv_mode_flags_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="unsupported WAGV benchmark axis"):
        wagv_mode_flags("unknown", True)


def test_bmm_route_status_reads_captured_runner_not_extension_flag() -> None:
    class Model:
        @staticmethod
        def rwkv7_native_graph_runner_copy_stats():
            return {
                "runners": [
                    {
                        "batch_size": 8,
                        "ada_wagv_bmm_requested": True,
                        "ada_wagv_bmm_selected": True,
                        "ada_wagv_bmm_effective": True,
                        "ada_wagv_bmm_selected_layers": [0, 1],
                        "ada_wagv_bmm_effective_layers": [0, 1],
                        "ada_wagv_bmm_effective_layer_count": 2,
                        "ada_wagv_bmm_full_model_effective": True,
                    }
                ]
            }

    status = wagv_bmm_route_status(Model(), 8)
    assert status == {
        "requested": True,
        "selected": True,
        "effective": True,
        "selected_layers": [0, 1],
        "effective_layers": [0, 1],
        "effective_layer_count": 2,
        "full_model_effective": True,
    }
    assert wagv_bmm_route_pass(status, requested=True, num_layers=2)

    sm120_status = wagv_bmm_route_status(
        SimpleNamespace(
            rwkv7_native_graph_runner_copy_stats=lambda: {
                "runners": [
                    {
                        "batch_size": 8,
                        "sm120_wagv_bmm_g_requested": True,
                        "sm120_wagv_bmm_g_selected": True,
                        "sm120_wagv_bmm_g_effective": True,
                        "sm120_wagv_bmm_g_selected_layers": [0, 1],
                        "sm120_wagv_bmm_g_effective_layers": [0, 1],
                        "sm120_wagv_bmm_g_effective_layer_count": 2,
                        "sm120_wagv_bmm_g_full_model_effective": True,
                    }
                ]
            }
        ),
        8,
        route_prefix="sm120_wagv_bmm_g",
    )
    assert wagv_bmm_route_pass(sm120_status, requested=True, num_layers=2)


def test_bmm_route_pass_rejects_requested_fallback() -> None:
    status = {
        "requested": True,
        "selected": False,
        "effective": False,
        "effective_layer_count": 0,
        "full_model_effective": False,
    }
    assert not wagv_bmm_route_pass(status, requested=True, num_layers=24)


def test_bmm_route_pass_requires_exact_selected_and_effective_layers() -> None:
    status = {
        "requested": True,
        "selected": True,
        "effective": True,
        "selected_layers": [0, 1],
        "effective_layers": [0, 2],
        "effective_layer_count": 2,
        "full_model_effective": True,
    }
    assert not wagv_bmm_route_pass(status, requested=True, num_layers=2)


def test_greedy_match_summary_is_boolean_and_fails_closed() -> None:
    assert greedy_match_summary([1, 2, 3], [1, 2, 3]) == (True, 3, 3)
    assert greedy_match_summary([1, 2, 3], [1, 2]) == (False, 2, 3)
    assert greedy_match_summary([], []) == (False, 0, 0)


def test_logits_pair_metrics_records_finite_alignment() -> None:
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metrics = logits_pair_metrics(reference, reference.clone(), batch_size=2)

    assert metrics["finite"] is True
    assert metrics["min_cosine"] == pytest.approx(1.0)
    assert metrics["max_abs_diff"] == 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_logits_pair_metrics_fails_closed_on_nonfinite(value: float) -> None:
    reference = torch.tensor([[1.0, 2.0]])
    candidate = torch.tensor([[1.0, value]])

    assert logits_pair_metrics(reference, candidate, batch_size=1) == {
        "finite": False,
        "min_cosine": None,
        "max_abs_diff": None,
    }
