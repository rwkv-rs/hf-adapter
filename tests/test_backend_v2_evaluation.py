from __future__ import annotations

import sys
from pathlib import Path

import torch


EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))

from benchmark_backend_v2 import add_speedups, percentile, routes_passed  # noqa: E402
from fla_common import (  # noqa: E402
    compare_states,
    gradient_metrics,
    gradient_rows_passed,
    tensor_metric,
)


def test_tensor_metric_preserves_tokenwise_argmax():
    reference = torch.tensor([[[0.0, 2.0], [4.0, 1.0]]])
    same_choices = torch.tensor([[[0.1, 1.9], [3.9, 1.1]]])
    different_choice = torch.tensor([[[2.1, 1.9], [3.9, 1.1]]])
    assert tensor_metric(same_choices, reference)["argmax_same"]
    assert not tensor_metric(different_choice, reference)["argmax_same"]


def test_compare_states_detects_fla_vk_layout():
    reference = [torch.arange(12).reshape(1, 1, 3, 4).float()]
    candidate = [{"recurrent_state": reference[0].transpose(-1, -2)}]
    rows = compare_states(candidate, reference)
    assert rows[0]["layout"] == "candidate_transposed"
    assert rows[0]["max_abs"] == 0.0


def test_gradient_report_rejects_missing_and_accepts_close_rows():
    reference = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    candidate = {name: value.clone() for name, value in reference.items()}
    report = gradient_metrics(candidate, reference)
    assert gradient_rows_passed(report, torch.bfloat16)
    report = gradient_metrics({"a": candidate["a"]}, reference)
    assert not gradient_rows_passed(report, torch.bfloat16)


def test_speed_report_and_actual_route_gate():
    prefill_route = {
        "selected": "optimized",
        "implementation": "native-nvidia-prefill-v2[self_chunk]",
    }
    decode_route = {
        "selected": "optimized",
        "implementation": "native-nvidia-fused-decode-v2[cuda_graph]",
    }
    report = {
        "models": {
            "0.4b": {
                "lanes": {
                    "reference": {
                        "prefill": {"case": {"median_ms": 12.0}},
                        "decode": {"case": {"median_ms": 6.0}},
                    },
                    "optimized": {
                        "prefill": {
                            "case": {"median_ms": 4.0, "route": prefill_route}
                        },
                        "decode": {
                            "case": {"median_ms": 2.0, "route": decode_route}
                        },
                    },
                    "fla": {
                        "prefill": {"case": {"median_ms": 3.0}},
                        "decode": {"case": {"median_ms": 3.0}},
                    },
                }
            }
        }
    }
    add_speedups(report)
    assert report["models"]["0.4b"]["lanes"]["optimized"]["prefill"][
        "case"
    ]["speedup_vs_reference"] == 3.0
    assert routes_passed(report)
    report["models"]["0.4b"]["lanes"]["optimized"]["decode"]["case"][
        "route"
    ] = {"implementation": "requested-native-only"}
    assert not routes_passed(report)


def test_percentile_uses_nearest_rank():
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == 5.0
