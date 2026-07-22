from __future__ import annotations

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))

import benchmark_backend_v2 as benchmark  # noqa: E402
from benchmark_backend_v2 import add_speedups, percentile, routes_passed  # noqa: E402
from validate_backend_v2_ecosystem import (  # noqa: E402
    adapter_fallback_route,
    expected_dense_training_route,
    native_training_route,
    reference_training_route,
    training_parameter_dtype,
)
from validate_backend_v2_training import (  # noqa: E402
    candidate_route_passed,
    tensor_metric as training_tensor_metric,
)
from validate_backend_v2_fla_sm70 import (  # noqa: E402
    optimized_route_passed as sm70_fla_route_passed,
)
from validate_backend_v2_inference import (  # noqa: E402
    annotate_metric as annotate_inference_metric,
    aspirational_metric_passed,
    release_metric_passed,
)
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


def test_training_tensor_metric_uses_stable_reductions_for_large_exact_logits():
    logits = torch.linspace(-8.0, 8.0, 16 * 65536).reshape(1, 16, 65536)
    row = training_tensor_metric(logits.clone(), logits)
    assert row["cosine"] == 1.0
    assert row["max_abs"] == 0.0
    assert row["relative_l2"] == 0.0


def test_inference_release_gate_keeps_strict_target_as_diagnostic():
    row = {
        "finite": True,
        "cosine": 0.99995,
        "max_abs": 0.20,
        "argmax_same": True,
    }
    assert release_metric_passed(row, torch.float16, logits=True)
    assert not aspirational_metric_passed(row, torch.float16, logits=True)
    annotated = annotate_inference_metric(row.copy(), torch.float16, logits=True)
    assert annotated["release_passed"]
    assert not annotated["aspirational_passed"]


def test_inference_bf16_release_floor_keeps_argmax_as_diagnostic():
    row = {
        "finite": True,
        "cosine": 0.9995,
        "max_abs": 1.0,
        "argmax_same": True,
    }
    assert release_metric_passed(row, torch.bfloat16, logits=True)
    assert not aspirational_metric_passed(row, torch.bfloat16, logits=True)
    row["argmax_same"] = False
    assert release_metric_passed(row, torch.bfloat16, logits=True)
    assert release_metric_passed(row, torch.bfloat16, logits=False)


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
                        "prefill": {"case": {"median_ms": 4.0, "route": prefill_route}},
                        "decode": {"case": {"median_ms": 2.0, "route": decode_route}},
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
    assert (
        report["models"]["0.4b"]["lanes"]["optimized"]["prefill"]["case"][
            "speedup_vs_reference"
        ]
        == 3.0
    )
    assert routes_passed(report)
    report["models"]["0.4b"]["lanes"]["optimized"]["decode"]["case"]["route"] = {
        "implementation": "requested-native-only"
    }
    assert not routes_passed(report)


def test_speed_route_gate_distinguishes_v100_training_fallback():
    fallback = {
        "selected": "reference",
        "phase": "training",
        "implementation": "torch-reference-model-v1",
        "reason": "native training requires BF16 and sm80 or newer",
    }
    report = {
        "models": {},
        "training": {
            "mode": "reference-fallback",
            "lanes": {"optimized": {"b1-t16": {"route": fallback}}},
        },
    }
    assert routes_passed(report)
    report["training"]["lanes"]["optimized"]["b1-t16"]["route"] = {
        "selected": "optimized",
        "phase": "training",
        "implementation": "native-nvidia-train-temp-autograd-v2",
    }
    assert not routes_passed(report)


def test_speed_route_gate_accepts_explicit_not_applicable_training():
    report = {
        "models": {},
        "training": {
            "mode": "skip-not-applicable",
            "status": "not_applicable",
            "reason": "native whole-model training requires BF16 and sm80 or newer",
        },
    }
    add_speedups(report)
    assert routes_passed(report)
    report["training"]["status"] = "passed"
    assert not routes_passed(report)


def test_percentile_uses_nearest_rank():
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == 5.0


def test_timed_training_uses_model_loss_once_and_preserves_sync_order(monkeypatch):
    trace = []
    ce_calls = []
    model_loss_backwards = []
    original_cross_entropy = benchmark.F.cross_entropy

    def counted_cross_entropy(*args, **kwargs):
        ce_calls.append((args, kwargs))
        return original_cross_entropy(*args, **kwargs)

    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1, 1, 5))

        def zero_grad(self, *, set_to_none):
            assert set_to_none
            self.weight.grad = None

        def __call__(self, *, input_ids, labels, use_cache, logits_to_keep):
            trace.append("step")
            assert not use_cache
            assert logits_to_keep == 0
            logits = self.weight.expand(*input_ids.shape, -1)
            loss = benchmark.F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss.register_hook(lambda gradient: model_loss_backwards.append(gradient))
            return SimpleNamespace(loss=loss, logits=logits)

    clock = itertools.count()

    def perf_counter():
        trace.append("clock")
        return next(clock) / 1000.0

    monkeypatch.setattr(benchmark.F, "cross_entropy", counted_cross_entropy)
    monkeypatch.setattr(benchmark, "synchronize", lambda: trace.append("sync"))
    monkeypatch.setattr(benchmark.time, "perf_counter", perf_counter)
    monkeypatch.setattr(benchmark.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(benchmark.torch.cuda, "max_memory_allocated", lambda: 0)

    ids = torch.tensor([[1, 2, 3, 4]])
    labels = ids.clone()
    result = benchmark.timed_training(Model(), ids, labels, warmup=1, repeats=2)

    steps = 1 + 1 + 2  # cold + warmup + measured repeats
    assert len(ce_calls) == steps
    assert len(model_loss_backwards) == steps
    assert result["loss_mode"] == "model-output-loss"
    assert trace == [
        "sync",
        "clock",
        "step",
        "sync",
        "clock",
        "step",
        "sync",
        "sync",
        "clock",
        "step",
        "sync",
        "clock",
        "sync",
        "clock",
        "step",
        "sync",
        "clock",
    ]


def test_timed_training_legacy_double_ce_is_explicit(monkeypatch):
    ce_calls = []
    model_loss_backwards = []
    original_cross_entropy = benchmark.F.cross_entropy

    def counted_cross_entropy(*args, **kwargs):
        ce_calls.append((args, kwargs))
        return original_cross_entropy(*args, **kwargs)

    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1, 1, 5))

        def zero_grad(self, *, set_to_none):
            self.weight.grad = None

        def __call__(self, *, input_ids, labels, use_cache, logits_to_keep):
            logits = self.weight.expand(*input_ids.shape, -1)
            loss = benchmark.F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss.register_hook(lambda gradient: model_loss_backwards.append(gradient))
            return SimpleNamespace(loss=loss, logits=logits)

    monkeypatch.setattr(benchmark.F, "cross_entropy", counted_cross_entropy)
    monkeypatch.setattr(benchmark, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(benchmark.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(benchmark.torch.cuda, "max_memory_allocated", lambda: 0)

    ids = torch.tensor([[1, 2, 3, 4]])
    result = benchmark.timed_training(
        Model(),
        ids,
        ids.clone(),
        warmup=0,
        repeats=1,
        legacy_double_ce=True,
    )

    steps = 1 + 0 + 1
    assert len(ce_calls) == steps * 2
    assert not model_loss_backwards
    assert result["loss_mode"] == "legacy-double-ce"


@pytest.mark.parametrize("legacy_double_ce", [False, True])
def test_training_lanes_share_the_same_loss_mode(monkeypatch, legacy_double_ce):
    calls = []
    route_calls = []

    class Generator:
        def __init__(self, *, device):
            assert device == "cuda"

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class Model:
        def __init__(self, kind):
            self.kind = kind
            self.config = SimpleNamespace(vocab_size=8)

    def load_model(kind, path, dtype, *, training):
        assert training
        return Model(kind)

    def randint(low, high, size, *, generator, device):
        assert (low, high, device) == (1, 8, "cuda")
        assert generator.seed == 17
        return torch.arange(size[0] * size[1]).reshape(size) % (high - low) + low

    def timed_training(model, ids, labels, **kwargs):
        calls.append(
            {
                "kind": model.kind,
                "ids": ids.clone(),
                "labels": labels.clone(),
                "kwargs": kwargs,
            }
        )
        return {"median_ms": 1.0}

    monkeypatch.setattr(benchmark, "load_model", load_model)
    monkeypatch.setattr(
        benchmark,
        "training_route_mode",
        lambda kind, mode: route_calls.append((kind, mode)),
    )
    monkeypatch.setattr(benchmark.torch, "Generator", Generator)
    monkeypatch.setattr(benchmark.torch, "randint", randint)
    monkeypatch.setattr(benchmark, "timed_training", timed_training)
    monkeypatch.setattr(benchmark, "last_route", lambda kind: {"kind": kind})
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    monkeypatch.setattr(benchmark.torch.cuda, "empty_cache", lambda: None)

    for kind in ("reference", "optimized", "fla"):
        benchmark.benchmark_training_lane(
            kind,
            Path("/checkpoint"),
            torch.bfloat16,
            "native",
            (1,),
            (16,),
            2,
            3,
            17,
            legacy_double_ce=legacy_double_ce,
        )

    assert route_calls == [
        ("reference", "native"),
        ("optimized", "native"),
        ("fla", "native"),
    ]
    assert [call["kind"] for call in calls] == ["reference", "optimized", "fla"]
    for call in calls:
        assert call["kwargs"] == {
            "warmup": 2,
            "repeats": 3,
            "legacy_double_ce": legacy_double_ce,
        }
        assert torch.equal(call["ids"], calls[0]["ids"])
        assert torch.equal(call["labels"], calls[0]["labels"])
        assert call["labels"][0, 8].item() == -100


def test_ecosystem_route_gates_distinguish_native_and_adapter_fallback():
    native = {
        "selected": "optimized",
        "phase": "training",
        "implementation": "native-nvidia-train-temp-autograd-v2",
    }
    fallback = {
        "selected": "reference",
        "phase": "training",
        "implementation": "torch-reference-model-v1",
        "reason": "adapter-wrapped FFN modules use reference autograd",
    }
    nested_fallback = {
        "selected": "reference",
        "phase": "training",
        "implementation": "torch-reference-model-v1",
        "reason": "native prefill requires the causal-LM boundary",
    }
    assert native_training_route(native)
    assert not adapter_fallback_route(native)
    assert adapter_fallback_route(fallback)
    assert adapter_fallback_route(nested_fallback)
    assert not native_training_route(fallback)


def test_ecosystem_fp16_uses_fp32_master_parameters():
    assert training_parameter_dtype("fp16") is torch.float32
    assert training_parameter_dtype("bf16") is torch.bfloat16


def test_sm70_fla_gate_requires_actual_optimized_recurrent_route():
    assert sm70_fla_route_passed(
        {
            "selected": "optimized",
            "implementation": "torch-cuda-graph-reference-v1",
        }
    )
    assert not sm70_fla_route_passed(
        {
            "selected": "reference",
            "implementation": "torch-reference-recurrence-v1",
        }
    )


def test_sm70_training_fallback_is_explicit_and_not_native():
    fallback = {
        "selected": "reference",
        "phase": "training",
        "implementation": "torch-reference-model-v1",
        "reason": "native training requires a BF16 checkpoint",
    }
    assert reference_training_route(fallback)
    assert expected_dense_training_route(fallback, "reference-fallback")
    assert not expected_dense_training_route(fallback, "native")
    assert candidate_route_passed(fallback, "reference-fallback")
    assert not candidate_route_passed(fallback, "native")
