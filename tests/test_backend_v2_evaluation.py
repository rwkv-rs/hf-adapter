from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))

import benchmark_backend_v2 as benchmark  # noqa: E402
from benchmark_backend_v2 import add_speedups, percentile, routes_passed  # noqa: E402
from common import input_ids_sha256, training_case_seed  # noqa: E402
from validate_backend_v2_ecosystem import (  # noqa: E402
    base_model_backend_environment,
    expected_dense_training_route,
    training_mixed_precision,
    training_parameter_dtype,
    training_smoke_learning_rate,
)
from validate_backend_v2_training import (  # noqa: E402
    candidate_route_passed,
    tensor_metric as training_tensor_metric,
)
from validate_model_training import select_lane as select_model_training_lane  # noqa: E402
from training_metrics import (  # noqa: E402
    adaptive_fast_domain_expected,
    candidate_numerics_not_worse_than_fla,
    checkpoint_input_hash_gate,
    global_gradient_metric,
    global_gradient_passed,
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


def test_global_gradient_gate_accepts_complete_vector_when_tiny_named_row_fails():
    reference = {
        "dominant.weight": torch.tensor([1000.0, -1000.0]),
        "tiny.bias": torch.tensor([1.0e-6]),
    }
    candidate = {
        "dominant.weight": reference["dominant.weight"].clone(),
        "tiny.bias": torch.zeros(1),
    }

    named_report = gradient_metrics(candidate, reference)
    assert not gradient_rows_passed(named_report, torch.bfloat16)
    assert named_report["parameters"]["tiny.bias"]["relative_l2"] == pytest.approx(1.0)

    global_metric = global_gradient_metric(candidate, reference)
    assert global_metric["candidate_only"] == []
    assert global_metric["reference_only"] == []
    assert global_metric["parameter_count"] == 2
    assert global_metric["element_count"] == 3
    assert global_gradient_passed(global_metric)


def test_global_gradient_gate_rejects_and_reports_missing_parameter_set():
    reference = {
        "dominant.weight": torch.tensor([1000.0, -1000.0]),
        "missing.bias": torch.tensor([1.0e-6]),
    }
    candidate = {"dominant.weight": reference["dominant.weight"].clone()}

    global_metric = global_gradient_metric(candidate, reference)
    assert global_metric["candidate_only"] == []
    assert global_metric["reference_only"] == ["missing.bias"]
    # The common portion is numerically exact, but an incomplete optimizer
    # update must never satisfy the formal release gate.
    assert global_metric["cosine"] == pytest.approx(1.0)
    assert global_metric["relative_l2"] == 0.0
    assert not global_gradient_passed(global_metric)


def test_global_gradient_metric_handles_zero_vector_boundaries():
    both_zero = global_gradient_metric(
        {"weight": torch.zeros(4)},
        {"weight": torch.zeros(4)},
    )
    assert both_zero["finite"]
    assert both_zero["cosine"] == 1.0
    assert both_zero["relative_l2"] == 0.0
    assert both_zero["max_abs"] == 0.0
    assert global_gradient_passed(both_zero)

    candidate_zero = global_gradient_metric(
        {"weight": torch.zeros(4)},
        {"weight": torch.ones(4)},
    )
    assert candidate_zero["finite"]
    assert candidate_zero["cosine"] == 0.0
    assert candidate_zero["relative_l2"] == 1.0
    assert not global_gradient_passed(candidate_zero)

    no_gradients = global_gradient_metric({}, {})
    assert no_gradients["parameter_count"] == 0
    assert no_gradients["element_count"] == 0
    assert not global_gradient_passed(no_gradients)


def test_global_gradient_gate_rejects_named_shape_mismatch():
    metric = global_gradient_metric(
        {"weight": torch.ones(2, 2)},
        {"weight": torch.ones(4)},
    )
    assert metric["shape_mismatch"] == {
        "weight": {"candidate": [2, 2], "reference": [4]}
    }
    assert metric["parameter_count"] == 0
    assert not global_gradient_passed(metric)


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
        "model": {
            "selected": "reference",
            "phase": "training",
            "implementation": "torch-reference-model-v1",
        },
        "recurrent": {
            "selected": "reference",
            "implementation": "torch-reference-v1",
        },
        "linear": {
            "selected": "reference",
            "implementation": "torch-reference-linear-v1",
        },
        "mix6": {
            "selected": "reference",
            "implementation": "torch-reference-mix6-v1",
        },
        "program": {
            "selected": "reference",
            "implementation": "torch-reference-training-program-v1",
        },
    }
    report = {
        "models": {},
        "training": {
            "mode": "reference",
            "lanes": {
                "optimized": {
                    "b1-t16": {
                        "shape": {"batch": 1, "tokens": 16},
                        "route": fallback,
                    }
                }
            },
        },
    }
    assert routes_passed(report)
    report["training"]["lanes"]["optimized"]["b1-t16"]["route"] = {
        "model": {
            "selected": "optimized",
            "phase": "training",
            "implementation": "native-nvidia-train-temp-autograd-v2",
        }
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


def test_speed_route_gate_accepts_clean_adaptive_leaf_bundle():
    routes = clean_training_routes(adaptive=True)
    routes["recurrent"] = {
        "selected": "optimized",
        "implementation": "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1",
    }
    report = {
        "models": {},
        "training": {
            "mode": "adaptive",
            "lanes": {
                "optimized": {
                    "b4-t17": {
                        "shape": {"batch": 4, "tokens": 17},
                        "route": routes,
                    }
                }
            },
        },
    }
    assert routes_passed(report)
    report["training"]["lanes"]["optimized"]["b4-t17"]["route"]["mix6"] = {
        "selected": "reference",
        "implementation": "torch-reference-mix6-v1",
    }
    assert not routes_passed(report)


def test_percentile_uses_nearest_rank():
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == 5.0


@pytest.mark.parametrize("spelling", ("adaptive", "native"))
def test_formal_adaptive_training_uses_auto_outer_and_independent_leaf_policy(
    monkeypatch, spelling
):
    for name in (
        "RWKV7_BACKEND",
        "RWKV7_KERNEL_IMPL",
        "RWKV7_MODEL_KERNEL_IMPL",
        "RWKV7_TRAINING_KERNEL_IMPL",
    ):
        monkeypatch.delenv(name, raising=False)
    benchmark.training_route_mode("optimized", spelling)
    assert benchmark.os.environ["RWKV7_BACKEND"] == "auto"
    assert benchmark.os.environ["RWKV7_MODEL_KERNEL_IMPL"] == "auto"
    assert benchmark.os.environ["RWKV7_TRAINING_KERNEL_IMPL"] == "adaptive"

    select_model_training_lane("candidate", candidate="adaptive")
    assert benchmark.os.environ["RWKV7_BACKEND"] == "auto"
    assert benchmark.os.environ["RWKV7_MODEL_KERNEL_IMPL"] == "auto"
    assert benchmark.os.environ["RWKV7_TRAINING_KERNEL_IMPL"] == "adaptive"


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
    monkeypatch.setattr(benchmark, "last_training_routes", lambda kind: {"kind": kind})
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    monkeypatch.setattr(benchmark.torch.cuda, "empty_cache", lambda: None)

    for kind in ("reference", "optimized", "fla"):
        benchmark.benchmark_training_lane(
            kind,
            Path("/checkpoint"),
            torch.bfloat16,
            "adaptive",
            (1,),
            (16,),
            2,
            3,
            17,
            legacy_double_ce=legacy_double_ce,
        )

    assert route_calls == [
        ("reference", "adaptive"),
        ("optimized", "adaptive"),
        ("fla", "adaptive"),
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


def clean_training_routes(*, adaptive: bool, fast_domain: bool = False) -> dict:
    return {
        "model": {
            "selected": "reference",
            "phase": "training",
            "implementation": "torch-reference-model-v1",
        },
        "recurrent": {
            "selected": "optimized" if adaptive else "reference",
            "implementation": (
                "native-nvidia-rwkv7-factorized-recurrent-training-v1"
                if adaptive and fast_domain
                else "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
                if adaptive
                else "torch-reference-v1"
            ),
        },
        "linear": {
            "selected": "optimized" if adaptive and fast_domain else "reference",
            "implementation": (
                "torch-cuda-rwkv7-flattened-linear-training-v1"
                if adaptive and fast_domain
                else "torch-reference-linear-v1"
            ),
        },
        "mix6": {
            "selected": "optimized" if adaptive else "reference",
            "implementation": (
                "native-nvidia-rwkv7-mix6-training-v1"
                if adaptive
                else "torch-reference-mix6-v1"
            ),
        },
        "program": {
            "selected": "optimized" if adaptive and fast_domain else "reference",
            "implementation": (
                "native-nvidia-rwkv7-adaptive-training-program-v1"
                if adaptive
                else "torch-reference-training-program-v1"
            ),
        },
    }


def test_ecosystem_route_gates_require_readable_model_and_actual_leafs():
    adaptive = clean_training_routes(adaptive=True)
    adaptive_fast = clean_training_routes(adaptive=True, fast_domain=True)
    reference = clean_training_routes(adaptive=False)
    assert expected_dense_training_route(adaptive, "adaptive")
    assert expected_dense_training_route(adaptive, "native")
    assert expected_dense_training_route(
        adaptive_fast,
        "adaptive",
        batch=4,
        tokens=128,
    )
    assert not expected_dense_training_route(
        adaptive_fast,
        "adaptive",
        batch=1,
        tokens=128,
    )
    assert not expected_dense_training_route(adaptive, "reference")
    assert expected_dense_training_route(reference, "reference")
    assert expected_dense_training_route(reference, "reference-fallback")
    assert not expected_dense_training_route(reference, "adaptive")


def test_ecosystem_fp16_uses_fp32_master_parameters():
    assert training_parameter_dtype("fp16") is torch.float32
    assert training_parameter_dtype("bf16") is torch.bfloat16
    assert training_mixed_precision("fp16") == "fp16"
    assert training_mixed_precision("bf16") == "no"
    assert training_smoke_learning_rate("fp16") == 1.0e-5
    assert training_smoke_learning_rate("bf16") == 1.0e-3


def test_base_auto_model_uses_fail_closed_fallback_environment(monkeypatch):
    for name in (
        "RWKV7_BACKEND",
        "RWKV7_MODEL_KERNEL_IMPL",
        "RWKV7_KERNEL_IMPL",
        "RWKV7_TRAINING_KERNEL_IMPL",
    ):
        monkeypatch.delenv(name, raising=False)

    base_model_backend_environment()

    assert os.environ["RWKV7_BACKEND"] == "auto"
    assert os.environ["RWKV7_MODEL_KERNEL_IMPL"] == "native"
    assert os.environ["RWKV7_KERNEL_IMPL"] == "auto"
    assert os.environ["RWKV7_TRAINING_KERNEL_IMPL"] == "auto"


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


def test_reference_training_is_explicit_and_not_adaptive():
    reference = clean_training_routes(adaptive=False)
    assert expected_dense_training_route(reference, "reference")
    assert not expected_dense_training_route(reference, "adaptive")
    assert candidate_route_passed(reference, "reference")
    assert candidate_route_passed(reference, "reference-fallback")
    assert not candidate_route_passed(reference, "adaptive")


def test_adaptive_route_uses_matrix_fallback_but_keeps_mix6_for_unaligned_tokens():
    routes = clean_training_routes(adaptive=True)
    assert candidate_route_passed(routes, "adaptive", batch=4, tokens=17)


def test_adaptive_route_uses_both_fast_leaves_only_in_certified_domain():
    routes = clean_training_routes(adaptive=True, fast_domain=True)
    assert candidate_route_passed(routes, "adaptive", batch=4, tokens=128)
    assert not candidate_route_passed(routes, "adaptive", batch=1, tokens=128)


def test_adaptive_fast_domain_evaluator_oracle_is_conservative():
    assert adaptive_fast_domain_expected(batch=4, tokens=128)
    assert not adaptive_fast_domain_expected(batch=1, tokens=128)
    assert not adaptive_fast_domain_expected(batch=4, tokens=16)
    assert not adaptive_fast_domain_expected(batch=8, tokens=128)
    assert not adaptive_fast_domain_expected(batch=4, tokens=256)
    assert not adaptive_fast_domain_expected(
        batch=4,
        tokens=128,
        fully_active=False,
    )


def test_training_case_provenance_is_order_independent_and_hashes_exact_ids():
    seed = training_case_seed(42, batch=4, tokens=128, padding="none")
    assert 0 <= seed < 2**63
    assert seed == training_case_seed(42, batch=4, tokens=128, padding="none")
    assert seed != training_case_seed(42, batch=4, tokens=128, padding="left")
    assert seed != training_case_seed(
        42,
        batch=4,
        tokens=128,
        padding="none",
        sample_index=1,
    )
    assert training_case_seed(42, batch=1, tokens=2000) != training_case_seed(
        42, batch=2, tokens=1000
    )

    ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    digest = input_ids_sha256(ids)
    assert digest == input_ids_sha256(ids.clone())
    assert digest != input_ids_sha256(ids.reshape(1, 4))
    changed = ids.clone()
    changed[0, 0] = 5
    assert digest != input_ids_sha256(changed)


def test_candidate_not_worse_than_fla_includes_same_input_loss():
    fla = {
        "logits": {"cosine": 0.9999},
        "loss": {"max_abs": 0.01},
        "global_gradient": {"cosine": 0.999, "relative_l2": 0.02},
    }
    candidate = {
        "logits": {"cosine": 1.0},
        "loss": {"max_abs": 0.009},
        "global_gradient": {"cosine": 0.9995, "relative_l2": 0.01},
    }
    assert candidate_numerics_not_worse_than_fla(candidate, fla)
    candidate["loss"]["max_abs"] = 0.011
    assert not candidate_numerics_not_worse_than_fla(candidate, fla)


def test_checkpoint_input_hash_gate_requires_both_modes_and_one_exact_hash():
    rows = [
        {
            "batch": 4,
            "tokens": 128,
            "checkpointing": mode,
            "input_ids_sha256": "same",
        }
        for mode in (False, True)
    ]
    assert checkpoint_input_hash_gate(rows, key_fields=("batch", "tokens"))["passed"]

    rows[1]["input_ids_sha256"] = "different"
    assert not checkpoint_input_hash_gate(rows, key_fields=("batch", "tokens"))[
        "passed"
    ]
    assert not checkpoint_input_hash_gate(rows[:1], key_fields=("batch", "tokens"))[
        "passed"
    ]
