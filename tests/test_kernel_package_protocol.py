from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def unload_kernel_package():
    for name in tuple(sys.modules):
        if name == "rwkv7_kernels" or name.startswith("rwkv7_kernels."):
            sys.modules.pop(name)


@pytest.fixture(autouse=True)
def source_kernel_package(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    monkeypatch.delenv("RWKV7_KERNEL_IMPL", raising=False)
    monkeypatch.delenv("RWKV7_MODEL_KERNEL_IMPL", raising=False)
    monkeypatch.delenv("RWKV7_TRAINING_KERNEL_IMPL", raising=False)
    unload_kernel_package()
    yield
    unload_kernel_package()


def cpu_inputs(tokens: int = 2):
    shape = (1, tokens, 1, 64)
    values = [torch.randn(shape, dtype=torch.float16) for _ in range(6)]
    state = torch.randn(1, 1, 64, 64, dtype=torch.float32)
    return (*values, state, None)


def test_public_kernel_surface_is_versioned_and_small():
    kernels = importlib.import_module("rwkv7_kernels")
    assert kernels.RWKV7_KERNEL_API_VERSION == 2
    assert kernels.__all__ == [
        "__version__",
        "RWKV7_KERNEL_API_VERSION",
        "execute_linear_training_v1",
        "execute_mix6_training_v1",
        "execute_recurrent_training_v1",
        "linear_training_v1",
        "mix6_training_v1",
        "model_forward_v1",
        "probe_linear_training_v1",
        "probe_mix6_training_v1",
        "probe_model_forward_v1",
        "probe_recurrent_v1",
        "probe_recurrent_training_v1",
        "recurrent_v1",
        "recurrent_training_v1",
    ]


def test_model_forward_auto_is_fail_closed_outside_validated_cuda():
    kernels = importlib.import_module("rwkv7_kernels")
    request = {
        "model_kind": "base",
        "training": False,
        "use_cache": True,
    }
    support = kernels.probe_model_forward_v1(object(), request)
    assert not support["supported"]
    assert support["phase"] == "prefill"
    assert "causal-LM boundary" in support["reason"]
    with pytest.raises(RuntimeError, match="causal-LM boundary"):
        kernels.model_forward_v1(object(), request)


def test_model_forward_auto_opens_only_validated_4080_fp16_envelope(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    monkeypatch.setattr(
        dispatcher,
        "_cuda_device_name",
        lambda _value: "NVIDIA GeForce RTX 4080",
    )
    monkeypatch.setattr(
        dispatcher,
        "_probe_native",
        lambda _owner, _request: {
            "supported": True,
            "implementation": "native-nvidia-prefill-v2",
            "reason": "native diagnostic accepted",
            "phase": "prefill",
        },
    )
    monkeypatch.setattr(
        importlib.import_module("rwkv7_kernels.quantization"),
        "quantization_report",
        lambda _owner: None,
    )

    class Config:
        hidden_size = 1024
        num_hidden_layers = 24

    class Base:
        embeddings = type(
            "Embeddings",
            (),
            {"weight": torch.zeros(4, 4, dtype=torch.float16)},
        )()

    owner = type(
        "Owner", (), {"model": Base(), "lm_head": object(), "config": Config()}
    )()
    request = {
        "model_kind": "causal_lm",
        "training": False,
        "grad_enabled": False,
        "use_cache": True,
        "input_ids": torch.ones(8, 2048, dtype=torch.long),
    }
    support = dispatcher.probe_model_forward_v1(owner, request)
    assert support == {
        "supported": True,
        "implementation": "native-nvidia-prefill-v2",
        "reason": (
            "validated RTX 4080 FP16 inference envelope selected by production auto"
        ),
        "phase": "prefill",
    }

    owner.model.embeddings.weight = torch.zeros(4, 4, dtype=torch.bfloat16)
    support = dispatcher.probe_model_forward_v1(owner, request)
    assert not support["supported"]
    assert "only for FP16" in support["reason"]


@pytest.mark.parametrize(
    ("training", "grad_enabled"),
    ((True, False), (True, True), (False, True)),
)
def test_model_forward_auto_rejects_autograd_before_native_probe(
    monkeypatch, training, grad_enabled
):
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    calls = {"native_probe": 0}

    def forbidden_native_probe(_owner, _request):
        calls["native_probe"] += 1
        raise AssertionError("auto autograd rejection must not inspect the model")

    monkeypatch.setattr(dispatcher, "_probe_native", forbidden_native_probe)
    support = dispatcher.probe_model_forward_v1(
        object(),
        {
            "model_kind": "causal_lm",
            "training": training,
            "grad_enabled": grad_enabled,
            "use_cache": False,
            "labels": torch.full((1, 16), -1, dtype=torch.long),
            "attention_mask": torch.zeros(1, 16, dtype=torch.long),
        },
    )

    assert not support["supported"]
    assert support["implementation"] == ("hf-readable-training-with-kernel-leaves-v1")
    assert support["phase"] == "training"
    assert support["reason"] == (
        "whole-model dispatch is inference-only; training stays in the readable "
        "HF layer loop and dispatches recurrent, linear, and Mix6 tensor leaves "
        "independently"
    )
    assert calls == {"native_probe": 0}


def test_explicit_dense_model_diagnostic_reports_unsupported_cpu(monkeypatch):
    monkeypatch.setenv("RWKV7_MODEL_KERNEL_IMPL", "dense")
    kernels = importlib.import_module("rwkv7_kernels")
    request = {
        "model_kind": "base",
        "training": False,
        "grad_enabled": False,
        "use_cache": True,
        "hidden_states": torch.zeros(1, 1, 8, dtype=torch.float16),
    }
    support = kernels.probe_model_forward_v1(object(), request)
    assert not support["supported"]
    assert support["implementation"] == "native-torchscript-dense-sequential-v2"
    assert support["phase"] == "decode"
    assert "CUDA" in support["reason"]


def test_explicit_native_prefill_reports_unsupported_cpu(monkeypatch):
    monkeypatch.setenv("RWKV7_MODEL_KERNEL_IMPL", "native")
    kernels = importlib.import_module("rwkv7_kernels")

    class Base:
        embeddings = type(
            "Embeddings",
            (),
            {"weight": torch.zeros(4, 4, dtype=torch.float16)},
        )()

    owner = type("Owner", (), {"model": Base(), "lm_head": object()})()
    request = {
        "model_kind": "causal_lm",
        "training": False,
        "grad_enabled": False,
        "use_cache": True,
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "inputs_embeds": None,
        "labels": None,
        "output_hidden_states": False,
        "output_attentions": False,
    }
    support = kernels.probe_model_forward_v1(owner, request)
    assert not support["supported"]
    assert support["implementation"] == "native-nvidia-prefill-v2"
    assert support["phase"] == "prefill"
    assert "CUDA" in support["reason"]


@pytest.mark.parametrize("implementation", ("auto", "native", "dense"))
@pytest.mark.parametrize(
    ("training", "grad_enabled"),
    ((True, False), (True, True), (False, True)),
)
def test_whole_model_public_protocol_is_inference_only(
    monkeypatch, implementation, training, grad_enabled
):
    monkeypatch.setenv("RWKV7_MODEL_KERNEL_IMPL", implementation)
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    calls = {"native_probe": 0, "dense_probe": 0}

    def forbidden_native_probe(_owner, _request):
        calls["native_probe"] += 1
        raise AssertionError("training must not inspect the whole-model backend")

    def forbidden_dense_probe(_owner, _request):
        calls["dense_probe"] += 1
        raise AssertionError("training must not inspect the dense diagnostic")

    monkeypatch.setattr(dispatcher, "_probe_native", forbidden_native_probe)
    monkeypatch.setattr(dispatcher, "_probe_dense", forbidden_dense_probe)
    request = {
        "model_kind": "causal_lm",
        "training": training,
        "grad_enabled": grad_enabled,
        "use_cache": False,
        "input_ids": torch.ones(1, 16, dtype=torch.long),
        "labels": torch.ones(1, 16, dtype=torch.long),
    }
    pristine_keys = tuple(request)

    support = dispatcher.probe_model_forward_v1(object(), request)

    assert support == {
        "supported": False,
        "implementation": "hf-readable-training-with-kernel-leaves-v1",
        "reason": (
            "whole-model dispatch is inference-only; training stays in the "
            "readable HF layer loop and dispatches recurrent, linear, and Mix6 "
            "tensor leaves independently"
        ),
        "phase": "training",
    }
    assert tuple(request) == pristine_keys
    assert calls == {"native_probe": 0, "dense_probe": 0}
    with pytest.raises(RuntimeError, match="whole-model dispatch is inference-only"):
        dispatcher.model_forward_v1(object(), request)
    assert calls == {"native_probe": 0, "dense_probe": 0}


def test_whole_model_dispatch_has_no_training_runtime_bridge():
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    diagnostic = importlib.import_module("rwkv7_kernels.nvidia.training_runtime")
    source = Path(dispatcher.__file__).read_text()

    assert "_NativeTrainingProbeTicket" not in source
    assert "_probe_native_training" not in source
    assert ".nvidia.training_runtime" not in source
    assert "run_training(owner, request)" not in source
    assert diagnostic.__all__ == []
    assert not hasattr(diagnostic, "run_training")
    assert callable(diagnostic._run_training_diagnostic)


def test_default_auto_prefill_reports_graph_implementation_on_cpu():
    kernels = importlib.import_module("rwkv7_kernels")
    support = kernels.probe_recurrent_v1(*cpu_inputs())
    assert not support["supported"]
    assert support["implementation"] == "torch-cuda-graph-reference-v1"
    assert "CUDA" in support["reason"]


def test_training_auto_is_fail_closed_and_factorized_checks_capability(
    monkeypatch,
):
    kernels = importlib.import_module("rwkv7_kernels")
    inputs = list(cpu_inputs())
    for value in inputs[:-2]:
        value.requires_grad_(True)

    support = kernels.probe_recurrent_training_v1(*inputs)
    assert not support["supported"]
    assert (
        support["implementation"]
        == "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    )
    assert "full-model release gate" in support["reason"]

    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "factorized")
    support = kernels.probe_recurrent_training_v1(*inputs)
    assert not support["supported"]
    assert (
        support["implementation"]
        == "native-nvidia-rwkv7-factorized-recurrent-training-v1"
    )
    assert "CUDA" in support["reason"]


def test_training_matrix_policy_is_exact_and_requires_cuda(monkeypatch):
    kernels = importlib.import_module("rwkv7_kernels")
    inputs = list(cpu_inputs())
    for value in inputs[:-2]:
        value.requires_grad_(True)

    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "matrix")
    support = kernels.probe_recurrent_training_v1(*inputs)
    assert not support["supported"]
    assert (
        support["implementation"]
        == "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    )
    assert "CUDA" in support["reason"]


def test_factorized_recurrent_probe_fails_closed_on_malformed_public_inputs():
    factorized = importlib.import_module("rwkv7_kernels.recurrent.training_factorized")
    inputs = list(cpu_inputs(tokens=16))
    inputs[0] = object()
    support = factorized.probe_recurrent_training_v1(*inputs)
    assert not support["supported"]
    assert "must be tensors" in support["reason"]

    inputs = list(cpu_inputs(tokens=16))
    inputs[-1] = object()
    support = factorized.probe_recurrent_training_v1(*inputs)
    assert not support["supported"]
    assert "attention_mask must be a tensor or None" in support["reason"]


def test_training_adaptive_policy_reports_the_actual_recurrent_leaf(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    inputs = list(cpu_inputs(tokens=16))
    for value in inputs[:-2]:
        value.requires_grad_(True)
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    factorized_probe_kwargs = []

    def probe_factorized(*_args, **kwargs):
        factorized_probe_kwargs.append(kwargs)
        return {
            "supported": True,
            "implementation": ("native-nvidia-rwkv7-factorized-recurrent-training-v1"),
            "reason": "dense test request",
        }

    monkeypatch.setattr(dispatcher, "_probe_factorized", probe_factorized)
    monkeypatch.setattr(
        dispatcher,
        "_probe_matrix",
        lambda *_args, **_kwargs: {
            "supported": True,
            "implementation": ("torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"),
            "reason": "exact test request",
        },
    )

    # Standalone callers have no model-owned zero-state provenance. Adaptive
    # therefore fails closed to the exact matrix leaf without examining the
    # mask (the object deliberately has no tensor operations).
    standalone_inputs = list(inputs)
    standalone_inputs[-1] = object()
    standalone = dispatcher.probe_recurrent_training_v1(*standalone_inputs)
    assert standalone["implementation"] == (
        "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    )
    assert "without model-proven zero initial-state provenance" in standalone["reason"]

    dense = dispatcher.probe_recurrent_training_v1(
        *inputs,
        fully_active=True,
        initial_state_zero=True,
        token_aligned=True,
    )
    assert dense["implementation"] == (
        "native-nvidia-rwkv7-factorized-recurrent-training-v1"
    )
    assert factorized_probe_kwargs == [{"initial_state_zero": True}]

    inputs[-1] = torch.tensor([[False, *([True] * 15)]])
    masked = dispatcher.probe_recurrent_training_v1(
        *inputs,
        fully_active=False,
        initial_state_zero=True,
        token_aligned=True,
    )
    assert masked["implementation"] == (
        "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    )
    assert "masked recurrent request" in masked["reason"]

    unaligned_inputs = list(cpu_inputs(tokens=17))
    for value in unaligned_inputs[:-2]:
        value.requires_grad_(True)
    unaligned = dispatcher.probe_recurrent_training_v1(
        *unaligned_inputs,
        fully_active=True,
        initial_state_zero=True,
        token_aligned=False,
    )
    assert unaligned["implementation"] == (
        "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    )
    assert "unaligned recurrent request" in unaligned["reason"]

    cached = dispatcher.probe_recurrent_training_v1(
        *inputs,
        fully_active=True,
        initial_state_zero=False,
        token_aligned=True,
    )
    assert cached["implementation"] == (
        "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    )
    assert "without model-proven zero initial-state provenance" in cached["reason"]


def test_training_recurrent_hints_reach_only_the_factorized_leaf(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    inputs = list(cpu_inputs(tokens=16))
    for value in inputs[:-2]:
        value.requires_grad_(True)
    probe_kwargs = []
    run_kwargs = []

    def probe(*_args, **kwargs):
        probe_kwargs.append(kwargs)
        return {
            "supported": True,
            "implementation": ("native-nvidia-rwkv7-factorized-recurrent-training-v1"),
            "reason": "hint protocol test",
        }

    def run(*_args, **kwargs):
        run_kwargs.append(kwargs)
        return object()

    monkeypatch.setattr(dispatcher, "_probe_factorized", probe)
    monkeypatch.setattr(dispatcher, "_run_factorized", run)

    result = dispatcher.recurrent_training_v1(
        *inputs,
        fully_active=True,
        initial_state_zero=True,
        token_aligned=True,
    )

    assert result is not None
    assert probe_kwargs == [{"initial_state_zero": True}]
    assert run_kwargs == [
        {
            "fully_active": True,
            "initial_state_zero": True,
            "token_aligned": True,
        }
    ]


def test_training_recurrent_atomic_fallback_probes_each_candidate_once(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    calls = {"factorized_probe": 0, "matrix_probe": 0, "matrix_run": 0}

    def factorized_probe(*_args, **_kwargs):
        calls["factorized_probe"] += 1
        return {
            "supported": False,
            "implementation": ("native-nvidia-rwkv7-factorized-recurrent-training-v1"),
            "reason": "factorized unavailable",
        }

    def matrix_probe(*_args, **_kwargs):
        calls["matrix_probe"] += 1
        return {
            "supported": True,
            "implementation": ("torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"),
            "reason": "matrix fallback",
        }

    def matrix_run(*args):
        calls["matrix_run"] += 1
        return args[3], args[6]

    monkeypatch.setattr(dispatcher, "_probe_factorized", factorized_probe)
    monkeypatch.setattr(dispatcher, "_probe_matrix", matrix_probe)
    monkeypatch.setattr(dispatcher, "_run_matrix", matrix_run)
    execution = dispatcher.execute_recurrent_training_v1(
        *cpu_inputs(tokens=16),
        fully_active=True,
        initial_state_zero=True,
        token_aligned=True,
    )

    assert execution["supported"]
    assert calls == {
        "factorized_probe": 1,
        "matrix_probe": 1,
        "matrix_run": 1,
    }


def test_training_recurrent_atomic_execution_error_does_not_reprobe(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    calls = {"probe": 0, "run": 0}

    def probe(*_args, **_kwargs):
        calls["probe"] += 1
        return {
            "supported": True,
            "implementation": ("native-nvidia-rwkv7-factorized-recurrent-training-v1"),
            "reason": "factorized accepted",
        }

    def run(*_args, **_kwargs):
        calls["run"] += 1
        raise RuntimeError("recurrent execution failed")

    monkeypatch.setattr(dispatcher, "_probe_factorized", probe)
    monkeypatch.setattr(dispatcher, "_run_factorized", run)
    with pytest.raises(RuntimeError, match="recurrent execution failed"):
        dispatcher.recurrent_training_v1(
            *cpu_inputs(tokens=16),
            fully_active=True,
            initial_state_zero=True,
            token_aligned=True,
        )
    assert calls == {"probe": 1, "run": 1}


@pytest.mark.parametrize(
    ("batch", "tokens", "fully_active", "initial_state_zero", "expected"),
    [
        (1, 16, True, True, "factorized"),
        (4, 16, True, True, "factorized"),
        (1, 17, True, True, "matrix"),
        (4, 17, True, True, "matrix"),
        (1, 128, True, True, "factorized"),
        (4, 128, True, True, "factorized"),
        (1, 16, False, True, "matrix"),
        (4, 128, False, True, "matrix"),
        (1, 16, True, False, "matrix"),
        (4, 128, True, False, "matrix"),
    ],
)
def test_training_adaptive_recurrent_route_matrix(
    monkeypatch,
    batch,
    tokens,
    fully_active,
    initial_state_zero,
    expected,
):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    implementations = {
        "factorized": "native-nvidia-rwkv7-factorized-recurrent-training-v1",
        "matrix": "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1",
    }
    monkeypatch.setattr(
        dispatcher,
        "_probe_factorized",
        lambda *_args, **_kwargs: {
            "supported": True,
            "implementation": implementations["factorized"],
            "reason": "factorized route-table test",
        },
    )
    monkeypatch.setattr(
        dispatcher,
        "_probe_matrix",
        lambda *_args, **_kwargs: {
            "supported": True,
            "implementation": implementations["matrix"],
            "reason": "matrix route-table test",
        },
    )
    shape = (batch, tokens, 1, 64)
    values = [torch.randn(shape, dtype=torch.float16) for _ in range(6)]
    state = torch.zeros(batch, 1, 64, 64, dtype=torch.float32)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    if not fully_active:
        mask[:, 0] = False

    support = dispatcher.probe_recurrent_training_v1(
        *values,
        state,
        mask,
        fully_active=fully_active,
        initial_state_zero=initial_state_zero,
        token_aligned=(tokens % 16 == 0),
    )
    assert support["implementation"] == implementations[expected]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("fully_active", 1),
        ("initial_state_zero", "yes"),
        ("token_aligned", object()),
    ],
)
def test_training_recurrent_hints_fail_closed_on_invalid_types(
    monkeypatch, name, value
):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    support = dispatcher.probe_recurrent_training_v1(
        *cpu_inputs(tokens=16), **{name: value}
    )
    assert not support["supported"]
    assert f"{name} must be a bool or None" in support["reason"]


def test_training_adaptive_policy_keeps_masked_linears_on_reference(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    value = torch.randn(2, 64, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)

    masked = dispatcher.probe_linear_training_v1(
        value,
        weight,
        None,
        fully_active=False,
        token_aligned=True,
    )
    assert not masked["supported"]
    assert masked["implementation"] == "torch-reference-linear-v1"

    monkeypatch.setattr(
        dispatcher,
        "_probe_flattened",
        lambda *_args, **_kwargs: {
            "supported": True,
            "implementation": "torch-cuda-rwkv7-flattened-linear-training-v1",
            "reason": "dense test request",
        },
    )
    dense = dispatcher.probe_linear_training_v1(
        value,
        weight,
        None,
        fully_active=True,
        token_aligned=True,
    )
    assert dense["supported"]
    assert dense["implementation"] == ("torch-cuda-rwkv7-flattened-linear-training-v1")

    unaligned = dispatcher.probe_linear_training_v1(
        value,
        weight,
        None,
        fully_active=True,
        token_aligned=False,
    )
    assert not unaligned["supported"]
    assert unaligned["implementation"] == "torch-reference-linear-v1"
    assert "token-length-unaligned" in unaligned["reason"]


def test_training_linear_atomic_execute_probes_once_on_success(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "factorized")
    calls = {"probe": 0, "run": 0}

    def probe(*_args, **_kwargs):
        calls["probe"] += 1
        return {
            "supported": True,
            "implementation": "torch-cuda-rwkv7-flattened-linear-training-v1",
            "reason": "atomic success",
        }

    def run(value, weight, bias):
        calls["run"] += 1
        return torch.nn.functional.linear(value, weight, bias)

    monkeypatch.setattr(dispatcher, "_probe_flattened", probe)
    monkeypatch.setattr(dispatcher, "_run_flattened", run)
    value = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)

    execution = dispatcher.execute_linear_training_v1(value, weight, None)

    assert execution["supported"]
    assert tuple(execution["output"].shape) == (2, 3, 5)
    assert calls == {"probe": 1, "run": 1}


def test_training_linear_atomic_execute_probes_once_on_fallback(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "factorized")
    calls = {"probe": 0, "run": 0}

    def probe(*_args, **_kwargs):
        calls["probe"] += 1
        return {
            "supported": False,
            "implementation": "torch-cuda-rwkv7-flattened-linear-training-v1",
            "reason": "atomic fallback",
        }

    def run(*_args, **_kwargs):
        calls["run"] += 1
        raise AssertionError("unsupported execution must not run")

    monkeypatch.setattr(dispatcher, "_probe_flattened", probe)
    monkeypatch.setattr(dispatcher, "_run_flattened", run)
    execution = dispatcher.execute_linear_training_v1(
        torch.randn(2, 3, 4, requires_grad=True),
        torch.randn(5, 4, requires_grad=True),
        None,
    )

    assert not execution["supported"]
    assert execution["output"] is None
    assert calls == {"probe": 1, "run": 0}


def test_training_linear_atomic_execute_probes_once_on_execution_error(
    monkeypatch,
):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "factorized")
    calls = {"probe": 0, "run": 0}

    def probe(*_args, **_kwargs):
        calls["probe"] += 1
        return {
            "supported": True,
            "implementation": "torch-cuda-rwkv7-flattened-linear-training-v1",
            "reason": "atomic error test",
        }

    def run(*_args, **_kwargs):
        calls["run"] += 1
        raise RuntimeError("linear execution failed")

    monkeypatch.setattr(dispatcher, "_probe_flattened", probe)
    monkeypatch.setattr(dispatcher, "_run_flattened", run)
    with pytest.raises(RuntimeError, match="linear execution failed"):
        dispatcher.linear_training_v1(
            torch.randn(2, 3, 4, requires_grad=True),
            torch.randn(5, 4, requires_grad=True),
            None,
        )

    assert calls == {"probe": 1, "run": 1}


def test_training_matrix_math_matches_reference_outputs_and_full_gradient():
    matrix = importlib.import_module("rwkv7_kernels.recurrent.training_matrix")
    from rwkv7_hf.ops_rwkv7 import rwkv7_recurrent_reference

    torch.manual_seed(307)
    shape = (4, 7, 3, 8)
    base = [(torch.randn(shape) * 0.1) for _ in range(6)]
    base[1] = torch.sigmoid(base[1].float())
    state = torch.randn(4, 3, 8, 8, dtype=torch.float32) * 0.01
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [False, True, True, True, True, True, True],
            [True, True, True, True, True, False, False],
            [False, False, False, False, False, False, False],
        ]
    )

    def collect(function):
        values = [item.detach().clone().requires_grad_() for item in base]
        initial_state = state.detach().clone().requires_grad_()
        output, final_state = function(*values, initial_state, mask)
        loss = output.square().mean() + final_state.square().mean()
        gradients = torch.autograd.grad(loss, (*values, initial_state))
        return output, final_state, gradients

    reference = collect(rwkv7_recurrent_reference)
    candidate = collect(matrix._batched_matrix_recurrence)
    for actual, expected in zip(candidate[:2], reference[:2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(candidate[2], reference[2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_training_linear_auto_is_fail_closed_and_factorized_requires_cuda(
    monkeypatch,
):
    kernels = importlib.import_module("rwkv7_kernels")
    value = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)

    support = kernels.probe_linear_training_v1(value, weight, None)
    assert not support["supported"]
    assert support["implementation"] == "torch-cuda-rwkv7-flattened-linear-training-v1"
    assert "full-model precision" in support["reason"]

    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "factorized")
    support = kernels.probe_linear_training_v1(value, weight, None)
    assert not support["supported"]
    assert support["implementation"] == "torch-cuda-rwkv7-flattened-linear-training-v1"
    assert "CUDA" in support["reason"]


def test_training_linear_trace_records_actual_flattened_leaf(monkeypatch, tmp_path):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    trace_path = tmp_path / "training-route.json"
    monkeypatch.setenv("RWKV7_KERNEL_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "factorized")
    monkeypatch.setattr(
        dispatcher,
        "_probe_flattened",
        lambda *_args, **_kwargs: {
            "supported": True,
            "implementation": "torch-cuda-rwkv7-flattened-linear-training-v1",
            "reason": "test",
        },
    )
    monkeypatch.setattr(
        dispatcher,
        "_run_flattened",
        lambda value, weight, bias, **_kwargs: torch.nn.functional.linear(
            value, weight, bias
        ),
    )
    value = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    output = dispatcher.linear_training_v1(value, weight, None)
    assert tuple(output.shape) == (2, 3, 5)

    importlib.import_module("rwkv7_kernels.trace").write_trace()
    payload = json.loads(trace_path.read_text())
    assert payload["requested_training_policy"] == "factorized"
    assert payload["actual_linear_calls"] == {
        "torch-cuda-rwkv7-flattened-linear-training-v1": 1
    }


def test_training_atomic_trace_records_recurrent_and_mix6_once(monkeypatch, tmp_path):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    trace_path = tmp_path / "training-atomic-route.json"
    monkeypatch.setenv("RWKV7_KERNEL_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    monkeypatch.setattr(
        dispatcher,
        "_probe_factorized",
        lambda *_args, **_kwargs: {
            "supported": True,
            "implementation": ("native-nvidia-rwkv7-factorized-recurrent-training-v1"),
            "reason": "atomic recurrent trace test",
        },
    )
    monkeypatch.setattr(
        dispatcher,
        "_run_factorized",
        lambda *args, **_kwargs: (args[3], args[6]),
    )
    monkeypatch.setattr(
        dispatcher,
        "_probe_mix6",
        lambda *_args: {
            "supported": True,
            "implementation": "native-nvidia-rwkv7-mix6-training-v1",
            "reason": "atomic Mix6 trace test",
        },
    )
    monkeypatch.setattr(
        dispatcher,
        "_run_mix6",
        lambda value, _shifted, *mixes: tuple(value for _ in mixes),
    )

    recurrent = dispatcher.execute_recurrent_training_v1(
        *cpu_inputs(tokens=16),
        fully_active=True,
        initial_state_zero=True,
        token_aligned=True,
    )
    value = torch.randn(2, 16, 8)
    shifted = torch.randn_like(value)
    mixes = tuple(torch.randn(8) for _ in range(6))
    mix6 = dispatcher.execute_mix6_training_v1(value, shifted, *mixes)
    assert recurrent["supported"] and mix6["supported"]

    importlib.import_module("rwkv7_kernels.trace").write_trace()
    payload = json.loads(trace_path.read_text())
    assert payload["actual_recurrent_calls"] == {
        "native-nvidia-rwkv7-factorized-recurrent-training-v1": 1
    }
    assert payload["actual_mix6_calls"] == {"native-nvidia-rwkv7-mix6-training-v1": 1}


def test_training_flattened_linear_declares_small_row_numerical_gate():
    source = (
        ROOT / "kernels" / "rwkv7_kernels" / "linear" / "training_flattened.py"
    ).read_text()
    assert "_MIN_FLATTENED_ROWS = 128" in source
    assert "smaller projections retain" in source


def test_explicit_triton_lane_reports_real_implementation_on_cpu(monkeypatch):
    monkeypatch.setenv("RWKV7_KERNEL_IMPL", "triton")
    kernels = importlib.import_module("rwkv7_kernels")
    support = kernels.probe_recurrent_v1(*cpu_inputs())
    assert not support["supported"]
    assert support["implementation"] == "native-triton-rank1-scan-v1"
    assert "CUDA" in support["reason"] or "Triton" in support["reason"]


def test_auto_routes_decode_to_triton_and_prefill_to_graph(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.dispatcher")

    def supported(name):
        def probe(*_args, **_kwargs):
            return {"supported": True, "implementation": name, "reason": name}

        return probe

    monkeypatch.setattr(dispatcher, "_probe_triton", supported("triton"))
    monkeypatch.setattr(dispatcher, "_probe_graph", supported("graph"))
    monkeypatch.setattr(dispatcher, "_run_triton", object())
    monkeypatch.setattr(dispatcher, "_run_graph", object())
    prefill = cpu_inputs()
    decode = tuple(
        value[:, :1] if isinstance(value, torch.Tensor) and value.ndim == 4 else value
        for value in prefill
    )

    decode_support, decode_run = dispatcher._select(*decode)
    prefill_support, prefill_run = dispatcher._select(*prefill)
    assert decode_support["implementation"] == "triton"
    assert decode_run is dispatcher._run_triton
    assert prefill_support["implementation"] == "graph"
    assert prefill_run is dispatcher._run_graph


def test_optional_route_trace_records_executed_implementation(monkeypatch, tmp_path):
    dispatcher = importlib.import_module("rwkv7_kernels.dispatcher")
    trace = tmp_path / "route.json"
    monkeypatch.setenv("RWKV7_KERNEL_TRACE_PATH", str(trace))
    monkeypatch.setattr(
        dispatcher,
        "_probe_triton",
        lambda *_args, **_kwargs: {
            "supported": True,
            "implementation": "triton-test",
            "reason": "test",
        },
    )
    monkeypatch.setattr(dispatcher, "_run_triton", lambda *_args, **_kwargs: "ran")
    prefill = cpu_inputs()
    decode = tuple(
        value[:, :1] if isinstance(value, torch.Tensor) and value.ndim == 4 else value
        for value in prefill
    )

    assert dispatcher.recurrent_v1(*decode) == "ran"
    dispatcher._write_trace()
    payload = json.loads(trace.read_text())
    assert payload["schema"] == "rwkv7-kernel-route-trace-v2"
    assert payload["requested_policy"] == "auto"
    assert payload["actual_recurrent_calls"] == {"triton-test": 1}


def test_route_trace_records_executed_whole_model_phase(monkeypatch, tmp_path):
    trace_path = tmp_path / "route.json"
    monkeypatch.setenv("RWKV7_KERNEL_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("RWKV7_MODEL_KERNEL_IMPL", "native")
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    monkeypatch.setattr(
        dispatcher,
        "_probe_native",
        lambda _owner, _request: {
            "supported": True,
            "implementation": "probe-name",
            "reason": "test",
            "phase": "prefill",
        },
    )
    monkeypatch.setattr(
        dispatcher,
        "_run_native_prefill",
        lambda _owner, _request: {
            "output_kind": "causal_lm",
            "logits": torch.zeros(1, 2, 3),
            "loss": None,
            "past_key_values": None,
            "hidden_states": None,
            "implementation": "native-prefill-test[fused]",
            "phase": "prefill",
        },
    )

    class EmptyCache:
        @staticmethod
        def get_seq_length():
            return 0

    result = dispatcher.model_forward_v1(
        object(),
        {
            "model_kind": "causal_lm",
            "training": False,
            "use_cache": True,
            "past_key_values": EmptyCache(),
        },
    )
    assert result["implementation"] == "native-prefill-test[fused]"
    importlib.import_module("rwkv7_kernels.trace").write_trace()
    payload = json.loads(trace_path.read_text())
    assert payload["requested_model_policy"] == "native"
    assert payload["actual_model_calls"] == {"native-prefill-test[fused]": 1}
    assert payload["actual_model_phases"] == {"prefill": 1}


def test_unknown_kernel_implementation_is_rejected(monkeypatch):
    monkeypatch.setenv("RWKV7_KERNEL_IMPL", "mystery")
    kernels = importlib.import_module("rwkv7_kernels")
    with pytest.raises(ValueError, match="RWKV7_KERNEL_IMPL"):
        kernels.probe_recurrent_v1(*cpu_inputs())


def test_graph_reference_math_is_batch_regrouping_invariant():
    graph = importlib.import_module("rwkv7_kernels.recurrent.graph")
    torch.manual_seed(41)
    batch, time, heads, width = 8, 3, 1, 8
    tensors = [
        torch.randn(batch, time, heads, width, dtype=torch.float16) for _ in range(6)
    ]
    state = torch.randn(batch, heads, width, width, dtype=torch.float32)
    mask = torch.ones(batch, time, dtype=torch.bool)
    mask[5, 0] = False

    grouped = graph._reference_recurrent(*tensors, state, mask)
    isolated = graph._reference_recurrent(
        *(value[5:6] for value in tensors), state[5:6], mask[5:6]
    )
    torch.testing.assert_close(grouped[0][5:6], isolated[0], rtol=0, atol=0)
    torch.testing.assert_close(grouped[1][5:6], isolated[1], rtol=0, atol=0)
