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


def cpu_inputs(tokens: int = 2, batch: int = 1):
    shape = (batch, tokens, 1, 64)
    values = [torch.randn(shape, dtype=torch.float16) for _ in range(6)]
    state = torch.randn(batch, 1, 64, 64, dtype=torch.float32)
    return (*values, state, None)


def test_public_kernel_surface_is_versioned_and_small():
    kernels = importlib.import_module("rwkv7_kernels")
    assert kernels.RWKV7_KERNEL_API_VERSION == 4
    assert kernels.__all__ == [
        "__version__",
        "RWKV7_KERNEL_API_VERSION",
        "execute_optional_v4",
    ]
    legacy_names = (
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
        "probe_training_program_v1",
        "recurrent_v1",
        "recurrent_training_v1",
    )
    assert all(not hasattr(kernels, name) for name in legacy_names)


def test_v4_training_program_issues_one_opaque_public_certificate(monkeypatch):
    backend = importlib.import_module("rwkv7_kernels.backend")
    calls = []

    def probe(hidden_states, attention_mask, **facts):
        calls.append((hidden_states, attention_mask, dict(facts)))
        return {
            "supported": True,
            "implementation": "native-nvidia-rwkv7-adaptive-training-program-v1",
            "reason": "program accepted",
        }

    monkeypatch.setattr(backend, "probe_training_program_v1", probe)
    hidden = torch.randn(4, 128, 64)
    mask = torch.ones(4, 128, dtype=torch.bool)
    result = backend.execute_optional_v4(
        "training_program",
        hidden,
        mask,
        fully_active=True,
        initial_state_zero=True,
        autograd_leaf_eligible=True,
        head_dim=64,
    )

    assert set(result) == {
        "api_version",
        "kind",
        "supported",
        "implementation",
        "reason",
        "result",
        "phase",
    }
    assert result["api_version"] == 4
    assert result["kind"] == "training_program"
    assert result["supported"] is True
    assert result["implementation"] == (
        "native-nvidia-rwkv7-adaptive-training-program-v1"
    )
    assert result["phase"] == "training"
    assert result["result"]["token_aligned"] is True
    assert result["result"]["program_id"].startswith(
        "native-nvidia-rwkv7-adaptive-training-program-v1:"
    )
    assert len(calls) == 1
    assert calls[0][2]["training"] is True
    assert calls[0][2]["token_aligned"] is True

    leaf_calls = []

    def execute_linear(value, weight, bias, **hints):
        leaf_calls.append(dict(hints))
        return {
            "supported": True,
            "implementation": "torch-cuda-rwkv7-flattened-linear-training-v1",
            "reason": "certificate accepted",
            "output": torch.nn.functional.linear(value, weight, bias),
        }

    monkeypatch.setattr(backend, "execute_linear_training_v1", execute_linear)
    program_id = result["result"]["program_id"]
    facts = {
        "fully_active": True,
        "initial_state_zero": True,
        "token_aligned": True,
        "autograd_leaf_eligible": True,
    }
    linear = backend.execute_optional_v4(
        "linear_training",
        hidden,
        torch.randn(8, 64),
        None,
        program_id=program_id,
        facts=facts,
    )
    assert linear["supported"] is True
    assert leaf_calls == [
        {
            "fully_active": True,
            "initial_state_zero": True,
            "token_aligned": True,
            "adaptive_fast_program": True,
        }
    ]

    mismatch = backend.execute_optional_v4(
        "linear_training",
        hidden[:1],
        torch.randn(8, 64),
        None,
        program_id=program_id,
        facts=facts,
    )
    assert mismatch["supported"] is False
    assert "batch/token shape" in mismatch["reason"]


def test_v4_training_program_force_reference_skips_optional_probe(monkeypatch):
    backend = importlib.import_module("rwkv7_kernels.backend")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("forced reference must not probe an optional program")

    monkeypatch.setattr(backend, "probe_training_program_v1", unexpected)
    result = backend.execute_optional_v4(
        "training_program",
        torch.randn(2, 17, 64),
        torch.ones(2, 17, dtype=torch.bool),
        fully_active=True,
        initial_state_zero=True,
        autograd_leaf_eligible=False,
        force_reference_program=True,
        head_dim=64,
    )

    assert result["supported"] is False
    assert result["implementation"] == "torch-reference-training-program-v1"
    assert result["result"] is None


def test_v4_envelope_rejects_payload_on_unsupported_result():
    protocol = importlib.import_module("rwkv7_kernels.protocol")
    with pytest.raises(ValueError, match="unsupported.*must be None"):
        protocol.optional_kernel_result(
            kind="recurrent",
            supported=False,
            implementation="mock",
            reason="unsupported",
            result=(torch.empty(0), torch.empty(0)),
            phase="prefill",
        )
    with pytest.raises(ValueError, match="unsupported.*must be None"):
        protocol.validate_optional_kernel_result(
            {
                "api_version": 4,
                "kind": "recurrent",
                "supported": False,
                "implementation": "mock",
                "reason": "unsupported",
                "result": {},
                "phase": "prefill",
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("api_version", True),
        ("api_version", 4.0),
        ("kind", b"recurrent"),
        ("supported", "false"),
        ("supported", 1),
        ("implementation", 7),
        ("reason", None),
        ("phase", False),
    ),
)
def test_v4_envelope_rejects_coercible_field_types(field, value):
    protocol = importlib.import_module("rwkv7_kernels.protocol")
    result = {
        "api_version": 4,
        "kind": "recurrent",
        "supported": False,
        "implementation": "mock",
        "reason": "unsupported",
        "result": None,
        "phase": "prefill",
    }
    result[field] = value
    with pytest.raises(TypeError, match=field):
        protocol.validate_optional_kernel_result(result)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kind", b"recurrent"),
        ("supported", "false"),
        ("supported", 1),
        ("implementation", 7),
        ("reason", None),
        ("phase", False),
    ),
)
def test_v4_envelope_builder_rejects_coercible_field_types(field, value):
    protocol = importlib.import_module("rwkv7_kernels.protocol")
    fields = {
        "kind": "recurrent",
        "supported": False,
        "implementation": "mock",
        "reason": "unsupported",
        "result": None,
        "phase": "prefill",
    }
    fields[field] = value
    with pytest.raises(TypeError, match=field):
        protocol.optional_kernel_result(**fields)


def test_v4_facade_rejects_coercible_legacy_probe_fields(monkeypatch):
    backend = importlib.import_module("rwkv7_kernels.backend")
    monkeypatch.setattr(
        backend,
        "probe_recurrent_v1",
        lambda *_args, **_kwargs: {
            "supported": "false",
            "implementation": "mock",
            "reason": "must not coerce",
        },
    )
    with pytest.raises(TypeError, match="supported"):
        backend.execute_optional_v4("recurrent", *cpu_inputs(), training=False)


def test_v4_forged_legacy_program_fails_closed_before_linear(monkeypatch):
    backend = importlib.import_module("rwkv7_kernels.backend")
    captured = []

    def execute(*args, **kwargs):
        captured.append((args, dict(kwargs)))
        return {
            "supported": True,
            "implementation": "torch-cuda-rwkv7-flattened-linear-training-v1",
            "reason": "linear accepted",
            "output": torch.randn(4, 128, 8),
        }

    monkeypatch.setattr(backend, "execute_linear_training_v1", execute)
    value = torch.randn(4, 128, 4)
    weight = torch.randn(8, 4)
    result = backend.execute_optional_v4(
        "linear_training",
        value,
        weight,
        None,
        program_id="native-nvidia-rwkv7-adaptive-training-program-v1",
        facts={
            "fully_active": True,
            "initial_state_zero": True,
            "token_aligned": True,
            "autograd_leaf_eligible": True,
        },
    )

    assert result["supported"] is False
    assert result["result"] is None
    assert result["phase"] == "training"
    assert "unknown optional training program_id" in result["reason"]
    assert captured == []


def test_v4_unknown_program_id_fails_closed_before_leaf(monkeypatch):
    backend = importlib.import_module("rwkv7_kernels.backend")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("an unrecognized certificate must not reach a leaf")

    monkeypatch.setattr(backend, "execute_mix6_training_v1", unexpected)
    result = backend.execute_optional_v4(
        "mix6_training",
        torch.randn(1, 2, 4),
        torch.randn(1, 2, 4),
        *(torch.randn(4) for _ in range(6)),
        program_id="forged-program",
        facts={"fully_active": True, "token_aligned": True},
    )

    assert result["supported"] is False
    assert result["result"] is None
    assert "unknown optional training program_id" in result["reason"]


def test_v4_force_reference_fact_fails_closed_before_leaf(monkeypatch):
    backend = importlib.import_module("rwkv7_kernels.backend")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("a reference program must not enter an optional leaf")

    monkeypatch.setattr(backend, "execute_linear_training_v1", unexpected)
    result = backend.execute_optional_v4(
        "linear_training",
        torch.randn(2, 3, 4),
        torch.randn(8, 4),
        None,
        program_id=None,
        facts={
            "fully_active": True,
            "initial_state_zero": True,
            "token_aligned": False,
            "autograd_leaf_eligible": False,
            "force_reference_program": True,
        },
    )

    assert result["supported"] is False
    assert result["implementation"] == "torch-reference-training-program-v1"
    assert result["result"] is None


def test_v4_recurrent_selects_inference_or_atomic_training(monkeypatch):
    backend = importlib.import_module("rwkv7_kernels.backend")
    inference_output = (torch.randn(1, 1, 1, 64), torch.randn(1, 1, 64, 64))
    calls = []

    def inference_probe(*_args, **_kwargs):
        calls.append("inference_probe")
        return {
            "supported": True,
            "implementation": "mock-recurrent-v1",
            "reason": "inference accepted",
        }

    def inference_run(*_args, **_kwargs):
        calls.append("inference_run")
        return inference_output

    monkeypatch.setattr(backend, "probe_recurrent_v1", inference_probe)
    monkeypatch.setattr(backend, "recurrent_v1", inference_run)
    inference = backend.execute_optional_v4(
        "recurrent", *cpu_inputs(tokens=1), training=False
    )
    assert inference["result"] is inference_output
    assert inference["phase"] == "decode"

    training_output = (torch.randn(1, 2, 1, 64), torch.randn(1, 1, 64, 64))

    def training_run(*_args, **kwargs):
        calls.append(("training_run", dict(kwargs)))
        return {
            "supported": True,
            "implementation": "mock-training-v1",
            "reason": "training accepted",
            "result": training_output,
        }

    monkeypatch.setattr(backend, "execute_recurrent_training_v1", training_run)
    training = backend.execute_optional_v4(
        "recurrent",
        *cpu_inputs(tokens=2),
        training=True,
        program_id=None,
        facts={
            "fully_active": True,
            "initial_state_zero": True,
            "token_aligned": False,
        },
    )
    assert training["result"] is training_output
    assert training["phase"] == "training"
    assert calls == [
        "inference_probe",
        "inference_run",
        (
            "training_run",
            {
                "fully_active": True,
                "initial_state_zero": True,
                "token_aligned": False,
                "adaptive_fast_program": None,
            },
        ),
    ]


def test_v4_model_forward_normalizes_probe_and_execution(monkeypatch):
    backend = importlib.import_module("rwkv7_kernels.backend")
    model_result = {
        "output_kind": "causal_lm",
        "logits": torch.randn(1, 2, 8),
        "implementation": "mock-prefill-v2[effective]",
        "phase": "prefill",
    }
    monkeypatch.setattr(
        backend,
        "probe_model_forward_v1",
        lambda *_args, **_kwargs: {
            "supported": True,
            "implementation": "mock-prefill-v2",
            "reason": "model accepted",
            "phase": "prefill",
        },
    )
    monkeypatch.setattr(
        backend,
        "model_forward_v1",
        lambda *_args, **_kwargs: model_result,
    )
    result = backend.execute_optional_v4(
        "model_forward",
        object(),
        {"model_kind": "causal_lm", "training": False, "use_cache": True},
    )

    assert result["supported"] is True
    assert result["result"] is model_result
    assert result["implementation"] == "mock-prefill-v2[effective]"
    assert result["phase"] == "prefill"


def test_model_forward_auto_is_fail_closed_outside_validated_cuda():
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    request = {
        "model_kind": "base",
        "training": False,
        "use_cache": True,
    }
    support = dispatcher.probe_model_forward_v1(object(), request)
    assert not support["supported"]
    assert support["phase"] == "prefill"
    assert "causal-LM boundary" in support["reason"]
    with pytest.raises(RuntimeError, match="causal-LM boundary"):
        dispatcher.model_forward_v1(object(), request)


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
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    request = {
        "model_kind": "base",
        "training": False,
        "grad_enabled": False,
        "use_cache": True,
        "hidden_states": torch.zeros(1, 1, 8, dtype=torch.float16),
    }
    support = dispatcher.probe_model_forward_v1(object(), request)
    assert not support["supported"]
    assert support["implementation"] == "native-torchscript-dense-sequential-v2"
    assert support["phase"] == "decode"
    assert "CUDA" in support["reason"]


def test_explicit_native_prefill_reports_unsupported_cpu(monkeypatch):
    monkeypatch.setenv("RWKV7_MODEL_KERNEL_IMPL", "native")
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")

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
    support = dispatcher.probe_model_forward_v1(owner, request)
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
    dispatcher = importlib.import_module("rwkv7_kernels.dispatcher")
    support = dispatcher.probe_recurrent_v1(*cpu_inputs())
    assert not support["supported"]
    assert support["implementation"] == "torch-cuda-graph-reference-v1"
    assert "CUDA" in support["reason"]


def test_training_auto_is_fail_closed_and_factorized_checks_capability(
    monkeypatch,
):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    inputs = list(cpu_inputs())
    for value in inputs[:-2]:
        value.requires_grad_(True)

    support = dispatcher.probe_recurrent_training_v1(*inputs)
    assert not support["supported"]
    assert (
        support["implementation"]
        == "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    )
    assert "full-model release gate" in support["reason"]

    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "factorized")
    support = dispatcher.probe_recurrent_training_v1(*inputs)
    assert not support["supported"]
    assert (
        support["implementation"]
        == "native-nvidia-rwkv7-factorized-recurrent-training-v1"
    )
    assert "CUDA" in support["reason"]


def test_training_matrix_policy_is_exact_and_requires_cuda(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    inputs = list(cpu_inputs())
    for value in inputs[:-2]:
        value.requires_grad_(True)

    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "matrix")
    support = dispatcher.probe_recurrent_training_v1(*inputs)
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
    inputs = list(cpu_inputs(tokens=128, batch=4))
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

    checkpoint_replay = dispatcher.probe_recurrent_training_v1(
        *inputs,
        fully_active=True,
        initial_state_zero=True,
        token_aligned=True,
        force_reference_recurrent=True,
    )
    assert checkpoint_replay["supported"] is False
    assert "pinned checkpoint forward and replay" in checkpoint_replay["reason"]

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

    inputs[-1] = torch.ones(4, 128, dtype=torch.bool)
    inputs[-1][:, 0] = False
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
    inputs = list(cpu_inputs(tokens=128, batch=4))
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
        *cpu_inputs(tokens=128, batch=4),
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
            *cpu_inputs(tokens=128, batch=4),
            fully_active=True,
            initial_state_zero=True,
            token_aligned=True,
        )
    assert calls == {"probe": 1, "run": 1}


@pytest.mark.parametrize(
    (
        "batch",
        "tokens",
        "fully_active",
        "initial_state_zero",
        "token_aligned",
        "expected",
    ),
    [
        (4, 128, True, True, True, True),
        (1, 128, True, True, True, False),
        (4, 16, True, True, True, False),
        (8, 128, True, True, True, False),
        (4, 256, True, True, True, False),
        (4, 128, False, True, True, False),
        (4, 128, True, False, True, False),
        (4, 128, True, True, False, False),
    ],
)
def test_training_adaptive_fast_domain_is_conservative(
    batch,
    tokens,
    fully_active,
    initial_state_zero,
    token_aligned,
    expected,
):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    assert (
        dispatcher.adaptive_training_fast_domain_v1(
            batch=batch,
            tokens=tokens,
            fully_active=fully_active,
            initial_state_zero=initial_state_zero,
            token_aligned=token_aligned,
        )
        is expected
    )


def test_training_program_preflight_rejects_frozen_or_reentrant_input(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    hidden = torch.randn(4, 128, 64)
    mask = torch.ones(4, 128, dtype=torch.bool)

    result = dispatcher.probe_training_program_v1(
        hidden,
        mask,
        training=True,
        fully_active=True,
        initial_state_zero=True,
        token_aligned=True,
        autograd_leaf_eligible=False,
        head_dim=64,
    )

    assert result["supported"] is False
    assert result["implementation"] == (
        "native-nvidia-rwkv7-adaptive-training-program-v1"
    )
    assert "gradient-bearing inputs" in result["reason"]


def test_training_program_preloads_both_native_dependencies(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.official_training_cuda")
    calls = []
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _self: True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 9))
    monkeypatch.setattr(
        train_temp,
        "recurrent_training_cuda_available",
        lambda *, build: calls.append(("recurrent", build)) or True,
    )
    monkeypatch.setattr(
        dispatcher,
        "load_mix6_training_cuda_extension",
        lambda *, device: calls.append(("mix6", device)),
    )
    monkeypatch.setattr(
        dispatcher,
        "_certify_recurrent_runtime",
        lambda device: calls.append(("certify", device)),
    )
    hidden = torch.randn(4, 128, 64, dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(4, 128, dtype=torch.bool)
    result = dispatcher.probe_training_program_v1(
        hidden,
        mask,
        training=True,
        fully_active=True,
        initial_state_zero=True,
        token_aligned=True,
        autograd_leaf_eligible=True,
        head_dim=64,
    )

    assert result["supported"] is True
    assert calls == [
        ("recurrent", True),
        ("mix6", hidden.device),
        ("certify", hidden.device),
    ]


def test_forged_adaptive_bool_does_not_bypass_recurrent_runtime_preflight(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    captured = []

    def probe(*_args, **kwargs):
        captured.append(dict(kwargs))
        return {
            "supported": True,
            "implementation": ("native-nvidia-rwkv7-factorized-recurrent-training-v1"),
            "reason": "runtime-preflight capture",
        }

    monkeypatch.setattr(dispatcher, "_probe_factorized", probe)
    result = dispatcher.probe_recurrent_training_v1(
        *cpu_inputs(tokens=128, batch=4),
        adaptive_fast_program=True,
        fully_active=True,
        initial_state_zero=True,
        token_aligned=True,
    )

    assert result["supported"] is True
    assert captured == [{"initial_state_zero": True}]


def test_recurrent_runtime_certificate_is_bound_to_exact_device():
    certificates = importlib.import_module("rwkv7_kernels._runtime_preflight")
    certificates._reset_for_tests()
    certificates._certify_recurrent_runtime(torch.device("cuda:1"))

    assert certificates.recurrent_runtime_certified(torch.device("cuda:1")) is True
    assert certificates.recurrent_runtime_certified(torch.device("cuda:0")) is False
    assert certificates.recurrent_runtime_certified(torch.device("cpu")) is False
    certificates._reset_for_tests()


def test_certified_recurrent_device_skips_capability_and_loader(monkeypatch):
    certificates = importlib.import_module("rwkv7_kernels._runtime_preflight")
    factorized = importlib.import_module("rwkv7_kernels.recurrent.training_factorized")

    class FakeCudaTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value, device_index=0):
            result = torch.Tensor._make_subclass(cls, value, value.requires_grad)
            result._device_index = device_index
            return result

        @property
        def is_cuda(self):
            return True

        @property
        def device(self):
            return torch.device("cuda", self._device_index)

    def fake_cuda(value, *, requires_grad=False):
        return FakeCudaTensor(value.requires_grad_(requires_grad))

    shape = (4, 128, 1, 64)
    vectors = [
        fake_cuda(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=True),
        fake_cuda(torch.zeros(shape, dtype=torch.float32), requires_grad=True),
        *[
            fake_cuda(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=True)
            for _ in range(4)
        ],
    ]
    state = fake_cuda(torch.zeros(4, 1, 64, 64, dtype=torch.float32))
    mask = fake_cuda(torch.ones(4, 128, dtype=torch.bool))
    certificates._reset_for_tests()
    certificates._certify_recurrent_runtime(torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("certified device repeated capability probing")
        ),
    )

    support = factorized.probe_recurrent_training_v1(
        *vectors,
        state,
        mask,
        initial_state_zero=True,
    )

    assert support["supported"] is True
    assert "preflight" in support["reason"]
    certificates._reset_for_tests()


def test_mix6_capability_probe_is_cached_per_device(monkeypatch):
    mix6 = importlib.import_module("rwkv7_kernels.time_mix.training_mix6")
    calls = []
    with mix6._CAPABILITY_LOCK:
        mix6._CAPABILITY_DEVICES.clear()
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: calls.append(torch.device(device)) or (8, 9),
    )

    assert mix6._bf16_capability_available(torch.device("cuda:0")) is True
    assert mix6._bf16_capability_available(torch.device("cuda:0")) is True
    assert mix6._bf16_capability_available(torch.device("cuda:1")) is True

    assert calls == [torch.device("cuda:0"), torch.device("cuda:1")]
    with mix6._CAPABILITY_LOCK:
        mix6._CAPABILITY_DEVICES.clear()


@pytest.mark.parametrize(
    ("batch", "tokens", "fully_active", "initial_state_zero", "expected"),
    [
        (1, 16, True, True, "matrix"),
        (4, 16, True, True, "matrix"),
        (1, 17, True, True, "matrix"),
        (4, 17, True, True, "matrix"),
        (1, 128, True, True, "matrix"),
        (4, 128, True, True, "factorized"),
        (8, 128, True, True, "matrix"),
        (4, 256, True, True, "matrix"),
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


def test_preflight_certified_recurrent_decline_does_not_mix_with_matrix(
    monkeypatch,
):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    matrix_calls = []
    monkeypatch.setattr(
        dispatcher,
        "_probe_factorized",
        lambda *_args, **_kwargs: {
            "supported": False,
            "implementation": ("native-nvidia-rwkv7-factorized-recurrent-training-v1"),
            "reason": "simulated late decline",
        },
    )

    def matrix(*_args, **_kwargs):
        matrix_calls.append(True)
        return {
            "supported": True,
            "implementation": ("torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"),
            "reason": "must not be selected after coupled preflight",
        }

    monkeypatch.setattr(dispatcher, "_probe_matrix", matrix)
    shape = (4, 128, 1, 64)
    values = [torch.randn(shape, dtype=torch.float16) for _ in range(6)]
    state = torch.zeros(4, 1, 64, 64, dtype=torch.float32)
    mask = torch.ones(4, 128, dtype=torch.bool)

    support = dispatcher.probe_recurrent_training_v1(
        *values,
        state,
        mask,
        adaptive_fast_program=True,
        fully_active=True,
        initial_state_zero=True,
        token_aligned=True,
    )

    assert not support["supported"]
    assert support["implementation"] == (
        "native-nvidia-rwkv7-factorized-recurrent-training-v1"
    )
    assert "preflight-certified" in support["reason"]
    assert not matrix_calls


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
        adaptive_fast_program=False,
        fully_active=False,
        initial_state_zero=True,
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
    outside_domain = dispatcher.probe_linear_training_v1(
        value,
        weight,
        None,
        adaptive_fast_program=True,
        fully_active=True,
        initial_state_zero=True,
        token_aligned=True,
    )
    assert not outside_domain["supported"]
    assert outside_domain["implementation"] == "torch-reference-linear-v1"
    assert "outside the certified adaptive fast domain" in outside_domain["reason"]

    fast_value = torch.randn(4, 128, 4, requires_grad=True)
    dense = dispatcher.probe_linear_training_v1(
        fast_value,
        weight,
        None,
        adaptive_fast_program=True,
        fully_active=True,
        initial_state_zero=True,
        token_aligned=True,
    )
    assert dense["supported"]
    assert dense["implementation"] == ("torch-cuda-rwkv7-flattened-linear-training-v1")

    unaligned = dispatcher.probe_linear_training_v1(
        value,
        weight,
        None,
        adaptive_fast_program=True,
        fully_active=True,
        initial_state_zero=True,
        token_aligned=False,
    )
    assert not unaligned["supported"]
    assert unaligned["implementation"] == "torch-reference-linear-v1"
    assert "token-length-unaligned" in unaligned["reason"]

    stateful = dispatcher.probe_linear_training_v1(
        fast_value,
        weight,
        None,
        adaptive_fast_program=True,
        fully_active=True,
        initial_state_zero=False,
        token_aligned=True,
    )
    assert not stateful["supported"]
    assert stateful["implementation"] == "torch-reference-linear-v1"
    assert "stateful" in stateful["reason"]


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
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    value = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)

    support = dispatcher.probe_linear_training_v1(value, weight, None)
    assert not support["supported"]
    assert support["implementation"] == "torch-cuda-rwkv7-flattened-linear-training-v1"
    assert "full-model precision" in support["reason"]

    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "factorized")
    support = dispatcher.probe_linear_training_v1(value, weight, None)
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
        *cpu_inputs(tokens=128, batch=4),
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
    dispatcher = importlib.import_module("rwkv7_kernels.dispatcher")
    support = dispatcher.probe_recurrent_v1(*cpu_inputs())
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
    dispatcher = importlib.import_module("rwkv7_kernels.dispatcher")
    with pytest.raises(ValueError, match="RWKV7_KERNEL_IMPL"):
        dispatcher.probe_recurrent_v1(*cpu_inputs())


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
