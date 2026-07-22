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
        "linear_training_v1",
        "model_forward_v1",
        "probe_linear_training_v1",
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


def test_explicit_native_training_reports_actual_training_capability(monkeypatch):
    monkeypatch.setenv("RWKV7_MODEL_KERNEL_IMPL", "native")
    kernels = importlib.import_module("rwkv7_kernels")

    class Config:
        head_dim = 64

    class Base:
        embeddings = type(
            "Embeddings",
            (),
            {"weight": torch.zeros(4, 4, dtype=torch.bfloat16)},
        )()
        layers = [
            type(
                "Layer",
                (),
                {
                    "ffn": type(
                        "FFN",
                        (),
                        {
                            "key": torch.nn.Linear(4, 4),
                            "value": torch.nn.Linear(4, 4),
                        },
                    )()
                },
            )()
        ]

    owner = type(
        "Owner",
        (),
        {"model": Base(), "lm_head": object(), "config": Config()},
    )()
    request = {
        "model_kind": "causal_lm",
        "training": True,
        "grad_enabled": True,
        "use_cache": False,
        "input_ids": torch.ones(1, 16, dtype=torch.long),
        "inputs_embeds": None,
        "labels": None,
        "output_hidden_states": False,
        "output_attentions": False,
    }
    support = kernels.probe_model_forward_v1(owner, request)
    assert not support["supported"]
    assert support["implementation"] == "native-nvidia-train-temp-autograd-v2"
    assert support["phase"] == "training"
    assert "CUDA" in support["reason"]


def test_native_training_forward_consumes_probe_ticket_once(monkeypatch):
    monkeypatch.setenv("RWKV7_MODEL_KERNEL_IMPL", "native")
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    runtime = importlib.import_module("rwkv7_kernels.nvidia.training_runtime")
    calls = {"probe": 0, "run": 0}

    def probe(_owner, _request):
        calls["probe"] += 1
        return {
            "supported": True,
            "implementation": "native-nvidia-train-temp-autograd-v2",
            "reason": "migrated train_temp autograd implementation selected",
            "phase": "training",
        }

    def run(_owner, _request):
        calls["run"] += 1
        return {
            "output_kind": "causal_lm",
            "logits": torch.zeros(1, 16, 4),
            "loss": None,
            "past_key_values": None,
            "hidden_states": None,
            "implementation": "native-nvidia-train-temp-autograd-v2",
            "phase": "training",
        }

    monkeypatch.setattr(dispatcher, "_probe_native", probe)
    monkeypatch.setattr(runtime, "run_training", run)
    owner = object()
    request = {
        "model_kind": "causal_lm",
        "training": True,
        "grad_enabled": True,
        "use_cache": False,
        "input_ids": torch.ones(1, 16, dtype=torch.long),
        "labels": torch.ones(1, 16, dtype=torch.long),
    }

    support = dispatcher.probe_model_forward_v1(owner, request)
    result = dispatcher.model_forward_v1(owner, request)

    assert support["supported"]
    assert result["phase"] == "training"
    assert calls == {"probe": 1, "run": 1}
    assert dispatcher._NATIVE_TRAINING_PROBE_TICKET_KEY not in request


def test_native_training_probe_ticket_rejects_mutated_label_values(monkeypatch):
    monkeypatch.setenv("RWKV7_MODEL_KERNEL_IMPL", "native")
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    calls = {"probe": 0}

    def probe(_owner, request):
        calls["probe"] += 1
        labels = request["labels"]
        invalid = (labels < -100) | ((labels < 0) & (labels != -100))
        supported = not bool(invalid.any())
        return {
            "supported": supported,
            "implementation": "native-nvidia-train-temp-autograd-v2",
            "reason": (
                "accepted" if supported else "native training labels are invalid"
            ),
            "phase": "training",
        }

    monkeypatch.setattr(dispatcher, "_probe_native", probe)
    owner = object()
    request = {
        "model_kind": "causal_lm",
        "training": True,
        "grad_enabled": True,
        "use_cache": False,
        "labels": torch.ones(1, 16, dtype=torch.long),
    }

    assert dispatcher.probe_model_forward_v1(owner, request)["supported"]
    request["labels"][0, 0] = -1
    with pytest.raises(RuntimeError, match="labels are invalid"):
        dispatcher.model_forward_v1(owner, request)

    assert calls == {"probe": 2}
    assert dispatcher._NATIVE_TRAINING_PROBE_TICKET_KEY not in request


def test_native_training_never_bypasses_wrapped_ffn_adapters(monkeypatch):
    monkeypatch.setenv("RWKV7_MODEL_KERNEL_IMPL", "native")
    kernels = importlib.import_module("rwkv7_kernels")

    class WrappedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 4)

        def forward(self, value):
            return self.base_layer(value)

    ffn = type(
        "FFN",
        (),
        {"key": WrappedLinear(), "value": WrappedLinear()},
    )()
    base = type(
        "Base",
        (),
        {
            "embeddings": type(
                "Embeddings",
                (),
                {"weight": torch.zeros(4, 4, dtype=torch.bfloat16)},
            )(),
            "layers": [type("Layer", (), {"ffn": ffn})()],
        },
    )()
    owner = type(
        "Owner",
        (),
        {
            "model": base,
            "lm_head": object(),
            "config": type("Config", (), {"head_dim": 64})(),
        },
    )()
    support = kernels.probe_model_forward_v1(
        owner,
        {
            "model_kind": "causal_lm",
            "training": True,
            "grad_enabled": True,
            "use_cache": False,
            "input_ids": torch.ones(1, 16, dtype=torch.long),
            "inputs_embeds": None,
            "labels": None,
            "output_hidden_states": False,
            "output_attentions": False,
        },
    )
    assert not support["supported"]
    assert "adapters use the reference autograd path" in support["reason"]


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
    assert support["implementation"] == "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    assert "full-model release gate" in support["reason"]

    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "factorized")
    support = kernels.probe_recurrent_training_v1(*inputs)
    assert not support["supported"]
    assert support["implementation"] == "native-nvidia-rwkv7-factorized-recurrent-training-v1"
    assert "CUDA" in support["reason"]


def test_training_matrix_policy_is_exact_and_requires_cuda(monkeypatch):
    kernels = importlib.import_module("rwkv7_kernels")
    inputs = list(cpu_inputs())
    for value in inputs[:-2]:
        value.requires_grad_(True)

    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "matrix")
    support = kernels.probe_recurrent_training_v1(*inputs)
    assert not support["supported"]
    assert support["implementation"] == "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    assert "CUDA" in support["reason"]


def test_training_adaptive_policy_reports_the_actual_recurrent_leaf(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    inputs = list(cpu_inputs(tokens=16))
    for value in inputs[:-2]:
        value.requires_grad_(True)
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    monkeypatch.setattr(
        dispatcher,
        "_probe_factorized",
        lambda *_args, **_kwargs: {
            "supported": True,
            "implementation": (
                "native-nvidia-rwkv7-factorized-recurrent-training-v1"
            ),
            "reason": "dense test request",
        },
    )
    monkeypatch.setattr(
        dispatcher,
        "_probe_matrix",
        lambda *_args, **_kwargs: {
            "supported": True,
            "implementation": (
                "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
            ),
            "reason": "exact test request",
        },
    )

    dense = dispatcher.probe_recurrent_training_v1(*inputs)
    assert dense["implementation"] == (
        "native-nvidia-rwkv7-factorized-recurrent-training-v1"
    )

    inputs[-1] = torch.tensor([[False, *([True] * 15)]])
    masked = dispatcher.probe_recurrent_training_v1(*inputs)
    assert masked["implementation"] == (
        "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    )
    assert "masked recurrent request" in masked["reason"]

    unaligned_inputs = list(cpu_inputs(tokens=17))
    for value in unaligned_inputs[:-2]:
        value.requires_grad_(True)
    unaligned = dispatcher.probe_recurrent_training_v1(*unaligned_inputs)
    assert unaligned["implementation"] == (
        "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    )
    assert "unaligned recurrent request" in unaligned["reason"]


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
    assert dense["implementation"] == (
        "torch-cuda-rwkv7-flattened-linear-training-v1"
    )

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


def test_training_flattened_linear_declares_small_row_numerical_gate():
    source = (
        ROOT
        / "kernels"
        / "rwkv7_kernels"
        / "linear"
        / "training_flattened.py"
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
