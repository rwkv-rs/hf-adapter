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
    unload_kernel_package()
    yield
    unload_kernel_package()


def cpu_inputs():
    shape = (1, 2, 1, 64)
    values = [torch.randn(shape, dtype=torch.float16) for _ in range(6)]
    state = torch.randn(1, 1, 64, 64, dtype=torch.float32)
    return (*values, state, None)


def test_public_kernel_surface_is_versioned_and_small():
    kernels = importlib.import_module("rwkv7_kernels")
    assert kernels.RWKV7_KERNEL_API_VERSION == 2
    assert kernels.__all__ == [
        "RWKV7_KERNEL_API_VERSION",
        "model_forward_v1",
        "probe_model_forward_v1",
        "probe_recurrent_v1",
        "recurrent_v1",
    ]


def test_model_forward_protocol_is_explicitly_capability_gated():
    kernels = importlib.import_module("rwkv7_kernels")
    request = {
        "model_kind": "base",
        "training": False,
        "use_cache": True,
    }
    support = kernels.probe_model_forward_v1(object(), request)
    assert support == {
        "supported": False,
        "implementation": "rwkv7-model-backend-v2",
        "reason": (
            "whole-model backend-v2 is not available for this shape; "
            "the adapter will use its readable reference layer loop"
        ),
        "phase": "prefill",
    }
    with pytest.raises(RuntimeError, match="whole-model backend-v2"):
        kernels.model_forward_v1(object(), request)


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


def test_default_auto_prefill_reports_graph_implementation_on_cpu():
    kernels = importlib.import_module("rwkv7_kernels")
    support = kernels.probe_recurrent_v1(*cpu_inputs())
    assert not support["supported"]
    assert support["implementation"] == "torch-cuda-graph-reference-v1"
    assert "CUDA" in support["reason"]


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
    assert payload["requested_policy"] == "auto"
    assert payload["actual_recurrent_calls"] == {"triton-test": 1}


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
        torch.randn(batch, time, heads, width, dtype=torch.float16)
        for _ in range(6)
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
