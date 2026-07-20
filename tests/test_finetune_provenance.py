from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch


def load_finetune_common():
    path = Path(__file__).resolve().parents[1] / "examples" / "finetune" / "common.py"
    spec = importlib.util.spec_from_file_location("rwkv7_finetune_common_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optional_artifact_and_adapter_route_provenance(tmp_path, monkeypatch):
    common = load_finetune_common()
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"immutable-wheel")
    row = common.optional_artifact(str(artifact))
    assert row["path"] == str(artifact.resolve())
    assert row["bytes"] == len(b"immutable-wheel")
    assert len(row["sha256"]) == 64

    route = {
        "requested": "auto",
        "selected": "reference",
        "implementation": "torch-reference-model-v1",
        "reason": "adapter-wrapped FFN modules use reference autograd",
        "phase": "training",
    }
    monkeypatch.setattr("rwkv7_hf.ops_rwkv7.get_last_model_route", lambda: dict(route))
    monkeypatch.setattr("rwkv7_hf.ops_rwkv7.get_last_recurrent_route", lambda: None)
    monkeypatch.setattr("rwkv7_hf.ops_rwkv7.get_last_linear_route", lambda: None)
    model = torch.nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    callback = common.ReproCallback(tmp_path)
    callback.saw_finite_loss = True
    callback.on_pre_optimizer_step(None, None, None, model=model)
    callback.write_status(1)

    routes = json.loads((tmp_path / "backend_routes.json").read_text())
    checks = json.loads((tmp_path / "training_checks.json").read_text())
    assert routes == [
        {"event": "pre_optimizer_step", "boundary": "model", **route}
    ]
    assert checks["adapter_reference_fallback"] is True
    assert checks["native_training"] is False
    assert checks["model_reference_training"] is True
    assert checks["matrix_recurrent_training"] is False
    assert checks["factorized_recurrent_training"] is False
    assert checks["flattened_linear_training"] is False
    assert checks["nonzero_gradient"] is True


def test_remote_model_namespace_route_is_resolved(tmp_path, monkeypatch):
    common = load_finetune_common()
    modeling_name = "transformers_modules.rwkv7_test.modeling_rwkv7"
    ops_name = "transformers_modules.rwkv7_test.ops_rwkv7"
    route = {
        "requested": "auto",
        "selected": "reference",
        "implementation": "torch-reference-model-v1",
        "reason": "native prefill requires the causal-LM boundary",
        "phase": "training",
    }
    recurrent_route = {
        "requested": "auto",
        "selected": "optimized",
        "implementation": "native-nvidia-rwkv7-factorized-recurrent-training-v1",
        "reason": "dense zero-state BF16 CUDA autograd request is supported",
    }
    linear_route = {
        "requested": "optimized",
        "selected": "optimized",
        "implementation": "torch-cuda-rwkv7-flattened-linear-training-v1",
        "reason": "contiguous CUDA training projection is supported by PyTorch cuBLAS",
    }
    modeling_module = types.ModuleType(modeling_name)
    ops_module = types.ModuleType(ops_name)

    def maybe_model_forward():
        raise AssertionError("the route resolver must not execute the dispatcher")

    maybe_model_forward.__module__ = ops_name
    modeling_module.maybe_model_forward = maybe_model_forward
    ops_module.get_last_model_route = lambda: dict(route)
    ops_module.get_last_recurrent_route = lambda: dict(recurrent_route)
    ops_module.get_last_linear_route = lambda: dict(linear_route)
    monkeypatch.setitem(sys.modules, modeling_name, modeling_module)
    monkeypatch.setitem(sys.modules, ops_name, ops_module)

    RemoteModel = type("RemoteModel", (), {"__module__": modeling_name})
    model = RemoteModel()
    callback = common.ReproCallback(tmp_path)
    callback._capture_backend_route("pre_optimizer_step", model)
    callback.saw_finite_loss = True
    callback.saw_nonzero_gradient = True
    callback.write_status(1)
    routes = json.loads((tmp_path / "backend_routes.json").read_text())
    checks = json.loads((tmp_path / "training_checks.json").read_text())
    assert routes == [
        {"event": "pre_optimizer_step", "boundary": "model", **route},
        {
            "event": "pre_optimizer_step",
            "boundary": "recurrent",
            **recurrent_route,
        },
        {
            "event": "pre_optimizer_step",
            "boundary": "linear",
            **linear_route,
        },
    ]
    assert checks["adapter_reference_fallback"] is True
    assert checks["model_reference_training"] is True
    assert checks["matrix_recurrent_training"] is False
    assert checks["factorized_recurrent_training"] is True
    assert checks["flattened_linear_training"] is True


def test_finetune_precision_arguments_are_explicit_and_standard():
    common = load_finetune_common()
    args = SimpleNamespace(
        model_revision="local-test-revision",
        torch_dtype="bfloat16",
    )

    assert common.model_load_kwargs(args) == {
        "revision": "local-test-revision",
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }
    assert common.trainer_precision_flags() == {"bf16": False, "fp16": False}
    assert common.gradient_checkpointing_kwargs() == {"use_reentrant": False}

    args.torch_dtype = "auto"
    assert common.model_load_kwargs(args) == {
        "revision": "local-test-revision",
        "trust_remote_code": True,
    }


def test_process_trace_reconciles_preference_training_routes(tmp_path):
    common = load_finetune_common()
    checks = {
        "matrix_recurrent_training": False,
        "factorized_recurrent_training": False,
        "flattened_linear_training": False,
    }
    trace = {
        "schema": "rwkv7-kernel-route-trace-v2",
        "actual_recurrent_calls": {
            "native-nvidia-rwkv7-factorized-recurrent-training-v1": 24
        },
        "actual_linear_calls": {
            "torch-cuda-rwkv7-flattened-linear-training-v1": 333
        },
    }
    (tmp_path / "training_checks.json").write_text(json.dumps(checks))
    (tmp_path / "kernel_route_trace.json").write_text(json.dumps(trace))

    common.reconcile_kernel_trace_checks(tmp_path)

    merged = json.loads((tmp_path / "training_checks.json").read_text())
    assert merged["matrix_recurrent_training"] is False
    assert merged["factorized_recurrent_training"] is True
    assert merged["flattened_linear_training"] is True
    assert merged["kernel_trace_schema"] == "rwkv7-kernel-route-trace-v2"


def test_lora_parameter_dtype_requires_one_explicit_dtype():
    common = load_finetune_common()

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(1, 2, bias=False, dtype=torch.bfloat16)}
            )

    assert common.lora_parameter_dtype(Adapter()) == torch.bfloat16
