from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

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
    model = torch.nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    callback = common.ReproCallback(tmp_path)
    callback.saw_finite_loss = True
    callback.on_pre_optimizer_step(None, None, None, model=model)
    callback.write_status(1)

    routes = json.loads((tmp_path / "backend_routes.json").read_text())
    checks = json.loads((tmp_path / "training_checks.json").read_text())
    assert routes == [{"event": "pre_optimizer_step", **route}]
    assert checks["adapter_reference_fallback"] is True
    assert checks["native_training"] is False
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
    modeling_module = types.ModuleType(modeling_name)
    ops_module = types.ModuleType(ops_name)

    def maybe_model_forward():
        raise AssertionError("the route resolver must not execute the dispatcher")

    maybe_model_forward.__module__ = ops_name
    modeling_module.maybe_model_forward = maybe_model_forward
    ops_module.get_last_model_route = lambda: dict(route)
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
    assert routes == [{"event": "pre_optimizer_step", **route}]
    assert checks["adapter_reference_fallback"] is True
