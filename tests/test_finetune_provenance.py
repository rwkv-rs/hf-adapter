from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from evaluation.validate_finetune_runs import (
    ADAPTIVE_TRAINING_PROGRAM_IMPLEMENTATION,
    EXPECTED_DATASETS,
    EXPECTED_TARGETS,
    FACTORIZED_RECURRENT_IMPLEMENTATION,
    FLATTENED_LINEAR_IMPLEMENTATION,
    MATRIX_RECURRENT_IMPLEMENTATION,
    MIX6_IMPLEMENTATION,
    validate_run,
)


def load_finetune_common():
    path = Path(__file__).resolve().parents[1] / "examples" / "finetune" / "common.py"
    spec = importlib.util.spec_from_file_location("rwkv7_finetune_common_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optional_artifact_and_readable_model_route_provenance(tmp_path, monkeypatch):
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
        "reason": "training preserves the readable HF layer loop",
        "phase": "training",
    }
    monkeypatch.setattr("rwkv7_hf.ops_rwkv7.get_last_model_route", lambda: dict(route))
    monkeypatch.setattr("rwkv7_hf.ops_rwkv7.get_last_recurrent_route", lambda: None)
    monkeypatch.setattr("rwkv7_hf.ops_rwkv7.get_last_linear_route", lambda: None)
    monkeypatch.setattr("rwkv7_hf.ops_rwkv7.get_last_mix6_route", lambda: None)
    monkeypatch.setattr(
        "rwkv7_hf.ops_rwkv7.get_last_training_program_route", lambda: None
    )
    model = torch.nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    callback = common.ReproCallback(tmp_path)
    callback.saw_finite_loss = True
    callback.on_pre_optimizer_step(None, None, None, model=model)
    callback.write_status(1)

    routes = json.loads((tmp_path / "backend_routes.json").read_text())
    checks = json.loads((tmp_path / "training_checks.json").read_text())
    assert routes == [{"event": "pre_optimizer_step", "boundary": "model", **route}]
    assert checks["readable_model_loop"] is True
    assert checks["matrix_recurrent_leaf"] is False
    assert checks["factorized_recurrent_leaf"] is False
    assert checks["flattened_linear_leaf"] is False
    assert checks["mix6_leaf"] is False
    assert checks["adaptive_fast_program"] is False
    assert checks["adaptive_program_fallback"] is False
    assert checks["clean_leaf_training"] is False
    assert checks["historical_whole_model_diagnostic"] is False
    assert checks["nonzero_gradient"] is True


def test_remote_model_namespace_route_is_resolved(tmp_path, monkeypatch):
    common = load_finetune_common()
    modeling_name = "transformers_modules.rwkv7_test.modeling_rwkv7"
    ops_name = "transformers_modules.rwkv7_test.ops_rwkv7"
    route = {
        "requested": "auto",
        "selected": "reference",
        "implementation": "torch-reference-model-v1",
        "reason": "training preserves the readable HF layer loop",
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
    mix6_route = {
        "requested": "optimized",
        "selected": "optimized",
        "implementation": "native-nvidia-rwkv7-mix6-training-v1",
        "reason": "explicit-shift BF16 CUDA Mix6 training leaf is supported",
    }
    program_route = {
        "selected": "optimized",
        "implementation": "native-nvidia-rwkv7-adaptive-training-program-v1",
        "reason": "the coupled B4/T128 program is certified",
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
    ops_module.get_last_mix6_route = lambda: dict(mix6_route)
    ops_module.get_last_training_program_route = lambda: dict(program_route)
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
        {
            "event": "pre_optimizer_step",
            "boundary": "mix6",
            **mix6_route,
        },
        {
            "event": "pre_optimizer_step",
            "boundary": "program",
            **program_route,
        },
    ]
    assert checks["readable_model_loop"] is True
    assert checks["matrix_recurrent_leaf"] is False
    assert checks["factorized_recurrent_leaf"] is True
    assert checks["flattened_linear_leaf"] is True
    assert checks["mix6_leaf"] is True
    assert checks["adaptive_fast_program"] is True
    assert checks["adaptive_program_fallback"] is False
    assert checks["clean_leaf_training"] is True
    assert checks["historical_whole_model_diagnostic"] is False


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
        "readable_model_loop": True,
        "matrix_recurrent_leaf": False,
        "factorized_recurrent_leaf": False,
        "flattened_linear_leaf": False,
        "mix6_leaf": False,
        "historical_whole_model_diagnostic": False,
    }
    trace = {
        "schema": "rwkv7-kernel-route-trace-v2",
        "actual_model_calls": {},
        "actual_recurrent_calls": {
            "native-nvidia-rwkv7-factorized-recurrent-training-v1": 24
        },
        "actual_linear_calls": {"torch-cuda-rwkv7-flattened-linear-training-v1": 333},
        "actual_mix6_calls": {"native-nvidia-rwkv7-mix6-training-v1": 144},
    }
    (tmp_path / "training_checks.json").write_text(json.dumps(checks))
    (tmp_path / "kernel_route_trace.json").write_text(json.dumps(trace))

    common.reconcile_kernel_trace_checks(tmp_path)

    merged = json.loads((tmp_path / "training_checks.json").read_text())
    assert merged["matrix_recurrent_leaf"] is False
    assert merged["factorized_recurrent_leaf"] is True
    assert merged["flattened_linear_leaf"] is True
    assert merged["mix6_leaf"] is True
    assert merged["clean_leaf_training"] is True
    assert merged["historical_whole_model_diagnostic"] is False
    assert merged["kernel_trace_schema"] == "rwkv7-kernel-route-trace-v2"


def test_historical_whole_model_route_is_diagnostic_and_rejected(tmp_path, monkeypatch):
    common = load_finetune_common()
    route = {
        "requested": "optimized",
        "selected": "optimized",
        "implementation": "native-nvidia-official-training-autograd-v2",
        "reason": "historical diagnostic",
        "phase": "training",
    }
    monkeypatch.setattr("rwkv7_hf.ops_rwkv7.get_last_model_route", lambda: route)
    monkeypatch.setattr("rwkv7_hf.ops_rwkv7.get_last_recurrent_route", lambda: None)
    monkeypatch.setattr("rwkv7_hf.ops_rwkv7.get_last_linear_route", lambda: None)
    monkeypatch.setattr("rwkv7_hf.ops_rwkv7.get_last_mix6_route", lambda: None)
    monkeypatch.setattr(
        "rwkv7_hf.ops_rwkv7.get_last_training_program_route", lambda: None
    )
    model = torch.nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    callback = common.ReproCallback(tmp_path)
    callback.saw_finite_loss = True
    callback.on_pre_optimizer_step(None, None, None, model=model)

    with pytest.raises(RuntimeError, match="training checks failed"):
        callback.write_status(1)

    checks = json.loads((tmp_path / "training_checks.json").read_text())
    assert checks["historical_whole_model_diagnostic"] is True


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


def write_clean_finetune_evidence(path: Path) -> None:
    dataset_name, dataset_revision = EXPECTED_DATASETS["sft"]
    payloads = {
        "exit_status.json": {"returncode": 0},
        "training_checks.json": {
            "finite_loss": True,
            "nonzero_gradient": True,
            "global_step": 100,
            "readable_model_loop": True,
            "matrix_recurrent_leaf": False,
            "factorized_recurrent_leaf": True,
            "flattened_linear_leaf": True,
            "mix6_leaf": True,
            "adaptive_fast_program": True,
            "adaptive_program_fallback": False,
            "clean_leaf_training": True,
            "historical_whole_model_diagnostic": False,
        },
        "adapter_reload.json": {"close": True, "max_abs": 0.0},
        "checkpoint_inventory.json": [{"path": "adapter_model.safetensors"}],
        "changed_parameters.json": ["base_model.model.layers.0.att.r_proj"],
        "resolved_config.json": {
            "seed": 42,
            "max_length": 512,
            "train_samples": 1024,
            "eval_samples": 128,
            "max_steps": 100,
            "gradient_accumulation_steps": 1,
            "report_to": "none",
            "dataset_name": dataset_name,
            "dataset_revision": dataset_revision,
            "target_modules": EXPECTED_TARGETS,
            "torch_dtype": "bfloat16",
            "lora_dtype": "model",
            "source_revision": "model-revision",
            "code_sha": "a" * 40,
        },
        "environment.json": {"transformers": "4.56.2", "trl": "0.20.0"},
        "model_provenance.json": {
            "resolved_revision": "model-revision",
            "files": {"model.safetensors": {"sha256": "b" * 64}},
        },
        "dataset_fingerprints.json": {
            "train": {"selected": [0]},
            "eval": {"selected": [1]},
        },
        "artifact_provenance.json": {
            "rwkv7_hf": {"sha256": "c" * 64},
            "rwkv7_kernels": {"sha256": "d" * 64},
        },
        "backend_routes.json": [
            {
                "event": "pre_optimizer_step",
                "boundary": "model",
                "selected": "reference",
                "phase": "training",
                "implementation": "torch-reference-model-v1",
            },
            {
                "event": "pre_optimizer_step",
                "boundary": "recurrent",
                "selected": "optimized",
                "implementation": (
                    "native-nvidia-rwkv7-factorized-recurrent-training-v1"
                ),
            },
            {
                "event": "pre_optimizer_step",
                "boundary": "linear",
                "selected": "optimized",
                "implementation": FLATTENED_LINEAR_IMPLEMENTATION,
            },
            {
                "event": "pre_optimizer_step",
                "boundary": "mix6",
                "selected": "optimized",
                "implementation": "native-nvidia-rwkv7-mix6-training-v1",
            },
            {
                "event": "pre_optimizer_step",
                "boundary": "program",
                "selected": "optimized",
                "implementation": ("native-nvidia-rwkv7-adaptive-training-program-v1"),
            },
        ],
        "kernel_route_trace.json": {
            "schema": "rwkv7-kernel-route-trace-v2",
            "requested_training_policy": "adaptive",
            "actual_model_calls": {},
            "actual_recurrent_calls": {
                "native-nvidia-rwkv7-factorized-recurrent-training-v1": 8
            },
            "actual_linear_calls": {
                "torch-cuda-rwkv7-flattened-linear-training-v1": 80
            },
            "actual_mix6_calls": {"native-nvidia-rwkv7-mix6-training-v1": 8},
        },
    }
    path.mkdir()
    for name, payload in payloads.items():
        (path / name).write_text(json.dumps(payload) + "\n")
    (path / "metrics.jsonl").write_text('{"loss": 1.0}\n')


def test_finetune_validator_accepts_only_readable_model_with_clean_leaves(tmp_path):
    run = tmp_path / "sft"
    write_clean_finetune_evidence(run)
    result = validate_run(
        run,
        "sft",
        100,
        require_backend_v2_routes=True,
        require_training_candidate="adaptive",
    )
    assert result["status"] == "passed", result["failures"]

    checks_path = run / "training_checks.json"
    checks = json.loads(checks_path.read_text())
    checks["historical_whole_model_diagnostic"] = True
    checks_path.write_text(json.dumps(checks) + "\n")
    trace_path = run / "kernel_route_trace.json"
    trace = json.loads(trace_path.read_text())
    trace["actual_model_calls"] = {"native-nvidia-official-training-autograd-v2": 1}
    trace_path.write_text(json.dumps(trace) + "\n")

    rejected = validate_run(
        run,
        "sft",
        100,
        require_backend_v2_routes=True,
        require_training_candidate="adaptive",
    )
    assert rejected["status"] == "failed"
    assert any("whole-model diagnostic" in row for row in rejected["failures"])


def test_finetune_validator_accepts_adaptive_program_fallback(tmp_path):
    run = tmp_path / "sft"
    write_clean_finetune_evidence(run)
    checks_path = run / "training_checks.json"
    checks = json.loads(checks_path.read_text())
    checks.update(
        {
            "factorized_recurrent_leaf": False,
            "flattened_linear_leaf": False,
            "clean_leaf_training": False,
            "adaptive_fast_program": False,
            "adaptive_program_fallback": True,
        }
    )
    checks_path.write_text(json.dumps(checks) + "\n")
    routes_path = run / "backend_routes.json"
    routes = json.loads(routes_path.read_text())
    routes = [row for row in routes if row["boundary"] == "model"]
    routes.extend(
        [
            {
                "event": "pre_optimizer_step",
                "boundary": "recurrent",
                "selected": "optimized",
                "implementation": MATRIX_RECURRENT_IMPLEMENTATION,
            },
            {
                "event": "pre_optimizer_step",
                "boundary": "linear",
                "selected": "reference",
                "implementation": "torch-reference-linear-v1",
            },
            {
                "event": "pre_optimizer_step",
                "boundary": "mix6",
                "selected": "optimized",
                "implementation": MIX6_IMPLEMENTATION,
            },
            {
                "event": "pre_optimizer_step",
                "boundary": "program",
                "selected": "reference",
                "implementation": ADAPTIVE_TRAINING_PROGRAM_IMPLEMENTATION,
            },
        ]
    )
    routes_path.write_text(json.dumps(routes) + "\n")
    trace_path = run / "kernel_route_trace.json"
    trace = json.loads(trace_path.read_text())
    trace["actual_recurrent_calls"] = {MATRIX_RECURRENT_IMPLEMENTATION: 8}
    trace["actual_linear_calls"] = {}
    trace_path.write_text(json.dumps(trace) + "\n")

    result = validate_run(
        run,
        "sft",
        100,
        require_backend_v2_routes=True,
        require_training_candidate="adaptive",
    )
    assert result["status"] == "passed", result["failures"]
    assert result["adaptive_route_evidence"]["fallback_program_observed"] is True

    trace["actual_linear_calls"] = {FLATTENED_LINEAR_IMPLEMENTATION: 1}
    trace_path.write_text(json.dumps(trace) + "\n")
    partial_result = validate_run(
        run,
        "sft",
        100,
        require_backend_v2_routes=True,
        require_training_candidate="adaptive",
    )
    assert partial_result["status"] == "failed"
    assert any(
        "fast training leaves were only partially observed" in failure
        for failure in partial_result["failures"]
    )

    # Preference trainers can execute several forwards before one optimizer
    # callback. The trace may therefore contain legal fast leaves even when the
    # last captured program decision was a shape-driven fallback.
    trace["actual_recurrent_calls"][FACTORIZED_RECURRENT_IMPLEMENTATION] = 2
    trace["actual_linear_calls"] = {FLATTENED_LINEAR_IMPLEMENTATION: 2}
    trace_path.write_text(json.dumps(trace) + "\n")
    aggregate_result = validate_run(
        run,
        "sft",
        100,
        require_backend_v2_routes=True,
        require_training_candidate="adaptive",
    )
    assert aggregate_result["status"] == "passed", aggregate_result["failures"]
    assert aggregate_result["adaptive_route_evidence"]["fast_program_observed"]
    assert not aggregate_result["adaptive_route_evidence"][
        "fast_program_route_observed"
    ]
    assert aggregate_result["adaptive_route_evidence"][
        "fast_program_inferred_from_complete_leaf_trace"
    ]

    trace["actual_linear_calls"] = {"unknown-training-linear": 1}
    trace_path.write_text(json.dumps(trace) + "\n")
    rejected = validate_run(
        run,
        "sft",
        100,
        require_backend_v2_routes=True,
        require_training_candidate="adaptive",
    )
    assert rejected["status"] == "failed"
    assert any(
        "unknown linear implementations" in failure for failure in rejected["failures"]
    )
