from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.build_backend_v2_device_validation import (
    EXPECTED_FLA_COMMIT,
    PRIMARY_REPORTS,
    REPORT_SCHEMAS,
    build,
)
from evaluation.validate_finetune_runs import (
    ADAPTIVE_TRAINING_PROGRAM_IMPLEMENTATION,
    MATRIX_RECURRENT_IMPLEMENTATION,
    MIX6_IMPLEMENTATION,
    REFERENCE_LINEAR_IMPLEMENTATION,
)
from scripts.release_route_contract import (
    ADAPTIVE_TRAINING_PROGRAM_ROUTE,
    FORMAL_ADAPTIVE_BACKEND_ENVIRONMENT,
    HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE,
    READABLE_TRAINING_MODEL_ROUTE,
    REQUIRED_TRAINING_LEAF_ROUTES,
)


SOURCE_SHA = "a" * 40
HARNESS_SHA = "b" * 40


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def wheel_rows(hf_sha: str, kernel_sha: str) -> dict:
    return {
        "rwkv7_hf": {"sha256": hf_sha},
        "rwkv7_kernels": {"sha256": kernel_sha},
    }


def setup_reports(tmp_path: Path) -> tuple[Namespace, dict[str, Path], str, str]:
    hf_wheel = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    kernel_wheel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    hf_wheel.write_bytes(b"hf-wheel")
    kernel_wheel.write_bytes(b"kernel-wheel")
    hf_sha = hashlib.sha256(hf_wheel.read_bytes()).hexdigest()
    kernel_sha = hashlib.sha256(kernel_wheel.read_bytes()).hexdigest()
    wheels = wheel_rows(hf_sha, kernel_sha)
    environment = {
        "command": ["python", "validator.py"],
        "gpu": "NVIDIA GeForce RTX 4080",
        "torch": "2.11.0+cu130",
        "cuda": "13.0",
        "backend_environment": dict(FORMAL_ADAPTIVE_BACKEND_ENVIRONMENT),
        "cuda_toolkit": {
            "cuda_home": "/toolkit",
            "torch_extensions_dir": "/extensions/backend-v2",
            "nvcc": "/toolkit/bin/nvcc",
            "nvcc_version": ["Cuda compilation tools, release 13.0, V13.0.88"],
            "provenance": {"sha256": "d" * 64},
        },
    }
    reports = {
        "correctness": {
            "models": [
                {
                    "cases": [
                        {
                            "route": {
                                "phase": "prefill",
                                "implementation": "native-nvidia-prefill-v2[self-chunk]",
                            }
                        },
                        {
                            "route": {
                                "phase": "decode",
                                "implementation": "native-nvidia-fused-decode-v2[dense]",
                            }
                        },
                    ]
                }
            ]
        },
        "hf_ecosystem": {
            "stages": [{"passed": True}],
            "training_expectation": {"mode": "adaptive"},
        },
        "training": {
            "settings": {"candidate_route": "adaptive"},
            "cases": [
                {
                    "model_route": {
                        "phase": "training",
                        "implementation": READABLE_TRAINING_MODEL_ROUTE,
                    },
                    "program_route": {
                        "phase": "training",
                        "implementation": ADAPTIVE_TRAINING_PROGRAM_ROUTE,
                    },
                    "leaf_routes": [
                        {"implementation": route}
                        for route in sorted(REQUIRED_TRAINING_LEAF_ROUTES)
                    ],
                }
            ],
        },
        "quantization": {
            "methods": [
                {
                    "method": "native_w8",
                    "status": "passed",
                    "prefill_route": {
                        "phase": "prefill",
                        "implementation": "native-nvidia-prefill-v2[native-w8]",
                    },
                }
            ]
        },
        "fla": {
            "fla": {"commit": EXPECTED_FLA_COMMIT},
            "release_gates": {"role": "blocking", "passed": True},
            "fla_diagnostics": {
                "role": "diagnostic-non-blocking",
                "complete": True,
                "passed_strict_envelope": False,
            },
        },
        "speed": {
            "fla": {"commit": EXPECTED_FLA_COMMIT},
            "training": {"mode": "adaptive"},
        },
    }
    paths = {}
    for label, report in reports.items():
        report.update(
            {
                "schema": REPORT_SCHEMAS[label],
                "status": "passed",
                "code_sha": HARNESS_SHA,
                "wheels": wheels,
                "environment": environment,
            }
        )
        paths[label] = write_json(tmp_path / f"{label}.json", report)

    artifacts = {
        "rwkv7_hf": {"sha256": hf_sha},
        "rwkv7_kernels": {"sha256": kernel_sha},
    }
    finetune = {
        "status": "passed",
        "runs": {
            name: {
                "status": "passed",
                "artifacts": artifacts,
                "backend_routes": [
                    {
                        "event": "pre_optimizer_step",
                        "boundary": "model",
                        "selected": "reference",
                        "phase": "training",
                        "implementation": READABLE_TRAINING_MODEL_ROUTE,
                        "reason": "training preserves the readable HF layer loop",
                    },
                    {
                        "event": "pre_optimizer_step",
                        "boundary": "program",
                        "selected": "reference",
                        "implementation": ADAPTIVE_TRAINING_PROGRAM_IMPLEMENTATION,
                    },
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
                        "implementation": REFERENCE_LINEAR_IMPLEMENTATION,
                    },
                    {
                        "event": "pre_optimizer_step",
                        "boundary": "mix6",
                        "selected": "optimized",
                        "implementation": MIX6_IMPLEMENTATION,
                    },
                ],
                "kernel_route_trace": {
                    "schema": "rwkv7-kernel-route-trace-v2",
                    "requested_training_policy": "adaptive",
                    "actual_model_calls": {},
                    "actual_recurrent_calls": {MATRIX_RECURRENT_IMPLEMENTATION: 12},
                    "actual_linear_calls": {},
                    "actual_mix6_calls": {MIX6_IMPLEMENTATION: 12},
                },
            }
            for name in ("sft", "dpo", "grpo")
        },
    }
    finetune_path = write_json(tmp_path / "finetune.json", finetune)
    lm_eval = {
        "schema": "rwkv7-lm-eval-three-way-validation-v1",
        "status": "passed",
        "units": 144,
        "require_model_routes": True,
        "aggregate_metrics": {
            lane: {f"unit-{index}": {"acc,none": 0.5} for index in range(48)}
            for lane in ("reference", "optimized", "fla")
        },
        "comparison_summary": {
            "candidate_comparisons": 96,
            "metric_failures": 0,
            "prediction_mismatches": 0,
            "continuous_mismatches": 0,
            "missing_docs": 0,
        },
        "artifacts": {
            lane: {
                "wheels": wheels,
                "fla": {"commit": EXPECTED_FLA_COMMIT},
            }
            for lane in ("reference", "optimized", "fla")
        },
    }
    lm_eval_path = write_json(tmp_path / "lm-eval.json", lm_eval)
    args = Namespace(
        device="rtx-4080",
        source_sha=SOURCE_SHA,
        harness_sha=HARNESS_SHA,
        hf_wheel=hf_wheel,
        kernel_wheel=kernel_wheel,
        correctness_report=paths["correctness"],
        hf_ecosystem_report=paths["hf_ecosystem"],
        training_report=paths["training"],
        quantization_report=paths["quantization"],
        fla_report=paths["fla"],
        speed_report=paths["speed"],
        finetune_report=finetune_path,
        lm_eval_report=lm_eval_path,
        output=tmp_path / "release-validation.json",
    )
    return args, paths, hf_sha, kernel_sha


def test_device_builder_consolidates_all_gates_and_actual_routes(tmp_path: Path):
    args, _, hf_sha, kernel_sha = setup_reports(tmp_path)
    report = build(args)
    assert report["status"] == "passed"
    assert report["hf_wheel_sha256"] == hf_sha
    assert report["kernel_wheel_sha256"] == kernel_sha
    assert report["lm_eval_units"] == 144
    assert report["training_policy"] == "adaptive"
    assert report["training_backend_environment"] == FORMAL_ADAPTIVE_BACKEND_ENVIRONMENT
    assert report["actual_routes"]["prefill"]
    assert report["actual_routes"]["decode"]
    assert report["actual_routes"]["training"]
    assert report["actual_routes"]["quantization"]
    assert all(report[f"{label}_status"] == "passed" for label in PRIMARY_REPORTS)
    assert json.loads(args.output.read_text()) == report


def test_device_builder_requires_complete_non_blocking_fla_diagnostics(tmp_path: Path):
    args, paths, _, _ = setup_reports(tmp_path)
    payload = json.loads(paths["fla"].read_text())
    payload["fla_diagnostics"]["complete"] = False
    write_json(paths["fla"], payload)
    with pytest.raises(ValueError, match="complete non-blocking diagnostics"):
        build(args)


def test_device_builder_rejects_failed_primary_gate(tmp_path: Path):
    args, paths, _, _ = setup_reports(tmp_path)
    payload = json.loads(paths["quantization"].read_text())
    payload["status"] = "failed"
    write_json(paths["quantization"], payload)
    with pytest.raises(ValueError, match="quantization report did not pass"):
        build(args)


def test_device_builder_rejects_report_from_another_wheel(tmp_path: Path):
    args, paths, _, _ = setup_reports(tmp_path)
    payload = json.loads(paths["training"].read_text())
    payload["wheels"]["rwkv7_kernels"]["sha256"] = "0" * 64
    write_json(paths["training"], payload)
    with pytest.raises(ValueError, match="training report wheel identity mismatch"):
        build(args)


def test_device_builder_rejects_missing_actual_route(tmp_path: Path):
    args, paths, _, _ = setup_reports(tmp_path)
    payload = json.loads(paths["correctness"].read_text())
    payload["models"] = []
    write_json(paths["correctness"], payload)
    with pytest.raises(ValueError, match="actual prefill route evidence is missing"):
        build(args)


def test_device_builder_rejects_non_release_v100_target(tmp_path: Path):
    args, _, _, _ = setup_reports(tmp_path)
    args.device = "tesla-v100"
    with pytest.raises(ValueError, match="unexpected release device"):
        build(args)


def test_device_builder_rejects_non_adaptive_formal_environment(tmp_path: Path):
    args, paths, _, _ = setup_reports(tmp_path)
    payload = json.loads(paths["training"].read_text())
    payload["environment"]["backend_environment"]["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"
    write_json(paths["training"], payload)
    with pytest.raises(ValueError, match="formal adaptive backend environment differs"):
        build(args)


def test_device_builder_rejects_historical_primary_training_route(tmp_path: Path):
    args, paths, _, _ = setup_reports(tmp_path)
    payload = json.loads(paths["training"].read_text())
    payload["cases"][0]["leaf_routes"].append(
        {"implementation": HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE}
    )
    write_json(paths["training"], payload)
    with pytest.raises(ValueError, match="historical whole-model train-temp"):
        build(args)


def test_device_builder_rejects_finetune_without_adaptive_route_evidence(
    tmp_path: Path,
):
    args, _, _, _ = setup_reports(tmp_path)
    payload = json.loads(args.finetune_report.read_text())
    payload["runs"]["sft"]["backend_routes"] = [
        {
            "phase": "training",
            "implementation": READABLE_TRAINING_MODEL_ROUTE,
        }
    ]
    write_json(args.finetune_report, payload)
    with pytest.raises(
        ValueError, match="sft report has invalid adaptive training routes"
    ):
        build(args)


def test_device_builder_rejects_unknown_finetune_leaf_execution(tmp_path: Path):
    args, _, _, _ = setup_reports(tmp_path)
    payload = json.loads(args.finetune_report.read_text())
    payload["runs"]["grpo"]["kernel_route_trace"]["actual_linear_calls"] = {
        "unknown-training-linear": 1
    }
    write_json(args.finetune_report, payload)
    with pytest.raises(ValueError, match="unknown linear implementations"):
        build(args)


def test_device_builder_rejects_historical_finetune_execution(tmp_path: Path):
    args, _, _, _ = setup_reports(tmp_path)
    payload = json.loads(args.finetune_report.read_text())
    payload["runs"]["dpo"]["kernel_route_trace"]["actual_model_calls"] = {
        HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE: 1
    }
    write_json(args.finetune_report, payload)
    with pytest.raises(ValueError, match="whole-model diagnostic"):
        build(args)


def test_device_builder_rejects_unpinned_fla(tmp_path: Path):
    args, paths, _, _ = setup_reports(tmp_path)
    payload = json.loads(paths["fla"].read_text())
    payload["fla"]["commit"] = "c" * 40
    write_json(paths["fla"], payload)
    with pytest.raises(ValueError, match="FLA commit mismatch"):
        build(args)


def test_device_builder_rejects_training_leaves_without_compiler_identity(
    tmp_path: Path,
):
    args, paths, _, _ = setup_reports(tmp_path)
    payload = json.loads(paths["training"].read_text())
    payload["environment"]["cuda_toolkit"]["provenance"] = None
    write_json(paths["training"], payload)
    with pytest.raises(ValueError, match="CUDA toolkit identity"):
        build(args)


def test_device_builder_rejects_training_leaves_with_mismatched_compiler(
    tmp_path: Path,
):
    args, paths, _, _ = setup_reports(tmp_path)
    payload = json.loads(paths["training"].read_text())
    payload["environment"]["cuda_toolkit"]["nvcc_version"] = [
        "Cuda compilation tools, release 12.8, V12.8.93"
    ]
    write_json(paths["training"], payload)
    with pytest.raises(ValueError, match="does not match PyTorch CUDA"):
        build(args)


def test_device_builder_rejects_unbound_native_build_paths(tmp_path: Path):
    args, paths, _, _ = setup_reports(tmp_path)
    payload = json.loads(paths["training"].read_text())
    payload["environment"]["cuda_toolkit"]["torch_extensions_dir"] = None
    write_json(paths["training"], payload)
    with pytest.raises(ValueError, match="CUDA build paths are not bound"):
        build(args)


def test_device_builder_rejects_lm_eval_without_compact_metric_matrix(
    tmp_path: Path,
):
    args, _, _, _ = setup_reports(tmp_path)
    payload = json.loads(args.lm_eval_report.read_text())
    payload["aggregate_metrics"]["optimized"].pop("unit-0")
    write_json(args.lm_eval_report, payload)
    with pytest.raises(ValueError, match="aggregate metric matrix is incomplete"):
        build(args)
