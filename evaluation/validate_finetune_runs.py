#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_DATASETS = {
    "sft": (
        "HuggingFaceH4/ultrachat_200k",
        "8049631c405ae6576f93f445c6b8166f76f5505a",
    ),
    "dpo": (
        "HuggingFaceH4/ultrafeedback_binarized",
        "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
    ),
    "grpo": (
        "openai/gsm8k",
        "740312add88f781978c0658806c59bc2815b9866",
    ),
}
EXPECTED_TARGETS = ["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"]


def read(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run(
    path: Path,
    name: str,
    expected_step: int,
    *,
    require_backend_v2_routes: bool,
    require_training_candidate: str | None,
) -> dict:
    exit_status = read(path / "exit_status.json")
    checks = read(path / "training_checks.json")
    reload = read(path / "adapter_reload.json")
    inventory = read(path / "checkpoint_inventory.json")
    changed = read(path / "changed_parameters.json")
    config = read(path / "resolved_config.json")
    environment = read(path / "environment.json")
    model = read(path / "model_provenance.json")
    fingerprints = read(path / "dataset_fingerprints.json")
    artifacts = read(path / "artifact_provenance.json")
    routes = read(path / "backend_routes.json")
    kernel_trace_path = path / "kernel_route_trace.json"
    kernel_trace = read(kernel_trace_path) if kernel_trace_path.is_file() else {}
    metrics_path = path / "metrics.jsonl"
    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    numeric = [
        float(value)
        for row in metrics
        for value in row.values()
        if isinstance(value, (int, float))
    ]
    failures = []
    if exit_status.get("returncode") != 0:
        failures.append("nonzero exit")
    if not checks.get("finite_loss"):
        failures.append("no finite loss")
    if not checks.get("nonzero_gradient"):
        failures.append("no nonzero gradient")
    if int(checks.get("global_step", -1)) != expected_step:
        failures.append(f"global_step != {expected_step}")
    if not reload.get("close"):
        failures.append("adapter reload mismatch")
    if not changed:
        failures.append("no changed parameters")
    if not inventory:
        failures.append("empty checkpoint inventory")
    if not metrics or any(not math.isfinite(value) for value in numeric):
        failures.append("missing or non-finite metrics")
    expected_dataset, expected_revision = EXPECTED_DATASETS[name]
    expected_config = {
        "seed": 42,
        "max_length": 512,
        "train_samples": 1024,
        "eval_samples": 128,
        "max_steps": 100,
        "gradient_accumulation_steps": 1,
        "report_to": "none",
        "dataset_name": expected_dataset,
        "dataset_revision": expected_revision,
        "target_modules": EXPECTED_TARGETS,
    }
    if name == "grpo":
        expected_config["max_completion_length"] = 64
    if require_training_candidate is not None:
        expected_config.update(
            {
                "torch_dtype": "bfloat16",
                "lora_dtype": "model",
            }
        )
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        failures.append(f"non-canonical resolved config: {mismatches}")
    if (
        environment.get("transformers") != "4.56.2"
        or environment.get("trl") != "0.20.0"
    ):
        failures.append("unexpected canonical Transformers/TRL environment")
    if not config.get("source_revision") or not config.get("code_sha"):
        failures.append("missing source revision")
    if not model.get("resolved_revision") or not model.get("files", {}).get(
        "model.safetensors"
    ):
        failures.append("missing model/weight provenance")
    if not all(
        fingerprints.get(split, {}).get("selected") for split in ("train", "eval")
    ):
        failures.append("missing deterministic dataset fingerprints")
    if require_backend_v2_routes:
        if not artifacts.get("rwkv7_hf", {}).get("sha256"):
            failures.append("missing rwkv7-hf wheel SHA256")
        if not artifacts.get("rwkv7_kernels", {}).get("sha256"):
            failures.append("missing rwkv7-kernels wheel SHA256")
        adapter_fallback = any(
            row.get("event") == "pre_optimizer_step"
            and row.get("selected") == "reference"
            and row.get("phase") == "training"
            and row.get("implementation") == "torch-reference-model-v1"
            and (
                "adapter" in str(row.get("reason", "")).lower()
                or "causal-lm boundary" in str(row.get("reason", "")).lower()
            )
            for row in routes
        )
        if not adapter_fallback or not checks.get("adapter_reference_fallback"):
            failures.append("LoRA training did not prove adapter reference fallback")
        if checks.get("native_training"):
            failures.append(
                "LoRA unexpectedly bypassed adapters through native training"
            )
    if require_training_candidate is not None:
        if kernel_trace.get("schema") != "rwkv7-kernel-route-trace-v2":
            failures.append("missing versioned process-wide kernel route trace")
        if kernel_trace.get("requested_training_policy") != require_training_candidate:
            failures.append("kernel trace training policy does not match the request")
        model_reference = any(
            row.get("event") == "pre_optimizer_step"
            and row.get("boundary") == "model"
            and row.get("selected") == "reference"
            and row.get("phase") == "training"
            and row.get("implementation") == "torch-reference-model-v1"
            for row in routes
        )
        recurrent_implementations = {
            "matrix": "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1",
            "factorized": "native-nvidia-rwkv7-factorized-recurrent-training-v1",
        }
        expected_recurrents = (
            set(recurrent_implementations.values())
            if require_training_candidate == "adaptive"
            else {recurrent_implementations[require_training_candidate]}
        )
        observed_recurrents = {
            str(row.get("implementation"))
            for row in routes
            if row.get("event") == "pre_optimizer_step"
            and row.get("boundary") == "recurrent"
            and row.get("selected") == "optimized"
            and row.get("implementation") in expected_recurrents
        }
        traced_recurrents = {
            str(implementation)
            for implementation, count in kernel_trace.get(
                "actual_recurrent_calls", {}
            ).items()
            if int(count) > 0 and implementation in expected_recurrents
        }
        observed_recurrents.update(traced_recurrents)
        recurrent_candidate = bool(observed_recurrents)
        matrix_recurrent = any(
            row.get("event") == "pre_optimizer_step"
            and row.get("boundary") == "recurrent"
            and row.get("selected") == "optimized"
            and row.get("implementation") == recurrent_implementations["matrix"]
            for row in routes
        ) or int(
            kernel_trace.get("actual_recurrent_calls", {}).get(
                recurrent_implementations["matrix"], 0
            )
        ) > 0
        factorized_recurrent = any(
            row.get("event") == "pre_optimizer_step"
            and row.get("boundary") == "recurrent"
            and row.get("selected") == "optimized"
            and row.get("implementation")
            == recurrent_implementations["factorized"]
            for row in routes
        ) or int(
            kernel_trace.get("actual_recurrent_calls", {}).get(
                recurrent_implementations["factorized"], 0
            )
        ) > 0
        if not model_reference or not checks.get("model_reference_training"):
            failures.append("training did not retain the readable HF layer loop")
        recurrent_check = bool(
            (matrix_recurrent and checks.get("matrix_recurrent_training"))
            or (
                factorized_recurrent
                and checks.get("factorized_recurrent_training")
            )
        )
        if not recurrent_candidate or not recurrent_check:
            failures.append(
                "training did not execute the requested recurrent boundary"
            )
        linear_reference = any(
            row.get("event") == "pre_optimizer_step"
            and row.get("boundary") == "linear"
            and row.get("selected") == "reference"
            and row.get("implementation") == "torch-reference-linear-v1"
            for row in routes
        )
        flattened_linear = any(
            row.get("event") == "pre_optimizer_step"
            and row.get("boundary") == "linear"
            and row.get("selected") == "optimized"
            and row.get("implementation")
            == "torch-cuda-rwkv7-flattened-linear-training-v1"
            for row in routes
        ) or int(
            kernel_trace.get("actual_linear_calls", {}).get(
                "torch-cuda-rwkv7-flattened-linear-training-v1", 0
            )
        ) > 0
        if require_training_candidate == "matrix":
            if not linear_reference:
                failures.append("matrix training did not retain reference linears")
        elif require_training_candidate == "factorized":
            if not flattened_linear or not checks.get(
                "flattened_linear_training"
            ):
                failures.append(
                    "factorized CUDA training did not execute the flattened linear boundary"
                )
        else:
            if matrix_recurrent and not linear_reference:
                failures.append(
                    "adaptive matrix fallback did not retain reference linears"
                )
            if factorized_recurrent and not (
                flattened_linear or linear_reference
            ):
                failures.append(
                    "adaptive factorized route did not record a linear boundary"
                )
        if checks.get("native_training"):
            failures.append("whole-model diagnostic training unexpectedly executed")
    if name == "grpo" and not any(
        float(row.get("reward_std", 0.0)) > 0.0 for row in metrics
    ):
        failures.append("GRPO never observed nonzero within-group reward variance")
    return {
        "path": str(path),
        "global_step": checks.get("global_step"),
        "metrics": len(metrics),
        "changed_parameters": len(changed),
        "inventory_files": len(inventory),
        "adapter_reload_max_abs": reload.get("max_abs"),
        "backend_routes": routes,
        "kernel_route_trace": kernel_trace,
        "artifacts": artifacts,
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate canonical SFT/DPO/GRPO artifacts"
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--require-backend-v2-routes", action="store_true")
    parser.add_argument(
        "--require-training-candidate",
        choices=("adaptive", "matrix", "factorized"),
    )
    args = parser.parse_args()
    runs = {
        name: validate_run(
            args.result_dir / name,
            name,
            100,
            require_backend_v2_routes=args.require_backend_v2_routes,
            require_training_candidate=args.require_training_candidate,
        )
        for name in ("sft", "dpo", "grpo")
    }
    resume = read(args.result_dir / "sft-resume" / "resume_check.json")
    resume_exit = read(args.result_dir / "sft-resume" / "exit_status.json")
    wandb = read(args.result_dir / "sft-wandb-offline" / "wandb.json")
    wandb_exit = read(args.result_dir / "sft-wandb-offline" / "exit_status.json")
    ancillary = {
        "resume": {
            "passed": bool(resume.get("advanced"))
            and resume_exit.get("returncode") == 0,
            **resume,
        },
        "wandb_offline": {
            "passed": wandb_exit.get("returncode") == 0 and bool(wandb.get("enabled")),
            **wandb,
        },
    }
    passed = all(row["status"] == "passed" for row in runs.values()) and all(
        row["passed"] for row in ancillary.values()
    )
    report = {
        "status": "passed" if passed else "failed",
        "runs": runs,
        "ancillary": ancillary,
    }
    (args.result_dir / "validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
