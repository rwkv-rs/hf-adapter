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


def validate_run(path: Path, name: str, expected_step: int) -> dict:
    exit_status = read(path / "exit_status.json")
    checks = read(path / "training_checks.json")
    reload = read(path / "adapter_reload.json")
    inventory = read(path / "checkpoint_inventory.json")
    changed = read(path / "changed_parameters.json")
    config = read(path / "resolved_config.json")
    environment = read(path / "environment.json")
    model = read(path / "model_provenance.json")
    fingerprints = read(path / "dataset_fingerprints.json")
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
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        failures.append(f"non-canonical resolved config: {mismatches}")
    if environment.get("transformers") != "4.56.2" or environment.get("trl") != "0.20.0":
        failures.append("unexpected canonical Transformers/TRL environment")
    if not config.get("source_revision") or not config.get("code_sha"):
        failures.append("missing source revision")
    if not model.get("resolved_revision") or not model.get("files", {}).get("model.safetensors"):
        failures.append("missing model/weight provenance")
    if not all(fingerprints.get(split, {}).get("selected") for split in ("train", "eval")):
        failures.append("missing deterministic dataset fingerprints")
    return {
        "path": str(path),
        "global_step": checks.get("global_step"),
        "metrics": len(metrics),
        "changed_parameters": len(changed),
        "inventory_files": len(inventory),
        "adapter_reload_max_abs": reload.get("max_abs"),
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical SFT/DPO/GRPO artifacts")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = {
        name: validate_run(args.result_dir / name, name, 100)
        for name in ("sft", "dpo", "grpo")
    }
    resume = read(args.result_dir / "sft-resume" / "resume_check.json")
    resume_exit = read(args.result_dir / "sft-resume" / "exit_status.json")
    wandb = read(args.result_dir / "sft-wandb-offline" / "wandb.json")
    wandb_exit = read(args.result_dir / "sft-wandb-offline" / "exit_status.json")
    ancillary = {
        "resume": {
            "passed": bool(resume.get("advanced")) and resume_exit.get("returncode") == 0,
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
