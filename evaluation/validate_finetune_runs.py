#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run(path: Path, expected_step: int) -> dict:
    exit_status = read(path / "exit_status.json")
    checks = read(path / "training_checks.json")
    reload = read(path / "adapter_reload.json")
    inventory = read(path / "checkpoint_inventory.json")
    changed = read(path / "changed_parameters.json")
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
        name: validate_run(args.result_dir / name, 100)
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
