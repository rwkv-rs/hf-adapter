#!/usr/bin/env python3
"""Verify and run a bounded copy of the pinned RWKV-LM train_temp recipe.

The official demo shell deletes selected checkpoints before training.  This
runner preserves the shell's model, data, optimizer, precision, and kernel
arguments while replacing that deletion with a fail-closed isolated output
directory.  ``--max-steps`` is the only training-control addition; it bounds
the acceptance slice without changing the official token-based LR schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = ROOT / "configs" / "train_temp_official_x070_12x768_b16.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_recipe(path: str | Path = DEFAULT_RECIPE) -> dict[str, Any]:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_recipe(recipe)
    return recipe


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def validate_recipe(recipe: dict[str, Any]) -> None:
    model = recipe["model"]
    dataset = recipe["dataset"]
    prepare = recipe["prepare"]
    run = recipe["run"]
    expected_reported_ffn = int((int(model["n_embd"]) * 3.5) // 32 * 32)
    if int(model["reported_dim_ffn"]) != expected_reported_ffn:
        raise ValueError(
            "reported_dim_ffn must match the pinned train.py 3.5x/32 default; "
            f"expected {expected_reported_ffn}"
        )
    expected_effective_ffn = 4 * int(model["n_embd"])
    if int(model["effective_dim_ffn"]) != expected_effective_ffn:
        raise ValueError(
            "effective_dim_ffn must match the pinned fast RWKV_CMix_x070 4x shape; "
            f"expected {expected_effective_ffn}"
        )
    if int(dataset["bin_bytes"]) != 2 * int(dataset["token_count"]):
        raise ValueError("Minipile bin_bytes must contain uint16 tokens")
    limit = int(dataset["token_count"]) // int(model["ctx_len"]) - 1
    magic_prime = int(dataset["magic_prime"])
    if magic_prime >= limit or magic_prime % 3 != 2 or not _is_prime(magic_prime):
        raise ValueError("magic_prime does not satisfy the official Minipile/T512 contract")
    for candidate in range(magic_prime + 3, limit, 3):
        if candidate % 3 == 2 and _is_prime(candidate):
            raise ValueError(f"magic_prime is not the largest valid prime below {limit}")
    if (int(prepare["micro_bsz"]), float(prepare["adam_eps"])) != (1, 1e-8):
        raise ValueError("prepare.sh contract must remain CPU B1 with adam_eps=1e-8")
    if (int(run["micro_bsz"]), float(run["adam_eps"])) != (16, 1e-18):
        raise ValueError("run.sh contract must remain GPU B16 with adam_eps=1e-18")
    if float(run["grad_clip"]) != 1.0:
        raise ValueError("run.sh contract must preserve train.py's implicit grad_clip=1.0")


def verify_checkout(checkout: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    failures: list[str] = []
    if commit != recipe["official_commit"]:
        failures.append(
            f"official checkout commit is {commit}, expected {recipe['official_commit']}"
        )
    script_rows: dict[str, Any] = {}
    for name, contract in recipe["scripts"].items():
        path = checkout / contract["path"]
        actual = sha256_file(path) if path.is_file() else None
        script_rows[name] = {"path": str(path), "sha256": actual}
        if actual != contract["sha256"]:
            failures.append(
                f"{name} script SHA256 is {actual}, expected {contract['sha256']}"
            )
    return {
        "status": "pass" if not failures else "fail",
        "commit": commit,
        "scripts": script_rows,
        "failures": failures,
    }


def verify_dataset(data_prefix: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    bin_path = data_prefix.with_suffix(".bin")
    idx_path = data_prefix.with_suffix(".idx")
    expected_bytes = int(recipe["dataset"]["bin_bytes"])
    expected_idx_bytes = int(recipe["dataset"]["idx_bytes"])
    failures: list[str] = []
    if not bin_path.is_file() or bin_path.stat().st_size != expected_bytes:
        actual = bin_path.stat().st_size if bin_path.is_file() else None
        failures.append(f"Minipile .bin bytes are {actual}, expected {expected_bytes}")
    if not idx_path.is_file() or idx_path.stat().st_size != expected_idx_bytes:
        actual = idx_path.stat().st_size if idx_path.is_file() else None
        failures.append(f"Minipile .idx bytes are {actual}, expected {expected_idx_bytes}")
    bin_sha256 = sha256_file(bin_path) if bin_path.is_file() and not failures else None
    idx_sha256 = sha256_file(idx_path) if idx_path.is_file() and not failures else None
    if bin_sha256 is not None and bin_sha256 != recipe["dataset"]["bin_sha256"]:
        failures.append("Minipile .bin SHA256 mismatch")
    if idx_sha256 is not None and idx_sha256 != recipe["dataset"]["idx_sha256"]:
        failures.append("Minipile .idx SHA256 mismatch")
    return {
        "status": "pass" if not failures else "fail",
        "data_prefix": str(data_prefix),
        "bin_bytes": bin_path.stat().st_size if bin_path.is_file() else None,
        "idx_bytes": idx_path.stat().st_size if idx_path.is_file() else None,
        "bin_sha256": bin_sha256,
        "idx_sha256": idx_sha256,
        "failures": failures,
    }


def _base_model_args(recipe: dict[str, Any], data_prefix: Path, output_dir: Path) -> list[str]:
    model = recipe["model"]
    dataset = recipe["dataset"]
    return [
        "--proj_dir", str(output_dir),
        "--data_file", str(data_prefix),
        "--data_type", str(dataset["data_type"]),
        "--vocab_size", str(model["vocab_size"]),
        "--my_testing", str(model["model_type"]),
        "--ctx_len", str(model["ctx_len"]),
        "--my_exit_tokens", str(dataset["token_count"]),
        "--magic_prime", str(dataset["magic_prime"]),
        "--n_layer", str(model["n_layer"]),
        "--n_embd", str(model["n_embd"]),
        "--head_size", str(model["head_size"]),
    ]


def build_prepare_command(
    checkout: Path, data_prefix: Path, output_dir: Path, recipe: dict[str, Any]
) -> list[str]:
    values = recipe["prepare"]
    train_py = checkout / "RWKV-v7" / "train_temp" / "train.py"
    return [
        sys.executable,
        str(train_py),
        "--wandb", "",
        *_base_model_args(recipe, data_prefix, output_dir),
        "--train_stage", str(values["train_stage"]),
        "--epoch_count", "1",
        "--epoch_begin", "0",
        "--epoch_save", "1",
        "--weight_decay", str(values["weight_decay"]),
        "--num_nodes", "1",
        "--micro_bsz", str(values["micro_bsz"]),
        "--lr_init", str(values["lr_init"]),
        "--lr_final", str(values["lr_final"]),
        "--warmup_steps", str(values["warmup_steps"]),
        "--beta1", str(values["beta1"]),
        "--beta2", str(values["beta2"]),
        "--adam_eps", str(values["adam_eps"]),
        "--accelerator", str(values["accelerator"]),
        "--devices", str(values["devices"]),
        "--precision", str(values["precision"]),
        "--strategy", str(values["strategy"]),
        "--grad_cp", str(values["grad_cp"]),
    ]


def build_run_command(
    checkout: Path,
    data_prefix: Path,
    output_dir: Path,
    recipe: dict[str, Any],
    *,
    max_steps: int,
) -> list[str]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive for a bounded acceptance run")
    values = recipe["run"]
    train_py = checkout / "RWKV-v7" / "train_temp" / "train.py"
    return [
        sys.executable,
        str(train_py),
        "--load_model", "0",
        "--wandb", "",
        *_base_model_args(recipe, data_prefix, output_dir),
        "--train_stage", str(values["train_stage"]),
        "--epoch_count", "999999",
        "--epoch_begin", "0",
        "--num_nodes", str(values["num_nodes"]),
        "--micro_bsz", str(values["micro_bsz"]),
        "--kernel", str(values["kernel"]),
        "--lr_init", str(values["lr_init"]),
        "--lr_final", str(values["lr_final"]),
        "--warmup_steps", str(values["warmup_steps"]),
        "--beta1", str(values["beta1"]),
        "--beta2", str(values["beta2"]),
        "--adam_eps", str(values["adam_eps"]),
        "--weight_decay", str(values["weight_decay"]),
        "--epoch_save", str(values["epoch_save"]),
        "--head_chunk", str(values["head_chunk"]),
        "--accelerator", str(values["accelerator"]),
        "--devices", str(values["devices"]),
        "--precision", str(values["precision"]),
        "--strategy", str(values["strategy"]),
        "--grad_cp", str(values["grad_cp"]),
        "--enable_progress_bar", "True",
        "--max_steps", str(max_steps),
    ]


def _ensure_isolated_output(output_dir: Path, *, prepare: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(output_dir.glob("rwkv-*.pth"))
    if prepare:
        if checkpoints:
            raise RuntimeError(
                "prepare output must start empty; found " + ", ".join(str(p) for p in checkpoints)
            )
        return
    expected = output_dir / "rwkv-init.pth"
    extras = [path for path in checkpoints if path != expected]
    if not expected.is_file() or extras:
        raise RuntimeError(
            "bounded run requires exactly rwkv-init.pth in the isolated output directory"
        )


def _prepend_search_paths(env: dict[str, str], name: str, paths: list[Path]) -> None:
    values = [str(path) for path in paths if path.is_dir()]
    existing = env.get(name)
    if existing:
        values.append(existing)
    if values:
        env[name] = os.pathsep.join(values)


def build_official_runtime_env(
    extension_dir: str | Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build a reproducible extension environment for conda and pip CUDA layouts."""

    env = os.environ.copy()
    python_bin = Path(sys.executable).resolve().parent
    _prepend_search_paths(env, "PATH", [python_bin])
    extension_dir = Path(extension_dir).resolve()
    env.setdefault("TORCH_EXTENSIONS_DIR", str(extension_dir))
    env.setdefault("MAX_JOBS", "2")

    include_paths: list[Path] = []
    cuda_home: Path | None = None
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME:
            cuda_home = Path(CUDA_HOME).resolve()
        else:
            environment_root = python_bin.parent
            if (environment_root / "bin" / "nvcc").is_file():
                cuda_home = environment_root
        if cuda_home is not None:
            include_paths.extend(
                [
                    cuda_home / "include",
                    cuda_home / "targets" / "x86_64-linux" / "include",
                ]
            )
            env.setdefault("CUDA_HOME", str(cuda_home))
        nvidia_packages = Path(torch.__file__).resolve().parent.parent / "nvidia"
        if nvidia_packages.is_dir():
            include_paths.extend(sorted(nvidia_packages.glob("*/include")))
    except (ImportError, OSError):
        pass

    unique_includes: list[Path] = []
    for path in include_paths:
        if path.is_dir() and path not in unique_includes:
            unique_includes.append(path)
    _prepend_search_paths(env, "CPATH", unique_includes)
    _prepend_search_paths(env, "CPLUS_INCLUDE_PATH", unique_includes)
    metadata = {
        "python_bin": str(python_bin),
        "cuda_home": str(cuda_home) if cuda_home is not None else None,
        "torch_extensions_dir": env["TORCH_EXTENSIONS_DIR"],
        "max_jobs": env["MAX_JOBS"],
        "include_paths": [str(path) for path in unique_includes],
    }
    return env, metadata


def _run_and_capture(
    command: list[str], cwd: Path, log_path: Path, *, env: dict[str, str]
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return process.wait()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("verify", "prepare", "run"))
    parser.add_argument("--recipe", default=str(DEFAULT_RECIPE))
    parser.add_argument("--official-checkout", required=True)
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--log")
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()

    recipe = load_recipe(args.recipe)
    checkout = Path(args.official_checkout).resolve()
    data_prefix = Path(args.data_prefix).resolve()
    output_dir = Path(args.output_dir).resolve()
    artifact_path = Path(args.artifact).resolve()
    checkout_report = verify_checkout(checkout, recipe)
    dataset_report = verify_dataset(data_prefix, recipe)
    result: dict[str, Any] = {
        "schema_version": 1,
        "axis": "train_temp_official_shell_recipe",
        "phase": args.phase,
        "recipe": str(Path(args.recipe).resolve()),
        "checkout": checkout_report,
        "dataset": dataset_report,
        "output_dir": str(output_dir),
    }
    if checkout_report["status"] != "pass" or dataset_report["status"] != "pass":
        result["status"] = "fail"
        write_json_atomic(artifact_path, result)
        return 2
    if args.phase == "verify":
        result["status"] = "pass"
        write_json_atomic(artifact_path, result)
        return 0

    prepare = args.phase == "prepare"
    _ensure_isolated_output(output_dir, prepare=prepare)
    command = (
        build_prepare_command(checkout, data_prefix, output_dir, recipe)
        if prepare
        else build_run_command(
            checkout, data_prefix, output_dir, recipe, max_steps=args.max_steps
        )
    )
    log_path = Path(args.log).resolve() if args.log else artifact_path.with_suffix(".log")
    runtime_env, runtime_metadata = build_official_runtime_env(
        log_path.parent / "torch_extensions"
    )
    started = time.time()
    exit_code = _run_and_capture(
        command,
        checkout / "RWKV-v7" / "train_temp",
        log_path,
        env=runtime_env,
    )
    result.update(
        {
            "status": "pass" if exit_code == 0 else "fail",
            "command": command,
            "bounded_max_steps": None if prepare else args.max_steps,
            "log": str(log_path),
            "build_environment": runtime_metadata,
            "exit_code": exit_code,
            "runtime_s": time.time() - started,
        }
    )
    write_json_atomic(artifact_path, result)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
