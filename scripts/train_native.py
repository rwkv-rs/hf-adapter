#!/usr/bin/env python3
"""Run configurable Native RWKV-7 full-parameter train_temp CUDA training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.bench_train_temp_alignment import (  # noqa: E402
    converge_hf,
    make_official_dataset_batch,
    make_official_dataset_sequence,
    write_json_atomic,
)
from scripts.run_train_temp_official_recipe import load_recipe  # noqa: E402


SCHEMA_VERSION = 1
PRESETS = ("native-default", "official-x070-12x768-b16")
PATH_FIELDS = {
    "model",
    "data_prefix",
    "sequence",
    "validation_batch",
    "output_dir",
    "checkpoint_out",
    "resume_from",
}
CONFIG_FIELDS = {
    *PATH_FIELDS,
    "preset",
    "batch_size",
    "seq_length",
    "steps",
    "learning_rate",
    "learning_rate_final",
    "schedule_total_steps",
    "warmup_steps",
    "weight_decay",
    "beta1",
    "beta2",
    "adam_eps",
    "grad_clip",
    "optimizer",
    "seed",
    "eval_interval",
    "gradient_checkpointing",
    "checkpoint_every",
    "stop_after_step",
    "checkpoint_sha256",
    "epoch",
    "validation_epoch",
    "magic_prime",
    "samples_per_epoch",
}


def _native_defaults() -> dict[str, Any]:
    return {
        "model": None,
        "data_prefix": None,
        "sequence": None,
        "validation_batch": None,
        "output_dir": None,
        "checkpoint_out": None,
        "resume_from": None,
        "batch_size": None,
        "seq_length": None,
        "steps": None,
        "learning_rate": 6.0e-4,
        "learning_rate_final": 6.0e-5,
        "schedule_total_steps": None,
        "warmup_steps": 10,
        "weight_decay": 0.001,
        "beta1": 0.9,
        "beta2": 0.99,
        "adam_eps": 1.0e-18,
        "grad_clip": 1.0,
        "optimizer": "fused_adam",
        "seed": 42,
        "eval_interval": 50,
        "gradient_checkpointing": True,
        "checkpoint_every": 0,
        "stop_after_step": 0,
        "checkpoint_sha256": None,
        "epoch": 0,
        "validation_epoch": 100,
        "magic_prime": 2_926_181,
        "samples_per_epoch": 40_320,
    }


def preset_config(name: str) -> dict[str, Any]:
    values = _native_defaults()
    if name == "native-default":
        return values
    if name != "official-x070-12x768-b16":
        raise ValueError(f"unknown preset: {name}")
    recipe = load_recipe()
    run = recipe["run"]
    model = recipe["model"]
    dataset = recipe["dataset"]
    values.update(
        {
            "batch_size": int(run["micro_bsz"]),
            "seq_length": int(model["ctx_len"]),
            "learning_rate": float(run["lr_init"]),
            "learning_rate_final": float(run["lr_final"]),
            "schedule_total_steps": 182_888,
            "warmup_steps": int(run["warmup_steps"]),
            "weight_decay": float(run["weight_decay"]),
            "beta1": float(run["beta1"]),
            "beta2": float(run["beta2"]),
            "adam_eps": float(run["adam_eps"]),
            "grad_clip": float(run["grad_clip"]),
            "gradient_checkpointing": bool(int(run["grad_cp"])),
            "magic_prime": int(dataset["magic_prime"]),
        }
    )
    return values


def load_user_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("training config must be a JSON object")
    schema_version = payload.pop("schema_version", SCHEMA_VERSION)
    if int(schema_version) != SCHEMA_VERSION:
        raise ValueError(f"unsupported training config schema: {schema_version}")
    unknown = sorted(set(payload) - CONFIG_FIELDS)
    if unknown:
        raise ValueError("unknown training config keys: " + ", ".join(unknown))
    return payload


def _shape(path: str | Path, *, ndim: int) -> tuple[int, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if not {"input_ids", "targets"}.issubset(keys):
            raise ValueError(f"{source} must contain input_ids and targets")
        input_shape = tuple(handle.get_slice("input_ids").get_shape())
        target_shape = tuple(handle.get_slice("targets").get_shape())
    if input_shape != target_shape or len(input_shape) != ndim:
        raise ValueError(
            f"{source} requires matching {ndim}D input_ids/targets; "
            f"got {input_shape} and {target_shape}"
        )
    return input_shape


def _positive(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _non_negative(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def validate_model_config(model: str | Path) -> None:
    """Reject known-incompatible local model shapes before compiling CUDA."""

    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("model config.json must contain a JSON object")
    head_dim = config.get("head_dim", config.get("head_size"))
    if head_dim is None:
        attention_width = config.get("attention_hidden_size")
        num_heads = config.get("num_heads", config.get("num_attention_heads"))
        if attention_width is not None and num_heads is not None:
            if int(attention_width) % int(num_heads):
                raise ValueError("attention_hidden_size must be divisible by num_heads")
            head_dim = int(attention_width) // int(num_heads)
    head_v_dim = config.get("head_v_dim", head_dim)
    known_dims = [int(value) for value in (head_dim, head_v_dim) if value is not None]
    if any(value != 64 for value in known_dims):
        raise ValueError(
            "train_temp CUDA currently requires K/V head dimensions of 64; "
            f"config reports {known_dims}"
        )


def validate_data_prefix(values: dict[str, Any]) -> None:
    prefix = Path(values["data_prefix"])
    bin_path = prefix.with_suffix(".bin")
    idx_path = prefix.with_suffix(".idx")
    if not bin_path.is_file() or not idx_path.is_file():
        raise FileNotFoundError("dataset requires matching .bin and .idx files")
    if bin_path.stat().st_size % 2:
        raise ValueError("dataset .bin size must be uint16-aligned")
    token_count = bin_path.stat().st_size // 2
    dataset_slots = token_count // int(values["seq_length"])
    magic_prime = int(values["magic_prime"])
    if magic_prime <= 0 or magic_prime >= dataset_slots or magic_prime % 3 != 2:
        raise ValueError(
            "magic_prime must be positive, below the dataset slot count, and congruent to 2 mod 3"
        )


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    user_config = load_user_config(args.config)
    requested_preset = args.preset or user_config.pop("preset", "native-default")
    if requested_preset not in PRESETS:
        raise ValueError(f"unknown preset: {requested_preset}")
    values = preset_config(requested_preset)
    values.update(user_config)
    for field in CONFIG_FIELDS - {"preset"}:
        if hasattr(args, field) and (value := getattr(args, field)) is not None:
            values[field] = value

    for field in PATH_FIELDS:
        if values[field] is not None:
            values[field] = str(Path(values[field]).expanduser().resolve())
    if values["model"] is None or values["output_dir"] is None:
        raise ValueError("--model and --output-dir are required")
    validate_model_config(values["model"])
    if values["data_prefix"] is not None and values["sequence"] is not None:
        raise ValueError("choose either --data-prefix or --sequence, not both")
    if values["data_prefix"] is None and values["sequence"] is None:
        raise ValueError("provide --data-prefix or --sequence")

    if values["sequence"] is not None:
        if values["validation_batch"] is None:
            raise ValueError("--sequence requires --validation-batch")
        sequence_shape = _shape(values["sequence"], ndim=3)
        validation_shape = _shape(values["validation_batch"], ndim=2)
        inferred_steps, inferred_batch, inferred_tokens = sequence_shape
        if validation_shape != (inferred_batch, inferred_tokens):
            raise ValueError(
                "validation batch shape must match training [batch, tokens]; "
                f"got {validation_shape} vs {(inferred_batch, inferred_tokens)}"
            )
        for field, inferred in (
            ("steps", inferred_steps),
            ("batch_size", inferred_batch),
            ("seq_length", inferred_tokens),
        ):
            if values[field] is None:
                values[field] = inferred
            elif int(values[field]) != int(inferred):
                raise ValueError(
                    f"configured {field}={values[field]} does not match sequence {inferred}"
                )
    else:
        values["batch_size"] = values["batch_size"] or 1
        values["seq_length"] = values["seq_length"] or 512
        if values["steps"] is None:
            raise ValueError("--data-prefix training requires --steps")

    values["steps"] = _positive(values["steps"], "steps")
    values["batch_size"] = _positive(values["batch_size"], "batch_size")
    values["seq_length"] = _positive(values["seq_length"], "seq_length")
    if values["seq_length"] % 16:
        raise ValueError("seq_length must be divisible by 16 for train_temp CUDA")
    if values["schedule_total_steps"] is None:
        values["schedule_total_steps"] = values["steps"]
    values["schedule_total_steps"] = _positive(
        values["schedule_total_steps"], "schedule_total_steps"
    )
    values["eval_interval"] = _positive(values["eval_interval"], "eval_interval")
    values["checkpoint_every"] = _non_negative(
        values["checkpoint_every"], "checkpoint_every"
    )
    values["stop_after_step"] = _non_negative(
        values["stop_after_step"], "stop_after_step"
    )
    values["epoch"] = _non_negative(values["epoch"], "epoch")
    values["validation_epoch"] = _non_negative(
        values["validation_epoch"], "validation_epoch"
    )
    values["samples_per_epoch"] = _positive(
        values["samples_per_epoch"], "samples_per_epoch"
    )
    values["magic_prime"] = _positive(values["magic_prime"], "magic_prime")
    if int(values["warmup_steps"]) < -1:
        raise ValueError("warmup_steps must be -1 or non-negative")
    if values["optimizer"] not in {"fused_adam", "torch_adamw"}:
        raise ValueError("optimizer must be fused_adam or torch_adamw")
    if not isinstance(values["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be a JSON boolean")
    for field in ("learning_rate", "learning_rate_final", "adam_eps", "grad_clip"):
        if float(values[field]) <= 0:
            raise ValueError(f"{field} must be positive")
    if float(values["weight_decay"]) < 0:
        raise ValueError("weight_decay must be non-negative")
    for field in ("beta1", "beta2"):
        if not 0 <= float(values[field]) < 1:
            raise ValueError(f"{field} must be in [0, 1)")
    if values["checkpoint_sha256"] is not None and not re.fullmatch(
        r"[0-9a-fA-F]{64}", str(values["checkpoint_sha256"])
    ):
        raise ValueError("checkpoint_sha256 must contain exactly 64 hexadecimal digits")
    if values["data_prefix"] is not None:
        validate_data_prefix(values)
    values["preset"] = requested_preset
    values["schema_version"] = SCHEMA_VERSION
    return values


def model_source_sha256(model: str | Path) -> str:
    source = Path(model)
    if not source.is_dir():
        raise ValueError(
            "a local model directory is required unless --checkpoint-sha256 is provided"
        )
    candidates = sorted(
        {
            *source.glob("*.safetensors"),
            *source.glob("*.bin"),
            *source.glob("*.index.json"),
            *source.glob("config.json"),
        },
        key=lambda path: path.name,
    )
    weight_files = [
        path for path in candidates if path.suffix in {".safetensors", ".bin"}
    ]
    if not weight_files:
        raise FileNotFoundError(f"no model weight files found under {source}")
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def planned_data_paths(values: dict[str, Any]) -> tuple[Path, Path]:
    if values["sequence"] is not None:
        return Path(values["sequence"]), Path(values["validation_batch"])

    output = Path(values["output_dir"]) / "data"
    shape = f"b{values['batch_size']}-t{values['seq_length']}"
    sequence = (
        output / f"train-{shape}-s{values['steps']}-e{values['epoch']}.safetensors"
    )
    validation = (
        output / f"validation-{shape}-e{values['validation_epoch']}.safetensors"
    )
    return sequence, validation


def prepare_data(values: dict[str, Any]) -> tuple[Path, Path]:
    sequence, validation = planned_data_paths(values)
    if values["sequence"] is not None:
        return sequence, validation

    sequence.parent.mkdir(parents=True, exist_ok=True)
    if not sequence.is_file():
        metadata = make_official_dataset_sequence(
            sequence,
            data_prefix=values["data_prefix"],
            batch_size=values["batch_size"],
            seq_len=values["seq_length"],
            steps=values["steps"],
            epoch=int(values["epoch"]),
            magic_prime=int(values["magic_prime"]),
            samples_per_epoch=int(values["samples_per_epoch"]),
        )
        write_json_atomic(sequence.with_suffix(".json"), metadata)
    if not validation.is_file():
        metadata = make_official_dataset_batch(
            validation,
            data_prefix=values["data_prefix"],
            batch_size=values["batch_size"],
            seq_len=values["seq_length"],
            epoch=int(values["validation_epoch"]),
            magic_prime=int(values["magic_prime"]),
            samples_per_epoch=int(values["samples_per_epoch"]),
        )
        write_json_atomic(validation.with_suffix(".json"), metadata)
    _shape(sequence, ndim=3)
    _shape(validation, ndim=2)
    return sequence, validation


def convergence_args(
    values: dict[str, Any],
    sequence: Path,
    validation: Path,
    *,
    calculate_model_hash: bool = True,
) -> SimpleNamespace:
    output_dir = Path(values["output_dir"])
    checkpoint_out = values["checkpoint_out"] or str(output_dir / "checkpoint.pt")
    checkpoint_sha256 = values["checkpoint_sha256"]
    if checkpoint_sha256 is None and calculate_model_hash:
        checkpoint_sha256 = model_source_sha256(values["model"])
    return SimpleNamespace(
        sequence=str(sequence),
        validation_batch=str(validation),
        output_json=str(output_dir / "result.json"),
        precision="bf16",
        device="cuda",
        seed=int(values["seed"]),
        learning_rate=float(values["learning_rate"]),
        learning_rate_final=float(values["learning_rate_final"]),
        schedule_total_steps=int(values["schedule_total_steps"]),
        warmup_steps=int(values["warmup_steps"]),
        weight_decay=float(values["weight_decay"]),
        beta1=float(values["beta1"]),
        beta2=float(values["beta2"]),
        adam_eps=float(values["adam_eps"]),
        grad_clip=float(values["grad_clip"]),
        eval_interval=int(values["eval_interval"]),
        resume_from=values["resume_from"],
        checkpoint_out=str(Path(checkpoint_out).expanduser().resolve()),
        checkpoint_every=int(values["checkpoint_every"]),
        stop_after_step=int(values["stop_after_step"]),
        optimizer=values["optimizer"],
        model=values["model"],
        checkpoint_sha256=checkpoint_sha256,
        native=True,
        train_temp_cuda=True,
        gradient_checkpointing=bool(values["gradient_checkpointing"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    values = resolve_config(args)
    output_dir = Path(values["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence, validation = planned_data_paths(values)
    runner_args = convergence_args(
        values,
        sequence,
        validation,
        calculate_model_hash=not args.dry_run,
    )
    resolved = {
        **values,
        "checkpoint_out": runner_args.checkpoint_out,
        "checkpoint_sha256": runner_args.checkpoint_sha256,
    }
    if values["sequence"] is not None:
        resolved.update(
            {
                "data_prefix": None,
                "sequence": str(sequence.resolve()),
                "validation_batch": str(validation.resolve()),
            }
        )
    write_json_atomic(output_dir / "resolved_config.json", resolved)
    if args.dry_run:
        return {
            "status": "dry_run",
            "backend": "hf_native_train_temp_cuda",
            "planned_sequence": str(sequence.resolve()),
            "planned_validation_batch": str(validation.resolve()),
            "resolved_config": resolved,
        }
    sequence, validation = prepare_data(values)
    resolved.update(
        {
            "data_prefix": None,
            "sequence": str(sequence.resolve()),
            "validation_batch": str(validation.resolve()),
        }
    )
    write_json_atomic(output_dir / "resolved_config.json", resolved)
    result = converge_hf(runner_args)
    return {
        "status": result["status"],
        "backend": result["backend"],
        "steps_completed": result["steps_completed"],
        "steps_requested": result["steps_requested"],
        "result_json": runner_args.output_json,
        "checkpoint": runner_args.checkpoint_out,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--preset", choices=PRESETS, help="built-in starting values")
    parser.add_argument("--config", help="JSON overrides; CLI flags take precedence")
    parser.add_argument(
        "--model",
        help="local HF model directory; a Hub ID also needs --checkpoint-sha256",
    )
    parser.add_argument(
        "--data-prefix",
        "--dataset",
        dest="data_prefix",
        help="MiniPile-style prefix with matching .bin and .idx files",
    )
    parser.add_argument("--sequence", help="packed [steps,batch,tokens] safetensors")
    parser.add_argument(
        "--validation-batch", help="packed [batch,tokens] validation safetensors"
    )
    parser.add_argument(
        "--output-dir", help="run config, result, data and checkpoint directory"
    )
    parser.add_argument("--checkpoint-out", help="training checkpoint destination")
    parser.add_argument("--resume-from", help="checkpoint created by this command")
    parser.add_argument("--checkpoint-sha256", help="optional source-model identity")
    parser.add_argument(
        "--batch-size",
        "--micro-bsz",
        dest="batch_size",
        type=int,
        help="micro batch size",
    )
    parser.add_argument(
        "--seq-length",
        "--ctx-len",
        dest="seq_length",
        type=int,
        help="dense context length",
    )
    parser.add_argument(
        "--steps",
        "--max-steps",
        dest="steps",
        type=int,
        help="optimizer steps to materialize and run",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr-init",
        dest="learning_rate",
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--learning-rate-final",
        "--lr-final",
        dest="learning_rate_final",
        type=float,
        help="final cosine learning rate",
    )
    parser.add_argument(
        "--schedule-total-steps", type=int, help="cosine schedule horizon"
    )
    parser.add_argument(
        "--warmup-steps", type=int, help="linear warmup steps; -1 disables"
    )
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--adam-eps", type=float)
    parser.add_argument("--grad-clip", type=float, help="global gradient norm limit")
    parser.add_argument(
        "--optimizer",
        choices=("fused_adam", "torch_adamw"),
        help="optimizer implementation",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-interval", type=int, help="validation cadence in steps")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="periodic checkpoint cadence; 0 means final only",
    )
    parser.add_argument(
        "--stop-after-step", type=int, help="stop early for a resumable partial run"
    )
    parser.add_argument("--epoch", type=int, help="training sampler epoch")
    parser.add_argument("--validation-epoch", type=int, help="held-out sampler epoch")
    parser.add_argument("--magic-prime", type=int, help="RWKV cubic sampler prime")
    parser.add_argument(
        "--samples-per-epoch", type=int, help="RWKV sampler epoch stride"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="trade compute for lower activation memory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and write config without hashing, packing data or starting CUDA",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(f"TRAIN NATIVE ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"pass", "partial", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
