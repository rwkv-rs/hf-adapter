#!/usr/bin/env python3
"""Process-isolated RWKV-LM train_temp versus HF alignment evidence runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv7_hf.train_temp_alignment import (
    build_train_temp_param_groups,
    compare_tensors,
    train_temp_cross_entropy,
)


SCHEMA_VERSION = 1
PROVENANCE_KEYS = ("phase", "precision", "checkpoint_sha256", "batch_sha256")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def save_safetensors_atomic(tensors: dict[str, torch.Tensor], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    cpu_tensors = {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in tensors.items()
    }
    save_file(cpu_tensors, temporary)
    os.replace(temporary, output)


def _make_token_tensor(
    *,
    generator: torch.Generator,
    prefix_shape: tuple[int, ...],
    seq_len: int,
    vocab_size: int,
    pattern: str,
    active_vocab_size: int | None,
) -> torch.Tensor:
    active_vocab = int(active_vocab_size or vocab_size)
    if active_vocab <= 1 or active_vocab > vocab_size:
        raise ValueError("active_vocab_size must be in [2, vocab_size]")
    if pattern == "random":
        return torch.randint(
            low=0,
            high=active_vocab,
            size=(*prefix_shape, int(seq_len) + 1),
            dtype=torch.long,
            generator=generator,
        )
    if pattern == "increment":
        starts = torch.randint(
            low=0,
            high=active_vocab,
            size=(*prefix_shape, 1),
            dtype=torch.long,
            generator=generator,
        )
        offsets = torch.arange(int(seq_len) + 1, dtype=torch.long)
        return (starts + offsets) % active_vocab
    raise ValueError(f"unsupported token pattern: {pattern}")


def make_deterministic_batch(
    output: str | Path,
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    pattern: str = "random",
    active_vocab_size: int | None = None,
) -> dict[str, Any]:
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than one")
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    tokens = _make_token_tensor(
        generator=generator,
        prefix_shape=(int(batch_size),),
        seq_len=seq_len,
        vocab_size=vocab_size,
        pattern=pattern,
        active_vocab_size=active_vocab_size,
    )
    output = Path(output)
    save_safetensors_atomic(
        {"input_ids": tokens[:, :-1], "targets": tokens[:, 1:]},
        output,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "axis": "train_temp_alignment_batch",
        "vocab_size": int(vocab_size),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "seed": int(seed),
        "pattern": pattern,
        "active_vocab_size": int(active_vocab_size or vocab_size),
        "content_sha256": sha256_file(output),
        "path": str(output),
    }


def make_deterministic_sequence(
    output: str | Path,
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    steps: int,
    seed: int,
    pattern: str = "random",
    active_vocab_size: int | None = None,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    tokens = _make_token_tensor(
        generator=generator,
        prefix_shape=(int(steps), int(batch_size)),
        seq_len=seq_len,
        vocab_size=vocab_size,
        pattern=pattern,
        active_vocab_size=active_vocab_size,
    )
    output = Path(output)
    save_safetensors_atomic(
        {"input_ids": tokens[..., :-1], "targets": tokens[..., 1:]},
        output,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "axis": "train_temp_alignment_sequence",
        "vocab_size": int(vocab_size),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "steps": int(steps),
        "seed": int(seed),
        "pattern": pattern,
        "active_vocab_size": int(active_vocab_size or vocab_size),
        "content_sha256": sha256_file(output),
        "path": str(output),
    }


def normalize_official_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    num_layers: int,
    prefix: str,
    translate,
) -> dict[str, torch.Tensor]:
    """Map official tensor names and layouts to converted HF snapshot keys."""

    normalized: dict[str, torch.Tensor] = {}
    for source_name, tensor in tensors.items():
        destination_name, transposed = translate(source_name, int(num_layers))
        if not destination_name:
            continue
        value = tensor.t().contiguous() if transposed else tensor.contiguous()
        preserves_mix_shape = (
            ".attn." in destination_name
            and destination_name.rsplit(".", 1)[-1]
            in {"x_r", "x_w", "x_k", "x_v", "x_a", "x_g"}
        )
        if (
            not preserves_mix_shape
            and value.ndim > 1
            and all(int(dim) == 1 for dim in value.shape[:-1])
        ):
            # The official model stores time-mix vectors as [1, 1, H], while
            # most converted HF modules expose the same parameters as [H].
            # Attention x_* parameters deliberately preserve [1, 1, H].
            value = value.reshape(value.shape[-1])
        key = prefix + destination_name
        if key in normalized:
            raise ValueError(f"duplicate normalized tensor key: {key}")
        normalized[key] = value
    return normalized


def _git_commit(path: str | Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _runtime_metadata(device: str) -> dict[str, Any]:
    gpu_name = None
    capability = None
    peak_memory_mb = None
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
        peak_memory_mb = torch.cuda.max_memory_allocated(index) / (1024**2)
    return {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_name": gpu_name,
        "compute_capability": capability,
        "peak_memory_mb": peak_memory_mb,
    }


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _optimizer_groups_metadata(
    groups: list[dict[str, Any]],
    name_normalizer: Callable[[str], str | None],
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for group in groups:
        source_names = list(group["param_names"])
        param_names = sorted(
            normalized
            for name in source_names
            if (normalized := name_normalizer(name)) is not None
        )
        metadata.append(
            {
                "group_name": group["group_name"],
                "param_names": param_names,
                "param_count": len(param_names),
                "source_param_count": len(source_names),
                "weight_decay": float(group["weight_decay"]),
                "my_lr_scale": float(group["my_lr_scale"]),
                "lr": float(group["lr"]),
            }
        )
    return metadata


def _capture_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "status",
            "backend",
            "phase",
            "precision",
            "loss",
            "runtime_s",
            "raw_grad_norm",
            "clip_returned_grad_norm",
            "post_step_loss",
            "snapshot_tensor_count",
            "snapshot_sha256",
            "gpu_name",
            "peak_memory_mb",
        )
    }


def _create_optimizer(
    groups: list[dict[str, Any]],
    *,
    optimizer_name: str,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
):
    if optimizer_name == "fused_adam":
        import deepspeed
        from deepspeed.ops.adam import FusedAdam

        optimizer = FusedAdam(
            groups,
            lr=float(learning_rate),
            betas=(float(beta1), float(beta2)),
            eps=float(adam_eps),
            weight_decay=0.0,
            bias_correction=True,
            adam_w_mode=True,
            amsgrad=False,
        )
        return optimizer, str(deepspeed.__version__)
    if optimizer_name == "torch_adamw":
        optimizer = torch.optim.AdamW(
            groups,
            lr=float(learning_rate),
            betas=(float(beta1), float(beta2)),
            eps=float(adam_eps),
            weight_decay=0.0,
            foreach=False,
            fused=False,
        )
        return optimizer, str(torch.__version__)
    raise ValueError(f"unsupported optimizer: {optimizer_name}")


def _capture_training_phase(
    *,
    model,
    backend: str,
    naming: str,
    loss_fn,
    normalizer,
    parameter_name_normalizer,
    batch_path: str | Path,
    checkpoint_sha256: str,
    output_json: str | Path,
    snapshot_path: str | Path,
    phase: str,
    precision: str,
    device: str,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
    grad_clip: float,
    optimizer_name: str,
    source_commit: str | None,
) -> dict[str, Any]:
    if phase not in {"forward", "backward", "step"}:
        raise ValueError(f"unsupported capture phase: {phase}")
    if precision != "bf16":
        raise ValueError("end-to-end train_temp capture currently requires bf16")
    _seed_everything(seed)
    batch_path = Path(batch_path)
    batch = load_file(batch_path)
    input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
    targets = batch["targets"].to(device=device, dtype=torch.long)
    if tuple(input_ids.shape) != tuple(targets.shape):
        raise ValueError("input_ids and targets must have the same shape")

    model.train()
    model.zero_grad(set_to_none=True)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters()]
    groups = build_train_temp_param_groups(
        named_parameters,
        weight_decay=float(weight_decay),
        naming=naming,
    )
    if not groups:
        raise ValueError("model has no trainable parameter groups")
    for group in groups:
        group["lr"] = float(learning_rate) * float(group["my_lr_scale"])
    group_metadata = _optimizer_groups_metadata(groups, parameter_name_normalizer)
    optimizer = None
    optimizer_version = None
    before: dict[str, torch.Tensor] = {}
    if phase == "step":
        optimizer, optimizer_version = _create_optimizer(
            groups,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            adam_eps=adam_eps,
        )
        before = {
            name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, parameter in named_parameters
            if parameter.requires_grad
        }

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = model(input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    loss = loss_fn(logits, targets)
    snapshot: dict[str, torch.Tensor] = {"logits": logits.detach()}
    raw_grad_norm = None
    clipped_grad_norm = None
    post_step_loss = None
    if phase in {"backward", "step"}:
        loss.backward()
        raw_gradient_tensors = {
            name: parameter.grad.detach()
            for name, parameter in named_parameters
            if parameter.requires_grad and parameter.grad is not None
        }
        raw_grad_norm = math.sqrt(
            sum(float(value.detach().float().pow(2).sum().item()) for value in raw_gradient_tensors.values())
        )
        snapshot.update(normalizer(raw_gradient_tensors, "grad::"))
        if phase == "step":
            clipped_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in named_parameters if parameter.requires_grad],
                    max_norm=float(grad_clip),
                ).item()
            )
            assert optimizer is not None
            optimizer.step()
            deltas = {
                name: parameter.detach().to(device="cpu", dtype=torch.float32) - before[name]
                for name, parameter in named_parameters
                if parameter.requires_grad
            }
            snapshot.update(normalizer(deltas, "delta::"))
            with torch.no_grad():
                post_outputs = model(input_ids)
                post_logits = post_outputs.logits if hasattr(post_outputs, "logits") else post_outputs
                post_step_loss = float(loss_fn(post_logits, targets).detach().float().item())
                snapshot["post_step_logits"] = post_logits.detach()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime_s = time.perf_counter() - started
    save_safetensors_atomic(snapshot, snapshot_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "axis": "train_temp_alignment_capture",
        "status": "pass",
        "backend": backend,
        "phase": phase,
        "precision": precision,
        "seed": int(seed),
        "checkpoint_sha256": str(checkpoint_sha256),
        "batch_sha256": sha256_file(batch_path),
        "batch_size": int(input_ids.shape[0]),
        "seq_len": int(input_ids.shape[1]),
        "loss": float(loss.detach().float().item()),
        "runtime_s": runtime_s,
        "raw_grad_norm": raw_grad_norm,
        "clip_returned_grad_norm": clipped_grad_norm,
        "post_step_loss": post_step_loss,
        "grad_clip": float(grad_clip),
        "optimizer": optimizer_name if phase == "step" else None,
        "optimizer_version": optimizer_version,
        "optimizer_groups": group_metadata,
        "betas": [float(beta1), float(beta2)],
        "adam_eps": float(adam_eps),
        "snapshot_file": str(Path(snapshot_path).resolve()),
        "snapshot_sha256": sha256_file(snapshot_path),
        "snapshot_tensor_count": len(snapshot),
        "source_commit": source_commit,
        **_runtime_metadata(device),
    }
    write_json_atomic(output_json, result)
    return result


def _learning_rate_at_step(
    step: int,
    *,
    learning_rate: float,
    learning_rate_final: float,
    schedule_total_steps: int,
    warmup_steps: int,
) -> float:
    if schedule_total_steps <= 0:
        lr = float(learning_rate)
    else:
        progress = (int(step) - max(0, int(warmup_steps))) / max(
            1, int(schedule_total_steps) - max(0, int(warmup_steps))
        )
        progress = max(0.0, min(1.0, progress))
        final_factor = float(learning_rate_final) / float(learning_rate)
        multiplier = (0.5 + final_factor / 2) + (0.5 - final_factor / 2) * math.cos(
            math.pi * progress
        )
        lr = float(learning_rate) * multiplier
    if warmup_steps > 0 and step < warmup_steps:
        lr *= 0.01 + 0.99 * int(step) / int(warmup_steps)
    return lr


def _run_convergence(
    *,
    model,
    backend: str,
    naming: str,
    loss_fn,
    parameter_name_normalizer,
    sequence_path: str | Path,
    validation_batch_path: str | Path,
    checkpoint_sha256: str,
    output_json: str | Path,
    precision: str,
    device: str,
    seed: int,
    learning_rate: float,
    learning_rate_final: float,
    schedule_total_steps: int,
    warmup_steps: int,
    weight_decay: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
    grad_clip: float,
    optimizer_name: str,
    eval_interval: int,
    source_commit: str | None,
) -> dict[str, Any]:
    if precision != "bf16":
        raise ValueError("end-to-end train_temp convergence currently requires bf16")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    _seed_everything(seed)
    sequence_path = Path(sequence_path)
    validation_batch_path = Path(validation_batch_path)
    sequence = load_file(sequence_path)
    train_inputs = sequence["input_ids"]
    train_targets = sequence["targets"]
    if train_inputs.ndim != 3 or tuple(train_inputs.shape) != tuple(train_targets.shape):
        raise ValueError("convergence sequence must have matching [steps, batch, tokens] tensors")
    validation = load_file(validation_batch_path)
    validation_inputs = validation["input_ids"].to(device=device, dtype=torch.long)
    validation_targets = validation["targets"].to(device=device, dtype=torch.long)

    named_parameters = [(name, parameter) for name, parameter in model.named_parameters()]
    groups = build_train_temp_param_groups(
        named_parameters,
        weight_decay=float(weight_decay),
        naming=naming,
    )
    for group in groups:
        group["lr"] = float(learning_rate) * float(group["my_lr_scale"])
    group_metadata = _optimizer_groups_metadata(groups, parameter_name_normalizer)
    optimizer, optimizer_version = _create_optimizer(
        groups,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        adam_eps=adam_eps,
    )

    def evaluate(step: int) -> dict[str, Any]:
        model.eval()
        with torch.no_grad():
            outputs = model(validation_inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = float(loss_fn(logits, validation_targets).detach().float().item())
        model.train()
        return {"step": int(step), "loss": loss, "finite": math.isfinite(loss)}

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    validation_curve = [evaluate(0)]
    train_curve: list[dict[str, Any]] = []
    status = "pass"
    failure = None
    for step in range(int(train_inputs.shape[0])):
        lr = _learning_rate_at_step(
            step,
            learning_rate=learning_rate,
            learning_rate_final=learning_rate_final,
            schedule_total_steps=schedule_total_steps,
            warmup_steps=warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group["my_lr_scale"])
        input_ids = train_inputs[step].to(device=device, dtype=torch.long)
        targets = train_targets[step].to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = loss_fn(logits, targets)
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named_parameters if parameter.requires_grad],
                max_norm=float(grad_clip),
            ).item()
        )
        optimizer.step()
        loss_value = float(loss.detach().float().item())
        finite = math.isfinite(loss_value) and math.isfinite(grad_norm)
        train_curve.append(
            {
                "step": step + 1,
                "loss": loss_value,
                "grad_norm": grad_norm,
                "lr": lr,
                "finite": finite,
            }
        )
        if not finite:
            status = "fail"
            failure = f"non-finite training value at step {step + 1}"
            break
        if (step + 1) % eval_interval == 0 or step + 1 == int(train_inputs.shape[0]):
            row = evaluate(step + 1)
            validation_curve.append(row)
            if not row["finite"]:
                status = "fail"
                failure = f"non-finite validation loss at step {step + 1}"
                break
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime_s = time.perf_counter() - started
    result = {
        "schema_version": SCHEMA_VERSION,
        "axis": "train_temp_alignment_convergence",
        "status": status,
        "failure": failure,
        "backend": backend,
        "precision": precision,
        "seed": int(seed),
        "checkpoint_sha256": str(checkpoint_sha256),
        "sequence_sha256": sha256_file(sequence_path),
        "validation_batch_sha256": sha256_file(validation_batch_path),
        "steps_requested": int(train_inputs.shape[0]),
        "steps_completed": len(train_curve),
        "batch_size": int(train_inputs.shape[1]),
        "seq_len": int(train_inputs.shape[2]),
        "learning_rate": float(learning_rate),
        "learning_rate_final": float(learning_rate_final),
        "schedule_total_steps": int(schedule_total_steps),
        "warmup_steps": int(warmup_steps),
        "grad_clip": float(grad_clip),
        "optimizer": optimizer_name,
        "optimizer_version": optimizer_version,
        "optimizer_groups": group_metadata,
        "train_curve": train_curve,
        "validation_curve": validation_curve,
        "runtime_s": runtime_s,
        "source_commit": source_commit,
        **_runtime_metadata(device),
    }
    write_json_atomic(output_json, result)
    return result


def _load_official_module(checkout: str | Path):
    train_temp = Path(checkout).resolve() / "RWKV-v7" / "train_temp"
    if not (train_temp / "src" / "model.py").is_file():
        raise FileNotFoundError(f"official train_temp model not found under {train_temp}")
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_KERNEL", "@rwkv3")
    # Current train_temp reuses module-level helper names between fused stages;
    # JIT mode binds each scripted wrapper before later definitions replace
    # those names. The production demo also relies on this route.
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ.setdefault("RWKV_HEAD_SIZE", "64")
    os.environ.setdefault("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
    os.chdir(train_temp)
    sys.path.insert(0, str(train_temp))
    return importlib.import_module("src.model")


def _load_official_config(path: str | Path) -> SimpleNamespace:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if "betas" in values:
        values["betas"] = tuple(values["betas"])
    return SimpleNamespace(**values)


def make_official_init(
    *,
    checkout: str | Path,
    config_path: str | Path,
    output: str | Path,
    metadata_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    official = _load_official_module(checkout)
    config = _load_official_config(config_path)
    _seed_everything(seed)
    model = official.RWKV(config)
    state = model.generate_init_weight()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, output)
    result = {
        "schema_version": SCHEMA_VERSION,
        "axis": "train_temp_alignment_init",
        "status": "pass",
        "seed": int(seed),
        "checkpoint": str(output.resolve()),
        "checkpoint_sha256": sha256_file(output),
        "official_commit": _git_commit(checkout),
        "config": vars(config),
    }
    write_json_atomic(metadata_path, result)
    return result


def capture_official(args) -> dict[str, Any]:
    from scripts.convert_rwkv7_to_hf import translate_name

    official = _load_official_module(args.official_checkout)
    config = _load_official_config(args.official_config)
    model = official.RWKV(config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device=args.device, dtype=torch.bfloat16)

    def normalize(tensors: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        return normalize_official_tensors(
            tensors,
            num_layers=int(config.n_layer),
            prefix=prefix,
            translate=translate_name,
        )

    def normalize_parameter_name(name: str) -> str | None:
        destination_name, _ = translate_name(name, int(config.n_layer))
        return destination_name or None

    return _capture_training_phase(
        model=model,
        backend="official_train_temp",
        naming="official",
        loss_fn=official.l2wrap_cross_entropy,
        normalizer=normalize,
        parameter_name_normalizer=normalize_parameter_name,
        batch_path=args.batch,
        checkpoint_sha256=sha256_file(args.checkpoint),
        output_json=args.output_json,
        snapshot_path=args.snapshot,
        phase=args.phase,
        precision=args.precision,
        device=args.device,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        grad_clip=args.grad_clip,
        optimizer_name=args.optimizer,
        source_commit=_git_commit(args.official_checkout),
    )


def capture_hf(args) -> dict[str, Any]:
    if args.native:
        os.environ["RWKV7_NATIVE_MODEL"] = "1"
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.config.use_cache = False

    def normalize(tensors: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        return {prefix + name: tensor for name, tensor in tensors.items()}

    return _capture_training_phase(
        model=model,
        backend="hf_native" if args.native else "hf_fla",
        naming="hf",
        loss_fn=train_temp_cross_entropy,
        normalizer=normalize,
        parameter_name_normalizer=lambda name: name,
        batch_path=args.batch,
        checkpoint_sha256=args.checkpoint_sha256,
        output_json=args.output_json,
        snapshot_path=args.snapshot,
        phase=args.phase,
        precision=args.precision,
        device=args.device,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        grad_clip=args.grad_clip,
        optimizer_name=args.optimizer,
        source_commit=_git_commit(Path(__file__).resolve().parents[1]),
    )


def converge_official(args) -> dict[str, Any]:
    from scripts.convert_rwkv7_to_hf import translate_name

    official = _load_official_module(args.official_checkout)
    config = _load_official_config(args.official_config)
    model = official.RWKV(config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device=args.device, dtype=torch.bfloat16)

    def normalize_parameter_name(name: str) -> str | None:
        destination_name, _ = translate_name(name, int(config.n_layer))
        return destination_name or None

    return _run_convergence(
        model=model,
        backend="official_train_temp",
        naming="official",
        loss_fn=official.l2wrap_cross_entropy,
        parameter_name_normalizer=normalize_parameter_name,
        sequence_path=args.sequence,
        validation_batch_path=args.validation_batch,
        checkpoint_sha256=sha256_file(args.checkpoint),
        output_json=args.output_json,
        precision=args.precision,
        device=args.device,
        seed=args.seed,
        learning_rate=args.learning_rate,
        learning_rate_final=args.learning_rate_final,
        schedule_total_steps=args.schedule_total_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        grad_clip=args.grad_clip,
        optimizer_name=args.optimizer,
        eval_interval=args.eval_interval,
        source_commit=_git_commit(args.official_checkout),
    )


def converge_hf(args) -> dict[str, Any]:
    if args.native:
        os.environ["RWKV7_NATIVE_MODEL"] = "1"
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.config.use_cache = False
    return _run_convergence(
        model=model,
        backend="hf_native" if args.native else "hf_fla",
        naming="hf",
        loss_fn=train_temp_cross_entropy,
        parameter_name_normalizer=lambda name: name,
        sequence_path=args.sequence,
        validation_batch_path=args.validation_batch,
        checkpoint_sha256=args.checkpoint_sha256,
        output_json=args.output_json,
        precision=args.precision,
        device=args.device,
        seed=args.seed,
        learning_rate=args.learning_rate,
        learning_rate_final=args.learning_rate_final,
        schedule_total_steps=args.schedule_total_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        grad_clip=args.grad_clip,
        optimizer_name=args.optimizer,
        eval_interval=args.eval_interval,
        source_commit=_git_commit(Path(__file__).resolve().parents[1]),
    )


def _snapshot_path(artifact_path: Path, artifact: dict[str, Any]) -> Path:
    snapshot = Path(str(artifact["snapshot_file"]))
    return snapshot if snapshot.is_absolute() else artifact_path.parent / snapshot


def _optimizer_group_contract(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    keys = ("group_name", "param_names", "param_count", "weight_decay", "my_lr_scale", "lr")
    return [
        {key: group.get(key) for key in keys}
        for group in artifact.get("optimizer_groups", [])
    ]


def compare_artifacts(
    reference_json: str | Path,
    candidate_json: str | Path,
    *,
    min_cosine: float | None = None,
    max_relative_l2: float | None = None,
    max_loss_relative_diff: float = 0.01,
) -> dict[str, Any]:
    reference_path = Path(reference_json)
    candidate_path = Path(candidate_json)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    phase = str(reference.get("phase", ""))
    if min_cosine is None:
        min_cosine = 0.9999 if phase == "forward" else 0.999
    if max_relative_l2 is None:
        max_relative_l2 = 0.02 if phase == "forward" else 0.025

    provenance_mismatches = [
        key for key in PROVENANCE_KEYS if reference.get(key) != candidate.get(key)
    ]
    reference_tensors = load_file(_snapshot_path(reference_path, reference))
    candidate_tensors = load_file(_snapshot_path(candidate_path, candidate))
    reference_keys = set(reference_tensors)
    candidate_keys = set(candidate_tensors)
    missing_in_reference = sorted(candidate_keys - reference_keys)
    missing_in_candidate = sorted(reference_keys - candidate_keys)
    common = sorted(reference_keys & candidate_keys)

    tensor_metrics: list[dict[str, Any]] = []
    tensor_failures: list[dict[str, Any]] = []
    for name in common:
        metrics = {"name": name, **compare_tensors(reference_tensors[name], candidate_tensors[name])}
        passed = bool(
            metrics["comparable"]
            and float(metrics["cosine"]) >= float(min_cosine)
            and float(metrics["relative_l2"]) <= float(max_relative_l2)
        )
        metrics["status"] = "pass" if passed else "fail"
        tensor_metrics.append(metrics)
        if not passed:
            tensor_failures.append(metrics)

    reference_loss = float(reference["loss"])
    candidate_loss = float(candidate["loss"])
    loss_abs_diff = abs(candidate_loss - reference_loss)
    loss_relative_diff = loss_abs_diff / max(abs(reference_loss), torch.finfo(torch.float64).eps)
    finite_loss = bool(torch.isfinite(torch.tensor([reference_loss, candidate_loss])).all().item())

    failures: list[str] = []
    optimizer_groups_match = None
    if phase == "step":
        if reference.get("optimizer") != candidate.get("optimizer"):
            failures.append("optimizer mismatch")
        optimizer_groups_match = _optimizer_group_contract(reference) == _optimizer_group_contract(
            candidate
        )
        if not optimizer_groups_match:
            failures.append("optimizer groups mismatch")
    if provenance_mismatches:
        failures.append("provenance mismatch: " + ", ".join(provenance_mismatches))
    if missing_in_reference:
        failures.append(f"candidate has {len(missing_in_reference)} unexpected tensors")
    if missing_in_candidate:
        failures.append(f"candidate is missing {len(missing_in_candidate)} tensors")
    if not common:
        failures.append("no common tensors")
    if tensor_failures:
        failures.append(f"{len(tensor_failures)} tensor gates failed")
    if not finite_loss:
        failures.append("non-finite loss")
    elif loss_relative_diff > max_loss_relative_diff:
        failures.append(
            f"loss relative difference {loss_relative_diff:.8g} exceeds {max_loss_relative_diff:.8g}"
        )

    comparable_metrics = [row for row in tensor_metrics if row["comparable"]]
    report = {
        "schema_version": SCHEMA_VERSION,
        "axis": "train_temp_alignment_compare",
        "status": "pass" if not failures else "fail",
        "phase": phase,
        "precision": reference.get("precision"),
        "reference_backend": reference.get("backend"),
        "candidate_backend": candidate.get("backend"),
        "checkpoint_sha256": reference.get("checkpoint_sha256"),
        "batch_sha256": reference.get("batch_sha256"),
        "tensor_count": len(common),
        "provenance_mismatches": provenance_mismatches,
        "missing_in_reference": missing_in_reference,
        "missing_in_candidate": missing_in_candidate,
        "reference_loss": reference_loss,
        "candidate_loss": candidate_loss,
        "loss_abs_diff": loss_abs_diff,
        "loss_relative_diff": loss_relative_diff,
        "min_cosine_target": float(min_cosine),
        "max_relative_l2_target": float(max_relative_l2),
        "max_loss_relative_diff_target": float(max_loss_relative_diff),
        "optimizer_groups_match": optimizer_groups_match,
        "reference_optimizer": reference.get("optimizer"),
        "candidate_optimizer": candidate.get("optimizer"),
        "worst_cosine": (
            min(float(row["cosine"]) for row in comparable_metrics)
            if comparable_metrics
            else None
        ),
        "max_relative_l2": (
            max(float(row["relative_l2"]) for row in comparable_metrics)
            if comparable_metrics
            else None
        ),
        "max_abs": (
            max(float(row["max_abs"]) for row in comparable_metrics)
            if comparable_metrics
            else None
        ),
        "tensor_failures": tensor_failures[:50],
        "failures": failures,
    }
    return report


def compare_convergence_artifacts(
    reference_json: str | Path,
    candidate_json: str | Path,
    *,
    max_train_auc_relative_diff: float = 0.02,
    max_final_validation_relative_diff: float = 0.02,
    max_validation_relative_diff: float = 0.03,
    max_grad_norm_auc_relative_diff: float = 0.05,
) -> dict[str, Any]:
    reference = json.loads(Path(reference_json).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_json).read_text(encoding="utf-8"))
    provenance_keys = (
        "precision",
        "checkpoint_sha256",
        "sequence_sha256",
        "validation_batch_sha256",
        "steps_requested",
        "batch_size",
        "seq_len",
        "learning_rate",
        "learning_rate_final",
        "schedule_total_steps",
        "warmup_steps",
        "grad_clip",
        "optimizer",
    )
    provenance_mismatches = [
        key for key in provenance_keys if reference.get(key) != candidate.get(key)
    ]
    optimizer_groups_match = _optimizer_group_contract(reference) == _optimizer_group_contract(
        candidate
    )
    reference_train = reference.get("train_curve", [])
    candidate_train = candidate.get("train_curve", [])
    reference_validation = reference.get("validation_curve", [])
    candidate_validation = candidate.get("validation_curve", [])
    train_steps_match = [row.get("step") for row in reference_train] == [
        row.get("step") for row in candidate_train
    ]
    validation_steps_match = [row.get("step") for row in reference_validation] == [
        row.get("step") for row in candidate_validation
    ]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    reference_train_losses = [float(row["loss"]) for row in reference_train]
    candidate_train_losses = [float(row["loss"]) for row in candidate_train]
    reference_grad_norms = [float(row["grad_norm"]) for row in reference_train]
    candidate_grad_norms = [float(row["grad_norm"]) for row in candidate_train]
    reference_validation_losses = [float(row["loss"]) for row in reference_validation]
    candidate_validation_losses = [float(row["loss"]) for row in candidate_validation]
    train_auc_reference = mean(reference_train_losses)
    train_auc_candidate = mean(candidate_train_losses)
    grad_auc_reference = mean(reference_grad_norms)
    grad_auc_candidate = mean(candidate_grad_norms)
    epsilon = torch.finfo(torch.float64).eps
    train_auc_relative_diff = abs(train_auc_candidate - train_auc_reference) / max(
        abs(train_auc_reference), epsilon
    )
    grad_norm_auc_relative_diff = abs(grad_auc_candidate - grad_auc_reference) / max(
        abs(grad_auc_reference), epsilon
    )
    validation_relative_diffs = [
        abs(candidate_loss - reference_loss) / max(abs(reference_loss), epsilon)
        for reference_loss, candidate_loss in zip(
            reference_validation_losses, candidate_validation_losses
        )
    ]
    final_validation_relative_diff = (
        validation_relative_diffs[-1] if validation_relative_diffs else float("inf")
    )
    max_observed_validation_relative_diff = (
        max(validation_relative_diffs) if validation_relative_diffs else float("inf")
    )
    finite = all(
        math.isfinite(value)
        for value in (
            reference_train_losses
            + candidate_train_losses
            + reference_grad_norms
            + candidate_grad_norms
            + reference_validation_losses
            + candidate_validation_losses
        )
    )
    failures: list[str] = []
    if reference.get("status") != "pass" or candidate.get("status") != "pass":
        failures.append("a convergence run did not pass")
    if provenance_mismatches:
        failures.append("provenance mismatch: " + ", ".join(provenance_mismatches))
    if not optimizer_groups_match:
        failures.append("optimizer groups mismatch")
    if not train_steps_match or not validation_steps_match:
        failures.append("curve steps mismatch")
    if not reference_train or not candidate_train:
        failures.append("empty training curve")
    if not reference_validation or not candidate_validation:
        failures.append("empty validation curve")
    if not finite:
        failures.append("non-finite curve value")
    if train_auc_relative_diff > max_train_auc_relative_diff:
        failures.append("train loss AUC relative difference exceeded target")
    if final_validation_relative_diff > max_final_validation_relative_diff:
        failures.append("final validation loss relative difference exceeded target")
    if max_observed_validation_relative_diff > max_validation_relative_diff:
        failures.append("validation curve relative difference exceeded target")
    if grad_norm_auc_relative_diff > max_grad_norm_auc_relative_diff:
        failures.append("gradient norm AUC relative difference exceeded target")
    return {
        "schema_version": SCHEMA_VERSION,
        "axis": "train_temp_alignment_convergence_compare",
        "status": "pass" if not failures else "fail",
        "reference_backend": reference.get("backend"),
        "candidate_backend": candidate.get("backend"),
        "steps": len(reference_train),
        "provenance_mismatches": provenance_mismatches,
        "optimizer_groups_match": optimizer_groups_match,
        "train_steps_match": train_steps_match,
        "validation_steps_match": validation_steps_match,
        "train_loss_auc_reference": train_auc_reference,
        "train_loss_auc_candidate": train_auc_candidate,
        "train_loss_auc_relative_diff": train_auc_relative_diff,
        "grad_norm_auc_reference": grad_auc_reference,
        "grad_norm_auc_candidate": grad_auc_candidate,
        "grad_norm_auc_relative_diff": grad_norm_auc_relative_diff,
        "final_validation_loss_reference": (
            reference_validation_losses[-1] if reference_validation_losses else None
        ),
        "final_validation_loss_candidate": (
            candidate_validation_losses[-1] if candidate_validation_losses else None
        ),
        "final_validation_relative_diff": final_validation_relative_diff,
        "max_validation_relative_diff": max_observed_validation_relative_diff,
        "targets": {
            "max_train_auc_relative_diff": float(max_train_auc_relative_diff),
            "max_final_validation_relative_diff": float(max_final_validation_relative_diff),
            "max_validation_relative_diff": float(max_validation_relative_diff),
            "max_grad_norm_auc_relative_diff": float(max_grad_norm_auc_relative_diff),
        },
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser("make-batch")
    batch.add_argument("--output", required=True)
    batch.add_argument("--metadata")
    batch.add_argument("--vocab-size", type=int, required=True)
    batch.add_argument("--batch-size", type=int, default=1)
    batch.add_argument("--seq-len", type=int, default=16)
    batch.add_argument("--seed", type=int, default=42)
    batch.add_argument("--pattern", choices=["random", "increment"], default="random")
    batch.add_argument("--active-vocab-size", type=int)

    sequence = subparsers.add_parser("make-sequence")
    sequence.add_argument("--output", required=True)
    sequence.add_argument("--metadata")
    sequence.add_argument("--vocab-size", type=int, required=True)
    sequence.add_argument("--batch-size", type=int, default=1)
    sequence.add_argument("--seq-len", type=int, default=16)
    sequence.add_argument("--steps", type=int, required=True)
    sequence.add_argument("--seed", type=int, required=True)
    sequence.add_argument("--pattern", choices=["random", "increment"], default="random")
    sequence.add_argument("--active-vocab-size", type=int)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference-json", required=True)
    compare.add_argument("--candidate-json", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--min-cosine", type=float)
    compare.add_argument("--max-relative-l2", type=float)
    compare.add_argument("--max-loss-relative-diff", type=float, default=0.01)

    compare_convergence = subparsers.add_parser("compare-convergence")
    compare_convergence.add_argument("--reference-json", required=True)
    compare_convergence.add_argument("--candidate-json", required=True)
    compare_convergence.add_argument("--output", required=True)
    compare_convergence.add_argument("--max-train-auc-relative-diff", type=float, default=0.02)
    compare_convergence.add_argument(
        "--max-final-validation-relative-diff", type=float, default=0.02
    )
    compare_convergence.add_argument("--max-validation-relative-diff", type=float, default=0.03)
    compare_convergence.add_argument(
        "--max-grad-norm-auc-relative-diff", type=float, default=0.05
    )

    init = subparsers.add_parser("make-official-init")
    init.add_argument("--official-checkout", required=True)
    init.add_argument("--official-config", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--metadata", required=True)
    init.add_argument("--seed", type=int, default=42)

    def add_capture_arguments(capture: argparse.ArgumentParser) -> None:
        capture.add_argument("--batch", required=True)
        capture.add_argument("--output-json", required=True)
        capture.add_argument("--snapshot", required=True)
        capture.add_argument("--phase", choices=["forward", "backward", "step"], required=True)
        capture.add_argument("--precision", choices=["bf16"], default="bf16")
        capture.add_argument("--device", default="cuda")
        capture.add_argument("--seed", type=int, default=42)
        capture.add_argument("--learning-rate", type=float, default=6.0e-4)
        capture.add_argument("--weight-decay", type=float, default=0.001)
        capture.add_argument("--beta1", type=float, default=0.9)
        capture.add_argument("--beta2", type=float, default=0.99)
        capture.add_argument("--adam-eps", type=float, default=1.0e-18)
        capture.add_argument("--grad-clip", type=float, default=1.0)
        capture.add_argument(
            "--optimizer",
            choices=["fused_adam", "torch_adamw"],
            default="fused_adam",
        )

    official = subparsers.add_parser("capture-official")
    add_capture_arguments(official)
    official.add_argument("--official-checkout", required=True)
    official.add_argument("--official-config", required=True)
    official.add_argument("--checkpoint", required=True)

    hf = subparsers.add_parser("capture-hf")
    add_capture_arguments(hf)
    hf.add_argument("--model", required=True)
    hf.add_argument("--checkpoint-sha256", required=True)
    hf.add_argument("--native", action="store_true")

    def add_convergence_arguments(convergence: argparse.ArgumentParser) -> None:
        convergence.add_argument("--sequence", required=True)
        convergence.add_argument("--validation-batch", required=True)
        convergence.add_argument("--output-json", required=True)
        convergence.add_argument("--precision", choices=["bf16"], default="bf16")
        convergence.add_argument("--device", default="cuda")
        convergence.add_argument("--seed", type=int, required=True)
        convergence.add_argument("--learning-rate", type=float, default=6.0e-4)
        convergence.add_argument("--learning-rate-final", type=float, default=1.0e-5)
        convergence.add_argument("--schedule-total-steps", type=int, default=500_000)
        convergence.add_argument("--warmup-steps", type=int, default=-1)
        convergence.add_argument("--weight-decay", type=float, default=0.001)
        convergence.add_argument("--beta1", type=float, default=0.9)
        convergence.add_argument("--beta2", type=float, default=0.99)
        convergence.add_argument("--adam-eps", type=float, default=1.0e-18)
        convergence.add_argument("--grad-clip", type=float, default=1.0)
        convergence.add_argument("--eval-interval", type=int, default=10)
        convergence.add_argument(
            "--optimizer",
            choices=["fused_adam", "torch_adamw"],
            default="fused_adam",
        )

    official_convergence = subparsers.add_parser("converge-official")
    add_convergence_arguments(official_convergence)
    official_convergence.add_argument("--official-checkout", required=True)
    official_convergence.add_argument("--official-config", required=True)
    official_convergence.add_argument("--checkpoint", required=True)

    hf_convergence = subparsers.add_parser("converge-hf")
    add_convergence_arguments(hf_convergence)
    hf_convergence.add_argument("--model", required=True)
    hf_convergence.add_argument("--checkpoint-sha256", required=True)
    hf_convergence.add_argument("--native", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "make-batch":
        metadata = make_deterministic_batch(
            args.output,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            seed=args.seed,
            pattern=args.pattern,
            active_vocab_size=args.active_vocab_size,
        )
        if args.metadata:
            write_json_atomic(args.metadata, metadata)
        print(json.dumps(metadata, ensure_ascii=False))
        return 0
    if args.command == "make-sequence":
        metadata = make_deterministic_sequence(
            args.output,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            steps=args.steps,
            seed=args.seed,
            pattern=args.pattern,
            active_vocab_size=args.active_vocab_size,
        )
        if args.metadata:
            write_json_atomic(args.metadata, metadata)
        print(json.dumps(metadata, ensure_ascii=False))
        return 0
    if args.command == "compare":
        report = compare_artifacts(
            args.reference_json,
            args.candidate_json,
            min_cosine=args.min_cosine,
            max_relative_l2=args.max_relative_l2,
            max_loss_relative_diff=args.max_loss_relative_diff,
        )
        write_json_atomic(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["status"] == "pass" else 1
    if args.command == "compare-convergence":
        report = compare_convergence_artifacts(
            args.reference_json,
            args.candidate_json,
            max_train_auc_relative_diff=args.max_train_auc_relative_diff,
            max_final_validation_relative_diff=args.max_final_validation_relative_diff,
            max_validation_relative_diff=args.max_validation_relative_diff,
            max_grad_norm_auc_relative_diff=args.max_grad_norm_auc_relative_diff,
        )
        write_json_atomic(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["status"] == "pass" else 1
    if args.command == "make-official-init":
        result = make_official_init(
            checkout=args.official_checkout,
            config_path=args.official_config,
            output=args.output,
            metadata_path=args.metadata,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "capture-official":
        result = capture_official(args)
        print(json.dumps(_capture_summary(result), ensure_ascii=False))
        return 0
    if args.command == "capture-hf":
        result = capture_hf(args)
        print(json.dumps(_capture_summary(result), ensure_ascii=False))
        return 0
    if args.command == "converge-official":
        result = converge_official(args)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "status",
                        "backend",
                        "steps_completed",
                        "runtime_s",
                        "peak_memory_mb",
                        "failure",
                    )
                },
                ensure_ascii=False,
            )
        )
        return 0 if result["status"] == "pass" else 1
    if args.command == "converge-hf":
        result = converge_hf(args)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "status",
                        "backend",
                        "steps_completed",
                        "runtime_s",
                        "peak_memory_mb",
                        "failure",
                    )
                },
                ensure_ascii=False,
            )
        )
        return 0 if result["status"] == "pass" else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
