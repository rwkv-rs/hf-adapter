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
from typing import Any

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


def make_deterministic_batch(
    output: str | Path,
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> dict[str, Any]:
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than one")
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    tokens = torch.randint(
        low=0,
        high=int(vocab_size),
        size=(int(batch_size), int(seq_len) + 1),
        dtype=torch.long,
        generator=generator,
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


def _optimizer_groups_metadata(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "group_name": group["group_name"],
            "param_names": list(group["param_names"]),
            "param_count": len(group["param_names"]),
            "weight_decay": float(group["weight_decay"]),
            "my_lr_scale": float(group["my_lr_scale"]),
            "lr": float(group["lr"]),
        }
        for group in groups
    ]


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
            "snapshot_tensor_count",
            "snapshot_sha256",
            "gpu_name",
            "peak_memory_mb",
        )
    }


def _capture_training_phase(
    *,
    model,
    backend: str,
    naming: str,
    loss_fn,
    normalizer,
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
    group_metadata = _optimizer_groups_metadata(groups)
    optimizer = None
    before: dict[str, torch.Tensor] = {}
    if phase == "step":
        optimizer = torch.optim.AdamW(
            groups,
            lr=float(learning_rate),
            betas=(float(beta1), float(beta2)),
            eps=float(adam_eps),
            weight_decay=0.0,
            foreach=False,
            fused=False,
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
        "grad_clip": float(grad_clip),
        "optimizer": "torch.optim.AdamW recipe oracle" if phase == "step" else None,
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

    return _capture_training_phase(
        model=model,
        backend="official_train_temp",
        naming="official",
        loss_fn=official.l2wrap_cross_entropy,
        normalizer=normalize,
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
        source_commit=_git_commit(Path(__file__).resolve().parents[1]),
    )


def _snapshot_path(artifact_path: Path, artifact: dict[str, Any]) -> Path:
    snapshot = Path(str(artifact["snapshot_file"]))
    return snapshot if snapshot.is_absolute() else artifact_path.parent / snapshot


def compare_artifacts(
    reference_json: str | Path,
    candidate_json: str | Path,
    *,
    min_cosine: float | None = None,
    max_relative_l2: float = 0.02,
    max_loss_relative_diff: float = 0.01,
) -> dict[str, Any]:
    reference_path = Path(reference_json)
    candidate_path = Path(candidate_json)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    phase = str(reference.get("phase", ""))
    if min_cosine is None:
        min_cosine = 0.9999 if phase == "forward" else 0.999

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

    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference-json", required=True)
    compare.add_argument("--candidate-json", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--min-cosine", type=float)
    compare.add_argument("--max-relative-l2", type=float, default=0.02)
    compare.add_argument("--max-loss-relative-diff", type=float, default=0.01)

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
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
