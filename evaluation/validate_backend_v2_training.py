#!/usr/bin/env python3
"""Validate backend-v2 training or an explicit hardware fallback.

Ampere-or-newer devices run the migrated BF16 train-temp autograd path. SM70
does not implement BF16 train-temp kernels, so its release profile instead
proves that the same installed wheel takes the readable FP16 autograd path
without changing loss, logits, or any gradient. The two outcomes are recorded
as different capabilities and are never conflated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from common import environment, git_revision, model_fingerprint, sha256_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument("--tokens", action="append", type=int, default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidate-route",
        choices=("native", "reference-fallback"),
        default="native",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    return parser.parse_args()


def route(candidate: bool, candidate_route: str) -> None:
    if not candidate:
        os.environ["RWKV7_BACKEND"] = "reference"
        os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "auto"
        return
    os.environ["RWKV7_BACKEND"] = "optimized" if candidate_route == "native" else "auto"
    os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "native"


def candidate_route_passed(route_value: dict[str, Any] | None, expected: str) -> bool:
    if expected == "native":
        return bool(
            route_value
            and route_value.get("selected") == "optimized"
            and route_value.get("phase") == "training"
            and route_value.get("implementation")
            == "native-nvidia-train-temp-autograd-v2"
        )
    return bool(
        route_value
        and route_value.get("selected") == "reference"
        and route_value.get("phase") == "training"
        and route_value.get("implementation") == "torch-reference-model-v1"
        and route_value.get("reason")
    )


def tensor_metric(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate = candidate.detach().float().reshape(-1)
    reference = reference.detach().float().reshape(-1)
    delta = (candidate - reference).abs()
    reference_norm = float(reference.norm())
    candidate_norm = float(candidate.norm())
    cosine = (
        1.0
        if reference_norm == 0.0 and candidate_norm == 0.0
        else float(
            torch.nn.functional.cosine_similarity(
                candidate.unsqueeze(0), reference.unsqueeze(0)
            )
        )
    )
    return {
        "finite": bool(
            torch.isfinite(candidate).all() and torch.isfinite(reference).all()
        ),
        "cosine": cosine,
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "relative_l2": float(delta.norm()) / max(reference_norm, 1.0e-12),
    }


def run_once(
    model, ids, labels, *, candidate: bool, candidate_route: str
) -> dict[str, Any]:
    from rwkv7_hf.ops_rwkv7 import get_last_model_route

    route(candidate, candidate_route)
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(input_ids=ids, labels=labels, use_cache=False, logits_to_keep=0)
    output.loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return {
        "logits": output.logits.detach().cpu(),
        "loss": output.loss.detach().cpu(),
        "gradients": gradients,
        "route": get_last_model_route(),
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    path = args.model.expanduser().resolve()
    if args.candidate_route == "native" and args.dtype != "bf16":
        raise ValueError("native train-temp acceptance requires --dtype bf16")
    if args.candidate_route == "reference-fallback" and args.dtype != "fp16":
        raise ValueError("SM70 fallback acceptance requires --dtype fp16")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model = RWKV7ForCausalLM.from_pretrained(path, dtype=dtype).cuda().train()
    vocab = int(model.config.vocab_size)
    batches = tuple(args.batch or (1, 4))
    tokens = tuple(args.tokens or (16, 128))
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    cases = []
    failures = []
    for checkpointing in (False, True):
        if checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        for batch in batches:
            for sequence in tokens:
                if sequence % 16:
                    raise ValueError(
                        "train_temp sequence lengths must be divisible by 16"
                    )
                ids = torch.randint(
                    0,
                    vocab,
                    (batch, sequence),
                    generator=generator,
                    device="cuda",
                )
                labels = ids.clone()
                labels[0, sequence // 2] = -100
                reference = run_once(
                    model,
                    ids,
                    labels,
                    candidate=False,
                    candidate_route=args.candidate_route,
                )
                candidate = run_once(
                    model,
                    ids,
                    labels,
                    candidate=True,
                    candidate_route=args.candidate_route,
                )
                logits = tensor_metric(candidate["logits"], reference["logits"])
                loss = tensor_metric(candidate["loss"], reference["loss"])
                gradient_rows = {}
                missing = sorted(
                    set(reference["gradients"]) ^ set(candidate["gradients"])
                )
                for name in sorted(
                    set(reference["gradients"]) & set(candidate["gradients"])
                ):
                    gradient_rows[name] = tensor_metric(
                        candidate["gradients"][name], reference["gradients"][name]
                    )
                gradient_passed = not missing and all(
                    row["finite"]
                    and row["cosine"] >= 0.999
                    and row["relative_l2"] <= 0.02
                    for row in gradient_rows.values()
                )
                actual_route = candidate["route"]
                passed = bool(
                    logits["finite"]
                    and logits["cosine"] >= 0.9999
                    and loss["finite"]
                    and abs(float(candidate["loss"] - reference["loss"])) <= 0.02
                    and gradient_passed
                    and candidate_route_passed(actual_route, args.candidate_route)
                )
                row = {
                    "case": (
                        f"b{batch}-t{sequence}-"
                        f"checkpointing-{str(checkpointing).lower()}"
                    ),
                    "passed": passed,
                    "logits": logits,
                    "loss": loss,
                    "loss_reference": float(reference["loss"]),
                    "loss_candidate": float(candidate["loss"]),
                    "gradients": gradient_rows,
                    "missing_gradients": missing,
                    "gradient_passed": gradient_passed,
                    "route": actual_route,
                    "reference_elapsed_seconds": reference["elapsed_seconds"],
                    "candidate_elapsed_seconds": candidate["elapsed_seconds"],
                    "speedup": reference["elapsed_seconds"]
                    / candidate["elapsed_seconds"],
                    "reference_peak_memory_bytes": reference["peak_memory_bytes"],
                    "candidate_peak_memory_bytes": candidate["peak_memory_bytes"],
                }
                cases.append(row)
                if not passed:
                    failures.append(row)

    wheels = {}
    for name, wheel in (
        ("rwkv7_hf", args.hf_wheel),
        ("rwkv7_kernels", args.kernel_wheel),
    ):
        if wheel is not None:
            wheels[name] = {"path": str(wheel), "sha256": sha256_file(wheel)}
    report = {
        "schema": "rwkv7-backend-v2-training-validation-v1",
        "status": "passed" if not failures else "failed",
        "code_sha": args.code_sha or git_revision(Path(__file__).resolve().parents[1]),
        "environment": environment(),
        "model": model_fingerprint(path),
        "wheels": wheels,
        "settings": {
            "candidate_route": args.candidate_route,
            "dtype": args.dtype,
            "batches": batches,
            "tokens": tokens,
            "seed": args.seed,
        },
        "cases": cases,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
