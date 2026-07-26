#!/usr/bin/env python3
"""Validate the readable HF training loop with adaptive optional tensor leaves.

The model structure always stays in ``modeling_rwkv7.py``.  The candidate lane
uses the public API-v4 training program and replaces only certified recurrent,
linear, and Mix6 tensor boundaries.  Unsupported shapes fail closed to the
same readable PyTorch operations and remain part of the formal candidate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from common import (
    environment,
    git_revision,
    input_ids_sha256,
    model_fingerprint,
    sha256_file,
    training_case_seed,
)
from training_metrics import (
    MODEL_GRADIENT_COSINE_MIN,
    MODEL_GRADIENT_RELATIVE_L2_MAX,
    MODEL_LOGITS_COSINE_MIN,
    MODEL_LOSS_MAX_ABS,
    NAMED_GRADIENT_COSINE_MIN_DIAGNOSTIC,
    NAMED_GRADIENT_RELATIVE_L2_MAX_DIAGNOSTIC,
    adaptive_fast_domain_expected,
    checkpoint_input_hash_gate,
    global_gradient_metric,
    global_gradient_passed,
    gradient_parameter_summary,
)


MATRIX_RECURRENT_IMPLEMENTATION = (
    "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
)
FACTORIZED_RECURRENT_IMPLEMENTATION = (
    "native-nvidia-rwkv7-factorized-recurrent-training-v1"
)
FLATTENED_LINEAR_IMPLEMENTATION = "torch-cuda-rwkv7-flattened-linear-training-v1"
MIX6_IMPLEMENTATION = "native-nvidia-rwkv7-mix6-training-v1"
PROGRAM_IMPLEMENTATION = "native-nvidia-rwkv7-adaptive-training-program-v1"


def canonical_candidate_route(value: str) -> str:
    aliases = {
        "adaptive": "adaptive",
        "native": "adaptive",
        "reference": "reference",
        "reference-fallback": "reference",
    }
    try:
        return aliases[value]
    except KeyError as exc:  # pragma: no cover - argparse validates the CLI
        raise ValueError(f"unknown candidate route: {value}") from exc


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument("--tokens", action="append", type=int, default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidate-route",
        choices=("adaptive", "reference", "native", "reference-fallback"),
        default="adaptive",
        help=(
            "adaptive validates the formal optional-kernel training program "
            "with shape-local reference fallback; reference forces the clean "
            "PyTorch baseline. native and reference-fallback are deprecated "
            "aliases."
        ),
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    return parser.parse_args()


def route(candidate: bool, candidate_route: str) -> None:
    candidate_route = canonical_candidate_route(candidate_route)
    os.environ["RWKV7_KERNEL_IMPL"] = "auto"
    os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "auto"
    if not candidate:
        os.environ["RWKV7_BACKEND"] = "reference"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"
        return
    if candidate_route == "reference":
        # Exercise the installed v4 preflight and prove that it selects the
        # complete reference program without entering any optional leaf.
        os.environ["RWKV7_BACKEND"] = "auto"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"
        return
    os.environ["RWKV7_BACKEND"] = "auto"
    os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "adaptive"


def candidate_route_passed(
    routes: dict[str, Any] | None,
    expected: str,
    *,
    batch: int = 1,
    tokens: int = 16,
) -> bool:
    expected = canonical_candidate_route(expected)
    routes = routes or {}
    model = routes.get("model") or {}
    recurrent = routes.get("recurrent") or {}
    linear = routes.get("linear") or {}
    mix6 = routes.get("mix6") or {}
    program = routes.get("program") or {}
    if not (
        model.get("selected") == "reference"
        and model.get("phase") == "training"
        and model.get("implementation") == "torch-reference-model-v1"
    ):
        return False
    if expected == "reference":
        return bool(
            recurrent.get("selected") == "reference"
            and recurrent.get("implementation") == "torch-reference-v1"
            and linear.get("selected") == "reference"
            and linear.get("implementation") == "torch-reference-linear-v1"
            and mix6.get("selected") == "reference"
            and mix6.get("implementation") == "torch-reference-mix6-v1"
            and program.get("selected") == "reference"
            and program.get("implementation") == "torch-reference-training-program-v1"
        )

    fast_domain = adaptive_fast_domain_expected(batch=batch, tokens=tokens)
    recurrent_implementation = (
        FACTORIZED_RECURRENT_IMPLEMENTATION
        if fast_domain
        else MATRIX_RECURRENT_IMPLEMENTATION
    )
    linear_passed = (
        linear.get("selected") == "optimized"
        and linear.get("implementation") == FLATTENED_LINEAR_IMPLEMENTATION
        if fast_domain
        else linear.get("selected") == "reference"
        and linear.get("implementation") == "torch-reference-linear-v1"
    )
    return bool(
        recurrent.get("selected") == "optimized"
        and recurrent.get("implementation") == recurrent_implementation
        and linear_passed
        and mix6.get("selected") == "optimized"
        and mix6.get("implementation") == MIX6_IMPLEMENTATION
        and program.get("selected") == ("optimized" if fast_domain else "reference")
        and program.get("implementation") == PROGRAM_IMPLEMENTATION
    )


def tensor_metric(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate = candidate.detach().float().reshape(-1)
    reference = reference.detach().float().reshape(-1)
    delta = (candidate - reference).abs()
    dot = (candidate * reference).sum(dtype=torch.float64)
    reference_norm = (reference * reference).sum(dtype=torch.float64).sqrt()
    candidate_norm = (candidate * candidate).sum(dtype=torch.float64).sqrt()
    cosine = (
        1.0
        if reference_norm == 0.0 and candidate_norm == 0.0
        else float(dot / (candidate_norm * reference_norm))
    )
    delta_l2 = (delta * delta).sum(dtype=torch.float64).sqrt()
    return {
        "finite": bool(
            torch.isfinite(candidate).all() and torch.isfinite(reference).all()
        ),
        "cosine": cosine,
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "relative_l2": float(delta_l2 / reference_norm.clamp_min(1.0e-12)),
    }


def run_once(
    model, ids, labels, *, candidate: bool, candidate_route: str
) -> dict[str, Any]:
    from rwkv7_hf.ops_rwkv7 import (
        get_last_linear_route,
        get_last_mix6_route,
        get_last_model_route,
        get_last_recurrent_route,
        get_last_training_program_route,
    )

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
        "route": {
            "model": get_last_model_route(),
            "recurrent": get_last_recurrent_route(),
            "linear": get_last_linear_route(),
            "mix6": get_last_mix6_route(),
            "program": get_last_training_program_route(),
        },
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    path = args.model.expanduser().resolve()
    candidate_route = canonical_candidate_route(args.candidate_route)
    if candidate_route == "adaptive" and args.dtype != "bf16":
        raise ValueError("adaptive clean-leaf acceptance requires --dtype bf16")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model = RWKV7ForCausalLM.from_pretrained(path, torch_dtype=dtype).cuda().train()
    vocab = int(model.config.vocab_size)
    batches = tuple(args.batch or (1, 4))
    tokens = tuple(args.tokens or (16, 17, 128))
    cases = []
    failures = []
    for checkpointing in (False, True):
        if checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        for batch in batches:
            for sequence in tokens:
                case_seed = training_case_seed(
                    args.seed,
                    batch=batch,
                    tokens=sequence,
                )
                generator = torch.Generator(device="cuda").manual_seed(case_seed)
                ids = torch.randint(
                    1,
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
                    candidate_route=candidate_route,
                )
                candidate = run_once(
                    model,
                    ids,
                    labels,
                    candidate=True,
                    candidate_route=candidate_route,
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
                strict_named_parameter_gate = not missing and all(
                    row["finite"]
                    and row["cosine"] >= NAMED_GRADIENT_COSINE_MIN_DIAGNOSTIC
                    and row["relative_l2"] <= NAMED_GRADIENT_RELATIVE_L2_MAX_DIAGNOSTIC
                    for row in gradient_rows.values()
                )
                gradient_report = {
                    "candidate_only": sorted(
                        set(candidate["gradients"]) - set(reference["gradients"])
                    ),
                    "reference_only": sorted(
                        set(reference["gradients"]) - set(candidate["gradients"])
                    ),
                    "parameters": gradient_rows,
                }
                global_gradient = global_gradient_metric(
                    candidate["gradients"],
                    reference["gradients"],
                )
                parameter_summary = gradient_parameter_summary(gradient_report)
                # A BF16 full model contains many tiny parameter gradients for
                # which a strict per-tensor relative error is ill-conditioned.
                # Keep every named row and the strict result as diagnostics, but
                # gate the actual optimizer update using the same complete-vector
                # contract as the clean-loop/FLA training validator.
                gradient_passed = global_gradient_passed(global_gradient)
                actual_route = candidate["route"]
                route_passed = candidate_route_passed(
                    actual_route,
                    candidate_route,
                    batch=batch,
                    tokens=sequence,
                )
                passed = bool(
                    logits["finite"]
                    and logits["cosine"] >= MODEL_LOGITS_COSINE_MIN
                    and loss["finite"]
                    and abs(float(candidate["loss"] - reference["loss"]))
                    <= MODEL_LOSS_MAX_ABS
                    and gradient_passed
                    and route_passed
                )
                row = {
                    "case": (
                        f"b{batch}-t{sequence}-"
                        f"checkpointing-{str(checkpointing).lower()}"
                    ),
                    "passed": passed,
                    "batch": batch,
                    "tokens": sequence,
                    "checkpointing": checkpointing,
                    "case_seed": case_seed,
                    "input_ids_sha256": input_ids_sha256(ids),
                    "logits": logits,
                    "loss": loss,
                    "loss_reference": float(reference["loss"]),
                    "loss_candidate": float(candidate["loss"]),
                    "gradients": gradient_rows,
                    "missing_gradients": missing,
                    "global_gradient": global_gradient,
                    "gradient_parameter_summary": parameter_summary,
                    "strict_named_parameter_gate": strict_named_parameter_gate,
                    "strict_named_parameter_diagnostic_passed": (
                        strict_named_parameter_gate
                    ),
                    "gradient_passed": gradient_passed,
                    "route_passed": route_passed,
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
    checkpoint_input_gate = checkpoint_input_hash_gate(
        cases,
        key_fields=("batch", "tokens"),
    )
    report = {
        "schema": "rwkv7-backend-v2-training-validation-v3",
        "status": (
            "passed" if not failures and checkpoint_input_gate["passed"] else "failed"
        ),
        "code_sha": args.code_sha or git_revision(Path(__file__).resolve().parents[1]),
        "environment": environment(),
        "model": model_fingerprint(path),
        "wheels": wheels,
        "settings": {
            "candidate_route": candidate_route,
            "requested_candidate_route": args.candidate_route,
            "dtype": args.dtype,
            "batches": batches,
            "tokens": tokens,
            "seed": args.seed,
            "case_seed_contract": (
                "order-independent by batch/tokens; checkpoint modes reuse "
                "identical input IDs"
            ),
            "gradient_thresholds": {
                "acceptance_basis": "complete-optimizer-gradient-vector",
                "global_cosine_min": MODEL_GRADIENT_COSINE_MIN,
                "global_relative_l2_max": MODEL_GRADIENT_RELATIVE_L2_MAX,
                "strict_named_cosine_min_diagnostic": (
                    NAMED_GRADIENT_COSINE_MIN_DIAGNOSTIC
                ),
                "strict_named_relative_l2_max_diagnostic": (
                    NAMED_GRADIENT_RELATIVE_L2_MAX_DIAGNOSTIC
                ),
            },
            "logits_cosine_min": MODEL_LOGITS_COSINE_MIN,
            "loss_max_abs": MODEL_LOSS_MAX_ABS,
        },
        "cases": cases,
        "checkpoint_input_hash_gate": checkpoint_input_gate,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
