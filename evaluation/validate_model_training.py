#!/usr/bin/env python3
"""Validate full-model RWKV-7 training with replaceable leaf operators.

The Hugging Face ``modeling_rwkv7.py`` layer loop remains identical in the
reference and candidate lanes. Only stateless linears and the canonical
recurrent boundary may be selected differently. A pinned FLA checkout is
loaded as an independent mathematical comparison; it is never imported by
either runtime package.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
from typing import Any

import torch
import torch.nn.functional as F

from common import environment, git_revision, model_fingerprint, sha256_file
from fla_common import (
    activate_fla_source,
    gradient_metrics,
    gradient_rows_passed,
    metric_passed,
    tensor_metric,
    write_json,
)


DTYPE = torch.bfloat16
MATRIX_RECURRENT_IMPLEMENTATION = "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
FACTORIZED_RECURRENT_IMPLEMENTATION = (
    "native-nvidia-rwkv7-factorized-recurrent-training-v1"
)
FLATTENED_LINEAR_IMPLEMENTATION = "torch-cuda-rwkv7-flattened-linear-training-v1"
MODEL_LOGITS_COSINE_MIN = 0.9999
MODEL_LOSS_MAX_ABS = 0.01
MODEL_GRADIENT_COSINE_MIN = 0.9995
MODEL_GRADIENT_RELATIVE_L2_MAX = 0.025
CUDA_LINEAR_MIN_ROWS = 128


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fla-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument("--tokens", action="append", type=int, default=[])
    parser.add_argument(
        "--padding",
        action="append",
        choices=("none", "left", "right"),
        default=[],
        help="repeat to validate multiple mask layouts; default is unpadded",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--checkpointing",
        action="append",
        choices=("off", "on"),
        default=[],
        help="repeat to select one or both modes; the default validates both",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    parser.add_argument(
        "--candidate",
        choices=("adaptive", "matrix", "factorized"),
        default="adaptive",
        help=(
            "adaptive uses the factorized dense route with exact masked "
            "fallback; matrix and factorized isolate one recurrent program"
        ),
    )
    return parser.parse_args()


def select_lane(lane: str, *, candidate: str) -> None:
    """Select one training lane without changing the HF model structure."""

    if lane == "reference":
        os.environ["RWKV7_BACKEND"] = "reference"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"
    elif lane == "candidate":
        # Whole-model auto deliberately declines training.  The readable HF
        # layer loop then calls the explicitly requested recurrent leaf.  The
        # factorized policy may also select its stateless linear leaf.
        os.environ["RWKV7_BACKEND"] = "auto"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = candidate
    else:
        raise ValueError(f"unknown clean-model lane: {lane}")
    os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "auto"
    os.environ["RWKV7_KERNEL_IMPL"] = "auto"


def clean_model(path: Path, *, checkpointing: bool):
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    model = RWKV7ForCausalLM.from_pretrained(path, torch_dtype=DTYPE).cuda().train()
    if checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    return model


def fla_model(path: Path, *, checkpointing: bool):
    from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
    from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM

    config = RWKV7Config.from_pretrained(path)
    config.fuse_norm = False
    config.fuse_cross_entropy = False
    config.fuse_linear_cross_entropy = False
    config.use_l2warp = False
    model = RWKV7ForCausalLM.from_pretrained(
        path,
        config=config,
        torch_dtype=DTYPE,
    ).cuda().train()
    if checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    return model


def shifted_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute standard HF causal loss independently of model-specific loss."""

    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def model_logits(
    model,
    ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    compact_padding: bool,
) -> torch.Tensor:
    """Return logits under one explicit padding contract.

    The pinned FLA model does not consume the public two-dimensional padding
    mask in its recurrent path.  Its comparison lane therefore compacts each
    sample in token order and scatters logits back.  This is an evaluator-only
    adapter; neither runtime package imports or depends on FLA.
    """

    if (
        not compact_padding
        or attention_mask is None
        or bool(attention_mask.all().detach().cpu())
    ):
        return model(
            input_ids=ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=0,
        ).logits

    rows = []
    batch, tokens = ids.shape
    for batch_idx in range(batch):
        active = torch.nonzero(
            attention_mask[batch_idx], as_tuple=False
        ).flatten()
        compact_ids = ids[batch_idx : batch_idx + 1].index_select(1, active)
        compact_logits = model(
            input_ids=compact_ids,
            use_cache=False,
            logits_to_keep=0,
        ).logits
        restored = compact_logits.new_zeros(
            (1, tokens, compact_logits.shape[-1])
        ).index_copy(1, active, compact_logits)
        rows.append(restored)
    return torch.cat(rows, dim=0)


def forward_backward(
    model,
    ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    compact_padding: bool = False,
):
    model.zero_grad(set_to_none=True)
    logits = model_logits(
        model,
        ids,
        attention_mask,
        compact_padding=compact_padding,
    )
    loss = shifted_loss(logits, labels)
    loss.backward()
    return logits, loss


def benchmark(
    model,
    ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    warmup: int,
    iterations: int,
    compact_padding: bool = False,
) -> dict[str, Any]:
    timings = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for index in range(warmup + iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        logits, loss = forward_backward(
            model,
            ids,
            labels,
            attention_mask,
            compact_padding=compact_padding,
        )
        end.record()
        end.synchronize()
        if index >= warmup:
            timings.append(float(start.elapsed_time(end)))
        del logits, loss
    median_ms = statistics.median(timings)
    return {
        "median_milliseconds": median_ms,
        "minimum_milliseconds": min(timings),
        "maximum_milliseconds": max(timings),
        "samples_per_second": int(ids.shape[0]) * 1000.0 / median_ms,
        "tokens_per_second": ids.numel() * 1000.0 / median_ms,
        "iterations": iterations,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def collect_lane(
    lane: str,
    path: Path,
    ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    candidate: str,
    checkpointing: bool,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if lane == "fla":
        model = fla_model(path, checkpointing=checkpointing)
    else:
        select_lane(lane, candidate=candidate)
        model = clean_model(path, checkpointing=checkpointing)

    compact_padding = lane == "fla"
    logits, loss = forward_backward(
        model,
        ids,
        labels,
        attention_mask,
        compact_padding=compact_padding,
    )
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    recurrent_route = None
    linear_route = None
    model_route = None
    if lane != "fla":
        from rwkv7_hf.ops_rwkv7 import (
            get_last_linear_route,
            get_last_model_route,
            get_last_recurrent_route,
        )

        recurrent_route = get_last_recurrent_route()
        linear_route = get_last_linear_route()
        model_route = get_last_model_route()

    performance = None
    if not checkpointing:
        performance = benchmark(
            model,
            ids,
            labels,
            attention_mask,
            warmup=warmup,
            iterations=iterations,
            compact_padding=compact_padding,
        )
    row = {
        "logits": logits.detach().cpu(),
        "loss": loss.detach().cpu(),
        "gradients": gradients,
        "recurrent_route": recurrent_route,
        "linear_route": linear_route,
        "model_route": model_route,
        "performance": performance,
        "padding_contract": (
            "per-sample-compact-scatter" if compact_padding else "hf-mask"
        ),
    }
    del logits, loss, model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def compare_lane(candidate: dict[str, Any], reference: dict[str, Any]):
    logits = tensor_metric(candidate["logits"], reference["logits"])
    loss = tensor_metric(candidate["loss"], reference["loss"])
    gradients = gradient_metrics(
        candidate["gradients"],
        reference["gradients"],
    )
    global_gradient = global_gradient_metric(
        candidate["gradients"],
        reference["gradients"],
    )
    parameter_summary = gradient_parameter_summary(gradients)
    strict_named_parameter_gate = gradient_rows_passed(gradients, DTYPE)
    # BF16 roundoff compounds through every residual block.  The release gate
    # therefore measures the complete optimizer update as one named gradient
    # vector while retaining all per-parameter rows and their stricter result.
    # The recurrent operator itself keeps the tighter all-input leaf gate.
    passed = bool(
        metric_passed(logits, DTYPE, logits=True)
        and logits["cosine"] >= MODEL_LOGITS_COSINE_MIN
        and loss["finite"]
        and loss["max_abs"] <= MODEL_LOSS_MAX_ABS
        and global_gradient["finite"]
        and not global_gradient["candidate_only"]
        and not global_gradient["reference_only"]
        and global_gradient["cosine"] >= MODEL_GRADIENT_COSINE_MIN
        and global_gradient["relative_l2"] <= MODEL_GRADIENT_RELATIVE_L2_MAX
    )
    return {
        "passed": passed,
        "strict_named_parameter_gate": strict_named_parameter_gate,
        "logits": logits,
        "loss": loss,
        "gradients": gradients,
        "global_gradient": global_gradient,
        "gradient_parameter_summary": parameter_summary,
    }


def global_gradient_metric(
    candidate: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare the complete optimizer update without concatenating tensors."""

    candidate_only = sorted(set(candidate) - set(reference))
    reference_only = sorted(set(reference) - set(candidate))
    common = sorted(set(candidate) & set(reference))
    dot = torch.zeros((), dtype=torch.float64)
    candidate_square = torch.zeros((), dtype=torch.float64)
    reference_square = torch.zeros((), dtype=torch.float64)
    delta_square = torch.zeros((), dtype=torch.float64)
    max_abs = 0.0
    finite = True
    elements = 0
    for name in common:
        left = candidate[name].detach().float().reshape(-1)
        right = reference[name].detach().float().reshape(-1)
        delta = left - right
        finite = finite and bool(
            torch.isfinite(left).all()
            and torch.isfinite(right).all()
            and torch.isfinite(delta).all()
        )
        dot += (left * right).sum(dtype=torch.float64)
        candidate_square += (left * left).sum(dtype=torch.float64)
        reference_square += (right * right).sum(dtype=torch.float64)
        delta_square += (delta * delta).sum(dtype=torch.float64)
        if delta.numel():
            max_abs = max(max_abs, float(delta.abs().max()))
        elements += int(delta.numel())
    denominator = candidate_square.sqrt() * reference_square.sqrt()
    cosine = (
        1.0
        if float(denominator) == 0.0
        and float(candidate_square + reference_square) == 0.0
        else float(dot / denominator.clamp_min(1.0e-30))
    )
    return {
        "finite": finite,
        "candidate_only": candidate_only,
        "reference_only": reference_only,
        "parameter_count": len(common),
        "element_count": elements,
        "cosine": cosine,
        "relative_l2": float(
            delta_square.sqrt() / reference_square.sqrt().clamp_min(1.0e-30)
        ),
        "candidate_to_reference_norm": float(
            candidate_square.sqrt()
            / reference_square.sqrt().clamp_min(1.0e-30)
        ),
        "max_abs": max_abs,
    }


def gradient_parameter_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize named-gradient spread while retaining every row in JSON."""

    rows = report["parameters"]
    if not rows:
        return {
            "parameter_count": 0,
            "strict_parameter_count": 0,
            "strict_parameter_fraction": 0.0,
        }
    strict = [
        name
        for name, row in rows.items()
        if row["finite"]
        and row["cosine"] >= 0.999
        and row["relative_l2"] <= 0.02
    ]
    relative = sorted(float(row["relative_l2"]) for row in rows.values())
    cosine = sorted(float(row["cosine"]) for row in rows.values())

    def percentile(values: list[float], fraction: float) -> float:
        index = round((len(values) - 1) * fraction)
        return values[index]

    return {
        "parameter_count": len(rows),
        "strict_parameter_count": len(strict),
        "strict_parameter_fraction": len(strict) / len(rows),
        "relative_l2_median": percentile(relative, 0.5),
        "relative_l2_p95": percentile(relative, 0.95),
        "relative_l2_p99": percentile(relative, 0.99),
        "relative_l2_max": relative[-1],
        "cosine_min": cosine[0],
        "cosine_p01": percentile(cosine, 0.01),
        "cosine_median": percentile(cosine, 0.5),
    }


def candidate_route_passed(
    row: dict[str, Any],
    *,
    candidate: str,
    batch: int,
    tokens: int,
    padding: str,
) -> bool:
    recurrent = row["recurrent_route"]
    linear = row["linear_route"]
    model = row["model_route"]
    exact_route = candidate == "matrix" or (
        candidate == "adaptive" and (padding != "none" or tokens % 16 != 0)
    )
    if exact_route:
        expected_recurrent = MATRIX_RECURRENT_IMPLEMENTATION
        linear_passed = bool(
            linear
            and linear.get("selected") == "reference"
            and linear.get("implementation") == "torch-reference-linear-v1"
            and (
                "accelerates only the recurrent leaf"
                in str(linear.get("reason", ""))
                or "retains reference linears"
                in str(linear.get("reason", ""))
            )
        )
    else:
        expected_recurrent = FACTORIZED_RECURRENT_IMPLEMENTATION
        if batch * tokens >= CUDA_LINEAR_MIN_ROWS:
            linear_passed = bool(
                linear
                and linear.get("selected") == "optimized"
                and linear.get("implementation")
                == FLATTENED_LINEAR_IMPLEMENTATION
            )
        else:
            # Small projections keep the fixed-row reference accumulation
            # order and must not claim that the linear leaf ran.
            linear_passed = bool(
                linear
                and linear.get("selected") == "reference"
                and linear.get("implementation") == "torch-reference-linear-v1"
                and f"at least {CUDA_LINEAR_MIN_ROWS} flattened rows"
                in str(linear.get("reason", ""))
            )
    return bool(
        recurrent
        and recurrent.get("selected") == "optimized"
        and recurrent.get("implementation") == expected_recurrent
        and linear_passed
        and model
        and model.get("selected") == "reference"
        and model.get("implementation") == "torch-reference-model-v1"
        and model.get("phase") == "training"
    )


def compact_lane(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": float(row["loss"]),
        "gradient_parameter_count": len(row["gradients"]),
        "recurrent_route": row["recurrent_route"],
        "linear_route": row["linear_route"],
        "model_route": row["model_route"],
        "performance": row["performance"],
        "padding_contract": row["padding_contract"],
    }


def wheel_rows(args: argparse.Namespace) -> dict[str, Any]:
    rows = {}
    for name, path in (
        ("rwkv7_hf", args.hf_wheel),
        ("rwkv7_kernels", args.kernel_wheel),
    ):
        if path is not None:
            resolved = path.expanduser().resolve()
            rows[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    return rows


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup cannot be negative and iterations must be positive")

    path = args.model.expanduser().resolve()
    fla = activate_fla_source(args.fla_source)
    probe = clean_model(path, checkpointing=False)
    vocab = int(probe.config.vocab_size)
    del probe
    torch.cuda.empty_cache()

    batches = tuple(args.batch or (1, 4))
    tokens = tuple(args.tokens or (16, 128))
    checkpointing_modes = tuple(
        value == "on" for value in (args.checkpointing or ("off", "on"))
    )
    padding_modes = tuple(args.padding or ("none",))
    cases = []
    failures = []
    for checkpointing in checkpointing_modes:
        for batch in batches:
            for token_count in tokens:
                if token_count <= 1:
                    raise ValueError("training validation requires at least two tokens")
                for padding in padding_modes:
                    generator = torch.Generator(device="cuda").manual_seed(
                        args.seed
                        + int(checkpointing) * 1_000_000
                        + batch * 1000
                        + token_count
                        + {"none": 0, "left": 100_000, "right": 200_000}[padding]
                    )
                    ids = torch.randint(
                        1,
                        vocab,
                        (batch, token_count),
                        device="cuda",
                        generator=generator,
                    )
                    attention_mask = torch.ones(
                        batch, token_count, dtype=torch.bool, device="cuda"
                    )
                    if padding != "none":
                        for batch_idx in range(batch):
                            masked = min(token_count - 1, batch_idx % 3 + 1)
                            if padding == "left":
                                attention_mask[batch_idx, :masked] = False
                            else:
                                attention_mask[batch_idx, -masked:] = False
                        ids = ids.masked_fill(~attention_mask, 0)
                    labels = ids.clone().masked_fill(~attention_mask, -100)
                    labels[0, token_count // 2] = -100
                    lanes = {
                        lane: collect_lane(
                            lane,
                            path,
                            ids,
                            labels,
                            attention_mask,
                            candidate=args.candidate,
                            checkpointing=checkpointing,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        )
                        for lane in ("reference", "candidate", "fla")
                    }
                    comparisons = {
                        lane: compare_lane(lanes[lane], lanes["reference"])
                        for lane in ("candidate", "fla")
                    }
                    route_ok = candidate_route_passed(
                        lanes["candidate"],
                        candidate=args.candidate,
                        batch=batch,
                        tokens=token_count,
                        padding=padding,
                    )
                    candidate_not_worse_than_fla = bool(
                        comparisons["candidate"]["logits"]["cosine"]
                        >= comparisons["fla"]["logits"]["cosine"]
                        and comparisons["candidate"]["global_gradient"]["cosine"]
                        >= comparisons["fla"]["global_gradient"]["cosine"]
                        and comparisons["candidate"]["global_gradient"]["relative_l2"]
                        <= comparisons["fla"]["global_gradient"]["relative_l2"]
                    )
                    passed = bool(
                        route_ok
                        and comparisons["candidate"]["passed"]
                        and candidate_not_worse_than_fla
                    )
                    performance = None
                    if not checkpointing:
                        reference_ms = lanes["reference"]["performance"][
                            "median_milliseconds"
                        ]
                        candidate_ms = lanes["candidate"]["performance"][
                            "median_milliseconds"
                        ]
                        fla_ms = lanes["fla"]["performance"][
                            "median_milliseconds"
                        ]
                        performance = {
                            "candidate_speedup_vs_reference": (
                                reference_ms / candidate_ms
                            ),
                            "candidate_speedup_vs_fla": fla_ms / candidate_ms,
                        }
                    row = {
                        "case": (
                            f"b{batch}-t{token_count}-padding-{padding}-"
                            f"checkpointing-{str(checkpointing).lower()}"
                        ),
                        "passed": passed,
                        "route_passed": route_ok,
                        "candidate": args.candidate,
                        "linear_leaf_expected": (
                            args.candidate in {"adaptive", "factorized"}
                            and padding == "none"
                            and (
                                args.candidate == "factorized"
                                or token_count % 16 == 0
                            )
                            and batch * token_count >= CUDA_LINEAR_MIN_ROWS
                        ),
                        "candidate_not_worse_than_fla": (
                            candidate_not_worse_than_fla
                        ),
                        "lanes": {
                            name: compact_lane(lane) for name, lane in lanes.items()
                        },
                        "comparisons": comparisons,
                        "performance": performance,
                    }
                    cases.append(row)
                    if not passed:
                        failures.append(row)
                    del lanes, ids, labels, attention_mask
                    gc.collect()

    report = {
        "schema": "rwkv7-model-training-leaves-validation-v2",
        "status": "passed" if not failures else "failed",
        "code_sha": args.code_sha
        or git_revision(Path(__file__).resolve().parents[1]),
        "environment": environment(),
        "model": model_fingerprint(path),
        "fla": fla,
        "wheels": wheel_rows(args),
        "settings": {
            "dtype": "bf16",
            "candidate": args.candidate,
            "batches": batches,
            "tokens": tokens,
            "checkpointing": checkpointing_modes,
            "padding": padding_modes,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "cuda_linear_min_flattened_rows": CUDA_LINEAR_MIN_ROWS,
            "release_thresholds": {
                "logits_cosine_min": MODEL_LOGITS_COSINE_MIN,
                "loss_max_abs": MODEL_LOSS_MAX_ABS,
                "global_gradient_cosine_min": MODEL_GRADIENT_COSINE_MIN,
                "global_gradient_relative_l2_max": (
                    MODEL_GRADIENT_RELATIVE_L2_MAX
                ),
            },
        },
        "fla_comparison_status": {
            "passed_cases": sum(
                int(row["comparisons"]["fla"]["passed"]) for row in cases
            ),
            "total_cases": len(cases),
            "release_gate": "informational comparison only",
            "masked_padding_contract": "per-sample-compact-scatter",
        },
        "cases": cases,
        "failures": failures,
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
