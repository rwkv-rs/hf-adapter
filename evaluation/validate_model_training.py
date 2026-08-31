#!/usr/bin/env python3
"""Validate full-model RWKV-7 training with replaceable leaf operators.

The Hugging Face ``modeling_rwkv7.py`` layer loop remains identical in the
reference and candidate lanes. Only the explicit-shift Mix6, stateless linear,
and canonical recurrent tensor boundaries may be selected differently. A
pinned FLA checkout is loaded as an independent mathematical comparison; it is
never imported by either runtime package.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
from typing import Any

# Fix pinned FLA/TorchInductor compilation to one worker.  The default
# 24-process pool was observed to hit its 300-second atexit TimeoutExpired
# after the validation JSON had already been written.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F

from common import (
    environment,
    git_revision,
    input_ids_sha256,
    model_fingerprint,
    sha256_file,
    training_case_seed,
)
from fla_common import (
    activate_fla_source,
    gradient_metrics,
    gradient_rows_passed,
    tensor_metric,
    write_json,
)
from training_metrics import (
    MODEL_GRADIENT_COSINE_MIN,
    MODEL_GRADIENT_RELATIVE_L2_MAX,
    MODEL_LOGITS_COSINE_MIN,
    MODEL_LOSS_MAX_ABS,
    adaptive_fast_domain_expected,
    classify_candidate_and_fla_reference_results,
    checkpoint_input_hash_gate,
    full_model_reference_release_envelope,
    global_gradient_metric,
    gradient_parameter_summary,
)


DTYPE = torch.bfloat16
MATRIX_RECURRENT_IMPLEMENTATION = (
    "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
)
FACTORIZED_RECURRENT_IMPLEMENTATION = (
    "native-nvidia-rwkv7-factorized-recurrent-training-v1"
)
FLATTENED_LINEAR_IMPLEMENTATION = "torch-cuda-rwkv7-flattened-linear-training-v1"
MIX6_IMPLEMENTATION = "native-nvidia-rwkv7-mix6-training-v1"
PROGRAM_IMPLEMENTATION = "native-nvidia-rwkv7-adaptive-training-program-v1"
REFERENCE_PROGRAM_IMPLEMENTATION = "torch-reference-training-program-v1"
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
        choices=("reference", "adaptive", "matrix", "factorized"),
        default="adaptive",
        help=(
            "adaptive is the formal optional-kernel program with shape-local "
            "reference fallback; reference forces the clean PyTorch baseline; "
            "matrix and factorized are recurrent-leaf diagnostics"
        ),
    )
    return parser.parse_args()


def select_lane(lane: str, *, candidate: str) -> None:
    """Select one training lane without changing the HF model structure."""

    if lane == "reference":
        os.environ["RWKV7_BACKEND"] = "reference"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"
    elif lane == "candidate":
        os.environ["RWKV7_BACKEND"] = "auto"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = (
            "auto" if candidate == "reference" else candidate
        )
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
    model = (
        RWKV7ForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=DTYPE,
        )
        .cuda()
        .train()
    )
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
        active = torch.nonzero(attention_mask[batch_idx], as_tuple=False).flatten()
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
    mix6_route = None
    model_route = None
    program_route = None
    if lane != "fla":
        from rwkv7_hf.ops_rwkv7 import (
            get_last_linear_route,
            get_last_mix6_route,
            get_last_model_route,
            get_last_recurrent_route,
            get_last_training_program_route,
        )

        recurrent_route = get_last_recurrent_route()
        linear_route = get_last_linear_route()
        mix6_route = get_last_mix6_route()
        model_route = get_last_model_route()
        program_route = get_last_training_program_route()

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
        "mix6_route": mix6_route,
        "model_route": model_route,
        "program_route": program_route,
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
    strict_named_parameter_diagnostic_passed = gradient_rows_passed(gradients, DTYPE)
    # BF16 roundoff compounds through every residual block.  The release gate
    # therefore measures the complete optimizer update as one named gradient
    # vector while retaining all per-parameter rows and their stricter result.
    # The recurrent operator itself keeps the tighter all-input leaf gate.
    release_envelope = full_model_reference_release_envelope(
        {
            "logits": logits,
            "loss": loss,
            "global_gradient": global_gradient,
        }
    )
    return {
        "passed": release_envelope["passed"],
        "reference_release_envelope": release_envelope,
        "strict_named_parameter_diagnostic_passed": (
            strict_named_parameter_diagnostic_passed
        ),
        "logits": logits,
        "loss": loss,
        "gradients": gradients,
        "global_gradient": global_gradient,
        "gradient_parameter_summary": parameter_summary,
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
    mix6 = row["mix6_route"]
    model = row["model_route"]
    program = row["program_route"]
    if candidate == "reference":
        return bool(
            recurrent
            and recurrent.get("selected") == "reference"
            and recurrent.get("implementation") == "torch-reference-v1"
            and linear
            and linear.get("selected") == "reference"
            and linear.get("implementation") == "torch-reference-linear-v1"
            and mix6
            and mix6.get("selected") == "reference"
            and mix6.get("implementation") == "torch-reference-mix6-v1"
            and model
            and model.get("selected") == "reference"
            and model.get("implementation") == "torch-reference-model-v1"
            and model.get("phase") == "training"
            and program
            and program.get("selected") == "reference"
            and program.get("implementation") == REFERENCE_PROGRAM_IMPLEMENTATION
        )
    adaptive_fast_domain = adaptive_fast_domain_expected(
        batch=batch,
        tokens=tokens,
        fully_active=padding == "none",
        initial_state_zero=True,
    )
    exact_route = candidate == "matrix" or (
        candidate == "adaptive" and not adaptive_fast_domain
    )
    if exact_route:
        expected_recurrent = MATRIX_RECURRENT_IMPLEMENTATION
        linear_passed = bool(
            linear
            and linear.get("selected") == "reference"
            and linear.get("implementation") == "torch-reference-linear-v1"
            and (
                "accelerates only the recurrent leaf" in str(linear.get("reason", ""))
                or "retains reference linears" in str(linear.get("reason", ""))
            )
        )
    else:
        expected_recurrent = FACTORIZED_RECURRENT_IMPLEMENTATION
        if batch * tokens >= CUDA_LINEAR_MIN_ROWS:
            linear_passed = bool(
                linear
                and linear.get("selected") == "optimized"
                and linear.get("implementation") == FLATTENED_LINEAR_IMPLEMENTATION
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
    # The readable model has already resolved masking and shift-state semantics
    # into the explicit ``shifted`` tensor.  Mix6 therefore has no padding or
    # 16-token recurrence-chunk restriction.
    mix6_expected = candidate in {"adaptive", "factorized"}
    if mix6_expected:
        mix6_passed = bool(
            mix6
            and mix6.get("selected") == "optimized"
            and mix6.get("implementation") == MIX6_IMPLEMENTATION
        )
    else:
        mix6_passed = bool(
            mix6
            and mix6.get("selected") == "reference"
            and mix6.get("implementation") == "torch-reference-mix6-v1"
        )
    return bool(
        recurrent
        and recurrent.get("selected") == "optimized"
        and recurrent.get("implementation") == expected_recurrent
        and linear_passed
        and mix6_passed
        and model
        and model.get("selected") == "reference"
        and model.get("implementation") == "torch-reference-model-v1"
        and model.get("phase") == "training"
        and program
        and program.get("selected")
        == (
            "optimized"
            if candidate == "adaptive" and adaptive_fast_domain
            else "reference"
        )
        and program.get("implementation") == PROGRAM_IMPLEMENTATION
    )


def compact_lane(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": float(row["loss"]),
        "gradient_parameter_count": len(row["gradients"]),
        "recurrent_route": row["recurrent_route"],
        "linear_route": row["linear_route"],
        "mix6_route": row["mix6_route"],
        "model_route": row["model_route"],
        "program_route": row["program_route"],
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
    tokens = tuple(args.tokens or (16, 17, 128))
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
                    case_seed = training_case_seed(
                        args.seed,
                        batch=batch,
                        tokens=token_count,
                        padding=padding,
                    )
                    generator = torch.Generator(device="cuda").manual_seed(case_seed)
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
                    numerical_roles = classify_candidate_and_fla_reference_results(
                        comparisons["candidate"], comparisons["fla"]
                    )
                    candidate_reference_release_gate = numerical_roles[
                        "candidate_reference_release_gate"
                    ]
                    fla_reference_diagnostic = numerical_roles[
                        "fla_reference_diagnostic"
                    ]
                    # The optional backend is the implementation under test.
                    # Pinned FLA is an external numerical/speed comparator and
                    # remains fully reported, but an FLA deviation cannot turn
                    # an otherwise valid candidate into a candidate failure.
                    passed = bool(
                        route_ok and candidate_reference_release_gate["passed"]
                    )
                    performance = None
                    if not checkpointing:
                        reference_ms = lanes["reference"]["performance"][
                            "median_milliseconds"
                        ]
                        candidate_ms = lanes["candidate"]["performance"][
                            "median_milliseconds"
                        ]
                        fla_ms = lanes["fla"]["performance"]["median_milliseconds"]
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
                        "batch": batch,
                        "tokens": token_count,
                        "padding": padding,
                        "checkpointing": checkpointing,
                        "case_seed": case_seed,
                        "input_ids_sha256": input_ids_sha256(ids),
                        "route_passed": route_ok,
                        "candidate": args.candidate,
                        "linear_leaf_expected": (
                            (
                                args.candidate == "factorized"
                                and batch * token_count >= CUDA_LINEAR_MIN_ROWS
                            )
                            or (
                                args.candidate == "adaptive"
                                and adaptive_fast_domain_expected(
                                    batch=batch,
                                    tokens=token_count,
                                    fully_active=padding == "none",
                                    initial_state_zero=True,
                                )
                            )
                        ),
                        "candidate_reference_release_gate": (
                            candidate_reference_release_gate
                        ),
                        "fla_reference_diagnostic": fla_reference_diagnostic,
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

    checkpoint_input_gate = checkpoint_input_hash_gate(
        cases,
        key_fields=("batch", "tokens", "padding"),
    )
    report = {
        "schema": "rwkv7-model-training-leaves-validation-v4",
        "status": (
            "passed" if not failures and checkpoint_input_gate["passed"] else "failed"
        ),
        "code_sha": args.code_sha or git_revision(Path(__file__).resolve().parents[1]),
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
            "case_seed_contract": (
                "order-independent by batch/tokens/padding; checkpoint modes "
                "reuse identical input IDs"
            ),
            "cuda_linear_min_flattened_rows": CUDA_LINEAR_MIN_ROWS,
            "full_model_reference_release_envelope": {
                "comparison_target": "readable-reference",
                "acceptance_basis": "fixed-full-model-reference-envelope",
                "thresholds": {
                    "logits_cosine_min": MODEL_LOGITS_COSINE_MIN,
                    "causal_loss_max_abs": MODEL_LOSS_MAX_ABS,
                    "optimizer_gradient_cosine_min": MODEL_GRADIENT_COSINE_MIN,
                    "optimizer_gradient_relative_l2_max": (
                        MODEL_GRADIENT_RELATIVE_L2_MAX
                    ),
                },
            },
        },
        "release_gate": {
            "name": "candidate-vs-readable-reference",
            "blocking": True,
            "passed": not failures and checkpoint_input_gate["passed"],
            "passed_cases": sum(int(row["passed"]) for row in cases),
            "total_cases": len(cases),
            "route_passed_cases": sum(int(row["route_passed"]) for row in cases),
            "reference_envelope_passed_cases": sum(
                int(row["candidate_reference_release_gate"]["passed"]) for row in cases
            ),
            "comparison_target": "readable-reference",
            "acceptance_basis": "fixed-full-model-reference-envelope",
        },
        "external_comparators": {
            "fla_vs_readable_reference": {
                "role": "diagnostic-non-blocking",
                "strict_envelope_passed": all(
                    row["fla_reference_diagnostic"]["passed"] for row in cases
                ),
                "passed_cases": sum(
                    int(row["fla_reference_diagnostic"]["passed"]) for row in cases
                ),
                "total_cases": len(cases),
                "failures_by_component": {
                    component: sum(
                        int(
                            not row["fla_reference_diagnostic"]["components"][component]
                        )
                        for row in cases
                    )
                    for component in (
                        "logits",
                        "causal_loss",
                        "optimizer_gradient_vector",
                    )
                },
                "pinned_revision": fla.get("commit"),
                "masked_padding_contract": "per-sample-compact-scatter",
            }
        },
        "diagnostics_complete": len(cases)
        == len(batches) * len(tokens) * len(padding_modes) * len(checkpointing_modes),
        "cases": cases,
        "checkpoint_input_hash_gate": checkpoint_input_gate,
        "failures": failures,
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
