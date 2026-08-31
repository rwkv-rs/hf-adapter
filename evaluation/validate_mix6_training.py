#!/usr/bin/env python3
"""Validate the native RWKV-7 explicit-shift Mix6 leaf and autograd policy.

The candidate lane always executes the public ``mix6-training-v1`` boundary.
The clean model supplies the explicit masked shifted tensor. Backward remains
shape-adaptive: small inputs replay the canonical
PyTorch expression, while larger inputs use the deterministic native CUDA
reduction.  This validator observes that dispatch directly, compares all six
outputs and all eight first-order gradients, proves repeatability, and checks
that ``create_graph=True`` retains a usable higher-order graph.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

from common import environment, git_revision, sha256_file


DTYPE = torch.bfloat16
OUTPUT_NAMES = ("r", "w", "k", "v", "a", "g")
INPUT_NAMES = ("x", "shifted", "x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
COSINE_MIN = 0.9999
MINIMUM_REPEATS = 5
CANONICAL_STRATEGY = "canonical-autograd-replay-v1"
NATIVE_STRATEGY = "native-deterministic-explicit-shift-cuda-backward-v1"
OPERATOR_NAMESPACE = "rwkv7_tmix_mix6_shifted_bf16_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument("--tokens", action="append", type=int, default=[])
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--repeats", type=int, default=MINIMUM_REPEATS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    parser.add_argument(
        "--verbose-build",
        action="store_true",
        help="show torch C++ extension build output",
    )
    return parser.parse_args()


def tensor_metric(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Compare one complete tensor using stable FP64 scalar reductions."""

    candidate_tensor = candidate.detach().float()
    reference_tensor = reference.detach().float()
    if candidate_tensor.shape != reference_tensor.shape:
        return {
            "shape_match": False,
            "candidate_shape": list(candidate_tensor.shape),
            "reference_shape": list(reference_tensor.shape),
            "finite": False,
            "cosine": 0.0,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "relative_l2": float("inf"),
            "bitwise_equal": False,
        }

    candidate_flat = candidate_tensor.reshape(-1)
    reference_flat = reference_tensor.reshape(-1)
    delta = candidate_flat - reference_flat
    dot = (candidate_flat * reference_flat).sum(dtype=torch.float64)
    candidate_norm = (candidate_flat * candidate_flat).sum(dtype=torch.float64).sqrt()
    reference_norm = (reference_flat * reference_flat).sum(dtype=torch.float64).sqrt()
    denominator = candidate_norm * reference_norm
    if float(denominator) == 0.0:
        cosine = 1.0 if torch.equal(candidate_flat, reference_flat) else 0.0
    else:
        cosine = float((dot / denominator).clamp(-1.0, 1.0))
    return {
        "shape_match": True,
        "candidate_shape": list(candidate_tensor.shape),
        "reference_shape": list(reference_tensor.shape),
        "finite": bool(
            torch.isfinite(candidate_flat).all()
            and torch.isfinite(reference_flat).all()
        ),
        "cosine": cosine,
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.abs().mean()) if delta.numel() else 0.0,
        "relative_l2": float(delta.norm(dtype=torch.float64))
        / max(float(reference_norm), 1.0e-12),
        "bitwise_equal": bool(torch.equal(candidate, reference)),
    }


def metric_passed(row: dict[str, Any]) -> bool:
    return bool(row["shape_match"] and row["finite"] and row["cosine"] >= COSINE_MIN)


def make_case(
    batch: int,
    tokens: int,
    channels: int,
    seed: int,
) -> dict[str, tuple[torch.Tensor, ...] | torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        batch,
        tokens,
        channels,
        dtype=DTYPE,
        device="cuda",
        generator=generator,
    )
    shifted = torch.randn(
        batch,
        tokens,
        channels,
        dtype=DTYPE,
        device="cuda",
        generator=generator,
    )
    mixes = tuple(
        torch.rand(
            channels,
            dtype=DTYPE,
            device="cuda",
            generator=generator,
        )
        for _ in OUTPUT_NAMES
    )
    output_gradients = tuple(
        torch.randn(
            batch,
            tokens,
            channels,
            dtype=DTYPE,
            device="cuda",
            generator=generator,
        )
        for _ in OUTPUT_NAMES
    )
    return {
        "inputs": (x, shifted, *mixes),
        "output_gradients": output_gradients,
    }


def clone_inputs(base: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().clone().requires_grad_(True) for value in base)


def canonical_mix6(
    x: torch.Tensor,
    shifted: torch.Tensor,
    *mixes: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    delta = shifted - x
    return tuple(x + delta * mix.view(1, 1, -1) for mix in mixes)


def collect_reference(
    base: dict[str, tuple[torch.Tensor, ...] | torch.Tensor],
) -> dict[str, tuple[torch.Tensor, ...]]:
    inputs = clone_inputs(base["inputs"])  # type: ignore[arg-type]
    outputs = canonical_mix6(*inputs)
    gradients = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=base["output_gradients"],  # type: ignore[arg-type]
    )
    torch.cuda.synchronize()
    return {
        "outputs": tuple(value.detach().clone() for value in outputs),
        "gradients": tuple(value.detach().clone() for value in gradients),
    }


class NativeCallObserver(AbstractContextManager["NativeCallObserver"]):
    """Count calls through the registered CUDA operator without replacing math."""

    def __init__(self) -> None:
        namespace = getattr(torch.ops, OPERATOR_NAMESPACE)
        self._namespace = namespace
        self._forward = namespace.forward
        self._backward = namespace.backward
        self.forward_calls = 0
        self.backward_calls = 0

    def __enter__(self) -> NativeCallObserver:
        def observed_forward(*args):
            self.forward_calls += 1
            return self._forward(*args)

        def observed_backward(*args):
            self.backward_calls += 1
            return self._backward(*args)

        setattr(self._namespace, "forward", observed_forward)
        setattr(self._namespace, "backward", observed_backward)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        setattr(self._namespace, "forward", self._forward)
        setattr(self._namespace, "backward", self._backward)
        return None

    def snapshot(self) -> tuple[int, int]:
        return self.forward_calls, self.backward_calls


def collect_candidate(
    kernel_api: Any,
    base: dict[str, tuple[torch.Tensor, ...] | torch.Tensor],
) -> dict[str, tuple[torch.Tensor, ...]]:
    inputs = clone_inputs(base["inputs"])  # type: ignore[arg-type]
    outputs = public_mix6_training_v1(kernel_api, *inputs)
    gradients = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=base["output_gradients"],  # type: ignore[arg-type]
    )
    torch.cuda.synchronize()
    return {
        "outputs": tuple(value.detach().clone() for value in outputs),
        "gradients": tuple(value.detach().clone() for value in gradients),
    }


def public_mix6_training_v1(
    kernel_api: Any,
    *inputs: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Execute Mix6 through the only public API-v4 package boundary."""

    envelope = kernel_api.execute_optional_v4(
        "mix6_training",
        *inputs,
        program_id=None,
        facts={},
    )
    if not isinstance(envelope, Mapping):
        raise TypeError("mix6_training returned a non-mapping API-v4 envelope")
    if envelope.get("kind") != "mix6_training":
        raise ValueError("mix6_training returned the wrong API-v4 kind")
    if envelope.get("supported") is not True:
        raise RuntimeError(str(envelope.get("reason", "Mix6 is unsupported")))
    result = envelope.get("result")
    if not isinstance(result, tuple) or len(result) != len(OUTPUT_NAMES):
        raise TypeError("mix6_training result must contain six tensors")
    if any(not isinstance(value, torch.Tensor) for value in result):
        raise TypeError("mix6_training result must contain only tensors")
    return result


def expected_strategy(rows: int, threshold: int) -> str:
    return CANONICAL_STRATEGY if rows < threshold else NATIVE_STRATEGY


def observed_strategy(
    call_rows: list[dict[str, int]],
    *,
    expected: str,
) -> tuple[str, bool]:
    expected_backward_calls = 0 if expected == CANONICAL_STRATEGY else 1
    passed = all(
        row["forward_calls"] == 1 and row["backward_calls"] == expected_backward_calls
        for row in call_rows
    )
    if not passed:
        return "unexpected-operator-call-pattern", False
    return expected, True


def validate_case(
    mix6: Any,
    *,
    batch: int,
    tokens: int,
    channels: int,
    repeats: int,
    seed: int,
    native_backward_min_rows: int,
) -> dict[str, Any]:
    base = make_case(batch, tokens, channels, seed)
    reference = collect_reference(base)
    candidates = []
    call_rows = []
    with NativeCallObserver() as observer:
        for repeat in range(repeats):
            before_forward, before_backward = observer.snapshot()
            candidates.append(collect_candidate(mix6, base))
            after_forward, after_backward = observer.snapshot()
            call_rows.append(
                {
                    "repeat": repeat,
                    "forward_calls": after_forward - before_forward,
                    "backward_calls": after_backward - before_backward,
                }
            )

    first = candidates[0]
    output_metrics = {
        name: tensor_metric(candidate, expected)
        for name, candidate, expected in zip(
            OUTPUT_NAMES,
            first["outputs"],
            reference["outputs"],
            strict=True,
        )
    }
    gradient_metrics = {
        name: tensor_metric(candidate, expected)
        for name, candidate, expected in zip(
            INPUT_NAMES,
            first["gradients"],
            reference["gradients"],
            strict=True,
        )
    }
    repeat_rows = []
    for repeat, candidate in enumerate(candidates[1:], start=1):
        output_equal = {
            name: bool(torch.equal(value, baseline))
            for name, value, baseline in zip(
                OUTPUT_NAMES,
                candidate["outputs"],
                first["outputs"],
                strict=True,
            )
        }
        gradient_equal = {
            name: bool(torch.equal(value, baseline))
            for name, value, baseline in zip(
                INPUT_NAMES,
                candidate["gradients"],
                first["gradients"],
                strict=True,
            )
        }
        repeat_rows.append(
            {
                "repeat": repeat,
                "outputs_bitwise_equal": output_equal,
                "gradients_bitwise_equal": gradient_equal,
                "passed": all(output_equal.values()) and all(gradient_equal.values()),
            }
        )

    rows = batch * tokens
    strategy_expected = expected_strategy(rows, native_backward_min_rows)
    strategy_observed, strategy_passed = observed_strategy(
        call_rows,
        expected=strategy_expected,
    )
    determinism_passed = all(row["passed"] for row in repeat_rows)
    outputs_passed = all(metric_passed(row) for row in output_metrics.values())
    gradients_passed = all(metric_passed(row) for row in gradient_metrics.values())
    passed = bool(
        outputs_passed and gradients_passed and determinism_passed and strategy_passed
    )
    return {
        "case": f"b{batch}-t{tokens}-c{channels}",
        "passed": passed,
        "shape": {
            "batch": batch,
            "tokens": tokens,
            "channels": channels,
            "flattened_rows": rows,
        },
        "strategy": {
            "expected": strategy_expected,
            "observed": strategy_observed,
            "passed": strategy_passed,
            "native_backward_min_rows": native_backward_min_rows,
            "operator_calls": call_rows,
        },
        "outputs": output_metrics,
        "outputs_passed": outputs_passed,
        "gradients": gradient_metrics,
        "gradients_passed": gradients_passed,
        "determinism": {
            "repeats": repeats,
            "comparison_runs": repeat_rows,
            "passed": determinism_passed,
        },
    }


def validate_higher_order(
    mix6: Any,
    *,
    channels: int,
    seed: int,
    native_backward_min_rows: int,
) -> dict[str, Any]:
    # Use a row count above the normal native threshold.  Observing zero native
    # backward calls therefore proves that create_graph, rather than shape,
    # selected the canonical replay.
    batch = 4
    tokens = 16
    base = make_case(batch, tokens, channels, seed)
    inputs = clone_inputs(base["inputs"])  # type: ignore[arg-type]
    with NativeCallObserver() as observer:
        outputs = public_mix6_training_v1(mix6, *inputs)
        first_gradients = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=base["output_gradients"],  # type: ignore[arg-type]
            create_graph=True,
        )
        objective = sum(
            gradient.float().square().mean() for gradient in first_gradients
        )
        second_gradients = torch.autograd.grad(
            objective,
            inputs,
            allow_unused=True,
        )
        torch.cuda.synchronize()

    first_rows = {
        name: {
            "requires_grad": bool(gradient.requires_grad),
            "finite": bool(torch.isfinite(gradient).all()),
        }
        for name, gradient in zip(INPUT_NAMES, first_gradients, strict=True)
    }
    second_rows = {
        name: {
            "present": gradient is not None,
            "finite": bool(torch.isfinite(gradient).all())
            if gradient is not None
            else False,
        }
        for name, gradient in zip(INPUT_NAMES, second_gradients, strict=True)
    }
    rows = batch * tokens
    call_pattern_passed = bool(
        rows >= native_backward_min_rows
        and observer.forward_calls == 1
        and observer.backward_calls == 0
    )
    passed = bool(
        all(row["requires_grad"] and row["finite"] for row in first_rows.values())
        and all(row["present"] and row["finite"] for row in second_rows.values())
        and call_pattern_passed
    )
    return {
        "passed": passed,
        "shape": {
            "batch": batch,
            "tokens": tokens,
            "channels": channels,
            "flattened_rows": rows,
        },
        "first_gradients": first_rows,
        "second_gradients": second_rows,
        "strategy": {
            "expected": CANONICAL_STRATEGY,
            "observed": (
                CANONICAL_STRATEGY
                if call_pattern_passed
                else "unexpected-operator-call-pattern"
            ),
            "reason": "create_graph=True",
            "native_backward_min_rows": native_backward_min_rows,
            "forward_calls": observer.forward_calls,
            "backward_calls": observer.backward_calls,
            "passed": call_pattern_passed,
        },
    }


def validate_non_default_stream(
    mix6: Any,
    *,
    channels: int,
    seed: int,
    native_backward_min_rows: int,
) -> dict[str, Any]:
    """Prove that both native launches follow the tensor device's current stream."""

    # Keep this above the threshold so both forward and backward launch on the
    # explicitly selected stream.  Seventeen tokens also proves that the raw
    # tensor leaf itself has no hidden 16-token alignment requirement.
    batch = 4
    tokens = 17
    base = make_case(batch, tokens, channels, seed)
    reference = collect_reference(base)
    stream = torch.cuda.Stream(device=torch.device("cuda", torch.cuda.current_device()))
    default_stream = torch.cuda.default_stream(stream.device)
    with NativeCallObserver() as observer:
        with torch.cuda.stream(stream):
            # The reference collection synchronized creation on the default
            # stream above.  The candidate therefore has no unresolved input
            # dependency when it is enqueued on this non-default stream.
            candidate = collect_candidate(mix6, base)

    output_metrics = {
        name: tensor_metric(actual, expected)
        for name, actual, expected in zip(
            OUTPUT_NAMES,
            candidate["outputs"],
            reference["outputs"],
            strict=True,
        )
    }
    gradient_metrics = {
        name: tensor_metric(actual, expected)
        for name, actual, expected in zip(
            INPUT_NAMES,
            candidate["gradients"],
            reference["gradients"],
            strict=True,
        )
    }
    distinct_stream = bool(stream.cuda_stream != default_stream.cuda_stream)
    call_pattern_passed = bool(
        batch * tokens >= native_backward_min_rows
        and observer.forward_calls == 1
        and observer.backward_calls == 1
    )
    passed = bool(
        distinct_stream
        and call_pattern_passed
        and all(metric_passed(row) for row in output_metrics.values())
        and all(metric_passed(row) for row in gradient_metrics.values())
    )
    return {
        "passed": passed,
        "shape": {
            "batch": batch,
            "tokens": tokens,
            "channels": channels,
            "flattened_rows": batch * tokens,
        },
        "stream": {
            "non_default": distinct_stream,
            "candidate_handle": int(stream.cuda_stream),
            "default_handle": int(default_stream.cuda_stream),
        },
        "operator_calls": {
            "forward": observer.forward_calls,
            "backward": observer.backward_calls,
            "passed": call_pattern_passed,
        },
        "outputs": output_metrics,
        "gradients": gradient_metrics,
    }


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def wheel_identities(args: argparse.Namespace) -> dict[str, Any]:
    wheels = {}
    for name, wheel in (
        ("rwkv7_hf", args.hf_wheel),
        ("rwkv7_kernels", args.kernel_wheel),
    ):
        if wheel is not None:
            wheels[name] = file_identity(wheel)
    return wheels


def source_identity(mix6: Any) -> dict[str, Any]:
    module = file_identity(Path(mix6.__file__))
    module_text = Path(mix6.__file__).read_text(encoding="utf-8")
    build_module = (
        Path(mix6.__file__).resolve().parents[1] / "nvidia" / "extension_build.py"
    )
    source_root = Path(mix6._source_root()).resolve()
    cpp = source_root / "rwkv7_tmix_mix6_shifted_bf16_v1.cpp"
    cuda = source_root / "rwkv7_tmix_mix6_shifted_bf16_v1.cu"
    cuda_text = cuda.read_text(encoding="utf-8")
    audit = {
        "parallel_partial_reduction_present": (
            "mix6_shifted_backward_partials_kernel" in cuda_text
        ),
        "fixed_order_finalize_present": (
            "mix6_shifted_backward_finalize_kernel" in cuda_text
            and "for (int64_t partial = 0; partial < partial_count; ++partial)"
            in cuda_text
        ),
        "bounded_row_tiles_present": (
            "MIX6_ROWS_PER_PARTIAL = 64" in cuda_text
            and "row += MIX6_ROW_WORKERS_PER_BLOCK" in cuda_text
        ),
        "full_row_serial_loop_absent": (
            "for (int64_t row = 0; row < rows; ++row)" not in cuda_text
        ),
        "fp32_partial_workspace_present": "parameter_partials" in cuda_text,
        "atomic_add_absent": "atomicAdd" not in cuda_text,
        "second_grid_dimension_absent": "blockIdx.y" not in cuda_text,
        "all_launches_checked": (
            cuda_text.count("C10_CUDA_KERNEL_LAUNCH_CHECK()") == 3
        ),
        "tensor_device_current_stream": (
            cuda_text.count("getCurrentCUDAStream(x.get_device())") == 2
        ),
        "legacy_train_temp_loader_absent": "train_temp_cuda" not in module_text,
        "fast_math_build_flag_absent": "--use_fast_math"
        not in tuple(mix6.CUDA_EXTENSION_OPTIMIZATION_FLAGS),
    }
    audit["passed"] = all(audit.values())
    return {
        "python_module": module,
        "cuda_build_module": file_identity(build_module),
        "cpp_source": file_identity(cpp),
        "cuda_source": file_identity(cuda),
        "cuda_build_flags": list(mix6.CUDA_EXTENSION_OPTIMIZATION_FLAGS),
        "deterministic_backward_source_audit": audit,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.channels <= 0 or args.channels % 2:
        raise ValueError("--channels must be a positive even integer")
    if args.repeats < MINIMUM_REPEATS:
        raise ValueError(f"--repeats must be at least {MINIMUM_REPEATS}")

    batches = tuple(args.batch or (1, 4))
    tokens = tuple(args.tokens or (1, 17, 128))
    if any(batch <= 0 for batch in batches):
        raise ValueError("--batch values must be positive")
    if any(token <= 0 for token in tokens):
        raise ValueError("--tokens values must be positive")

    import rwkv7_kernels as kernel_api
    import rwkv7_kernels.time_mix.training_mix6 as mix6

    # Exercise the same versioned dispatcher called by rwkv7_hf.ops_rwkv7,
    # rather than validating only the private raw implementation.
    os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "adaptive"
    mix6.load_mix6_training_cuda_extension(verbose=args.verbose_build)
    native_backward_min_rows = int(mix6.NATIVE_BACKWARD_MIN_ROWS)
    source = source_identity(mix6)

    cases = []
    failures = []
    for batch in batches:
        for token_count in tokens:
            case_seed = args.seed + batch * 10_000 + token_count
            try:
                row = validate_case(
                    kernel_api,
                    batch=batch,
                    tokens=token_count,
                    channels=args.channels,
                    repeats=args.repeats,
                    seed=case_seed,
                    native_backward_min_rows=native_backward_min_rows,
                )
            except Exception as exc:
                row = {
                    "case": f"b{batch}-t{token_count}-c{args.channels}",
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            cases.append(row)
            if not row["passed"]:
                failures.append(row)
            torch.cuda.empty_cache()

    try:
        higher_order = validate_higher_order(
            kernel_api,
            channels=args.channels,
            seed=args.seed + 900_000,
            native_backward_min_rows=native_backward_min_rows,
        )
    except Exception as exc:
        higher_order = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not higher_order["passed"]:
        failures.append({"case": "higher-order", **higher_order})

    try:
        non_default_stream = validate_non_default_stream(
            kernel_api,
            channels=args.channels,
            seed=args.seed + 950_000,
            native_backward_min_rows=native_backward_min_rows,
        )
    except Exception as exc:
        non_default_stream = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not non_default_stream["passed"]:
        failures.append({"case": "non-default-stream", **non_default_stream})

    source_audit_passed = source["deterministic_backward_source_audit"]["passed"]
    if not source_audit_passed:
        failures.append(
            {
                "case": "deterministic-backward-source-audit",
                "passed": False,
                "details": source["deterministic_backward_source_audit"],
            }
        )
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema": "rwkv7-mix6-training-validation-v1",
        "status": "passed" if not failures else "failed",
        "command": list(sys.argv),
        "code_sha": args.code_sha or git_revision(root),
        "environment": environment(),
        "wheels": wheel_identities(args),
        "implementation": {
            "operator_namespace": OPERATOR_NAMESPACE,
            "operator_registered": bool(
                hasattr(getattr(torch.ops, OPERATOR_NAMESPACE), "forward")
                and hasattr(getattr(torch.ops, OPERATOR_NAMESPACE), "backward")
            ),
            "native_backward_min_rows": native_backward_min_rows,
            "policy": {
                "rows_below_threshold": CANONICAL_STRATEGY,
                "rows_at_or_above_threshold": NATIVE_STRATEGY,
                "create_graph": CANONICAL_STRATEGY,
            },
            "source": source,
        },
        "settings": {
            "dtype": "bfloat16",
            "batches": list(batches),
            "tokens": list(tokens),
            "channels": args.channels,
            "repeats": args.repeats,
            "seed": args.seed,
            "output_count": len(OUTPUT_NAMES),
            "gradient_count": len(INPUT_NAMES),
            "output_names": list(OUTPUT_NAMES),
            "gradient_names": list(INPUT_NAMES),
            "cosine_min": COSINE_MIN,
            "determinism": "bitwise-equal",
        },
        "cases": cases,
        "higher_order": higher_order,
        "non_default_stream": non_default_stream,
        "failures": failures,
    }
    write_report(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
