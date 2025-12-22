#!/usr/bin/env python3
# coding=utf-8
"""Observational launch-grid tuning for the native MM4 CUDA kernels.

The benchmark loads one Hugging Face model exactly once, extracts the requested
dense ``nn.Linear`` weights, and sweeps the launch overrides already exposed by
``rwkv7_hf.native_quant_mm4``.  Batch one uses ``mm4_gemv_triton``; larger
batches use ``mm4_batched_gemv_triton`` and are also compared with direct
``mm4_batched_dot_triton`` candidates.

Every JSONL candidate row contains CUDA-event median latency and correctness
against both the original dense weight and ``mm4_matmul`` reference.  This tool
is deliberately observational: it never changes environment variables, kernel
policy files, model policy, or production dispatch eligibility.

T4-oriented example::

    python bench/bench_native_quant_mm4_tuning.py \
      --hf-dir /models/rwkv7-g1g-1.5b-hf \
      --modules lm_head --batches 1 2 4 8 \
      --results bench/t4_mm4_tuning.jsonl
"""
from __future__ import annotations

import argparse
import fnmatch
import itertools
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F

from rwkv7_hf.native_quant_mm4 import (
    mm4_batched_dot_enabled,
    mm4_batched_dot_triton,
    mm4_batched_gemv_triton,
    mm4_effective_launch_config,
    mm4_gemv_available,
    mm4_gemv_triton,
    mm4_matmul,
    quantize_mm4,
)


VALID_GEMV_BLOCKS = (16, 32, 64, 128)
VALID_DOT_BLOCK_B = (16, 32, 64)
VALID_DOT_WARPS = (1, 2, 4, 8)

# A useful but bounded T4 grid.  The CLI can expand or reduce every dimension.
DEFAULT_BATCHES = (1, 2, 4, 8)
DEFAULT_GEMV_BLOCK_PAIRS = (32, 64, 128)
DEFAULT_GEMV_BLOCK_N = (32, 64, 128)
DEFAULT_DOT_BLOCK_B = (16,)
DEFAULT_DOT_BLOCK_PAIRS = (32, 64, 128)
DEFAULT_DOT_BLOCK_N = (32, 64)
DEFAULT_DOT_WARPS = (2, 4)


@dataclass(frozen=True)
class GemvLaunch:
    block_pairs: int
    block_n: int


@dataclass(frozen=True)
class DotLaunch:
    block_b: int
    block_pairs: int
    block_n: int
    num_warps: int


def _unique_ints(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    """Return stable, positive, duplicate-free CLI integers."""

    result: list[int] = []
    for raw in values:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{name} values must be positive; got {value}")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(result)


def _validate_choices(values: Iterable[int], allowed: Sequence[int], *, name: str) -> tuple[int, ...]:
    result = _unique_ints(values, name=name)
    invalid = [value for value in result if value not in allowed]
    if invalid:
        raise ValueError(f"{name} contains {invalid}; allowed values are {list(allowed)}")
    return result


def gemv_launch_grid(block_pairs: Iterable[int], block_n: Iterable[int]) -> list[GemvLaunch]:
    """Build the exact cross-product accepted by the native GEMV APIs."""

    pairs = _validate_choices(block_pairs, VALID_GEMV_BLOCKS, name="gemv block-pairs")
    widths = _validate_choices(block_n, VALID_GEMV_BLOCKS, name="gemv block-n")
    return [GemvLaunch(pairs_value, width) for pairs_value, width in itertools.product(pairs, widths)]


def dot_launch_grid(
    block_b: Iterable[int],
    block_pairs: Iterable[int],
    block_n: Iterable[int],
    num_warps: Iterable[int],
) -> list[DotLaunch]:
    """Build the exact cross-product accepted by ``mm4_batched_dot_triton``."""

    rows = _validate_choices(block_b, VALID_DOT_BLOCK_B, name="dot block-b")
    pairs = _validate_choices(block_pairs, VALID_GEMV_BLOCKS, name="dot block-pairs")
    widths = _validate_choices(block_n, VALID_GEMV_BLOCKS, name="dot block-n")
    warps = _validate_choices(num_warps, VALID_DOT_WARPS, name="dot warps")
    return [
        DotLaunch(rows_value, pairs_value, width, warps_value)
        for rows_value, pairs_value, width, warps_value in itertools.product(
            rows, pairs, widths, warps
        )
    ]


def resolve_linear_modules(model: torch.nn.Module, selectors: Sequence[str]) -> list[tuple[str, torch.nn.Module]]:
    """Resolve exact names, unique suffixes, or shell-style module globs."""

    if not selectors:
        raise ValueError("at least one --modules selector is required")
    named = dict(model.named_modules())
    selected: list[tuple[str, torch.nn.Module]] = []
    seen: set[str] = set()
    for selector in selectors:
        if any(char in selector for char in "*?["):
            matches = [(name, module) for name, module in named.items() if fnmatch.fnmatchcase(name, selector)]
        elif selector in named:
            matches = [(selector, named[selector])]
        else:
            suffix = "." + selector
            matches = [(name, module) for name, module in named.items() if name.endswith(suffix)]
            if len(matches) > 1:
                names = ", ".join(name for name, _ in matches[:8])
                raise ValueError(f"module selector {selector!r} is ambiguous: {names}")
        if not matches:
            raise ValueError(f"module selector {selector!r} matched nothing")
        for name, module in matches:
            weight = getattr(module, "weight", None)
            if not isinstance(module, torch.nn.Linear) or weight is None or weight.dim() != 2:
                raise ValueError(f"module {name!r} is not a 2-D torch.nn.Linear")
            if name not in seen:
                selected.append((name, module))
                seen.add(name)
    return selected


def correctness_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Return finite-safe error metrics, with cosine reduced over batch rows."""

    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    af = actual.detach().float()
    ef = expected.detach().float()
    finite = bool(torch.isfinite(af).all().item() and torch.isfinite(ef).all().item())
    if not finite:
        return {"finite": False, "max_abs": None, "mean_abs": None, "min_cosine": None}
    diff = (af - ef).abs()
    af_rows = af.reshape(1, -1) if af.dim() == 1 else af.reshape(-1, af.shape[-1])
    ef_rows = ef.reshape(1, -1) if ef.dim() == 1 else ef.reshape(-1, ef.shape[-1])
    cosine = F.cosine_similarity(af_rows, ef_rows, dim=-1)
    return {
        "finite": True,
        "max_abs": float(diff.max().cpu()),
        "mean_abs": float(diff.mean().cpu()),
        "min_cosine": float(cosine.min().cpu()),
    }


def cuda_median_ms(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    iterations: int,
) -> tuple[float, list[float]]:
    """Measure median device latency with independent CUDA-event samples."""

    if device.type != "cuda":
        raise ValueError("cuda_median_ms requires a CUDA device")
    if warmup < 0 or repeats <= 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and repeats/iterations must be > 0")
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize(device)
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            events.append((start, end))
    torch.cuda.synchronize(device)
    samples = [float(start.elapsed_time(end)) / float(iterations) for start, end in events]
    return float(statistics.median(samples)), samples


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append compact, one-object-per-line benchmark records."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _device_map(device: torch.device) -> dict[str, int]:
    return {"": int(device.index if device.index is not None else torch.cuda.current_device())}


def load_hf_model(hf_dir: str, *, device: torch.device) -> torch.nn.Module:
    """Load the only HF model used by this process."""

    # Keep transformers lazy so CPU-only unit tests can import the grid helpers.
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        hf_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=_device_map(device),
    ).eval()


def _quantize_weight(
    weight: torch.Tensor,
    *,
    device: torch.device,
    quantize_on: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Quantize with the repository API and place its tensors on ``device``."""

    if quantize_on == "cpu":
        source = weight.detach().to("cpu").t().contiguous()
    elif quantize_on == "cuda":
        source = weight.detach().to(device).t().contiguous()
    else:
        raise ValueError(f"unsupported quantization device: {quantize_on}")
    packed, mx, rx_s, my, ry_s, m_orig, m_padded = quantize_mm4(source)
    tensors = tuple(value.to(device) for value in (packed, mx, rx_s, my, ry_s))
    return (*tensors, int(m_orig), int(m_padded))


def _model_metadata(hf_dir: str, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    capability = torch.cuda.get_device_capability(device)
    config = getattr(model, "config", None)
    return {
        "model_id_or_path": hf_dir,
        "model_name": Path(hf_dir.rstrip("/")).name,
        "model_type": getattr(config, "model_type", None),
        "model_load_count": 1,
        "dtype": "fp16",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_compute_capability": [int(capability[0]), int(capability[1])],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _candidate_row(
    common: dict[str, Any],
    *,
    kernel: str,
    launch: GemvLaunch | DotLaunch,
    fn: Callable[[], torch.Tensor],
    dense: torch.Tensor,
    reference: torch.Tensor,
    dense_median_ms: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    row = {
        **common,
        "kernel": kernel,
        "launch": asdict(launch),
        "dense_median_ms": round(float(dense_median_ms), 6),
    }
    try:
        with torch.inference_mode():
            candidate = fn()
        torch.cuda.synchronize(device)
        versus_dense = correctness_metrics(candidate, dense)
        versus_reference = correctness_metrics(candidate, reference)
        median_ms, samples = cuda_median_ms(
            fn,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            iterations=args.iterations,
        )
        reference_max_abs = versus_reference["max_abs"]
        dense_cosine = versus_dense["min_cosine"]
        passed = bool(
            versus_reference["finite"]
            and versus_dense["finite"]
            and reference_max_abs is not None
            and reference_max_abs <= args.reference_max_abs
            and dense_cosine is not None
            and dense_cosine >= args.dense_cosine_min
        )
        row.update(
            {
                "status": "pass" if passed else "fail",
                "correctness_pass": passed,
                "candidate_median_ms": round(float(median_ms), 6),
                "latency_samples_ms": [round(float(value), 6) for value in samples],
                "speedup_vs_dense": round(float(dense_median_ms) / float(median_ms), 6)
                if median_ms > 0
                else None,
                "candidate_vs_dense": versus_dense,
                "candidate_vs_reference": versus_reference,
            }
        )
    except Exception as exc:  # A tuning grid must preserve other candidate evidence.
        row.update(
            {
                "status": "error",
                "correctness_pass": False,
                "candidate_median_ms": None,
                "latency_samples_ms": [],
                "speedup_vs_dense": None,
                "candidate_vs_dense": None,
                "candidate_vs_reference": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return row


def benchmark_model(
    model: torch.nn.Module,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Benchmark selected weights from an already-loaded model."""

    batches = _unique_ints(args.batches, name="batches")
    gemv_grid = gemv_launch_grid(args.gemv_block_pairs, args.gemv_block_n)
    dot_grid = dot_launch_grid(
        args.dot_block_b,
        args.dot_block_pairs,
        args.dot_block_n,
        args.dot_warps,
    )
    modules = resolve_linear_modules(model, args.modules)
    metadata = _model_metadata(args.hf_dir, model, device)
    active_policy = mm4_effective_launch_config(device)
    production_dot_enabled = bool(mm4_batched_dot_enabled(device))
    rows: list[dict[str, Any]] = []

    for module_name, module in modules:
        weight = module.weight.detach()
        if weight.device != device:
            raise ValueError(f"module {module_name!r} is on {weight.device}, expected {device}")
        if weight.dtype != torch.float16:
            raise TypeError(f"module {module_name!r} has {weight.dtype}; MM4 dot tuning requires fp16")
        packed, mx, rx_s, my, ry_s, m_orig, m_padded = _quantize_weight(
            weight,
            device=device,
            quantize_on=args.quantize_on,
        )
        in_features = int(weight.shape[1])
        out_features = int(weight.shape[0])

        for batch_size in batches:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(args.seed) + batch_size)
            x = torch.randn(
                batch_size,
                in_features,
                dtype=torch.float16,
                device=device,
                generator=generator,
            )
            with torch.inference_mode():
                # Bias is intentionally excluded: the launch candidates expose
                # dequant-matmul only, and RWKV projection weights are biasless.
                dense = F.linear(x, weight, None)
                reference = mm4_matmul(x, packed, mx, rx_s, my, ry_s, m_orig)
            reference_vs_dense = correctness_metrics(reference, dense)
            dense_median_ms, _ = cuda_median_ms(
                lambda: F.linear(x, weight, None),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                iterations=args.iterations,
            )
            common = {
                "axis": "native_quant_mm4_tuning",
                "schema_version": 1,
                **metadata,
                "module": module_name,
                "weight_shape_out_in": [out_features, in_features],
                "batch_size": int(batch_size),
                "quantized_output_features": int(m_orig),
                "quantized_padded_output_features": int(m_padded),
                "module_has_bias": bool(module.bias is not None),
                "bias_included": False,
                "quantize_on": args.quantize_on,
                "warmup": int(args.warmup),
                "repeats": int(args.repeats),
                "iterations_per_repeat": int(args.iterations),
                "reference_max_abs_limit": float(args.reference_max_abs),
                "dense_cosine_min": float(args.dense_cosine_min),
                "reference_vs_dense": reference_vs_dense,
                "production_batched_dot_enabled": production_dot_enabled,
                "production_effective_launch": active_policy,
                "observational_only": True,
                "policy_promotion": "disabled",
                "policy_promoted": False,
            }

            for launch in gemv_grid:
                if batch_size == 1:
                    kernel = "mm4_gemv_triton"

                    def candidate_fn(launch: GemvLaunch = launch) -> torch.Tensor:
                        return mm4_gemv_triton(
                            x[0],
                            packed,
                            mx,
                            rx_s,
                            my,
                            ry_s,
                            m_orig,
                            block_pairs=launch.block_pairs,
                            block_n=launch.block_n,
                        ).unsqueeze(0)

                else:
                    kernel = "mm4_batched_gemv_triton"

                    def candidate_fn(launch: GemvLaunch = launch) -> torch.Tensor:
                        return mm4_batched_gemv_triton(
                            x,
                            packed,
                            mx,
                            rx_s,
                            my,
                            ry_s,
                            m_orig,
                            block_pairs=launch.block_pairs,
                            block_n=launch.block_n,
                        )

                row = _candidate_row(
                    common,
                    kernel=kernel,
                    launch=launch,
                    fn=candidate_fn,
                    dense=dense,
                    reference=reference,
                    dense_median_ms=dense_median_ms,
                    args=args,
                    device=device,
                )
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)

            if batch_size >= args.dot_min_batch:
                for launch in dot_grid:

                    def dot_candidate_fn(launch: DotLaunch = launch) -> torch.Tensor:
                        return mm4_batched_dot_triton(
                            x,
                            packed,
                            mx,
                            rx_s,
                            my,
                            ry_s,
                            m_orig,
                            block_b=launch.block_b,
                            block_pairs=launch.block_pairs,
                            block_n=launch.block_n,
                            num_warps=launch.num_warps,
                        )

                    row = _candidate_row(
                        common,
                        kernel="mm4_batched_dot_triton",
                        launch=launch,
                        fn=dot_candidate_fn,
                        dense=dense,
                        reference=reference,
                        dense_median_ms=dense_median_ms,
                        args=args,
                        device=device,
                    )
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)

        del packed, mx, rx_s, my, ry_s
        torch.cuda.empty_cache()
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep native MM4 GEMV and batched-dot launches without promoting policy."
    )
    parser.add_argument("--hf-dir", required=True, help="HF model ID or local converted model directory")
    parser.add_argument("--device", default="cuda:0", help="single CUDA device (default: cuda:0)")
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["lm_head"],
        help="exact names, unique suffixes, or globs of nn.Linear modules",
    )
    parser.add_argument("--batches", nargs="+", type=int, default=list(DEFAULT_BATCHES))
    parser.add_argument(
        "--gemv-block-pairs", nargs="+", type=int, default=list(DEFAULT_GEMV_BLOCK_PAIRS)
    )
    parser.add_argument("--gemv-block-n", nargs="+", type=int, default=list(DEFAULT_GEMV_BLOCK_N))
    parser.add_argument("--dot-block-b", nargs="+", type=int, default=list(DEFAULT_DOT_BLOCK_B))
    parser.add_argument(
        "--dot-block-pairs", nargs="+", type=int, default=list(DEFAULT_DOT_BLOCK_PAIRS)
    )
    parser.add_argument("--dot-block-n", nargs="+", type=int, default=list(DEFAULT_DOT_BLOCK_N))
    parser.add_argument("--dot-warps", nargs="+", type=int, default=list(DEFAULT_DOT_WARPS))
    parser.add_argument(
        "--dot-min-batch",
        type=int,
        default=2,
        help="first batch size that receives direct batched-dot candidates",
    )
    parser.add_argument(
        "--quantize-on",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU reduces transient T4 VRAM use; candidate tensors always run on CUDA",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reference-max-abs", type=float, default=0.5)
    parser.add_argument("--dense-cosine-min", type=float, default=0.98)
    parser.add_argument(
        "--results",
        default=str(Path(__file__).with_name("results_native_quant_mm4_tuning.jsonl")),
        help="append JSONL here; pass an empty string to disable the file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return exit code 2 when any candidate errors or misses correctness limits",
    )
    return parser


def _validate_runtime_args(args: argparse.Namespace) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("native MM4 tuning requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must name one CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    if not mm4_gemv_available(device):
        raise RuntimeError("native MM4 Triton GEMV is unavailable")
    _unique_ints(args.batches, name="batches")
    if args.dot_min_batch <= 0:
        raise ValueError("--dot-min-batch must be positive")
    if args.warmup < 0 or args.repeats <= 0 or args.iterations <= 0:
        raise ValueError("warmup must be >= 0 and repeats/iterations must be > 0")
    return device


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = _validate_runtime_args(args)
    torch.cuda.empty_cache()
    print(f"loading one model: {args.hf_dir}", file=sys.stderr, flush=True)
    model = load_hf_model(args.hf_dir, device=device)
    rows = benchmark_model(model, args, device=device)
    if args.results:
        append_jsonl(args.results, rows)
        print(f"appended {len(rows)} rows -> {args.results}", file=sys.stderr, flush=True)
    passed = sum(row["status"] == "pass" for row in rows)
    failed = sum(row["status"] == "fail" for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    print(
        f"candidates={len(rows)} pass={passed} fail={failed} error={errors}; policy promotion disabled",
        file=sys.stderr,
        flush=True,
    )
    if not rows or passed == 0:
        return 2
    if args.strict and (failed or errors):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
