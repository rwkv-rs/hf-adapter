#!/usr/bin/env python3
"""Isolated RTX 5090 B8 dense-FFN compilation probe.

This script does not alter the HF runtime.  It measures whether one shared
``torch.compile`` callable can accelerate the exact dense RWKV FFN boundary

    linear(H, 4H) -> relu^2 -> linear(4H, H) + residual

for the two decode shapes that matter to the 0.4B and 1.5B models.  Each timing
sample replays 24 distinct layer-weight pairs inside a CUDA graph so the result
is reported directly as microseconds saved per autoregressive model step.

The probe is deliberately fail-closed: a publishable pass requires an exact
RTX 5090 / sm_120, FP16, B8, H in {1024, 2048}, a real full-graph compile that
is reused across distinct weight pointers, finite outputs, cosine >= 0.9999,
identical row-wise argmaxes, and strictly more than the requested saving.
Even a pass remains diagnostic evidence until the route is integrated and a
full-model multi-step greedy/cache check passes.
"""

from __future__ import annotations

# Support direct ``python bench/<category>/<script>.py`` execution.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import math
import statistics
import traceback
from pathlib import Path
from typing import Any, Callable


SUPPORTED_HIDDEN = (1024, 2048)
EXPECTED_BATCH = 8
DEFAULT_LAYERS = 24
DEFAULT_MIN_COSINE = 0.9999
DEFAULT_MIN_SAVED_US = 50.0


def is_exact_rtx5090(device_name: str, capability: tuple[int, int]) -> bool:
    normalized = " ".join(str(device_name).lower().split())
    return "rtx 5090" in normalized and tuple(capability) == (12, 0)


def strict_saving_pass(saved_us: float | None, threshold_us: float) -> bool:
    if saved_us is None:
        return False
    value = float(saved_us)
    threshold = float(threshold_us)
    return bool(math.isfinite(value) and math.isfinite(threshold) and value > threshold)


def validate_shape(batch_size: int, hidden: int, layers: int) -> None:
    if int(batch_size) != EXPECTED_BATCH:
        raise ValueError(f"probe is restricted to batch size {EXPECTED_BATCH}")
    if int(hidden) not in SUPPORTED_HIDDEN:
        raise ValueError(f"hidden must be one of {SUPPORTED_HIDDEN}")
    if int(layers) <= 0:
        raise ValueError("layers must be positive")


def summarize_us(samples: list[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("timing samples must be non-empty, finite, and positive")
    ordered = sorted(values)
    return {
        "samples_us_per_step": [round(value, 6) for value in values],
        "median_us_per_step": round(float(statistics.median(values)), 6),
        "min_us_per_step": round(ordered[0], 6),
        "max_us_per_step": round(ordered[-1], 6),
    }


def correctness_gate_pass(
    summary: dict[str, Any],
    *,
    min_cosine: float = DEFAULT_MIN_COSINE,
) -> bool:
    try:
        compared = int(summary["vectors_compared"])
        cosine = float(summary["min_cosine"])
        finite = bool(summary["all_finite"])
        argmax_equal = bool(summary["argmax_all_equal"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        compared > 0
        and math.isfinite(cosine)
        and cosine >= float(min_cosine)
        and finite
        and argmax_equal
    )


def probe_status(
    *,
    exact_rtx5090: bool,
    compile_effective: bool,
    compile_reused: bool,
    correctness_passed: bool,
    saving_passed: bool,
) -> tuple[str, str]:
    if not compile_effective:
        return "fail", "fullgraph_compile_not_effective"
    if not compile_reused:
        return "fail", "compiled_callable_recompiled_for_layer_weights"
    if not correctness_passed:
        return "fail", "correctness_gate_failed"
    if not exact_rtx5090:
        return "diagnostic_only", "proxy_device_no_5090_claim"
    if not saving_passed:
        return "diagnostic_miss", "strict_step_saving_not_met"
    return "diagnostic_pass", "strict_step_saving_met"


def choose_fastest_passing_candidate(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if bool(row.get("compile_effective"))
        and bool(row.get("compile_reused"))
        and bool(row.get("correctness_pass"))
        and math.isfinite(float(row.get("median_us_per_step", math.nan)))
    ]
    if not eligible:
        return None
    return min(
        eligible, key=lambda row: (float(row["median_us_per_step"]), str(row["label"]))
    )


def _counter_value(torch_module, group: str, name: str) -> int:
    counters = getattr(getattr(torch_module, "_dynamo", None), "utils", None)
    counters = getattr(counters, "counters", {})
    return int(counters.get(group, {}).get(name, 0))


def _compile_shared(
    torch_module,
    fn: Callable[..., Any],
    example_args: tuple[Any, ...],
    *,
    mode: str,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    counters = torch_module._dynamo.utils.counters
    counters.clear()
    compiled = torch_module.compile(
        fn,
        backend="inductor",
        mode=mode,
        fullgraph=True,
        dynamic=False,
    )
    for _ in range(3):
        compiled(*example_args)
    torch_module.cuda.synchronize()
    unique_after_first = _counter_value(torch_module, "stats", "unique_graphs")
    graph_breaks_after_first = sum(
        int(value) for value in counters.get("graph_break", {}).values()
    )
    return compiled, {
        "unique_graphs_after_first_weight": unique_after_first,
        "graph_breaks_after_first_weight": graph_breaks_after_first,
    }


def _capture(torch_module, fn: Callable[[], Any], *, warmup: int):
    side_stream = torch_module.cuda.Stream()
    side_stream.wait_stream(torch_module.cuda.current_stream())
    with torch_module.cuda.stream(side_stream), torch_module.inference_mode():
        for _ in range(int(warmup)):
            fn()
    torch_module.cuda.current_stream().wait_stream(side_stream)
    torch_module.cuda.synchronize()
    graph = torch_module.cuda.CUDAGraph()
    with torch_module.cuda.graph(graph):
        fn()
    torch_module.cuda.synchronize()
    return graph


def _time_graph_us(torch_module, graph, *, replays: int) -> float:
    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(replays)):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / float(replays)


def _correctness_summary(
    torch_module, reference: list[Any], candidate: list[Any]
) -> dict[str, Any]:
    if len(reference) != len(candidate) or not reference:
        return {}
    cosines: list[float] = []
    max_abs = 0.0
    all_finite = True
    argmax_all_equal = True
    vectors = 0
    for expected, actual in zip(reference, candidate, strict=True):
        expected32 = expected.float()
        actual32 = actual.float()
        all_finite = bool(
            all_finite
            and torch_module.isfinite(expected32).all().item()
            and torch_module.isfinite(actual32).all().item()
        )
        max_abs = max(max_abs, float((expected32 - actual32).abs().max().item()))
        cosine = torch_module.nn.functional.cosine_similarity(
            expected32, actual32, dim=-1
        )
        cosines.append(float(cosine.min().item()))
        argmax_all_equal = bool(
            argmax_all_equal
            and torch_module.equal(expected32.argmax(dim=-1), actual32.argmax(dim=-1))
        )
        vectors += int(expected32.shape[0])
    return {
        "vectors_compared": vectors,
        "min_cosine": min(cosines),
        "max_abs_diff": max_abs,
        "all_finite": all_finite,
        "argmax_all_equal": argmax_all_equal,
        "argmax_scope": "row-wise FFN-output structural proxy; full-model greedy gate still required",
    }


def _measure_shape(
    torch_module, args: argparse.Namespace, hidden: int
) -> dict[str, Any]:
    validate_shape(args.batch_size, hidden, args.layers)
    batch = int(args.batch_size)
    intermediate = 4 * int(hidden)
    layers = int(args.layers)
    generator = torch_module.Generator(device=args.device)
    generator.manual_seed(int(args.seed) + int(hidden))

    def randn(*shape: int, scale: float = 1.0):
        return torch_module.randn(
            *shape,
            generator=generator,
            device=args.device,
            dtype=torch_module.float16,
        ).mul_(float(scale))

    xs = [randn(batch, hidden) for _ in range(layers)]
    residuals = [randn(batch, hidden) for _ in range(layers)]
    up_weights = [
        randn(intermediate, hidden, scale=hidden**-0.5) for _ in range(layers)
    ]
    down_weights = [
        randn(hidden, intermediate, scale=intermediate**-0.5) for _ in range(layers)
    ]

    def reference_ffn(x, up, down, residual):
        hidden_values = torch_module.relu(
            torch_module.nn.functional.linear(x, up)
        ).square()
        return residual + torch_module.nn.functional.linear(hidden_values, down)

    def compiled_linear_add(x, up, down, residual):
        hidden_values = torch_module.relu(
            torch_module.nn.functional.linear(x, up)
        ).square()
        return residual + torch_module.nn.functional.linear(hidden_values, down)

    def compiled_addmm(x, up, down, residual):
        hidden_values = torch_module.relu(
            torch_module.nn.functional.linear(x, up)
        ).square()
        return torch_module.addmm(residual, hidden_values, down.t())

    variants = {
        "compiled_linear_add": compiled_linear_add,
        "compiled_addmm": compiled_addmm,
    }
    baseline_outputs: list[Any] = []

    def baseline_step() -> None:
        baseline_outputs[:] = [
            reference_ffn(x, up, down, residual)
            for x, up, down, residual in zip(
                xs, up_weights, down_weights, residuals, strict=True
            )
        ]

    baseline_step()
    baseline_graph = _capture(torch_module, baseline_step, warmup=args.warmup)
    candidate_rows: list[dict[str, Any]] = []
    candidate_graphs: dict[str, Any] = {}
    for mode in args.compile_modes:
        for variant_name, variant_fn in variants.items():
            label = f"{variant_name}:{mode}"
            row: dict[str, Any] = {
                "label": label,
                "mode": mode,
                "variant": variant_name,
            }
            try:
                compiled, compile_stats = _compile_shared(
                    torch_module,
                    variant_fn,
                    (xs[0], up_weights[0], down_weights[0], residuals[0]),
                    mode=mode,
                )
                # Different pointers with identical shapes must reuse the same graph.
                for layer_index in range(1, min(layers, 4)):
                    compiled(
                        xs[layer_index],
                        up_weights[layer_index],
                        down_weights[layer_index],
                        residuals[layer_index],
                    )
                torch_module.cuda.synchronize()
                unique_after_reuse = _counter_value(
                    torch_module, "stats", "unique_graphs"
                )
                graph_breaks_after_reuse = sum(
                    int(value)
                    for value in torch_module._dynamo.utils.counters.get(
                        "graph_break", {}
                    ).values()
                )
                compile_effective = bool(
                    compile_stats["unique_graphs_after_first_weight"] == 1
                    and compile_stats["graph_breaks_after_first_weight"] == 0
                )
                compile_reused = bool(
                    unique_after_reuse == 1 and graph_breaks_after_reuse == 0
                )
                candidate_outputs: list[Any] = []

                def candidate_step() -> None:
                    candidate_outputs[:] = [
                        compiled(x, up, down, residual)
                        for x, up, down, residual in zip(
                            xs, up_weights, down_weights, residuals, strict=True
                        )
                    ]

                # Compare fresh clones allocated outside either CUDA-graph pool.
                # Keeping references to tensors produced while a graph is being
                # captured is not a sound oracle: a later capture may legally
                # reuse that private pool and make unrelated lanes appear to
                # differ catastrophically.
                fresh_reference = [
                    reference_ffn(x, up, down, residual).detach().clone()
                    for x, up, down, residual in zip(
                        xs, up_weights, down_weights, residuals, strict=True
                    )
                ]
                fresh_candidate = [
                    compiled(x, up, down, residual).detach().clone()
                    for x, up, down, residual in zip(
                        xs, up_weights, down_weights, residuals, strict=True
                    )
                ]
                torch_module.cuda.synchronize()
                correctness = _correctness_summary(
                    torch_module, fresh_reference, fresh_candidate
                )
                correctness_passed = correctness_gate_pass(
                    correctness, min_cosine=args.min_cosine
                )
                candidate_graph = _capture(
                    torch_module, candidate_step, warmup=args.warmup
                )
                candidate_graphs[label] = candidate_graph
                row.update(
                    {
                        **compile_stats,
                        "unique_graphs_after_distinct_weights": unique_after_reuse,
                        "graph_breaks_after_distinct_weights": graph_breaks_after_reuse,
                        "compile_effective": compile_effective,
                        "compile_reused": compile_reused,
                        "correctness": correctness,
                        "correctness_pass": correctness_passed,
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "compile_effective": False,
                        "compile_reused": False,
                        "correctness_pass": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            candidate_rows.append(row)

    raw_samples: dict[str, list[float]] = {
        "baseline_eager_dense": [],
        **{label: [] for label in candidate_graphs},
    }
    timing_lanes = [("baseline_eager_dense", baseline_graph), *candidate_graphs.items()]
    for repeat_index in range(int(args.repeats)):
        order = timing_lanes if repeat_index % 2 == 0 else list(reversed(timing_lanes))
        for label, graph in order:
            for _ in range(int(args.graph_warmup_replays)):
                graph.replay()
            torch_module.cuda.synchronize()
            raw_samples[label].append(
                _time_graph_us(torch_module, graph, replays=args.replays)
            )

    baseline_timing = summarize_us(raw_samples["baseline_eager_dense"])
    baseline_us = float(baseline_timing["median_us_per_step"])
    for row in candidate_rows:
        samples = raw_samples.get(str(row["label"]))
        if samples:
            timing = summarize_us(samples)
            candidate_us = float(timing["median_us_per_step"])
            row.update(timing)
            row["saved_us_per_step"] = round(baseline_us - candidate_us, 6)
            row["speedup"] = round(baseline_us / candidate_us, 6)

    best = choose_fastest_passing_candidate(candidate_rows)
    best_saved = None if best is None else float(best["saved_us_per_step"])
    saving_passed = strict_saving_pass(best_saved, args.min_saved_us)
    compile_effective = bool(best is not None and best.get("compile_effective"))
    compile_reused = bool(best is not None and best.get("compile_reused"))
    correctness_passed = bool(best is not None and best.get("correctness_pass"))
    return {
        "hidden": int(hidden),
        "intermediate": intermediate,
        "batch_size": batch,
        "layers_per_step": layers,
        "baseline": baseline_timing,
        "candidates": candidate_rows,
        "best_candidate": None if best is None else str(best["label"]),
        "best_saved_us_per_step": best_saved,
        "strict_saving_gate_us": float(args.min_saved_us),
        "strict_saving_pass": saving_passed,
        "compile_effective": compile_effective,
        "compile_reused": compile_reused,
        "correctness_pass": correctness_passed,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this probe accepts CUDA devices only")
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    device_name = torch.cuda.get_device_name(index)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
    exact = is_exact_rtx5090(device_name, capability)
    if not exact and not args.allow_proxy_device:
        raise RuntimeError(
            f"expected exact RTX 5090 sm_120, got {device_name!r} "
            f"sm_{capability[0]}{capability[1]}"
        )
    if args.dtype != "fp16":
        raise ValueError("dense acceptance lane is FP16 only")
    shapes = [_measure_shape(torch, args, hidden) for hidden in args.hidden_sizes]
    compile_effective = all(bool(row["compile_effective"]) for row in shapes)
    compile_reused = all(bool(row["compile_reused"]) for row in shapes)
    correctness_passed = all(bool(row["correctness_pass"]) for row in shapes)
    saving_passed = all(bool(row["strict_saving_pass"]) for row in shapes)
    status, conclusion = probe_status(
        exact_rtx5090=exact,
        compile_effective=compile_effective,
        compile_reused=compile_reused,
        correctness_passed=correctness_passed,
        saving_passed=saving_passed,
    )
    return {
        "axis": "sm120_b8_dense_ffn_shared_compile_probe",
        "status": status,
        "conclusion": conclusion,
        "diagnostic_only": True,
        "publishable_speed_claim": False,
        "device": device_name,
        "compute_capability": list(capability),
        "exact_rtx5090": exact,
        "proxy_device_allowed": bool(args.allow_proxy_device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "batch_size": int(args.batch_size),
        "hidden_sizes": [int(value) for value in args.hidden_sizes],
        "layers_per_step": int(args.layers),
        "min_cosine_gate": float(args.min_cosine),
        "strict_saving_gate_us_per_step": float(args.min_saved_us),
        "compile_modes": list(args.compile_modes),
        "timing_scope": {
            "clock": "CUDA events",
            "sample_unit": f"{int(args.layers)} distinct dense FFNs",
            "includes": ["two dense FP16 GEMMs", "ReLU square", "residual add"],
            "excludes": ["model load", "attention", "norm/mix", "compile", "capture"],
            "integration_gate": (
                "full model must separately pass >=512 decode steps, cosine >=0.9999, "
                "all greedy tokens, cache semantics, and the paired Qwen table"
            ),
        },
        "shapes": shapes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16",), default="fp16")
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        choices=SUPPORTED_HIDDEN,
        default=list(SUPPORTED_HIDDEN),
    )
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument(
        "--compile-modes",
        nargs="+",
        default=["default", "max-autotune-no-cudagraphs"],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--graph-warmup-replays", type=int, default=20)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--min-cosine", type=float, default=DEFAULT_MIN_COSINE)
    parser.add_argument("--min-saved-us", type=float, default=DEFAULT_MIN_SAVED_US)
    parser.add_argument("--allow-proxy-device", action="store_true")
    parser.add_argument("--results", default="")
    args = parser.parse_args(argv)
    for name in ("layers", "warmup", "graph_warmup_replays", "replays", "repeats"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.batch_size) != EXPECTED_BATCH:
        parser.error(f"--batch-size must be {EXPECTED_BATCH}")
    if (
        not math.isfinite(float(args.min_cosine))
        or not 0.0 < float(args.min_cosine) <= 1.0
    ):
        parser.error("--min-cosine must be finite and in (0, 1]")
    if not math.isfinite(float(args.min_saved_us)) or float(args.min_saved_us) <= 0.0:
        parser.error("--min-saved-us must be finite and positive")
    if len(set(args.hidden_sizes)) != len(args.hidden_sizes):
        parser.error("--hidden-sizes must not contain duplicates")
    return args


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        row = run_probe(args)
    except Exception as exc:
        row = {
            "axis": "sm120_b8_dense_ffn_shared_compile_probe",
            "status": "fail",
            "diagnostic_only": True,
            "publishable_speed_claim": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        }
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    _append_jsonl(args.results, row)
    return 0 if row.get("status") in {"diagnostic_pass", "diagnostic_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
