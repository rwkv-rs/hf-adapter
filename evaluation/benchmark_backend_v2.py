#!/usr/bin/env python3
"""Benchmark readable, backend-v2, and pinned FLA RWKV-7 paths fairly.

Every lane uses the same checkpoint and input tensors.  Cold compile/capture is
reported separately from steady-state samples.  Cached-decode preparation is
performed outside the timed region so prompt prefill is not mislabeled as
decode throughput.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from common import environment, git_revision, model_fingerprint, sha256_file
from fla_common import activate_fla_source, write_json
from training_metrics import adaptive_fast_domain_expected


MATRIX_RECURRENT_IMPLEMENTATION = (
    "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
)
FACTORIZED_RECURRENT_IMPLEMENTATION = (
    "native-nvidia-rwkv7-factorized-recurrent-training-v1"
)
FLATTENED_LINEAR_IMPLEMENTATION = "torch-cuda-rwkv7-flattened-linear-training-v1"
MIX6_IMPLEMENTATION = "native-nvidia-rwkv7-mix6-training-v1"
PROGRAM_IMPLEMENTATION = "native-nvidia-rwkv7-adaptive-training-program-v1"


def canonical_training_mode(value: str) -> str:
    """Normalize historical CLI spellings without restoring old routing."""

    aliases = {
        "adaptive": "adaptive",
        "native": "adaptive",
        "reference": "reference",
        "reference-fallback": "reference",
        "skip-not-applicable": "skip-not-applicable",
    }
    try:
        return aliases[value]
    except KeyError as exc:  # pragma: no cover - argparse owns CLI validation
        raise ValueError(f"unknown training mode: {value}") from exc


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="label=path")
    parser.add_argument("--fla-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument("--tokens", action="append", type=int, default=[])
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--operator-batch", action="append", type=int, default=[])
    parser.add_argument("--operator-tokens", action="append", type=int, default=[])
    parser.add_argument("--operator-warmup", type=int, default=2)
    parser.add_argument("--operator-repeats", type=int, default=5)
    parser.add_argument("--training-model", type=Path)
    parser.add_argument("--training-batch", action="append", type=int, default=[])
    parser.add_argument("--training-tokens", action="append", type=int, default=[])
    parser.add_argument("--training-warmup", type=int, default=1)
    parser.add_argument("--training-repeats", type=int, default=3)
    parser.add_argument(
        "--training-mode",
        choices=(
            "adaptive",
            "reference",
            "skip-not-applicable",
            "native",
            "reference-fallback",
        ),
        default="adaptive",
        help=(
            "adaptive benchmarks the formal optional-kernel training program "
            "with shape-local reference fallback; reference measures the clean "
            "PyTorch baseline; native and reference-fallback are deprecated "
            "aliases; skip-not-applicable records a hardware limitation"
        ),
    )
    parser.add_argument("--training-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    return parser.parse_args()


def parse_models(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, path = value.split("=", 1)
        result[label] = Path(path).expanduser().resolve()
    return result


def route_mode(kind: str) -> None:
    if kind == "optimized":
        os.environ["RWKV7_BACKEND"] = "optimized"
        os.environ["RWKV7_KERNEL_IMPL"] = "auto"
        os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "native"
    else:
        os.environ["RWKV7_BACKEND"] = "reference"
        os.environ["RWKV7_KERNEL_IMPL"] = "auto"
        os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "auto"


def training_route_mode(kind: str, training_mode: str) -> None:
    training_mode = canonical_training_mode(training_mode)
    os.environ["RWKV7_KERNEL_IMPL"] = "auto"
    os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "auto"
    if kind == "optimized" and training_mode == "adaptive":
        # The adaptive training candidate is a composition of independent
        # leaves. Unsupported small/masked shapes deliberately retain the
        # readable PyTorch operation, so the outer failure-containment mode
        # must be ``auto`` rather than strict all-leaves-or-error.
        os.environ["RWKV7_BACKEND"] = "auto"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "adaptive"
    elif kind == "optimized" and training_mode == "reference":
        os.environ["RWKV7_BACKEND"] = "auto"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"
    else:
        os.environ["RWKV7_BACKEND"] = "reference"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"


def last_route(kind: str) -> dict[str, Any] | None:
    if kind == "fla":
        return None
    from rwkv7_hf.ops_rwkv7 import get_last_model_route

    return get_last_model_route()


def last_training_routes(kind: str) -> dict[str, Any] | None:
    """Return every actual boundary used by one clean HF training step."""

    if kind == "fla":
        return None
    from rwkv7_hf.ops_rwkv7 import (
        get_last_linear_route,
        get_last_mix6_route,
        get_last_model_route,
        get_last_recurrent_route,
        get_last_training_program_route,
    )

    return {
        "model": get_last_model_route(),
        "recurrent": get_last_recurrent_route(),
        "linear": get_last_linear_route(),
        "mix6": get_last_mix6_route(),
        "program": get_last_training_program_route(),
    }


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def synchronize() -> None:
    torch.cuda.synchronize()


def timed_inference(
    function: Callable[[], Any], *, warmup: int, repeats: int
) -> dict[str, Any]:
    with torch.inference_mode():
        synchronize()
        started = time.perf_counter()
        cold = function()
        synchronize()
        cold_ms = (time.perf_counter() - started) * 1000.0
        del cold
        for _ in range(warmup):
            value = function()
            synchronize()
            del value
        samples = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            synchronize()
            started = time.perf_counter()
            value = function()
            synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
            del value
    return {
        "cold_ms": cold_ms,
        "compile_or_capture_ms_upper_bound": max(
            0.0, cold_ms - statistics.median(samples)
        ),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "warmup": warmup,
        "repeats": repeats,
    }


def timed_decode(
    prepare: Callable[[], Any],
    decode: Callable[[Any], Any],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        cache = prepare()
        synchronize()
        started = time.perf_counter()
        value = decode(cache)
        synchronize()
        cold_ms = (time.perf_counter() - started) * 1000.0
        del cache, value
        for _ in range(warmup):
            cache = prepare()
            value = decode(cache)
            synchronize()
            del cache, value
        samples = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            cache = prepare()
            synchronize()
            started = time.perf_counter()
            value = decode(cache)
            synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
            del cache, value
    return {
        "cold_ms": cold_ms,
        "compile_or_capture_ms_upper_bound": max(
            0.0, cold_ms - statistics.median(samples)
        ),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "warmup": warmup,
        "repeats": repeats,
    }


def timed_autograd(
    step: Callable[[], Any], *, warmup: int, repeats: int
) -> dict[str, Any]:
    synchronize()
    started = time.perf_counter()
    cold = step()
    synchronize()
    cold_ms = (time.perf_counter() - started) * 1000.0
    del cold
    for _ in range(warmup):
        value = step()
        synchronize()
        del value
    samples = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        value = step()
        synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
        del value
    return {
        "cold_ms": cold_ms,
        "compile_ms_upper_bound": max(0.0, cold_ms - statistics.median(samples)),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "warmup": warmup,
        "repeats": repeats,
    }


def benchmark_operator_lane(
    kind: str,
    dtype: torch.dtype,
    batches: tuple[int, ...],
    tokens: tuple[int, ...],
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if kind == "fla":
        from fla.ops.rwkv7 import chunk_rwkv7
    else:
        from rwkv7_hf.ops_rwkv7 import (
            get_last_recurrent_route,
            rwkv7_recurrent,
            rwkv7_recurrent_reference,
        )

    route_mode(kind)
    rows = {}
    for batch in batches:
        for sequence in tokens:
            generator = torch.Generator(device="cuda").manual_seed(
                seed + batch * 1000 + sequence
            )
            shape = (batch, sequence, 2, 64)
            base = {
                name: torch.randn(
                    shape, device="cuda", dtype=dtype, generator=generator
                )
                * 0.1
                for name in ("r", "k", "v", "a", "b")
            }
            base["w"] = -(
                torch.rand(shape, device="cuda", dtype=dtype, generator=generator) * 0.5
                + 0.1
            )
            base["state"] = (
                torch.randn(
                    (batch, 2, 64, 64),
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                )
                * 0.01
            )

            def invoke(values=base):
                if kind == "fla":
                    return chunk_rwkv7(
                        r=values["r"],
                        w=values["w"],
                        k=values["k"],
                        v=values["v"],
                        a=values["a"],
                        b=values["b"],
                        initial_state=values["state"],
                        output_final_state=True,
                    )
                if kind == "optimized":
                    return rwkv7_recurrent(
                        values["r"],
                        values["w"].exp(),
                        values["k"],
                        values["v"],
                        values["a"],
                        values["b"],
                        values["state"],
                    )
                return rwkv7_recurrent_reference(
                    values["r"],
                    values["w"].exp(),
                    values["k"],
                    values["v"],
                    values["a"],
                    values["b"],
                    values["state"],
                )

            forward = timed_inference(invoke, warmup=warmup, repeats=repeats)
            forward["tokens_per_second"] = (
                batch * sequence / (forward["median_ms"] / 1000.0)
            )
            forward["route"] = (
                get_last_recurrent_route() if kind == "optimized" else None
            )
            if kind == "optimized":
                backward: dict[str, Any] = {
                    "status": "not_applicable",
                    "reason": (
                        "the recurrent-v1 optimized operators are inference-only; "
                        "optimized full-model autograd is benchmarked separately"
                    ),
                }
            else:
                values = {
                    key: value.detach().clone().requires_grad_(True)
                    for key, value in base.items()
                }

                def backward_step():
                    for value in values.values():
                        value.grad = None
                    output, state = (
                        chunk_rwkv7(
                            r=values["r"],
                            w=values["w"],
                            k=values["k"],
                            v=values["v"],
                            a=values["a"],
                            b=values["b"],
                            initial_state=values["state"],
                            output_final_state=True,
                        )
                        if kind == "fla"
                        else rwkv7_recurrent_reference(
                            values["r"],
                            values["w"].exp(),
                            values["k"],
                            values["v"],
                            values["a"],
                            values["b"],
                            values["state"],
                        )
                    )
                    loss = (
                        output.float().square().mean() + state.float().square().mean()
                    )
                    loss.backward()
                    return loss

                backward = timed_autograd(
                    backward_step,
                    warmup=max(1, min(warmup, 2)),
                    repeats=max(3, min(repeats, 5)),
                )
                backward["tokens_per_second"] = (
                    batch * sequence / (backward["median_ms"] / 1000.0)
                )
            rows[f"b{batch}-t{sequence}"] = {
                "forward": forward,
                "forward_backward": backward,
            }
            del base
    return rows


def load_model(kind: str, path: Path, dtype: torch.dtype, *, training: bool = False):
    if kind == "fla":
        from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
        from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM

        config = RWKV7Config.from_pretrained(path)
        config.fuse_norm = False
        config.fuse_cross_entropy = False
        config.fuse_linear_cross_entropy = False
        config.use_l2warp = False
    else:
        from rwkv7_hf.configuration_rwkv7 import RWKV7Config
        from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

        config = RWKV7Config.from_pretrained(path)
        route_mode(kind)
    model = RWKV7ForCausalLM.from_pretrained(
        path, config=config, torch_dtype=dtype
    ).cuda()
    return model.train() if training else model.eval()


def benchmark_inference_lane(
    kind: str,
    path: Path,
    dtype: torch.dtype,
    batches: tuple[int, ...],
    tokens: tuple[int, ...],
    decode_steps: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    model = load_model(kind, path, dtype)
    route_mode(kind)
    vocab = int(model.config.vocab_size)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows: dict[str, Any] = {"prefill": {}, "decode": {}}
    for batch in batches:
        for sequence in tokens:
            ids = torch.randint(
                1, vocab, (batch, sequence), generator=generator, device="cuda"
            )

            def prefill(ids=ids, model=model):
                return model(input_ids=ids, use_cache=True, logits_to_keep=1)

            result = timed_inference(prefill, warmup=warmup, repeats=repeats)
            result["tokens_per_second"] = (
                batch * sequence / (result["median_ms"] / 1000.0)
            )
            result["route"] = last_route(kind)
            rows["prefill"][f"b{batch}-t{sequence}"] = result

    for batch in batches:
        prompt = torch.randint(
            1, vocab, (batch, 128), generator=generator, device="cuda"
        )
        continuation = torch.randint(
            1, vocab, (batch, decode_steps), generator=generator, device="cuda"
        )

        def prepare(prompt=prompt, model=model):
            return model(
                input_ids=prompt, use_cache=True, logits_to_keep=1
            ).past_key_values

        def decode(cache, continuation=continuation, model=model):
            output = None
            for index in range(decode_steps):
                output = model(
                    input_ids=continuation[:, index : index + 1],
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = output.past_key_values
            return output

        result = timed_decode(
            prepare,
            decode,
            warmup=max(1, min(warmup, 2)),
            repeats=max(3, min(repeats, 5)),
        )
        result["steps"] = decode_steps
        result["milliseconds_per_step"] = result["median_ms"] / decode_steps
        result["tokens_per_second"] = (
            batch * decode_steps / (result["median_ms"] / 1000.0)
        )
        result["route"] = last_route(kind)
        rows["decode"][f"b{batch}"] = result
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def timed_training(
    model,
    ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
    legacy_double_ce: bool = False,
) -> dict[str, Any]:
    """Time model training, including exactly one model-provided CE by default.

    ``legacy_double_ce`` reproduces the old diagnostic path: the model first
    computes ``output.loss`` because labels are supplied, then a second CE is
    built from its logits and used for backward.  It is intentionally a Python
    API-only escape hatch so published benchmark runs keep the fair default.
    """

    def step():
        model.zero_grad(set_to_none=True)
        output = model(
            input_ids=ids,
            labels=labels,
            use_cache=False,
            logits_to_keep=0,
        )
        if legacy_double_ce:
            loss = F.cross_entropy(
                output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        else:
            loss = output.loss
            if loss is None:
                raise RuntimeError("training model did not return loss for labels")
        loss.backward()
        return loss

    synchronize()
    started = time.perf_counter()
    cold = step()
    synchronize()
    cold_ms = (time.perf_counter() - started) * 1000.0
    del cold
    for _ in range(warmup):
        value = step()
        synchronize()
        del value
    samples = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        value = step()
        synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
        del value
    return {
        "cold_ms": cold_ms,
        "compile_ms_upper_bound": max(0.0, cold_ms - statistics.median(samples)),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "warmup": warmup,
        "repeats": repeats,
        "loss_mode": ("legacy-double-ce" if legacy_double_ce else "model-output-loss"),
    }


def benchmark_training_lane(
    kind: str,
    path: Path,
    dtype: torch.dtype,
    training_mode: str,
    batches: tuple[int, ...],
    tokens: tuple[int, ...],
    warmup: int,
    repeats: int,
    seed: int,
    *,
    legacy_double_ce: bool = False,
) -> dict[str, Any]:
    model = load_model(kind, path, dtype, training=True)
    training_route_mode(kind, training_mode)
    vocab = int(model.config.vocab_size)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = {}
    for batch in batches:
        for sequence in tokens:
            ids = torch.randint(
                1, vocab, (batch, sequence), generator=generator, device="cuda"
            )
            labels = ids.clone()
            labels[0, sequence // 2] = -100
            result = timed_training(
                model,
                ids,
                labels,
                warmup=warmup,
                repeats=repeats,
                legacy_double_ce=legacy_double_ce,
            )
            result["tokens_per_second"] = (
                batch * sequence / (result["median_ms"] / 1000.0)
            )
            result["shape"] = {"batch": batch, "tokens": sequence}
            result["route"] = last_training_routes(kind)
            rows[f"b{batch}-t{sequence}"] = result
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def add_speedups(report: dict[str, Any]) -> None:
    operator = report.get("operator", {}).get("lanes", {})
    if operator:
        reference = operator["reference"]
        optimized = operator["optimized"]
        fla = operator["fla"]
        for case in reference.keys() & optimized.keys() & fla.keys():
            reference_forward = reference[case]["forward"]["median_ms"]
            optimized_forward = optimized[case]["forward"]
            fla_forward = fla[case]["forward"]
            optimized_forward["speedup_vs_reference"] = (
                reference_forward / optimized_forward["median_ms"]
            )
            optimized_forward["speedup_vs_fla"] = (
                fla_forward["median_ms"] / optimized_forward["median_ms"]
            )
            fla_forward["speedup_vs_reference"] = (
                reference_forward / fla_forward["median_ms"]
            )
            reference_backward = reference[case]["forward_backward"]["median_ms"]
            fla_backward = fla[case]["forward_backward"]
            fla_backward["speedup_vs_reference"] = (
                reference_backward / fla_backward["median_ms"]
            )
    for model in report["models"].values():
        for phase in ("prefill", "decode"):
            reference = model["lanes"]["reference"][phase]
            optimized = model["lanes"]["optimized"][phase]
            fla = model["lanes"]["fla"][phase]
            for case in reference.keys() & optimized.keys() & fla.keys():
                base_ms = reference[case]["median_ms"]
                optimized[case]["speedup_vs_reference"] = (
                    base_ms / optimized[case]["median_ms"]
                )
                optimized[case]["speedup_vs_fla"] = (
                    fla[case]["median_ms"] / optimized[case]["median_ms"]
                )
                fla[case]["speedup_vs_reference"] = base_ms / fla[case]["median_ms"]
                fla[case]["speedup_vs_optimized"] = (
                    optimized[case]["median_ms"] / fla[case]["median_ms"]
                )
    training = report.get("training")
    if training and "lanes" in training:
        reference = training["lanes"]["reference"]
        optimized = training["lanes"]["optimized"]
        fla = training["lanes"]["fla"]
        for case in reference.keys() & optimized.keys() & fla.keys():
            base_ms = reference[case]["median_ms"]
            optimized[case]["speedup_vs_reference"] = (
                base_ms / optimized[case]["median_ms"]
            )
            optimized[case]["speedup_vs_fla"] = (
                fla[case]["median_ms"] / optimized[case]["median_ms"]
            )
            fla[case]["speedup_vs_reference"] = base_ms / fla[case]["median_ms"]


def routes_passed(report: dict[str, Any]) -> bool:
    for row in (
        report.get("operator", {}).get("lanes", {}).get("optimized", {}).values()
    ):
        implementation = str(
            (row["forward"].get("route") or {}).get("implementation", "")
        )
        if implementation not in {
            "native-triton-rank1-scan-v1",
            "torch-cuda-graph-reference-v1",
        }:
            return False
    for model in report["models"].values():
        for row in model["lanes"]["optimized"]["prefill"].values():
            if not str((row.get("route") or {}).get("implementation", "")).startswith(
                "native-nvidia-prefill-v2["
            ):
                return False
        for row in model["lanes"]["optimized"]["decode"].values():
            if not str((row.get("route") or {}).get("implementation", "")).startswith(
                "native-nvidia-fused-decode-v2["
            ):
                return False
    training = report.get("training")
    if training:
        mode = canonical_training_mode(training.get("mode", "adaptive"))
        if mode == "skip-not-applicable":
            return training.get("status") == "not_applicable"
        for row in training["lanes"]["optimized"].values():
            routes = row.get("route") or {}
            model_route = routes.get("model") or {}
            recurrent_route = routes.get("recurrent") or {}
            linear_route = routes.get("linear") or {}
            mix6_route = routes.get("mix6") or {}
            program_route = routes.get("program") or {}
            if not (
                model_route.get("selected") == "reference"
                and model_route.get("phase") == "training"
                and model_route.get("implementation") == "torch-reference-model-v1"
            ):
                return False
            if mode == "reference":
                if not (
                    recurrent_route.get("selected") == "reference"
                    and recurrent_route.get("implementation") == "torch-reference-v1"
                    and linear_route.get("selected") == "reference"
                    and linear_route.get("implementation")
                    == "torch-reference-linear-v1"
                    and mix6_route.get("selected") == "reference"
                    and mix6_route.get("implementation") == "torch-reference-mix6-v1"
                    and program_route.get("selected") == "reference"
                    and program_route.get("implementation")
                    == "torch-reference-training-program-v1"
                ):
                    return False
                continue

            shape = row.get("shape") or {}
            batch = int(shape.get("batch", 0))
            tokens = int(shape.get("tokens", 0))
            fast_domain = adaptive_fast_domain_expected(batch=batch, tokens=tokens)
            recurrent_implementation = (
                FACTORIZED_RECURRENT_IMPLEMENTATION
                if fast_domain
                else MATRIX_RECURRENT_IMPLEMENTATION
            )
            if not (
                recurrent_route.get("selected") == "optimized"
                and recurrent_route.get("implementation") == recurrent_implementation
            ):
                return False
            if fast_domain:
                if not (
                    linear_route.get("selected") == "optimized"
                    and linear_route.get("implementation")
                    == FLATTENED_LINEAR_IMPLEMENTATION
                ):
                    return False
            elif not (
                linear_route.get("selected") == "reference"
                and linear_route.get("implementation") == "torch-reference-linear-v1"
            ):
                return False
            if not (
                mix6_route.get("selected") == "optimized"
                and mix6_route.get("implementation") == MIX6_IMPLEMENTATION
            ):
                return False
            if not (
                program_route.get("selected")
                == ("optimized" if fast_domain else "reference")
                and program_route.get("implementation") == PROGRAM_IMPLEMENTATION
            ):
                return False
    return True


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    training_mode = canonical_training_mode(args.training_mode)
    if (
        args.training_model is not None
        and training_mode == "adaptive"
        and args.training_dtype != "bf16"
    ):
        raise ValueError("adaptive clean-leaf training requires --training-dtype bf16")
    fla = activate_fla_source(args.fla_source)
    models = parse_models(args.model)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    batches = tuple(args.batch or (1, 4, 8))
    tokens = tuple(args.tokens or (128, 512, 2048))
    report: dict[str, Any] = {
        "schema": "rwkv7-backend-v2-three-way-speed-v1",
        "status": "running",
        "code_sha": args.code_sha or git_revision(Path(__file__).resolve().parents[1]),
        "dtype": args.dtype,
        "fla": fla,
        "environment": environment(),
        "settings": {
            "batches": batches,
            "tokens": tokens,
            "decode_steps": args.decode_steps,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "torch_compile": False,
            "preparation_inside_decode_timing": False,
            "seed": args.seed,
        },
        "wheels": {},
        "models": {},
    }
    for name, wheel in (
        ("rwkv7_hf", args.hf_wheel),
        ("rwkv7_kernels", args.kernel_wheel),
    ):
        if wheel is not None:
            wheel = wheel.expanduser().resolve()
            report["wheels"][name] = {
                "path": str(wheel),
                "sha256": sha256_file(wheel),
            }
    operator_batches = tuple(args.operator_batch or (1, 4, 8))
    operator_tokens = tuple(args.operator_tokens or (1, 17, 128, 512, 2048))
    report["operator"] = {
        "settings": {
            "batches": operator_batches,
            "tokens": operator_tokens,
            "warmup": args.operator_warmup,
            "repeats": args.operator_repeats,
        },
        "lanes": {
            kind: benchmark_operator_lane(
                kind,
                dtype,
                operator_batches,
                operator_tokens,
                args.operator_warmup,
                args.operator_repeats,
                args.seed + 500_000,
            )
            for kind in ("reference", "optimized", "fla")
        },
    }
    for model_index, (label, path) in enumerate(models.items()):
        lanes = {}
        for kind in ("reference", "optimized", "fla"):
            lanes[kind] = benchmark_inference_lane(
                kind,
                path,
                dtype,
                batches,
                tokens,
                args.decode_steps,
                args.warmup,
                args.repeats,
                args.seed + model_index * 10_000,
            )
        report["models"][label] = {
            "model": model_fingerprint(path),
            "lanes": lanes,
        }
    if args.training_model is not None:
        training_path = args.training_model.expanduser().resolve()
        train_batches = tuple(args.training_batch or (1, 4))
        train_tokens = tuple(args.training_tokens or (128, 512))
        if training_mode == "skip-not-applicable":
            report["training"] = {
                "model": model_fingerprint(training_path),
                "dtype": args.training_dtype,
                "mode": training_mode,
                "requested_mode": args.training_mode,
                "status": "not_applicable",
                "reason": (
                    "native BF16 tensor leaves require CUDA sm80 or newer; the "
                    "readable HF training loop itself remains available"
                ),
            }
        else:
            train_dtype = (
                torch.bfloat16 if args.training_dtype == "bf16" else torch.float16
            )
            training_lanes = {}
            for kind in ("reference", "optimized", "fla"):
                training_lanes[kind] = benchmark_training_lane(
                    kind,
                    training_path,
                    train_dtype,
                    training_mode,
                    train_batches,
                    train_tokens,
                    args.training_warmup,
                    args.training_repeats,
                    args.seed + 100_000,
                )
            report["training"] = {
                "model": model_fingerprint(training_path),
                "dtype": args.training_dtype,
                "mode": training_mode,
                "requested_mode": args.training_mode,
                "settings": {
                    "batches": train_batches,
                    "tokens": train_tokens,
                    "warmup": args.training_warmup,
                    "repeats": args.training_repeats,
                },
                "lanes": training_lanes,
            }
    add_speedups(report)
    report["route_gate"] = routes_passed(report)
    report["status"] = "passed" if report["route_gate"] else "failed"
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["route_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
