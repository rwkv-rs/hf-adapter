#!/usr/bin/env python3
"""Measure reference versus one optional recurrent-v1 implementation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import torch

from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM
from rwkv7_hf.ops_rwkv7 import get_last_recurrent_route


@contextmanager
def backend_mode(mode: str):
    previous = os.environ.get("RWKV7_BACKEND")
    os.environ["RWKV7_BACKEND"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RWKV7_BACKEND", None)
        else:
            os.environ["RWKV7_BACKEND"] = previous


def kernel_status() -> dict:
    import rwkv7_kernels

    return {
        "api_version": rwkv7_kernels.RWKV7_KERNEL_API_VERSION,
        "package_version": package_version("rwkv7-kernels"),
        "implementation_mode": os.environ.get("RWKV7_KERNEL_IMPL", "auto"),
        "last_route": get_last_recurrent_route(),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--implementation", choices=("auto", "graph", "triton"), required=True
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha", help="source revision when .git is unavailable")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def measure(function, *, warmup: int, repeats: int) -> dict:
    samples = []
    with torch.inference_mode():
        for _ in range(warmup):
            value = function()
            del value
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            value = function()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
            del value
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "route": get_last_recurrent_route(),
    }


def benchmark_backend(
    model,
    *,
    backend: str,
    generator: torch.Generator,
    warmup: int,
    repeats: int,
    decode_tokens: int,
) -> dict:
    rows = {}
    device = torch.device("cuda")
    vocab_size = int(model.config.vocab_size)
    with backend_mode(backend):
        for batch, length in (
            (1, 1),
            (1, 128),
            (1, 512),
            (1, 2048),
            (4, 128),
            (4, 512),
        ):
            ids = torch.randint(
                1,
                vocab_size,
                (batch, length),
                generator=generator,
                device=device,
            )

            def prefill(ids=ids):
                return model(
                    input_ids=ids,
                    use_cache=True,
                    logits_to_keep=1,
                )

            result = measure(prefill, warmup=warmup, repeats=repeats)
            result["tokens_per_second"] = batch * length / (
                result["median_ms"] / 1000.0
            )
            rows[f"generation_prefill_b{batch}_t{length}"] = result

        for batch in (1, 4):
            prompt = torch.randint(
                1, vocab_size, (batch, 128), generator=generator, device=device
            )
            tokens = torch.randint(
                1,
                vocab_size,
                (batch, decode_tokens),
                generator=generator,
                device=device,
            )
            with torch.inference_mode():
                initial_cache = model(
                    input_ids=prompt,
                    use_cache=True,
                    logits_to_keep=1,
                ).past_key_values

            def decode():
                cache = initial_cache.clone()
                output = None
                for index in range(decode_tokens):
                    output = model(
                        input_ids=tokens[:, index : index + 1],
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    cache = output.past_key_values
                return output

            result = measure(
                decode,
                warmup=1,
                repeats=max(3, repeats // 2),
            )
            result["milliseconds_per_step"] = result["median_ms"] / decode_tokens
            result["tokens_per_second"] = batch * decode_tokens / (
                result["median_ms"] / 1000.0
            )
            rows[f"cached_decode_b{batch}"] = result
    return rows


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    os.environ["RWKV7_KERNEL_IMPL"] = args.implementation
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda")
    model = RWKV7ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    report = {
        "schema": "rwkv7-native-backend-speed-v1",
        "code_sha": args.code_sha or git_sha(),
        "model": str(args.model.resolve()),
        "dtype": args.dtype,
        "implementation": args.implementation,
        "settings": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "decode_tokens": args.decode_tokens,
            "cuda_graph": False,
            "torch_compile": False,
            "logits_to_keep": 1,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "triton": package_version("triton"),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "kernel": kernel_status(),
        },
        "backends": {},
    }
    for index, backend in enumerate(("reference", "optimized")):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        report["backends"][backend] = benchmark_backend(
            model,
            backend=backend,
            generator=generator,
            warmup=args.warmup,
            repeats=args.repeats,
            decode_tokens=args.decode_tokens,
        )
    reference = report["backends"]["reference"]
    automatic = report["backends"]["optimized"]
    for name in reference.keys() & automatic.keys():
        automatic[name]["speedup_vs_reference"] = (
            reference[name]["median_ms"] / automatic[name]["median_ms"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(reference)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
