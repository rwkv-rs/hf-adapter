#!/usr/bin/env python3
"""Benchmark the reference RWKV-7 path against a pinned FLA checkout.

This is a throughput diagnostic rather than a correctness gate.  Both
backends load the same Hugging Face weights, use the same inputs and run
without CUDA graphs or compilation wrappers.  Triton compilation is excluded
by warmup iterations.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path

import torch


EXPECTED_FLA_COMMIT = "80e494f6c588e091fc8316b612870df29375c5b8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fla-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def synchronize() -> None:
    torch.cuda.synchronize()


def measure(function, *, warmup: int, repeats: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            value = function()
            del value
        synchronize()
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
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def model_cases(model, *, warmup: int, repeats: int, decode_tokens: int) -> dict:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(42)
    rows = {}
    for batch, length in ((1, 128), (1, 512), (1, 2048), (4, 128), (4, 512)):
        ids = torch.randint(1, 8192, (batch, length), generator=generator, device=device)

        def forward_no_cache(ids=ids):
            return model(input_ids=ids, use_cache=False).logits

        result = measure(forward_no_cache, warmup=warmup, repeats=repeats)
        result["tokens_per_second"] = batch * length / (result["median_ms"] / 1000.0)
        rows[f"prefill_no_cache_b{batch}_t{length}"] = result

    for batch in (1, 4):
        prompt = torch.randint(1, 8192, (batch, 128), generator=generator, device=device)
        tokens = torch.randint(
            1, 8192, (batch, decode_tokens), generator=generator, device=device
        )
        with torch.inference_mode():
            initial = model(input_ids=prompt, use_cache=True)
            cache = initial.past_key_values
            # Compile/warm all one-token paths before measuring the continuous
            # recurrent decode loop.
            token = tokens[:, :1]
            for _ in range(4):
                output = model(input_ids=token, past_key_values=cache, use_cache=True)
                cache = output.past_key_values

        def decode_sequence():
            nonlocal cache
            last = None
            for index in range(decode_tokens):
                last = model(
                    input_ids=tokens[:, index : index + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = last.past_key_values
            return last.logits

        result = measure(decode_sequence, warmup=1, repeats=max(3, repeats // 2))
        result["milliseconds_per_token_step"] = result["median_ms"] / decode_tokens
        result["tokens_per_second"] = (
            batch * decode_tokens / (result["median_ms"] / 1000.0)
        )
        rows[f"cached_decode_b{batch}"] = result
    return rows


def operator_cases(kind: str, dtype: torch.dtype, *, warmup: int, repeats: int) -> dict:
    if kind == "reference":
        from rwkv7_hf.ops_rwkv7 import rwkv7_recurrent

        def invoke(values):
            return rwkv7_recurrent(
                values["r"],
                values["w"].exp(),
                values["k"],
                values["v"],
                values["a"],
                values["b"],
                values["state"],
            )
    else:
        from fla.ops.rwkv7 import chunk_rwkv7

        def invoke(values):
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

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(7)
    rows = {}
    for batch, length in ((1, 128), (1, 512), (1, 2048), (4, 128), (4, 512)):
        shape = (batch, length, 2, 64)
        values = {
            name: torch.randn(shape, generator=generator, device=device, dtype=dtype) * 0.1
            for name in ("r", "w", "k", "v", "a", "b")
        }
        values["w"] = -(values["w"].abs() + 0.1)
        values["state"] = torch.randn(
            (batch, 2, 64, 64), generator=generator, device=device, dtype=dtype
        ) * 0.01

        result = measure(lambda: invoke(values), warmup=warmup, repeats=repeats)
        result["tokens_per_second"] = batch * length / (result["median_ms"] / 1000.0)
        rows[f"wkv_b{batch}_t{length}"] = result
        del values
    return rows


def load_model(kind: str, model_path: Path, dtype: torch.dtype):
    if kind == "reference":
        from rwkv7_hf.configuration_rwkv7 import RWKV7Config
        from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM
    else:
        from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
        from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM
    config = RWKV7Config.from_pretrained(model_path)
    return RWKV7ForCausalLM.from_pretrained(
        model_path, config=config, dtype=dtype
    ).cuda().eval()


def add_speedups(report: dict) -> None:
    for section in ("operator", "model"):
        reference = report["backends"]["reference"][section]
        fla = report["backends"]["fla"][section]
        for name in reference.keys() & fla.keys():
            fla[name]["speedup_vs_reference"] = (
                reference[name]["median_ms"] / fla[name]["median_ms"]
            )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    marker = args.fla_source / ".fla-upstream-commit"
    commit = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    if commit != EXPECTED_FLA_COMMIT:
        raise SystemExit(f"unexpected FLA commit: {commit!r}")
    sys.path.insert(0, str(args.fla_source.resolve()))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    torch.manual_seed(args.seed)

    report = {
        "schema_version": 1,
        "purpose": "non-blocking throughput diagnostic",
        "model": str(args.model.resolve()),
        "dtype": args.dtype,
        "fla_commit": commit,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "triton": package_version("triton"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "settings": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "decode_tokens": args.decode_tokens,
            "cuda_graph": False,
            "torch_compile": False,
        },
        "backends": {},
    }
    for kind in ("reference", "fla"):
        started = time.perf_counter()
        operator = operator_cases(
            kind, dtype, warmup=args.warmup, repeats=args.repeats
        )
        model = load_model(kind, args.model, dtype)
        model_rows = model_cases(
            model,
            warmup=args.warmup,
            repeats=args.repeats,
            decode_tokens=args.decode_tokens,
        )
        report["backends"][kind] = {
            "operator": operator,
            "model": model_rows,
            "elapsed_seconds": time.perf_counter() - started,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
    add_speedups(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
