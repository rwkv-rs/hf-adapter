#!/usr/bin/env python3
"""Three-way whole-model speed matrix for reference, auto kernels, and FLA."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import gc
from importlib import metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


EXPECTED_FLA_COMMIT = "80e494f6c588e091fc8316b612870df29375c5b8"
CASES = tuple(
    (batch, length)
    for batch in (1, 4, 8)
    for length in (128, 512, 2048)
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fla-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha", required=True)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@contextmanager
def clean_backend(kind: str):
    previous_backend = os.environ.get("RWKV7_BACKEND")
    previous_impl = os.environ.get("RWKV7_KERNEL_IMPL")
    os.environ["RWKV7_BACKEND"] = kind
    os.environ["RWKV7_KERNEL_IMPL"] = "auto"
    try:
        yield
    finally:
        if previous_backend is None:
            os.environ.pop("RWKV7_BACKEND", None)
        else:
            os.environ["RWKV7_BACKEND"] = previous_backend
        if previous_impl is None:
            os.environ.pop("RWKV7_KERNEL_IMPL", None)
        else:
            os.environ["RWKV7_KERNEL_IMPL"] = previous_impl


def measure(function, *, warmup: int, repeats: int) -> dict[str, Any]:
    samples = []
    with torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(warmup):
            value = function()
            del value
        torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - started) * 1000.0
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            value = function()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
            del value
    return {
        "warmup_ms": warmup_ms,
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[max(0, round((len(samples) - 1) * 0.95))],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def actual_route(kind: str) -> dict[str, str] | None:
    if kind == "fla":
        return {"selected": "fla", "implementation": "fla-rwkv7-pinned"}
    from rwkv7_hf.ops_rwkv7 import get_last_recurrent_route

    return get_last_recurrent_route()


def model_cases(
    kind: str,
    model,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    decode_tokens: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    vocab_size = int(model.config.vocab_size)
    generator = torch.Generator(device=device).manual_seed(seed)
    rows: dict[str, Any] = {}
    context = clean_backend(kind) if kind != "fla" else nullcontext()

    with context:
        for batch, length in CASES:
            ids = torch.randint(
                1, vocab_size, (batch, length), generator=generator, device=device
            )

            def prefill(ids=ids):
                return model(input_ids=ids, use_cache=True, logits_to_keep=1)

            row = measure(prefill, warmup=warmup, repeats=repeats)
            row["tokens_per_second"] = batch * length / (row["median_ms"] / 1000)
            row["route"] = actual_route(kind)
            rows[f"prefill_b{batch}_t{length}"] = row

        for batch in (1, 4, 8):
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
                cache = model(
                    input_ids=prompt, use_cache=True, logits_to_keep=1
                ).past_key_values
                for _ in range(4):
                    warmed = model(
                        input_ids=tokens[:, :1],
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    cache = warmed.past_key_values

            def decode():
                nonlocal cache
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

            row = measure(decode, warmup=1, repeats=max(3, repeats))
            row["milliseconds_per_step"] = row["median_ms"] / decode_tokens
            row["tokens_per_second"] = batch * decode_tokens / (
                row["median_ms"] / 1000
            )
            row["route"] = actual_route(kind)
            rows[f"cached_decode_b{batch}"] = row
    return rows


def load_model(kind: str, model_path: Path):
    if kind == "fla":
        from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
        from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM
    else:
        from rwkv7_hf.configuration_rwkv7 import RWKV7Config
        from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM
    config = RWKV7Config.from_pretrained(model_path)
    return RWKV7ForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.float16
    ).cuda().eval()


def main() -> int:
    args = arguments()
    source = args.fla_source.expanduser().resolve()
    commit = (source / ".fla-upstream-commit").read_text().strip()
    if commit != EXPECTED_FLA_COMMIT:
        raise SystemExit(f"unexpected FLA commit: {commit!r}")
    sys.path.insert(0, str(source))
    torch.manual_seed(args.seed)
    report: dict[str, Any] = {
        "schema": "rwkv7-production-speed-v1",
        "code_sha": args.code_sha,
        "model": str(args.model.expanduser().resolve()),
        "fla_commit": commit,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "triton": package_version("triton"),
            "rwkv7_hf": package_version("rwkv7-hf"),
            "rwkv7_kernels": package_version("rwkv7-kernels"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "settings": {
            "dtype": "fp16",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "decode_tokens": args.decode_tokens,
            "logits_to_keep": 1,
            "optimized_policy": "auto: T=1 Triton, T>1 exact CUDA Graph",
        },
        "backends": {},
    }
    for kind in ("reference", "optimized", "fla"):
        model = load_model(kind, args.model)
        report["backends"][kind] = model_cases(
            kind,
            model,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            decode_tokens=args.decode_tokens,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    reference = report["backends"]["reference"]
    optimized = report["backends"]["optimized"]
    fla = report["backends"]["fla"]
    for case in reference.keys() & optimized.keys() & fla.keys():
        optimized[case]["speedup_vs_reference"] = (
            reference[case]["median_ms"] / optimized[case]["median_ms"]
        )
        fla[case]["speedup_vs_reference"] = (
            reference[case]["median_ms"] / fla[case]["median_ms"]
        )
        optimized[case]["speedup_vs_fla"] = (
            fla[case]["median_ms"] / optimized[case]["median_ms"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(reference)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
