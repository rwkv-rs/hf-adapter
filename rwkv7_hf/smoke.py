# coding=utf-8
"""Download, load, generate, and report the effective RWKV-7 runtime route."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence


DEFAULT_MODEL = "wangyue114514/rwkv7-g1d-0.1b-hf"
DEFAULT_REVISION = "v0.7.0"


def _choose_device(torch_module: Any, requested: str) -> Any:
    if requested == "auto":
        if bool(torch_module.cuda.is_available()):
            return torch_module.device("cuda")
        mps = getattr(getattr(torch_module, "backends", None), "mps", None)
        if bool(getattr(mps, "is_available", lambda: False)()):
            return torch_module.device("mps")
        return torch_module.device("cpu")
    device = torch_module.device(requested)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch_module.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _choose_dtype(torch_module: Any, requested: str, device: Any) -> Any:
    values = {
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
        "fp32": torch_module.float32,
    }
    if requested != "auto":
        return values[requested]
    return torch_module.float32 if device.type == "cpu" else torch_module.float16


def _synchronize(torch_module: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)
    elif device.type == "mps":
        synchronize = getattr(torch_module.mps, "synchronize", None)
        if callable(synchronize):
            synchronize()


def _runtime_version() -> str | None:
    try:
        return importlib.metadata.version("rwkv7-hf")
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prompt", default="User: Hello! Assistant:")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="auto"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _choose_device(torch, args.device)
    dtype = _choose_dtype(torch, args.dtype, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=True,
            dtype=dtype,
            local_files_only=args.local_files_only,
        )
        .to(device)
        .eval()
    )
    _synchronize(torch, device)
    load_seconds = time.perf_counter() - started

    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[1])
    if prompt_tokens < 2:
        raise RuntimeError(
            "the smoke prompt tokenized to fewer than two tokens; use a longer "
            "--prompt so both prefill and decode routes are exercised"
        )
    with torch.inference_mode():
        _synchronize(torch, device)
        prefill_started = time.perf_counter()
        output = model(**encoded, use_cache=True, return_dict=True)
        _synchronize(torch, device)
        prefill_seconds = time.perf_counter() - prefill_started

        if not bool(torch.isfinite(output.logits).all().item()):
            raise RuntimeError("prefill produced non-finite logits")
        cache = output.past_key_values
        next_token = output.logits[:, -1:].argmax(dim=-1)
        generated = [next_token]

        decode_steps = max(0, args.max_new_tokens - 1)
        _synchronize(torch, device)
        decode_started = time.perf_counter()
        for _ in range(decode_steps):
            output = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            if not bool(torch.isfinite(output.logits).all().item()):
                raise RuntimeError("decode produced non-finite logits")
            cache = output.past_key_values
            next_token = output.logits[:, -1:].argmax(dim=-1)
            generated.append(next_token)
        _synchronize(torch, device)
        decode_seconds = time.perf_counter() - decode_started

    generated_ids = torch.cat(generated, dim=1)
    runtime_report = (
        model.rwkv7_runtime_report()
        if hasattr(model, "rwkv7_runtime_report")
        else {"available": False}
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "model": args.model,
        "revision": args.revision,
        "device": str(device),
        "dtype": str(dtype),
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "rwkv7_hf": _runtime_version(),
        },
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(generated_ids.shape[1]),
        "generated_token_ids": generated_ids[0].tolist(),
        "generated_text": tokenizer.decode(generated_ids[0], skip_special_tokens=True),
        "timing": {
            "load_seconds": round(load_seconds, 6),
            "prefill_seconds": round(prefill_seconds, 6),
            "prefill_tokens_per_second": round(prompt_tokens / prefill_seconds, 3),
            "decode_steps": decode_steps,
            "decode_seconds": round(decode_seconds, 6),
            "decode_tokens_per_second": (
                round(decode_steps / decode_seconds, 3) if decode_steps else None
            ),
        },
        "peak_memory_bytes": peak_memory,
        "runtime": runtime_report,
    }


def render_text(report: dict[str, Any]) -> str:
    timing = report["timing"]
    runtime = report["runtime"]
    kernels = runtime.get("kernels", {}).get("package", {})
    return "\n".join(
        [
            "RWKV-7 Hugging Face public-model smoke",
            f"Model: {report['model']}@{report['revision']}",
            f"Device: {report['device']} dtype={report['dtype']}",
            f"Prefill: {timing['prefill_tokens_per_second']} tok/s",
            f"Decode: {timing['decode_tokens_per_second']} tok/s",
            f"Backend: prefill={runtime.get('last_prefill_backend')} decode={runtime.get('last_decode_backend')}",
            f"Kernel package: {kernels.get('status', 'unknown')}",
            f"Generated: {report['generated_text']!r}",
            "RESULT: PASS",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_smoke(args)
    rendered_json = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered_json + "\n", encoding="utf-8")
    print(rendered_json if args.json else render_text(report))
    return 0


def cli() -> None:
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"RESULT: FAIL: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    cli()
