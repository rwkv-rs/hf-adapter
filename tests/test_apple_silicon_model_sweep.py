#!/usr/bin/env python3
# coding=utf-8
"""Apple Silicon / MPS converted-model generation sweep.

This is a local hardware harness, not a Linux CI benchmark. On non-Apple hosts it
emits a SKIP row and exits 0 so the entry point remains syntax-checkable.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
os.environ.setdefault("RWKV7_FAST_FORWARD", "0")
os.environ.setdefault("RWKV7_FAST_CACHE", "0")
os.environ.setdefault("RWKV7_FAST_TOKEN_BACKEND", "native_jit")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def append_result(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def emit(path: str, row: dict[str, Any]) -> None:
    print(json.dumps(row, ensure_ascii=False))
    append_result(path, row)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def darwin_sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return "unknown"


def apple_memory_gb() -> int | str:
    raw = darwin_sysctl("hw.memsize")
    try:
        return round(int(raw) / 1024 / 1024 / 1024)
    except Exception:
        return "unknown"


def infer_model_size_label(model_path: str, explicit: str = "") -> str:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(model_path).name.lower())
    return match.group(1) if match else "unknown"


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not mps_is_available(torch):
            raise RuntimeError("requested --device mps but MPS is unavailable")
        return requested
    return "mps" if mps_is_available(torch) else "cpu"


def dtype_for(torch: Any, name: str) -> Any:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def parse_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    if any(v <= 0 for v in values):
        raise ValueError(f"all values must be positive: {values}")
    return values


def sync(torch: Any, device: str) -> None:
    if device == "mps" and getattr(torch, "mps", None) is not None:
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()



def mps_backend(torch: Any) -> Any | None:
    return getattr(getattr(torch, "backends", None), "mps", None)


def mps_is_available(torch: Any) -> bool:
    mps = mps_backend(torch)
    if mps is None or not hasattr(mps, "is_available"):
        return False
    try:
        return bool(mps.is_available())
    except Exception:
        return False


def mps_is_built(torch: Any) -> bool:
    mps = mps_backend(torch)
    if mps is None or not hasattr(mps, "is_built"):
        return False
    try:
        return bool(mps.is_built())
    except Exception:
        return False


def mps_memory_stats(torch: Any) -> dict[str, int]:
    if not mps_is_available(torch):
        return {}
    stats: dict[str, int] = {}
    for key, func_name in (
        ("mps_current_allocated_memory_bytes", "current_allocated_memory"),
        ("mps_driver_allocated_memory_bytes", "driver_allocated_memory"),
        ("mps_recommended_max_memory_bytes", "recommended_max_memory"),
    ):
        try:
            stats[key] = int(getattr(torch.mps, func_name)())
        except Exception:
            pass
    return stats


def make_prompt_ids(tokenizer: Any, torch: Any, length: int, device: str) -> dict[str, Any]:
    seed_text = "User: Apple Silicon RWKV generation sweep. Assistant: "
    text = seed_text
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    while encoded.shape[1] < length:
        text += seed_text
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = encoded[:, :length].to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, device=device),
    }


def run_sweep(torch: Any, args: argparse.Namespace, device: str, dtype: Any) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    ).eval()
    model.to(device)

    model_name = Path(args.model).name
    model_size_label = infer_model_size_label(args.model, args.model_size_label)
    prompt_lengths = parse_ints(args.prompt_lengths)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

    for prompt_length in prompt_lengths:
        batch = make_prompt_ids(tokenizer, torch, prompt_length, device)
        with torch.no_grad():
            sync(torch, device)
            t0 = time.perf_counter()
            out = model(**batch, use_cache=True, logits_to_keep=1)
            sync(torch, device)
            t1 = time.perf_counter()
            gen = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=None,
            )
            sync(torch, device)
            t2 = time.perf_counter()
        assert out.logits.shape[0] == 1
        assert gen.shape[1] >= batch["input_ids"].shape[1]
        generated = int(gen.shape[1] - batch["input_ids"].shape[1])
        prefill_s = max(t1 - t0, 1e-12)
        generate_s = max(t2 - t1, 1e-12)
        row = {
            "axis": "apple_silicon_model_generate_sweep",
            "status": "pass",
            "model": model_name,
            "model_size_label": model_size_label,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "prompt_tokens": int(batch["input_ids"].shape[1]),
            "generated_tokens": generated,
            "prefill_s": round(prefill_s, 6),
            "generate_s": round(generate_s, 6),
            "total_s": round(t2 - t0, 6),
            "prefill_tokens_per_second": round(prompt_length / prefill_s, 3),
            "decode_tokens_per_second": round(generated / generate_s, 3) if generated else 0.0,
            "backend_class": model.__class__.__name__,
        }
        row.update(mps_memory_stats(torch))
        emit(args.results, row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Converted RWKV-7 HF model directory")
    ap.add_argument("--model-size-label", default="")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--prompt-lengths", default="16,64,128")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--results", default="")
    ap.add_argument("--require-apple", action="store_true")
    args = ap.parse_args()

    if not is_apple_silicon():
        row = {
            "axis": "apple_silicon_model_generate_sweep",
            "status": "skip",
            "reason": "not Darwin/arm64",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": Path(args.model).name,
        }
        emit(args.results, row)
        if args.require_apple:
            raise SystemExit(2)
        return 0

    import torch

    device = choose_device(torch, args.device)
    dtype = dtype_for(torch, args.dtype)
    header = {
        "axis": "apple_silicon_model_sweep_env",
        "status": "info",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": darwin_sysctl("machdep.cpu.brand_string"),
        "memory_gb": apple_memory_gb(),
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": package_version("transformers"),
        "mps_built": mps_is_built(torch),
        "mps_available": mps_is_available(torch),
        "device": device,
        "dtype": args.dtype,
        "model": Path(args.model).name,
        "model_size_label": infer_model_size_label(args.model, args.model_size_label),
        "prompt_lengths": parse_ints(args.prompt_lengths),
        "max_new_tokens": args.max_new_tokens,
        "native_model": os.environ.get("RWKV7_NATIVE_MODEL"),
    }
    header.update(mps_memory_stats(torch))
    emit(args.results, header)
    run_sweep(torch, args, device, dtype)
    print("APPLE SILICON MODEL SWEEP PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
