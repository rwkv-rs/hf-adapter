#!/usr/bin/env python3
# coding=utf-8
"""Apple Silicon / MPS smoke for the RWKV-7 native HF backend.

Default behavior is intentionally small: always run a tiny native model, and run
an actual converted HF model only when --model is supplied. On non-Apple hosts the
script exits 0 with a SKIP row unless --require-apple is used, so Linux CI can
syntax-check and keep the entry point importable.
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

# MPS can fall back individual unsupported ops to CPU. This keeps hardware smoke
# useful while the dedicated Metal/MLX backend is still a follow-up target.
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


def mps_memory_stats(torch: Any) -> dict[str, int]:
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
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


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("requested --device mps but torch.backends.mps.is_available() is false")
        return requested
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for(torch: Any, name: str) -> Any:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def tensor_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def run_tiny_native(torch: Any, device: str, dtype: Any, max_new_tokens: int) -> dict[str, Any]:
    from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM

    torch.manual_seed(20260704)
    cfg = NativeRWKV7Config(
        vocab_size=37,
        hidden_size=16,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=32,
        decay_low_rank_dim=4,
        gate_low_rank_dim=4,
        a_low_rank_dim=4,
        v_low_rank_dim=4,
        use_cache=True,
    )
    model = NativeRWKV7ForCausalLM(cfg).eval().to(device=device, dtype=dtype)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
    elapsed = time.perf_counter() - t0
    assert out.shape == (1, input_ids.shape[1] + max_new_tokens), tuple(out.shape)
    return {
        "axis": "apple_silicon_tiny_native",
        "status": "pass",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "generated_tokens": int(max_new_tokens),
        "elapsed_s": round(elapsed, 4),
    }


def run_hf_model(torch: Any, args: argparse.Namespace, device: str, dtype: Any) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    ).eval()
    model.to(device)
    batch = tensor_to_device(tok(args.prompt, return_tensors="pt"), device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**batch, use_cache=True, logits_to_keep=1)
        gen = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(tok, "pad_token_id", None) or 0,
            eos_token_id=getattr(tok, "eos_token_id", None),
        )
    elapsed = time.perf_counter() - t0
    assert out.logits.shape[0] == 1
    assert gen.shape[1] >= batch["input_ids"].shape[1]
    row = {
        "axis": "apple_silicon_hf_model",
        "status": "pass",
        "model": Path(args.model).name,
        "model_size_label": infer_model_size_label(args.model, args.model_size_label),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "prompt_tokens": int(batch["input_ids"].shape[1]),
        "generated_tokens": int(gen.shape[1] - batch["input_ids"].shape[1]),
        "elapsed_s": round(elapsed, 4),
        "backend_class": model.__class__.__name__,
    }
    row.update(mps_memory_stats(torch))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="Converted RWKV-7 HF model dir. Optional for tiny smoke only.")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max-new-tokens", type=int, default=2)
    ap.add_argument("--prompt", default="User: Hello from Apple Silicon.\n\nAssistant:")
    ap.add_argument("--results", default="")
    ap.add_argument("--model-size-label", default="")
    ap.add_argument("--require-apple", action="store_true")
    ap.add_argument("--skip-tiny", action="store_true")
    args = ap.parse_args()

    if not is_apple_silicon():
        row = {
            "axis": "apple_silicon_smoke",
            "status": "skip",
            "reason": "not Darwin/arm64",
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
        print(json.dumps(row, ensure_ascii=False))
        append_result(args.results, row)
        if args.require_apple:
            raise SystemExit(2)
        return 0

    import torch

    device = choose_device(torch, args.device)
    dtype = dtype_for(torch, args.dtype)
    header = {
        "axis": "apple_silicon_env",
        "status": "info",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": darwin_sysctl("machdep.cpu.brand_string"),
        "memory_gb": apple_memory_gb(),
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": package_version("transformers"),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "device": device,
        "dtype": args.dtype,
        "native_model": os.environ.get("RWKV7_NATIVE_MODEL"),
    }
    header.update(mps_memory_stats(torch))
    emit(args.results, header)

    if not args.skip_tiny:
        row = run_tiny_native(torch, device, dtype, args.max_new_tokens)
        emit(args.results, row)
    if args.model:
        row = run_hf_model(torch, args, device, dtype)
        emit(args.results, row)
    print("APPLE SILICON SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
