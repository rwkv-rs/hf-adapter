#!/usr/bin/env python3
# coding=utf-8
"""End-to-end decode benchmark for native mm8/mm4 quantized RWKV-7 heads.

This measures the repository-native quantization paths from
``native_quant_mm8.py`` / ``native_quant_mm4.py`` after applying the size-gated
``quantize_model_mm*`` replacement to a loaded HF model. It is intentionally
separate from ``bench_quantization.py``, which measures bitsandbytes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 200


def infer_model_size_label(hf_dir: str, explicit: str = "") -> str | None:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(hf_dir).name.lower())
    return match.group(1) if match else None


def device_map_for(device: str):
    if not device.startswith("cuda"):
        return None
    if ":" in device:
        return {"": int(device.split(":", 1)[1])}
    return {"": 0}


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def encode(tok, n: int) -> torch.LongTensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids
    return ids[:, :n]


def model_metadata(args: argparse.Namespace, model=None) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "model_name": Path(args.hf_dir).name,
        "model_size_label": infer_model_size_label(args.hf_dir, args.model_size_label),
        "hf_model_dir": args.hf_dir,
        "hidden_size": getattr(cfg, "hidden_size", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "head_dim": getattr(cfg, "head_dim", None),
        "num_heads": getattr(cfg, "num_heads", None),
    }


def module_counts(model) -> dict[str, int]:
    counts = {"MM8Linear": 0, "MM4Linear": 0, "Linear": 0}
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name in counts:
            counts[cls_name] += 1
        elif type(module) is torch.nn.Linear:
            counts["Linear"] += 1
    return counts


def apply_native_quant(model, quantization: str, min_params: int) -> int:
    if quantization == "none":
        return 0
    if quantization == "mm8":
        from rwkv7_hf.native_quant_mm8 import quantize_model_mm8

        return int(quantize_model_mm8(model, min_params=min_params))
    if quantization == "mm4":
        from rwkv7_hf.native_quant_mm4 import quantize_model_mm4

        return int(quantize_model_mm4(model, min_params=min_params))
    raise ValueError(f"unsupported quantization: {quantization}")


def skip_row(args: argparse.Namespace, quantization: str, exc: BaseException) -> dict[str, Any]:
    return {
        "axis": "native_mm_quantization",
        "backend": "hf_adapter",
        "quantization": f"native_{quantization}" if quantization != "none" else "none",
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else args.device,
        **model_metadata(args),
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "native_mm_min_params": args.min_params,
        "status": "skip",
        "error": repr(exc),
    }


def bench_one(args: argparse.Namespace, tok, quantization: str, dtype: torch.dtype) -> dict[str, Any]:
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map_for(args.device) if args.device.startswith("cuda") else None,
    ).eval()
    replaced = apply_native_quant(model, quantization, args.min_params)
    cuda_sync(args.device)
    load_s = time.time() - t0

    input_device = next(model.parameters()).device
    ids = encode(tok, args.prompt_tokens).to(input_device)
    prompt_tokens = int(ids.shape[1])

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(ids, use_cache=True)
    cuda_sync(args.device)
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(args.runs):
            out = model(ids, use_cache=True)
    cuda_sync(args.device)
    prefill_tokps = prompt_tokens / ((time.time() - t0) / args.runs)

    with torch.inference_mode():
        out = model(ids[:, :8], use_cache=True)
        state = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.warmup):
            step = model(nxt, past_key_values=state, use_cache=True)
            state = step.past_key_values
            nxt = step.logits[:, -1:].argmax(dim=-1)
    cuda_sync(args.device)
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(args.decode_tokens):
            step = model(nxt, past_key_values=state, use_cache=True)
            state = step.past_key_values
            nxt = step.logits[:, -1:].argmax(dim=-1)
    cuda_sync(args.device)
    decode_s = time.time() - t0

    footprint_mb = None
    if hasattr(model, "get_memory_footprint"):
        footprint_mb = round(float(model.get_memory_footprint()) / 1024 / 1024, 1)

    return {
        "axis": "native_mm_quantization",
        "backend": "hf_adapter",
        "quantization": f"native_{quantization}" if quantization != "none" else "none",
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else args.device,
        **model_metadata(args, model),
        "prompt_tokens": prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "prefill_tokps": round(prefill_tokps, 1),
        "decode_tokps": round(args.decode_tokens / decode_s, 1),
        "decode_ms_per_tok": round(1000 * decode_s / args.decode_tokens, 2),
        "native_mm_min_params": args.min_params,
        "native_mm_replaced_modules": replaced,
        "module_counts": module_counts(model),
        "model_footprint_mb": footprint_mb,
        "peak_vram_mb": peak_mb(args.device),
        "load_s": round(load_s, 3),
        "cache_type": type(state).__name__ if state is not None else None,
        "status": "pass",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--model-size-label", default="")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quantizations", nargs="+", choices=["none", "mm8", "mm4"], default=["mm8", "mm4"])
    ap.add_argument("--min-params", type=int, default=8_000_000)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--optional", action="store_true", help="Append skip rows instead of failing when a native quant path is unavailable")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    rows: list[dict[str, Any]] = []
    had_error = False
    for quantization in args.quantizations:
        print(f"\n===== native quantization: {quantization} =====", flush=True)
        try:
            row = bench_one(args, tok, quantization, dtype)
        except Exception as exc:
            if not args.optional:
                raise
            had_error = True
            row = skip_row(args, quantization, exc)
        rows.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended {len(rows)} rows -> {out}", flush=True)
    return 1 if had_error and not args.optional else 0


if __name__ == "__main__":
    raise SystemExit(main())
