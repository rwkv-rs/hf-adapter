#!/usr/bin/env python3
# coding=utf-8
"""Benchmark experimental NativeRWKV7ForCausalLM cached decode.

This is intentionally separate from the production wrapper fast-token benches.
It tracks the FLA-free native PyTorch fallback path and its optional
``RWKV7_NATIVE_MODEL_JIT`` cached-decode acceleration so native/upstream/AMD work
has a reproducible speed row without claiming this path replaces the wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "User: Summarize recurrent neural networks and cache reuse.\n\nAssistant:" * 16


@contextmanager
def native_model_jit(enabled: bool):
    old = os.environ.get("RWKV7_NATIVE_MODEL_JIT")
    os.environ["RWKV7_NATIVE_MODEL_JIT"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RWKV7_NATIVE_MODEL_JIT", None)
        else:
            os.environ["RWKV7_NATIVE_MODEL_JIT"] = old


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") else device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def encode(tok, prompt_tokens: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    return ids.to(device) if device.startswith("cuda") else ids


def load_model(args, dtype: torch.dtype):
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.hf_dir,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    return model


def run_backend(args, model, ids: torch.Tensor, *, jit_enabled: bool) -> dict[str, Any]:
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with native_model_jit(jit_enabled), torch.inference_mode():
        out = model(ids, use_cache=True)
        state = out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
        first = model(token, past_key_values=state, use_cache=True)
        first_backend = model.rwkv7_native_model_last_decode_backend()
        first_next = first.logits[:, -1:].argmax(dim=-1)
        state = first.past_key_values
        token = first_next
        for _ in range(args.warmup):
            out = model(token, past_key_values=state, use_cache=True)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        t0 = time.perf_counter()
        for _ in range(args.decode_steps):
            out = model(token, past_key_values=state, use_cache=True)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        dt = time.perf_counter() - t0
    return {
        "axis": "native_model_decode",
        "backend": "hf_native_model",
        "decode_backend": "native_jit" if jit_enabled else "eager",
        "effective_decode_backend": first_backend,
        "dtype": args.dtype,
        "device": device_name(args.device),
        "prompt_tokens": int(ids.shape[1]),
        "decode_steps": args.decode_steps,
        "decode_tokps": round(args.decode_steps / dt, 2),
        "decode_ms_per_tok": round(1000 * dt / max(args.decode_steps, 1), 4),
        "first_next_token": int(first_next.reshape(-1)[0].detach().cpu()),
        "peak_vram_mb": peak_mb(args.device),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp32", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--backends", nargs="+", default=["eager", "native_jit"], choices=["eager", "native_jit"])
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, DTYPES[args.dtype])
    ids = encode(tok, args.prompt_tokens, args.device)
    rows = []
    for backend in args.backends:
        row = run_backend(args, model, ids, jit_enabled=backend == "native_jit")
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
