#!/usr/bin/env python3
# coding=utf-8
"""Benchmark NativeRWKV7ForCausalLM cached decode.

This is intentionally separate from the legacy wrapper fast-token benches. It
tracks the FLA-free eager, JIT, and CUDA-graph paths through the public native
model API, including fixed-batch graph reuse and greedy-token evidence.
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
def native_model_backend(backend: str):
    old = os.environ.get("RWKV7_NATIVE_MODEL_BACKEND")
    os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = backend
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RWKV7_NATIVE_MODEL_BACKEND", None)
        else:
            os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = old


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") else device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def encode(tok, prompt_tokens: int, batch_size: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    ids = ids.repeat(batch_size, 1)
    return ids.to(device) if device.startswith("cuda") else ids


def load_model(args, dtype: torch.dtype):
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.hf_dir,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    return model


def run_backend(args, model, ids: torch.Tensor, *, backend: str) -> dict[str, Any]:
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if hasattr(model, "rwkv7_clear_native_graph_cache"):
        model.rwkv7_clear_native_graph_cache()
    batch_size = int(ids.shape[0])
    greedy_tokens: list[list[int]] = [[] for _ in range(batch_size)]
    with native_model_backend(backend), torch.inference_mode():
        out = model(ids, use_cache=True, logits_to_keep=1)
        state = out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
        first = model(token, past_key_values=state, use_cache=True, logits_to_keep=1)
        first_backend = model.rwkv7_native_model_last_decode_backend()
        first_next = first.logits[:, -1:].argmax(dim=-1)
        state = first.past_key_values
        token = first_next
        for _ in range(args.warmup):
            out = model(token, past_key_values=state, use_cache=True, logits_to_keep=1)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        t0 = time.perf_counter()
        for _ in range(args.decode_steps):
            out = model(token, past_key_values=state, use_cache=True, logits_to_keep=1)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
            token_values = token.reshape(batch_size).detach().cpu().tolist()
            for row, value in enumerate(token_values):
                greedy_tokens[row].append(int(value))
        cuda_sync(args.device)
        dt = time.perf_counter() - t0
    graph_stats = (
        model.rwkv7_native_graph_cache_stats()
        if hasattr(model, "rwkv7_native_graph_cache_stats")
        else None
    )
    graph_overrides = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("RWKV7_NATIVE_GRAPH_", "RWKV7_FUSED_"))
    }
    return {
        "axis": "native_model_decode",
        "backend": "hf_native_model",
        "decode_backend": backend,
        "effective_decode_backend": first_backend,
        "dtype": args.dtype,
        "device": device_name(args.device),
        "batch_size": batch_size,
        "prompt_tokens": int(ids.shape[1]),
        "decode_steps": args.decode_steps,
        "decode_tokps": round(batch_size * args.decode_steps / dt, 2),
        "decode_per_sequence_tokps": round(args.decode_steps / dt, 2),
        "decode_ms_per_tok": round(1000 * dt / max(args.decode_steps, 1), 4),
        "first_next_tokens": [int(value) for value in first_next.reshape(-1).detach().cpu().tolist()],
        "greedy_tokens": greedy_tokens,
        "native_graph_cache": graph_stats,
        "native_graph_overrides": graph_overrides,
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
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    ap.add_argument(
        "--backends",
        nargs="+",
        default=["eager", "native_jit"],
        choices=["eager", "native_jit", "native_graph"],
    )
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, DTYPES[args.dtype])
    rows = []
    batch_sizes = args.batch_sizes or [args.batch_size]
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        ids = encode(tok, args.prompt_tokens, batch_size, args.device)
        for backend in args.backends:
            row = run_backend(args, model, ids, backend=backend)
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
