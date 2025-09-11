#!/usr/bin/env python3
# coding=utf-8
"""HF single-call TTFT/TPOT benchmark for RWKV-7.

HF scope is one `model()` / `model.generate()` call and batch-generate behavior,
not multi-user serving concurrency. This script records:

* TTFT: one prefill call with `use_cache=True` and `logits_to_keep=1`.
* TPOT: cached one-token decode latency, bsz=1, carrying RWKV recurrent state.
* Batch generate: optional `model.generate()` throughput for common batch sizes.

Rows are JSON-serializable so they can be appended to `bench/results.jsonl` and
consumed by the normal analyzer/gates.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from transformers import AutoModelForCausalLM

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return float(ordered[idx])


def device_name(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(torch.device(device).index or 0)
    return device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def append_result(path: str | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model(args):
    dtype = DTYPES[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")
    if args.attn_mode:
        model.config.attn_mode = args.attn_mode
        for layer in getattr(model.model, "layers", []):
            attn = getattr(layer, "attn", None)
            if hasattr(attn, "mode"):
                attn.mode = args.attn_mode
    return model


def random_ids(vocab_size: int, batch_size: int, seq_len: int, device: str) -> torch.Tensor:
    ids = torch.randint(0, int(vocab_size), (int(batch_size), int(seq_len)))
    return ids.to(device) if device.startswith("cuda") else ids


def bench_ttft(args, model) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vocab = int(model.config.vocab_size)
    for isl in args.isl:
        ids = random_ids(vocab, 1, isl, args.device)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(ids, use_cache=True, logits_to_keep=1)
            cuda_sync(args.device)
            times = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                model(ids, use_cache=True, logits_to_keep=1)
                cuda_sync(args.device)
                times.append((time.perf_counter() - t0) * 1000.0)
        row = {
            "axis": "ttft_tpot",
            "metric": "ttft",
            "backend": "hf_adapter",
            "dtype": args.dtype,
            "device": device_name(args.device),
            "attn_mode": args.attn_mode,
            "fuse_norm": getattr(model.config, "fuse_norm", None),
            "fast_forward_env": os.environ.get("RWKV7_FAST_FORWARD", "1"),
            "prompt_tokens": int(isl),
            "batch_size": 1,
            "reps": int(args.reps),
            "p50_ms": round(percentile(times, 0.50), 4),
            "p99_ms": round(percentile(times, 0.99), 4),
            "mean_ms": round(mean(times), 4),
            "prefill_tokps_p50": round(float(isl) / max(percentile(times, 0.50), 1e-9) * 1000.0, 1),
            "peak_vram_mb": peak_mb(args.device),
            "status": "pass",
        }
        rows.append(row)
    return rows


def bench_tpot(args, model) -> dict[str, Any]:
    vocab = int(model.config.vocab_size)
    seed = random_ids(vocab, 1, args.tpot_seed_tokens, args.device)
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        out = model(seed, use_cache=True, logits_to_keep=1)
        state = out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.warmup):
            out = model(token, past_key_values=state, use_cache=True, logits_to_keep=1)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        times = []
        for _ in range(args.decode_tokens):
            t0 = time.perf_counter()
            out = model(token, past_key_values=state, use_cache=True, logits_to_keep=1)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
            cuda_sync(args.device)
            times.append((time.perf_counter() - t0) * 1000.0)
    p50 = percentile(times, 0.50)
    return {
        "axis": "ttft_tpot",
        "metric": "tpot",
        "backend": "hf_adapter",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_forward_env": os.environ.get("RWKV7_FAST_FORWARD", "1"),
        "fast_token_backend_effective": getattr(model, "rwkv7_last_fast_token_backend", lambda: None)(),
        "prompt_tokens": int(args.tpot_seed_tokens),
        "decode_tokens": int(args.decode_tokens),
        "batch_size": 1,
        "p50_ms": round(p50, 4),
        "p99_ms": round(percentile(times, 0.99), 4),
        "mean_ms": round(mean(times), 4),
        "decode_tokps_p50": round(1000.0 / max(p50, 1e-9), 1),
        "peak_vram_mb": peak_mb(args.device),
        "status": "pass",
    }


def bench_batch_generate(args, model) -> list[dict[str, Any]]:
    if args.generate_tokens <= 0:
        return []
    rows: list[dict[str, Any]] = []
    vocab = int(model.config.vocab_size)
    for batch_size in args.batch_sizes:
        ids = random_ids(vocab, batch_size, args.generate_prompt_tokens, args.device)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range(args.generate_warmup):
                model.generate(ids, max_new_tokens=args.generate_tokens, do_sample=False, use_cache=True)
            cuda_sync(args.device)
            t0 = time.perf_counter()
            out = model.generate(ids, max_new_tokens=args.generate_tokens, do_sample=False, use_cache=True)
            cuda_sync(args.device)
            elapsed = time.perf_counter() - t0
        generated = max(0, int(out.shape[1]) - int(ids.shape[1])) * int(batch_size)
        rows.append(
            {
                "axis": "ttft_tpot",
                "metric": "batch_generate",
                "backend": "hf_adapter",
                "dtype": args.dtype,
                "device": device_name(args.device),
                "attn_mode": args.attn_mode,
                "fuse_norm": getattr(model.config, "fuse_norm", None),
                "fast_forward_env": os.environ.get("RWKV7_FAST_FORWARD", "1"),
                "fast_token_backend_effective": getattr(model, "rwkv7_last_fast_token_backend", lambda: None)(),
                "prompt_tokens": int(args.generate_prompt_tokens),
                "max_new_tokens": int(args.generate_tokens),
                "batch_size": int(batch_size),
                "generated_tokens": int(generated),
                "elapsed_s": round(elapsed, 4),
                "generate_tokps": round(float(generated) / max(elapsed, 1e-9), 1),
                "peak_vram_mb": peak_mb(args.device),
                "status": "pass",
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--isl", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tpot-seed-tokens", type=int, default=64)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--generate-prompt-tokens", type=int, default=128)
    ap.add_argument("--generate-tokens", type=int, default=32)
    ap.add_argument("--generate-warmup", type=int, default=1)
    ap.add_argument("--results", default=None, help="Optional JSONL file to append rows to")
    args = ap.parse_args()

    model = load_model(args)
    rows = [*bench_ttft(args, model), bench_tpot(args, model), *bench_batch_generate(args, model)]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
        append_result(args.results, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
