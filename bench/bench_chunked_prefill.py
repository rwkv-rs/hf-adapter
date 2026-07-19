#!/usr/bin/env python3
# coding=utf-8
"""Benchmark RWKV-7 full prefill vs HF chunked prefill helper."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 512


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name()
    return device


def reset_peak(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024**2, 1)


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def timed(fn, warmup: int, runs: int, device: str) -> tuple[float, Any]:
    out = None
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
        cuda_sync(device)
        t0 = time.perf_counter()
        for _ in range(runs):
            out = fn()
        cuda_sync(device)
    return (time.perf_counter() - t0) / max(1, runs), out


def append(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-diff", type=float, default=0.15)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=DTYPES[args.dtype],
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    set_attn_mode(model, args.attn_mode)
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")
    if not hasattr(model, "rwkv7_prefill_chunks"):
        raise AttributeError("Model does not expose rwkv7_prefill_chunks")

    enc = tok(SEED, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[:, : args.prompt_tokens].repeat(args.batch_size, 1)
    if args.device.startswith("cuda"):
        ids = ids.cuda()
    total_tokens = int(ids.numel())

    def full_prefill():
        return model(ids, use_cache=True, logits_to_keep=1)

    reset_peak(args.device)
    full_dt, full_out = timed(full_prefill, args.warmup, args.runs, args.device)
    full_peak = peak_mb(args.device)
    rows: list[dict[str, Any]] = []
    full_tokps = total_tokens / full_dt
    rows.append(
        {
            "axis": "chunked_prefill",
            "backend": "hf_adapter",
            "prefill_mode": "full",
            "dtype": args.dtype,
            "device": device_name(args.device),
            "attn_mode": args.attn_mode,
            "fuse_norm": getattr(model.config, "fuse_norm", None),
            "batch_size": args.batch_size,
            "prompt_tokens": int(ids.shape[1]),
            "total_tokens": total_tokens,
            "prefill_ms": round(full_dt * 1000, 2),
            "prefill_tokps_total": round(full_tokps, 1),
            "peak_vram_mb": full_peak,
            "seq_length": full_out.past_key_values.get_seq_length(),
        }
    )

    full_seq_length = full_out.past_key_values.get_seq_length()
    next_token = full_out.logits[:, -1:].argmax(dim=-1)
    # ``timed`` returns inference tensors.  Keep all cache-continuation calls in
    # inference mode as well, otherwise PyTorch may try to save those tensors
    # for backward and reject them before the benchmark starts.
    with torch.inference_mode():
        full_next = model(next_token, past_key_values=full_out.past_key_values, use_cache=True, logits_to_keep=1)
    for chunk_size in args.chunk_sizes:
        def chunked_prefill():
            return model.rwkv7_prefill_chunks(ids, chunk_size=chunk_size, logits_to_keep=1)

        reset_peak(args.device)
        dt, out = timed(chunked_prefill, args.warmup, args.runs, args.device)
        peak = peak_mb(args.device)
        diff = float((full_out.logits.float() - out.logits.float()).abs().max().detach().cpu())
        chunk_seq_length = out.past_key_values.get_seq_length()
        seq_match = full_seq_length == chunk_seq_length
        with torch.inference_mode():
            chunk_next = model(next_token, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
        decode_diff = float((full_next.logits.float() - chunk_next.logits.float()).abs().max().detach().cpu())
        row = {
            "axis": "chunked_prefill",
            "backend": "hf_adapter",
            "prefill_mode": "chunked",
            "dtype": args.dtype,
            "device": device_name(args.device),
            "attn_mode": args.attn_mode,
            "fuse_norm": getattr(model.config, "fuse_norm", None),
            "batch_size": args.batch_size,
            "prompt_tokens": int(ids.shape[1]),
            "total_tokens": total_tokens,
            "chunk_size": int(chunk_size),
            "prefill_ms": round(dt * 1000, 2),
            "prefill_tokps_total": round(total_tokens / dt, 1),
            "speed_ratio_vs_full": round((total_tokens / dt) / full_tokps, 4) if full_tokps else None,
            "peak_vram_mb": peak,
            "peak_vram_ratio_vs_full": round(peak / full_peak, 4) if peak is not None and full_peak else None,
            "max_abs_diff": round(diff, 6),
            "decode_max_abs_diff": round(decode_diff, 6),
            "seq_length_match": bool(seq_match),
            "seq_length": chunk_seq_length,
        }
        rows.append(row)
        if diff > args.max_diff or decode_diff > args.max_diff or not seq_match:
            raise AssertionError(f"chunk_size={chunk_size} failed: {row}")

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
        append(args.results, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
