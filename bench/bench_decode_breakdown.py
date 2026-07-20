#!/usr/bin/env python3
# coding=utf-8
"""RWKV-7 decode performance breakdown.

The existing `bench_speed.py` gives the high-level HF-vs-official numbers. This
script narrows the decode bottleneck by measuring:

- HF prefill with `logits_to_keep=1` vs full logits.
- HF recurrent decode with greedy argmax vs fixed-token decode.
- HF `chunk` config vs `fused_recurrent` config.
- Official `rwkv` decode on the same prompt.

The fixed-token decode path removes sampling/argmax overhead, so the difference
between greedy and fixed-token estimates the Python/sampling part of decode. The
remaining gap vs official points at model/cache/kernel overhead.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 256


def cuda_sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def encode(tok, n: int, device: str):
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :n]
    return ids.to(device) if device.startswith("cuda") else ids


def timed(fn, device: str, runs: int = 1) -> float:
    cuda_sync(device)
    t0 = time.time()
    for _ in range(runs):
        fn()
    cuda_sync(device)
    return (time.time() - t0) / runs


@contextmanager
def reference_forward_env():
    old = os.environ.get("RWKV7_FAST_FORWARD")
    old_native_backend = os.environ.get("RWKV7_NATIVE_MODEL_BACKEND")
    os.environ["RWKV7_FAST_FORWARD"] = "0"
    os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = "eager"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RWKV7_FAST_FORWARD", None)
        else:
            os.environ["RWKV7_FAST_FORWARD"] = old
        if old_native_backend is None:
            os.environ.pop("RWKV7_NATIVE_MODEL_BACKEND", None)
        else:
            os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = old_native_backend


def load_hf(args, dtype, attn_mode: str):
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode
    return model


def bench_hf_variant(args, tok, dtype, attn_mode: str) -> dict[str, Any]:
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_hf(args, dtype, attn_mode)
    ids = encode(tok, args.prompt_tokens, args.device)
    fixed = ids[:, -1:]

    # Warmup both full and cached paths.
    with torch.no_grad():
        _ = model(ids[:, : min(32, ids.shape[1])], use_cache=True, logits_to_keep=1)
        for _ in range(args.warmup):
            _ = model(ids, use_cache=True, logits_to_keep=1)
    cuda_sync(args.device)

    with torch.no_grad():
        dt_prefill_keep1 = timed(lambda: model(ids, use_cache=True, logits_to_keep=1), args.device, args.runs)
        dt_prefill_full = timed(lambda: model(ids, use_cache=False), args.device, max(1, args.runs // 2))

        out = model(ids[:, :8], use_cache=True, logits_to_keep=1)
        state = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.warmup):
            with reference_forward_env():
                out = model(nxt, past_key_values=state, use_cache=True, logits_to_keep=1)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        t0 = time.time()
        for _ in range(args.decode_tokens):
            with reference_forward_env():
                out = model(nxt, past_key_values=state, use_cache=True, logits_to_keep=1)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        greedy_dt = time.time() - t0

        out = model(ids[:, :8], use_cache=True, logits_to_keep=1)
        state = out.past_key_values
        for _ in range(args.warmup):
            with reference_forward_env():
                out = model(fixed, past_key_values=state, use_cache=True, logits_to_keep=1)
            state = out.past_key_values
        cuda_sync(args.device)
        t0 = time.time()
        for _ in range(args.decode_tokens):
            with reference_forward_env():
                out = model(fixed, past_key_values=state, use_cache=True, logits_to_keep=1)
            state = out.past_key_values
        cuda_sync(args.device)
        fixed_dt = time.time() - t0

        fast_greedy_dt = None
        fast_fixed_dt = None
        fast_fn = getattr(model, "rwkv7_forward_token", None)
        fast_name = "rwkv7_forward_token" if fast_fn is not None else None
        if fast_fn is None:
            fast_fn = getattr(model, "rwkv7_forward_one", None)
            fast_name = "rwkv7_forward_one" if fast_fn is not None else None
        run_fast_decode = args.fast_decode_api != "false" and fast_fn is not None
        if args.fast_decode_api == "true" and fast_fn is None:
            raise ValueError("Loaded model does not expose a fast one-token decode API")
        if run_fast_decode:
            out = model(ids[:, :8], use_cache=True, logits_to_keep=1)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
            for _ in range(args.warmup):
                out = fast_fn(nxt, past_key_values=state)
                state = out.past_key_values
                nxt = out.logits[:, -1:].argmax(dim=-1)
            cuda_sync(args.device)
            t0 = time.time()
            for _ in range(args.decode_tokens):
                out = fast_fn(nxt, past_key_values=state)
                state = out.past_key_values
                nxt = out.logits[:, -1:].argmax(dim=-1)
            cuda_sync(args.device)
            fast_greedy_dt = time.time() - t0

            out = model(ids[:, :8], use_cache=True, logits_to_keep=1)
            state = out.past_key_values
            for _ in range(args.warmup):
                out = fast_fn(fixed, past_key_values=state)
                state = out.past_key_values
            cuda_sync(args.device)
            t0 = time.time()
            for _ in range(args.decode_tokens):
                out = fast_fn(fixed, past_key_values=state)
                state = out.past_key_values
            cuda_sync(args.device)
            fast_fixed_dt = time.time() - t0

    result = {
        "axis": "decode_breakdown",
        "backend": "hf_adapter",
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "attn_mode": attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in {"0", "false", "False", "no", "off"},
        "cache_type": type(state).__name__ if state is not None else None,
        "prompt_tokens": int(ids.shape[1]),
        "decode_tokens": args.decode_tokens,
        "prefill_keep1_tokps": round(int(ids.shape[1]) / dt_prefill_keep1, 1),
        "prefill_full_tokps": round(int(ids.shape[1]) / dt_prefill_full, 1),
        "prefill_keep1_ms": round(dt_prefill_keep1 * 1000, 2),
        "prefill_full_ms": round(dt_prefill_full * 1000, 2),
        "decode_greedy_tokps": round(args.decode_tokens / greedy_dt, 1),
        "decode_fixed_tokps": round(args.decode_tokens / fixed_dt, 1),
        "decode_greedy_ms_per_tok": round(1000 * greedy_dt / args.decode_tokens, 2),
        "decode_fixed_ms_per_tok": round(1000 * fixed_dt / args.decode_tokens, 2),
        "argmax_sampling_overhead_ms_per_tok": round(1000 * (greedy_dt - fixed_dt) / args.decode_tokens, 2),
        "fast_decode_api": bool(fast_greedy_dt is not None),
        "fast_decode_api_name": fast_name if fast_greedy_dt is not None else None,
        "decode_fast_api_greedy_tokps": round(args.decode_tokens / fast_greedy_dt, 1) if fast_greedy_dt else None,
        "decode_fast_api_fixed_tokps": round(args.decode_tokens / fast_fixed_dt, 1) if fast_fixed_dt else None,
        "decode_fast_api_greedy_ms_per_tok": round(1000 * fast_greedy_dt / args.decode_tokens, 2) if fast_greedy_dt else None,
        "decode_fast_api_fixed_ms_per_tok": round(1000 * fast_fixed_dt / args.decode_tokens, 2) if fast_fixed_dt else None,
        "peak_vram_mb": peak_mb(args.device),
    }
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def bench_official(args, tok, dtype) -> dict[str, Any] | None:
    if not args.pth:
        return None
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    from rwkv.model import RWKV
    pth = args.pth[:-4] if args.pth.lower().endswith(".pth") else args.pth
    strategy_dtype = "fp16" if dtype is torch.float16 else "bf16" if dtype is torch.bfloat16 else "fp32"
    model = RWKV(model=pth, strategy=f"{args.device} {strategy_dtype}")
    id_list = encode(tok, args.prompt_tokens, "cpu")[0].tolist()

    # Prefill and decode match bench_speed.py semantics.
    model.forward(id_list[:8], None)
    cuda_sync(args.device)
    t0 = time.time()
    logits, state = model.forward(id_list, None)
    cuda_sync(args.device)
    prefill_dt = time.time() - t0

    logits, state = model.forward(id_list[:8], None)
    for _ in range(args.warmup):
        nt = int(logits.argmax())
        logits, state = model.forward([nt], state)
    cuda_sync(args.device)
    t0 = time.time()
    for _ in range(args.decode_tokens):
        nt = int(logits.argmax())
        logits, state = model.forward([nt], state)
    cuda_sync(args.device)
    decode_dt = time.time() - t0
    return {
        "axis": "decode_breakdown",
        "backend": "official_rwkv",
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "attn_mode": "rwkv_package",
        "prompt_tokens": len(id_list),
        "decode_tokens": args.decode_tokens,
        "prefill_tokps": round(len(id_list) / prefill_dt, 1),
        "prefill_ms": round(prefill_dt * 1000, 2),
        "decode_tokps": round(args.decode_tokens / decode_dt, 1),
        "decode_ms_per_tok": round(1000 * decode_dt / args.decode_tokens, 2),
        "peak_vram_mb": peak_mb(args.device),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--pth", default=None)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--attn-modes", nargs="+", default=["chunk", "fused_recurrent"], choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto",
                    help="Override config.fuse_norm for HF load; false is faster on V100 in current tests")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto",
                    help="HF only: use the lightweight RWKV7StateCache hot path (default via model env is enabled)")
    ap.add_argument("--fast-decode-api", choices=["auto", "true", "false"], default="auto",
                    help="Also benchmark rwkv7_forward_token/rwkv7_forward_one when the loaded model exposes it")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    results = []
    for mode in args.attn_modes:
        print(f"\n===== HF {mode} =====", flush=True)
        r = bench_hf_variant(args, tok, dtype, mode)
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)
    if args.pth:
        print("\n===== official rwkv =====", flush=True)
        r = bench_official(args, tok, dtype)
        if r is not None:
            results.append(r)
            print(json.dumps(r, indent=2), flush=True)

    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nappended {len(results)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
