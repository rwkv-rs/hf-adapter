#!/usr/bin/env python3
# coding=utf-8
"""RWKV-7 speed & memory benchmark — HF adapter vs official `rwkv`.

Decode is measured by threading the recurrent state directly (one token at a
time), NOT via transformers.generate() — that's the true recurrent decode path
and it matches how the official `rwkv` package decodes, so the two backends are
compared fairly. Prefill is one full forward. Peak VRAM covers prefill+decode.

Official runs the pure-torch reference path (RWKV_CUDA_ON unset => no fused
WKV7 kernel, which would need nvcc). That makes official prefill artificially
slow (sequential), but per-token DECODE is a fair same-box comparison.

Usage:
  python bench/bench_speed.py --hf-dir <dir> --pth <.pth> --backend both --dtype fp16
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("RWKV_V7_ON", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 200


def encode(tok, n):
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids
    return ids[:, :n]


def set_attn_mode(model, attn_mode: str) -> None:
    if attn_mode == "auto":
        return
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def bench_hf(args, dt):
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    if args.fast_token_layout != "auto":
        os.environ["RWKV7_FAST_TOKEN_LAYOUT"] = args.fast_token_layout
    if args.fast_token_backend != "auto":
        os.environ["RWKV7_FAST_TOKEN_BACKEND"] = args.fast_token_backend
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, trust_remote_code=True, torch_dtype=dt,
        device_map=args.device).eval()
    set_attn_mode(model, args.attn_mode)
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")
    ids = encode(tok, args.prompt_tokens).to(args.device)
    L = ids.shape[1]

    torch.cuda.reset_peak_memory_stats()
    # prefill
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(ids, use_cache=True, logits_to_keep=args.hf_logits_to_keep)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(args.runs):
            _ = model(ids, use_cache=True, logits_to_keep=args.hf_logits_to_keep)
    torch.cuda.synchronize()
    prefill_tokps = L / ((time.time() - t0) / args.runs)

    # decode via direct state threading
    use_fast_decode = args.hf_decode_api in {"rwkv7_forward_one", "rwkv7_forward_token"}
    if use_fast_decode and not hasattr(model, args.hf_decode_api):
        raise ValueError(f"Loaded model does not expose {args.hf_decode_api}")
    fast_decode_fn = getattr(model, args.hf_decode_api) if use_fast_decode else None

    def decode_step(token, state):
        if use_fast_decode:
            return fast_decode_fn(token, past_key_values=state)
        return model(token, past_key_values=state, use_cache=True, logits_to_keep=args.hf_logits_to_keep)

    with torch.inference_mode():
        out = model(ids[:, :8], use_cache=True, logits_to_keep=args.hf_logits_to_keep)
        state = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.warmup):
            out = decode_step(nxt, state)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(args.decode_tokens):
            out = decode_step(nxt, state)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    dt_decode = time.time() - t0
    res = _res("hf_adapter", args, model, L, prefill_tokps,
               args.decode_tokens / dt_decode,
               torch.cuda.max_memory_allocated() / 1024 / 1024,
               getattr(model.config, "attn_mode", "?"))
    res["hf_logits_to_keep"] = args.hf_logits_to_keep
    res["hf_prefill_use_cache"] = True
    res["fuse_norm"] = getattr(model.config, "fuse_norm", None)
    res["fast_cache"] = os.environ.get("RWKV7_FAST_CACHE", "1") not in {"0", "false", "False", "no", "off"}
    res["cache_type"] = type(state).__name__ if state is not None else None
    res["hf_decode_api"] = args.hf_decode_api
    if use_fast_decode:
        res["fast_token_layout"] = os.environ.get("RWKV7_FAST_TOKEN_LAYOUT", "3d")
        res["fast_token_backend"] = os.environ.get("RWKV7_FAST_TOKEN_BACKEND", "fla")
    return res


def bench_official(args, dt):
    from rwkv.model import RWKV
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    pth = args.pth[:-4] if args.pth.lower().endswith(".pth") else args.pth
    strat = f"{args.device} " + ("fp16" if dt == torch.float16 else
                                 "bf16" if dt == torch.bfloat16 else "fp32")
    m = RWKV(model=pth, strategy=strat)
    id_list = encode(tok, args.prompt_tokens)[0].tolist()
    L = len(id_list)

    torch.cuda.reset_peak_memory_stats()
    # prefill (official torch path is sequential => slow; 1 timed run)
    m.forward(id_list[:8], None)
    t0 = time.time()
    logits = m.forward(id_list, None)
    logits = logits[0] if isinstance(logits, tuple) else logits
    torch.cuda.synchronize()
    prefill_tokps = L / (time.time() - t0)

    # decode via state threading
    logits, state = m.forward(id_list[:8], None)
    for _ in range(args.warmup):
        nt = int(logits.argmax())
        logits, state = m.forward([nt], state)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.decode_tokens):
        nt = int(logits.argmax())
        logits, state = m.forward([nt], state)
    torch.cuda.synchronize()
    dt_decode = time.time() - t0
    return _res("official_rwkv", args, None, L, prefill_tokps,
                args.decode_tokens / dt_decode,
                torch.cuda.max_memory_allocated() / 1024 / 1024,
                "torch_ref(no_fused_kernel)")


def _res(backend, args, model, L, prefill, decode, vram, attn):
    return {
        "axis": "speed_mem", "backend": backend, "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0), "attn_mode": attn,
        "prompt_tokens": L, "decode_tokens": args.decode_tokens,
        "prefill_tokps": round(prefill, 1),
        "decode_tokps": round(decode, 1),
        "decode_ms_per_tok": round(1000 / decode, 2),
        "peak_vram_mb": round(vram, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--pth", default=None)
    ap.add_argument("--backend", default="both", choices=["hf", "official", "both"])
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--hf-logits-to-keep", type=int, default=1,
                    help="HF prefill/decode logits_to_keep; 1 matches serving needs and reduces memory")
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto",
                    help="Override config.fuse_norm for HF load; false is faster on V100 in current tests")
    ap.add_argument("--attn-mode", choices=["auto", "chunk", "fused_recurrent"], default="auto",
                    help="HF attention mode override; fused_recurrent avoids V100 Triton chunk compile instability")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto",
                    help="HF only: use the lightweight RWKV7StateCache hot path (default via model env is enabled)")
    ap.add_argument("--hf-decode-api", choices=["forward", "rwkv7_forward_one", "rwkv7_forward_token"], default="forward",
                    help="HF decode loop implementation; rwkv7_forward_token is the batched inference-only fast path")
    ap.add_argument("--fast-token-layout", choices=["auto", "3d", "2d"], default="auto",
                    help="HF fast-token layout; 3d is the validated baseline, 2d is an experimental A/B path")
    ap.add_argument("--fast-token-backend", choices=["auto", "fla", "native_jit", "native_graph"], default="auto",
                    help="HF fast-token backend; native_graph uses CUDA graph for bsz=1 and falls back to native_jit for batched requests")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"),
                    help="JSONL output path; set empty string to disable appending")
    args = ap.parse_args()
    dt = DTYPES[args.dtype]
    out = Path(args.results) if args.results else None
    results = []
    if args.backend in ("hf", "both"):
        print(f"\n===== backend: hf_adapter ({args.dtype}) =====", flush=True)
        r = bench_hf(args, dt); results.append(r); print(json.dumps(r, indent=2), flush=True)
    if args.backend in ("official", "both"):
        if not args.pth:
            print("--pth required for official backend", flush=True)
        else:
            print(f"\n===== backend: official_rwkv ({args.dtype}) =====", flush=True)
            r = bench_official(args, dt); results.append(r); print(json.dumps(r, indent=2), flush=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nappended {len(results)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
