#!/usr/bin/env python3
# coding=utf-8
"""Profile RWKV-7 one-token decode on HF adapter and official rwkv.

This is the next step after `bench_decode_breakdown.py`: it records the CUDA/CPU
operator hotspots for the slow recurrent decode path. Keep active steps small;
profiler overhead is high and we only need a stable top-op ranking.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 64


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") else device


def _table(prof, sort_by: str, row_limit: int) -> str:
    try:
        return prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    except Exception as exc:  # pragma: no cover - profiler version differences
        return f"<failed to create profiler table sort_by={sort_by}: {exc}>"


def _event_device_total_us(evt) -> float:
    # PyTorch renamed CUDA timing fields to device timing in newer releases.
    return float(
        getattr(evt, "device_time_total", None)
        or getattr(evt, "cuda_time_total", None)
        or 0.0
    )


def _event_self_device_total_us(evt) -> float:
    return float(
        getattr(evt, "self_device_time_total", None)
        or getattr(evt, "self_cuda_time_total", None)
        or 0.0
    )


def _top_json(prof, row_limit: int) -> list[dict[str, Any]]:
    rows = []
    events = sorted(prof.key_averages(), key=_event_device_total_us, reverse=True)
    for evt in events[:row_limit]:
        device_total = _event_device_total_us(evt)
        rows.append({
            "key": evt.key,
            "count": evt.count,
            "device_time_total_us": device_total,
            "device_time_avg_us": device_total / max(evt.count, 1),
            "self_device_time_total_us": _event_self_device_total_us(evt),
            "cpu_time_total_us": float(getattr(evt, "cpu_time_total", 0.0)),
            "self_cpu_time_total_us": float(getattr(evt, "self_cpu_time_total", 0.0)),
        })
    return rows


def _encode(tok, n: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :n]
    return ids.to(device) if device.startswith("cuda") else ids


def profile_hf(args, dtype) -> dict[str, Any]:
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
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
    model.config.attn_mode = args.attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = args.attn_mode
    ids = _encode(tok, args.prompt_tokens, args.device)
    fixed = ids[:, -1:]
    use_fast_decode = args.hf_decode_api in {"rwkv7_forward_one", "rwkv7_forward_token"}
    if use_fast_decode and not hasattr(model, args.hf_decode_api):
        raise ValueError(f"Loaded model does not expose {args.hf_decode_api}")
    fast_decode_fn = getattr(model, args.hf_decode_api) if use_fast_decode else None

    def decode_step(token, state):
        if use_fast_decode:
            return fast_decode_fn(token, past_key_values=state)
        return model(token, past_key_values=state, use_cache=True, logits_to_keep=1)

    with torch.inference_mode():
        out = model(ids[:, :8], use_cache=True, logits_to_keep=1)
        state = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.prewarm):
            token = fixed if args.fixed_token else nxt
            out = decode_step(token, state)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
    _sync(args.device)

    activities = [ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    sched = schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=1)
    steps = args.wait + args.warmup + args.active
    t0 = time.time()
    with profile(
        activities=activities,
        schedule=sched,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as prof:
        with torch.inference_mode():
            for _ in range(steps):
                token = fixed if args.fixed_token else nxt
                out = decode_step(token, state)
                state = out.past_key_values
                nxt = out.logits[:, -1:].argmax(dim=-1)
                prof.step()
    _sync(args.device)
    elapsed = time.time() - t0
    return {
        "axis": "decode_profile",
        "backend": "hf_adapter",
        "dtype": args.dtype,
        "device": _device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in {"0", "false", "False", "no", "off"},
        "cache_type": type(state).__name__ if state is not None else None,
        "hf_decode_api": args.hf_decode_api,
        "fixed_token": args.fixed_token,
        "wait": args.wait,
        "warmup": args.warmup,
        "active": args.active,
        "elapsed_wall_s": round(elapsed, 4),
        "top_cuda": _top_json(prof, args.row_limit),
        "cuda_table": _table(prof, "cuda_time_total", args.row_limit),
        "cpu_table": _table(prof, "cpu_time_total", args.row_limit),
    }


def profile_official(args, dtype) -> dict[str, Any]:
    if not args.pth:
        raise ValueError("--pth is required for official backend")
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    from rwkv.model import RWKV
    pth = args.pth[:-4] if args.pth.lower().endswith(".pth") else args.pth
    strat_dtype = "fp16" if dtype is torch.float16 else "bf16" if dtype is torch.bfloat16 else "fp32"
    model = RWKV(model=pth, strategy=f"{args.device} {strat_dtype}")
    ids = _encode(tok, args.prompt_tokens, "cpu")[0].tolist()
    logits, state = model.forward(ids[:8], None)
    fixed = ids[-1]
    for _ in range(args.prewarm):
        nt = fixed if args.fixed_token else int(logits.argmax())
        logits, state = model.forward([nt], state)
    _sync(args.device)

    activities = [ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    sched = schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=1)
    steps = args.wait + args.warmup + args.active
    t0 = time.time()
    with profile(
        activities=activities,
        schedule=sched,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as prof:
        for _ in range(steps):
            nt = fixed if args.fixed_token else int(logits.argmax())
            logits, state = model.forward([nt], state)
            prof.step()
    _sync(args.device)
    elapsed = time.time() - t0
    return {
        "axis": "decode_profile",
        "backend": "official_rwkv",
        "dtype": args.dtype,
        "device": _device_name(args.device),
        "attn_mode": "rwkv_package",
        "fixed_token": args.fixed_token,
        "wait": args.wait,
        "warmup": args.warmup,
        "active": args.active,
        "elapsed_wall_s": round(elapsed, 4),
        "top_cuda": _top_json(prof, args.row_limit),
        "cuda_table": _table(prof, "cuda_time_total", args.row_limit),
        "cpu_table": _table(prof, "cpu_time_total", args.row_limit),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--pth", default=None)
    ap.add_argument("--backend", default="hf", choices=["hf", "official"])
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="chunk", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto",
                    help="Override config.fuse_norm for HF load; false is faster on V100 in current tests")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto",
                    help="HF only: use the lightweight RWKV7StateCache hot path (default via model env is enabled)")
    ap.add_argument("--hf-decode-api", choices=["forward", "rwkv7_forward_one", "rwkv7_forward_token"], default="forward",
                    help="HF decode implementation to profile; rwkv7_forward_token is the batched inference-only fast path")
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--prewarm", type=int, default=8)
    ap.add_argument("--wait", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--active", type=int, default=8)
    ap.add_argument("--row-limit", type=int, default=20)
    ap.add_argument("--fixed-token", action="store_true", help="Remove argmax/sampling from profiled loop")
    ap.add_argument("--record-shapes", action="store_true")
    ap.add_argument("--profile-memory", action="store_true")
    ap.add_argument("--out", default=None, help="Optional JSON file for the profile summary")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.backend == "hf":
        result = profile_hf(args, dtype)
    else:
        result = profile_official(args, dtype)

    print("\n=== CUDA time table ===")
    print(result["cuda_table"])
    print("\n=== CPU time table ===")
    print(result["cpu_table"])
    print("\n=== summary ===")
    summary = {k: v for k, v in result.items() if k not in {"cuda_table", "cpu_table"}}
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
