#!/usr/bin/env python3
# coding=utf-8
"""A/B benchmark for native_graph with fused attention output prep enabled.

`bench_fused_attn_output.py` proves the isolated group-norm/correction/gate
output-prep kernel is fast. This script checks the production-facing question:
after capture inside the HF native_graph fast-token backend, does enabling
`RWKV7_NATIVE_GRAPH_FUSED_OUTPUT=1` preserve logits/greedy behavior and move
end-to-end decode latency?
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV7_FAST_TOKEN_BACKEND", "native_graph")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 128
_FALSE_VALUES = {"0", "false", "False", "no", "off"}


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") else device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def load_model(args: argparse.Namespace, dtype: torch.dtype):
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    os.environ["RWKV7_FAST_TOKEN_BACKEND"] = "native_graph"
    os.environ["RWKV7_NATIVE_GRAPH_CACHE_SIZE"] = str(args.native_graph_cache_size)
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
    set_attn_mode(model, args.attn_mode)
    for name in ("rwkv7_forward_token", "rwkv7_clear_native_graph_cache"):
        if not hasattr(model, name):
            raise ValueError(f"Loaded model does not expose {name}")
    return model


def encode(tok, prompt_tokens: int, batch_size: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    ids = ids.repeat(batch_size, 1)
    return ids.to(device) if device.startswith("cuda") else ids


def prefill(model, ids: torch.Tensor):
    out = model(ids, use_cache=True, logits_to_keep=1)
    token = out.logits[:, -1:].argmax(dim=-1)
    return token, out.past_key_values


def run_mode(model, token: torch.Tensor, base_state, args: argparse.Namespace, *, enabled: bool) -> dict[str, Any]:
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT"] = "1" if args.fused_recurrent else "0"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_OUTPUT"] = "1" if enabled else "0"
    model.rwkv7_clear_native_graph_cache()
    if hasattr(model, "rwkv7_reset_native_graph_cache_stats"):
        model.rwkv7_reset_native_graph_cache_stats()
    state = base_state.clone()
    tok = token.clone()
    with torch.inference_mode():
        first = model.rwkv7_forward_token(tok, past_key_values=state)
        effective_backend = getattr(model, "rwkv7_last_fast_token_backend", lambda: None)()
        # Warmup from a fresh clone so correctness logits remain first-token.
        state = base_state.clone()
        tok = token.clone()
        for _ in range(args.warmup):
            out = model.rwkv7_forward_token(tok, past_key_values=state)
            if not args.fixed_token:
                tok = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        t0 = time.perf_counter()
        greedy_tokens: list[int] = []
        for _ in range(args.steps):
            out = model.rwkv7_forward_token(tok, past_key_values=state)
            if not args.fixed_token:
                tok = out.logits[:, -1:].argmax(dim=-1)
            else:
                tok = token
            greedy_tokens.extend(int(v) for v in out.logits[:, -1, :].argmax(dim=-1).detach().cpu().reshape(-1))
        cuda_sync(args.device)
    ms_per_step = (time.perf_counter() - t0) * 1000.0 / float(args.steps)
    stats = model.rwkv7_native_graph_cache_stats() if hasattr(model, "rwkv7_native_graph_cache_stats") else {}
    return {
        "enabled": enabled,
        "effective_backend": effective_backend,
        "first_logits": first.logits.detach().clone(),
        "ms_per_step": ms_per_step,
        "tokps_total": 1000.0 * int(token.numel()) / ms_per_step if ms_per_step > 0 else None,
        "greedy_tokens": greedy_tokens,
        "cache_stats": stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--fixed-token", action="store_true")
    ap.add_argument("--native-graph-cache-size", type=int, default=8)
    ap.add_argument("--fused-recurrent", action="store_true", help="Keep RWKV7_NATIVE_GRAPH_FUSED_RECURRENT=1 in both A/B modes to test combined output integration.")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, dtype)
    ids = encode(tok, args.prompt_tokens, args.batch_size, args.device)
    with torch.inference_mode():
        # Use the normal path for prefill, then A/B only the captured decode graph.
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT"] = "1" if args.fused_recurrent else "0"
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_OUTPUT"] = "0"
        token, base_state = prefill(model, ids)
        baseline = run_mode(model, token, base_state, args, enabled=False)
        fused = run_mode(model, token, base_state, args, enabled=True)

    max_abs = float((baseline["first_logits"].float() - fused["first_logits"].float()).abs().max().detach().cpu())
    cosine = float(torch.nn.functional.cosine_similarity(
        baseline["first_logits"].float().reshape(args.batch_size, -1),
        fused["first_logits"].float().reshape(args.batch_size, -1),
        dim=-1,
    ).min().detach().cpu())
    greedy_total = min(len(baseline["greedy_tokens"]), len(fused["greedy_tokens"]))
    greedy_match = sum(int(a == b) for a, b in zip(baseline["greedy_tokens"], fused["greedy_tokens"], strict=False))
    row = {
        "axis": "native_graph_fused_output",
        "backend": "hf_adapter",
        "status": "pass",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES,
        "batch_size": args.batch_size,
        "prompt_tokens": int(ids.shape[1]),
        "steps": args.steps,
        "fixed_token": args.fixed_token,
        "fused_recurrent_enabled": bool(args.fused_recurrent),
        "baseline_effective_backend": baseline["effective_backend"],
        "fused_effective_backend": fused["effective_backend"],
        "baseline_fused_output": False,
        "fused_output": True,
        "baseline_ms_per_step": round(float(baseline["ms_per_step"]), 4),
        "fused_ms_per_step": round(float(fused["ms_per_step"]), 4),
        "speedup": round(float(baseline["ms_per_step"]) / float(fused["ms_per_step"]), 4) if fused["ms_per_step"] else None,
        "baseline_tokps_total": round(float(baseline["tokps_total"]), 1) if baseline["tokps_total"] else None,
        "fused_tokps_total": round(float(fused["tokps_total"]), 1) if fused["tokps_total"] else None,
        "max_abs_diff_first_step": round(max_abs, 6),
        "min_cosine_first_step": cosine,
        "greedy_match": greedy_match,
        "greedy_total": greedy_total,
        "baseline_cache_stats": baseline["cache_stats"],
        "fused_cache_stats": fused["cache_stats"],
        "peak_vram_mb": peak_mb(args.device),
    }
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended 1 row -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
