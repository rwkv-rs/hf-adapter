#!/usr/bin/env python3
# coding=utf-8
"""Measure native-graph replay overhead around the captured RWKV-7 decode graph.

The CUDA graph captures the model math, but the serving wrapper still has a
small amount of work around replay: copying cache state into fixed graph
buffers, copying the next token id, replaying the graph, and rebinding the HF
cache object to graph-buffer views. This benchmark records that overhead so
changes to the production-facing native-graph path are visible in JSONL gates.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("RWKV_V7_ON", "1")

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


def event_time_ms(fn: Callable[[], Any], device: str) -> float:
    if not device.startswith("cuda"):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def wall_time_ms(fn: Callable[[], Any], device: str) -> float:
    cuda_sync(device)
    t0 = time.perf_counter()
    fn()
    cuda_sync(device)
    return (time.perf_counter() - t0) * 1000.0


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
    for name in ("_rwkv7_native_jit_packs", "_rwkv7_native_graph_runner", "rwkv7_forward_token"):
        if not hasattr(model, name):
            raise ValueError(f"Loaded model does not expose {name}")
    return model


def encode(tok, prompt_tokens: int, batch_size: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    ids = ids.repeat(batch_size, 1)
    return ids.to(device) if device.startswith("cuda") else ids


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
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, dtype)
    ids = encode(tok, args.prompt_tokens, args.batch_size, args.device)
    fixed = ids[:, -1:]

    with torch.inference_mode():
        out = model(ids, use_cache=True, logits_to_keep=1)
        base_state = out.past_key_values
        token = fixed if args.fixed_token else out.logits[:, -1:].argmax(dim=-1)
        packs = model._rwkv7_native_jit_packs()
        runner = model._rwkv7_native_graph_runner(packs, int(token.numel()))
        state_manual = base_state.clone()
        state_api = base_state.clone()

        # Correctness sanity check for the direct runner path vs public API.
        logits_manual = runner.replay(token.reshape(-1), state_manual)
        api_out = model.rwkv7_forward_token(token, past_key_values=state_api)
        max_abs_diff = float((logits_manual.float() - api_out.logits.float()).abs().max().detach().cpu())
        effective_backend = getattr(model, "rwkv7_last_fast_token_backend", lambda: None)()

        # Reinitialize states so timing starts from a normal prefill cache.
        state_parts = base_state.clone()
        state_api = base_state.clone()
        token_parts = token.clone()
        token_api = token.clone()
        for _ in range(args.warmup):
            runner.replay(token_parts.reshape(-1), state_parts)
            out_api = model.rwkv7_forward_token(token_api, past_key_values=state_api)
            if not args.fixed_token:
                token_parts = runner.logits.view(args.batch_size, 1, -1).argmax(dim=-1)
                token_api = out_api.logits[:, -1:].argmax(dim=-1)

        copy_ms = token_ms = replay_ms = bind_ms = argmax_ms = 0.0
        wall_ms = 0.0
        for _ in range(args.steps):
            def step_parts() -> None:
                nonlocal copy_ms, token_ms, replay_ms, bind_ms, argmax_ms, token_parts
                copy_ms += event_time_ms(lambda: runner.copy_from_cache(state_parts), args.device)
                token_ms += event_time_ms(lambda: runner.tok_id.copy_(token_parts.reshape(args.batch_size)), args.device)
                replay_ms += event_time_ms(lambda: runner.graph.replay(), args.device)
                t_bind0 = time.perf_counter()
                runner.bind_cache(state_parts)
                bind_ms += (time.perf_counter() - t_bind0) * 1000.0
                if not args.fixed_token:
                    argmax_ms += event_time_ms(
                        lambda: runner.logits.view(args.batch_size, 1, -1).argmax(dim=-1),
                        args.device,
                    )
                    token_parts = runner.logits.view(args.batch_size, 1, -1).argmax(dim=-1)

            wall_ms += wall_time_ms(step_parts, args.device)

        def api_loop() -> None:
            nonlocal token_api, state_api
            for _ in range(args.steps):
                out_api = model.rwkv7_forward_token(token_api, past_key_values=state_api)
                state_api = out_api.past_key_values
                if not args.fixed_token:
                    token_api = out_api.logits[:, -1:].argmax(dim=-1)

        api_wall_ms = wall_time_ms(api_loop, args.device)

    denom = float(args.steps)
    wall_ms_per_token = wall_ms / denom
    api_ms_per_token = api_wall_ms / denom
    row = {
        "axis": "native_graph_replay_overhead",
        "backend": "hf_adapter",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES,
        "fast_token_backend": "native_graph",
        "fast_token_backend_effective": effective_backend,
        "batch_size": args.batch_size,
        "prompt_tokens": int(ids.shape[1]),
        "steps": args.steps,
        "fixed_token": args.fixed_token,
        "max_abs_diff_runner_vs_api": round(max_abs_diff, 6),
        "copy_from_cache_ms": round(copy_ms / denom, 4),
        "token_copy_ms": round(token_ms / denom, 4),
        "graph_replay_ms": round(replay_ms / denom, 4),
        "bind_cache_ms": round(bind_ms / denom, 4),
        "argmax_ms": round(argmax_ms / denom, 4),
        "manual_wall_ms_per_token": round(wall_ms_per_token, 4),
        "api_ms_per_token": round(api_ms_per_token, 4),
        "manual_decode_tokps_total": round(1000.0 * args.batch_size / wall_ms_per_token, 1) if wall_ms_per_token > 0 else None,
        "api_decode_tokps_total": round(1000.0 * args.batch_size / api_ms_per_token, 1) if api_ms_per_token > 0 else None,
        "copy_share_of_manual_wall": round((copy_ms / denom) / wall_ms_per_token, 4) if wall_ms_per_token > 0 else None,
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
