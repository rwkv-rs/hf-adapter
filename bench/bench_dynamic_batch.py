#!/usr/bin/env python3
# coding=utf-8
"""Dynamic-batch decode benchmark for the RWKV-7 HF adapter.

This simulates serving behavior that repeatedly reorders active rows and drops
completed rows from the recurrent state cache. It does not model request arrival,
but it exercises the core state-cache operations needed by dynamic batching and
records total decoded tokens/s for standard HF `forward` and the fast token API.
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
_FALSE_VALUES = {"0", "false", "False", "no", "off"}
PROMPTS = [
    "Alpha user asks about graph theory, eigenvalues, and sparse matrices. ",
    "Beta dialogue covers cooking rice, mountain weather, and train tickets. ",
    "Gamma note discusses compilers, register allocation, and loop fusion. ",
    "Delta report mentions batteries, camera lenses, and market volatility. ",
    "Epsilon story has robots, ancient maps, and a quiet library at night. ",
    "Zeta memo compares quantization, cache locality, and instruction fusion. ",
    "Eta transcript includes astronomy, ocean currents, and classical poetry. ",
    "Theta example mixes probability puzzles, APIs, and hardware counters. ",
]


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


def load_model(args, dtype):
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    os.environ["RWKV7_FAST_TOKEN_BACKEND"] = args.fast_token_backend
    os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = args.fast_token_backend
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
    return model


def last_fast_token_backend(model):
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


def state_cache_metrics(state: Any) -> dict[str, Any]:
    getter = getattr(state, "rwkv7_cache_metrics", None)
    if not callable(getter):
        return {}
    metrics = getter()
    return dict(metrics) if isinstance(metrics, dict) else {}


def native_graph_cache_stats(model) -> dict[str, Any]:
    getter = getattr(model, "rwkv7_native_graph_cache_stats", None)
    if not callable(getter):
        return {}
    stats = getter()
    return dict(stats) if isinstance(stats, dict) else {}


def native_graph_copy_stats(model) -> dict[str, Any]:
    getter = getattr(model, "rwkv7_native_graph_runner_copy_stats", None)
    if not callable(getter):
        return {}
    stats = getter()
    return dict(stats) if isinstance(stats, dict) else {}


def native_graph_copy_totals(stats: dict[str, Any]) -> dict[str, int]:
    totals = stats.get("totals") if isinstance(stats, dict) else {}
    if not isinstance(totals, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("copy_from_cache_calls", "copy_from_cache_fast_skips", "bind_cache_calls", "bind_cache_fast_skips"):
        try:
            out[key] = int(totals.get(key, 0))
        except (TypeError, ValueError):
            out[key] = 0
    return out


def diff_native_graph_copy_stats(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_totals = native_graph_copy_totals(before)
    after_totals = native_graph_copy_totals(after)
    deltas = {
        key: int(after_totals.get(key, 0)) - int(before_totals.get(key, 0))
        for key in ("copy_from_cache_calls", "copy_from_cache_fast_skips", "bind_cache_calls", "bind_cache_fast_skips")
    }
    copy_calls = int(deltas["copy_from_cache_calls"])
    bind_calls = int(deltas["bind_cache_calls"])
    deltas["copy_from_cache_fast_skip_rate"] = (
        float(deltas["copy_from_cache_fast_skips"]) / float(copy_calls) if copy_calls else None
    )
    deltas["bind_cache_fast_skip_rate"] = (
        float(deltas["bind_cache_fast_skips"]) / float(bind_calls) if bind_calls else None
    )
    return deltas


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


def encode_prompts(tok, batch_size: int, prompt_tokens: int, device: str) -> torch.Tensor:
    rows = []
    for i in range(batch_size):
        text = PROMPTS[i % len(PROMPTS)] * 128
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if ids.numel() < prompt_tokens:
            raise ValueError(f"Prompt {i} only tokenized to {ids.numel()} tokens; need {prompt_tokens}")
        rows.append(ids[:prompt_tokens])
    out = torch.stack(rows, dim=0)
    return out.to(device) if device.startswith("cuda") else out


def step_decode(model, decode_api: str, token: torch.Tensor, state: Any, logits_to_keep: int):
    if decode_api == "forward":
        with reference_forward_env():
            return model(token, past_key_values=state, use_cache=True, logits_to_keep=logits_to_keep)
    if decode_api == "rwkv7_forward_token":
        return model.rwkv7_forward_token(token, past_key_values=state)
    raise ValueError(decode_api)


def maybe_reorder_or_drop(state, token: torch.Tensor, step_idx: int, args) -> tuple[Any, torch.Tensor, int, int]:
    active = int(token.shape[0])
    if active <= 1:
        return state, token, 0, 0
    do_reorder = args.reorder_every > 0 and step_idx % args.reorder_every == 0
    do_drop = args.drop_every > 0 and step_idx % args.drop_every == 0 and active > args.min_batch_size
    if not do_reorder and not do_drop:
        return state, token, 0, 0
    perm = list(range(active))
    if do_reorder:
        perm = [active - 1, *range(active - 1)]
    if do_drop:
        perm = perm[: active - 1]
    perm_t = torch.tensor(perm, dtype=torch.long, device=token.device)
    if hasattr(state, "select_batch"):
        state = state.select_batch(perm_t, inplace=True)
    elif hasattr(state, "reorder_cache"):
        state.reorder_cache(perm_t.detach().cpu())
    else:
        raise ValueError(f"Cache type {type(state).__name__} does not expose select_batch/reorder_cache")
    token = token.index_select(0, perm_t)
    return state, token, int(do_reorder), int(do_drop)


def run_loop(args, model, ids: torch.Tensor, decode_api: str) -> dict[str, Any]:
    if decode_api == "rwkv7_forward_token" and not hasattr(model, "rwkv7_forward_token"):
        raise ValueError("Loaded model does not expose rwkv7_forward_token")
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    requested_backend = os.environ.get("RWKV7_FAST_TOKEN_BACKEND", "auto")
    if decode_api == "rwkv7_forward_token" and requested_backend in {"auto", "native_jit", "native_graph"}:
        # Shape changes are expected in dynamic batching. TorchScript specializes
        # the native block-step for new batch sizes, so compile the active sizes
        # outside the timed region to measure steady-state serving throughput.
        with torch.inference_mode():
            for active in range(int(ids.shape[0]), args.min_batch_size - 1, -1):
                warm_ids = ids[:active]
                out = model(warm_ids, use_cache=True, logits_to_keep=args.hf_logits_to_keep)
                warm_state = out.past_key_values
                warm_token = out.logits[:, -1:].argmax(dim=-1)
                for _ in range(2):
                    out = step_decode(model, decode_api, warm_token, warm_state, args.hf_logits_to_keep)
                    warm_state = out.past_key_values
                    warm_token = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        out = model(ids, use_cache=True, logits_to_keep=args.hf_logits_to_keep)
        state = out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
        for i in range(args.warmup):
            out = step_decode(model, decode_api, token, state, args.hf_logits_to_keep)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
            state, token, _, _ = maybe_reorder_or_drop(state, token, i + 1, args)

        out = model(ids, use_cache=True, logits_to_keep=args.hf_logits_to_keep)
        state = out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
        if hasattr(model, "rwkv7_reset_native_graph_cache_stats"):
            model.rwkv7_reset_native_graph_cache_stats()
        copy_stats_before = native_graph_copy_stats(model)
        total_tokens = 0
        reorder_count = 0
        drop_count = 0
        cuda_sync(args.device)
        t0 = time.time()
        for i in range(args.decode_steps):
            out = step_decode(model, decode_api, token, state, args.hf_logits_to_keep)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
            total_tokens += int(token.shape[0])
            state, token, r, d = maybe_reorder_or_drop(state, token, i + 1, args)
            reorder_count += r
            drop_count += d
        cuda_sync(args.device)
        dt = time.time() - t0

    cache_metrics = state_cache_metrics(state)
    graph_stats = native_graph_cache_stats(model)
    copy_delta = diff_native_graph_copy_stats(copy_stats_before, native_graph_copy_stats(model))
    row = {
        "axis": "dynamic_batch",
        "backend": "hf_adapter",
        "decode_api": decode_api,
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES,
        "cache_type": type(state).__name__ if state is not None else None,
        "cache_select_api": bool(hasattr(state, "select_batch")) if state is not None else False,
        "final_cache_batch_size": state.get_batch_size() if hasattr(state, "get_batch_size") else None,
        "prompt_tokens": int(ids.shape[1]),
        "initial_batch_size": int(ids.shape[0]),
        "final_batch_size": int(token.shape[0]),
        "min_batch_size": args.min_batch_size,
        "decode_steps": args.decode_steps,
        "total_decode_tokens": total_tokens,
        "reorder_every": args.reorder_every,
        "drop_every": args.drop_every,
        "reorder_count": reorder_count,
        "drop_count": drop_count,
        "decode_tokps_total": round(total_tokens / dt, 1),
        "decode_ms_per_step": round(1000 * dt / args.decode_steps, 2),
        "decode_ms_per_token": round(1000 * dt / max(total_tokens, 1), 4),
        "cache_updates": cache_metrics.get("updates"),
        "cache_new_layers": cache_metrics.get("new_layers"),
        "cache_select_batch_calls": cache_metrics.get("select_batch_calls"),
        "cache_native_graph_bound_selects": cache_metrics.get("native_graph_bound_selects"),
        "cache_batch_select_calls": cache_metrics.get("batch_select_calls"),
        "cache_reorder_calls": cache_metrics.get("reorder_calls"),
        "cache_device_moves": cache_metrics.get("device_moves"),
        "cache_resets": cache_metrics.get("resets"),
        "cache_seen_tokens": cache_metrics.get("seen_tokens"),
        "state_cache_metrics": cache_metrics,
        "native_graph_cache_requests": graph_stats.get("requests"),
        "native_graph_cache_hits": graph_stats.get("hits"),
        "native_graph_cache_misses": graph_stats.get("misses"),
        "native_graph_cache_evictions": graph_stats.get("evictions"),
        "native_graph_cache_hit_rate": round(float(graph_stats["hit_rate"]), 4)
        if graph_stats.get("hit_rate") is not None
        else None,
        "native_graph_cache_batch_sizes": graph_stats.get("batch_sizes"),
        "native_graph_cache_stats": graph_stats,
        "native_graph_copy_from_cache_calls": copy_delta.get("copy_from_cache_calls"),
        "native_graph_copy_from_cache_fast_skips": copy_delta.get("copy_from_cache_fast_skips"),
        "native_graph_copy_from_cache_fast_skip_rate": round(float(copy_delta["copy_from_cache_fast_skip_rate"]), 4)
        if copy_delta.get("copy_from_cache_fast_skip_rate") is not None
        else None,
        "native_graph_bind_cache_calls": copy_delta.get("bind_cache_calls"),
        "native_graph_bind_cache_fast_skips": copy_delta.get("bind_cache_fast_skips"),
        "native_graph_bind_cache_fast_skip_rate": round(float(copy_delta["bind_cache_fast_skip_rate"]), 4)
        if copy_delta.get("bind_cache_fast_skip_rate") is not None
        else None,
        "native_graph_runner_copy_delta": copy_delta,
        "peak_vram_mb": peak_mb(args.device),
    }
    if decode_api == "rwkv7_forward_token":
        row["fast_token_backend"] = requested_backend
        row["fast_token_backend_effective"] = last_fast_token_backend(model) or requested_backend
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-token-backend", choices=["auto", "fla", "native_jit", "native_graph"], default="auto")
    ap.add_argument("--decode-apis", nargs="+", default=["forward", "rwkv7_forward_token"], choices=["forward", "rwkv7_forward_token"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-batch-size", type=int, default=2)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--reorder-every", type=int, default=4)
    ap.add_argument("--drop-every", type=int, default=32)
    ap.add_argument("--hf-logits-to-keep", type=int, default=1)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    if args.min_batch_size < 1 or args.min_batch_size > args.batch_size:
        raise ValueError("--min-batch-size must be between 1 and --batch-size")
    dtype = DTYPES[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, dtype)
    ids = encode_prompts(tok, args.batch_size, args.prompt_tokens, args.device)

    rows = []
    for decode_api in args.decode_apis:
        print(f"\n===== dynamic batch decode_api={decode_api} =====", flush=True)
        row = run_loop(args, model, ids, decode_api)
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
