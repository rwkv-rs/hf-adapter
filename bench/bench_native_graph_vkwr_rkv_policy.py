#!/usr/bin/env python3
# coding=utf-8
"""A/B benchmark for VKWR-inspired stacked R/K/V native_graph projection.

Candidate mode:

    RWKV7_NATIVE_GRAPH_RKV_POLICY=vkwr_auto

The policy mirrors VKWR's grouped RKV rule: use a stacked batched projection
for one-row decode and medium tiny-row batches (4..64 by default), while
leaving rows 2/3 on the historical three-``F.linear`` path.  This is a
telemetry-only probe; it must beat the default end-to-end path before any
default-on decision.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_HELPER_PATH = Path(__file__).with_name("bench_native_graph_fused_wavg_lora.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("_rwkv7_wavg_bench_helper", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError(f"failed to load helper benchmark from {_HELPER_PATH}")
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER)

DTYPES = _HELPER.DTYPES
_FALSE_VALUES = _HELPER._FALSE_VALUES
cuda_sync = _HELPER.cuda_sync
device_name = _HELPER.device_name
encode = _HELPER.encode
load_model = _HELPER.load_model
peak_mb = _HELPER.peak_mb
prefill = _HELPER.prefill


def _set_common_env(args: argparse.Namespace) -> None:
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT"] = "1" if args.fused_recurrent else "0"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT"] = "1" if args.fused_recurrent_output else "0"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_OUTPUT"] = "1" if args.fused_output else "0"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT"] = "1" if args.fused_output_project else "0"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_PROJECTION"] = "0"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA"] = "0"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA"] = "0"
    os.environ["RWKV7_NATIVE_GRAPH_RKV_MIN_HIDDEN"] = str(args.policy_min_hidden)
    os.environ["RWKV7_NATIVE_GRAPH_RKV_MAX_ROWS"] = str(args.policy_max_rows)


def run_policy(model: Any, token: Any, base_state: Any, args: argparse.Namespace, *, policy: str) -> dict[str, Any]:
    _set_common_env(args)
    os.environ["RWKV7_NATIVE_GRAPH_RKV_POLICY"] = policy
    model.rwkv7_clear_native_graph_cache()
    if hasattr(model, "rwkv7_reset_native_graph_cache_stats"):
        model.rwkv7_reset_native_graph_cache_stats()

    import torch

    state = base_state.clone()
    tok = token.clone()
    with torch.inference_mode():
        first = model.rwkv7_forward_token(tok, past_key_values=state)
        effective_backend = getattr(model, "rwkv7_last_fast_token_backend", lambda: None)()
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
        "policy": policy,
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
    ap.add_argument("--fused-recurrent-output", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fused-output", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fused-output-project", action="store_true")
    ap.add_argument("--fused-recurrent", action="store_true")
    ap.add_argument("--policy-min-hidden", type=int, default=1)
    ap.add_argument("--policy-max-rows", type=int, default=64)
    ap.add_argument("--baseline-policy", default="manual", choices=["manual", "off"])
    ap.add_argument("--candidate-policy", default="vkwr_auto", choices=["vkwr_auto", "vkwr", "auto", "stacked"])
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, dtype)
    hidden_size = int(getattr(model.model.layers[0].attn, "hidden_size"))
    ids = encode(tok, args.prompt_tokens, args.batch_size, args.device)

    with torch.inference_mode():
        _set_common_env(args)
        os.environ["RWKV7_NATIVE_GRAPH_RKV_POLICY"] = args.baseline_policy
        token, base_state = prefill(model, ids)
        baseline = run_policy(model, token, base_state, args, policy=args.baseline_policy)
        candidate = run_policy(model, token, base_state, args, policy=args.candidate_policy)

    max_abs = float((baseline["first_logits"].float() - candidate["first_logits"].float()).abs().max().detach().cpu())
    cosine = float(torch.nn.functional.cosine_similarity(
        baseline["first_logits"].float().reshape(args.batch_size, -1),
        candidate["first_logits"].float().reshape(args.batch_size, -1),
        dim=-1,
    ).min().detach().cpu())
    greedy_total = min(len(baseline["greedy_tokens"]), len(candidate["greedy_tokens"]))
    greedy_match = sum(int(a == b) for a, b in zip(baseline["greedy_tokens"], candidate["greedy_tokens"], strict=False))
    active_by_rule = args.batch_size == 1 or 4 <= args.batch_size <= args.policy_max_rows
    row = {
        "axis": "native_graph_vkwr_rkv_policy",
        "backend": "hf_adapter",
        "status": "pass",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES,
        "batch_size": args.batch_size,
        "hidden_size": hidden_size,
        "prompt_tokens": int(ids.shape[1]),
        "steps": args.steps,
        "fixed_token": args.fixed_token,
        "policy_min_hidden": int(args.policy_min_hidden),
        "policy_max_rows": int(args.policy_max_rows),
        "policy_active_by_rule": bool(active_by_rule and hidden_size >= int(args.policy_min_hidden)),
        "fused_recurrent_enabled": bool(args.fused_recurrent),
        "fused_recurrent_output_enabled": bool(args.fused_recurrent_output),
        "fused_output_enabled": bool(args.fused_output),
        "fused_output_project_enabled": bool(args.fused_output_project),
        "baseline_policy": baseline["policy"],
        "candidate_policy": candidate["policy"],
        "baseline_effective_backend": baseline["effective_backend"],
        "candidate_effective_backend": candidate["effective_backend"],
        "baseline_ms_per_step": round(float(baseline["ms_per_step"]), 4),
        "candidate_ms_per_step": round(float(candidate["ms_per_step"]), 4),
        "speedup": round(float(baseline["ms_per_step"]) / float(candidate["ms_per_step"]), 4) if candidate["ms_per_step"] else None,
        "baseline_tokps_total": round(float(baseline["tokps_total"]), 1) if baseline["tokps_total"] else None,
        "candidate_tokps_total": round(float(candidate["tokps_total"]), 1) if candidate["tokps_total"] else None,
        "max_abs_diff_first_step": round(max_abs, 6),
        "min_cosine_first_step": cosine,
        "greedy_match": greedy_match,
        "greedy_total": greedy_total,
        "baseline_cache_stats": baseline["cache_stats"],
        "candidate_cache_stats": candidate["cache_stats"],
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
