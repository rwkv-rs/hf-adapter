#!/usr/bin/env python3
# coding=utf-8
"""Paired native-graph decode A/B for recurrent-state precision.

The baseline explicitly uses FP32 state.  The candidate either follows the
hardware policy (the default) or forces the Triton FP16-state route for an
exploratory shape.  Graph-bound caches are detached and graph pools are
released between modes so large checkpoints can be compared on 16 GiB cards.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV7_FAST_TOKEN_BACKEND", "native_graph")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM


SEED = "The quick brown fox jumps over the lazy dog. " * 128


def _set_mode(*, candidate: bool, force_candidate: bool) -> None:
    if not candidate:
        os.environ["RWKV7_NATIVE_GRAPH_STATE_DTYPE"] = "fp32"
        os.environ["RWKV7_NATIVE_GRAPH_TRITON_FP16_STATE"] = "0"
        os.environ["RWKV7_NATIVE_GRAPH_FP16_RECURRENT"] = "0"
        return
    os.environ.pop("RWKV7_NATIVE_GRAPH_STATE_DTYPE", None)
    os.environ.pop("RWKV7_NATIVE_GRAPH_FP16_RECURRENT", None)
    if force_candidate:
        os.environ["RWKV7_NATIVE_GRAPH_TRITON_FP16_STATE"] = "1"
    else:
        os.environ.pop("RWKV7_NATIVE_GRAPH_TRITON_FP16_STATE", None)


def _release_mode(model, state, output) -> None:
    runner = state._native_graph_bound_runner()
    state._invalidate_native_graph_binding()
    if runner is not None:
        runner._bound_cache_ref = None
    del output, state, runner
    model.rwkv7_clear_native_graph_cache()
    gc.collect()
    torch.cuda.empty_cache()


def _run_mode(
    model,
    base_state,
    start_token: torch.Tensor,
    args: argparse.Namespace,
    *,
    candidate: bool,
) -> dict[str, Any]:
    _set_mode(candidate=candidate, force_candidate=args.force_candidate)
    model.rwkv7_clear_native_graph_cache()
    if hasattr(model, "rwkv7_reset_native_graph_cache_stats"):
        model.rwkv7_reset_native_graph_cache_stats()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    state = base_state.clone()
    token = start_token.clone()
    greedy: list[int] = []
    with torch.inference_mode():
        output = model.rwkv7_forward_token(token, past_key_values=state)
        first_logits = output.logits.detach().float().cpu().clone()
        for _ in range(args.correctness_steps):
            output = model.rwkv7_forward_token(token, past_key_values=state)
            token = output.logits[:, -1:].argmax(dim=-1)
            greedy.extend(int(value) for value in token.detach().cpu().reshape(-1))

        fixed_token = start_token.clone()
        for _ in range(args.warmup):
            output = model.rwkv7_forward_token(
                fixed_token,
                past_key_values=state,
                copy_logits=False,
            )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.steps):
            output = model.rwkv7_forward_token(
                fixed_token,
                past_key_values=state,
                copy_logits=False,
            )
        end.record()
        torch.cuda.synchronize()

    ms_per_step = float(start.elapsed_time(end)) / float(args.steps)
    runner = state._native_graph_bound_runner()
    route = {
        "state_dtype": str(getattr(runner, "state_dtype", "unknown")),
        "triton_fp16_state": bool(getattr(runner, "triton_fp16_state", False)),
        "native_fp16_recurrent": bool(getattr(runner, "fp16_recurrent", False)),
    }
    result = {
        "first_logits": first_logits,
        "greedy": greedy,
        "ms_per_step": ms_per_step,
        "tokps_total": 1000.0 * int(start_token.numel()) / ms_per_step,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated()) / (1024.0**2),
        "route": route,
        "cache_stats": model.rwkv7_native_graph_cache_stats(),
    }
    _release_mode(model, state, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--correctness-steps", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--candidate-first", action="store_true")
    parser.add_argument(
        "--force-candidate",
        action="store_true",
        help="force Triton FP16 state instead of exercising the default policy",
    )
    parser.add_argument("--results", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.hf_dir,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()
    ids = tokenizer(
        SEED,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[:, : args.prompt_tokens]
    ids = ids.repeat(args.batch_size, 1).cuda()
    with torch.inference_mode():
        prefill = model(ids, use_cache=True, logits_to_keep=1)
        base_state = prefill.past_key_values
        start_token = prefill.logits[:, -1:].argmax(dim=-1)
    del prefill

    order = (True, False) if args.candidate_first else (False, True)
    modes: dict[bool, dict[str, Any]] = {}
    for candidate in order:
        modes[candidate] = _run_mode(
            model,
            base_state,
            start_token,
            args,
            candidate=candidate,
        )
    baseline = modes[False]
    candidate = modes[True]

    max_abs = float(
        (baseline["first_logits"] - candidate["first_logits"]).abs().max()
    )
    cosine = float(
        F.cosine_similarity(
            baseline["first_logits"].reshape(args.batch_size, -1),
            candidate["first_logits"].reshape(args.batch_size, -1),
            dim=-1,
        ).min()
    )
    greedy_total = min(len(baseline["greedy"]), len(candidate["greedy"]))
    greedy_match = sum(
        int(left == right)
        for left, right in zip(
            baseline["greedy"],
            candidate["greedy"],
            strict=False,
        )
    )
    route_pass = bool(
        candidate["route"]["state_dtype"] == "torch.float16"
        and candidate["route"]["triton_fp16_state"]
        and not candidate["route"]["native_fp16_recurrent"]
    )
    correctness_pass = bool(greedy_match == greedy_total and cosine >= 0.999)
    row = {
        "axis": "native_graph_triton_fp16_state",
        "status": "pass" if route_pass and correctness_pass else "fail",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model": str(Path(args.hf_dir).expanduser()),
        "hidden_size": int(model.config.hidden_size),
        "num_layers": int(model.config.num_hidden_layers),
        "batch_size": args.batch_size,
        "prompt_tokens": int(ids.shape[1]),
        "correctness_steps": args.correctness_steps,
        "timing_steps": args.steps,
        "candidate_forced": args.force_candidate,
        "candidate_first": args.candidate_first,
        "baseline_route": baseline["route"],
        "candidate_route": candidate["route"],
        "baseline_ms_per_step": baseline["ms_per_step"],
        "candidate_ms_per_step": candidate["ms_per_step"],
        "speedup": baseline["ms_per_step"] / candidate["ms_per_step"],
        "baseline_tokps_total": baseline["tokps_total"],
        "candidate_tokps_total": candidate["tokps_total"],
        "baseline_peak_vram_mb": baseline["peak_vram_mb"],
        "candidate_peak_vram_mb": candidate["peak_vram_mb"],
        "peak_vram_delta_mb": candidate["peak_vram_mb"] - baseline["peak_vram_mb"],
        "max_abs_diff_first_step": max_abs,
        "min_cosine_first_step": cosine,
        "greedy_match": greedy_match,
        "greedy_total": greedy_total,
        "route_pass": route_pass,
        "correctness_pass": correctness_pass,
        "baseline_cache_stats": baseline["cache_stats"],
        "candidate_cache_stats": candidate["cache_stats"],
    }
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    if args.results:
        path = Path(args.results)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"appended 1 row -> {path}", flush=True)
    return 0 if row["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
