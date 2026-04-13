#!/usr/bin/env python3
# coding=utf-8
"""Balanced native-graph decode A/B for state precision and fused features.

The baseline explicitly uses FP32 state.  The candidate either follows the
hardware policy (the default) or forces one exploratory feature. Graph-bound
caches are detached and graph pools are released between modes so candidates
can be compared without reloading the model or reusing a stale capture.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from statistics import median
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV7_FAST_TOKEN_BACKEND", "native_graph")

import torch
import torch.nn.functional as F

from rwkv7_hf import native_jit
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM
from rwkv7_hf.tokenization_rwkv7 import RWKV7Tokenizer


SEED = "The quick brown fox jumps over the lazy dog. " * 128


def balanced_mode_order(rounds: int, *, candidate_first: bool) -> tuple[bool, ...]:
    """Alternate pair order so each mode occupies early and late positions."""

    if rounds <= 0:
        raise ValueError("paired rounds must be positive")
    order: list[bool] = []
    starts_with_candidate = bool(candidate_first)
    for round_index in range(rounds):
        first = starts_with_candidate if round_index % 2 == 0 else not starts_with_candidate
        order.extend((first, not first))
    return tuple(order)


def aggregate_mode_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated mode runs while retaining one correctness payload."""

    if not results:
        raise ValueError("cannot aggregate an empty mode result list")
    first = dict(results[0])
    first["ms_per_step"] = median(float(row["ms_per_step"]) for row in results)
    first["tokps_total"] = median(float(row["tokps_total"]) for row in results)
    first["peak_vram_mb"] = max(float(row["peak_vram_mb"]) for row in results)
    first["cache_stats"] = results[-1]["cache_stats"]
    first["timing_samples_ms"] = [float(row["ms_per_step"]) for row in results]
    return first


def _set_mode(
    *,
    candidate: bool,
    force_candidate: bool,
    candidate_feature: str,
) -> None:
    if candidate_feature == "fused-norm-mix":
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX"] = "1" if candidate else "0"
        return
    if candidate_feature == "fused-recurrent-raw":
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW"] = "1" if candidate else "0"
        return
    if candidate_feature == "precompute-embedding":
        os.environ["RWKV7_NATIVE_GRAPH_PRECOMPUTE_EMB_LN0"] = "1" if candidate else "0"
        return
    if candidate_feature == "fused-output-project":
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT"] = "1" if candidate else "0"
        return
    if candidate_feature == "fused-projection":
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_PROJECTION"] = "1" if candidate else "0"
        return
    if candidate_feature == "ada-linear":
        os.environ["RWKV7_NATIVE_GRAPH_ADA_LINEAR"] = "1" if candidate else "0"
        return
    if candidate_feature == "fused-wavg-lora":
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA"] = "1" if candidate else "0"
        return
    if candidate_feature == "fused-wag-lora":
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA"] = "1" if candidate else "0"
        return
    if candidate_feature.startswith("norm-mix-warps-"):
        target = int(candidate_feature.rsplit("-", 1)[1])
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX"] = "1"
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS"] = str(
            target if candidate else 4
        )
        return
    if candidate_feature.startswith("recurrent-raw-warps-"):
        target = int(candidate_feature.rsplit("-", 1)[1])
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW"] = "1"
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW_NUM_WARPS"] = str(
            target if candidate else 8
        )
        return
    if candidate_feature != "fp16-state":
        raise ValueError(f"unsupported candidate feature: {candidate_feature}")
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
    _set_mode(
        candidate=candidate,
        force_candidate=args.force_candidate,
        candidate_feature=args.candidate_feature,
    )
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
        "fused_norm_mix": bool(
            native_jit._native_graph_fused_norm_mix_enabled(
                int(start_token.shape[0]),
                int(model.config.hidden_size),
            )
        ),
        "fused_recurrent_raw": bool(
            native_jit._native_graph_fused_recurrent_raw_enabled(
                int(start_token.shape[0]),
                int(model.config.hidden_size),
            )
        ),
        "precomputed_embedding_ln0": bool(
            getattr(runner, "precomputed_embedding_ln0", False)
        ),
        "fused_output_project": bool(
            native_jit._native_graph_fused_output_project_enabled()
        ),
        "fused_projection": bool(
            native_jit._native_graph_fused_projection_enabled()
        ),
        "ada_linear": bool(native_jit._native_graph_ada_linear_enabled()),
        "fused_wavg_lora": bool(
            native_jit._native_graph_fused_wavg_lora_enabled(
                int(start_token.shape[0]),
                int(model.config.hidden_size),
            )
        ),
        "fused_wag_lora": bool(
            native_jit._native_graph_fused_wag_lora_enabled()
        ),
        "norm_mix_num_warps": int(
            native_jit._native_graph_fused_norm_mix_num_warps()
        ),
        "recurrent_raw_num_warps": int(
            native_jit._native_graph_fused_recurrent_raw_num_warps()
        ),
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
        "--candidate-feature",
        choices=(
            "fp16-state",
            "fused-norm-mix",
            "fused-recurrent-raw",
            "precompute-embedding",
            "fused-output-project",
            "fused-projection",
            "ada-linear",
            "fused-wavg-lora",
            "fused-wag-lora",
            "norm-mix-warps-1",
            "norm-mix-warps-2",
            "norm-mix-warps-8",
            "recurrent-raw-warps-1",
            "recurrent-raw-warps-2",
            "recurrent-raw-warps-4",
        ),
        default="fp16-state",
    )
    parser.add_argument(
        "--paired-rounds",
        type=int,
        default=1,
        help="number of paired runs; 2 uses an ABBA-balanced order",
    )
    parser.add_argument(
        "--force-candidate",
        action="store_true",
        help="force Triton FP16 state instead of exercising the default policy",
    )
    parser.add_argument("--results", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    tokenizer = RWKV7Tokenizer.from_pretrained(args.hf_dir)
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

    mode_runs: dict[bool, list[dict[str, Any]]] = {False: [], True: []}
    order = balanced_mode_order(
        args.paired_rounds,
        candidate_first=args.candidate_first,
    )
    for candidate in order:
        mode_runs[candidate].append(_run_mode(
            model,
            base_state,
            start_token,
            args,
            candidate=candidate,
        ))
    baseline = aggregate_mode_results(mode_runs[False])
    candidate = aggregate_mode_results(mode_runs[True])

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
    if args.candidate_feature == "fp16-state":
        route_pass = bool(
            candidate["route"]["state_dtype"] == "torch.float16"
            and candidate["route"]["triton_fp16_state"]
            and not candidate["route"]["native_fp16_recurrent"]
        )
    elif args.candidate_feature.startswith("norm-mix-warps-"):
        target_warps = int(args.candidate_feature.rsplit("-", 1)[1])
        route_pass = bool(
            baseline["route"]["fused_norm_mix"]
            and candidate["route"]["fused_norm_mix"]
            and baseline["route"]["norm_mix_num_warps"] == 4
            and candidate["route"]["norm_mix_num_warps"] == target_warps
        )
    elif args.candidate_feature.startswith("recurrent-raw-warps-"):
        target_warps = int(args.candidate_feature.rsplit("-", 1)[1])
        route_pass = bool(
            baseline["route"]["fused_recurrent_raw"]
            and candidate["route"]["fused_recurrent_raw"]
            and baseline["route"]["recurrent_raw_num_warps"] == 8
            and candidate["route"]["recurrent_raw_num_warps"] == target_warps
        )
    else:
        route_key = {
            "fused-norm-mix": "fused_norm_mix",
            "fused-recurrent-raw": "fused_recurrent_raw",
            "precompute-embedding": "precomputed_embedding_ln0",
            "fused-output-project": "fused_output_project",
            "fused-projection": "fused_projection",
            "ada-linear": "ada_linear",
            "fused-wavg-lora": "fused_wavg_lora",
            "fused-wag-lora": "fused_wag_lora",
        }[args.candidate_feature]
        route_pass = bool(
            candidate["route"][route_key]
            and not baseline["route"][route_key]
        )
    correctness_pass = bool(greedy_match == greedy_total and cosine >= 0.999)
    row = {
        "axis": "native_graph_" + args.candidate_feature.replace("-", "_"),
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
        "candidate_feature": args.candidate_feature,
        "candidate_first": args.candidate_first,
        "paired_rounds": args.paired_rounds,
        "mode_order": ["candidate" if value else "baseline" for value in order],
        "baseline_route": baseline["route"],
        "candidate_route": candidate["route"],
        "baseline_ms_per_step": baseline["ms_per_step"],
        "candidate_ms_per_step": candidate["ms_per_step"],
        "baseline_timing_samples_ms": baseline["timing_samples_ms"],
        "candidate_timing_samples_ms": candidate["timing_samples_ms"],
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
