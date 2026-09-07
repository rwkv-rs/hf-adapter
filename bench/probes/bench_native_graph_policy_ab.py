#!/usr/bin/env python3
# coding=utf-8
"""A/B the complete card policy against a conservative decode graph.

Unlike the one-feature fusion probes, this benchmark enables every decode
fusion selected by :mod:`rwkv7_hf.kernel_policy` at the same time.  Correctness
steps are kept outside the timed window so per-token CPU reads do not distort
the production decode result.
"""
from __future__ import annotations

# Support direct ``python bench/<category>/<script>.py`` execution.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from bench.benchlib.paths import DEFAULT_RESULTS_PATH

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

from rwkv7_hf.kernel_policy import current_kernel_policy


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 128
POLICY_ENV = {
    "fused_recurrent_output": "RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT",
    "fused_recurrent_raw": "RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW",
    "fused_output": "RWKV7_NATIVE_GRAPH_FUSED_OUTPUT",
    "fused_norm_mix": "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX",
}


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def set_attn_mode(model, mode: str) -> None:
    model.config.attn_mode = mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = mode


def apply_mode(model, flags: dict[str, bool], *, norm_mix_num_warps: int) -> None:
    for field, env_name in POLICY_ENV.items():
        os.environ[env_name] = "1" if flags[field] else "0"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS"] = str(
        norm_mix_num_warps
    )
    model.rwkv7_clear_native_graph_cache()
    if hasattr(model, "rwkv7_reset_native_graph_cache_stats"):
        model.rwkv7_reset_native_graph_cache_stats()


def decode_step(model, token: torch.Tensor, state):
    return model.rwkv7_forward_token(token, past_key_values=state)


def run_mode(
    model,
    token: torch.Tensor,
    base_state,
    args: argparse.Namespace,
    flags: dict[str, bool],
    *,
    norm_mix_num_warps: int,
) -> dict[str, Any]:
    apply_mode(model, flags, norm_mix_num_warps=norm_mix_num_warps)

    with torch.inference_mode():
        state = base_state.clone()
        first = decode_step(model, token.clone(), state)
        effective_backend = getattr(
            model, "rwkv7_last_fast_token_backend", lambda: None
        )()

        # Correctness is intentionally outside the timed window.
        state = base_state.clone()
        current = token.clone()
        greedy: list[int] = []
        for _ in range(args.correctness_steps):
            out = decode_step(model, current, state)
            next_token = out.logits[:, -1:].argmax(dim=-1)
            greedy.extend(int(v) for v in next_token.detach().cpu().reshape(-1))
            current = token if args.fixed_token else next_token

        # Warm and time a fresh state. No device-to-host read occurs per step.
        state = base_state.clone()
        current = token.clone()
        for _ in range(args.warmup):
            out = decode_step(model, current, state)
            if not args.fixed_token:
                current = out.logits[:, -1:].argmax(dim=-1)
        sync(args.device)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            out = decode_step(model, current, state)
            if not args.fixed_token:
                current = out.logits[:, -1:].argmax(dim=-1)
        sync(args.device)
        elapsed = time.perf_counter() - t0

    ms_per_step = elapsed * 1000.0 / float(args.steps)
    stats = (
        model.rwkv7_native_graph_cache_stats()
        if hasattr(model, "rwkv7_native_graph_cache_stats")
        else {}
    )
    return {
        "effective_backend": effective_backend,
        "first_logits": first.logits.detach().clone(),
        "greedy_tokens": greedy,
        "ms_per_step": ms_per_step,
        "tokps_total": 1000.0 * int(token.numel()) / ms_per_step,
        "cache_stats": stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--attn-mode", choices=["chunk", "fused_recurrent"], default="fused_recurrent"
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--correctness-steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--fixed-token", action="store_true")
    ap.add_argument("--min-speedup", type=float, default=0.0)
    ap.add_argument("--require-accelerated-policy", action="store_true")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    set_attn_mode(model, args.attn_mode)
    for name in ("rwkv7_forward_token", "rwkv7_clear_native_graph_cache"):
        if not hasattr(model, name):
            raise ValueError(f"loaded model does not expose {name}")

    ids = tokenizer(
        SEED, return_tensors="pt", add_special_tokens=False
    ).input_ids[:, : args.prompt_tokens]
    ids = ids.repeat(args.batch_size, 1).to(args.device)
    conservative = {field: False for field in POLICY_ENV}
    policy = current_kernel_policy(device=args.device, torch_module=torch)
    selected = {field: bool(getattr(policy, field)) for field in POLICY_ENV}
    policy_active = any(selected.values())
    if args.require_accelerated_policy and not policy_active:
        raise RuntimeError(
            f"no accelerated decode policy for {policy.profile.family}/"
            f"{policy.profile.architecture or policy.profile.name}"
        )

    # Prefill is held on the conservative path; this A/B owns decode only.
    apply_mode(model, conservative, norm_mix_num_warps=policy.norm_mix_num_warps)
    with torch.inference_mode():
        prefill = model(ids, use_cache=True, logits_to_keep=1)
    token = prefill.logits[:, -1:].argmax(dim=-1)
    base_state = prefill.past_key_values

    baseline = run_mode(
        model,
        token,
        base_state,
        args,
        conservative,
        norm_mix_num_warps=policy.norm_mix_num_warps,
    )
    candidate = run_mode(
        model,
        token,
        base_state,
        args,
        selected,
        norm_mix_num_warps=policy.norm_mix_num_warps,
    )

    baseline_logits = baseline["first_logits"].float().reshape(args.batch_size, -1)
    candidate_logits = candidate["first_logits"].float().reshape(args.batch_size, -1)
    max_abs = float((baseline_logits - candidate_logits).abs().max().cpu())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            baseline_logits, candidate_logits, dim=-1
        ).min().cpu()
    )
    greedy_total = min(
        len(baseline["greedy_tokens"]), len(candidate["greedy_tokens"])
    )
    greedy_match = sum(
        int(a == b)
        for a, b in zip(
            baseline["greedy_tokens"], candidate["greedy_tokens"], strict=False
        )
    )
    speedup = float(baseline["ms_per_step"]) / float(candidate["ms_per_step"])
    correctness_pass = bool(greedy_match == greedy_total and cosine >= 0.999)
    speed_pass = bool(speedup >= args.min_speedup)
    backend_pass = bool(
        baseline["effective_backend"] == "native_graph"
        and candidate["effective_backend"] == "native_graph"
    )
    passed = correctness_pass and speed_pass and backend_pass

    profile = policy.profile
    row = {
        "axis": "native_graph_policy_ab",
        "backend": "hf_adapter",
        "status": "pass" if passed else "fail",
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "gpu_family": profile.family,
        "gpu_architecture": profile.architecture,
        "model_name": Path(args.hf_dir).name,
        "hf_model_dir": args.hf_dir,
        "hidden_size": getattr(model.config, "hidden_size", None),
        "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
        "batch_size": args.batch_size,
        "prompt_tokens": int(ids.shape[1]),
        "correctness_steps": args.correctness_steps,
        "timing_steps": args.steps,
        "fixed_token": args.fixed_token,
        "selected_policy": selected,
        "norm_mix_num_warps": policy.norm_mix_num_warps,
        "baseline_effective_backend": baseline["effective_backend"],
        "candidate_effective_backend": candidate["effective_backend"],
        "baseline_ms_per_step": round(float(baseline["ms_per_step"]), 4),
        "candidate_ms_per_step": round(float(candidate["ms_per_step"]), 4),
        "baseline_tokps_total": round(float(baseline["tokps_total"]), 1),
        "candidate_tokps_total": round(float(candidate["tokps_total"]), 1),
        "speedup": round(speedup, 4),
        "min_speedup": args.min_speedup,
        "max_abs_diff_first_step": round(max_abs, 6),
        "min_cosine_first_step": cosine,
        "greedy_match": greedy_match,
        "greedy_total": greedy_total,
        "correctness_pass": correctness_pass,
        "speed_pass": speed_pass,
        "backend_pass": backend_pass,
        "baseline_cache_stats": baseline["cache_stats"],
        "candidate_cache_stats": candidate["cache_stats"],
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
            if args.device.startswith("cuda")
            else None
        ),
    }
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    if args.results:
        output = Path(args.results)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
