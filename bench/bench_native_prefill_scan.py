#!/usr/bin/env python3
# coding=utf-8
"""Benchmark the native layer-wise prefill path and optional fused scan.

Rows from this script are the end-to-end prefill counterpart to
`bench_fused_recurrent_scan.py`: the recurrent scan kernel is useful only if it
survives full-layer projection/output/FFN overhead and produces a cache that the
native_graph decode path can continue from.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rwkv7_hf import native_jit


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 256


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def wall_ms(fn: Callable[[], Any], device: str) -> float:
    cuda_sync(device)
    t0 = time.perf_counter()
    fn()
    cuda_sync(device)
    return (time.perf_counter() - t0) * 1000.0


def build_ids(tok, batch_size: int, prompt_tokens: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    if int(ids.shape[1]) < prompt_tokens:
        raise ValueError(f"seed produced only {ids.shape[1]} tokens, need {prompt_tokens}")
    return ids.repeat(batch_size, 1).to(device)


def median(vals: list[float]) -> float:
    vals = sorted(vals)
    return vals[len(vals) // 2]


def infer_model_size_label(model_path: str) -> str | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", str(model_path).lower())
    return f"{match.group(1)}b" if match else None


def scan_block_m(model) -> int | None:
    raw = os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    try:
        return int(model._rwkv7_native_jit_packs()[0][2])
    except Exception:
        return None


def cosine_min(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=-1).min().detach().cpu())


def run_case(args: argparse.Namespace, tok, model, batch_size: int, prompt_tokens: int) -> dict[str, Any]:
    ids = build_ids(tok, batch_size, prompt_tokens, args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # FLA/HF reference prefill.
    with torch.inference_mode():
        ref = model(ids, use_cache=True, logits_to_keep=1, return_dict=True)
        native = model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)
        ref_logits = ref.logits[:, -1, :].detach()
        native_logits = native.logits[:, -1, :].detach()
        max_abs = float((ref_logits.float() - native_logits.float()).abs().max().detach().cpu())
        min_cos = cosine_min(ref_logits, native_logits)
        greedy_match = bool(torch.equal(ref_logits.argmax(dim=-1).detach().cpu(), native_logits.argmax(dim=-1).detach().cpu()))

        next_token = ref_logits.argmax(dim=-1, keepdim=True)
        ref_next = model(next_token, past_key_values=ref.past_key_values, use_cache=True, logits_to_keep=1, return_dict=True)
        native_next = model.rwkv7_forward_token(next_token, past_key_values=native.past_key_values, return_dict=True)
        decode_max_abs = float((ref_next.logits[:, -1].float() - native_next.logits[:, -1].float()).abs().max().detach().cpu())
        decode_greedy_match = bool(torch.equal(ref_next.logits[:, -1].argmax(dim=-1).detach().cpu(), native_next.logits[:, -1].argmax(dim=-1).detach().cpu()))
        decode_backend = getattr(model, "rwkv7_last_fast_token_backend", lambda: None)()

    for _ in range(args.warmup):
        with torch.inference_mode():
            model(ids, use_cache=True, logits_to_keep=1, return_dict=True)
            model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)

    ref_times: list[float] = []
    native_times: list[float] = []
    with torch.inference_mode():
        for _ in range(args.steps):
            ref_times.append(wall_ms(lambda: model(ids, use_cache=True, logits_to_keep=1, return_dict=True), args.device))
        for _ in range(args.steps):
            native_times.append(wall_ms(lambda: model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True), args.device))

    ref_ms = median(ref_times)
    native_ms = median(native_times)
    peak = None
    if args.device.startswith("cuda"):
        peak = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    return {
        "axis": "native_prefill_scan",
        "backend": "hf_adapter",
        "status": "pass" if greedy_match and decode_greedy_match else "fail",
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "model_path": args.model,
        "model_size_label": infer_model_size_label(args.model),
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "tokens_total": batch_size * prompt_tokens,
        "fused_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SCAN", "0") not in {"0", "false", "False", "no", "off"},
        "scan_block_m": scan_block_m(model),
        "prefill_fused_state_prep_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_prep_effective": native_jit._native_prefill_fused_state_prep_enabled(),
        "prefill_fused_wavg_lora_requested": native_jit._native_prefill_fused_wavg_lora_requested(),
        "prefill_fused_wavg_lora_effective": native_jit._native_prefill_fused_wavg_lora_enabled(batch_size * prompt_tokens),
        "prefill_fused_wavg_lora_max_m": native_jit._native_prefill_fused_wavg_lora_max_m(),
        "fast_token_backend_after_native_prefill": decode_backend,
        "hf_prefill_ms": round(ref_ms, 4),
        "native_prefill_ms": round(native_ms, 4),
        "native_vs_hf_speedup": round(ref_ms / native_ms, 4) if native_ms > 0 else None,
        "hf_prefill_tokps_total": round(1000.0 * batch_size * prompt_tokens / ref_ms, 1) if ref_ms > 0 else None,
        "native_prefill_tokps_total": round(1000.0 * batch_size * prompt_tokens / native_ms, 1) if native_ms > 0 else None,
        "max_abs_diff": round(max_abs, 6),
        "min_cosine": round(min_cos, 8),
        "greedy_match": greedy_match,
        "decode_after_prefill_max_abs_diff": round(decode_max_abs, 6),
        "decode_after_prefill_greedy_match": decode_greedy_match,
        "peak_vram_mb": peak,
    }


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def append_row(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=DTYPES, default="fp16")
    ap.add_argument("--batch-sizes", default="1,4")
    ap.add_argument("--prompt-tokens", default="128")
    ap.add_argument("--fused-scan", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    if args.fused_scan != "auto":
        os.environ["RWKV7_NATIVE_PREFILL_FUSED_SCAN"] = "1" if args.fused_scan == "true" else "0"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=DTYPES[args.dtype],
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    for bsz in parse_ints(args.batch_sizes):
        for prompt_tokens in parse_ints(args.prompt_tokens):
            row = run_case(args, tok, model, bsz, prompt_tokens)
            print(json.dumps(row, ensure_ascii=False))
            append_row(args.results, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
