#!/usr/bin/env python3
# coding=utf-8
"""Benchmark HF-compatible RWKV speculative decoding with a real draft model.

The correctness contract is target-greedy equality: speculative output must
match `target.generate(..., do_sample=False)` even when the draft is a smaller
RWKV/HF model and acceptance is partial. The row records acceptance telemetry so
future optimization work can decide whether draft choice and block size are
useful for latency.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = int(device.split(":", 1)[1]) if ":" in device else 0
        return torch.cuda.get_device_name(idx)
    return device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def load_model(path: str, dtype: torch.dtype, device: str, attn_mode: str):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device if device.startswith("cuda") else None,
    ).eval()
    set_attn_mode(model, attn_mode)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", required=True, help="Target HF model directory")
    ap.add_argument("--draft-model", required=True, help="Draft HF model directory")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", choices=["chunk", "fused_recurrent"], default="fused_recurrent")
    ap.add_argument("--prompt", default="User: Hello!\n\nAssistant:")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--draft-tokens", type=int, default=4)
    ap.add_argument("--optional", action="store_true", help="Append a skip row when a model path is missing")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    target_path = Path(args.target_model)
    draft_path = Path(args.draft_model)
    if not target_path.exists() or not draft_path.exists():
        row = {
            "axis": "speculative_decode",
            "backend": "hf_adapter",
            "status": "skip",
            "reason": "missing target or draft HF directory",
            "target_hf_dir": args.target_model,
            "draft_hf_dir": args.draft_model,
        }
        print(json.dumps(row, ensure_ascii=False))
        if args.results:
            out = Path(args.results)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return 0 if args.optional else 1

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dtype = DTYPES[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    target = load_model(args.target_model, dtype, args.device, args.attn_mode)
    draft = target if str(target_path.resolve()) == str(draft_path.resolve()) else load_model(args.draft_model, dtype, args.device, args.attn_mode)
    cuda_sync(args.device)

    model_device = next(target.parameters()).device
    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model_device)
    pad_token_id = getattr(tok, "pad_token_id", None) or 0

    with torch.inference_mode():
        t0 = time.time()
        expected = target.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
        cuda_sync(args.device)
        target_s = time.time() - t0

        t0 = time.time()
        spec = target.rwkv7_speculative_generate(
            input_ids,
            draft_model=draft,
            max_new_tokens=args.max_new_tokens,
            draft_tokens=args.draft_tokens,
            return_stats=True,
        )
        cuda_sync(args.device)
        speculative_s = time.time() - t0

    seq = spec["sequences"]
    stats: dict[str, Any] = dict(spec["stats"])
    generated_equal = bool(torch.equal(seq.detach().cpu(), expected.detach().cpu()))
    generated_tokens = int(seq.shape[1] - input_ids.shape[1])
    target_tail = expected[0, -args.max_new_tokens :].detach().cpu().tolist() if args.max_new_tokens else []
    spec_tail = seq[0, -args.max_new_tokens :].detach().cpu().tolist() if args.max_new_tokens else []
    status = "pass" if generated_equal and generated_tokens == args.max_new_tokens else "fail"
    row: dict[str, Any] = {
        "axis": "speculative_decode",
        "backend": "hf_adapter",
        "status": status,
        "dtype": args.dtype,
        "device": device_name(args.device),
        "target_model_name": target_path.name,
        "draft_model_name": draft_path.name,
        "target_hf_dir": args.target_model,
        "draft_hf_dir": args.draft_model,
        "same_model": str(target_path.resolve()) == str(draft_path.resolve()),
        "prompt_tokens": int(input_ids.shape[1]),
        "max_new_tokens": int(args.max_new_tokens),
        "draft_tokens": int(args.draft_tokens),
        "generated_tokens": generated_tokens,
        "generated_equal": generated_equal,
        "target_tail": target_tail,
        "speculative_tail": spec_tail,
        "decoded_preview": tok.decode(seq[0].detach().cpu().tolist(), skip_special_tokens=True)[:160],
        "target_generate_s": round(target_s, 4),
        "speculative_s": round(speculative_s, 4),
        "target_generate_tokps": round(args.max_new_tokens / target_s, 2) if target_s > 0 and args.max_new_tokens else None,
        "speculative_tokps": round(args.max_new_tokens / speculative_s, 2) if speculative_s > 0 and args.max_new_tokens else None,
        "speedup_vs_target_generate": round(target_s / speculative_s, 4) if speculative_s > 0 else None,
        "peak_vram_mb": peak_mb(args.device),
    }
    for key, value in stats.items():
        row[f"stats_{key}"] = value

    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"appended 1 row -> {out}", flush=True)
    if status != "pass":
        raise SystemExit(1)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
