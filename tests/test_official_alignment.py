#!/usr/bin/env python3
# coding=utf-8
"""RWKV-7 HF adapter correctness test against the official `rwkv` package.

This is a smoke-sized acceptance test, not a full eval. It verifies:

1. Prompt last-logit alignment on several prompts.
2. Greedy decode token equality for a configurable window.

For dtype-sensitive checks, fp32 is the correctness reference. fp16/bf16 are allowed
larger max-abs error, but should preserve top-k/argmax behavior.
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time, in a faraway land,",
    "User: Hello!\n\nAssistant:",
    "import torch\nx = torch.randn(",
    "The capital of France is",
]

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
DEFAULT_MAX_ABS = {"fp32": 0.05, "fp16": 0.15, "bf16": 0.70}
DEFAULT_TOP5 = {"fp32": 1.0, "fp16": 0.90, "bf16": 0.80}


def official_forward(model: Any, ids: list[int], state: Any = None):
    out = model.forward(ids, state)
    if isinstance(out, tuple):
        logits, state = out
    else:
        logits, state = out, None
    if logits.dim() > 1:
        logits = logits[-1]
    return logits.float().cpu(), state


def metric_row(hf_logits: torch.Tensor, off_logits: torch.Tensor) -> dict[str, float | int]:
    hf_top5 = set(torch.topk(hf_logits, 5).indices.tolist())
    off_top5 = set(torch.topk(off_logits, 5).indices.tolist())
    return {
        "top5_match": len(hf_top5 & off_top5) / 5,
        "argmax_match": int(hf_logits.argmax().item() == off_logits.argmax().item()),
        "cosine": float(torch.nn.functional.cosine_similarity(hf_logits.unsqueeze(0), off_logits.unsqueeze(0)).item()),
        "max_abs_diff": float((hf_logits - off_logits).abs().max().item()),
        "mean_abs_diff": float((hf_logits - off_logits).abs().mean().item()),
    }


def greedy_window(model, off, prompt_ids: list[int], device: str, window: int) -> dict[str, Any]:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        hf_out = model(ids, use_cache=True, logits_to_keep=1)
    hf_state = hf_out.past_key_values
    hf_logits = hf_out.logits[0, -1].float().cpu()
    off_logits, off_state = official_forward(off, prompt_ids, None)

    matched = 0
    mismatches: list[dict[str, int]] = []
    for step in range(window):
        hf_next = int(hf_logits.argmax().item())
        off_next = int(off_logits.argmax().item())
        if hf_next != off_next:
            mismatches.append({"step": step, "hf": hf_next, "official": off_next})
            break
        matched += 1
        next_ids = torch.tensor([[hf_next]], dtype=torch.long, device=device)
        with torch.no_grad():
            hf_out = model(next_ids, past_key_values=hf_state, use_cache=True, logits_to_keep=1)
        hf_state = hf_out.past_key_values
        hf_logits = hf_out.logits[0, -1].float().cpu()
        off_logits, off_state = official_forward(off, [off_next], off_state)
    return {"requested": window, "matched": matched, "mismatches": mismatches}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--official-strategy", default="cpu fp32")
    ap.add_argument("--greedy-window", type=int, default=64)
    ap.add_argument("--max-abs-target", type=float, default=None)
    ap.add_argument("--top5-target", type=float, default=None)
    ap.add_argument("--cosine-target", type=float, default=0.9999)
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto", help="Override config.fuse_norm for HF load")
    ap.add_argument("--results", default=None, help="Optional JSONL path to append summary")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    max_abs_target = DEFAULT_MAX_ABS[args.dtype] if args.max_abs_target is None else args.max_abs_target
    top5_target = DEFAULT_TOP5[args.dtype] if args.top5_target is None else args.top5_target

    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(args.hf_dir, trust_remote_code=True)
    if args.fuse_norm != "auto":
        cfg.fuse_norm = args.fuse_norm == "true"
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        config=cfg,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()

    from rwkv.model import RWKV
    pth_name = args.pth[:-4] if args.pth.lower().endswith(".pth") else args.pth
    off = RWKV(model=pth_name, strategy=args.official_strategy)

    rows = []
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc.input_ids.to(args.device)
        id_list = enc.input_ids[0].tolist()
        with torch.no_grad():
            hf_out = model(input_ids, use_cache=False, logits_to_keep=1)
        hf_logits = hf_out.logits[0, -1].float().cpu()
        off_logits, _ = official_forward(off, id_list, None)
        row = metric_row(hf_logits, off_logits)
        row["prompt"] = prompt
        rows.append(row)
        print(
            f"{prompt!r:50s} top5={row['top5_match']:.2f} "
            f"argmax={row['argmax_match']} cos={row['cosine']:.6f} "
            f"max={row['max_abs_diff']:.4f} mean={row['mean_abs_diff']:.5f}",
            flush=True,
        )

    greedy = greedy_window(model, off, tok(PROMPTS[2], add_special_tokens=False).input_ids, args.device, args.greedy_window)
    summary = {
        "axis": "official_alignment",
        "ts": int(time.time()),
        "hf_dir": args.hf_dir,
        "pth": args.pth,
        "dtype": args.dtype,
        "official_strategy": args.official_strategy,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "n_prompts": len(rows),
        "top5_match": sum(float(r["top5_match"]) for r in rows) / len(rows),
        "argmax_match": sum(int(r["argmax_match"]) for r in rows) / len(rows),
        "cosine": sum(float(r["cosine"]) for r in rows) / len(rows),
        "max_abs_diff": max(float(r["max_abs_diff"]) for r in rows),
        "mean_abs_diff": sum(float(r["mean_abs_diff"]) for r in rows) / len(rows),
        "greedy_window": greedy,
        "targets": {
            "top5_match": top5_target,
            "cosine": args.cosine_target,
            "max_abs_diff": max_abs_target,
            "greedy_matched": args.greedy_window,
        },
    }
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    failures = []
    if summary["top5_match"] < top5_target:
        failures.append(f"top5_match {summary['top5_match']:.4f} < {top5_target}")
    if summary["cosine"] < args.cosine_target:
        failures.append(f"cosine {summary['cosine']:.6f} < {args.cosine_target}")
    if summary["max_abs_diff"] > max_abs_target:
        failures.append(f"max_abs_diff {summary['max_abs_diff']:.6f} > {max_abs_target}")
    if greedy["matched"] < args.greedy_window:
        failures.append(f"greedy matched {greedy['matched']} < {args.greedy_window}")

    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"appended -> {out}")

    if failures:
        print("FAIL: " + "; ".join(failures), flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
