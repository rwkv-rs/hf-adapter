#!/usr/bin/env python3
# coding=utf-8
"""RWKV-7 HF adapter benchmark — precision axis (iteration 1).

Compares the HF adapter path against the official `rwkv` package on the SAME
input token ids. Both models are the same weights (HF dir is converted from the
same .pth), so any divergence is purely the HF/FLA math path vs the official
WKV7 path.

Reference side runs on CPU fp32 (exact) so the comparison isolates the HF fp16
path. Results are appended to bench/results.jsonl, one JSON line per run.

Usage:
  python bench/bench.py --hf-dir <hf model dir> --pth <official .pth>
                        [--dtype fp16] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Official rwkv package needs this for v7 models.
os.environ.setdefault("RWKV_V7_ON", "1")

TARGET = {
    "top5_match": 1.0,
    "cosine": 0.9999,
    "max_abs_diff": 0.05,
    "greedy_window": 64,
}

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time, in a faraway land,",
    "User: Hello!\n\nAssistant:",
    "import torch\nx = torch.randn(",
    "The capital of France is",
]


def hf_last_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    return out.logits[0, -1].float().cpu()


def official_logits(rwkv_model, ids: list[int]) -> torch.Tensor:
    out = rwkv_model.forward(ids, None)
    logits = out[0] if isinstance(out, tuple) else out
    if logits.dim() > 1:
        logits = logits[-1]
    return logits.float().cpu()


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    top5_a = set(torch.topk(a, 5).indices.tolist())
    top5_b = set(torch.topk(b, 5).indices.tolist())
    cos = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()
    max_abs = (a - b).abs().max().item()
    return {
        "top5_match": len(top5_a & top5_b) / 5,
        "cosine": cos,
        "max_abs_diff": max_abs,
        "argmax_match": int(a.argmax().item() == b.argmax().item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--official-strategy", default="cpu fp32",
                    help="official rwkv strategy (cpu fp32 = exact reference)")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]

    print(f"[hf] loading {args.hf_dir} ({args.dtype} on {args.device})...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, trust_remote_code=True, torch_dtype=dtype,
        device_map=args.device,
    ).eval()

    # The official rwkv package appends '.pth' internally, so strip it.
    pth_name = args.pth[:-4] if args.pth.lower().endswith(".pth") else args.pth
    print(f"[official] loading {pth_name}.pth ({args.official_strategy})...", flush=True)
    from rwkv.model import RWKV
    off = RWKV(model=pth_name, strategy=args.official_strategy)

    rows = []
    for p in PROMPTS:
        enc = tok(p, return_tensors="pt", add_special_tokens=False)
        ids = enc.input_ids.to(args.device)
        id_list = enc.input_ids[0].tolist()
        a = hf_last_logits(model, ids)
        b = official_logits(off, id_list)
        if a.shape != b.shape:
            print(f"  SHAPE MISMATCH prompt={p!r}: hf {a.shape} vs off {b.shape}")
            continue
        m = metrics(a, b)
        m["prompt"] = p
        rows.append(m)
        print(f"  {p!r:50s} top5={m['top5_match']:.2f} "
              f"cos={m['cosine']:.6f} maxabs={m['max_abs_diff']:.4f} "
              f"argmax={m['argmax_match']}")

    if not rows:
        print("No usable rows; aborting.", flush=True)
        return 1

    summary = {
        "axis": "precision",
        "ts": int(time.time()),
        "hf_dir": args.hf_dir,
        "pth": args.pth,
        "dtype": args.dtype,
        "official_strategy": args.official_strategy,
        "n_prompts": len(rows),
        "top5_match": sum(r["top5_match"] for r in rows) / len(rows),
        "cosine": sum(r["cosine"] for r in rows) / len(rows),
        "max_abs_diff": max(r["max_abs_diff"] for r in rows),
        "argmax_match": sum(r["argmax_match"] for r in rows) / len(rows),
    }
    print("\n=== summary vs target ===", flush=True)
    for k in ("top5_match", "cosine", "max_abs_diff"):
        v = summary[k]
        t = TARGET[k]
        ok = (v >= t) if k != "max_abs_diff" else (v <= t)
        print(f"  {k:14s} {v:.6f}  (target {t})  {'PASS' if ok else 'FAIL'}")

    out = Path(__file__).parent / "results.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"\nappended -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
