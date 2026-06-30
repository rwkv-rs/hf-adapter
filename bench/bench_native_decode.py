#!/usr/bin/env python3
# coding=utf-8
"""Benchmark the native RWKV-7 one-token decode path.

This measures `rwkv7_hf.native_jit`, which ports the official per-token
TMix/CMix math into a TorchScript block step and optionally captures the fixed
single-batch greedy decode step in a CUDA graph.  It is intentionally tracked as
an explicit `native_decode` axis: this path is a performance prototype alongside
the full HF `forward` / `generate` compatibility path, not a replacement for
serving-style dynamic batching yet.
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rwkv7_hf.native_jit import cuda_graph_decode, decode_speed, extract, forward, greedy_graph, greedy_jit

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 64


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") else device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    dtype = DTYPES[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    return tok, model


def encode(tok, args: argparse.Namespace) -> torch.Tensor:
    if args.prompt:
        ids = tok(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids
    else:
        ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, : args.prompt_tokens]
    return ids.to(args.device) if args.device.startswith("cuda") else ids


def correctness(model, ids: torch.Tensor, packs, args: argparse.Namespace) -> dict[str, Any]:
    with torch.inference_mode():
        hf = model(ids).logits[0, -1].float()
        native = forward(model, ids, packs).float()
    diff = hf - native
    return {
        "logit_cosine": round(float(F.cosine_similarity(hf.unsqueeze(0), native.unsqueeze(0)).item()), 8),
        "logit_max_abs_diff": round(float(diff.abs().max().item()), 6),
        "logit_mean_abs_diff": round(float(diff.abs().mean().item()), 6),
        "logit_argmax_match": bool(int(hf.argmax() == native.argmax())),
    }


def append_jsonl(path: str | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--greedy-check-tokens", type=int, default=16)
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--skip-graph", action="store_true")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    torch.set_grad_enabled(False)
    tok, model = load_model(args)
    ids = encode(tok, args)
    packs, H, N, eps = extract(model)

    row: dict[str, Any] = {
        "axis": "native_decode",
        "backend": "hf_native_jit",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "prompt_tokens": int(ids.shape[1]),
        "decode_tokens": args.decode_tokens,
        "hidden_size": int(H * N),
        "num_heads": int(H),
        "head_dim": int(N),
        "graph_enabled": not args.skip_graph,
    }

    if not args.skip_correctness:
        row.update(correctness(model, ids, packs, args))

    with torch.inference_mode():
        jit_tokens = greedy_jit(model, ids, packs, n=args.greedy_check_tokens)
        graph_tokens = None
        if not args.skip_graph:
            graph_tokens = greedy_graph(model, ids, packs, n=args.greedy_check_tokens)
        row["greedy_check_tokens"] = args.greedy_check_tokens
        if graph_tokens is not None:
            row["graph_vs_jit_tokens_matched"] = int(sum(int(a == b) for a, b in zip(jit_tokens, graph_tokens)))
            row["graph_vs_jit_tokens_total"] = len(jit_tokens)

        t0 = time.time()
        jit_tokps = decode_speed(model, ids, packs, n=args.decode_tokens)
        row["native_jit_tokps"] = round(float(jit_tokps), 2)
        row["native_jit_ms_per_tok"] = round(1000.0 / float(jit_tokps), 4) if jit_tokps else None
        row["native_jit_wall_s"] = round(time.time() - t0, 4)

        if not args.skip_graph:
            t0 = time.time()
            graph_tokps = cuda_graph_decode(model, ids, packs, n=args.decode_tokens)
            row["native_graph_tokps"] = round(float(graph_tokps), 2)
            row["native_graph_ms_per_tok"] = round(1000.0 / float(graph_tokps), 4) if graph_tokps else None
            row["native_graph_wall_s"] = round(time.time() - t0, 4)

    row["peak_vram_mb"] = peak_mb(args.device)
    print(json.dumps(row, indent=2, ensure_ascii=False))
    append_jsonl(args.results, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
