#!/usr/bin/env python3
# coding=utf-8
"""Load and generate from a converted RWKV-7 checkpoint larger than the 0.1B dev model.

This is intentionally a short production-facing smoke benchmark: it proves that
the shape-inferred converter emits a standard HF directory that can be loaded by
AutoConfig/AutoTokenizer/AutoModelForCausalLM, run a cached forward pass, and
complete greedy generation on the target GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = 0
        if ":" in device:
            idx = int(device.split(":", 1)[1])
        return torch.cuda.get_device_name(idx)
    return device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_checkpoint_metadata(path: str | None, sha_arg: str | None, size_arg: int | None) -> tuple[str | None, int | None]:
    if not path:
        return sha_arg, size_arg
    p = Path(path)
    sha = sha_arg
    size = size_arg
    if p.is_file():
        if sha is None:
            sha = sha256_file(p)
        if size is None:
            size = p.stat().st_size
    return sha, size


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def last_fast_token_backend(model) -> str | None:
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


def value_dim_summary(value_dim: Any) -> dict[str, Any]:
    if isinstance(value_dim, (list, tuple)) and value_dim:
        return {
            "value_dim_first": int(value_dim[0]),
            "value_dim_last": int(value_dim[-1]),
            "value_dim_unique": sorted({int(v) for v in value_dim}),
        }
    if value_dim is not None:
        val = int(value_dim)
        return {"value_dim_first": val, "value_dim_last": val, "value_dim_unique": [val]}
    return {"value_dim_first": None, "value_dim_last": None, "value_dim_unique": []}


def config_summary(config) -> dict[str, Any]:
    hidden = int(getattr(config, "hidden_size", 0) or 0)
    head_dim = int(getattr(config, "head_dim", 0) or 0)
    out: dict[str, Any] = {
        "vocab_size": int(getattr(config, "vocab_size", 0) or 0),
        "hidden_size": hidden,
        "intermediate_size": int(getattr(config, "intermediate_size", 0) or 0),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0) or 0),
        "head_dim": head_dim,
        "num_heads": hidden // head_dim if hidden and head_dim else None,
        "attn_mode": getattr(config, "attn_mode", None),
        "fuse_norm": getattr(config, "fuse_norm", None),
        "config_torch_dtype": str(getattr(config, "torch_dtype", None)),
    }
    out.update(value_dim_summary(getattr(config, "value_dim", None)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--model-size-label", required=True, help="Human-readable size label such as 0.4b")
    ap.add_argument("--checkpoint-path", default=None, help="Optional source .pth path used for sha/size provenance")
    ap.add_argument("--checkpoint-sha256", default=None)
    ap.add_argument("--checkpoint-size-bytes", type=int, default=None)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", choices=["chunk", "fused_recurrent", "config"], default="fused_recurrent")
    ap.add_argument("--fast-token-backend", choices=["auto", "fla", "native_jit", "native_graph"], default="auto")
    ap.add_argument("--prompt", default="User: Hello!\n\nAssistant:")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    os.environ["RWKV7_FAST_TOKEN_BACKEND"] = args.fast_token_backend
    os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = args.fast_token_backend
    os.environ.setdefault("RWKV7_FAST_FORWARD", "1")

    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dtype = DTYPES[args.dtype]
    checkpoint_sha, checkpoint_size = maybe_checkpoint_metadata(
        args.checkpoint_path,
        args.checkpoint_sha256,
        args.checkpoint_size_bytes,
    )

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(args.hf_dir, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    if args.attn_mode != "config":
        set_attn_mode(model, args.attn_mode)
    cuda_sync(args.device)
    load_s = time.time() - t0

    model_device = next(model.parameters()).device
    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model_device) for k, v in enc.items()}

    t0 = time.time()
    with torch.inference_mode():
        out = model(**enc, use_cache=True, logits_to_keep=1)
    cuda_sync(args.device)
    forward_s = time.time() - t0
    logits = out.logits.detach().float()
    if not logits.isfinite().all():
        raise RuntimeError("forward produced non-finite logits")
    top5 = logits[0, -1].topk(5).indices.detach().cpu().tolist()

    t0 = time.time()
    with torch.inference_mode():
        gen = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    cuda_sync(args.device)
    generate_s = time.time() - t0
    new_tail = gen[0, -args.max_new_tokens :].detach().cpu().tolist() if args.max_new_tokens else []
    decoded = tok.decode(gen[0].detach().cpu().tolist(), skip_special_tokens=True)
    footprint_mb = None
    if hasattr(model, "get_memory_footprint"):
        footprint_mb = round(float(model.get_memory_footprint()) / 1024 / 1024, 1)

    cfg_fields = config_summary(cfg)
    row: dict[str, Any] = {
        "axis": "larger_model_smoke",
        "backend": "hf_adapter",
        "status": "pass",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "model_size_label": args.model_size_label,
        "model_name": Path(args.hf_dir).name,
        "hf_model_dir": args.hf_dir,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": checkpoint_size,
        "fast_token_backend": args.fast_token_backend,
        "fast_token_backend_effective": last_fast_token_backend(model),
        "prompt_tokens": int(enc["input_ids"].shape[1]),
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": int(gen.shape[1] - enc["input_ids"].shape[1]),
        "logits_shape": [int(v) for v in out.logits.shape],
        "generated_shape": [int(v) for v in gen.shape],
        "top5": top5,
        "generated_tail": new_tail,
        "decoded_preview": decoded[:160],
        "load_s": round(load_s, 3),
        "forward_s": round(forward_s, 4),
        "generate_s": round(generate_s, 4),
        "generate_tokps": round(args.max_new_tokens / generate_s, 2) if generate_s > 0 else None,
        "model_footprint_mb": footprint_mb,
        "peak_vram_mb": peak_mb(args.device),
        "torch_dtype": str(next(model.parameters()).dtype),
    }
    row.update(cfg_fields)

    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    if args.results:
        out_path = Path(args.results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended 1 row -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
