#!/usr/bin/env python3
# coding=utf-8
"""HF device_map smoke test for RWKV-7 generate on multiple GPUs.

This validates the HF/Accelerate pipeline-parallel direction: a manually split
RWKV-7 model should run normal cached forward and greedy generate without the
single-device fast-token shortcut crossing CUDA devices.  The optimized
single-device fast path remains enabled globally; the adapter must skip it when
`hf_device_map` spans more than one CUDA device.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV7_FAST_FORWARD", "1")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_by_device() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    out: dict[str, float] = {}
    for i in range(torch.cuda.device_count()):
        try:
            out[str(i)] = round(torch.cuda.max_memory_allocated(i) / 1024 / 1024, 1)
        except Exception:
            out[str(i)] = 0.0
    return out


def reset_peak_stats() -> None:
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(idx)
        except Exception:
            pass


def manual_pp_device_map(num_layers: int, split_layer: int) -> dict[str, int]:
    split_layer = max(1, min(int(split_layer), int(num_layers) - 1))
    device_map: dict[str, int] = {"model.embeddings": 0}
    for layer_idx in range(int(num_layers)):
        device_map[f"model.layers.{layer_idx}"] = 0 if layer_idx < split_layer else 1
    device_map["model.norm"] = 1
    device_map["lm_head"] = 1
    return device_map


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def last_fast_backend(model) -> str | None:
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--attn-mode", choices=["chunk", "fused_recurrent"], default="fused_recurrent")
    ap.add_argument("--prompt", default="User: Hello!\n\nAssistant:")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--split-layer", type=int, default=None)
    ap.add_argument("--compare-single-device", action="store_true")
    ap.add_argument("--optional", action="store_true", help="Skip successfully when fewer than two CUDA devices are visible")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        row = {
            "axis": "device_map_smoke",
            "backend": "hf_adapter",
            "status": "skip",
            "reason": "requires at least two CUDA devices",
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        print(json.dumps(row, ensure_ascii=False))
        return 0 if args.optional else 1

    reset_peak_stats()
    dtype = DTYPES[args.dtype]
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    split_layer = args.split_layer if args.split_layer is not None else max(1, num_layers // 2)
    device_map = manual_pp_device_map(num_layers, split_layer)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=False)
    enc0 = {k: v.to("cuda:0") for k, v in enc.items()}

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    ).eval()
    set_attn_mode(model, args.attn_mode)
    cuda_sync()
    load_s = time.time() - t0

    t0 = time.time()
    with torch.inference_mode():
        out = model(**enc0, use_cache=True, logits_to_keep=1)
        logits_finite = bool(out.logits.detach().float().isfinite().all().item())
        generated = model.generate(**enc0, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
    cuda_sync()
    generate_s = time.time() - t0
    tail = generated[0, -args.max_new_tokens :].detach().cpu().tolist() if args.max_new_tokens else []

    reference_tail = None
    generated_equal_reference = None
    if args.compare_single_device:
        ref = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": 0},
        ).eval()
        set_attn_mode(ref, args.attn_mode)
        with torch.inference_mode():
            ref_generated = ref.generate(**enc0, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        cuda_sync()
        reference_tail = ref_generated[0, -args.max_new_tokens :].detach().cpu().tolist() if args.max_new_tokens else []
        generated_equal_reference = bool(torch.equal(generated.detach().cpu(), ref_generated.detach().cpu()))
        del ref

    last_backend = last_fast_backend(model)
    multi_cuda_device_map = bool(getattr(model, "_rwkv7_has_multi_cuda_device_map")())
    generated_tokens = int(generated.shape[1] - enc["input_ids"].shape[1])
    status = "pass"
    reasons = []
    if not multi_cuda_device_map:
        reasons.append("model was not split across multiple CUDA devices")
    if not logits_finite:
        reasons.append("forward produced non-finite logits")
    if generated_tokens != int(args.max_new_tokens):
        reasons.append(f"generated {generated_tokens} tokens, expected {args.max_new_tokens}")
    if last_backend is not None:
        reasons.append(f"multi-GPU device_map selected single-device backend {last_backend!r}")
    if args.compare_single_device and generated_equal_reference is not True:
        reasons.append("multi-GPU greedy output differs from the single-GPU reference")
    if reasons:
        status = "fail"

    row: dict[str, Any] = {
        "axis": "device_map_smoke",
        "backend": "hf_adapter",
        "status": status,
        "reason": "; ".join(reasons) if reasons else None,
        "dtype": args.dtype,
        "device": ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())),
        "device_count": torch.cuda.device_count(),
        "device_map_kind": "manual_pp_split",
        "split_layer": int(split_layer),
        "num_hidden_layers": num_layers,
        "hf_device_map_devices": sorted({str(v) for v in getattr(model, "hf_device_map", {}).values()}),
        "multi_cuda_device_map": multi_cuda_device_map,
        "fast_forward_env": os.environ.get("RWKV7_FAST_FORWARD", "1"),
        "last_fast_token_backend": last_backend,
        "prompt_tokens": int(enc["input_ids"].shape[1]),
        "max_new_tokens": int(args.max_new_tokens),
        "generated_tokens": generated_tokens,
        "generated_tail": tail,
        "reference_tail": reference_tail,
        "generated_equal_reference": generated_equal_reference,
        "logits_shape": [int(v) for v in out.logits.shape],
        "logits_device": str(out.logits.device),
        "logits_finite": logits_finite,
        "load_s": round(load_s, 3),
        "generate_s": round(generate_s, 4),
        "generate_tokps": round(args.max_new_tokens / generate_s, 2) if generate_s > 0 and args.max_new_tokens else None,
        "peak_vram_mb_by_device": peak_by_device(),
    }
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    if args.results:
        out_path = Path(args.results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"appended 1 row -> {out_path}", flush=True)
    print(status.upper(), flush=True)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
