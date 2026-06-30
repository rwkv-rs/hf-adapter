#!/usr/bin/env python3
# coding=utf-8
"""Quantized inference smoke test for RWKV-7 HF loading.

The adapter should remain loadable through standard HF quantization configs.
This script intentionally exercises the normal HF `forward`/`generate` path
rather than the native-JIT fast-token helper, because bitsandbytes replaces
Linear modules with quantized modules whose packed weights are not compatible
with the native weight extractor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def device_map_for(device: str):
    if not device.startswith("cuda"):
        return None
    if ":" in device:
        return {"": int(device.split(":", 1)[1])}
    return {"": 0}


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def load_model(args, dtype):
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": device_map_for(args.device) if args.device.startswith("cuda") else None,
    }
    if args.quantization != "none":
        if importlib.util.find_spec("bitsandbytes") is None:
            if args.optional:
                print(json.dumps({"axis": "quantized_inference", "quantization": args.quantization, "status": "skip", "reason": "bitsandbytes missing"}))
                return None
            raise RuntimeError("bitsandbytes is required for 8bit/4bit quantized inference")
        from transformers import BitsAndBytesConfig

        if args.quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif args.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            )
        else:  # pragma: no cover - argparse choices
            raise ValueError(args.quantization)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    set_attn_mode(model, args.attn_mode)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="8bit")
    ap.add_argument("--prompt", default="User: Summarize RWKV in one sentence.\n\nAssistant:")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--optional", action="store_true", help="Return success when the quantization backend is not installed/supported")
    ap.add_argument("--bnb-4bit-quant-type", choices=["fp4", "nf4"], default="nf4")
    ap.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.time()
    try:
        model = load_model(args, dtype)
    except Exception as exc:
        if args.optional:
            print(json.dumps({"axis": "quantized_inference", "quantization": args.quantization, "status": "skip", "reason": repr(exc)}))
            return 0
        raise
    if model is None:
        return 0
    cuda_sync(args.device)
    load_s = time.time() - t0

    input_device = next(model.parameters()).device
    enc = tok(args.prompt, return_tensors="pt")
    enc = {k: v.to(input_device) for k, v in enc.items()}

    with torch.no_grad():
        t0 = time.time()
        out = model(**enc, use_cache=True, logits_to_keep=1)
        cuda_sync(args.device)
        forward_s = time.time() - t0
        logits = out.logits.detach().float()
        assert logits.isfinite().all(), "non-finite logits from quantized forward"
        generated = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        cuda_sync(args.device)
    new_tokens = generated[0, -args.max_new_tokens :].detach().cpu().tolist() if args.max_new_tokens > 0 else []
    footprint_mb = None
    if hasattr(model, "get_memory_footprint"):
        footprint_mb = round(float(model.get_memory_footprint()) / 1024 / 1024, 1)
    row = {
        "axis": "quantized_inference",
        "backend": "hf_adapter",
        "quantization": args.quantization,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else args.device,
        "prompt_tokens": int(enc["input_ids"].shape[1]),
        "max_new_tokens": args.max_new_tokens,
        "load_s": round(load_s, 3),
        "forward_ms": round(1000 * forward_s, 3),
        "logits_shape": list(logits.shape),
        "next_token": int(logits[:, -1].argmax(dim=-1)[0].item()),
        "generated_tail": new_tokens,
        "model_footprint_mb": footprint_mb,
        "peak_vram_mb": peak_mb(args.device),
        "status": "pass",
    }
    print(json.dumps(row, ensure_ascii=False))
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
