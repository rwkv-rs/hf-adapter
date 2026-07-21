#!/usr/bin/env python3
# coding=utf-8
"""Benchmark HF adapter inference under fp16 / bitsandbytes 8bit / 4bit loads.

Decode compares the reference cached HF forward against the fast-forward path.
On promoted hardware, graph-safe bitsandbytes W8/W4 modules remain live
operands inside native prefill/native-graph execution; unsupported quantizers
or unsafe W8 outlier settings retain the compatibility fallback.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = (
    "RWKV is a recurrent language model with transformer-like parallel training. "
    "This prompt is repeated to create a stable quantization benchmark. "
    * 80
)


def infer_model_size_label(hf_dir: str, explicit: str = "") -> str | None:
    if explicit:
        return explicit.lower()
    name = Path(hf_dir).name.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", name)
    return match.group(1) if match else None


def model_metadata(args: argparse.Namespace, model) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "model_name": Path(args.hf_dir).name,
        "model_size_label": infer_model_size_label(args.hf_dir, args.model_size_label),
        "hf_model_dir": args.hf_dir,
        "vocab_size": getattr(cfg, "vocab_size", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "head_dim": getattr(cfg, "head_dim", None),
        "num_heads": getattr(cfg, "num_heads", None),
    }


def device_map_for(device: str):
    if not device.startswith("cuda"):
        return None
    if ":" in device:
        return {"": int(device.split(":", 1)[1])}
    return {"": 0}


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def encode(tok, n: int) -> torch.LongTensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids
    return ids[:, :n]


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


@contextmanager
def fast_forward_env(enabled: bool):
    old = os.environ.get("RWKV7_FAST_FORWARD")
    old_native_backend = os.environ.get("RWKV7_NATIVE_MODEL_BACKEND")
    os.environ["RWKV7_FAST_FORWARD"] = "1" if enabled else "0"
    if not enabled:
        os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = "eager"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RWKV7_FAST_FORWARD", None)
        else:
            os.environ["RWKV7_FAST_FORWARD"] = old
        if old_native_backend is None:
            os.environ.pop("RWKV7_NATIVE_MODEL_BACKEND", None)
        else:
            os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = old_native_backend


def clone_cache(past_key_values):
    if hasattr(past_key_values, "clone"):
        return past_key_values.clone()
    return past_key_values


def last_fast_token_backend(model):
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


def clear_last_fast_token_backend(model) -> None:
    if hasattr(model, "_rwkv7_last_fast_token_backend"):
        model._rwkv7_last_fast_token_backend = None


def load_model(args: argparse.Namespace, quantization: str, dtype: torch.dtype):
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": device_map_for(args.device) if args.device.startswith("cuda") else None,
    }
    if quantization != "none":
        kwargs["rwkv7_bnb_skip_policy"] = args.quant_skip_policy
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("bitsandbytes missing")
        from transformers import BitsAndBytesConfig

        if quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            )
        else:  # pragma: no cover - argparse choices
            raise ValueError(quantization)
    model = AutoModelForCausalLM.from_pretrained(args.hf_dir, **kwargs).eval()
    set_attn_mode(model, args.attn_mode)
    return model


def quant_module_counts(model) -> dict[str, Any]:
    counts = {
        "linear_dense": 0,
        "linear_8bit": 0,
        "linear_4bit": 0,
        "dense_lora_rank_linear": 0,
        "quantized_lora_rank_linear": 0,
    }
    dense_lora_names: list[str] = []
    quantized_lora_names: list[str] = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        is_lora_rank_linear = "_lora.lora.0" in name or "_lora.lora.2" in name
        if type(module) is torch.nn.Linear:
            counts["linear_dense"] += 1
            if is_lora_rank_linear:
                counts["dense_lora_rank_linear"] += 1
                dense_lora_names.append(name)
        elif "Linear8bit" in cls_name:
            counts["linear_8bit"] += 1
            if is_lora_rank_linear:
                counts["quantized_lora_rank_linear"] += 1
                quantized_lora_names.append(name)
        elif "Linear4bit" in cls_name:
            counts["linear_4bit"] += 1
            if is_lora_rank_linear:
                counts["quantized_lora_rank_linear"] += 1
                quantized_lora_names.append(name)
    counts["dense_lora_rank_linear_sample"] = dense_lora_names[:6]
    counts["quantized_lora_rank_linear_sample"] = quantized_lora_names[:6]
    return counts


def quant_skip_modules(model) -> list[str]:
    qconfig = getattr(getattr(model, "config", None), "quantization_config", None)
    if qconfig is None and getattr(model, "hf_quantizer", None) is not None:
        qconfig = getattr(model.hf_quantizer, "quantization_config", None)
    return list(getattr(qconfig, "llm_int8_skip_modules", None) or [])


def quant_skip_policy(model) -> str:
    return str(getattr(model, "_rwkv7_bnb_skip_policy", None) or getattr(getattr(model, "config", None), "rwkv7_bnb_skip_policy", "memory"))


def bench_one(args: argparse.Namespace, tok, quantization: str, dtype: torch.dtype) -> dict[str, Any]:
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        model = load_model(args, quantization, dtype)
        cuda_sync(args.device)
        load_s = time.time() - t0
    except Exception as exc:
        if args.optional:
            return {
                "axis": "quantization",
                "backend": "hf_adapter",
                "quantization": quantization,
                "dtype": args.dtype,
                "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else args.device,
                "model_name": Path(args.hf_dir).name,
                "model_size_label": infer_model_size_label(args.hf_dir, args.model_size_label),
                "hf_model_dir": args.hf_dir,
                "status": "skip",
                "error": repr(exc),
            }
        raise

    input_device = next(model.parameters()).device
    ids = encode(tok, args.prompt_tokens).to(input_device)
    prompt_tokens = int(ids.shape[1])

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(ids, use_cache=True, logits_to_keep=1)
    cuda_sync(args.device)
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(args.runs):
            out = model(ids, use_cache=True, logits_to_keep=1)
    cuda_sync(args.device)
    prefill_tokps = prompt_tokens / ((time.time() - t0) / args.runs)
    logits = out.logits.detach().float()
    if not logits.isfinite().all():
        raise RuntimeError(f"{quantization} produced non-finite logits")

    def decode_timed(enabled: bool) -> tuple[float, Any, str | None]:
        clear_last_fast_token_backend(model)
        state = clone_cache(out.past_key_values)
        nxt = out.logits[:, -1:].argmax(dim=-1)
        with torch.inference_mode():
            for _ in range(args.warmup):
                with fast_forward_env(enabled):
                    step = model(nxt, past_key_values=state, use_cache=True, logits_to_keep=1)
                state = step.past_key_values
                nxt = step.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(args.decode_tokens):
                with fast_forward_env(enabled):
                    step = model(nxt, past_key_values=state, use_cache=True, logits_to_keep=1)
                state = step.past_key_values
                nxt = step.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        backend = last_fast_token_backend(model) if enabled else None
        return time.time() - t0, step, backend

    with torch.inference_mode():
        nxt = out.logits[:, -1:].argmax(dim=-1)
        with fast_forward_env(False):
            ref_check = model(nxt, past_key_values=clone_cache(out.past_key_values), use_cache=True, logits_to_keep=1)
        with fast_forward_env(True):
            fast_check = model(nxt, past_key_values=clone_cache(out.past_key_values), use_cache=True, logits_to_keep=1)
    fast_diff = float((ref_check.logits.float() - fast_check.logits.float()).abs().max().detach().cpu())
    fast_same_token = bool(torch.equal(ref_check.logits[:, -1].argmax(dim=-1), fast_check.logits[:, -1].argmax(dim=-1)))
    if not fast_check.logits.detach().float().isfinite().all():
        raise RuntimeError(f"{quantization} produced non-finite fast-forward logits")

    ref_decode_s = None
    ref_backend = None
    fast_decode_s = None
    fast_backend = None
    if args.decode_mode in {"reference", "compare"}:
        ref_decode_s, _, ref_backend = decode_timed(False)
    if args.decode_mode in {"fast", "compare"}:
        fast_decode_s, _, fast_backend = decode_timed(True)
    if ref_decode_s is not None and fast_decode_s is not None:
        if fast_decode_s <= ref_decode_s:
            primary_decode_s = fast_decode_s
            selected_decode_path = "fast_forward"
        else:
            primary_decode_s = ref_decode_s
            selected_decode_path = "reference"
    else:
        primary_decode_s = fast_decode_s if fast_decode_s is not None else ref_decode_s
        selected_decode_path = "fast_forward" if fast_decode_s is not None else "reference"
    assert primary_decode_s is not None
    footprint_mb = None
    if hasattr(model, "get_memory_footprint"):
        footprint_mb = round(float(model.get_memory_footprint()) / 1024 / 1024, 1)
    module_counts = quant_module_counts(model)
    reference_decode_tokps = round(args.decode_tokens / ref_decode_s, 1) if ref_decode_s is not None else None
    fast_decode_tokps = round(args.decode_tokens / fast_decode_s, 1) if fast_decode_s is not None else None
    speedup = (
        round(fast_decode_tokps / reference_decode_tokps, 4)
        if fast_decode_tokps is not None and reference_decode_tokps not in (None, 0)
        else None
    )
    return {
        "axis": "quantization",
        "backend": "hf_adapter",
        "quantization": quantization,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else args.device,
        **model_metadata(args, model),
        "attn_mode": getattr(model.config, "attn_mode", "?"),
        "prompt_tokens": prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "prefill_tokps": round(prefill_tokps, 1),
        "decode_mode": args.decode_mode,
        "selected_decode_path": selected_decode_path,
        "decode_tokps": round(args.decode_tokens / primary_decode_s, 1),
        "decode_ms_per_tok": round(1000 * primary_decode_s / args.decode_tokens, 2),
        "reference_decode_tokps": reference_decode_tokps,
        "fast_decode_tokps": fast_decode_tokps,
        "fast_decode_speedup": speedup,
        "fast_forward_backend": fast_backend,
        "reference_forward_backend": ref_backend,
        "fast_forward_max_abs_diff": round(fast_diff, 6),
        "fast_forward_same_next_token": fast_same_token,
        "quant_skip_policy": quant_skip_policy(model),
        "quant_skip_modules": quant_skip_modules(model),
        "module_counts": module_counts,
        "model_footprint_mb": footprint_mb,
        "peak_vram_mb": peak_mb(args.device),
        "load_s": round(load_s, 3),
        "status": "pass",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--model-size-label", default="", help="Optional size label such as 0.1b or 0.4b; inferred from --hf-dir when omitted")
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--quantizations", nargs="+", choices=["none", "8bit", "4bit"], default=["none", "8bit", "4bit"])
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--decode-mode", choices=["reference", "fast", "compare"], default="compare")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--optional", action="store_true", help="Append skip rows instead of failing when a quant backend is unavailable")
    ap.add_argument("--bnb-4bit-quant-type", choices=["fp4", "nf4"], default="nf4")
    ap.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    ap.add_argument(
        "--quant-skip-policy",
        choices=["memory", "decode_hot", "dense"],
        default=os.environ.get("RWKV7_BNB_SKIP_POLICY", "memory"),
        help=(
            "bitsandbytes skip policy: memory keeps only lm_head/small LoRA dense; "
            "decode_hot also keeps attention r/k/v/o projections dense for faster cached decode; "
            "dense keeps all large projections dense as a diagnostic upper bound"
        ),
    )
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()
    if args.decode_tokens <= 0:
        raise ValueError("--decode-tokens must be positive")
    dtype = DTYPES[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)

    rows = []
    for quantization in args.quantizations:
        print(f"\n===== quantization: {quantization} =====", flush=True)
        row = bench_one(args, tok, quantization, dtype)
        rows.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
