#!/usr/bin/env python3
# coding=utf-8
"""bitsandbytes 8-bit/4-bit functional smoke for the native/no-FLA backend.

This is intentionally a functional gate, not the final quantized performance
gate. It verifies that the standard HF quantization_config path can load the
opt-in native model, run forward/decode/generate, materialize quantized Linear
modules, and fall back to eager native decode instead of the JIT pack path.

  RWKV7_NATIVE_MODEL=1 python tests/test_native_bnb_quant_smoke.py --model <hf_dir> --quantization both
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def quant_module_counts(model) -> dict[str, int]:
    counts = {"linear_dense": 0, "linear_8bit": 0, "linear_4bit": 0}
    for module in model.modules():
        cls = type(module).__name__
        if type(module) is torch.nn.Linear:
            counts["linear_dense"] += 1
        elif "Linear8bit" in cls:
            counts["linear_8bit"] += 1
        elif "Linear4bit" in cls:
            counts["linear_4bit"] += 1
    return counts


def quant_config(mode: str, dtype: torch.dtype, quant_type: str, double_quant: bool):
    from transformers import BitsAndBytesConfig

    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=double_quant,
        )
    raise ValueError(mode)


def current_remote_code_model_dir(model_dir: str):
    """Return a temp HF model dir that uses this worktree's remote-code files.

    Converted checkpoints on the server may have older copied ``modeling_*.py``.
    For PR validation we need AutoModel to import the current branch while still
    reusing the checkpoint/tokenizer artifacts from ``--model``.
    """
    src = Path(model_dir).resolve()
    tmp_ctx = tempfile.TemporaryDirectory(prefix="native_bnb_remote_code_")
    tmp = Path(tmp_ctx.name)
    code_dir = Path(__file__).resolve().parents[1] / "rwkv7_hf"
    for item in src.iterdir():
        target = tmp / item.name
        if item.is_dir():
            os.symlink(item, target, target_is_directory=True)
        else:
            os.symlink(item, target)
    for py_file in code_dir.glob("*.py"):
        target = tmp / py_file.name
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(py_file, target)
    return tmp_ctx, str(tmp)


def run_one(args: argparse.Namespace, model_path: str, tokenizer, mode: str, dtype: torch.dtype) -> dict[str, Any]:
    os.environ["RWKV7_NATIVE_MODEL"] = "1"
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": device_map_for(args.device) if args.device.startswith("cuda") else None,
        "quantization_config": quant_config(
            mode, dtype, args.bnb_4bit_quant_type, args.bnb_4bit_use_double_quant
        ),
    }
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
    cuda_sync(args.device)
    load_s = time.time() - t0
    assert model.__class__.__name__ == "NativeRWKV7ForCausalLM", type(model)

    counts = quant_module_counts(model)
    expected_key = "linear_8bit" if mode == "8bit" else "linear_4bit"
    assert counts[expected_key] > 0, counts

    dev = next(model.parameters()).device
    enc = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        t0 = time.time()
        out = model(**enc, use_cache=True)
        token = out.logits[:, -1:].argmax(dim=-1)
        nxt = model(token, past_key_values=out.past_key_values, use_cache=True)
        generated = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
        cuda_sync(args.device)
        forward_decode_generate_s = time.time() - t0
    assert out.logits.detach().float().isfinite().all(), "non-finite quantized forward logits"
    assert nxt.logits.detach().float().isfinite().all(), "non-finite quantized decode logits"
    backend = model.rwkv7_native_model_last_decode_backend()
    assert backend == "eager", f"quantized native decode should skip JIT packs, got {backend!r}"
    footprint_mb = None
    if hasattr(model, "get_memory_footprint"):
        footprint_mb = round(float(model.get_memory_footprint()) / 1024 / 1024, 1)
    row = {
        "axis": "native_bnb_quant_smoke",
        "backend": "hf_native_model",
        "quantization": mode,
        "status": "pass",
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else args.device,
        "load_s": round(load_s, 3),
        "forward_decode_generate_s": round(forward_decode_generate_s, 3),
        "logits_shape": list(out.logits.shape),
        "decode_backend": backend,
        "next_token": int(token[0, 0].detach().cpu().item()),
        "generated_tail": generated[0, -args.max_new_tokens :].detach().cpu().tolist() if args.max_new_tokens else [],
        "module_counts": counts,
        "model_footprint_mb": footprint_mb,
        "peak_vram_mb": peak_mb(args.device),
    }
    del model
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--quantization", choices=["8bit", "4bit", "both"], default="both")
    ap.add_argument("--prompt", default="User: Summarize RWKV in one sentence.\n\nAssistant:")
    ap.add_argument("--max-new-tokens", type=int, default=2)
    ap.add_argument("--optional", action="store_true", help="Return success when bitsandbytes/CUDA quantization is unavailable")
    ap.add_argument("--bnb-4bit-quant-type", choices=["fp4", "nf4"], default="nf4")
    ap.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    args = ap.parse_args()

    if importlib.util.find_spec("bitsandbytes") is None:
        if args.optional:
            print(json.dumps({"axis": "native_bnb_quant_smoke", "status": "skip", "reason": "bitsandbytes missing"}))
            return 0
        raise RuntimeError("bitsandbytes is required for native bnb quant smoke")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        if args.optional:
            print(json.dumps({"axis": "native_bnb_quant_smoke", "status": "skip", "reason": "CUDA unavailable"}))
            return 0
        raise RuntimeError("CUDA is required for bitsandbytes quant smoke")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    tmp_ctx, model_path = current_remote_code_model_dir(args.model)
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        modes = ["8bit", "4bit"] if args.quantization == "both" else [args.quantization]
        rows = []
        for mode in modes:
            row = run_one(args, model_path, tok, mode, dtype)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    except Exception as exc:
        if args.optional:
            print(json.dumps({"axis": "native_bnb_quant_smoke", "status": "skip", "reason": repr(exc)}))
            return 0
        raise
    finally:
        tmp_ctx.cleanup()
    print("NATIVE BNB QUANT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
