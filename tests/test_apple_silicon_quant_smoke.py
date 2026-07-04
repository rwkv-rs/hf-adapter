#!/usr/bin/env python3
# coding=utf-8
"""Apple Silicon / MPS native MM8/MM4 quantization smoke.

This is the Apple-side counterpart to the CUDA bitsandbytes path: it exercises
the bitsandbytes-free native MM8/MM4 module replacement on MPS.  Tiny mode
requires no external checkpoint and verifies config-driven from_pretrained
round-trip.  When ``--model`` is supplied it also loads a converted RWKV-7 HF
directory through ``RWKV7_NATIVE_MODEL=1``, applies MM8/MM4 in-place, and runs a
short forward/generate smoke with packed-footprint telemetry.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
os.environ.setdefault("RWKV7_FAST_FORWARD", "0")
os.environ.setdefault("RWKV7_FAST_CACHE", "0")
os.environ.setdefault("RWKV7_FAST_TOKEN_BACKEND", "native_jit")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def append_result(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def emit(path: str, row: dict[str, Any]) -> None:
    print(json.dumps(row, ensure_ascii=False))
    append_result(path, row)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def darwin_sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return "unknown"


def apple_memory_gb() -> int | str:
    raw = darwin_sysctl("hw.memsize")
    try:
        return round(int(raw) / 1024 / 1024 / 1024)
    except Exception:
        return "unknown"


def infer_model_size_label(model_path: str, explicit: str = "") -> str:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(model_path).name.lower())
    return match.group(1) if match else "unknown"


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not mps_is_available(torch):
            raise RuntimeError("requested --device mps but MPS is unavailable")
        return requested
    return "mps" if mps_is_available(torch) else "cpu"


def dtype_for(torch: Any, name: str) -> Any:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]



def mps_backend(torch: Any) -> Any | None:
    return getattr(getattr(torch, "backends", None), "mps", None)


def mps_is_available(torch: Any) -> bool:
    mps = mps_backend(torch)
    if mps is None or not hasattr(mps, "is_available"):
        return False
    try:
        return bool(mps.is_available())
    except Exception:
        return False


def mps_is_built(torch: Any) -> bool:
    mps = mps_backend(torch)
    if mps is None or not hasattr(mps, "is_built"):
        return False
    try:
        return bool(mps.is_built())
    except Exception:
        return False


def mps_memory_stats(torch: Any) -> dict[str, int]:
    if not mps_is_available(torch):
        return {}
    stats: dict[str, int] = {}
    for key, func_name in (
        ("mps_current_allocated_memory_bytes", "current_allocated_memory"),
        ("mps_driver_allocated_memory_bytes", "driver_allocated_memory"),
        ("mps_recommended_max_memory_bytes", "recommended_max_memory"),
    ):
        try:
            stats[key] = int(getattr(torch.mps, func_name)())
        except Exception:
            pass
    return stats


def parse_quantizations(raw: str) -> list[str]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one quantization")
    bad = [v for v in values if v not in {"mm8", "mm4"}]
    if bad:
        raise ValueError(f"unsupported quantization(s): {bad}")
    return values


def module_count(model: Any, class_name: str) -> int:
    return sum(1 for module in model.modules() if type(module).__name__ == class_name)


def selected_linear_weight_bytes(model: Any, torch: Any, min_params: int) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight.numel() >= min_params:
            total += int(module.weight.numel()) * int(module.weight.element_size())
            if module.bias is not None:
                total += int(module.bias.numel()) * int(module.bias.element_size())
    return total


def quantized_linear_buffer_bytes(model: Any, class_name: str) -> int:
    total = 0
    for module in model.modules():
        if type(module).__name__ == class_name:
            for buffer in module.buffers(recurse=False):
                total += int(buffer.numel()) * int(buffer.element_size())
    return total


def configure_quant(model: Any, quantization: str, min_params: int) -> int:
    model.config.use_native_mm8 = quantization == "mm8"
    model.config.use_native_mm4 = quantization == "mm4"
    model.config.native_mm8_min_params = int(min_params)
    model.config.native_mm4_min_params = int(min_params)
    if hasattr(model, "apply_native_mm_quantization_from_config"):
        return int(model.apply_native_mm_quantization_from_config())
    if quantization == "mm8":
        from rwkv7_hf.native_quant_mm8 import quantize_model_mm8

        return int(quantize_model_mm8(model, min_params=min_params))
    from rwkv7_hf.native_quant_mm4 import quantize_model_mm4

    return int(quantize_model_mm4(model, min_params=min_params))


def build_tiny_config(quantization: str):
    from rwkv7_hf.native_model import NativeRWKV7Config

    return NativeRWKV7Config(
        vocab_size=97,
        hidden_size=32,
        num_hidden_layers=2,
        head_dim=8,
        intermediate_size=64,
        decay_low_rank_dim=8,
        gate_low_rank_dim=8,
        a_low_rank_dim=8,
        v_low_rank_dim=8,
        use_cache=True,
        use_native_mm8=quantization == "mm8",
        use_native_mm4=quantization == "mm4",
        native_mm8_min_params=1,
        native_mm4_min_params=1,
    )


def run_tiny_quant(torch: Any, args: argparse.Namespace, device: str, dtype: Any, quantization: str) -> dict[str, Any]:
    from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

    torch.manual_seed(20260704)
    dense = NativeRWKV7ForCausalLM(build_tiny_config(quantization="")).eval()
    source = NativeRWKV7ForCausalLM(build_tiny_config(quantization=quantization)).eval()
    source.load_state_dict(dense.state_dict())
    dense_bytes = selected_linear_weight_bytes(dense, torch, min_params=1)

    tmp = tempfile.mkdtemp(prefix=f"apple_tiny_{quantization}_")
    try:
        source.save_pretrained(tmp)
        model = NativeRWKV7ForCausalLM.from_pretrained(tmp, torch_dtype=dtype).eval().to(device)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    class_name = "MM8Linear" if quantization == "mm8" else "MM4Linear"
    replaced = int(getattr(model, "_rwkv7_native_mm_replaced_modules", 0))
    count = module_count(model, class_name)
    quant_bytes = quantized_linear_buffer_bytes(model, class_name)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        next_id = out.logits[:, -1:].argmax(dim=-1)
        dec = model(next_id, past_key_values=out.past_key_values, use_cache=True)
        gen = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
    elapsed = time.perf_counter() - t0
    assert count >= 1 and replaced == count
    assert out.logits.shape == (1, input_ids.shape[1], model.config.vocab_size)
    assert dec.logits.shape == (1, 1, model.config.vocab_size)
    assert gen.shape[1] == input_ids.shape[1] + args.max_new_tokens
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(dec.logits).all()
    row = {
        "axis": "apple_silicon_native_quant_tiny",
        "status": "pass",
        "quantization": quantization,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "replaced_modules": replaced,
        "module_class": class_name,
        "dense_linear_weight_bytes": dense_bytes,
        "quantized_linear_buffer_bytes": quant_bytes,
        "footprint_ratio": round(quant_bytes / max(dense_bytes, 1), 6),
        "generated_tokens": int(args.max_new_tokens),
        "decode_backend": model.rwkv7_native_model_last_decode_backend(),
        "elapsed_s": round(elapsed, 4),
    }
    row.update(mps_memory_stats(torch))
    return row


def run_model_quant(torch: Any, args: argparse.Namespace, device: str, dtype: Any, quantization: str) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    ).eval()
    model.to(device)
    dense_bytes = selected_linear_weight_bytes(model, torch, min_params=args.min_params)
    replaced = configure_quant(model, quantization, args.min_params)
    class_name = "MM8Linear" if quantization == "mm8" else "MM4Linear"
    count = module_count(model, class_name)
    quant_bytes = quantized_linear_buffer_bytes(model, class_name)
    batch = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    batch = {k: v.to(device) for k, v in batch.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**batch, use_cache=True)
        gen = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or 0,
            eos_token_id=None,
        )
    elapsed = time.perf_counter() - t0
    assert replaced == count and count >= 1
    assert out.logits.shape[0] == batch["input_ids"].shape[0]
    assert gen.shape[1] >= batch["input_ids"].shape[1]
    assert torch.isfinite(out.logits).all()
    row = {
        "axis": "apple_silicon_native_quant_model",
        "status": "pass",
        "model": Path(args.model).name,
        "model_size_label": infer_model_size_label(args.model, args.model_size_label),
        "quantization": quantization,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "min_params": int(args.min_params),
        "replaced_modules": replaced,
        "module_class": class_name,
        "dense_linear_weight_bytes": dense_bytes,
        "quantized_linear_buffer_bytes": quant_bytes,
        "footprint_ratio": round(quant_bytes / max(dense_bytes, 1), 6),
        "prompt_tokens": int(batch["input_ids"].shape[1]),
        "generated_tokens": int(gen.shape[1] - batch["input_ids"].shape[1]),
        "backend_class": model.__class__.__name__,
        "elapsed_s": round(elapsed, 4),
    }
    row.update(mps_memory_stats(torch))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="Optional converted RWKV-7 HF model dir for real-model quant smoke.")
    ap.add_argument("--model-size-label", default="")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--quantizations", default="mm8,mm4")
    ap.add_argument("--min-params", type=int, default=8_000_000)
    ap.add_argument("--max-new-tokens", type=int, default=1)
    ap.add_argument("--prompt", default="User: Apple native quant smoke.\n\nAssistant:")
    ap.add_argument("--results", default="")
    ap.add_argument("--require-apple", action="store_true")
    ap.add_argument("--skip-tiny", action="store_true")
    args = ap.parse_args()

    quantizations = parse_quantizations(args.quantizations)
    if not is_apple_silicon():
        row = {
            "axis": "apple_silicon_native_quant",
            "status": "skip",
            "reason": "not Darwin/arm64",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": Path(args.model).name if args.model else "",
        }
        emit(args.results, row)
        if args.require_apple:
            raise SystemExit(2)
        return 0

    import torch

    device = choose_device(torch, args.device)
    dtype = dtype_for(torch, args.dtype)
    header = {
        "axis": "apple_silicon_native_quant_env",
        "status": "info",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": darwin_sysctl("machdep.cpu.brand_string"),
        "memory_gb": apple_memory_gb(),
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": package_version("transformers"),
        "mps_built": mps_is_built(torch),
        "mps_available": mps_is_available(torch),
        "device": device,
        "dtype": args.dtype,
        "quantizations": quantizations,
        "model": Path(args.model).name if args.model else "",
    }
    header.update(mps_memory_stats(torch))
    emit(args.results, header)

    if not args.skip_tiny:
        for quantization in quantizations:
            emit(args.results, run_tiny_quant(torch, args, device, dtype, quantization))
    if args.model:
        for quantization in quantizations:
            emit(args.results, run_model_quant(torch, args, device, dtype, quantization))

    print("APPLE SILICON NATIVE QUANT SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
