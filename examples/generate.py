#!/usr/bin/env python3
"""Generate text from a converted RWKV-7 Hugging Face model."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
TRUE_VALUES = {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Converted HF model directory or Hub id")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", *DTYPES], default="auto")
    parser.add_argument("--backend", choices=["auto", "fla", "native"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is unavailable")
    return device


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested != "auto":
        return DTYPES[requested]
    return torch.float32 if device.type == "cpu" else torch.float16


def select_native_backend(
    requested: str,
    *,
    device_type: str,
    fla_available: bool,
    native_env_enabled: bool,
) -> bool:
    if requested == "native":
        return True
    if requested == "fla":
        if not fla_available:
            raise RuntimeError("--backend fla requires flash-linear-attention")
        return False
    return native_env_enabled or device_type != "cuda" or not fla_available


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    fla_available = importlib.util.find_spec("fla") is not None
    native_env = os.environ.get("RWKV7_NATIVE_MODEL", "0").strip().lower() in TRUE_VALUES
    use_native = select_native_backend(
        args.backend,
        device_type=device.type,
        fla_available=fla_available,
        native_env_enabled=native_env,
    )
    if use_native:
        os.environ["RWKV7_NATIVE_MODEL"] = "1"
    else:
        os.environ.pop("RWKV7_NATIVE_MODEL", None)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    backend = "native" if use_native else "fla"
    print(f"Loading {args.model} on {device} as {dtype} with {backend} backend", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).eval()
    model.to(device)

    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    generation = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.temperature > 0:
        generation.update({"temperature": args.temperature, "top_p": args.top_p})

    with torch.inference_mode():
        output = model.generate(**encoded, **generation)
    prompt_length = int(encoded["input_ids"].shape[1])
    answer = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
