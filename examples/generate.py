#!/usr/bin/env python3
"""Generate text from a converted RWKV-7 Hugging Face model."""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Converted HF model directory or Hub id"
    )
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "metax", "npu", "mps", "musa", "cpu"],
        default="auto",
    )
    parser.add_argument("--dtype", choices=["auto", *DTYPES], default="auto")
    parser.add_argument("--backend", choices=["auto", "native"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _musa_available() -> bool:
    musa = getattr(torch, "musa", None)
    is_available = getattr(musa, "is_available", None)
    try:
        return bool(callable(is_available) and is_available())
    except Exception:
        return False


def _ascend_available() -> bool:
    try:
        from rwkv7_hf.ascend_runtime import ascend_available

        return bool(ascend_available())
    except Exception:
        return False


def _metax_available() -> bool:
    try:
        from rwkv7_hf.metax_runtime import metax_available

        return bool(metax_available())
    except Exception:
        return False


def _enable_metax() -> torch.device:
    from rwkv7_hf.metax_runtime import enable_metax

    info = enable_metax("cuda:0", required=True)
    return torch.device(info.device)


def _enable_ascend() -> torch.device:
    from rwkv7_hf.ascend_runtime import enable_ascend

    info = enable_ascend("npu:0", backend="eager", required=True)
    return torch.device(info.device)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if _metax_available():
            return _enable_metax()
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _ascend_available():
            return _enable_ascend()
        if _musa_available():
            return torch.device("musa")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "musa" and not _musa_available():
        raise RuntimeError("--device musa was requested, but MUSA is unavailable")
    if requested == "npu":
        return _enable_ascend()
    if requested == "metax" or (requested == "cuda" and _metax_available()):
        return _enable_metax()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is unavailable")
    return device


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested != "auto":
        return DTYPES[requested]
    if device.type == "cpu":
        return torch.float32
    if device.type == "npu":
        return torch.bfloat16
    return torch.float16


def select_native_backend(
    requested: str,
) -> bool:
    """Return the canonical user-facing backend selection.

    Converted checkpoints now point directly at the native Auto classes. FLA
    remains available only in dedicated reference benchmarks, where the model
    metadata and effective operator bindings can be checked explicitly.
    """

    if requested not in {"auto", "native"}:
        raise ValueError(f"unsupported user-facing backend: {requested}")
    return True


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
    select_native_backend(args.backend)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(
        f"Loading {args.model} on {device} as {dtype} with native backend",
        file=sys.stderr,
    )

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
