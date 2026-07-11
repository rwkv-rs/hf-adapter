#!/usr/bin/env python3
# coding=utf-8
"""Greedy text generation with the optional MLX RWKV-7 reference backend."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from rwkv7_hf.mlx_bridge import mlx_available, mlx_memory_telemetry, reset_mlx_peak_memory
from rwkv7_hf.mlx_model import load_mlx_generation_session


def append_result(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Converted RWKV-7 HF model directory.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    ap.add_argument("--skip-special-tokens", action="store_true")
    ap.add_argument("--quantization", default="none", choices=["none", "mm8", "mm4"], help="Optional MLX packed W8/W4 projection path.")
    ap.add_argument("--quant-min-params", type=int, default=8_000_000)
    ap.add_argument("--quant-rkv-min-params", type=int, default=-1, help="Separate min-params threshold for attention r/k/v projection quantization; -1 preserves --quant-min-params.")
    ap.add_argument("--quant-backend", default="affine", choices=["affine", "reference", "metal", "auto", "groupwise"])
    ap.add_argument("--wkv-backend", default="reference", choices=["reference", "metal", "auto"])
    ap.add_argument("--decode-backend", default="auto", choices=["eager", "compiled", "auto"])
    ap.add_argument("--decode-norm-backend", default="reference", choices=["reference", "fast"])
    ap.add_argument("--prepare-compiled-decode", action="store_true")
    ap.add_argument("--compiled-decode-validation-tokens", type=int, default=32)
    ap.add_argument("--compiled-decode-logits-atol", type=float, default=0.0)
    ap.add_argument("--compiled-decode-state-atol", type=float, default=0.0)
    ap.add_argument("--compiled-decode-reference-logits-atol", type=float, default=0.25)
    ap.add_argument("--compiled-decode-reference-state-atol", type=float, default=0.5)
    ap.add_argument("--require-mlx", action="store_true")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--results", default="", help="Optional JSONL file to append a generation result row.")
    args = ap.parse_args()
    if args.compiled_decode_validation_tokens <= 0:
        raise ValueError("--compiled-decode-validation-tokens must be positive")
    if min(
        args.compiled_decode_logits_atol,
        args.compiled_decode_state_atol,
        args.compiled_decode_reference_logits_atol,
        args.compiled_decode_reference_state_atol,
    ) < 0:
        raise ValueError("compiled decode tolerances must be non-negative")

    if not mlx_available():
        row = {
            "axis": "mlx_generate",
            "status": "skip",
            "reason": "mlx not installed",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": Path(args.model).name,
            "quantization": args.quantization,
            "quant_min_params": int(args.quant_min_params),
            "quant_rkv_min_params": None if int(args.quant_rkv_min_params) < 0 else int(args.quant_rkv_min_params),
            "quant_backend": args.quant_backend,
            "wkv_backend": args.wkv_backend,
            "decode_backend": args.decode_backend,
            "decode_norm_backend": args.decode_norm_backend,
            "prepare_compiled_decode": bool(args.prepare_compiled_decode),
        }
        print(json.dumps(row, ensure_ascii=False))
        append_result(args.results, row)
        return 2 if args.require_mlx else 0

    reset_mlx_peak_memory()
    session = load_mlx_generation_session(
        args.model,
        args.prompt,
        dtype=args.dtype,
        skip_special_tokens=bool(args.skip_special_tokens),
        quantization=args.quantization,
        quant_min_params=int(args.quant_min_params),
        quant_rkv_min_params=None if int(args.quant_rkv_min_params) < 0 else int(args.quant_rkv_min_params),
        quant_backend=args.quant_backend,
        wkv_backend=args.wkv_backend,
        decode_backend=args.decode_backend,
        decode_norm_backend=args.decode_norm_backend,
        prepare_compiled_decode=bool(args.prepare_compiled_decode),
        compiled_decode_validation_tokens=int(args.compiled_decode_validation_tokens),
        compiled_decode_logits_atol=float(args.compiled_decode_logits_atol),
        compiled_decode_state_atol=float(args.compiled_decode_state_atol),
        compiled_decode_reference_logits_atol=float(
            args.compiled_decode_reference_logits_atol
        ),
        compiled_decode_reference_state_atol=float(
            args.compiled_decode_reference_state_atol
        ),
    )
    session.decode(int(args.max_new_tokens))
    output = session.output()
    row = {
        "axis": "mlx_generate",
        "status": "pass",
        "model": Path(args.model).name,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "quant_min_params": int(args.quant_min_params),
        "quant_rkv_min_params": None if int(args.quant_rkv_min_params) < 0 else int(args.quant_rkv_min_params),
        "quant_backend": args.quant_backend,
        "wkv_backend": args.wkv_backend,
        "decode_backend": args.decode_backend,
        "decode_norm_backend": args.decode_norm_backend,
        "prepare_compiled_decode": bool(args.prepare_compiled_decode),
        "prompt_preview": args.prompt[:80],
        "text": output.text,
        **output.telemetry(),
        **session.model.telemetry(),
        **mlx_memory_telemetry(),
    }
    if not args.json_only:
        print(output.text)
    print(json.dumps(row, ensure_ascii=False))
    append_result(args.results, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
