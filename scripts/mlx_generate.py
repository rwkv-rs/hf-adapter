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
from rwkv7_hf.mlx_model import generate_text_from_hf


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
    ap.add_argument("--quant-backend", default="affine", choices=["affine", "reference", "metal", "auto"])
    ap.add_argument("--require-mlx", action="store_true")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--results", default="", help="Optional JSONL file to append a generation result row.")
    args = ap.parse_args()

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
        }
        print(json.dumps(row, ensure_ascii=False))
        append_result(args.results, row)
        return 2 if args.require_mlx else 0

    reset_mlx_peak_memory()
    output = generate_text_from_hf(
        args.model,
        args.prompt,
        dtype=args.dtype,
        max_new_tokens=int(args.max_new_tokens),
        skip_special_tokens=bool(args.skip_special_tokens),
        quantization=args.quantization,
        quant_min_params=int(args.quant_min_params),
        quant_rkv_min_params=None if int(args.quant_rkv_min_params) < 0 else int(args.quant_rkv_min_params),
        quant_backend=args.quant_backend,
    )
    row = {
        "axis": "mlx_generate",
        "status": "pass",
        "model": Path(args.model).name,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "quant_min_params": int(args.quant_min_params),
        "quant_rkv_min_params": None if int(args.quant_rkv_min_params) < 0 else int(args.quant_rkv_min_params),
        "quant_backend": args.quant_backend,
        "prompt_preview": args.prompt[:80],
        "text": output.text,
        **output.telemetry(),
        **mlx_memory_telemetry(),
    }
    if not args.json_only:
        print(output.text)
    print(json.dumps(row, ensure_ascii=False))
    append_result(args.results, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
