#!/usr/bin/env python3
# coding=utf-8
"""Export selected RWKV-7 HF safetensors to MLX safetensors.

This is the first concrete MLX backend artifact: a deterministic bridge from a
converted HuggingFace model directory into an MLX-readable tensor bundle plus a
manifest.  Full MLX RWKV inference will build on the same tensor names.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from rwkv7_hf.mlx_bridge import (
    copy_hf_metadata_files,
    list_hf_safetensor_keys,
    load_selected_hf_tensors_as_mlx,
    mlx_available,
    save_mlx_safetensors,
    summarize_mlx_arrays,
    write_mlx_manifest,
)


def append_result(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Converted RWKV-7 HF model directory")
    ap.add_argument("output", help="Output MLX directory")
    ap.add_argument("--dtype", default="keep", choices=["keep", "fp32", "fp16", "bf16"])
    ap.add_argument(
        "--tensor-regex",
        default="",
        help="Regex selecting tensors to export. Empty means use --include.",
    )
    ap.add_argument(
        "--include",
        action="append",
        default=[],
        help="Exact tensor name to export. Can be repeated.",
    )
    ap.add_argument("--max-tensors", type=int, default=0, help="Optional cap for smoke exports.")
    ap.add_argument("--copy-metadata", action="store_true", help="Copy config/tokenizer metadata files.")
    ap.add_argument("--require-mlx", action="store_true", help="Fail if MLX is not importable.")
    ap.add_argument("--list-keys", action="store_true", help="Print HF safetensor keys and exit.")
    ap.add_argument("--results", default="", help="Optional JSONL file to append an export result row.")
    args = ap.parse_args()

    if args.list_keys:
        for key in list_hf_safetensor_keys(args.model):
            print(key)
        return 0

    if not mlx_available():
        row = {
            "axis": "mlx_hf_export",
            "status": "skip",
            "reason": "mlx not installed",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": Path(args.model).name,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_result(args.results, row)
        if args.require_mlx:
            return 2
        return 0

    include = list(args.include)
    tensor_regex = args.tensor_regex or None
    if not include and tensor_regex is None:
        # Safe default for a lightweight MLX bridge smoke: one real projection
        # matrix is enough to validate HF -> MLX loading and matmul layout.
        include = ["model.layers.0.attn.r_proj.weight"]
    arrays = load_selected_hf_tensors_as_mlx(
        args.model,
        include=include,
        tensor_regex=tensor_regex,
        dtype=args.dtype,
        max_tensors=args.max_tensors or None,
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_file = save_mlx_safetensors(
        arrays,
        out_dir / "model.safetensors",
        metadata={
            "format": "rwkv7_hf_mlx_safetensors_v1",
            "source_model": str(args.model),
            "dtype": args.dtype,
        },
    )
    copied = copy_hf_metadata_files(args.model, out_dir) if args.copy_metadata else []
    manifest = write_mlx_manifest(
        out_dir,
        source_model=args.model,
        arrays=arrays,
        dtype=args.dtype,
        extra={
            "tensor_file": tensor_file.name,
            "copied_metadata_files": copied,
            "selected_tensors": sorted(arrays),
        },
    )
    row = {
        "axis": "mlx_hf_export",
        "status": "pass",
        "model": Path(args.model).name,
        "output": str(out_dir),
        "manifest": str(manifest),
        **summarize_mlx_arrays(arrays),
    }
    print(json.dumps(row, ensure_ascii=False))
    append_result(args.results, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
