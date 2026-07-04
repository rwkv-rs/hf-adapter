#!/usr/bin/env python3
# coding=utf-8
"""Apple Silicon MLX bridge smoke for RWKV-7 HF checkpoints.

This does not claim a full MLX RWKV backend yet. It proves the first native-MLX
building block: selected tensors from a converted HF checkpoint can be loaded as
MLX arrays, saved as MLX safetensors, and used in MLX matmul on Apple Silicon.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from rwkv7_hf.mlx_bridge import (
    load_selected_hf_tensors_as_mlx,
    mlx_array_nbytes,
    mlx_available,
    require_mlx,
    save_mlx_safetensors,
    summarize_mlx_arrays,
)


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


def run_tiny_mlx(args: argparse.Namespace) -> dict[str, Any]:
    mx = require_mlx()
    t0 = time.perf_counter()
    hidden = 16
    out_features = 24
    x = mx.arange(hidden, dtype=mx.float32).reshape(1, hidden) / hidden
    w = mx.arange(out_features * hidden, dtype=mx.float32).reshape(out_features, hidden) / (hidden * out_features)
    y = x @ w.T
    mx.eval(y)
    with tempfile.TemporaryDirectory(prefix="rwkv7_mlx_tiny_") as tmp:
        path = Path(tmp) / "tiny.safetensors"
        save_mlx_safetensors({"x": x, "w": w, "y": y}, path, metadata={"format": "rwkv7_mlx_tiny_smoke"})
        loaded = mx.load(path)
        y2 = loaded["x"] @ loaded["w"].T
        mx.eval(y2)
        assert bool(mx.allclose(y, y2))
    elapsed = time.perf_counter() - t0
    arrays = {"x": x, "w": w, "y": y}
    return {
        "axis": "apple_silicon_mlx_tiny",
        "status": "pass",
        "dtype": str(y.dtype),
        "output_shape": list(y.shape),
        "elapsed_s": round(elapsed, 6),
        **summarize_mlx_arrays(arrays),
    }


def run_hf_projection(args: argparse.Namespace) -> dict[str, Any]:
    mx = require_mlx()
    tensor_name = args.tensor_name
    t0 = time.perf_counter()
    arrays = load_selected_hf_tensors_as_mlx(args.model, include=[tensor_name], dtype=args.dtype)
    w = arrays[tensor_name]
    hidden = int(w.shape[1])
    x = mx.ones((int(args.batch_size), hidden), dtype=w.dtype)
    y = x @ w.T
    mx.eval(y)
    elapsed = time.perf_counter() - t0
    assert y.shape == (int(args.batch_size), int(w.shape[0]))
    assert bool(mx.all(mx.isfinite(y)))
    summary = summarize_mlx_arrays(arrays)
    return {
        "axis": "apple_silicon_mlx_projection_smoke",
        "status": "pass",
        "model": Path(args.model).name,
        "model_size_label": infer_model_size_label(args.model, args.model_size_label),
        "tensor": tensor_name,
        "dtype": str(w.dtype),
        "batch_size": int(args.batch_size),
        "input_features": hidden,
        "output_features": int(w.shape[0]),
        "projection_output_shape": list(y.shape),
        "selected_tensor_bytes": mlx_array_nbytes(w),
        "elapsed_s": round(elapsed, 6),
        **summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="Optional converted RWKV-7 HF model dir.")
    ap.add_argument("--model-size-label", default="")
    ap.add_argument("--tensor-name", default="model.layers.0.attn.r_proj.weight")
    ap.add_argument("--dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--results", default="")
    ap.add_argument("--require-apple", action="store_true")
    ap.add_argument("--require-mlx", action="store_true")
    ap.add_argument("--skip-tiny", action="store_true")
    args = ap.parse_args()

    if not is_apple_silicon():
        row = {
            "axis": "apple_silicon_mlx_smoke",
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

    if not mlx_available():
        row = {
            "axis": "apple_silicon_mlx_smoke",
            "status": "skip",
            "reason": "mlx not installed",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": Path(args.model).name if args.model else "",
        }
        emit(args.results, row)
        if args.require_mlx:
            raise SystemExit(2)
        return 0

    import mlx.core as mx

    header = {
        "axis": "apple_silicon_mlx_env",
        "status": "info",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": darwin_sysctl("machdep.cpu.brand_string"),
        "memory_gb": apple_memory_gb(),
        "mlx": package_version("mlx"),
        "mlx_default_device": str(mx.default_device()),
        "dtype": args.dtype,
        "model": Path(args.model).name if args.model else "",
    }
    emit(args.results, header)

    if not args.skip_tiny:
        emit(args.results, run_tiny_mlx(args))
    if args.model:
        emit(args.results, run_hf_projection(args))

    print("APPLE SILICON MLX SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
