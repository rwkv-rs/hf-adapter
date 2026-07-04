#!/usr/bin/env python3
# coding=utf-8
"""Microbenchmark MLX W8/W4 quantized projection backends.

This harness isolates the Apple quant projection seam from the full recurrent
model.  It times dense fp16 ``x @ W.T`` and the packed quant backends
(reference/affine/metal/auto) for the same random linear weight, then records
correctness versus the quantized reference path and speed ratios versus dense.

The goal is not to claim production speed; it is to make the current Metal
projection kernel bottleneck reproducible before fusing projection + WKV. With
``--groups >1`` it also measures a one-launch grouped projection prototype for
R/K/V-style decode-hot projection groups.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwkv7_hf.mlx_bridge import mlx_available, mlx_memory_telemetry, reset_mlx_peak_memory


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_csv_strs(raw: str) -> list[str]:
    return [x.strip().lower() for x in str(raw).split(",") if x.strip()]


def _append_jsonl(path: str | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _round_float(value: Any, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _time_call(mx: Any, fn, *, warmup: int, runs: int) -> tuple[Any, float]:
    y = None
    for _ in range(max(0, int(warmup))):
        y = fn()
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(max(1, int(runs))):
        y = fn()
        mx.eval(y)
    elapsed = time.perf_counter() - start
    return y, elapsed / max(1, int(runs))


def _cosine(mx: Any, a: Any, b: Any) -> float:
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    denom = mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    value = mx.sum(af * bf) / mx.maximum(denom, 1e-12)
    mx.eval(value)
    return float(value)


def _bench_one(
    mx: Any,
    *,
    bits: int,
    rows: int,
    in_features: int,
    out_features: int,
    dtype: str,
    backends: Iterable[str],
    warmup: int,
    runs: int,
    seed: int,
    results: str | None,
    json_only: bool,
) -> None:
    from rwkv7_hf.mlx_quant import MLXQuantizedLinear, metal_quant_available, mm4_matmul_mlx, mm8_matmul_mlx

    dtype_obj = {"fp16": mx.float16, "fp32": mx.float32, "bf16": mx.bfloat16}[dtype]
    mx.random.seed(int(seed) + int(bits) * 1000 + int(rows))
    x = mx.random.normal((int(rows), int(in_features))).astype(dtype_obj)
    dense_weight = mx.random.normal((int(out_features), int(in_features))).astype(dtype_obj)
    mx.eval(x, dense_weight)

    reset_mlx_peak_memory()
    dense_y, dense_s = _time_call(mx, lambda: x @ dense_weight.T, warmup=warmup, runs=runs)
    mx.eval(dense_y)
    dense_row = {
        "axis": "mlx_quant_projection_bench",
        "status": "pass",
        "backend": "dense_fp",
        "bits": None,
        "dtype": dtype,
        "rows": int(rows),
        "in_features": int(in_features),
        "out_features": int(out_features),
        "warmup": int(warmup),
        "runs": int(runs),
        "avg_s": _round_float(dense_s, 9),
        "avg_ms": _round_float(dense_s * 1000.0, 6),
        "tok_s_equiv": _round_float(float(rows) / dense_s if dense_s > 0 else None, 6),
        "metal_quant_available": bool(metal_quant_available()),
        **mlx_memory_telemetry(),
    }
    _append_jsonl(results, dense_row)
    if not json_only:
        print(json.dumps(dense_row, ensure_ascii=False))

    # Use one reference quantization for correctness and one backend-specific
    # quantized linear for layout/backend selection.  The reference output is the
    # exact packed-quant formula, not the original fp16 dense output.
    ref_backend = "metal" if metal_quant_available() else "affine"
    q_ref = MLXQuantizedLinear.from_linear_weight(dense_weight, bits=int(bits), backend=ref_backend)
    if bits == 8:
        qref_y = mm8_matmul_mlx(x, q_ref.weight, backend="reference")
    else:
        qref_y = mm4_matmul_mlx(x, q_ref.weight, backend="reference")
    mx.eval(qref_y)

    for backend in backends:
        backend = backend.lower().strip()
        if backend == "dense_fp":
            continue
        if backend == "metal" and not metal_quant_available():
            row = {
                "axis": "mlx_quant_projection_bench",
                "status": "skip",
                "reason": "metal quant kernel unavailable",
                "backend": backend,
                "bits": int(bits),
                "dtype": dtype,
                "rows": int(rows),
                "in_features": int(in_features),
                "out_features": int(out_features),
                "metal_quant_available": False,
            }
            _append_jsonl(results, row)
            if not json_only:
                print(json.dumps(row, ensure_ascii=False))
            continue
        q = MLXQuantizedLinear.from_linear_weight(dense_weight, bits=int(bits), backend=backend)
        reset_mlx_peak_memory()
        y, avg_s = _time_call(mx, lambda q=q: q(x), warmup=warmup, runs=runs)
        mx.eval(y)
        max_abs_ref = mx.max(mx.abs(y.astype(mx.float32) - qref_y.astype(mx.float32)))
        max_abs_dense = mx.max(mx.abs(y.astype(mx.float32) - dense_y.astype(mx.float32)))
        mx.eval(max_abs_ref, max_abs_dense)
        tel = q.telemetry()
        actual_backend = tel.get("last_backend") or backend
        row = {
            "axis": "mlx_quant_projection_bench",
            "status": "pass",
            "backend": backend,
            "actual_backend": actual_backend,
            "bits": int(bits),
            "dtype": dtype,
            "rows": int(rows),
            "in_features": int(in_features),
            "out_features": int(out_features),
            "warmup": int(warmup),
            "runs": int(runs),
            "avg_s": _round_float(avg_s, 9),
            "avg_ms": _round_float(avg_s * 1000.0, 6),
            "tok_s_equiv": _round_float(float(rows) / avg_s if avg_s > 0 else None, 6),
            "speedup_vs_dense": _round_float(dense_s / avg_s if avg_s > 0 else None, 6),
            "max_abs_vs_quant_reference": _round_float(float(max_abs_ref), 8),
            "max_abs_vs_dense_fp": _round_float(float(max_abs_dense), 8),
            "cosine_vs_dense_fp": _round_float(_cosine(mx, y, dense_y), 8),
            "storage_bytes": int(tel.get("storage_bytes") or 0),
            "dense_weight_bytes": int(dense_weight.nbytes),
            "footprint_ratio": _round_float(float(tel.get("storage_bytes") or 0) / max(int(dense_weight.nbytes), 1), 6),
            "backend_counts": tel.get("backend_counts"),
            "auto_metal_max_rows": int(tel.get("auto_metal_max_rows") or 0),
            "metal_quant_available": bool(metal_quant_available()),
            **mlx_memory_telemetry(),
        }
        _append_jsonl(results, row)
        if not json_only:
            print(json.dumps(row, ensure_ascii=False))


def _bench_group(
    mx: Any,
    *,
    bits: int,
    rows: int,
    groups: int,
    in_features: int,
    out_features: int,
    dtype: str,
    warmup: int,
    runs: int,
    seed: int,
    results: str | None,
    json_only: bool,
) -> None:
    from rwkv7_hf.mlx_quant import (
        MLXQuantizedLinear,
        metal_quant_available,
        mm4_group_matmul_metal,
        mm4_matmul_mlx,
        mm8_group_matmul_metal,
        mm8_matmul_mlx,
        pack_mlx_mm4_group,
        pack_mlx_mm8_group,
    )

    dtype_obj = {"fp16": mx.float16, "fp32": mx.float32, "bf16": mx.bfloat16}[dtype]
    groups = int(groups)
    if groups <= 1:
        return
    mx.random.seed(int(seed) + int(bits) * 2000 + int(rows) * 17 + groups)
    x = mx.random.normal((int(rows), int(in_features))).astype(dtype_obj)
    dense_weights = [
        mx.random.normal((int(out_features), int(in_features))).astype(dtype_obj)
        for _ in range(groups)
    ]
    mx.eval(x, *dense_weights)

    reset_mlx_peak_memory()
    dense_group_y, dense_group_s = _time_call(
        mx,
        lambda: mx.stack([x @ w.T for w in dense_weights], axis=0),
        warmup=warmup,
        runs=runs,
    )
    mx.eval(dense_group_y)
    dense_bytes = sum(int(w.nbytes) for w in dense_weights)
    dense_row = {
        "axis": "mlx_quant_group_projection_bench",
        "status": "pass",
        "backend": "dense_fp_group",
        "bits": None,
        "dtype": dtype,
        "rows": int(rows),
        "groups": groups,
        "in_features": int(in_features),
        "out_features": int(out_features),
        "warmup": int(warmup),
        "runs": int(runs),
        "avg_s": _round_float(dense_group_s, 9),
        "avg_ms": _round_float(dense_group_s * 1000.0, 6),
        "tok_s_equiv": _round_float(float(rows * groups) / dense_group_s if dense_group_s > 0 else None, 6),
        "metal_quant_available": bool(metal_quant_available()),
        **mlx_memory_telemetry(),
    }
    _append_jsonl(results, dense_row)
    if not json_only:
        print(json.dumps(dense_row, ensure_ascii=False))

    if not metal_quant_available():
        row = {
            "axis": "mlx_quant_group_projection_bench",
            "status": "skip",
            "reason": "metal quant kernel unavailable",
            "backend": "metal_group",
            "bits": int(bits),
            "rows": int(rows),
            "groups": groups,
            "in_features": int(in_features),
            "out_features": int(out_features),
            "metal_quant_available": False,
        }
        _append_jsonl(results, row)
        if not json_only:
            print(json.dumps(row, ensure_ascii=False))
        return

    qlines = [
        MLXQuantizedLinear.from_linear_weight(w, bits=int(bits), backend="metal")
        for w in dense_weights
    ]
    if bits == 8:
        qref_y = mx.stack([mm8_matmul_mlx(x, q.weight, backend="reference") for q in qlines], axis=0)
    else:
        qref_y = mx.stack([mm4_matmul_mlx(x, q.weight, backend="reference") for q in qlines], axis=0)
    mx.eval(qref_y)

    reset_mlx_peak_memory()
    separate_y, separate_s = _time_call(
        mx,
        lambda: mx.stack([q(x) for q in qlines], axis=0),
        warmup=warmup,
        runs=runs,
    )
    mx.eval(separate_y)
    max_abs_sep = mx.max(mx.abs(separate_y.astype(mx.float32) - qref_y.astype(mx.float32)))
    mx.eval(max_abs_sep)
    storage_bytes = sum(int(q.storage_bytes) for q in qlines)
    separate_row = {
        "axis": "mlx_quant_group_projection_bench",
        "status": "pass",
        "backend": "metal_separate",
        "bits": int(bits),
        "dtype": dtype,
        "rows": int(rows),
        "groups": groups,
        "in_features": int(in_features),
        "out_features": int(out_features),
        "warmup": int(warmup),
        "runs": int(runs),
        "avg_s": _round_float(separate_s, 9),
        "avg_ms": _round_float(separate_s * 1000.0, 6),
        "tok_s_equiv": _round_float(float(rows * groups) / separate_s if separate_s > 0 else None, 6),
        "speedup_vs_dense_group": _round_float(dense_group_s / separate_s if separate_s > 0 else None, 6),
        "max_abs_vs_quant_reference": _round_float(float(max_abs_sep), 8),
        "storage_bytes": int(storage_bytes),
        "dense_weight_bytes": int(dense_bytes),
        "footprint_ratio": _round_float(float(storage_bytes) / max(dense_bytes, 1), 6),
        **mlx_memory_telemetry(),
    }
    _append_jsonl(results, separate_row)
    if not json_only:
        print(json.dumps(separate_row, ensure_ascii=False))

    weights = [q.weight for q in qlines]
    group_weight = pack_mlx_mm8_group(weights) if bits == 8 else pack_mlx_mm4_group(weights)
    reset_mlx_peak_memory()
    if bits == 8:
        grouped_fn = lambda: mm8_group_matmul_metal(x, group_weight)
    else:
        grouped_fn = lambda: mm4_group_matmul_metal(x, group_weight)
    grouped_y, grouped_s = _time_call(mx, grouped_fn, warmup=warmup, runs=runs)
    mx.eval(grouped_y)
    max_abs_group = mx.max(mx.abs(grouped_y.astype(mx.float32) - qref_y.astype(mx.float32)))
    max_abs_group_vs_sep = mx.max(mx.abs(grouped_y.astype(mx.float32) - separate_y.astype(mx.float32)))
    mx.eval(max_abs_group, max_abs_group_vs_sep)
    row = {
        "axis": "mlx_quant_group_projection_bench",
        "status": "pass",
        "backend": "metal_group",
        "bits": int(bits),
        "dtype": dtype,
        "rows": int(rows),
        "groups": groups,
        "in_features": int(in_features),
        "out_features": int(out_features),
        "warmup": int(warmup),
        "runs": int(runs),
        "avg_s": _round_float(grouped_s, 9),
        "avg_ms": _round_float(grouped_s * 1000.0, 6),
        "tok_s_equiv": _round_float(float(rows * groups) / grouped_s if grouped_s > 0 else None, 6),
        "speedup_vs_dense_group": _round_float(dense_group_s / grouped_s if grouped_s > 0 else None, 6),
        "speedup_vs_separate_metal": _round_float(separate_s / grouped_s if grouped_s > 0 else None, 6),
        "max_abs_vs_quant_reference": _round_float(float(max_abs_group), 8),
        "max_abs_vs_separate_metal": _round_float(float(max_abs_group_vs_sep), 8),
        "storage_bytes": int(storage_bytes),
        "dense_weight_bytes": int(dense_bytes),
        "footprint_ratio": _round_float(float(storage_bytes) / max(dense_bytes, 1), 6),
        **mlx_memory_telemetry(),
    }
    _append_jsonl(results, row)
    if not json_only:
        print(json.dumps(row, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,4", help="Comma-separated row counts to test.")
    parser.add_argument("--bits", default="4,8", help="Comma-separated quant bit-widths: 4,8.")
    parser.add_argument("--in-features", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=2048)
    parser.add_argument("--dtype", choices=["fp16", "fp32", "bf16"], default="fp16")
    parser.add_argument("--backends", default="reference,affine,metal,auto")
    parser.add_argument("--groups", type=int, default=1, help="If >1, also run a grouped Metal projection microbench.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--require-mlx", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--results")
    args = parser.parse_args()

    if not mlx_available():
        row = {
            "axis": "mlx_quant_projection_bench",
            "status": "skip",
            "reason": "mlx not installed",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "rows": _parse_csv_ints(args.rows),
            "bits": _parse_csv_ints(args.bits),
            "groups": int(args.groups),
            "in_features": int(args.in_features),
            "out_features": int(args.out_features),
            "backends": _parse_csv_strs(args.backends),
        }
        _append_jsonl(args.results, row)
        print(json.dumps(row, ensure_ascii=False))
        return 2 if args.require_mlx else 0

    import mlx.core as mx

    rows_list = _parse_csv_ints(args.rows)
    bits_list = _parse_csv_ints(args.bits)
    backends = _parse_csv_strs(args.backends)
    env_row = {
        "axis": "mlx_quant_projection_bench_env",
        "status": "info",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx_version": getattr(mx, "__version__", None),
        "rows": rows_list,
        "bits": bits_list,
        "groups": int(args.groups),
        "in_features": int(args.in_features),
        "out_features": int(args.out_features),
        "dtype": args.dtype,
        "backends": backends,
        "warmup": int(args.warmup),
        "runs": int(args.runs),
    }
    _append_jsonl(args.results, env_row)
    if not args.json_only:
        print(json.dumps(env_row, ensure_ascii=False))

    for bits in bits_list:
        if bits not in {4, 8}:
            raise ValueError(f"unsupported bits {bits}; expected 4 or 8")
        for rows in rows_list:
            _bench_one(
                mx,
                bits=bits,
                rows=rows,
                in_features=args.in_features,
                out_features=args.out_features,
                dtype=args.dtype,
                backends=backends,
                warmup=args.warmup,
                runs=args.runs,
                seed=args.seed,
                results=args.results,
                json_only=args.json_only,
            )
            _bench_group(
                mx,
                bits=bits,
                rows=rows,
                groups=args.groups,
                in_features=args.in_features,
                out_features=args.out_features,
                dtype=args.dtype,
                warmup=args.warmup,
                runs=args.runs,
                seed=args.seed,
                results=args.results,
                json_only=args.json_only,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
