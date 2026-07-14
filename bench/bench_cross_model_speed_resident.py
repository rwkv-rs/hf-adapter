#!/usr/bin/env python3
"""Single-load RWKV/Qwen speed sweep for large acceptance matrices.

The fresh-process harness remains the final isolation check.  This companion
keeps one model resident while sweeping shapes, amortizing 7B/9B load time.  It
orders cells by batch and prompt, uses one graph per active batch, records every
row in the same schema, and marks resident rows explicitly.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

# Direct execution puts ``bench/`` first on sys.path, where the historical
# ``bench.py`` file shadows the namespace package named ``bench``. Import the
# sibling worker explicitly so both ``python bench/...py`` and module imports
# resolve the same implementation.
BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
import bench_cross_model_speed as speed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-kind", required=True, choices=["rwkv", "qwen35"])
    ap.add_argument("--model-role", required=True, choices=["candidate", "reference"])
    ap.add_argument("--model-pair", required=True)
    ap.add_argument("--model-size-label", required=True)
    ap.add_argument("--benchmark-matrix", default="qwen35_hf")
    ap.add_argument("--dtype", default="fp16", choices=sorted(speed.DTYPES))
    ap.add_argument(
        "--quantization",
        default="none",
        choices=[
            "none",
            "bnb8",
            "bnb4",
            "bnb8_a8w8_head",
            "torchao_w8",
            "torchao_w4",
            "a8w8",
            "mm8",
            "mm4",
        ],
    )
    ap.add_argument("--native-quant-min-params", type=int, default=1_000_000)
    ap.add_argument("--native-quant-policy", choices=["memory", "speed"], default="memory")
    ap.add_argument("--torchao-group-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--prompt-tokens", nargs="+", type=int, default=[128, 512, 2048])
    ap.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        metavar="BxP",
        help="Optional exact batch/prompt pairs, e.g. 2x512 8x128; avoids a Cartesian sweep.",
    )
    ap.add_argument(
        "--cells",
        nargs="+",
        default=None,
        metavar="BxPxD",
        help=(
            "Optional exact batch/prompt/decode cells, e.g. 8x512x512; "
            "avoids both shape/decode Cartesian products for isolated red-cell reruns."
        ),
    )
    ap.add_argument("--decode-tokens", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--prefill-chunk-size", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rwkv-attn-mode", choices=["chunk", "fused_recurrent"], default="fused_recurrent")
    ap.add_argument("--rwkv-code-source", choices=["repo", "model"], default="repo")
    ap.add_argument("--qwen-backend", choices=["auto", "fla", "torch"], default="auto")
    ap.add_argument("--require-qwen-fast-path", action="store_true")
    ap.add_argument(
        "--probe-output",
        default="",
        help="Optional backend-probe output path forwarded to the shared worker.",
    )
    ap.add_argument("--probe-tokens", type=int, default=8)
    ap.add_argument("--results", required=True)
    ap.add_argument("--fail-fast", action="store_true")
    return ap.parse_args()


def resolve_sweep_shapes(args: argparse.Namespace) -> list[tuple[int, int]]:
    if not args.shapes:
        return [
            (int(batch_size), int(prompt_tokens))
            for batch_size in args.batch_sizes
            for prompt_tokens in args.prompt_tokens
        ]
    shapes: list[tuple[int, int]] = []
    for raw in args.shapes:
        try:
            batch_size, prompt_tokens = (int(value) for value in str(raw).lower().split("x"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid --shapes value {raw!r}; expected BxP") from exc
        if batch_size <= 0 or prompt_tokens <= 0:
            raise ValueError(f"invalid --shapes value {raw!r}; dimensions must be positive")
        shape = (batch_size, prompt_tokens)
        if shape not in shapes:
            shapes.append(shape)
    return shapes


def resolve_sweep_cells(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    """Resolve an ordered exact-cell list or the legacy Cartesian sweep."""

    if not getattr(args, "cells", None):
        return [
            (batch_size, prompt_tokens, int(decode_tokens))
            for batch_size, prompt_tokens in resolve_sweep_shapes(args)
            for decode_tokens in args.decode_tokens
        ]
    if getattr(args, "shapes", None):
        raise ValueError("--cells and --shapes are mutually exclusive")
    cells: list[tuple[int, int, int]] = []
    for raw in args.cells:
        try:
            batch_size, prompt_tokens, decode_tokens = (
                int(value) for value in str(raw).lower().split("x")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid --cells value {raw!r}; expected BxPxD") from exc
        if batch_size <= 0 or prompt_tokens <= 0 or decode_tokens <= 0:
            raise ValueError(f"invalid --cells value {raw!r}; dimensions must be positive")
        cell = (batch_size, prompt_tokens, decode_tokens)
        if cell not in cells:
            cells.append(cell)
    return cells


def cell_args(args: argparse.Namespace, batch_size: int, prompt_tokens: int, decode_tokens: int) -> Namespace:
    return Namespace(
        model=args.model,
        model_kind=args.model_kind,
        model_role=args.model_role,
        model_pair=args.model_pair,
        model_size_label=args.model_size_label,
        benchmark_matrix=args.benchmark_matrix,
        dtype=args.dtype,
        quantization=args.quantization,
        native_quant_min_params=args.native_quant_min_params,
        native_quant_policy=args.native_quant_policy,
        torchao_group_size=args.torchao_group_size,
        device=args.device,
        batch_size=int(batch_size),
        prompt_tokens=int(prompt_tokens),
        decode_tokens=int(decode_tokens),
        prefill_chunk_size=int(args.prefill_chunk_size),
        warmup=args.warmup,
        runs=args.runs,
        rwkv_attn_mode=args.rwkv_attn_mode,
        rwkv_code_source=args.rwkv_code_source,
        qwen_backend=args.qwen_backend,
        require_qwen_fast_path=args.require_qwen_fast_path,
        probe_output=args.probe_output,
        probe_tokens=args.probe_tokens,
        results=args.results,
        optional=False,
    )


def clear_shape_caches(model) -> None:
    # Fixed-shape graphs retain pointers to lazy packed projection weights.
    # Once those graphs are gone, release the pack as well so a later batch
    # that deliberately routes to separate GEMMs is measured with its real
    # production footprint and can recover the cuBLAS workspace it owns.
    for name in (
        "rwkv7_clear_native_graph_cache",
        "rwkv7_clear_native_prefill_graph_cache",
        "rwkv7_clear_native_prefill_stacked_rkv_cache",
    ):
        fn = getattr(model, name, None)
        if callable(fn):
            fn()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    for values in (args.batch_sizes, args.prompt_tokens, args.decode_tokens):
        if not values or any(int(value) <= 0 for value in values):
            raise ValueError("all sweep shapes must be positive")

    # Bound graph-pool growth while retaining reuse for both decode lengths at
    # the same (batch, prompt) cell group.
    os.environ.setdefault("RWKV7_NATIVE_GRAPH_CACHE_SIZE", "1")
    os.environ.setdefault("RWKV7_NATIVE_PREFILL_GRAPH_CACHE_SIZE", "1")

    cells = resolve_sweep_cells(args)
    effective_model_path = args.model
    temporary = None
    model = None
    failures = 0
    rows = 0
    started = time.perf_counter()
    try:
        if args.model_kind == "rwkv":
            effective_model_path, temporary = speed.prepare_rwkv_model_dir(
                args.model,
                args.rwkv_code_source,
            )
        tokenizer = speed.AutoTokenizer.from_pretrained(
            effective_model_path,
            trust_remote_code=args.model_kind == "rwkv",
        )
        seed_args = cell_args(
            args,
            cells[0][0],
            cells[0][1],
            cells[0][2],
        )
        speed.validate_args(seed_args)
        model = speed.load_model(seed_args, speed.DTYPES[args.dtype], effective_model_path)
        qwen_contract = speed.enforce_qwen_backend(model, seed_args)
        speed.validate_loaded_model(seed_args, model)
        load_s = time.perf_counter() - started

        total = len(cells)
        active_batch = None
        for batch_size, prompt_tokens, decode_tokens in cells:
            if batch_size != active_batch:
                clear_shape_caches(model)
                active_batch = batch_size
            current = cell_args(args, batch_size, prompt_tokens, decode_tokens)
            speed.validate_args(current)
            try:
                row = speed.benchmark_loaded(
                    current,
                    tokenizer,
                    model,
                    load_s=load_s,
                    qwen_contract=qwen_contract,
                )
            except Exception as exc:
                failures += 1
                row = speed.failure_row(current, exc)
                if args.fail_fast:
                    speed.append_row(args.results, row)
                    print("QWEN35_CROSS_MODEL_SPEED_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
                    raise
            rows += 1
            row["resident_sweep"] = True
            row["resident_cell_index"] = rows
            row["resident_cells_total"] = total
            row["load_amortized"] = rows > 1
            speed.append_row(args.results, row)
            print("QWEN35_CROSS_MODEL_SPEED_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            "QWEN35_RESIDENT_SWEEP_RESULT "
            + json.dumps(
                {
                    "status": "pass" if failures == 0 else "fail",
                    "rows": rows,
                    "failures": failures,
                    "load_s": round(load_s, 3),
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "results": str(Path(args.results)),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1 if failures else 0
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
