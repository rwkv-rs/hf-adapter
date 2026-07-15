#!/usr/bin/env python3
"""Measure dense -> native-quant transition on the same loaded RWKV model.

Separate-process matrix rows remain the deployment acceptance source.  This
companion removes model-load, allocator, and recurrent-body variance when a
speed policy changes only ``lm_head``: it measures exact dense cells, mutates
that same model to the requested native quant backend, clears every captured
graph, and measures the identical cells again.  Both phases retain the normal
cross-model JSON schema and physical model/peak memory telemetry.
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

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
import bench_cross_model_speed as speed
from bench_cross_model_speed_resident import clear_shape_caches, resolve_sweep_cells


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-pair", required=True)
    ap.add_argument("--model-size-label", required=True)
    ap.add_argument("--quantization", required=True, choices=["torchao_w4", "mm4", "a8w8"])
    ap.add_argument("--cells", nargs="+", required=True, metavar="BxPxD")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=sorted(speed.DTYPES))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=9)
    ap.add_argument("--native-quant-min-params", type=int, default=1)
    ap.add_argument("--torchao-group-size", type=int, default=128)
    ap.add_argument("--benchmark-matrix", default="qwen35_native_quant_transition")
    ap.add_argument("--results", required=True)
    return ap.parse_args()


def cell_args(args, batch_size: int, prompt_tokens: int, decode_tokens: int, quantization: str) -> Namespace:
    return Namespace(
        model=args.model,
        model_kind="rwkv",
        model_role="candidate",
        model_pair=args.model_pair,
        model_size_label=args.model_size_label,
        benchmark_matrix=args.benchmark_matrix,
        dtype=args.dtype,
        quantization=quantization,
        native_quant_min_params=args.native_quant_min_params,
        native_quant_policy="speed",
        torchao_group_size=args.torchao_group_size,
        device=args.device,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        prefill_chunk_size=0,
        warmup=args.warmup,
        runs=args.runs,
        rwkv_attn_mode="fused_recurrent",
        rwkv_code_source="repo",
        qwen_backend="auto",
        require_qwen_fast_path=False,
        results=args.results,
        optional=False,
    )


def quantize_in_place(model, args) -> int:
    if args.quantization == "torchao_w4":
        from rwkv7_hf.native_quant_torchao import quantize_model_torchao

        replaced = quantize_model_torchao(
            model,
            "torchao_w4",
            min_params=args.native_quant_min_params,
            policy="speed",
            group_size=args.torchao_group_size,
        )
    elif args.quantization == "mm4":
        from rwkv7_hf.native_quant_mm4 import quantize_model_mm4

        replaced = quantize_model_mm4(
            model, min_params=args.native_quant_min_params, policy="speed"
        )
    else:
        from rwkv7_hf.native_quant_a8w8 import quantize_model_a8w8

        replaced = quantize_model_a8w8(
            model, min_params=args.native_quant_min_params, policy="speed"
        )
    setattr(model, "_rwkv7_cross_model_quant_backend", args.quantization)
    setattr(model, "_rwkv7_cross_model_quant_replaced_modules", int(replaced))
    clear_shape_caches(model)
    return int(replaced)


def run_phase(args, tokenizer, model, cells, *, quantization: str, phase: str, load_s: float) -> int:
    rows = 0
    active_batch = None
    for batch_size, prompt_tokens, decode_tokens in cells:
        if batch_size != active_batch:
            clear_shape_caches(model)
            active_batch = batch_size
        current = cell_args(args, batch_size, prompt_tokens, decode_tokens, quantization)
        speed.validate_args(current)
        row = speed.benchmark_loaded(current, tokenizer, model, load_s=load_s)
        row["quant_transition_same_model"] = True
        row["quant_transition_phase"] = phase
        row["resident_sweep"] = True
        row["resident_cell_index"] = rows + 1
        row["resident_cells_total"] = len(cells)
        row["load_amortized"] = rows > 0
        speed.append_row(args.results, row)
        print("QWEN35_CROSS_MODEL_SPEED_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
        rows += 1
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    args = parse_args()
    if args.dtype != "fp16" and args.quantization == "torchao_w4":
        raise ValueError("the transition bridge specifically validates an fp16 model")
    sweep_args = Namespace(
        cells=args.cells,
        shapes=None,
        batch_sizes=[1],
        prompt_tokens=[128],
        decode_tokens=[128],
    )
    cells = resolve_sweep_cells(sweep_args)
    os.environ.setdefault("RWKV7_FAST_TOKEN_BACKEND", "native_graph")
    os.environ.setdefault("RWKV7_NATIVE_PREFILL_GRAPH", "1")
    os.environ.setdefault("RWKV7_NATIVE_GRAPH_CACHE_SIZE", "1")
    os.environ.setdefault("RWKV7_NATIVE_PREFILL_GRAPH_CACHE_SIZE", "1")
    Path(args.results).unlink(missing_ok=True)

    started = time.perf_counter()
    effective_path, temporary = speed.prepare_rwkv_model_dir(args.model, "repo")
    model = None
    try:
        tokenizer = speed.AutoTokenizer.from_pretrained(effective_path, trust_remote_code=True)
        seed = cell_args(args, *cells[0], "none")
        model = speed.load_model(seed, speed.DTYPES[args.dtype], effective_path)
        speed.validate_loaded_model(seed, model)
        load_s = time.perf_counter() - started
        dense_rows = run_phase(
            args, tokenizer, model, cells, quantization="none", phase="before_quant", load_s=load_s
        )
        replaced = quantize_in_place(model, args)
        quant_rows = run_phase(
            args,
            tokenizer,
            model,
            cells,
            quantization=args.quantization,
            phase="after_quant",
            load_s=load_s,
        )
        print(
            "QWEN35_QUANT_TRANSITION_RESULT "
            + json.dumps(
                {
                    "status": "pass",
                    "dense_rows": dense_rows,
                    "quant_rows": quant_rows,
                    "replaced_modules": replaced,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "results": args.results,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
