#!/usr/bin/env python3
# coding=utf-8
"""Micro-benchmark the prefill shift-mix + W/A/G/V LoRA fusion.

The end-to-end native prefill profiler sees ``attn_shift_wavg_lora_fused`` as
the largest non-scan component after the shift-WAVG route.  This benchmark
uses the actual checkpoint layer weights but random ``h``/``prev_h`` inputs to
split the fused route into:

* down/materialize phase: time-mix on the fly, produce ``xr/xk/xv`` for cuBLAS
  R/K/V, and produce W/A/G/V low-rank intermediates;
* up phase: low-rank up projections and output activations;
* full helper: the production helper that calls both phases.

It is an evidence probe only; HF end-to-end rows remain the promotion gate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from transformers import AutoModelForCausalLM

from bench_native_prefill_scan import prepare_model_dir
from rwkv7_hf import native_jit
from rwkv7_hf.fused_lora import (
    _shift_wavg_lora_down_kernel,
    _wavg_lora_up_kernel,
    fused_shift_wavg_lora,
)

try:  # pragma: no cover - remote CUDA/Triton probe
    import triton
except Exception as exc:  # pragma: no cover
    raise RuntimeError("bench_shift_wavg_lora_micro requires triton") from exc


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def median(vals: list[float]) -> float:
    vals = sorted(vals)
    return vals[len(vals) // 2]


def time_cuda(fn: Callable[[], Any], *, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        vals.append(float(start.elapsed_time(end)))
    return median(vals)


def append_row(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=DTYPES, default="fp16")
    ap.add_argument("--code-source", choices=["model", "repo"], default="repo")
    ap.add_argument("--layer-index", type=int, default=1)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--block-m", type=int, default=128)
    ap.add_argument("--block-r", type=int, default=64)
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--down-warps", type=int, default=4)
    ap.add_argument("--up-warps", type=int, default=4)
    ap.add_argument("--lean-down", action="store_true")
    ap.add_argument("--lean-up", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    if not args.device.startswith("cuda"):
        raise ValueError("bench_shift_wavg_lora_micro currently requires cuda")
    torch.manual_seed(1234)
    effective_model_path, tmp_model_dir = prepare_model_dir(args.model, code_source=args.code_source)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            effective_model_path,
            trust_remote_code=True,
            torch_dtype=DTYPES[args.dtype],
            device_map=args.device,
        ).eval()
        packs = model._rwkv7_native_jit_packs()
        p = native_jit._ensure_rkv_pack(packs[int(args.layer_index)])
        (
            i,
            H,
            N,
            _eps,
            _has_pre,
            _pre_w,
            _pre_b,
            _an_w,
            _an_b,
            _fn_w,
            _fn_b,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            _k_k,
            _k_a,
            _r_k,
            _Rw,
            _Kw,
            _Vw,
            _Ow,
            w1,
            w2,
            w0,
            a1,
            a2,
            a0,
            v1,
            v2,
            v0,
            g1,
            g2,
            _gn_w,
            _gn_b,
            _fx_k,
            _fK,
            _fV,
            _RKVw,
        ) = p
        H = int(H)
        N = int(N)
        hidden = H * N
        rows = int(args.rows)
        dtype = DTYPES[args.dtype]
        device = torch.device(args.device)
        h = torch.randn(rows, hidden, device=device, dtype=dtype)
        prev_h = torch.randn(rows, hidden, device=device, dtype=dtype)

        w_rank = int(w1.shape[0])
        a_rank = int(a1.shape[0])
        g_rank = int(g1.shape[0])
        v_rank = int(v1.shape[0])
        max_rank = max(w_rank, a_rank, g_rank, v_rank)

        def alloc_phase_tensors():
            w_mid = torch.empty((rows, w_rank), device=device, dtype=dtype)
            a_mid = torch.empty((rows, a_rank), device=device, dtype=dtype)
            g_mid = torch.empty((rows, g_rank), device=device, dtype=dtype)
            v_mid = torch.empty((rows, v_rank), device=device, dtype=dtype)
            xr = torch.empty((rows, hidden), device=device, dtype=dtype)
            xk = torch.empty_like(xr)
            xv = torch.empty_like(xr)
            w_out = torch.empty_like(xr)
            a_out = torch.empty_like(xr)
            g_out = torch.empty_like(xr)
            v_gate = torch.empty_like(xr)
            return w_mid, a_mid, g_mid, v_mid, xr, xk, xv, w_out, a_out, g_out, v_gate

        def run_down(tensors):
            w_mid, a_mid, g_mid, v_mid, xr, xk, xv, *_ = tensors
            _shift_wavg_lora_down_kernel[(rows, triton.cdiv(max_rank, int(args.block_r)))](
                h,
                prev_h,
                x_r,
                x_w,
                x_k,
                x_v,
                x_a,
                x_g,
                w1,
                a1,
                g1,
                v1,
                w_mid,
                a_mid,
                g_mid,
                v_mid,
                xr,
                xk,
                xv,
                hidden,
                w_rank,
                a_rank,
                g_rank,
                v_rank,
                max_rank,
                BLOCK_R=int(args.block_r),
                BLOCK_K=int(args.block_k),
                LEAN_DOWN=bool(args.lean_down),
                num_warps=int(args.down_warps),
            )

        def run_up(tensors):
            w_mid, a_mid, g_mid, v_mid, _xr, _xk, _xv, w_out, a_out, g_out, v_gate = tensors
            _wavg_lora_up_kernel[(rows, triton.cdiv(hidden, int(args.block_m)))](
                w_mid,
                a_mid,
                g_mid,
                v_mid,
                w2,
                a2,
                g2,
                v2,
                w0,
                a0,
                g2,
                v0,
                w_out,
                a_out,
                g_out,
                v_gate,
                hidden,
                w_rank,
                a_rank,
                g_rank,
                v_rank,
                max_rank,
                HAS_W_BIAS=True,
                HAS_A_BIAS=True,
                HAS_G_BIAS=False,
                HAS_V_BIAS=True,
                OUTPUT_W_DECAY=False,
                LEAN_UP=bool(args.lean_up),
                BLOCK_M=int(args.block_m),
                BLOCK_R=int(args.block_r),
                num_warps=int(args.up_warps),
            )

        def run_full():
            return fused_shift_wavg_lora(
                h,
                prev_h,
                x_r,
                x_w,
                x_k,
                x_v,
                x_a,
                x_g,
                w1,
                a1,
                g1,
                v1,
                w2,
                a2,
                g2,
                v2,
                w0,
                a0,
                None,
                v0,
                block_m=int(args.block_m),
                block_r=int(args.block_r),
                block_k=int(args.block_k),
                down_num_warps=int(args.down_warps),
                up_num_warps=int(args.up_warps),
                lean_down=bool(args.lean_down),
                lean_up=bool(args.lean_up),
            )

        # Correctness against torch fallback and explicit phase composition.
        full = run_full()
        ref = fused_shift_wavg_lora(
            h,
            prev_h,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            w1,
            a1,
            g1,
            v1,
            w2,
            a2,
            g2,
            v2,
            w0,
            a0,
            None,
            v0,
            block_m=int(args.block_m),
            block_r=int(args.block_r),
            block_k=int(args.block_k),
            force_fallback=True,
        )
        max_diff = max(float((a.float() - b.float()).abs().max().detach().cpu()) for a, b in zip(full, ref))
        phase_tensors = alloc_phase_tensors()
        run_down(phase_tensors)
        run_up(phase_tensors)
        phase_out = (phase_tensors[4], phase_tensors[5], phase_tensors[6], phase_tensors[7], phase_tensors[8], phase_tensors[9], phase_tensors[10])
        phase_diff = max(float((a.float() - b.float()).abs().max().detach().cpu()) for a, b in zip(phase_out, full))
        status = "pass" if max_diff <= 0.125 and phase_diff == 0.0 else "fail"

        down_tensors = alloc_phase_tensors()
        run_down(down_tensors)
        down_ms = time_cuda(lambda: run_down(down_tensors), warmup=args.warmup, steps=args.steps)
        up_ms = time_cuda(lambda: run_up(down_tensors), warmup=args.warmup, steps=args.steps)
        full_ms = time_cuda(run_full, warmup=args.warmup, steps=args.steps)

        common = {
            "axis": "shift_wavg_lora_micro",
            "backend": "hf_adapter",
            "status": status,
            "device": torch.cuda.get_device_name(0),
            "dtype": args.dtype,
            "model_path": args.model,
            "effective_model_path": effective_model_path,
            "code_source": args.code_source,
            "layer_index": int(i),
            "rows": rows,
            "hidden": hidden,
            "heads": H,
            "head_dim": N,
            "w_rank": w_rank,
            "a_rank": a_rank,
            "g_rank": g_rank,
            "v_rank": v_rank,
            "max_rank": max_rank,
            "block_m": int(args.block_m),
            "block_r": int(args.block_r),
            "block_k": int(args.block_k),
            "down_warps": int(args.down_warps),
            "up_warps": int(args.up_warps),
            "lean_down": bool(args.lean_down),
            "lean_up": bool(args.lean_up),
            "max_abs_diff_vs_fallback": round(max_diff, 6),
            "phase_max_abs_diff_vs_full": round(phase_diff, 6),
        }
        for phase, ms in (("down", down_ms), ("up", up_ms), ("full", full_ms), ("summary", full_ms)):
            row = dict(common)
            row.update(
                {
                    "phase": phase,
                    "cuda_ms": round(float(ms), 6),
                    "rows_per_ms": round(float(rows) / float(ms), 3) if ms > 0 else None,
                    "down_ms": round(float(down_ms), 6),
                    "up_ms": round(float(up_ms), 6),
                    "full_ms": round(float(full_ms), 6),
                    "down_ratio_of_full": round(float(down_ms) / float(full_ms), 4) if full_ms > 0 else None,
                    "up_ratio_of_full": round(float(up_ms) / float(full_ms), 4) if full_ms > 0 else None,
                }
            )
            print(json.dumps(row, ensure_ascii=False))
            append_row(args.results, row)
    finally:
        if tmp_model_dir is not None:
            tmp_model_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
