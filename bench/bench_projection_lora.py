#!/usr/bin/env python3
# coding=utf-8
"""Projection/LoRA microbenchmarks for RWKV-7 fast-token decode.

The component benchmark identifies `attn_linears_lora` as the largest remaining
fast-token group. This script drills into that group and also times simple
PyTorch-level candidate fusions:

- current separate R/K/V projections vs a stacked batched-matmul candidate
- current W/A LoRA modules vs a same-rank batched-matmul candidate

The candidates are not used by the model; they are measurement scaffolding for
choosing the next implementation/fusion target.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_FALSE_VALUES = {"0", "false", "False", "no", "off"}


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") else device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def load_model(args, dtype):
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")
    set_attn_mode(model, args.attn_mode)
    return model


def make_inputs(hidden_size: int, batch_size: int, device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    gen_device = device if device.startswith("cuda") else "cpu"
    base = torch.randn(batch_size, 1, hidden_size, device=gen_device, dtype=dtype)
    prev = torch.randn_like(base)
    delta = prev - base
    # Deterministic-ish non-identical inputs without using model time-mix params.
    return {
        "xr": base + 0.01 * delta,
        "xw": base + 0.02 * delta,
        "xk": base + 0.03 * delta,
        "xv": base + 0.04 * delta,
        "xa": base + 0.05 * delta,
        "xg": base + 0.06 * delta,
    }


def timed(fn: Callable[[], Any], device: str, warmup: int, steps: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    cuda_sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            fn()
    cuda_sync(device)
    return (time.perf_counter() - t0) * 1000.0 / steps


def activation(module, x):
    # FLA LoRA is Sequential(Linear, activation, Linear). Avoid assuming the
    # activation class; call it from the module.
    return module.lora[1](x)


def lora_current(module, x):
    return module(x)


def lora_manual(module, x):
    h = activation(module, F.linear(x, module.lora[0].weight, None))
    return F.linear(h, module.lora[2].weight, module.lora[2].bias)


def rkv_current(attn, xs):
    return attn.r_proj(xs["xr"]), attn.k_proj(xs["xk"]), attn.v_proj(xs["xv"])


def rkv_bmm_candidate(attn, xs):
    # Shape: [3, batch, hidden] x [3, hidden, hidden] -> [3, batch, hidden]
    x = torch.stack([xs["xr"].squeeze(1), xs["xk"].squeeze(1), xs["xv"].squeeze(1)], dim=0)
    w = torch.stack([attn.r_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0).transpose(1, 2)
    y = torch.bmm(x, w).transpose(0, 1).unsqueeze(2)  # [batch, 3, 1, hidden]
    return y[:, 0], y[:, 1], y[:, 2]


def wa_lora_current(attn, xs):
    return attn.w_lora(xs["xw"]), attn.a_lora(xs["xa"])


def wa_lora_bmm_candidate(attn, xs):
    x = torch.stack([xs["xw"].squeeze(1), xs["xa"].squeeze(1)], dim=0)
    w1 = torch.stack([attn.w_lora.lora[0].weight, attn.a_lora.lora[0].weight], dim=0).transpose(1, 2)
    h = torch.bmm(x, w1)
    h = torch.stack([activation(attn.w_lora, h[0]), activation(attn.a_lora, h[1])], dim=0)
    w2 = torch.stack([attn.w_lora.lora[2].weight, attn.a_lora.lora[2].weight], dim=0).transpose(1, 2)
    y = torch.bmm(h, w2)
    b0 = attn.w_lora.lora[2].bias
    b1 = attn.a_lora.lora[2].bias
    if b0 is not None:
        y[0] = y[0] + b0
    if b1 is not None:
        y[1] = y[1] + b1
    y = y.transpose(0, 1).unsqueeze(2)
    return y[:, 0], y[:, 1]


def maxdiff_tuple(a, b) -> float:
    diffs = []
    for x, y in zip(a, b, strict=False):
        diffs.append(float((x.float() - y.float()).abs().max().detach().cpu()))
    return max(diffs) if diffs else 0.0


def _mb(num_bytes: float) -> float:
    return round(float(num_bytes) / 1024.0 / 1024.0, 5)


def linear_matrix_profile(
    name: str,
    module,
    batch_size: int,
    *,
    input_source: str,
    fused_group: str,
    timing_ms: float | None = None,
    quant_candidate: bool = True,
) -> dict[str, Any]:
    weight = module.weight
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    bias = getattr(module, "bias", None)
    params = int(weight.numel()) + (int(bias.numel()) if bias is not None else 0)
    weight_params = int(weight.numel())
    flops = int(2 * int(batch_size) * in_features * out_features)
    return {
        "name": name,
        "input_source": input_source,
        "fused_group": fused_group,
        "shape": [out_features, in_features],
        "has_bias": bias is not None,
        "params": params,
        "weight_params": weight_params,
        "flops_per_token": flops,
        "fp16_weight_mb": _mb(weight_params * 2),
        "int8_weight_mb": _mb(weight_params),
        "int4_weight_mb": _mb(weight_params / 2.0),
        "timing_ms": round(float(timing_ms), 5) if timing_ms is not None else None,
        "quant_candidate": bool(quant_candidate),
    }


def lora_matrix_profile(
    name: str,
    module,
    batch_size: int,
    *,
    input_source: str,
    fused_group: str,
    timing_ms: float | None = None,
) -> list[dict[str, Any]]:
    down = module.lora[0]
    up = module.lora[2]
    activation_name = type(module.lora[1]).__name__
    rows = [
        linear_matrix_profile(
            f"{name}.down",
            down,
            batch_size,
            input_source=input_source,
            fused_group=fused_group,
            timing_ms=None,
            quant_candidate=False,
        ),
        linear_matrix_profile(
            f"{name}.up",
            up,
            batch_size,
            input_source=f"{name}.activation",
            fused_group=fused_group,
            timing_ms=None,
            quant_candidate=False,
        ),
    ]
    for row in rows:
        row["lora_module"] = name
        row["lora_activation"] = activation_name
        row["lora_timing_group_ms"] = round(float(timing_ms), 5) if timing_ms is not None else None
    return rows


def matrix_profile_summary(matrix_profile: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, dict[str, Any]] = {}
    for row in matrix_profile:
        group = str(row["fused_group"])
        agg = by_group.setdefault(
            group,
            {
                "matrix_count": 0,
                "params": 0,
                "flops_per_token": 0,
                "fp16_weight_mb": 0.0,
                "int8_weight_mb": 0.0,
                "int4_weight_mb": 0.0,
                "timed_members": [],
            },
        )
        agg["matrix_count"] += 1
        agg["params"] += int(row.get("params") or 0)
        agg["flops_per_token"] += int(row.get("flops_per_token") or 0)
        agg["fp16_weight_mb"] += float(row.get("fp16_weight_mb") or 0.0)
        agg["int8_weight_mb"] += float(row.get("int8_weight_mb") or 0.0)
        agg["int4_weight_mb"] += float(row.get("int4_weight_mb") or 0.0)
        if row.get("timing_ms") is not None:
            agg["timed_members"].append([row["name"], row["timing_ms"]])
        elif row.get("lora_timing_group_ms") is not None and not any(
            member[0] == row.get("lora_module") for member in agg["timed_members"]
        ):
            agg["timed_members"].append([row["lora_module"], row["lora_timing_group_ms"]])
    for agg in by_group.values():
        for key in ("fp16_weight_mb", "int8_weight_mb", "int4_weight_mb"):
            agg[key] = round(float(agg[key]), 5)
    return by_group


def build_fused_kernel_plan(
    timings: dict[str, float],
    matrix_profile: list[dict[str, Any]],
    *,
    current_sum_ms: float,
    candidate_sum_ms: float,
) -> dict[str, Any]:
    has_v_lora = any(row.get("lora_module") == "v_lora" for row in matrix_profile)
    first_target_members = ["r_proj", "k_proj", "v_proj", "w_lora", "a_lora", "g_lora"]
    if has_v_lora:
        first_target_members.insert(5, "v_lora")
    rkv_current = float(timings.get("rkv_current") or 0.0)
    rkv_candidate = float(timings.get("rkv_bmm_candidate") or 0.0)
    wa_current = float(timings.get("wa_lora_current") or 0.0)
    wa_candidate = float(timings.get("wa_lora_bmm_candidate") or 0.0)
    return {
        "first_fused_fp16_target": {
            "group": "attn_time_mix_linears_lora",
            "members": first_target_members,
            "current_ms": round(float(current_sum_ms), 5),
            "naive_candidate_ms": round(float(candidate_sum_ms), 5),
            "naive_candidate_speedup": round(float(current_sum_ms) / float(candidate_sum_ms), 4) if candidate_sum_ms else None,
            "reason": "largest decode_components bucket; PyTorch bmm candidate is only a measurement scaffold, not the final kernel",
            "required_output_tensors": ["r", "w", "k", "v", "a", "g"],
        },
        "fused_groups": [
            {
                "name": "attn_rkv_dense",
                "members": ["r_proj", "k_proj", "v_proj"],
                "current_ms": round(rkv_current, 5),
                "naive_candidate_ms": round(rkv_candidate, 5),
                "naive_candidate_speedup": round(rkv_current / rkv_candidate, 4) if rkv_candidate else None,
                "target": "single fp16 fused GEMV/GEMM producing r,k,v without intermediate stack materialization",
            },
            {
                "name": "attn_wa_lora",
                "members": ["w_lora", "a_lora"],
                "current_ms": round(wa_current, 5),
                "naive_candidate_ms": round(wa_candidate, 5),
                "naive_candidate_speedup": round(wa_current / wa_candidate, 4) if wa_candidate else None,
                "target": "fuse two same-rank LoRA down/activation/up chains",
            },
        ],
        "native_quant_candidates": [
            {
                "name": "ffn_key_value",
                "status": "planned_not_measured_by_projection_lora",
                "reason": "largest memory-saving W8/W4 candidate; requires fused dequant-GEMV before bnb can be replaced",
            },
            {
                "name": "attn_dense_rkv_o",
                "status": "fp16_first_quant_later",
                "reason": "decode-hot path; current decode_hot policy keeps it dense because generic bnb kernels are slow on V100",
            },
        ],
        "matrix_profile_summary": matrix_profile_summary(matrix_profile),
    }


def bench_layer(attn, xs, args) -> dict[str, Any]:
    row: dict[str, Any] = {}
    funcs: dict[str, Callable[[], Any]] = {
        "r_proj": lambda: attn.r_proj(xs["xr"]),
        "k_proj": lambda: attn.k_proj(xs["xk"]),
        "v_proj": lambda: attn.v_proj(xs["xv"]),
        "rkv_current": lambda: rkv_current(attn, xs),
        "rkv_bmm_candidate": lambda: rkv_bmm_candidate(attn, xs),
        "w_lora": lambda: attn.w_lora(xs["xw"]),
        "a_lora": lambda: attn.a_lora(xs["xa"]),
        "g_lora": lambda: attn.g_lora(xs["xg"]),
        "wa_lora_current": lambda: wa_lora_current(attn, xs),
        "wa_lora_bmm_candidate": lambda: wa_lora_bmm_candidate(attn, xs),
    }
    if getattr(attn, "v_lora", None) is not None:
        funcs["v_lora"] = lambda: attn.v_lora(xs["xv"])
    else:
        funcs["v_lora"] = lambda: None

    timings = {name: timed(fn, args.device, args.warmup, args.steps) for name, fn in funcs.items()}
    with torch.inference_mode():
        rkv_diff = maxdiff_tuple(rkv_current(attn, xs), rkv_bmm_candidate(attn, xs))
        wa_diff = maxdiff_tuple(wa_lora_current(attn, xs), wa_lora_bmm_candidate(attn, xs))
        w_diff = float((lora_current(attn.w_lora, xs["xw"]).float() - lora_manual(attn.w_lora, xs["xw"]).float()).abs().max().detach().cpu())
    row["timings_ms"] = {k: round(v, 5) for k, v in timings.items()}
    row["candidate_diffs"] = {
        "rkv_bmm_max_abs_diff": rkv_diff,
        "wa_lora_bmm_max_abs_diff": wa_diff,
        "w_lora_manual_max_abs_diff": w_diff,
    }
    row["candidate_speedups"] = {
        "rkv_bmm_vs_current": round(timings["rkv_current"] / timings["rkv_bmm_candidate"], 4) if timings["rkv_bmm_candidate"] else None,
        "wa_bmm_vs_current": round(timings["wa_lora_current"] / timings["wa_lora_bmm_candidate"], 4) if timings["wa_lora_bmm_candidate"] else None,
    }
    row["current_linears_lora_sum_ms"] = round(
        timings["rkv_current"] + timings["w_lora"] + timings["a_lora"] + timings.get("v_lora", 0.0) + timings["g_lora"],
        5,
    )
    row["candidate_linears_lora_sum_ms"] = round(
        timings["rkv_bmm_candidate"] + timings["wa_lora_bmm_candidate"] + timings.get("v_lora", 0.0) + timings["g_lora"],
        5,
    )
    matrix_profile = [
        linear_matrix_profile("r_proj", attn.r_proj, args.batch_size, input_source="xr", fused_group="attn_rkv_dense", timing_ms=timings["r_proj"]),
        linear_matrix_profile("k_proj", attn.k_proj, args.batch_size, input_source="xk", fused_group="attn_rkv_dense", timing_ms=timings["k_proj"]),
        linear_matrix_profile("v_proj", attn.v_proj, args.batch_size, input_source="xv", fused_group="attn_rkv_dense", timing_ms=timings["v_proj"]),
        linear_matrix_profile("o_proj", attn.o_proj, args.batch_size, input_source="attn_out_gated", fused_group="attn_output_dense", quant_candidate=True),
        *lora_matrix_profile("w_lora", attn.w_lora, args.batch_size, input_source="xw", fused_group="attn_wa_lora", timing_ms=timings["w_lora"]),
        *lora_matrix_profile("a_lora", attn.a_lora, args.batch_size, input_source="xa", fused_group="attn_wa_lora", timing_ms=timings["a_lora"]),
        *lora_matrix_profile("g_lora", attn.g_lora, args.batch_size, input_source="xg", fused_group="attn_g_lora", timing_ms=timings["g_lora"]),
    ]
    if getattr(attn, "v_lora", None) is not None:
        matrix_profile.extend(
            lora_matrix_profile("v_lora", attn.v_lora, args.batch_size, input_source="xv", fused_group="attn_v_lora", timing_ms=timings["v_lora"])
        )
    row["matrix_profile"] = matrix_profile
    row["matrix_profile_summary"] = matrix_profile_summary(matrix_profile)
    row["fused_kernel_plan"] = build_fused_kernel_plan(
        timings,
        matrix_profile,
        current_sum_ms=row["current_linears_lora_sum_ms"],
        candidate_sum_ms=row["candidate_linears_lora_sum_ms"],
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, 11])
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_model(args, dtype)
    hidden_size = int(model.config.hidden_size)
    xs = make_inputs(hidden_size, args.batch_size, args.device, dtype)

    layer_rows = []
    for layer_idx in args.layers:
        attn = model.model.layers[layer_idx].attn
        layer_row = bench_layer(attn, xs, args)
        layer_row["layer_idx"] = layer_idx
        layer_row["has_v_lora"] = getattr(attn, "v_lora", None) is not None
        layer_rows.append(layer_row)

    def avg(key: str) -> float:
        vals = [row["timings_ms"][key] for row in layer_rows if key in row["timings_ms"] and row["timings_ms"][key] is not None]
        return round(sum(vals) / len(vals), 5) if vals else 0.0

    avg_timings = {k: avg(k) for k in sorted(layer_rows[0]["timings_ms"].keys())}
    current_sum = round(sum(row["current_linears_lora_sum_ms"] for row in layer_rows) / len(layer_rows), 5)
    candidate_sum = round(sum(row["candidate_linears_lora_sum_ms"] for row in layer_rows) / len(layer_rows), 5)
    sample_matrix_profile = layer_rows[0]["matrix_profile"]
    fused_kernel_plan = build_fused_kernel_plan(
        avg_timings,
        sample_matrix_profile,
        current_sum_ms=current_sum,
        candidate_sum_ms=candidate_sum,
    )
    fused_kernel_plan["sample_layer_idx"] = layer_rows[0]["layer_idx"]
    fused_kernel_plan["sampled_layers"] = args.layers
    row = {
        "axis": "projection_lora",
        "backend": "hf_adapter",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES,
        "batch_size": args.batch_size,
        "hidden_size": hidden_size,
        "layers": args.layers,
        "steps": args.steps,
        "avg_timings_ms": avg_timings,
        "avg_current_linears_lora_sum_ms": current_sum,
        "avg_candidate_linears_lora_sum_ms": candidate_sum,
        "avg_candidate_speedup": round(current_sum / candidate_sum, 4) if candidate_sum else None,
        "sample_matrix_profile": sample_matrix_profile,
        "sample_matrix_profile_summary": matrix_profile_summary(sample_matrix_profile),
        "fused_kernel_plan": fused_kernel_plan,
        "layer_rows": layer_rows,
        "peak_vram_mb": peak_mb(args.device),
    }
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended 1 row -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
