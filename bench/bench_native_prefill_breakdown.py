#!/usr/bin/env python3
# coding=utf-8
"""Component breakdown for the native RWKV-7 prefill path.

`bench_native_prefill_scan.py` gives the end-to-end prefill number.  This
script answers the next engineering question: for a slow batch/prompt case, is
the time going into projection/LoRA, the recurrent scan, output prep, FFN, or
the final head?
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_native_prefill_scan import prepare_model_dir
from rwkv7_hf import native_jit


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 2048


def infer_model_size_label(model_path: str) -> str | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", str(model_path).lower())
    return f"{match.group(1)}b" if match else None


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def scan_block_m(model) -> int | None:
    raw = os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    try:
        head_dim = int(model._rwkv7_native_jit_packs()[0][2])
        return native_jit._native_prefill_scan_block_m(head_dim)
    except Exception:
        return None


def scan_num_warps(model, block_m: int | None) -> int | None:
    raw = os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    try:
        head_dim = int(model._rwkv7_native_jit_packs()[0][2])
        return native_jit._native_prefill_scan_num_warps(head_dim, block_m)
    except Exception:
        return None


def scan_num_stages() -> int | None:
    raw = os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_NUM_STAGES")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    try:
        return native_jit._native_prefill_scan_num_stages()
    except Exception:
        return None


def scan_algebraic_output() -> bool:
    return getattr(native_jit, "_native_prefill_scan_algebraic_output_enabled", lambda: False)()


def scan_nomask64() -> bool:
    return getattr(native_jit, "_native_prefill_scan_nomask64_enabled", lambda: False)()


def scan_precompute_w() -> bool:
    return getattr(native_jit, "_native_prefill_scan_precompute_w_enabled", lambda: False)()


def scan_precompute_w_dtype() -> str:
    return getattr(native_jit, "_native_prefill_scan_precompute_w_dtype", lambda: "fp32")()


def median(vals: list[float]) -> float:
    vals = sorted(vals)
    return vals[len(vals) // 2]


def build_ids(tok, batch_size: int, prompt_tokens: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    if int(ids.shape[1]) < prompt_tokens:
        raise ValueError(f"seed produced only {ids.shape[1]} tokens, need {prompt_tokens}")
    return ids.repeat(batch_size, 1).to(device)


class EventProfiler:
    def __init__(self, device: torch.device):
        self.use_cuda = device.type == "cuda"
        self.events: list[tuple[str, int | None, torch.cuda.Event, torch.cuda.Event]] = []
        self.timings: dict[str, float] = defaultdict(float)
        self.layer_timings: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.current_layer: int | None = None
        self.total_start = None
        self.total_end = None
        self.total_wall_start = 0.0
        self.total_wall_ms = 0.0

    def start_total(self) -> None:
        if self.use_cuda:
            self.total_start = torch.cuda.Event(enable_timing=True)
            self.total_end = torch.cuda.Event(enable_timing=True)
            self.total_start.record()
        else:
            self.total_wall_start = time.perf_counter()

    def stop_total(self) -> None:
        if self.use_cuda:
            assert self.total_end is not None
            self.total_end.record()
        else:
            self.total_wall_ms = (time.perf_counter() - self.total_wall_start) * 1000.0

    def measure(self, name: str, fn: Callable[[], Any]) -> Any:
        if self.use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            self.events.append((name, self.current_layer, start, end))
            return out
        start_wall = time.perf_counter()
        out = fn()
        elapsed = (time.perf_counter() - start_wall) * 1000.0
        self.timings[name] += elapsed
        if self.current_layer is not None:
            self.layer_timings[int(self.current_layer)][name] += elapsed
        return out

    def finish(self) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
        if self.use_cuda:
            torch.cuda.synchronize()
            assert self.total_start is not None and self.total_end is not None
            self.timings["total_gpu"] = float(self.total_start.elapsed_time(self.total_end))
            for name, layer_idx, start, end in self.events:
                elapsed = float(start.elapsed_time(end))
                self.timings[name] += elapsed
                if layer_idx is not None:
                    self.layer_timings[int(layer_idx)][name] += elapsed
        else:
            self.timings["total_gpu"] = self.total_wall_ms
        return (
            dict(self.timings),
            {int(layer): dict(values) for layer, values in self.layer_timings.items()},
        )


def profiled_native_prefill(
    model,
    ids: torch.Tensor,
    packs,
    *,
    logits_to_keep: int = 1,
    fine_attention_breakdown: bool = False,
):
    """Mirror ``native_jit.prefill`` while recording component timings."""

    base = model.model
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    B = int(ids.shape[0])
    T = int(ids.shape[1])
    H0 = int(packs[0][1])
    N0 = int(packs[0][2])
    hidden0 = H0 * N0
    dtype = base.embeddings.weight.dtype
    state, xpa, xpf = native_jit._init_batched_from_packs(packs, B, ids.device, dtype)

    profiler = EventProfiler(ids.device)
    profiler.start_total()
    x = profiler.measure("embedding", lambda: F.embedding(ids, base.embeddings.weight).reshape(B, T, hidden0))
    v_first_seq = torch.zeros(B, T, hidden0, device=ids.device, dtype=dtype)

    for p in packs:
        p = native_jit._ensure_rkv_pack(p)
        (i, H, N, eps, has_pre,
         pre_w, pre_b, an_w, an_b, fn_w, fn_b,
         x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
         Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
         gn_w, gn_b, fx_k, fK, fV, RKVw) = p
        layer_idx = int(i)
        profiler.current_layer = layer_idx
        H = int(H)
        N = int(N)
        hidden = H * N
        use_shift_wavg_lora = (
            layer_idx > 0
            and not getattr(native_jit, "_native_prefill_fused_projection_enabled", lambda _rows: False)(B * T)
            and getattr(native_jit, "_native_prefill_fused_shift_wavg_lora_enabled", lambda _rows: False)(B * T)
            and not getattr(native_jit, "_native_prefill_fused_norm_mix_enabled", lambda: False)()
        )
        shift_wavg_lora_done = False
        shift_wavg_values = None

        def norm_shift_mix():
            if getattr(native_jit, "_native_prefill_fused_norm_mix_enabled", lambda: False)():
                out = native_jit.fused_attn_norm_shift_mix_prefill(
                    x,
                    xpa[layer_idx].view(B, hidden),
                    x_r,
                    x_w,
                    x_k,
                    x_v,
                    x_a,
                    x_g,
                    pre_norm_weight=pre_w,
                    pre_norm_bias=pre_b,
                    norm_weight=an_w,
                    norm_bias=an_b,
                    has_pre_norm=int(has_pre) == 1,
                )
                residual_local, h_local, xr_local, xw_local, xk_local, xv_local, xa_local, xg_local = out.as_tuple()
                prev_h_local = None
            else:
                residual_local = F.layer_norm(x, [hidden], pre_w, pre_b, 1e-5) if int(has_pre) == 1 else x
                h_local = F.layer_norm(residual_local, [hidden], an_w, an_b, 1e-5)
                prev_h_local = torch.cat([xpa[layer_idx].view(B, 1, hidden), h_local[:, :-1, :]], dim=1)
                if use_shift_wavg_lora:
                    xr_local = xw_local = xk_local = xv_local = xa_local = xg_local = None
                elif native_jit._native_prefill_fused_shift_mix_enabled():
                    xr_local, xw_local, xk_local, xv_local, xa_local, xg_local = native_jit.fused_attn_shift_mix(
                        h_local, prev_h_local, x_r, x_w, x_k, x_v, x_a, x_g
                    )
                else:
                    xx = prev_h_local - h_local
                    xr_local = h_local + xx * x_r.view(1, 1, hidden)
                    xw_local = h_local + xx * x_w.view(1, 1, hidden)
                    xk_local = h_local + xx * x_k.view(1, 1, hidden)
                    xv_local = h_local + xx * x_v.view(1, 1, hidden)
                    xa_local = h_local + xx * x_a.view(1, 1, hidden)
                    xg_local = h_local + xx * x_g.view(1, 1, hidden)
            return (
                residual_local,
                h_local,
                prev_h_local,
                xr_local,
                xw_local,
                xk_local,
                xv_local,
                xa_local,
                xg_local,
            )

        residual, h, prev_h, xr, xw, xk, xv, xa, xg = profiler.measure("attn_norm_shift_mix", norm_shift_mix)

        if use_shift_wavg_lora:
            block_m, block_r, block_k = native_jit._native_prefill_fused_shift_wavg_lora_blocks()
            down_warps, up_warps = native_jit._native_prefill_fused_shift_wavg_lora_warps()

            def fused_shift_wavg():
                return native_jit.fused_shift_wavg_lora(
                    h.reshape(B * T, hidden),
                    prev_h.reshape(B * T, hidden),
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
                    block_m=block_m,
                    block_r=block_r,
                    block_k=block_k,
                    down_num_warps=down_warps,
                    up_num_warps=up_warps,
                )

            xr2, xk2, xv2, w2_out, a2_out, g2_out, v_gate2 = profiler.measure(
                "attn_shift_wavg_lora_fused",
                fused_shift_wavg,
            )
            xr = xr2.view(B, T, hidden)
            xk = xk2.view(B, T, hidden)
            xv = xv2.view(B, T, hidden)
            xw = xa = xg = None
            shift_wavg_values = (
                w2_out.view(B, T, hidden),
                torch.sigmoid(a2_out.view(B, T, hidden)),
                g2_out.view(B, T, hidden),
                v_gate2.view(B, T, hidden),
            )
            shift_wavg_lora_done = True

        if fine_attention_breakdown:
            r = profiler.measure("attn_dense_r_proj", lambda: F.linear(xr, Rw))
            k = profiler.measure("attn_dense_k_proj", lambda: F.linear(xk, Kw))
            v = profiler.measure("attn_dense_v_proj", lambda: F.linear(xv, Vw))
        else:
            def dense_rkv():
                return F.linear(xr, Rw), F.linear(xk, Kw), F.linear(xv, Vw)

            r, k, v = profiler.measure("attn_dense_rkv", dense_rkv)

        use_fused_scan_output = native_jit._native_prefill_fused_scan_output_enabled()
        use_fused_state_scan_raw_output = (
            getattr(native_jit, "_native_prefill_fused_state_scan_raw_output_enabled", lambda: False)()
            and not use_fused_scan_output
        )
        use_fused_state_scan_sk_output = (
            getattr(native_jit, "_native_prefill_fused_state_scan_sk_output_enabled", lambda: False)()
            and not use_fused_scan_output
            and not use_fused_state_scan_raw_output
        )
        use_clampw_scan = (
            native_jit._native_prefill_fused_clampw_scan_enabled()
            and not use_fused_scan_output
            and not use_fused_state_scan_raw_output
            and not use_fused_state_scan_sk_output
        )
        use_fused_state_scan = (
            native_jit._native_prefill_fused_state_scan_enabled()
            and not use_fused_scan_output
            and not use_fused_state_scan_raw_output
            and not use_fused_state_scan_sk_output
        )
        use_fused_state_scan_any = use_fused_state_scan or use_fused_state_scan_raw_output or use_fused_state_scan_sk_output
        sk_scale = None
        use_cuda_state_scan_sk = False
        use_dplr_scan = (
            native_jit._native_prefill_dplr_scan_enabled()
            and not native_jit._native_prefill_fused_scan_enabled()
            and not use_fused_scan_output
            and not use_fused_state_scan_any
        )
        if use_clampw_scan and native_jit._native_prefill_fused_state_prep_enabled() and native_jit.fused_prefill_kv_kk_prep is None:
            use_clampw_scan = False

        def lora_and_state_prep():
            v_gate_local = None
            if shift_wavg_lora_done:
                w_local, a_local, g_local, v_gate_local = shift_wavg_values
            elif layer_idx > 0 and native_jit._native_prefill_fused_wavg_lora_enabled(B * T):
                block_m, block_r, block_k = native_jit._native_prefill_fused_wavg_lora_blocks()
                w_local, a_local, g_local, v_gate_local = native_jit.fused_wavg_lora(
                    xw.reshape(B * T, hidden),
                    xa.reshape(B * T, hidden),
                    xg.reshape(B * T, hidden),
                    xv.reshape(B * T, hidden),
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
                    block_m=block_m,
                    block_r=block_r,
                    block_k=block_k,
                )
                w_local = w_local.view(B, T, hidden)
                a_local = torch.sigmoid(a_local.view(B, T, hidden))
                g_local = g_local.view(B, T, hidden)
                v_gate_local = v_gate_local.view(B, T, hidden)
            else:
                w_local = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
                a_local = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
                g_local = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
                if layer_idx != 0:
                    v_gate_local = torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
            if use_fused_state_scan_any:
                k_local = k
                v_local = v
                kk_local = None
                v_first_local = v if layer_idx == 0 else v_first_seq
            elif native_jit._native_prefill_fused_state_prep_enabled():
                if use_clampw_scan:
                    if layer_idx == 0:
                        k_local, v_local, kk_local = native_jit.fused_prefill_kv_kk_prep(
                            k,
                            v,
                            a_local,
                            k_k,
                            k_a,
                            num_heads=H,
                            head_dim=N,
                        )
                        v_first_local = v_local
                    else:
                        k_local, v_local, kk_local = native_jit.fused_prefill_kv_kk_prep(
                            k,
                            v,
                            a_local,
                            k_k,
                            k_a,
                            v_first=v_first_seq,
                            v_gate=v_gate_local,
                            num_heads=H,
                            head_dim=N,
                        )
                        v_first_local = v_first_seq
                else:
                    if layer_idx == 0:
                        w_local, k_local, v_local, kk_local = native_jit.fused_prefill_state_prep(
                            w_local,
                            k,
                            v,
                            a_local,
                            k_k,
                            k_a,
                            num_heads=H,
                            head_dim=N,
                            w_out_dtype=native_jit._native_prefill_state_prep_w_dtype(),
                        )
                        v_first_local = v_local
                    else:
                        w_local, k_local, v_local, kk_local = native_jit.fused_prefill_state_prep(
                            w_local,
                            k,
                            v,
                            a_local,
                            k_k,
                            k_a,
                            v_first=v_first_seq,
                            v_gate=v_gate_local,
                            num_heads=H,
                            head_dim=N,
                            w_out_dtype=native_jit._native_prefill_state_prep_w_dtype(),
                        )
                        v_first_local = v_first_seq
            else:
                kk_local = F.normalize((k * k_k.view(1, 1, hidden)).view(B, T, H, N), dim=-1, p=2.0).view(B, T, hidden)
                k_local = k * (1 + (a_local - 1) * k_a.view(1, 1, hidden))
                if layer_idx == 0:
                    v_first_local = v
                    v_local = v
                else:
                    v_first_local = v_first_seq
                    v_local = v + (v_first_seq - v) * v_gate_local
                if not use_clampw_scan:
                    w_local = torch.exp(-0.606531 * torch.sigmoid(w_local.float()))
            return w_local, k_local, v_local, a_local, g_local, kk_local, v_first_local, v_gate_local

        def fine_lora_and_state_prep():
            v_gate_local = None
            if shift_wavg_lora_done:
                w_local, a_local, g_local, v_gate_local = shift_wavg_values
            elif layer_idx > 0 and native_jit._native_prefill_fused_wavg_lora_enabled(B * T):
                block_m, block_r, block_k = native_jit._native_prefill_fused_wavg_lora_blocks()

                def fused_wavg():
                    w_out, a_out, g_out, v_gate_out = native_jit.fused_wavg_lora(
                        xw.reshape(B * T, hidden),
                        xa.reshape(B * T, hidden),
                        xg.reshape(B * T, hidden),
                        xv.reshape(B * T, hidden),
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
                        block_m=block_m,
                        block_r=block_r,
                        block_k=block_k,
                    )
                    return (
                        w_out.view(B, T, hidden),
                        torch.sigmoid(a_out.view(B, T, hidden)),
                        g_out.view(B, T, hidden),
                        v_gate_out.view(B, T, hidden),
                    )

                w_local, a_local, g_local, v_gate_local = profiler.measure("attn_lora_wavg_fused", fused_wavg)
            else:
                w_local = profiler.measure("attn_lora_w", lambda: F.linear(torch.tanh(F.linear(xw, w1)), w2, w0))
                a_local = profiler.measure("attn_lora_a", lambda: torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2)))
                g_local = profiler.measure("attn_lora_g", lambda: F.linear(torch.sigmoid(F.linear(xg, g1)), g2))
                if layer_idx != 0:
                    v_gate_local = profiler.measure("attn_lora_v_gate", lambda: torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2)))

            if use_fused_state_scan_any:
                k_local = k
                v_local = v
                kk_local = None
                v_first_local = v if layer_idx == 0 else v_first_seq
            elif native_jit._native_prefill_fused_state_prep_enabled():
                if use_clampw_scan:
                    if layer_idx == 0:
                        k_local, v_local, kk_local = profiler.measure(
                            "attn_state_prep_no_w_fused",
                            lambda: native_jit.fused_prefill_kv_kk_prep(
                                k,
                                v,
                                a_local,
                                k_k,
                                k_a,
                                num_heads=H,
                                head_dim=N,
                            ),
                        )
                        v_first_local = v_local
                    else:
                        k_local, v_local, kk_local = profiler.measure(
                            "attn_state_prep_no_w_fused",
                            lambda: native_jit.fused_prefill_kv_kk_prep(
                                k,
                                v,
                                a_local,
                                k_k,
                                k_a,
                                v_first=v_first_seq,
                                v_gate=v_gate_local,
                                num_heads=H,
                                head_dim=N,
                            ),
                        )
                        v_first_local = v_first_seq
                else:
                    if layer_idx == 0:
                        w_local, k_local, v_local, kk_local = profiler.measure(
                            "attn_state_prep_fused",
                            lambda: native_jit.fused_prefill_state_prep(
                                w_local,
                                k,
                                v,
                                a_local,
                                k_k,
                                k_a,
                                num_heads=H,
                                head_dim=N,
                                w_out_dtype=native_jit._native_prefill_state_prep_w_dtype(),
                            ),
                        )
                        v_first_local = v_local
                    else:
                        w_local, k_local, v_local, kk_local = profiler.measure(
                            "attn_state_prep_fused",
                            lambda: native_jit.fused_prefill_state_prep(
                                w_local,
                                k,
                                v,
                                a_local,
                                k_k,
                                k_a,
                                v_first=v_first_seq,
                                v_gate=v_gate_local,
                                num_heads=H,
                                head_dim=N,
                                w_out_dtype=native_jit._native_prefill_state_prep_w_dtype(),
                            ),
                        )
                        v_first_local = v_first_seq
            else:
                kk_local = profiler.measure(
                    "attn_kk_norm",
                    lambda: F.normalize((k * k_k.view(1, 1, hidden)).view(B, T, H, N), dim=-1, p=2.0).view(B, T, hidden),
                )
                k_local = profiler.measure("attn_k_adjust", lambda: k * (1 + (a_local - 1) * k_a.view(1, 1, hidden)))
                if layer_idx == 0:
                    v_first_local = v
                    v_local = v
                else:
                    v_first_local = v_first_seq
                    v_local = profiler.measure("attn_v_interp", lambda: v + (v_first_seq - v) * v_gate_local)
                if not use_clampw_scan:
                    w_local = profiler.measure("attn_w_decay", lambda: torch.exp(-0.606531 * torch.sigmoid(w_local.float())))
            return w_local, k_local, v_local, a_local, g_local, kk_local, v_first_local, v_gate_local

        if fine_attention_breakdown:
            w, k, v, a, g, kk, v_first_seq, v_gate = fine_lora_and_state_prep()
        else:
            w, k, v, a, g, kk, v_first_seq, v_gate = profiler.measure("attn_lora_state_prep", lora_and_state_prep)

        if use_fused_state_scan_raw_output:
            state_scan_block_m = native_jit._native_prefill_scan_block_m(N)
            state_scan_num_warps = native_jit._native_prefill_scan_num_warps(N, state_scan_block_m)
            state_scan_num_stages = native_jit._native_prefill_scan_num_stages()

            def state_scan_raw_output():
                if layer_idx == 0:
                    return native_jit.fused_recurrent_scan_state_prep_nokv(
                        r.view(B, T, H, N),
                        w.view(B, T, H, N),
                        k.view(B, T, H, N),
                        v.view(B, T, H, N),
                        a.view(B, T, H, N),
                        state[layer_idx],
                        k_k,
                        k_a,
                        block_n=N,
                        block_m=state_scan_block_m,
                        num_warps=state_scan_num_warps,
                        num_stages=state_scan_num_stages,
                    )
                return native_jit.fused_recurrent_scan_state_prep_nokv(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    v_first=v_first_seq.view(B, T, H, N),
                    v_gate=v_gate.view(B, T, H, N),
                    block_n=N,
                    block_m=state_scan_block_m,
                    num_warps=state_scan_num_warps,
                    num_stages=state_scan_num_stages,
                )

            out, new_state = profiler.measure("recurrent_scan_state_prep_nokv_fused", state_scan_raw_output)
            out = out.reshape(B, T, hidden)
            if layer_idx == 0:
                v_first_seq = v
        elif use_fused_state_scan_sk_output:
            state_scan_block_m = native_jit._native_prefill_scan_block_m(N)
            state_scan_num_warps = native_jit._native_prefill_scan_num_warps(N, state_scan_block_m)
            state_scan_num_stages = native_jit._native_prefill_scan_num_stages()
            state_scan_nomask64 = getattr(native_jit, "_native_prefill_scan_nomask64_enabled", lambda: False)()

            def state_scan_sk_output():
                if layer_idx == 0:
                    return native_jit.fused_recurrent_scan_state_prep_sk(
                        r.view(B, T, H, N),
                        w.view(B, T, H, N),
                        k.view(B, T, H, N),
                        v.view(B, T, H, N),
                        a.view(B, T, H, N),
                        state[layer_idx],
                        k_k,
                        k_a,
                        r_k,
                        block_n=N,
                        num_warps=state_scan_num_warps,
                        num_stages=state_scan_num_stages,
                        nomask64=state_scan_nomask64,
                    )
                return native_jit.fused_recurrent_scan_state_prep_sk(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    r_k,
                    v_first=v_first_seq.view(B, T, H, N),
                    v_gate=v_gate.view(B, T, H, N),
                    block_n=N,
                    num_warps=state_scan_num_warps,
                    num_stages=state_scan_num_stages,
                    nomask64=state_scan_nomask64,
                )

            out, new_state, sk_scale = profiler.measure("recurrent_scan_state_prep_sk_fused", state_scan_sk_output)
            out = out.reshape(B, T, hidden)
            sk_scale = sk_scale.reshape(B * T, H)
            if layer_idx == 0:
                v_first_seq = v
        elif use_fused_state_scan:
            state_scan_block_m = native_jit._native_prefill_scan_block_m(N)
            state_scan_num_warps = native_jit._native_prefill_scan_num_warps(N, state_scan_block_m)
            state_scan_num_stages = native_jit._native_prefill_scan_num_stages()
            state_scan_algebraic_output = native_jit._native_prefill_scan_algebraic_output_enabled()
            state_scan_nomask64 = native_jit._native_prefill_scan_nomask64_enabled()
            use_cuda_state_scan = (
                getattr(native_jit, "_native_prefill_cuda_state_scan_enabled", lambda: False)()
                and N == 64
                and state_scan_block_m == 64
                and h.dtype == torch.float16
            )
            use_cuda_state_scan_sk = bool(
                use_cuda_state_scan and getattr(native_jit, "_native_prefill_cuda_state_scan_sk_enabled", lambda: False)()
            )
            cuda_state_scan_lanes = (
                native_jit._native_prefill_cuda_state_scan_lanes_per_row() if use_cuda_state_scan else 1
            )
            cuda_state_scan_precompute = (
                native_jit._native_prefill_cuda_state_scan_precompute_enabled() if use_cuda_state_scan else False
            )
            cuda_state_scan_precompute_mode = (
                native_jit._native_prefill_cuda_state_scan_precompute_mode() if use_cuda_state_scan else "none"
            )
            cuda_state_scan_rows_per_block = (
                native_jit._native_prefill_cuda_state_scan_rows_per_block() if use_cuda_state_scan else 1
            )
            cuda_state_scan_schedule = (
                native_jit._native_prefill_cuda_state_scan_schedule() if use_cuda_state_scan else "default"
            )
            state_scan_precompute_w = (
                getattr(native_jit, "_native_prefill_scan_precompute_w_enabled", lambda: False)()
                and not use_cuda_state_scan
            )
            state_scan_precompute_w_dtype = getattr(
                native_jit, "_native_prefill_scan_precompute_w_dtype", lambda: "fp32"
            )()
            w_for_state_scan = w
            if state_scan_precompute_w:
                def precompute_w_decay():
                    w_decay = torch.sigmoid(w.float()).mul_(-0.606531).exp_()
                    if state_scan_precompute_w_dtype == "input":
                        w_decay = w_decay.to(dtype=w.dtype)
                    return w_decay

                w_for_state_scan = profiler.measure("attn_w_decay_precompute", precompute_w_decay)

            def state_scan():
                if use_cuda_state_scan_sk and layer_idx == 0:
                    return native_jit.cuda_state_scan_prep_sk(
                        r.view(B, T, H, N),
                        w.view(B, T, H, N),
                        k.view(B, T, H, N),
                        v.view(B, T, H, N),
                        a.view(B, T, H, N),
                        state[layer_idx],
                        k_k,
                        k_a,
                        r_k,
                        rows_per_block=cuda_state_scan_rows_per_block,
                        schedule=cuda_state_scan_schedule,
                    )
                if use_cuda_state_scan_sk:
                    return native_jit.cuda_state_scan_prep_sk(
                        r.view(B, T, H, N),
                        w.view(B, T, H, N),
                        k.view(B, T, H, N),
                        v.view(B, T, H, N),
                        a.view(B, T, H, N),
                        state[layer_idx],
                        k_k,
                        k_a,
                        r_k,
                        v_first=v_first_seq.view(B, T, H, N),
                        v_gate=v_gate.view(B, T, H, N),
                        rows_per_block=cuda_state_scan_rows_per_block,
                        schedule=cuda_state_scan_schedule,
                    )
                if use_cuda_state_scan and layer_idx == 0:
                    return native_jit.cuda_state_scan_prep(
                        r.view(B, T, H, N),
                        w.view(B, T, H, N),
                        k.view(B, T, H, N),
                        v.view(B, T, H, N),
                        a.view(B, T, H, N),
                        state[layer_idx],
                        k_k,
                        k_a,
                        lanes_per_row=cuda_state_scan_lanes,
                        precompute_vector=cuda_state_scan_precompute,
                        precompute_mode=cuda_state_scan_precompute_mode,
                        rows_per_block=cuda_state_scan_rows_per_block,
                        schedule=cuda_state_scan_schedule,
                    )
                if use_cuda_state_scan:
                    return native_jit.cuda_state_scan_prep(
                        r.view(B, T, H, N),
                        w.view(B, T, H, N),
                        k.view(B, T, H, N),
                        v.view(B, T, H, N),
                        a.view(B, T, H, N),
                        state[layer_idx],
                        k_k,
                        k_a,
                        v_first=v_first_seq.view(B, T, H, N),
                        v_gate=v_gate.view(B, T, H, N),
                        lanes_per_row=cuda_state_scan_lanes,
                        precompute_vector=cuda_state_scan_precompute,
                        precompute_mode=cuda_state_scan_precompute_mode,
                        rows_per_block=cuda_state_scan_rows_per_block,
                        schedule=cuda_state_scan_schedule,
                    )
                if layer_idx == 0:
                    return native_jit.fused_recurrent_scan_state_prep(
                        r.view(B, T, H, N),
                        w_for_state_scan.view(B, T, H, N),
                        k.view(B, T, H, N),
                        v.view(B, T, H, N),
                        a.view(B, T, H, N),
                        state[layer_idx],
                        k_k,
                        k_a,
                        block_n=N,
                        block_m=state_scan_block_m,
                        num_warps=state_scan_num_warps,
                        num_stages=state_scan_num_stages,
                        algebraic_output=state_scan_algebraic_output,
                        nomask64=state_scan_nomask64,
                        precomputed_w=state_scan_precompute_w,
                    )
                return native_jit.fused_recurrent_scan_state_prep(
                    r.view(B, T, H, N),
                    w_for_state_scan.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    v_first=v_first_seq.view(B, T, H, N),
                    v_gate=v_gate.view(B, T, H, N),
                    block_n=N,
                    block_m=state_scan_block_m,
                    num_warps=state_scan_num_warps,
                    num_stages=state_scan_num_stages,
                    algebraic_output=state_scan_algebraic_output,
                    nomask64=state_scan_nomask64,
                    precomputed_w=state_scan_precompute_w,
                )

            component_name = (
                "recurrent_scan_state_prep_cuda_sk"
                if use_cuda_state_scan_sk
                else ("recurrent_scan_state_prep_cuda" if use_cuda_state_scan else "recurrent_scan_state_prep_fused")
            )
            scan_result = profiler.measure(component_name, state_scan)
            if use_cuda_state_scan_sk:
                out, new_state, sk_scale = scan_result
            else:
                out, new_state, k, v = scan_result
            out = out.reshape(B, T, hidden)
            if use_cuda_state_scan_sk:
                sk_scale = sk_scale.reshape(B * T, H)
                if layer_idx == 0:
                    v_first_seq = v.reshape(B, T, hidden)
            else:
                k = k.reshape(B, T, hidden)
                v = v.reshape(B, T, hidden)
                if layer_idx == 0:
                    v_first_seq = v
        elif use_fused_scan_output:
            out, new_state = profiler.measure(
                "recurrent_scan_output_prep_fused",
                lambda: native_jit.fused_recurrent_scan_output_prepare(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    kk.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    g.view(B, T, H, N),
                    r_k,
                    gn_w,
                    gn_b,
                    eps=eps,
                    block_n=N,
                ),
            )
            out = out.reshape(B, T, hidden)
        else:
            out, new_state = profiler.measure(
                "recurrent_scan_clampw" if use_clampw_scan else ("recurrent_scan_dplr" if use_dplr_scan else "recurrent_scan"),
                lambda: native_jit._native_prefill_scan(r, w, k, v, kk, a, state[layer_idx], B, T, H, N, w_is_raw=use_clampw_scan),
            )

        def output_prep_project():
            if use_fused_state_scan_raw_output:
                if layer_idx == 0:
                    out_local = native_jit.fused_attn_output_prepare_raw_kv(
                        out.reshape(B * T, hidden),
                        r.reshape(B * T, H, N),
                        k.reshape(B * T, H, N),
                        v.reshape(B * T, H, N),
                        a.reshape(B * T, H, N),
                        g.reshape(B * T, hidden),
                        k_a.view(H, N),
                        r_k,
                        gn_w,
                        gn_b,
                        num_heads=H,
                        head_dim=N,
                        head_v_dim=N,
                        eps=eps,
                    ).view(B, T, hidden)
                else:
                    out_local = native_jit.fused_attn_output_prepare_raw_kv(
                        out.reshape(B * T, hidden),
                        r.reshape(B * T, H, N),
                        k.reshape(B * T, H, N),
                        v.reshape(B * T, H, N),
                        a.reshape(B * T, H, N),
                        g.reshape(B * T, hidden),
                        k_a.view(H, N),
                        r_k,
                        gn_w,
                        gn_b,
                        v_first=v_first_seq.reshape(B * T, H, N),
                        v_gate=v_gate.reshape(B * T, H, N),
                        num_heads=H,
                        head_dim=N,
                        head_v_dim=N,
                        eps=eps,
                    ).view(B, T, hidden)
            elif use_fused_state_scan_sk_output or use_cuda_state_scan_sk:
                if layer_idx == 0:
                    out_local = native_jit.fused_attn_output_prepare_from_sk_raw_v(
                        out.reshape(B * T, hidden),
                        sk_scale,
                        v.reshape(B * T, H, N),
                        g.reshape(B * T, hidden),
                        gn_w,
                        gn_b,
                        num_heads=H,
                        head_v_dim=N,
                        eps=eps,
                    ).view(B, T, hidden)
                else:
                    out_local = native_jit.fused_attn_output_prepare_from_sk_raw_v(
                        out.reshape(B * T, hidden),
                        sk_scale,
                        v.reshape(B * T, H, N),
                        g.reshape(B * T, hidden),
                        gn_w,
                        gn_b,
                        v_first=v_first_seq.reshape(B * T, H, N),
                        v_gate=v_gate.reshape(B * T, H, N),
                        num_heads=H,
                        head_v_dim=N,
                        eps=eps,
                    ).view(B, T, hidden)
            elif native_jit._native_prefill_fused_output_enabled():
                out_local = native_jit.fused_attn_output_prepare(
                    out.reshape(B * T, hidden),
                    r.reshape(B * T, H, N),
                    k.reshape(B * T, H, N),
                    v.reshape(B * T, H, N),
                    g.reshape(B * T, hidden),
                    r_k,
                    gn_w,
                    gn_b,
                    num_heads=H,
                    head_dim=N,
                    head_v_dim=N,
                    eps=eps,
                ).view(B, T, hidden)
            else:
                out_local = F.group_norm(out.reshape(B * T, hidden), H, gn_w, gn_b, eps).view(B, T, hidden)
                sk = (r.view(B, T, H, N) * k.view(B, T, H, N) * r_k.view(1, 1, H, N)).sum(dim=-1, keepdim=True)
                out_local = (out_local + (sk * v.view(B, T, H, N)).view(B, T, hidden)) * g
            out_local = F.linear(out_local, Ow)
            return residual + out_local

        def fine_output_prep_project():
            if use_fused_state_scan_raw_output:
                def raw_output_prep():
                    if layer_idx == 0:
                        return native_jit.fused_attn_output_prepare_raw_kv(
                            out.reshape(B * T, hidden),
                            r.reshape(B * T, H, N),
                            k.reshape(B * T, H, N),
                            v.reshape(B * T, H, N),
                            a.reshape(B * T, H, N),
                            g.reshape(B * T, hidden),
                            k_a.view(H, N),
                            r_k,
                            gn_w,
                            gn_b,
                            num_heads=H,
                            head_dim=N,
                            head_v_dim=N,
                            eps=eps,
                        ).view(B, T, hidden)
                    return native_jit.fused_attn_output_prepare_raw_kv(
                        out.reshape(B * T, hidden),
                        r.reshape(B * T, H, N),
                        k.reshape(B * T, H, N),
                        v.reshape(B * T, H, N),
                        a.reshape(B * T, H, N),
                        g.reshape(B * T, hidden),
                        k_a.view(H, N),
                        r_k,
                        gn_w,
                        gn_b,
                        v_first=v_first_seq.reshape(B * T, H, N),
                        v_gate=v_gate.reshape(B * T, H, N),
                        num_heads=H,
                        head_dim=N,
                        head_v_dim=N,
                        eps=eps,
                    ).view(B, T, hidden)

                prepared = profiler.measure("attn_output_prep_raw_kv_fused", raw_output_prep)
            elif use_fused_state_scan_sk_output or use_cuda_state_scan_sk:
                def sk_output_prep():
                    if layer_idx == 0:
                        return native_jit.fused_attn_output_prepare_from_sk_raw_v(
                            out.reshape(B * T, hidden),
                            sk_scale,
                            v.reshape(B * T, H, N),
                            g.reshape(B * T, hidden),
                            gn_w,
                            gn_b,
                            num_heads=H,
                            head_v_dim=N,
                            eps=eps,
                        ).view(B, T, hidden)
                    return native_jit.fused_attn_output_prepare_from_sk_raw_v(
                        out.reshape(B * T, hidden),
                        sk_scale,
                        v.reshape(B * T, H, N),
                        g.reshape(B * T, hidden),
                        gn_w,
                        gn_b,
                        v_first=v_first_seq.reshape(B * T, H, N),
                        v_gate=v_gate.reshape(B * T, H, N),
                        num_heads=H,
                        head_v_dim=N,
                        eps=eps,
                    ).view(B, T, hidden)

                prepared = profiler.measure("attn_output_prep_sk_raw_v_fused", sk_output_prep)
            elif native_jit._native_prefill_fused_output_enabled():
                prepared = profiler.measure(
                    "attn_output_prep_fused",
                    lambda: native_jit.fused_attn_output_prepare(
                        out.reshape(B * T, hidden),
                        r.reshape(B * T, H, N),
                        k.reshape(B * T, H, N),
                        v.reshape(B * T, H, N),
                        g.reshape(B * T, hidden),
                        r_k,
                        gn_w,
                        gn_b,
                        num_heads=H,
                        head_dim=N,
                        head_v_dim=N,
                        eps=eps,
                    ).view(B, T, hidden),
                )
            else:
                def output_prep():
                    out_local = F.group_norm(out.reshape(B * T, hidden), H, gn_w, gn_b, eps).view(B, T, hidden)
                    sk = (r.view(B, T, H, N) * k.view(B, T, H, N) * r_k.view(1, 1, H, N)).sum(dim=-1, keepdim=True)
                    return (out_local + (sk * v.view(B, T, H, N)).view(B, T, hidden)) * g

                prepared = profiler.measure("attn_output_prep", output_prep)
            projected = profiler.measure("attn_output_o_proj", lambda: F.linear(prepared, Ow))
            return residual + projected

        if use_fused_scan_output:
            projected = profiler.measure("attn_output_o_proj", lambda: F.linear(out, Ow))
            x = residual + projected
        elif fine_attention_breakdown:
            x = fine_output_prep_project()
        else:
            x = profiler.measure("attn_output_project", output_prep_project)
        xpa[layer_idx] = h[:, -1, :].contiguous()
        state[layer_idx] = new_state.contiguous()

        def ffn_block():
            residual_local = x
            apply_norm_shift = getattr(native_jit, "_native_prefill_apply_ffn_norm_shift", None)
            if apply_norm_shift is None:
                h2 = F.layer_norm(x, [hidden], fn_w, fn_b, 1e-5)
                prev_h2 = torch.cat([xpf[layer_idx].view(B, 1, hidden), h2[:, :-1, :]], dim=1)
                fk = h2 + (prev_h2 - h2) * fx_k.view(1, 1, hidden)
                h2_last = h2[:, -1, :].contiguous()
            else:
                fk, h2_last = apply_norm_shift(x, xpf[layer_idx], fx_k, fn_w, fn_b, hidden)
            fk = F.linear(fk, fK)
            fk = getattr(native_jit, "_native_prefill_apply_ffn_activation", lambda _fk: torch.relu(_fk) ** 2)(fk)
            return residual_local + F.linear(fk, fV), h2_last.contiguous()

        x, xpf_last = profiler.measure("ffn", ffn_block)
        xpf[layer_idx] = xpf_last

    def final_norm_head():
        normed = F.layer_norm(x, [hidden0], base.norm.weight, base.norm.bias, 1e-5)
        keep = T if logits_to_keep is None or int(logits_to_keep) <= 0 else min(int(logits_to_keep), T)
        return F.linear(normed[:, -keep:, :], model.lm_head.weight, model.lm_head.bias)

    profiler.current_layer = None
    logits = profiler.measure("final_norm_head", final_norm_head)
    profiler.stop_total()
    timings, layer_timings = profiler.finish()
    return logits, state, xpa, xpf, timings, layer_timings


def run_case(args: argparse.Namespace, tok, model, batch_size: int, prompt_tokens: int) -> dict[str, Any]:
    ids = build_ids(tok, batch_size, prompt_tokens, args.device)
    packs = model._rwkv7_native_jit_packs()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        ref = model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)
        prof_logits, *_ = profiled_native_prefill(model, ids, packs, logits_to_keep=1, fine_attention_breakdown=args.fine_attn)
        ref_logits = ref.logits[:, -1, :].detach()
        prof_logits_last = prof_logits[:, -1, :].detach()
        max_abs = float((ref_logits.float() - prof_logits_last.float()).abs().max().detach().cpu())
        greedy_match = bool(torch.equal(ref_logits.argmax(dim=-1).detach().cpu(), prof_logits_last.argmax(dim=-1).detach().cpu()))

    for _ in range(args.warmup):
        with torch.inference_mode():
            profiled_native_prefill(model, ids, packs, logits_to_keep=1, fine_attention_breakdown=args.fine_attn)

    timing_runs: list[dict[str, float]] = []
    layer_timing_runs: list[dict[int, dict[str, float]]] = []
    with torch.inference_mode():
        for _ in range(args.steps):
            _, _, _, _, timings, layer_timings = profiled_native_prefill(
                model,
                ids,
                packs,
                logits_to_keep=1,
                fine_attention_breakdown=args.fine_attn,
            )
            timing_runs.append(timings)
            layer_timing_runs.append(layer_timings)

    keys = sorted({k for row in timing_runs for k in row})
    med = {k: median([float(row.get(k, 0.0)) for row in timing_runs]) for k in keys}
    component_keys = [k for k in keys if k != "total_gpu"]
    component_sum = sum(float(med.get(k, 0.0)) for k in component_keys)
    total_gpu = float(med.get("total_gpu") or component_sum)
    component_ms = {k: round(float(med.get(k, 0.0)), 4) for k in component_keys}
    component_share = {
        k: round(float(med.get(k, 0.0)) / component_sum, 4) if component_sum > 0 else None
        for k in component_keys
    }
    top_components = sorted(
        [[k, component_ms[k], component_share[k]] for k in component_keys],
        key=lambda row: float(row[1]),
        reverse=True,
    )
    layer_component_ms = None
    layer_total_ms = None
    top_layers_by_total = None
    layer_top_components = None
    if args.layer_breakdown:
        layer_ids = sorted({int(layer) for run in layer_timing_runs for layer in run})
        layer_component_ms = {}
        layer_total_ms = {}
        layer_top_components = {}
        for layer_idx in layer_ids:
            layer_keys = sorted({k for run in layer_timing_runs for k in run.get(layer_idx, {})})
            layer_med = {
                k: median([float(run.get(layer_idx, {}).get(k, 0.0)) for run in layer_timing_runs])
                for k in layer_keys
            }
            layer_components = {k: round(float(v), 4) for k, v in layer_med.items()}
            layer_component_ms[str(layer_idx)] = layer_components
            total = sum(float(v) for v in layer_med.values())
            layer_total_ms[str(layer_idx)] = round(total, 4)
            layer_top_components[str(layer_idx)] = sorted(
                [[k, round(float(v), 4)] for k, v in layer_med.items()],
                key=lambda row: float(row[1]),
                reverse=True,
            )[:5]
        top_layers_by_total = sorted(
            [[int(layer), float(ms)] for layer, ms in layer_total_ms.items()],
            key=lambda row: float(row[1]),
            reverse=True,
        )[:8]
    peak = None
    if args.device.startswith("cuda"):
        peak = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    scan_m = scan_block_m(model)
    shift_wavg_blocks = getattr(native_jit, "_native_prefill_fused_shift_wavg_lora_blocks", lambda: (None, None, None))()
    shift_wavg_warps = getattr(native_jit, "_native_prefill_fused_shift_wavg_lora_warps", lambda: (None, None))()
    try:
        rkv_pack = native_jit._ensure_rkv_pack(model._rwkv7_native_jit_packs()[0])
        rkv_hidden = int(rkv_pack[1]) * int(rkv_pack[2])
        rkv_weight = rkv_pack[-1]
        prefill_rkv_bmm_effective = getattr(native_jit, "_native_prefill_rkv_bmm_enabled", lambda *_args: False)(
            batch_size * prompt_tokens,
            rkv_hidden,
            rkv_weight,
        )
    except Exception:
        prefill_rkv_bmm_effective = False
    row = {
        "axis": "native_prefill_breakdown",
        "backend": "hf_adapter",
        "bench_case": os.environ.get("RWKV7_BENCH_CASE"),
        "status": "pass" if greedy_match else "fail",
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "model_path": args.model,
        "effective_model_path": getattr(args, "effective_model_path", args.model),
        "code_source": getattr(args, "code_source", "model"),
        "model_size_label": infer_model_size_label(args.model),
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "tokens_total": batch_size * prompt_tokens,
        "fused_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SCAN", "0").lower() not in {"0", "false", "no", "off"},
        "scan_block_m": scan_m,
        "scan_num_warps": scan_num_warps(model, scan_m),
        "scan_num_stages": scan_num_stages(),
        "scan_algebraic_output": scan_algebraic_output(),
        "scan_nomask64": scan_nomask64(),
        "scan_precompute_w": scan_precompute_w(),
        "scan_precompute_w_dtype": scan_precompute_w_dtype(),
        "prefill_rkv_bmm_requested": getattr(native_jit, "_native_prefill_rkv_bmm_requested", lambda: False)(),
        "prefill_rkv_bmm_effective": bool(prefill_rkv_bmm_effective),
        "prefill_rkv_bmm_max_rows": getattr(native_jit, "_native_prefill_rkv_bmm_max_rows", lambda: None)(),
        "prefill_cuda_state_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_cuda_state_scan_effective": getattr(native_jit, "_native_prefill_cuda_state_scan_enabled", lambda: False)(),
        "prefill_cuda_state_scan_sk_requested": os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SK", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_cuda_state_scan_sk_effective": getattr(native_jit, "_native_prefill_cuda_state_scan_sk_enabled", lambda: False)(),
        "prefill_cuda_state_scan_lanes": getattr(native_jit, "_native_prefill_cuda_state_scan_lanes_per_row", lambda: 1)(),
        "prefill_cuda_state_scan_precompute": getattr(native_jit, "_native_prefill_cuda_state_scan_precompute_enabled", lambda: False)(),
        "prefill_cuda_state_scan_precompute_mode": getattr(native_jit, "_native_prefill_cuda_state_scan_precompute_mode", lambda: "none")(),
        "prefill_cuda_state_scan_rows_per_block": getattr(native_jit, "_native_prefill_cuda_state_scan_rows_per_block", lambda: 1)(),
        "prefill_cuda_state_scan_schedule": getattr(native_jit, "_native_prefill_cuda_state_scan_schedule", lambda: "default")(),
        "prefill_cuda_state_scan_w_precomputed": getattr(native_jit, "_native_prefill_fused_shift_wavg_lora_w_decay_enabled", lambda _rows: False)(batch_size * prompt_tokens),
        "fine_attention_breakdown": bool(args.fine_attn),
        "prefill_fused_scan_output_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_scan_output_effective": native_jit._native_prefill_fused_scan_output_enabled(),
        "prefill_fused_state_scan_output_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_OUTPUT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_scan_output_effective": native_jit._native_prefill_fused_state_scan_output_enabled(),
        "prefill_fused_state_scan_correction_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_CORRECTION", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_scan_correction_effective": getattr(native_jit, "_native_prefill_fused_state_scan_correction_enabled", lambda: False)(),
        "prefill_fused_state_scan_raw_output_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_RAW_OUTPUT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_scan_raw_output_effective": getattr(native_jit, "_native_prefill_fused_state_scan_raw_output_enabled", lambda: False)(),
        "prefill_fused_state_scan_sk_output_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_SK_OUTPUT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_scan_sk_output_effective": getattr(native_jit, "_native_prefill_fused_state_scan_sk_output_enabled", lambda: False)(),
        "prefill_fused_clampw_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_clampw_scan_effective": native_jit._native_prefill_fused_clampw_scan_enabled(),
        "prefill_dplr_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_DPLR_SCAN", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_dplr_scan_effective": (
            native_jit._native_prefill_dplr_scan_enabled()
            and not native_jit._native_prefill_fused_scan_enabled()
            and not native_jit._native_prefill_fused_scan_output_enabled()
        ),
        "prefill_dplr_algorithm": os.environ.get("RWKV7_DPLR_PREFILL_ALGORITHM"),
        "prefill_dplr_chunk_size": native_jit._native_prefill_dplr_chunk_size(),
        "prefill_fused_shift_mix_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_shift_mix_effective": native_jit._native_prefill_fused_shift_mix_enabled(),
        "prefill_fused_norm_mix_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_NORM_MIX", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_norm_mix_effective": getattr(native_jit, "_native_prefill_fused_norm_mix_enabled", lambda: False)(),
        "prefill_fused_state_prep_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_prep_effective": native_jit._native_prefill_fused_state_prep_enabled(),
        "prefill_fused_state_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_scan_effective": native_jit._native_prefill_fused_state_scan_enabled(),
        "prefill_state_prep_w_dtype": native_jit._native_prefill_state_prep_w_dtype(),
        "prefill_fused_output_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_OUTPUT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_output_effective": native_jit._native_prefill_fused_output_enabled(),
        "prefill_fused_wavg_lora_requested": native_jit._native_prefill_fused_wavg_lora_requested(),
        "prefill_fused_wavg_lora_effective": native_jit._native_prefill_fused_wavg_lora_enabled(batch_size * prompt_tokens),
        "prefill_fused_wavg_lora_max_m": native_jit._native_prefill_fused_wavg_lora_max_m(),
        "prefill_fused_shift_wavg_lora_requested": getattr(native_jit, "_native_prefill_fused_shift_wavg_lora_requested", lambda: False)(),
        "prefill_fused_shift_wavg_lora_effective": getattr(native_jit, "_native_prefill_fused_shift_wavg_lora_enabled", lambda _rows: False)(batch_size * prompt_tokens),
        "prefill_fused_shift_wavg_lora_block_m": shift_wavg_blocks[0],
        "prefill_fused_shift_wavg_lora_block_r": shift_wavg_blocks[1],
        "prefill_fused_shift_wavg_lora_block_k": shift_wavg_blocks[2],
        "prefill_fused_shift_wavg_lora_down_warps": shift_wavg_warps[0],
        "prefill_fused_shift_wavg_lora_up_warps": shift_wavg_warps[1],
        "prefill_fused_shift_wavg_lora_w_decay_requested": getattr(native_jit, "_native_prefill_fused_shift_wavg_lora_w_decay_requested", lambda: False)(),
        "prefill_fused_shift_wavg_lora_w_decay_effective": getattr(native_jit, "_native_prefill_fused_shift_wavg_lora_w_decay_enabled", lambda _rows: False)(batch_size * prompt_tokens),
        "prefill_ffn_fused_act_requested": getattr(native_jit, "_native_prefill_ffn_fused_act_requested", lambda: False)(),
        "prefill_ffn_fused_act_effective": getattr(native_jit, "_native_prefill_ffn_fused_act_enabled", lambda: False)(),
        "prefill_ffn_fused_act_mode": getattr(native_jit, "_native_prefill_ffn_fused_act_mode", lambda: "triton")(),
        "prefill_ffn_fused_act_block_size": getattr(native_jit, "_native_prefill_ffn_fused_act_block_size", lambda: None)(),
        "prefill_ffn_fused_norm_shift_requested": getattr(native_jit, "_native_prefill_ffn_fused_norm_shift_requested", lambda: False)(),
        "prefill_ffn_fused_norm_shift_effective": getattr(native_jit, "_native_prefill_ffn_fused_norm_shift_enabled", lambda: False)(),
        "prefill_ffn_fused_norm_shift_block_h": getattr(native_jit, "_native_prefill_ffn_fused_norm_shift_block_h", lambda: None)(),
        "prefill_fused_projection_requested": getattr(native_jit, "_native_prefill_fused_projection_requested", lambda: False)(),
        "prefill_fused_projection_effective": getattr(native_jit, "_native_prefill_fused_projection_enabled", lambda _rows: False)(batch_size * prompt_tokens),
        "prefill_fused_projection_max_m": getattr(native_jit, "_native_prefill_fused_projection_max_m", lambda: None)(),
        "profiled_total_gpu_ms": round(total_gpu, 4),
        "component_sum_ms": round(component_sum, 4),
        "profiled_tokps_total": round(1000.0 * batch_size * prompt_tokens / total_gpu, 1) if total_gpu > 0 else None,
        "component_ms": component_ms,
        "component_share": component_share,
        "top_components": top_components,
        "max_abs_diff_vs_native_prefill": round(max_abs, 6),
        "greedy_match_vs_native_prefill": greedy_match,
        "peak_vram_mb": peak,
    }
    if args.layer_breakdown:
        row.update(
            {
                "layer_breakdown": True,
                "layer_component_ms": layer_component_ms,
                "layer_total_ms": layer_total_ms,
                "top_layers_by_total": top_layers_by_total,
                "layer_top_components": layer_top_components,
            }
        )
    return row


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
    ap.add_argument("--batch-sizes", default="1,4")
    ap.add_argument("--prompt-tokens", default="512")
    ap.add_argument("--fused-scan", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--code-source", choices=["model", "repo"], default="model", help="load trust_remote_code from checkpoint files or overlay current repo rwkv7_hf/*.py")
    ap.add_argument("--fine-attn", action="store_true", help="split attn_lora_state_prep into LoRA/state-prep subcomponents")
    ap.add_argument("--layer-breakdown", action="store_true", help="also record per-layer component timings for bsz=1 bottleneck attribution")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    if args.fused_scan != "auto":
        os.environ["RWKV7_NATIVE_PREFILL_FUSED_SCAN"] = "1" if args.fused_scan == "true" else "0"

    effective_model_path, tmp_model_dir = prepare_model_dir(args.model, code_source=args.code_source)
    args.effective_model_path = effective_model_path
    try:
        tok = AutoTokenizer.from_pretrained(effective_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            effective_model_path,
            trust_remote_code=True,
            torch_dtype=DTYPES[args.dtype],
            device_map=args.device if args.device.startswith("cuda") else None,
        ).eval()
        for bsz in parse_ints(args.batch_sizes):
            for prompt_tokens in parse_ints(args.prompt_tokens):
                row = run_case(args, tok, model, bsz, prompt_tokens)
                print(json.dumps(row, ensure_ascii=False))
                append_row(args.results, row)
    finally:
        if tmp_model_dir is not None:
            tmp_model_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
