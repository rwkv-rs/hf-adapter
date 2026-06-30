#!/usr/bin/env python3
# coding=utf-8
"""Component-level timing for RWKV-7 HF fast one-token decode.

`bench_decode_micro.py` shows whole-step fast decode vs HF forward. This script
instruments the fast-token path itself so the next optimization can target the
largest components (per-layer projection groups, recurrent kernel, norm/output,
FFN, final lm_head) instead of guessing from raw profiler tables.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from fla.ops.rwkv7.fused_recurrent import fused_mul_recurrent_rwkv7
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 128
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


class SectionTimer:
    def __init__(self, device: str) -> None:
        self.device = device
        self.cuda = device.startswith("cuda")
        self.totals_ms: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self._pending = []

    @contextmanager
    def section(self, name: str):
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._pending.append((name, start, end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self.totals_ms[name] += (time.perf_counter() - t0) * 1000.0
                self.counts[name] += 1

    def finalize_step(self) -> None:
        if not self.cuda:
            return
        torch.cuda.synchronize()
        for name, start, end in self._pending:
            self.totals_ms[name] += float(start.elapsed_time(end))
            self.counts[name] += 1
        self._pending.clear()


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
    if not hasattr(model, "rwkv7_forward_token"):
        raise ValueError("Loaded model does not expose rwkv7_forward_token")
    return model


def encode(tok, prompt_tokens: int, batch_size: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    ids = ids.repeat(batch_size, 1)
    return ids.to(device) if device.startswith("cuda") else ids


def attn_one_components(attn, hidden_states: torch.Tensor, state: dict[str, Any], v_first: torch.Tensor | None, timer: SectionTimer, prefix: str):
    batch_size, seq_len, _ = hidden_states.shape
    if seq_len != 1:
        raise ValueError("attn_one_components expects [batch, 1, hidden]")
    num_heads, head_dim = attn.num_heads, attn.head_dim

    with timer.section(f"{prefix}.attn_shift_mix"):
        conv_cache = state.get("conv_state")
        if conv_cache is None:
            prev = torch.zeros_like(hidden_states)
        else:
            prev = conv_cache.unsqueeze(1) if conv_cache.dim() == 2 else conv_cache
        delta = prev - hidden_states
        xr = torch.addcmul(hidden_states, delta, attn.x_r)
        xw = torch.addcmul(hidden_states, delta, attn.x_w)
        xk = torch.addcmul(hidden_states, delta, attn.x_k)
        xv = torch.addcmul(hidden_states, delta, attn.x_v)
        xa = torch.addcmul(hidden_states, delta, attn.x_a)
        xg = torch.addcmul(hidden_states, delta, attn.x_g)

    with timer.section(f"{prefix}.attn_linears_lora"):
        r = attn.r_proj(xr)
        w = -0.6065306597126334 * attn.w_lora(xw).sigmoid()
        k = attn.k_proj(xk)
        v = attn.v_proj(xv)
        if attn.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, attn.v_lora(xv).sigmoid())
        a = attn.a_lora(xa).sigmoid()
        g = attn.g_lora(xg)

    with timer.section(f"{prefix}.attn_key_mix_norm"):
        kk = F.normalize(
            (k * attn.k_k).view(batch_size, seq_len, num_heads, head_dim),
            dim=-1,
            p=2.0,
        )
        k = k.addcmul(k * (a - 1), attn.k_a)
        r, w, k, a = (t.view(batch_size, seq_len, num_heads, head_dim) for t in (r, w, k, a))
        v = v.view(batch_size, seq_len, num_heads, attn.head_v_dim)

    with timer.section(f"{prefix}.attn_recurrent"):
        o, recurrent_state = fused_mul_recurrent_rwkv7(
            r=r,
            w=w,
            k=k,
            v=v,
            kk=kk,
            a=a,
            scale=1.0,
            initial_state=state.get("recurrent_state"),
            output_final_state=True,
        )

    with timer.section(f"{prefix}.attn_norm_out_proj"):
        o = attn.g_norm(o.reshape(batch_size * seq_len, attn.value_dim)).view(batch_size, seq_len, attn.value_dim)
        correction = ((r * k * attn.r_k.view(1, 1, num_heads, head_dim)).sum(-1, keepdim=True) * v).reshape(o.shape)
        o = attn.o_proj((o + correction) * g)
    return o, recurrent_state, hidden_states[:, -1], v_first


def ffn_one_components(ffn, hidden_states: torch.Tensor, state: dict[str, Any], timer: SectionTimer, prefix: str):
    with timer.section(f"{prefix}.ffn_shift_mix"):
        ffn_cache = state.get("ffn_state")
        if ffn_cache is None:
            prev = torch.zeros_like(hidden_states)
        else:
            prev = ffn_cache.unsqueeze(1) if ffn_cache.dim() == 2 else ffn_cache
        delta = prev - hidden_states
        k = torch.addcmul(hidden_states, delta, ffn.x_k.view(1, 1, -1))
    with timer.section(f"{prefix}.ffn_key_relu"):
        h = torch.relu(ffn.key(k)) ** 2
    with timer.section(f"{prefix}.ffn_value"):
        out = ffn.value(h)
    return out, hidden_states[:, -1]


def instrumented_forward_token(model, input_ids: torch.Tensor, past_key_values, timer: SectionTimer):
    if input_ids.dim() == 1:
        token = input_ids
    elif input_ids.dim() == 2 and input_ids.shape[1] == 1:
        token = input_ids[:, 0]
    else:
        raise ValueError("input_ids must be [batch] or [batch, 1]")

    with timer.section("embedding"):
        x = model.model.embeddings(token.view(-1, 1))
    v_first = None
    for layer_idx, layer in enumerate(model.model.layers):
        prefix = f"layer_{layer_idx:02d}"
        with timer.section(f"{prefix}.total"):
            state = past_key_values._ensure_layer(layer_idx)
            with timer.section(f"{prefix}.pre_attn_norm"):
                residual = layer.pre_norm(x) if hasattr(layer, "pre_norm") else x
                attn_input = layer.attn_norm(residual)
            attn_out, recurrent_state, conv_state, v_first = attn_one_components(layer.attn, attn_input, state, v_first, timer, prefix)
            with timer.section(f"{prefix}.attn_residual"):
                hidden_states = residual + attn_out
                residual = hidden_states
            with timer.section(f"{prefix}.ffn_norm"):
                ffn_input = layer.ffn_norm(hidden_states)
            ffn_out, ffn_state = ffn_one_components(layer.ffn, ffn_input, state, timer, prefix)
            with timer.section(f"{prefix}.ffn_residual_state"):
                x = residual + ffn_out
                state["recurrent_state"] = recurrent_state
                state["conv_state"] = conv_state
                state["ffn_state"] = ffn_state
                state["attn_state"] = None
    with timer.section("final_norm_lm_head"):
        hidden_states = model.model.norm(x)
        logits = model.lm_head(hidden_states)
    past_key_values._seen_tokens += 1
    return logits, past_key_values


def component_name(section: str) -> str | None:
    if not section.startswith("layer_"):
        return section
    parts = section.split(".", 1)
    if len(parts) != 2:
        return None
    if parts[1] == "total":
        return None
    return parts[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--fixed-token", action="store_true")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, dtype)
    ids = encode(tok, args.prompt_tokens, args.batch_size, args.device)
    fixed = ids[:, -1:]

    with torch.inference_mode():
        out = model(ids, use_cache=True, logits_to_keep=1)
        state = out.past_key_values
        token = fixed if args.fixed_token else out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.warmup):
            out = model.rwkv7_forward_token(token, past_key_values=state)
            state = out.past_key_values
            token = fixed if args.fixed_token else out.logits[:, -1:].argmax(dim=-1)

        out = model(ids, use_cache=True, logits_to_keep=1)
        state = out.past_key_values
        token = fixed if args.fixed_token else out.logits[:, -1:].argmax(dim=-1)
        timer = SectionTimer(args.device)
        wall_ms = 0.0
        for _ in range(args.steps):
            cuda_sync(args.device)
            t0 = time.perf_counter()
            logits, state = instrumented_forward_token(model, token, state, timer)
            with timer.section("argmax"):
                next_token = logits[:, -1:].argmax(dim=-1)
            timer.finalize_step()
            wall_ms += (time.perf_counter() - t0) * 1000.0
            token = fixed if args.fixed_token else next_token

    section_ms = {name: round(total / args.steps, 4) for name, total in sorted(timer.totals_ms.items())}
    component_totals: dict[str, float] = defaultdict(float)
    for name, ms in section_ms.items():
        comp = component_name(name)
        if comp is not None:
            component_totals[comp] += ms
    component_ms = {name: round(ms, 4) for name, ms in sorted(component_totals.items())}
    layer_total_ms = {name: ms for name, ms in section_ms.items() if name.startswith("layer_") and name.endswith(".total")}
    top_components = sorted(component_ms.items(), key=lambda kv: kv[1], reverse=True)[:12]
    top_layers = sorted(layer_total_ms.items(), key=lambda kv: kv[1], reverse=True)[:8]
    wall_ms_per_token = wall_ms / args.steps

    row = {
        "axis": "decode_components",
        "backend": "hf_adapter",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES,
        "cache_type": type(state).__name__ if state is not None else None,
        "decode_api": "rwkv7_forward_token",
        "batch_size": args.batch_size,
        "prompt_tokens": int(ids.shape[1]),
        "steps": args.steps,
        "fixed_token": args.fixed_token,
        "wall_ms_per_token": round(wall_ms_per_token, 4),
        "decode_tokps_wall": round(1000.0 * args.batch_size / wall_ms_per_token, 1) if wall_ms_per_token > 0 else None,
        "component_ms": component_ms,
        "top_components": [[name, ms] for name, ms in top_components],
        "top_layers": [[name, ms] for name, ms in top_layers],
        "section_ms": section_ms,
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
