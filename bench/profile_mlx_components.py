#!/usr/bin/env python3
# coding=utf-8
"""Synchronized component profiler for the Apple MLX RWKV-7 backend.

The normal Apple/Qwen harness measures end-to-end generation.  This tool is a
kernel-work guide: it inserts MLX synchronization boundaries around selected
RWKV MLX components and records where a prefill+decode row spends time.  The
numbers are not a replacement for end-to-end tok/s because synchronization after
components perturbs MLX scheduling, but the ranking is useful for deciding which
pieces need fused MLX/Metal kernels next.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DEFAULT_PROMPT_SEED = (
    "User: Compare RWKV-7 and Qwen3.5 on Apple Silicon. "
    "Report throughput, latency, memory, state-cache behavior, and quantization stability.\n"
    "Assistant: "
)
AXIS = "mlx_component_profile"


def append_jsonl(path: str | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_prompt(seed: str, target_chars: int) -> str:
    target = int(target_chars)
    if target <= 0:
        raise ValueError("target_chars must be positive")
    repeats = (target + len(seed) - 1) // len(seed)
    return (seed * repeats)[:target]


def device_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        try:
            import subprocess

            info["macos_product_version"] = subprocess.check_output(
                ["sw_vers", "-productVersion"], text=True
            ).strip()
            info["mac_hw_model"] = subprocess.check_output(
                ["sysctl", "-n", "hw.model"], text=True
            ).strip()
            info["mac_memsize_bytes"] = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            )
        except Exception:
            pass
    return info


@dataclass
class ComponentStats:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        value = float(elapsed_s)
        self.count += 1
        self.total_s += value
        self.max_s = max(self.max_s, value)

    def row(self, *, total_profiled_s: float) -> dict[str, Any]:
        avg_s = self.total_s / max(self.count, 1)
        return {
            "count": int(self.count),
            "total_s": round(float(self.total_s), 6),
            "avg_ms": round(float(avg_s * 1000.0), 6),
            "max_ms": round(float(self.max_s * 1000.0), 6),
            "pct_profiled": round(float(100.0 * self.total_s / total_profiled_s), 4)
            if total_profiled_s > 0
            else None,
        }


@dataclass
class ComponentRecorder:
    stats: dict[str, ComponentStats] = field(default_factory=dict)

    def add(self, name: str, elapsed_s: float) -> None:
        self.stats.setdefault(str(name), ComponentStats()).add(float(elapsed_s))

    @property
    def total_profiled_s(self) -> float:
        return sum(stat.total_s for stat in self.stats.values())

    def summary(self, *, limit: int = 32) -> dict[str, Any]:
        total = self.total_profiled_s
        rows = {
            name: stat.row(total_profiled_s=total)
            for name, stat in sorted(self.stats.items(), key=lambda item: item[1].total_s, reverse=True)[:limit]
        }
        return {
            "total_profiled_s": round(float(total), 6),
            "components": rows,
            "top_components": [
                {"name": name, **stat.row(total_profiled_s=total)}
                for name, stat in sorted(self.stats.items(), key=lambda item: item[1].total_s, reverse=True)[: min(limit, 12)]
            ],
        }


def component_name(method_name: str, args: tuple[Any, ...]) -> str:
    if method_name == "_layer_norm" and len(args) >= 2:
        prefix = str(args[1])
        if prefix == "model.norm":
            return "layer_norm:final"
        if prefix.endswith(".pre_norm"):
            return "layer_norm:pre_norm"
        if prefix.endswith(".attn_norm"):
            return "layer_norm:attn_norm"
        if prefix.endswith(".ffn_norm"):
            return "layer_norm:ffn_norm"
        return "layer_norm:other"
    if method_name == "_embedding":
        return "embedding"
    if method_name == "_attn_step":
        return "attention_step"
    if method_name == "_ffn_step":
        return "ffn_step"
    if method_name == "_attn_sequence":
        return "attention_sequence_scan"
    if method_name == "_attn_sequence_dplr":
        return "attention_sequence_dplr"
    if method_name == "_ffn_sequence":
        return "ffn_sequence"
    if method_name == "_logits_from_hidden":
        return "final_norm_lm_head"
    return method_name.lstrip("_")


def flatten_mlx_arrays(value: Any) -> list[Any]:
    arrays: list[Any] = []
    if value is None:
        return arrays
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return [value]
    if isinstance(value, dict):
        for item in value.values():
            arrays.extend(flatten_mlx_arrays(item))
        return arrays
    if isinstance(value, (list, tuple)):
        for item in value:
            arrays.extend(flatten_mlx_arrays(item))
        return arrays
    return arrays


def install_component_wrappers(model: Any, recorder: ComponentRecorder, *, sync: bool = True) -> list[tuple[str, Callable[..., Any]]]:
    import mlx.core as mx

    wrapped: list[tuple[str, Callable[..., Any]]] = []
    methods = [
        "_embedding",
        "_layer_norm",
        "_attn_step",
        "_ffn_step",
        "_attn_sequence",
        "_attn_sequence_dplr",
        "_ffn_sequence",
        "_logits_from_hidden",
    ]
    for method_name in methods:
        original = getattr(model, method_name)

        def make_wrapper(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                label = component_name(name, args)
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                if sync:
                    arrays = flatten_mlx_arrays(out)
                    if arrays:
                        mx.eval(*arrays)
                recorder.add(label, time.perf_counter() - t0)
                return out

            return wrapper

        setattr(model, method_name, make_wrapper(method_name, original))
        wrapped.append((method_name, original))
    return wrapped


def restore_component_wrappers(model: Any, wrapped: list[tuple[str, Callable[..., Any]]]) -> None:
    for name, original in wrapped:
        setattr(model, name, original)


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    import mlx.core as mx

    from rwkv7_hf.mlx_bridge import mlx_memory_telemetry, reset_mlx_peak_memory
    from rwkv7_hf.mlx_model import load_mlx_rwkv7_model

    prompt = make_prompt(args.prompt_seed, args.prompt_target_chars)
    t_load = time.perf_counter()
    model = load_mlx_rwkv7_model(
        args.model_dir,
        dtype=args.rwkv_dtype,
        quantization=args.rwkv_quantization,
        quant_min_params=int(args.rwkv_quant_min_params),
        quant_rkv_min_params=None if int(args.rwkv_quant_rkv_min_params) < 0 else int(args.rwkv_quant_rkv_min_params),
        quant_backend=args.rwkv_quant_backend,
        wkv_backend=args.rwkv_wkv_backend,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    load_s = time.perf_counter() - t_load
    prompt_ids = [int(x) for x in tokenizer(prompt, add_special_tokens=False).input_ids]
    if not prompt_ids:
        raise ValueError("prompt produced no token ids")

    # Warm up kernels/caches outside the component wrappers.
    for _ in range(max(0, int(args.warmup_repeats))):
        logits, state = model.prefill([prompt_ids] * int(args.batch_size))
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(next_token)
        for _ in range(int(args.decode_length)):
            logits, state = model.decode_step(next_token, state)
            mx.eval(logits)
            next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
            mx.eval(next_token)

    reset_counters = getattr(model, "reset_telemetry_counters", None)
    if callable(reset_counters):
        reset_counters()
    reset_mlx_peak_memory()
    recorder = ComponentRecorder()
    wrapped = install_component_wrappers(model, recorder, sync=not args.no_component_sync)
    try:
        t_prefill = time.perf_counter()
        logits, state = model.prefill([prompt_ids] * int(args.batch_size))
        mx.eval(logits)
        prefill_s = time.perf_counter() - t_prefill
        next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(next_token)
        generated: list[int] = []
        t_decode = time.perf_counter()
        for _ in range(int(args.decode_length)):
            generated.extend(int(x) for x in next_token.reshape(-1).tolist())
            logits, state = model.decode_step(next_token, state)
            mx.eval(logits)
            next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
            mx.eval(next_token)
        decode_s = time.perf_counter() - t_decode
    finally:
        restore_component_wrappers(model, wrapped)

    telemetry = model.telemetry()
    row = {
        "axis": AXIS,
        "status": "pass",
        "model": Path(args.model_dir).name,
        "model_path": str(args.model_dir),
        "prompt_target_chars": int(args.prompt_target_chars),
        "prompt_chars": len(prompt),
        "prompt_eval_tokens": len(prompt_ids),
        "batch_size": int(args.batch_size),
        "prompt_eval_tokens_total": len(prompt_ids) * int(args.batch_size),
        "decode_length": int(args.decode_length),
        "generated_tokens": int(args.decode_length) * int(args.batch_size),
        "warmup_repeats": int(args.warmup_repeats),
        "component_sync": not bool(args.no_component_sync),
        "load_s": round(float(load_s), 6),
        "prefill_s": round(float(prefill_s), 6),
        "decode_s": round(float(decode_s), 6),
        "prefill_tok_s": round(float(len(prompt_ids) * int(args.batch_size) / prefill_s), 6)
        if prefill_s > 0
        else None,
        "decode_tok_s": round(float(args.decode_length * int(args.batch_size) / decode_s), 6)
        if decode_s > 0
        else None,
        "generated_preview": generated[:16],
        "dtype": args.rwkv_dtype,
        "quantization": args.rwkv_quantization,
        "quant_min_params": int(args.rwkv_quant_min_params),
        "quant_rkv_min_params": None if int(args.rwkv_quant_rkv_min_params) < 0 else int(args.rwkv_quant_rkv_min_params),
        "quant_backend": args.rwkv_quant_backend,
        "wkv_backend": args.rwkv_wkv_backend,
        "wkv_backend_counts": telemetry.get("wkv_backend_counts"),
        "quantized_linear_last_backend_counts": telemetry.get("quantized_linear_last_backend_counts"),
        "group_rkv_quant_projection_counts": telemetry.get("group_rkv_quant_projection_counts"),
        **recorder.summary(limit=int(args.component_limit)),
        **mlx_memory_telemetry(),
        **device_info(),
    }
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--results", default="")
    ap.add_argument("--prompt-target-chars", type=int, default=512)
    ap.add_argument("--prompt-seed", default=DEFAULT_PROMPT_SEED)
    ap.add_argument("--decode-length", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup-repeats", type=int, default=1)
    ap.add_argument("--rwkv-dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    ap.add_argument("--rwkv-quantization", default="mm4", choices=["none", "mm8", "mm4"])
    ap.add_argument("--rwkv-quant-min-params", type=int, default=8_000_000)
    ap.add_argument("--rwkv-quant-rkv-min-params", type=int, default=0)
    ap.add_argument("--rwkv-quant-backend", default="auto", choices=["affine", "reference", "metal", "auto", "groupwise"])
    ap.add_argument("--rwkv-wkv-backend", default="auto", choices=["reference", "metal", "auto"])
    ap.add_argument("--component-limit", type=int, default=32)
    ap.add_argument("--no-component-sync", action="store_true", help="Record wrapper overhead without adding mx.eval boundaries.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.prompt_target_chars <= 0:
        ap.error("--prompt-target-chars must be positive")
    if args.decode_length < 0:
        ap.error("--decode-length must be non-negative")
    if args.batch_size <= 0:
        ap.error("--batch-size must be positive")
    if args.warmup_repeats < 0:
        ap.error("--warmup-repeats must be non-negative")
    if args.dry_run:
        row = {
            "axis": AXIS + "_plan",
            "status": "plan",
            "model_dir": args.model_dir,
            "prompt_target_chars": int(args.prompt_target_chars),
            "decode_length": int(args.decode_length),
            "warmup_repeats": int(args.warmup_repeats),
            "component_sync": not bool(args.no_component_sync),
            "quantization": args.rwkv_quantization,
        }
    else:
        row = run_profile(args)
    print(json.dumps(row, ensure_ascii=False))
    append_jsonl(args.results or None, row)
    return 0 if row.get("status") in {"pass", "plan"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
