#!/usr/bin/env python3
# coding=utf-8
"""Analyze RWKV-7 benchmark JSONL rows against performance targets.

This turns raw benchmark rows into a compact gap report. It intentionally works
with partially populated `bench/results.jsonl`: missing newer axes are reported
as pending instead of failing, while existing speed/memory rows are compared
against the current target ratios.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        row["_lineno"] = lineno
        rows.append(row)
    return rows


def filt(rows: Iterable[dict[str, Any]], *, device: str | None, dtype: str | None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        row_device = row.get("device")
        if device and row_device is not None and device.lower() not in str(row_device).lower():
            continue
        if dtype and row.get("dtype") != dtype:
            continue
        out.append(row)
    return out


def latest(rows: Iterable[dict[str, Any]], pred) -> dict[str, Any] | None:
    matches = [r for r in rows if pred(r)]
    return matches[-1] if matches else None


def latest_by_key(rows: Iterable[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    out = []
    for key, vals in groups.items():
        out.append(max(vals, key=lambda v: int(v.get("_lineno", 0))))
    return sorted(out, key=lambda r: str(key_fn(r)))


def num(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    val = row.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def verdict_ge(value: float | None, target: float) -> str:
    if value is None:
        return "PENDING"
    return "PASS" if value >= target else "GAP"


def verdict_le(value: float | None, target: float) -> str:
    if value is None:
        return "PENDING"
    return "PASS" if value <= target else "GAP"


def compact(row: dict[str, Any] | None, keys: list[str]) -> dict[str, Any] | None:
    if row is None:
        return None
    return {k: row[k] for k in keys if k in row}


def fast_token_backend_effective(row: dict[str, Any]) -> str | None:
    return row.get("fast_token_backend_effective") or row.get("fast_token_backend")


def analyze(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rows = filt(rows, device=args.device, dtype=args.dtype)
    target_decode_ratio = args.target_decode_ratio
    target_prefill_ratio = args.target_prefill_ratio
    target_memory_ratio = args.target_memory_ratio

    # Keep the formal speed/memory target anchored to the low-memory serving
    # row. `native_graph` is reported through fast_decode as an optional
    # reduced-launch speed path because its captured buffers intentionally trade
    # extra VRAM for lower latency.
    speed_hf = latest(
        rows,
        lambda r: r.get("axis") == "speed_mem"
        and r.get("backend") == "hf_adapter"
        and fast_token_backend_effective(r) != "native_graph",
    ) or latest(rows, lambda r: r.get("axis") == "speed_mem" and r.get("backend") == "hf_adapter")
    speed_official = latest(rows, lambda r: r.get("axis") == "speed_mem" and r.get("backend") == "official_rwkv")
    speed_decode_ratio = ratio(num(speed_hf, "decode_tokps"), num(speed_official, "decode_tokps"))
    speed_prefill_ratio = ratio(num(speed_hf, "prefill_tokps"), num(speed_official, "prefill_tokps"))
    speed_memory_ratio = ratio(num(speed_hf, "peak_vram_mb"), num(speed_official, "peak_vram_mb"))

    breakdown_official = latest(rows, lambda r: r.get("axis") == "decode_breakdown" and r.get("backend") == "official_rwkv")
    breakdown_hf_rows = [r for r in rows if r.get("axis") == "decode_breakdown" and r.get("backend") == "hf_adapter"]
    best_breakdown_hf = max(
        breakdown_hf_rows,
        key=lambda r: (float(r.get("decode_fixed_tokps") or r.get("decode_greedy_tokps") or 0), int(r.get("_lineno", 0))),
        default=None,
    )
    breakdown_decode_ratio = ratio(
        num(best_breakdown_hf, "decode_fixed_tokps") or num(best_breakdown_hf, "decode_greedy_tokps"),
        num(breakdown_official, "decode_tokps"),
    )

    fast_candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("axis") == "speed_mem" and row.get("backend") == "hf_adapter" and row.get("hf_decode_api") in {"rwkv7_forward_one", "rwkv7_forward_token"}:
            fast_candidates.append(row)
        if row.get("axis") == "decode_breakdown" and row.get("backend") == "hf_adapter" and row.get("fast_decode_api") is True:
            fast_candidates.append(row)
    best_fast = max(
        fast_candidates,
        key=lambda r: float(r.get("decode_tokps") or r.get("decode_fast_api_fixed_tokps") or r.get("decode_fast_api_greedy_tokps") or 0),
        default=None,
    )
    best_fast_tokps = None
    if best_fast:
        best_fast_tokps = num(best_fast, "decode_tokps") or num(best_fast, "decode_fast_api_fixed_tokps") or num(best_fast, "decode_fast_api_greedy_tokps")
    fast_decode_ratio = ratio(best_fast_tokps, num(speed_official, "decode_tokps") or num(breakdown_official, "decode_tokps"))

    latest_precision = latest(rows, lambda r: r.get("axis") in {"precision", "official_alignment"})
    greedy = latest_precision.get("greedy_window") if latest_precision else None
    greedy_ratio = None
    if isinstance(greedy, dict) and greedy.get("requested"):
        greedy_ratio = float(greedy.get("matched", 0)) / float(greedy["requested"])

    batch_rows = [r for r in rows if r.get("axis") == "batch_sweep" and r.get("backend") == "hf_adapter"]
    batch_latest = latest_by_key(
        batch_rows,
        lambda r: (r.get("batch_size"), r.get("decode_api")),
    )
    native_graph_batch_sizes = sorted(
        {
            int(r.get("batch_size"))
            for r in batch_latest
            if r.get("decode_api") == "rwkv7_forward_token"
            and fast_token_backend_effective(r) == "native_graph"
            and r.get("batch_size") is not None
        }
    )
    dynamic_rows = [r for r in rows if r.get("axis") == "dynamic_batch" and r.get("backend") == "hf_adapter"]
    dynamic_latest = latest_by_key(
        dynamic_rows,
        lambda r: r.get("decode_api"),
    )
    native_graph_dynamic = any(
        r.get("decode_api") == "rwkv7_forward_token"
        and fast_token_backend_effective(r) == "native_graph"
        for r in dynamic_latest
    )
    chunked_rows = [r for r in rows if r.get("axis") == "chunked_prefill" and r.get("backend") == "hf_adapter"]
    chunked_latest = latest_by_key(
        chunked_rows,
        lambda r: (r.get("prefill_mode"), r.get("chunk_size")),
    )
    micro = latest(rows, lambda r: r.get("axis") == "decode_micro" and r.get("backend") == "hf_adapter")
    forward_fast_path = latest(rows, lambda r: r.get("axis") == "forward_fast_path" and r.get("backend") == "hf_adapter")
    generate_fast_path = latest(rows, lambda r: r.get("axis") == "generate_fast_path" and r.get("backend") == "hf_adapter")
    fast_token_warmup = latest(rows, lambda r: r.get("axis") == "fast_token_warmup" and r.get("backend") == "hf_adapter")
    native_graph_overhead = latest_by_key(
        [r for r in rows if r.get("axis") == "native_graph_replay_overhead" and r.get("backend") == "hf_adapter"],
        lambda r: r.get("batch_size"),
    )
    components = latest(rows, lambda r: r.get("axis") == "decode_components" and r.get("backend") == "hf_adapter")
    projection_lora = latest(rows, lambda r: r.get("axis") == "projection_lora" and r.get("backend") == "hf_adapter")
    quant_rows = [r for r in rows if r.get("axis") == "quantization" and r.get("backend") == "hf_adapter"]
    quant_latest = latest_by_key(quant_rows, lambda r: r.get("quantization"))
    larger_rows = [r for r in rows if r.get("axis") == "larger_model_smoke" and r.get("backend") == "hf_adapter"]
    larger_latest = latest_by_key(
        larger_rows,
        lambda r: r.get("model_size_label") or r.get("model_name"),
    )
    native_rows = [r for r in rows if r.get("axis") == "native_decode" and r.get("backend") == "hf_native_jit"]
    best_native = max(
        native_rows,
        key=lambda r: (float(r.get("native_graph_tokps") or r.get("native_jit_tokps") or 0), int(r.get("_lineno", 0))),
        default=None,
    )
    native_best_tokps = None
    native_best_path = None
    if best_native:
        native_best_tokps = num(best_native, "native_graph_tokps")
        native_best_path = "native_graph"
        if native_best_tokps is None:
            native_best_tokps = num(best_native, "native_jit_tokps")
            native_best_path = "native_jit"
    native_decode_ratio = ratio(native_best_tokps, num(speed_official, "decode_tokps") or num(breakdown_official, "decode_tokps"))

    focus = []
    if speed_decode_ratio is not None and speed_decode_ratio < target_decode_ratio:
        focus.append(f"decode throughput {speed_decode_ratio:.2f}x official; optimize one-token layer/kernel path")
    if speed_memory_ratio is not None and speed_memory_ratio > target_memory_ratio:
        focus.append(f"peak VRAM {speed_memory_ratio:.2f}x official; inspect logits/cache allocation")
    if fast_decode_ratio is None:
        focus.append("formal fast token API rows pending")
    elif fast_decode_ratio < target_decode_ratio:
        focus.append(f"fast token API {fast_decode_ratio:.2f}x official; continue reducing tiny kernels/dispatch")
    if not batch_latest:
        focus.append("batch_sweep rows pending")
    else:
        native_fallback_batches = [
            r.get("batch_size")
            for r in batch_latest
            if r.get("decode_api") == "rwkv7_forward_token"
            and r.get("fast_token_backend") == "native_jit"
            and r.get("fast_token_backend_effective") != "native_jit"
        ]
        if native_fallback_batches:
            sizes = "/".join(str(b) for b in native_fallback_batches)
            focus.append(f"native_jit fast-token backend did not activate for bsz={sizes}; check backend fallback")
    if not dynamic_latest:
        focus.append("dynamic_batch rows pending")
    if not chunked_latest:
        focus.append("chunked_prefill rows pending")
    if micro is None:
        focus.append("decode_micro rows pending")
    if fast_token_warmup is None:
        focus.append("fast_token_warmup rows pending")
    if not native_graph_overhead:
        focus.append("native_graph_replay_overhead rows pending")
    else:
        max_copy_share = max(
            (float(r["copy_share_of_manual_wall"]) for r in native_graph_overhead if r.get("copy_share_of_manual_wall") is not None),
            default=None,
        )
        if max_copy_share is not None and max_copy_share > 0.15:
            focus.append(f"native_graph cache-copy overhead max is {max_copy_share:.2%}; inspect runner cache binding")
    if components is None:
        focus.append("decode_components rows pending")
    elif components.get("top_components"):
        top = components["top_components"][0]
        if isinstance(top, (list, tuple)) and len(top) >= 2:
            focus.append(f"largest fast-token component: {top[0]} {top[1]} ms/token")
    if projection_lora is None:
        focus.append("projection_lora rows pending")
    else:
        speedup = projection_lora.get("avg_candidate_speedup")
        if speedup is not None and float(speedup) < 1.0:
            focus.append(f"naive PyTorch projection/LoRA bmm candidate is slower ({float(speedup):.2f}x); custom fusion needed")
    quant_pass_modes = {r.get("quantization") for r in quant_latest if r.get("status") == "pass"}
    if quant_rows and not {"8bit", "4bit"}.issubset(quant_pass_modes):
        missing = sorted({"8bit", "4bit"} - quant_pass_modes)
        focus.append(f"quantization validation incomplete: missing passing {missing}")
    quant_by_mode = {r.get("quantization"): r for r in quant_latest if r.get("status") == "pass"}
    quant_base_decode = num(quant_by_mode.get("none"), "decode_tokps")
    if quant_base_decode:
        slow = []
        for mode in ("8bit", "4bit"):
            q_decode = num(quant_by_mode.get(mode), "decode_tokps")
            q_ratio = ratio(q_decode, quant_base_decode)
            if q_ratio is not None and q_ratio < 1.0:
                slow.append(f"{mode} {q_ratio:.2f}x")
        if slow:
            focus.append("generic bnb quantized decode is slower than fp16: " + ", ".join(slow))
    larger_by_label = {str(r.get("model_size_label", "")).lower(): r for r in larger_latest}
    for required_label, display_label in (
        ("0.4b", "0.4B"),
        ("1.5b", "1.5B"),
        ("2.9b", "2.9B"),
        ("7.2b", "7.2B"),
        ("13.3b", "13.3B"),
    ):
        if required_label not in larger_by_label:
            focus.append(f"{display_label} converted-model load/generate smoke row pending")
            continue
        row = larger_by_label[required_label]
        if row.get("status") != "pass":
            focus.append(f"{display_label} larger-model smoke did not pass: {row.get('status')}")
        else:
            focus.append(
                f"{display_label} converted HF model loads and generates on "
                f"{row.get('device')} with hidden={row.get('hidden_size')}, layers={row.get('num_hidden_layers')}"
            )
    if best_native is None:
        focus.append("native JIT/CUDA-graph decode rows pending")
    elif native_decode_ratio is not None and native_decode_ratio >= target_decode_ratio:
        if native_graph_batch_sizes and native_graph_dynamic:
            sizes = "/".join(str(v) for v in native_graph_batch_sizes)
            focus.append(
                f"{native_best_path} reaches {native_decode_ratio:.2f}x official; HF native_graph integrated for bsz={sizes} plus dynamic active batches"
            )
        else:
            focus.append(
                f"{native_best_path} reaches {native_decode_ratio:.2f}x official; validate integration with HF fast-token/dynamic batching"
            )
    if not focus:
        focus.append("targets met for available rows; rerun larger models/new GPUs")

    return {
        "filters": {"device": args.device, "dtype": args.dtype},
        "targets": {
            "prefill_ratio_ge": target_prefill_ratio,
            "decode_ratio_ge": target_decode_ratio,
            "memory_ratio_le": target_memory_ratio,
        },
        "speed_mem": {
            "hf": compact(speed_hf, ["_lineno", "device", "attn_mode", "fuse_norm", "fast_cache", "hf_decode_api", "fast_token_layout", "fast_token_backend", "fast_token_backend_effective", "prefill_tokps", "decode_tokps", "decode_ms_per_tok", "peak_vram_mb"]),
            "official": compact(speed_official, ["_lineno", "device", "attn_mode", "prefill_tokps", "decode_tokps", "decode_ms_per_tok", "peak_vram_mb"]),
            "prefill_ratio": round(speed_prefill_ratio, 4) if speed_prefill_ratio is not None else None,
            "decode_ratio": round(speed_decode_ratio, 4) if speed_decode_ratio is not None else None,
            "memory_ratio": round(speed_memory_ratio, 4) if speed_memory_ratio is not None else None,
            "prefill_status": verdict_ge(speed_prefill_ratio, target_prefill_ratio),
            "decode_status": verdict_ge(speed_decode_ratio, target_decode_ratio),
            "memory_status": verdict_le(speed_memory_ratio, target_memory_ratio),
        },
        "decode_breakdown": {
            "best_hf": compact(best_breakdown_hf, ["_lineno", "attn_mode", "fuse_norm", "fast_cache", "cache_type", "prefill_keep1_tokps", "decode_greedy_tokps", "decode_fixed_tokps", "argmax_sampling_overhead_ms_per_tok", "peak_vram_mb"]),
            "official": compact(breakdown_official, ["_lineno", "prefill_tokps", "decode_tokps", "decode_ms_per_tok", "peak_vram_mb"]),
            "decode_ratio": round(breakdown_decode_ratio, 4) if breakdown_decode_ratio is not None else None,
            "decode_status": verdict_ge(breakdown_decode_ratio, target_decode_ratio),
        },
        "fast_decode": {
            "best_row": compact(best_fast, ["_lineno", "axis", "hf_decode_api", "fast_decode_api_name", "fast_token_layout", "fast_token_backend", "fast_token_backend_effective", "attn_mode", "decode_tokps", "decode_fast_api_greedy_tokps", "decode_fast_api_fixed_tokps", "peak_vram_mb"]),
            "decode_tokps": round(best_fast_tokps, 4) if best_fast_tokps is not None else None,
            "decode_ratio": round(fast_decode_ratio, 4) if fast_decode_ratio is not None else None,
            "decode_status": verdict_ge(fast_decode_ratio, target_decode_ratio),
        },
        "precision": {
            "latest": compact(latest_precision, ["_lineno", "axis", "dtype", "top5_match", "argmax_match", "cosine", "max_abs_diff", "mean_abs_diff", "greedy_window"]),
            "greedy_ratio": round(greedy_ratio, 4) if greedy_ratio is not None else None,
        },
        "batch_sweep": [compact(r, ["_lineno", "batch_size", "decode_api", "fast_token_backend", "fast_token_backend_effective", "decode_tokps_total", "decode_tokps_per_seq", "decode_ms_per_step", "peak_vram_mb"]) for r in batch_latest],
        "dynamic_batch": [compact(r, ["_lineno", "decode_api", "fast_token_backend", "fast_token_backend_effective", "initial_batch_size", "final_batch_size", "final_cache_batch_size", "cache_select_api", "total_decode_tokens", "reorder_count", "drop_count", "decode_tokps_total", "decode_ms_per_token", "peak_vram_mb"]) for r in dynamic_latest],
        "chunked_prefill": [compact(r, ["_lineno", "prefill_mode", "batch_size", "prompt_tokens", "chunk_size", "prefill_tokps_total", "speed_ratio_vs_full", "peak_vram_mb", "peak_vram_ratio_vs_full", "max_abs_diff", "decode_max_abs_diff", "seq_length_match"]) for r in chunked_latest],
        "decode_micro": compact(micro, ["_lineno", "fast_decode_api_name", "fast_token_layout", "fast_token_backend", "fast_token_backend_effective", "hf_forward_fixed", "hf_forward_greedy", "hf_forward_auto_fixed", "hf_forward_auto_greedy", "hf_forward_auto_backend", "fast_decode_fixed", "fast_decode_greedy", "norm_lm_head", "lm_head", "argmax", "empty_loop", "peak_vram_mb"]),
        "forward_fast_path": compact(forward_fast_path, ["_lineno", "fast_token_backend", "fast_token_layout", "reference_forward", "hf_forward_fast", "direct_fast_token", "hf_forward_fast_backend", "direct_fast_token_backend", "max_abs_diff_auto_vs_reference", "max_abs_diff_direct_vs_reference", "peak_vram_mb"]),
        "generate_fast_path": compact(generate_fast_path, ["_lineno", "fast_token_backend", "fast_token_backend_effective", "batch_size", "reference_generate", "hf_generate_fast", "speedup_vs_reference", "generated_equal", "generated_tokens_matched", "generated_tokens_total", "prompt_tokens", "max_new_tokens", "peak_vram_mb"]),
        "fast_token_warmup": compact(fast_token_warmup, ["_lineno", "fast_token_backend", "batch_sizes", "effective_backend_by_batch", "native_graph_cache_batch_sizes", "native_graph_cache_size_limit", "cleared_before", "warmup_s", "peak_vram_mb"]),
        "native_graph_replay_overhead": [
            compact(r, ["_lineno", "fast_token_backend", "fast_token_backend_effective", "batch_size", "prompt_tokens", "steps", "fixed_token", "max_abs_diff_runner_vs_api", "copy_from_cache_ms", "token_copy_ms", "graph_replay_ms", "bind_cache_ms", "argmax_ms", "manual_wall_ms_per_token", "api_ms_per_token", "manual_decode_tokps_total", "api_decode_tokps_total", "copy_share_of_manual_wall", "native_graph_cache_requests", "native_graph_cache_hits", "native_graph_cache_misses", "native_graph_cache_evictions", "native_graph_cache_hit_rate", "native_graph_cache_batch_sizes", "peak_vram_mb"])
            for r in native_graph_overhead
        ],
        "decode_components": compact(components, ["_lineno", "decode_api", "batch_size", "wall_ms_per_token", "decode_tokps_wall", "top_components", "top_layers", "peak_vram_mb"]),
        "projection_lora": compact(projection_lora, ["_lineno", "batch_size", "hidden_size", "layers", "avg_timings_ms", "avg_current_linears_lora_sum_ms", "avg_candidate_linears_lora_sum_ms", "avg_candidate_speedup", "peak_vram_mb"]),
        "larger_model_smoke": [
            compact(
                r,
                [
                    "_lineno",
                    "status",
                    "model_size_label",
                    "model_name",
                    "checkpoint_sha256",
                    "checkpoint_size_bytes",
                    "vocab_size",
                    "hidden_size",
                    "intermediate_size",
                    "num_hidden_layers",
                    "head_dim",
                    "num_heads",
                    "value_dim_first",
                    "value_dim_last",
                    "value_dim_unique",
                    "attn_mode",
                    "fuse_norm",
                    "fast_token_backend",
                    "fast_token_backend_effective",
                    "prompt_tokens",
                    "max_new_tokens",
                    "generated_tokens",
                    "top5",
                    "generated_tail",
                    "load_s",
                    "forward_s",
                    "generate_s",
                    "generate_tokps",
                    "model_footprint_mb",
                    "peak_vram_mb",
                    "device",
                ],
            )
            for r in larger_latest
        ],
        "quantization": [
            compact(
                r,
                [
                    "_lineno",
                    "quantization",
                    "status",
                    "prefill_tokps",
                    "decode_mode",
                    "decode_tokps",
                    "reference_decode_tokps",
                    "fast_decode_tokps",
                    "fast_decode_speedup",
                    "fast_forward_backend",
                    "fast_forward_max_abs_diff",
                    "fast_forward_same_next_token",
                    "decode_ms_per_tok",
                    "model_footprint_mb",
                    "peak_vram_mb",
                    "error",
                ],
            )
            for r in quant_latest
        ],
        "native_decode": {
            "best_row": compact(best_native, ["_lineno", "device", "prompt_tokens", "decode_tokens", "hidden_size", "num_heads", "head_dim", "native_jit_tokps", "native_jit_ms_per_tok", "native_graph_tokps", "native_graph_ms_per_tok", "graph_vs_jit_tokens_matched", "graph_vs_jit_tokens_total", "logit_cosine", "logit_max_abs_diff", "peak_vram_mb"]),
            "best_path": native_best_path,
            "decode_tokps": round(native_best_tokps, 4) if native_best_tokps is not None else None,
            "decode_ratio": round(native_decode_ratio, 4) if native_decode_ratio is not None else None,
            "decode_status": verdict_ge(native_decode_ratio, target_decode_ratio),
        },
        "next_focus": focus,
    }


def print_text(report: dict[str, Any]) -> None:
    print("# RWKV-7 benchmark gap report")
    print(f"filters={report['filters']} targets={report['targets']}")
    speed = report["speed_mem"]
    print("\n## speed_mem")
    print(json.dumps(speed, ensure_ascii=False))
    breakdown = report["decode_breakdown"]
    print("\n## decode_breakdown")
    print(json.dumps(breakdown, ensure_ascii=False))
    fast = report["fast_decode"]
    print("\n## fast_decode")
    print(json.dumps(fast, ensure_ascii=False))
    print("\n## precision")
    print(json.dumps(report["precision"], ensure_ascii=False))
    print("\n## batch_sweep")
    if report["batch_sweep"]:
        for row in report["batch_sweep"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## dynamic_batch")
    if report["dynamic_batch"]:
        for row in report["dynamic_batch"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## chunked_prefill")
    if report["chunked_prefill"]:
        for row in report["chunked_prefill"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## decode_micro")
    print(json.dumps(report["decode_micro"], ensure_ascii=False) if report["decode_micro"] else "PENDING")
    print("\n## forward_fast_path")
    print(json.dumps(report["forward_fast_path"], ensure_ascii=False) if report["forward_fast_path"] else "PENDING")
    print("\n## generate_fast_path")
    print(json.dumps(report["generate_fast_path"], ensure_ascii=False) if report["generate_fast_path"] else "PENDING")
    print("\n## fast_token_warmup")
    print(json.dumps(report["fast_token_warmup"], ensure_ascii=False) if report["fast_token_warmup"] else "PENDING")
    print("\n## native_graph_replay_overhead")
    if report["native_graph_replay_overhead"]:
        for row in report["native_graph_replay_overhead"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## decode_components")
    print(json.dumps(report["decode_components"], ensure_ascii=False) if report["decode_components"] else "PENDING")
    print("\n## projection_lora")
    print(json.dumps(report["projection_lora"], ensure_ascii=False) if report["projection_lora"] else "PENDING")
    print("\n## larger_model_smoke")
    if report["larger_model_smoke"]:
        for row in report["larger_model_smoke"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## quantization")
    if report["quantization"]:
        for row in report["quantization"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## native_decode")
    print(json.dumps(report["native_decode"], ensure_ascii=False))
    print("\n## next_focus")
    for item in report["next_focus"]:
        print(f"- {item}")


def has_gap(report: dict[str, Any]) -> bool:
    statuses = [
        report["speed_mem"]["prefill_status"],
        report["speed_mem"]["decode_status"],
        report["speed_mem"]["memory_status"],
        report["decode_breakdown"]["decode_status"],
        report["fast_decode"]["decode_status"],
    ]
    return any(status == "GAP" for status in statuses)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    ap.add_argument("--device", default=None, help="Case-insensitive substring match")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--target-prefill-ratio", type=float, default=0.9)
    ap.add_argument("--target-decode-ratio", type=float, default=0.9)
    ap.add_argument("--target-memory-ratio", type=float, default=1.1)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fail-on-gap", action="store_true", help="Exit nonzero when an available ratio misses target")
    args = ap.parse_args()

    report = analyze(load_rows(Path(args.results)), args)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_text(report)
    if args.fail_on_gap and has_gap(report):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
