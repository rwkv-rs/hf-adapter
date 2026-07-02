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
import re
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


def model_size_label(row: dict[str, Any] | None) -> str | None:
    """Best-effort normalized model size label such as ``0.4b``.

    Benchmark rows grew this field over time.  Older HF rows often only contain
    a checkpoint path (`rwkv7-g1d-0.4b-hf`), while Albatross rows usually record
    ``model_size_label`` directly.  Use this for apples-to-apples comparisons
    when possible, and fall back to unlabeled rows only when no exact-size row
    exists.
    """

    if not row:
        return None
    for key in ("model_size_label", "model_label", "model_path", "model"):
        raw = row.get(key)
        if raw is None:
            continue
        text = str(raw).lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*b", text)
        if match:
            return f"{match.group(1)}b"
    return None


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


def is_default_native_graph_batch_sweep(row: dict[str, Any]) -> bool:
    """Return whether a batch_sweep row represents the production native_graph path.

    Experimental flag sweeps are useful telemetry, but they must not overwrite
    the Albatross gate just because they were appended later in results.jsonl.
    Treat missing historical flag fields as non-experimental for compatibility,
    but reject any row that explicitly enables an opt-in probe or disables a
    default native-graph fusion.
    """

    if row.get("axis") != "batch_sweep" or row.get("backend") != "hf_adapter":
        return False
    if row.get("decode_api") != "rwkv7_forward_token":
        return False
    if fast_token_backend_effective(row) != "native_graph":
        return False
    for flag in (
        "native_graph_fused_recurrent",
        "native_graph_fused_output_project",
        "native_graph_fused_wag_lora",
        "native_graph_fused_wavg_lora",
        "native_graph_fused_projection",
    ):
        if bool(row.get(flag)):
            return False
    if row.get("native_graph_fused_recurrent_output") is False:
        return False
    if row.get("native_graph_fused_output") is False:
        return False
    return True


def native_graph_flag_signature(row: dict[str, Any]) -> str:
    flags = []
    for flag in (
        "native_graph_fused_recurrent",
        "native_graph_fused_recurrent_output",
        "native_graph_fused_output",
        "native_graph_fused_output_project",
        "native_graph_fused_wag_lora",
        "native_graph_fused_wavg_lora",
        "native_graph_fused_projection",
    ):
        if flag in row:
            flags.append(f"{flag.replace('native_graph_', '')}={bool(row.get(flag))}")
    return ",".join(flags) or "native_graph_flags_unreported"


def model_label(row: dict[str, Any]) -> str:
    label = row.get("model_size_label")
    if label:
        return str(label).lower()
    name = str(row.get("model_name") or row.get("hf_model_dir") or "")
    lowered = name.lower()
    for candidate in ("0.1b", "0.4b", "1.5b", "2.9b", "7.2b", "13.3b"):
        if candidate in lowered:
            return candidate
    return "unknown"


def is_canonical_quant_model(row: dict[str, Any]) -> bool:
    """Keep the main quantization gate anchored to the 0.1B baseline.

    Larger-model quant rows are reported separately via quantization_model_sweep
    so adding a 0.4B/1.5B sweep cannot accidentally overwrite the existing
    canonical W8/W4 memory gate.
    """

    label = model_label(row)
    return label in {"unknown", "0.1b"}


def analyze(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    raw_rows = list(rows)
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
    batch_default_latest = latest_by_key(
        [r for r in batch_rows if is_default_native_graph_batch_sweep(r)],
        lambda r: r.get("batch_size"),
    )
    batch_experimental_latest = latest_by_key(
        [
            r
            for r in batch_rows
            if r.get("decode_api") == "rwkv7_forward_token"
            and fast_token_backend_effective(r) == "native_graph"
            and not is_default_native_graph_batch_sweep(r)
        ],
        lambda r: (native_graph_flag_signature(r), r.get("batch_size")),
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
    native_prefill_scan = latest_by_key(
        [r for r in rows if r.get("axis") == "native_prefill_scan" and r.get("backend") == "hf_adapter"],
        lambda r: (
            r.get("batch_size"),
            r.get("prompt_tokens"),
            r.get("code_source"),
            r.get("native_jit_module"),
            bool(r.get("fused_scan_requested")),
            r.get("scan_block_m"),
            r.get("scan_num_warps"),
            bool(r.get("fine_attention_breakdown")),
            bool(r.get("layer_breakdown")),
            bool(r.get("prefill_fused_scan_output_effective")),
            bool(r.get("prefill_fused_clampw_scan_effective")),
            bool(r.get("prefill_dplr_scan_effective")),
            r.get("prefill_dplr_chunk_size"),
            bool(r.get("prefill_fused_shift_mix_effective")),
            bool(r.get("prefill_fused_state_prep_effective")),
            bool(r.get("prefill_fused_state_scan_effective")),
            r.get("prefill_state_prep_w_dtype"),
            bool(r.get("prefill_fused_output_effective")),
            bool(r.get("prefill_fused_wavg_lora_effective")),
            r.get("prefill_fused_wavg_lora_max_m"),
        ),
    )
    native_prefill_breakdown = latest_by_key(
        [r for r in rows if r.get("axis") == "native_prefill_breakdown" and r.get("backend") == "hf_adapter"],
        lambda r: (
            model_size_label(r),
            r.get("batch_size"),
            r.get("prompt_tokens"),
            bool(r.get("fused_scan_requested")),
            r.get("scan_block_m"),
            r.get("scan_num_warps"),
            bool(r.get("fine_attention_breakdown")),
            bool(r.get("prefill_fused_scan_output_effective")),
            bool(r.get("prefill_fused_clampw_scan_effective")),
            bool(r.get("prefill_dplr_scan_effective")),
            r.get("prefill_dplr_chunk_size"),
            bool(r.get("prefill_fused_shift_mix_effective")),
            bool(r.get("prefill_fused_state_prep_effective")),
            bool(r.get("prefill_fused_state_scan_effective")),
            r.get("prefill_state_prep_w_dtype"),
            bool(r.get("prefill_fused_output_effective")),
            bool(r.get("prefill_fused_wavg_lora_effective")),
            r.get("prefill_fused_wavg_lora_max_m"),
        ),
    )
    micro = latest(rows, lambda r: r.get("axis") == "decode_micro" and r.get("backend") == "hf_adapter")
    forward_fast_path = latest(rows, lambda r: r.get("axis") == "forward_fast_path" and r.get("backend") == "hf_adapter")
    generate_fast_path = latest(rows, lambda r: r.get("axis") == "generate_fast_path" and r.get("backend") == "hf_adapter")
    fast_token_warmup = latest(rows, lambda r: r.get("axis") == "fast_token_warmup" and r.get("backend") == "hf_adapter")
    ttft_tpot = latest_by_key(
        [r for r in rows if r.get("axis") == "ttft_tpot" and r.get("backend") == "hf_adapter"],
        lambda r: (
            r.get("metric"),
            r.get("batch_size"),
            r.get("prompt_tokens"),
            r.get("fast_prefill_env"),
            r.get("native_prefill_fused_scan_env"),
        ),
    )
    native_graph_overhead = latest_by_key(
        [r for r in rows if r.get("axis") == "native_graph_replay_overhead" and r.get("backend") == "hf_adapter"],
        lambda r: r.get("batch_size"),
    )
    components = latest(rows, lambda r: r.get("axis") == "decode_components" and r.get("backend") == "hf_adapter")
    projection_lora = latest(rows, lambda r: r.get("axis") == "projection_lora" and r.get("backend") == "hf_adapter")
    fused_projection_proto = latest(rows, lambda r: r.get("axis") == "fused_projection_proto" and r.get("backend") == "hf_adapter")
    albatross_projection_layout = latest_by_key(
        [r for r in rows if r.get("axis") == "albatross_projection_layout_tune" and r.get("backend") == "hf_adapter"],
        lambda r: r.get("batch_size"),
    )
    fused_wa_lora_proto = latest(rows, lambda r: r.get("axis") == "fused_wa_lora_proto" and r.get("backend") == "hf_adapter")
    fused_wag_lora_proto = latest(rows, lambda r: r.get("axis") == "fused_wag_lora_proto" and r.get("backend") == "hf_adapter")
    fused_wavg_lora_proto = latest(rows, lambda r: r.get("axis") == "fused_wavg_lora_proto" and r.get("backend") == "hf_adapter")
    fused_rkv_wag_projection_proto = latest(rows, lambda r: r.get("axis") == "fused_rkv_wag_projection_proto" and r.get("backend") == "hf_adapter")
    fused_attn_output_proto = latest(rows, lambda r: r.get("axis") == "fused_attn_output_proto" and r.get("backend") == "hf_adapter")
    fused_attn_output_project_proto = latest(rows, lambda r: r.get("axis") == "fused_attn_output_project_proto" and r.get("backend") == "hf_adapter")
    fused_ffn_proto = latest(rows, lambda r: r.get("axis") == "fused_ffn_proto" and r.get("backend") == "hf_adapter")
    fused_shift_mix_proto = latest(rows, lambda r: r.get("axis") == "fused_shift_mix_proto" and r.get("backend") == "hf_adapter")
    fused_recurrent_proto = latest(rows, lambda r: r.get("axis") == "fused_recurrent_proto" and r.get("backend") == "hf_adapter")
    fused_recurrent_scan_proto = latest_by_key(
        [r for r in rows if r.get("axis") == "fused_recurrent_scan_proto" and r.get("backend") == "hf_adapter"],
        lambda r: (r.get("batch_size"), r.get("tokens"), r.get("heads"), r.get("head_dim"), r.get("block_m"), r.get("num_warps")),
    )
    dplr_prefill_scan_proto = latest_by_key(
        [r for r in rows if r.get("axis") == "dplr_prefill_scan_proto" and r.get("backend") == "hf_adapter"],
        lambda r: (
            r.get("B", r.get("batch_size")),
            r.get("T", r.get("tokens")),
            r.get("H", r.get("heads")),
            r.get("N", r.get("head_dim")),
            r.get("requested_algorithm", r.get("algorithm")),
            r.get("effective_algorithm"),
            r.get("algorithm_family"),
            r.get("chunk_size"),
            r.get("dtype"),
            r.get("device"),
        ),
    )
    dplr_chunk_summary_proto = latest_by_key(
        [r for r in rows if r.get("axis") == "dplr_chunk_summary_proto" and r.get("backend") == "hf_adapter"],
        lambda r: (
            r.get("B", r.get("batch_size")),
            r.get("T", r.get("tokens")),
            r.get("H", r.get("heads")),
            r.get("N", r.get("head_dim")),
            r.get("algorithm"),
            r.get("chunk_size"),
            r.get("dtype"),
            r.get("device"),
        ),
    )
    dplr_dense3_stage_proto = latest_by_key(
        [r for r in rows if r.get("axis") == "dplr_dense3_stage_proto" and r.get("backend") == "hf_adapter"],
        lambda r: (
            r.get("B", r.get("batch_size")),
            r.get("T", r.get("tokens")),
            r.get("H", r.get("heads")),
            r.get("N", r.get("head_dim")),
            r.get("algorithm"),
            r.get("stage"),
            r.get("chunk_size"),
            r.get("dtype"),
            r.get("device"),
        ),
    )
    fused_recurrent_output_proto = latest(rows, lambda r: r.get("axis") == "fused_recurrent_output_proto" and r.get("backend") == "hf_adapter")
    native_graph_fused_recurrent = latest(rows, lambda r: r.get("axis") == "native_graph_fused_recurrent" and r.get("backend") == "hf_adapter")
    native_graph_fused_recurrent_output = latest(rows, lambda r: r.get("axis") == "native_graph_fused_recurrent_output" and r.get("backend") == "hf_adapter")
    native_graph_fused_output = latest(rows, lambda r: r.get("axis") == "native_graph_fused_output" and r.get("backend") == "hf_adapter")
    native_graph_fused_output_project = latest(rows, lambda r: r.get("axis") == "native_graph_fused_output_project" and r.get("backend") == "hf_adapter")
    native_graph_fused_wag_lora = latest(rows, lambda r: r.get("axis") == "native_graph_fused_wag_lora" and r.get("backend") == "hf_adapter")
    native_graph_fused_wavg_lora = latest(rows, lambda r: r.get("axis") == "native_graph_fused_wavg_lora" and r.get("backend") == "hf_adapter")
    native_graph_fused_projection = latest(rows, lambda r: r.get("axis") == "native_graph_fused_projection" and r.get("backend") == "hf_adapter")
    native_graph_fused_output_sweep = latest_by_key(
        [
            r
            for r in rows
            if r.get("axis") == "native_graph_fused_output"
            and r.get("backend") == "hf_adapter"
            and not bool(r.get("fused_recurrent_enabled"))
        ],
        lambda r: r.get("batch_size"),
    )
    native_graph_fused_recurrent_output_sweep = latest_by_key(
        [
            r
            for r in rows
            if r.get("axis") == "native_graph_fused_recurrent_output"
            and r.get("backend") == "hf_adapter"
        ],
        lambda r: r.get("batch_size"),
    )
    native_graph_fused_output_project_sweep = latest_by_key(
        [
            r
            for r in rows
            if r.get("axis") == "native_graph_fused_output_project"
            and r.get("backend") == "hf_adapter"
            and not bool(r.get("fused_recurrent_enabled"))
            and not bool(r.get("fused_projection_enabled"))
        ],
        lambda r: r.get("batch_size"),
    )
    native_graph_fused_wag_lora_sweep = latest_by_key(
        [
            r
            for r in rows
            if r.get("axis") == "native_graph_fused_wag_lora"
            and r.get("backend") == "hf_adapter"
            and not bool(r.get("fused_recurrent_enabled"))
            and not bool(r.get("fused_output_project_enabled"))
        ],
        lambda r: r.get("batch_size"),
    )
    native_graph_fused_wavg_lora_sweep = latest_by_key(
        [
            r
            for r in rows
            if r.get("axis") == "native_graph_fused_wavg_lora"
            and r.get("backend") == "hf_adapter"
            and not bool(r.get("fused_recurrent_enabled"))
            and bool(r.get("fused_recurrent_output_enabled", True))
            and bool(r.get("fused_output_enabled", True))
            and not bool(r.get("fused_output_project_enabled"))
        ],
        lambda r: r.get("batch_size"),
    )
    native_graph_fused_projection_sweep = latest_by_key(
        [
            r
            for r in rows
            if r.get("axis") == "native_graph_fused_projection"
            and r.get("backend") == "hf_adapter"
            and not bool(r.get("fused_recurrent_enabled"))
            and bool(r.get("fused_output_enabled", True))
        ],
        lambda r: r.get("batch_size"),
    )
    native_quant_gemv_proto = latest(rows, lambda r: r.get("axis") == "native_quant_gemv_proto" and r.get("backend") == "hf_adapter")
    native_quant_w4_gemv_proto = latest(rows, lambda r: r.get("axis") == "native_quant_w4_gemv_proto" and r.get("backend") == "hf_adapter")
    native_quant_rkv_proto = latest(rows, lambda r: r.get("axis") == "native_quant_rkv_proto" and r.get("backend") == "hf_adapter")
    native_quant_w4_rkv_proto = latest(rows, lambda r: r.get("axis") == "native_quant_w4_rkv_proto" and r.get("backend") == "hf_adapter")
    native_quant_rkv_sweep = latest_by_key(
        [
            r
            for r in rows
            if r.get("axis") == "native_quant_rkv_sweep"
            and r.get("backend") == "hf_adapter"
        ],
        lambda r: r.get("quantization"),
    )
    quant_rows_all = [r for r in rows if r.get("axis") == "quantization" and r.get("backend") == "hf_adapter"]
    quant_rows_canonical = [r for r in quant_rows_all if is_canonical_quant_model(r)]
    if not quant_rows_canonical:
        quant_rows_canonical = quant_rows_all
    # Keep the target gate anchored to the canonical memory-first quantization
    # policy. Hybrid policies such as decode_hot are useful optimization probes,
    # but they trade some W4 footprint for speed and should not overwrite the
    # memory-target rows just because they were measured later.
    quant_rows = [r for r in quant_rows_canonical if r.get("quant_skip_policy") in (None, "", "memory", "small_lora", "minimal")]
    quant_latest = latest_by_key(quant_rows, lambda r: r.get("quantization"))
    quant_variant_latest = latest_by_key(
        quant_rows_canonical,
        lambda r: (r.get("quantization"), r.get("quant_skip_policy") or "memory"),
    )
    quant_model_latest = latest_by_key(
        quant_rows_all,
        lambda r: (model_label(r), r.get("quantization"), r.get("quant_skip_policy") or "memory"),
    )
    quant_variant_pass = [r for r in quant_variant_latest if r.get("status") == "pass"]
    quant_base_for_variants = max(
        [r for r in quant_variant_pass if r.get("quantization") == "none"],
        key=lambda r: (float(r.get("decode_tokps") or 0.0), int(r.get("_lineno", 0))),
        default=None,
    )
    quant_base_decode_for_variants = num(quant_base_for_variants, "decode_tokps")
    quant_base_footprint_for_variants = num(quant_base_for_variants, "model_footprint_mb")
    quant_base_peak_for_variants = num(quant_base_for_variants, "peak_vram_mb")
    quant_best_variants = []
    for mode in ("8bit", "4bit"):
        candidates = [r for r in quant_variant_pass if r.get("quantization") == mode]
        if not candidates:
            continue
        best_speed = max(
            candidates,
            key=lambda r: (float(r.get("decode_tokps") or 0.0), -float(r.get("model_footprint_mb") or 1e30), int(r.get("_lineno", 0))),
        )
        best_memory = min(
            candidates,
            key=lambda r: (float(r.get("model_footprint_mb") or 1e30), -float(r.get("decode_tokps") or 0.0), -int(r.get("_lineno", 0))),
        )
        best_decode = num(best_speed, "decode_tokps")
        best_footprint = num(best_speed, "model_footprint_mb")
        best_peak = num(best_speed, "peak_vram_mb")
        quant_best_variants.append(
            {
                "quantization": mode,
                "best_speed": compact(
                    best_speed,
                    [
                        "_lineno",
                        "quant_skip_policy",
                        "decode_tokps",
                        "reference_decode_tokps",
                        "fast_decode_tokps",
                        "fast_forward_backend",
                        "model_footprint_mb",
                        "peak_vram_mb",
                    ],
                ),
                "best_memory": compact(
                    best_memory,
                    [
                        "_lineno",
                        "quant_skip_policy",
                        "decode_tokps",
                        "model_footprint_mb",
                        "peak_vram_mb",
                    ],
                ),
                "decode_ratio_vs_fp16": round(ratio(best_decode, quant_base_decode_for_variants), 4)
                if quant_base_decode_for_variants
                else None,
                "footprint_ratio_vs_fp16": round(ratio(best_footprint, quant_base_footprint_for_variants), 4)
                if quant_base_footprint_for_variants
                else None,
                "peak_vram_ratio_vs_fp16": round(ratio(best_peak, quant_base_peak_for_variants), 4)
                if quant_base_peak_for_variants
                else None,
            }
        )
    device_map_smoke = latest(rows, lambda r: r.get("axis") == "device_map_smoke" and r.get("backend") == "hf_adapter")
    speculative_decode = latest(rows, lambda r: r.get("axis") == "speculative_decode" and r.get("backend") == "hf_adapter" and r.get("status") != "skip")
    larger_rows = [r for r in rows if r.get("axis") == "larger_model_smoke" and r.get("backend") == "hf_adapter"]
    larger_latest = latest_by_key(
        larger_rows,
        lambda r: r.get("model_size_label") or r.get("model_name"),
    )
    # Training smoke rows intentionally ignore the inference dtype filter: the
    # stable V100 training smoke currently runs with train_dtype=fp32 while most
    # serving regression reports are filtered with dtype=fp16. Device filtering
    # still applies so reports stay tied to the requested hardware.
    training_rows = [
        r for r in filt(raw_rows, device=args.device, dtype=None)
        if r.get("axis") == "training_smoke" and r.get("backend") == "hf_adapter"
    ]
    training_latest = latest_by_key(training_rows, lambda r: r.get("trainer_backend"))
    deepspeed_rows = [
        r for r in filt(raw_rows, device=args.device, dtype=None)
        if r.get("axis") == "deepspeed_training_smoke" and r.get("backend") == "hf_adapter"
    ]
    deepspeed_latest = latest_by_key(deepspeed_rows, lambda r: r.get("zero_stage"))
    albatross_rows = [
        r for r in rows
        if r.get("axis") == "albatross_speed" and r.get("backend") == "albatross"
    ]
    albatross_latest = latest_by_key(
        albatross_rows,
        lambda r: (
            r.get("engine"),
            r.get("model_size_label") or r.get("model_path"),
            r.get("batch_size"),
            r.get("tokens_per_sequence"),
        ),
    )
    albatross_best_by_case: dict[tuple[int, int], dict[str, Any]] = {}
    for row in albatross_latest:
        if row.get("batch_size") is None or row.get("tokens_per_sequence") is None:
            continue
        key = (int(row["batch_size"]), int(row["tokens_per_sequence"]))
        old = albatross_best_by_case.get(key)
        if old is None or float(row.get("tokps_p50") or 0.0) >= float(old.get("tokps_p50") or 0.0):
            albatross_best_by_case[key] = row

    hf_decode_by_bsz_all: dict[int, dict[str, Any]] = {}
    for row in batch_latest:
        if row.get("decode_api") != "rwkv7_forward_token" or row.get("batch_size") is None:
            continue
        bsz = int(row["batch_size"])
        old = hf_decode_by_bsz_all.get(bsz)
        if old is None or int(row.get("_lineno", 0)) >= int(old.get("_lineno", 0)):
            hf_decode_by_bsz_all[bsz] = row
    hf_decode_by_bsz_default: dict[int, dict[str, Any]] = {}
    for row in batch_default_latest:
        if row.get("batch_size") is None:
            continue
        hf_decode_by_bsz_default[int(row["batch_size"])] = row

    hf_prefill_by_case: dict[tuple[int, int, str | None], dict[str, Any]] = {}

    def hf_prefill_tokps(row: dict[str, Any] | None) -> tuple[float | None, str | None]:
        """Return the HF-adapter prefill throughput metric used for A/B.

        Different benchmark axes record different prefill paths.  For the
        native fused scan axis the useful adapter number is the native prefill
        throughput, not the FLA/HF reference throughput measured beside it.
        """

        if row is None:
            return None, None
        if row.get("axis") == "native_prefill_scan":
            val = num(row, "native_prefill_tokps_total")
            if val is not None:
                return val, "native_prefill_tokps_total"
            val = num(row, "hf_prefill_tokps_total")
            if val is not None:
                return val, "hf_prefill_tokps_total"
        val = num(row, "prefill_tokps_total")
        if val is not None:
            return val, "prefill_tokps_total"
        val = num(row, "prefill_tokps")
        if val is not None:
            return val, "prefill_tokps"
        return None, None

    for row in rows:
        if row.get("backend") != "hf_adapter":
            continue
        prompt_tokens = row.get("prompt_tokens")
        batch_size = row.get("batch_size")
        tokps, metric = hf_prefill_tokps(row)
        label = model_size_label(row)
        if tokps is not None and prompt_tokens is not None and batch_size is not None:
            key = (int(batch_size), int(prompt_tokens), label)
        elif row.get("axis") == "speed_mem" and tokps is not None and prompt_tokens is not None:
            key = (1, int(prompt_tokens), label)
        else:
            continue
        old = hf_prefill_by_case.get(key)
        old_tokps, _ = hf_prefill_tokps(old)
        if (
            old is None
            or old_tokps is None
            or float(tokps) > float(old_tokps)
            or (float(tokps) == float(old_tokps) and int(row.get("_lineno", 0)) >= int(old.get("_lineno", 0)))
        ):
            row = dict(row)
            row["_prefill_metric"] = metric
            row["_prefill_tokps_for_compare"] = tokps
            row["_model_size_label_for_compare"] = label
            hf_prefill_by_case[key] = row

    albatross_decode_comparison = []
    albatross_prefill_comparison = []
    for (bsz, tokens), alb in sorted(albatross_best_by_case.items()):
        alb_tokps = num(alb, "tokps_p50")
        if tokens == 1:
            hf = hf_decode_by_bsz_default.get(bsz) or hf_decode_by_bsz_all.get(bsz)
            hf_tokps = num(hf, "decode_tokps_total")
            if hf is not None and hf_tokps is not None:
                albatross_decode_comparison.append(
                    {
                        "batch_size": bsz,
                        "hf_decode_api": hf.get("decode_api"),
                        "hf_fast_token_backend_effective": fast_token_backend_effective(hf),
                        "hf_tokps_total": round(hf_tokps, 4),
                        "albatross_engine": alb.get("engine"),
                        "albatross_engine_config": alb.get("engine_config"),
                        "albatross_tokps_p50": round(alb_tokps, 4) if alb_tokps is not None else None,
                        "hf_vs_albatross_ratio": round(ratio(hf_tokps, alb_tokps), 4) if alb_tokps else None,
                    }
                )
            continue
        if tokens < 512:
            continue
        alb_label = model_size_label(alb)
        hf = hf_prefill_by_case.get((bsz, tokens, alb_label)) if alb_label is not None else None
        if hf is None:
            hf = hf_prefill_by_case.get((bsz, tokens, None))
        hf_tokps = num(hf, "_prefill_tokps_for_compare") if hf is not None else None
        if hf is not None and hf_tokps is not None:
            albatross_prefill_comparison.append(
                {
                    "batch_size": bsz,
                    "tokens_per_sequence": tokens,
                    "model_size_label": alb_label,
                    "hf_axis": hf.get("axis"),
                    "hf_model_size_label": hf.get("_model_size_label_for_compare"),
                    "hf_prefill_metric": hf.get("_prefill_metric"),
                    "hf_fused_scan_requested": hf.get("fused_scan_requested"),
                    "hf_scan_block_m": hf.get("scan_block_m"),
                    "hf_scan_num_warps": hf.get("scan_num_warps"),
                    "hf_prefill_fused_scan_output_effective": hf.get("prefill_fused_scan_output_effective"),
                    "hf_prefill_fused_clampw_scan_effective": hf.get("prefill_fused_clampw_scan_effective"),
                    "hf_prefill_dplr_scan_effective": hf.get("prefill_dplr_scan_effective"),
                    "hf_prefill_dplr_chunk_size": hf.get("prefill_dplr_chunk_size"),
                    "hf_prefill_fused_shift_mix_effective": hf.get("prefill_fused_shift_mix_effective"),
                    "hf_prefill_fused_state_prep_effective": hf.get("prefill_fused_state_prep_effective"),
                    "hf_prefill_state_prep_w_dtype": hf.get("prefill_state_prep_w_dtype"),
                    "hf_prefill_fused_output_effective": hf.get("prefill_fused_output_effective"),
                    "hf_prefill_fused_wavg_lora_effective": hf.get("prefill_fused_wavg_lora_effective"),
                    "hf_tokps_total": round(hf_tokps, 4),
                    "albatross_engine": alb.get("engine"),
                    "albatross_engine_config": alb.get("engine_config"),
                    "albatross_tokps_p50": round(alb_tokps, 4) if alb_tokps is not None else None,
                    "hf_vs_albatross_ratio": round(ratio(hf_tokps, alb_tokps), 4) if alb_tokps else None,
                }
            )

    albatross_decode_ratios = [
        float(row["hf_vs_albatross_ratio"])
        for row in albatross_decode_comparison
        if row.get("hf_vs_albatross_ratio") is not None
    ]
    albatross_prefill_ratios = [
        float(row["hf_vs_albatross_ratio"])
        for row in albatross_prefill_comparison
        if row.get("hf_vs_albatross_ratio") is not None
    ]
    albatross_decode_min = min(albatross_decode_ratios) if albatross_decode_ratios else None
    albatross_decode_max = max(albatross_decode_ratios) if albatross_decode_ratios else None
    albatross_prefill_min = min(albatross_prefill_ratios) if albatross_prefill_ratios else None
    albatross_prefill_max = max(albatross_prefill_ratios) if albatross_prefill_ratios else None
    quant_target_by_mode = {
        "8bit": {"decode_ratio_ge": 1.0, "footprint_ratio_le": 0.75},
        "4bit": {"decode_ratio_ge": 1.0, "footprint_ratio_le": 0.55},
    }
    fused_quant_targets = []
    for row in quant_best_variants:
        mode = row.get("quantization")
        targets = quant_target_by_mode.get(str(mode), {"decode_ratio_ge": 1.0, "footprint_ratio_le": None})
        decode_ratio = row.get("decode_ratio_vs_fp16")
        footprint_ratio = row.get("footprint_ratio_vs_fp16")
        footprint_target = targets.get("footprint_ratio_le")
        fused_quant_targets.append(
            {
                "quantization": mode,
                "best_speed_policy": (row.get("best_speed") or {}).get("quant_skip_policy") or "memory",
                "decode_ratio_vs_fp16": decode_ratio,
                "decode_target_ge": targets["decode_ratio_ge"],
                "decode_status": verdict_ge(decode_ratio, targets["decode_ratio_ge"]),
                "footprint_ratio_vs_fp16": footprint_ratio,
                "footprint_target_le": footprint_target,
                "footprint_status": verdict_le(footprint_ratio, footprint_target) if footprint_target is not None else "PENDING",
            }
        )
    fused_backend_targets = {
        "phase": "rwkv7_hf_fused_backend",
        "purpose": "Track the new native fused fp16 -> native W8/W4 backend against Albatross and fp16 speed targets.",
        "albatross_decode": {
            "current_ratio_min": round(albatross_decode_min, 4) if albatross_decode_min is not None else None,
            "current_ratio_max": round(albatross_decode_max, 4) if albatross_decode_max is not None else None,
            "p1_ratio_ge": 0.55,
            "p1_status": verdict_ge(albatross_decode_min, 0.55),
            "p2_ratio_ge": 0.75,
            "p2_status": verdict_ge(albatross_decode_min, 0.75),
            "p3_ratio_ge": 0.90,
            "p3_status": verdict_ge(albatross_decode_min, 0.90),
        },
        "albatross_prefill": {
            "current_ratio_min": round(albatross_prefill_min, 4) if albatross_prefill_min is not None else None,
            "current_ratio_max": round(albatross_prefill_max, 4) if albatross_prefill_max is not None else None,
            "p1_ratio_ge": 0.60,
            "p1_status": verdict_ge(albatross_prefill_min, 0.60),
            "p2_ratio_ge": 0.80,
            "p2_status": verdict_ge(albatross_prefill_min, 0.80),
        },
        "quantization": fused_quant_targets,
        "next_kernel_steps": [
            "preserve split-row prefill scan occupancy; do not promote full-head scan+output-prep fusion",
            "fuse larger norm/shift/projection/LoRA/state-prep regions instead of shallow output or FFN buckets",
            "keep cuBLAS-backed dense projections unless a deeper kernel wins end-to-end at bsz=1/4",
            "continue decode P1 with fused recurrent-output plus deeper projection/LoRA work",
            "add native W8/W4 pack plus fused dequant-GEMV and optimize packed kernels until W8/W4 >= fp16 end-to-end",
        ],
    }

    native_rows = [r for r in rows if r.get("axis") == "native_decode" and r.get("backend") == "hf_native_jit"]
    # The experimental FLA-free native_model correctness smoke normally runs
    # fp32, while the serving report is commonly filtered with --dtype fp16.
    # Keep it visible like training telemetry, but still honor the device
    # filter so reports remain hardware-specific.
    native_model_smoke = latest(
        filt(raw_rows, device=args.device, dtype=None),
        lambda r: r.get("axis") == "native_model_smoke" and r.get("backend") == "hf_native_model",
    )
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
    if batch_experimental_latest and batch_default_latest:
        default_by_bsz = {int(r["batch_size"]): r for r in batch_default_latest if r.get("batch_size") is not None}
        grouped_experimental: dict[str, list[float]] = defaultdict(list)
        for row in batch_experimental_latest:
            if row.get("batch_size") is None:
                continue
            default = default_by_bsz.get(int(row["batch_size"]))
            exp_tokps = num(row, "decode_tokps_total")
            default_tokps = num(default, "decode_tokps_total")
            exp_ratio = ratio(exp_tokps, default_tokps)
            if exp_ratio is not None:
                grouped_experimental[native_graph_flag_signature(row)].append(exp_ratio)
        for signature, ratios in sorted(grouped_experimental.items()):
            if ratios:
                focus.append(
                    f"experimental native_graph flags {signature} vs default recurrent-output path: "
                    f"min_ratio={min(ratios):.2f}x max_ratio={max(ratios):.2f}x"
                )
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
        fused_plan = projection_lora.get("fused_kernel_plan")
        if isinstance(fused_plan, dict):
            first_target = fused_plan.get("first_fused_fp16_target") or {}
            group = first_target.get("group")
            current_ms = first_target.get("current_ms")
            members = first_target.get("members") or []
            if group:
                focus.append(
                    f"fused projection first target: {group} current={current_ms} ms "
                    f"members={','.join(str(m) for m in members)}"
                )
        else:
            focus.append("projection_lora row lacks matrix-level fused kernel plan; rerun bench_projection_lora")
    if fused_projection_proto is None:
        focus.append("fused_projection_proto row pending")
    else:
        proto_speedup = fused_projection_proto.get("avg_speedup")
        backend = fused_projection_proto.get("prototype_backend")
        if proto_speedup is not None and float(proto_speedup) < 1.0:
            focus.append(
                f"fused R/K/V projection prototype backend={backend} is slower "
                f"({float(proto_speedup):.2f}x); optimize kernel before HF integration"
            )
        elif proto_speedup is not None:
            focus.append(
                f"fused R/K/V projection prototype backend={backend} speedup={float(proto_speedup):.2f}x; "
                "validate end-to-end fast-token integration"
            )
    if not albatross_projection_layout:
        focus.append("Albatross-inspired projection layout sweep pending")
    else:
        summaries = []
        slow_batches = []
        for row in albatross_projection_layout:
            best = row.get("best_config") if isinstance(row.get("best_config"), dict) else {}
            speed = best.get("avg_speedup")
            label = f"bsz={row.get('batch_size')} {best.get('backend')} m{best.get('block_m')} k{best.get('block_k')} {speed}x"
            summaries.append(label)
            if speed is not None and float(speed) < 1.0:
                slow_batches.append(row.get("batch_size"))
        if slow_batches:
            focus.append(
                "Albatross-inspired simple RKV split-K/layout sweep is slower than cuBLAS "
                f"for bsz={slow_batches}; best={'; '.join(summaries)}. "
                "Do not integrate this shallow GEMV; borrow deeper fusion instead (LN+mix+projection/output)."
            )
        else:
            focus.append(
                "Albatross-inspired projection layout sweep found faster configs: "
                f"{'; '.join(summaries)}; validate native_graph integration"
            )
    if fused_wa_lora_proto is None:
        focus.append("fused_wa_lora_proto row pending")
    else:
        wa_speedup = fused_wa_lora_proto.get("avg_speedup")
        backend = fused_wa_lora_proto.get("prototype_backend")
        if wa_speedup is not None and float(wa_speedup) < 1.0:
            focus.append(
                f"fused W/A LoRA prototype backend={backend} is slower "
                f"({float(wa_speedup):.2f}x); two-kernel LoRA grouping is not enough, fuse with R/K/V or more LoRA paths"
            )
        elif wa_speedup is not None:
            focus.append(
                f"fused W/A LoRA prototype backend={backend} speedup={float(wa_speedup):.2f}x; "
                "consider grouping with R/K/V projection path"
            )
    if fused_wag_lora_proto is None:
        focus.append("fused_wag_lora_proto row pending")
    else:
        wag_speedup = fused_wag_lora_proto.get("avg_speedup")
        backend = fused_wag_lora_proto.get("prototype_backend")
        if wag_speedup is not None and float(wag_speedup) < 1.0:
            focus.append(
                f"fused W/A/G LoRA prototype backend={backend} is slower "
                f"({float(wag_speedup):.2f}x); grouped LoRA alone still needs R/K/V/state fusion"
            )
        elif wag_speedup is not None:
            focus.append(
                f"fused W/A/G LoRA prototype backend={backend} speedup={float(wag_speedup):.2f}x; "
                "next combine with R/K/V projection path"
            )
    if fused_wavg_lora_proto is None:
        focus.append("fused_wavg_lora_proto row pending")
    else:
        wavg_speedup = fused_wavg_lora_proto.get("avg_speedup")
        backend = fused_wavg_lora_proto.get("prototype_backend")
        if wavg_speedup is not None and float(wavg_speedup) < 1.0:
            focus.append(
                f"fused W/A/G/V-gate LoRA prototype backend={backend} is slower "
                f"({float(wavg_speedup):.2f}x); V-gate grouping alone is not enough, fuse with R/K/V or skip native_graph integration"
            )
        elif wavg_speedup is not None:
            focus.append(
                f"fused W/A/G/V-gate LoRA prototype backend={backend} speedup={float(wavg_speedup):.2f}x; "
                "compare against W/A/G-only and then test native_graph integration"
            )
    if fused_rkv_wag_projection_proto is None:
        focus.append("fused_rkv_wag_projection_proto row pending")
    else:
        combo_rows = [r for r in rows if r.get("axis") == "fused_rkv_wag_projection_proto" and r.get("backend") == "hf_adapter"]
        combo_speedups = [float(r["avg_speedup"]) for r in combo_rows if r.get("avg_speedup") is not None]
        combo_cases = sorted(
            {(r.get("batch_size"), r.get("sequence_length"), r.get("tokens_total"), r.get("block_m"), r.get("block_r"), r.get("block_k")) for r in combo_rows},
            key=str,
        )
        combo_speedup = fused_rkv_wag_projection_proto.get("avg_speedup")
        backend = fused_rkv_wag_projection_proto.get("prototype_backend")
        if combo_speedups and min(combo_speedups) < 1.0:
            focus.append(
                f"fused R/K/V + W/A/G projection prototype backend={backend} is slower "
                f"for cases={combo_cases}; speedup min={min(combo_speedups):.2f}x max={max(combo_speedups):.2f}x; "
                "optimize larger norm/shift/projection fusion before HF integration"
            )
        elif combo_speedup is not None:
            focus.append(
                f"fused R/K/V + W/A/G projection prototype backend={backend} "
                f"speedup={float(combo_speedup):.2f}x tokens_total={fused_rkv_wag_projection_proto.get('tokens_total')}; "
                "next target full attention fusion/integration"
            )
    if fused_attn_output_proto is None:
        focus.append("fused_attn_output_proto row pending")
    else:
        output_speedup = fused_attn_output_proto.get("avg_speedup")
        backend = fused_attn_output_proto.get("prototype_backend")
        if output_speedup is not None and float(output_speedup) < 1.0:
            focus.append(
                f"fused attention output prototype backend={backend} is slower "
                f"({float(output_speedup):.2f}x); keep o_proj/cuBLAS and fuse output prep only after deeper integration"
            )
        elif output_speedup is not None:
            focus.append(
                f"fused attention output prototype backend={backend} speedup={float(output_speedup):.2f}x; "
                "consider integration/full attention fusion"
            )
    if fused_attn_output_project_proto is None:
        focus.append("fused_attn_output_project_proto row pending")
    else:
        best_project = fused_attn_output_project_proto.get("best_fused_project") or {}
        project_speedup = best_project.get("speedup_vs_prep_cublas")
        project_current_speedup = best_project.get("speedup_vs_current")
        backend = fused_attn_output_project_proto.get("prototype_backend")
        block_m = best_project.get("block_m")
        if project_speedup is not None and float(project_speedup) >= 1.0:
            focus.append(
                f"fused output+o_proj prototype backend={backend} block_m={block_m} "
                f"beats prep+cuBLAS by {float(project_speedup):.2f}x "
                f"({float(project_current_speedup):.2f}x vs current); validate native_graph integration"
            )
        elif project_speedup is not None:
            focus.append(
                f"fused output+o_proj prototype backend={backend} block_m={block_m} "
                f"is slower than prep+cuBLAS ({float(project_speedup):.2f}x); keep o_proj on cuBLAS"
            )
    if fused_ffn_proto is None:
        focus.append("fused_ffn_proto row pending")
    else:
        ffn_speedup = fused_ffn_proto.get("avg_speedup")
        backend = fused_ffn_proto.get("prototype_backend")
        if ffn_speedup is not None and float(ffn_speedup) < 1.0:
            focus.append(
                f"fused FFN prototype backend={backend} is slower "
                f"({float(ffn_speedup):.2f}x); two-kernel FFN is not enough, keep cuBLAS or fuse into larger graph"
            )
        elif ffn_speedup is not None:
            focus.append(
                f"fused FFN prototype backend={backend} speedup={float(ffn_speedup):.2f}x; "
                "validate native_graph integration"
            )
    if fused_shift_mix_proto is None:
        focus.append("fused_shift_mix_proto row pending")
    else:
        shift_speedup = fused_shift_mix_proto.get("avg_speedup")
        backend = fused_shift_mix_proto.get("prototype_backend")
        if shift_speedup is not None and float(shift_speedup) < 1.0:
            focus.append(
                f"fused attention shift-mix prototype backend={backend} is slower "
                f"({float(shift_speedup):.2f}x); keep it as telemetry and fuse deeper with projection/state update"
            )
        elif shift_speedup is not None:
            focus.append(
                f"fused attention shift-mix prototype backend={backend} speedup={float(shift_speedup):.2f}x; "
                + ("validate native prefill opt-in" if fused_shift_mix_proto.get("input_rank") == 3 else "validate native_graph integration")
            )
    if fused_recurrent_proto is None:
        focus.append("fused_recurrent_proto row pending")
    else:
        rec_speedup = fused_recurrent_proto.get("avg_speedup")
        backend = fused_recurrent_proto.get("prototype_backend")
        out_diff = fused_recurrent_proto.get("out_max_abs_diff")
        if rec_speedup is not None and float(rec_speedup) >= 1.0:
            if native_graph_fused_recurrent is None:
                focus.append(
                    f"fused recurrent prototype backend={backend} speedup={float(rec_speedup):.2f}x "
                    f"out_max_abs_diff={out_diff}; validate end-to-end native_graph integration"
                )
            else:
                focus.append(
                    f"fused recurrent prototype backend={backend} speedup={float(rec_speedup):.2f}x "
                    f"out_max_abs_diff={out_diff}; opt-in native_graph integration row present"
                )
        elif rec_speedup is not None:
            focus.append(
                f"fused recurrent prototype backend={backend} is slower "
                f"({float(rec_speedup):.2f}x); keep optimizing before integration"
            )
    if not fused_recurrent_scan_proto:
        focus.append("fused_recurrent_scan_proto row pending")
    else:
        scan_speedups = [
            float(r["native_vs_fla_speedup"])
            for r in fused_recurrent_scan_proto
            if r.get("native_vs_fla_speedup") is not None
        ]
        scan_cosines = [
            float(r["native_vs_torch_out_min_cosine"])
            for r in fused_recurrent_scan_proto
            if r.get("native_vs_torch_out_min_cosine") is not None
        ]
        cases = sorted(
            {(r.get("batch_size"), r.get("tokens"), r.get("block_m"), r.get("num_warps")) for r in fused_recurrent_scan_proto},
            key=str,
        )
        if scan_speedups:
            scan_next = "use full native-prefill rows for promotion and fuse projection/output" if native_prefill_scan else "next wire into full native prefill and fuse projection/output"
            focus.append(
                f"fused recurrent scan prefill prototype present for cases={cases}; "
                f"native/FLA recurrent-only speedup min={min(scan_speedups):.2f}x max={max(scan_speedups):.2f}x; "
                + scan_next
            )
        else:
            focus.append(
                f"fused recurrent scan prefill prototype present for cases={cases}; "
                "add FLA recurrent-only timing for speed ratio"
            )
        if scan_cosines and min(scan_cosines) < 0.999:
            focus.append(
                f"fused recurrent scan native-vs-torch cosine min={min(scan_cosines):.6f}; "
                "tighten numerical validation before integration"
            )
    if not native_prefill_scan:
        focus.append("native_prefill_scan end-to-end row pending")
    else:
        ok_rows = [
            r for r in native_prefill_scan
            if r.get("greedy_match") is True and r.get("decode_after_prefill_greedy_match") is True
        ]
        speedups = [
            float(r["native_vs_hf_speedup"])
            for r in native_prefill_scan
            if r.get("native_vs_hf_speedup") is not None
        ]
        cases = sorted(
            {
                (
                    r.get("batch_size"),
                    r.get("prompt_tokens"),
                    bool(r.get("fused_scan_requested")),
                    r.get("scan_block_m"),
                    r.get("scan_num_warps"),
                    bool(r.get("prefill_fused_scan_output_effective")),
                    bool(r.get("prefill_fused_clampw_scan_effective")),
                    bool(r.get("prefill_fused_shift_mix_effective")),
                    bool(r.get("prefill_fused_state_prep_effective")),
                    bool(r.get("prefill_fused_state_scan_effective")),
                    bool(r.get("prefill_fused_output_effective")),
                    bool(r.get("prefill_fused_wavg_lora_effective")),
                    r.get("prefill_fused_wavg_lora_max_m"),
                )
                for r in native_prefill_scan
            },
            key=str,
        )
        if speedups:
            focus.append(
                f"native_prefill_scan end-to-end rows present for cases={cases}; "
                f"speedup min={min(speedups):.2f}x max={max(speedups):.2f}x; "
                f"correct_cache_rows={len(ok_rows)}/{len(native_prefill_scan)}"
            )
            scan_output_rows = [r for r in native_prefill_scan if r.get("prefill_fused_scan_output_effective")]
            scan_output_ratios: list[float] = []
            for row in scan_output_rows:
                base_candidates = [
                    r
                    for r in native_prefill_scan
                    if r.get("batch_size") == row.get("batch_size")
                    and r.get("prompt_tokens") == row.get("prompt_tokens")
                    and bool(r.get("prefill_fused_state_prep_effective")) == bool(row.get("prefill_fused_state_prep_effective"))
                    and not bool(r.get("prefill_fused_clampw_scan_effective"))
                    and not bool(r.get("prefill_fused_scan_output_effective"))
                    and not bool(r.get("prefill_fused_output_effective"))
                    and not bool(r.get("prefill_fused_wavg_lora_effective"))
                ]
                base = max(base_candidates, key=lambda r: float(r.get("native_prefill_tokps_total") or 0.0), default=None)
                row_tokps = num(row, "native_prefill_tokps_total")
                base_tokps = num(base, "native_prefill_tokps_total")
                if row_tokps is not None and base_tokps:
                    scan_output_ratios.append(float(row_tokps) / float(base_tokps))
            if scan_output_ratios:
                focus.append(
                    f"native prefill fused scan+output-prep opt-in A/B ratio min={min(scan_output_ratios):.3f}x "
                    f"max={max(scan_output_ratios):.3f}x; promote only if it beats split scan across target bsz"
                )
            clampw_rows = [r for r in native_prefill_scan if r.get("prefill_fused_clampw_scan_effective")]
            clampw_ratios: list[float] = []
            for row in clampw_rows:
                base_candidates = [
                    r
                    for r in native_prefill_scan
                    if r.get("batch_size") == row.get("batch_size")
                    and r.get("prompt_tokens") == row.get("prompt_tokens")
                    and r.get("scan_block_m") == row.get("scan_block_m")
                    and r.get("scan_num_warps") == row.get("scan_num_warps")
                    and bool(r.get("prefill_fused_state_prep_effective")) == bool(row.get("prefill_fused_state_prep_effective"))
                    and bool(r.get("prefill_fused_shift_mix_effective")) == bool(row.get("prefill_fused_shift_mix_effective"))
                    and bool(r.get("prefill_fused_output_effective")) == bool(row.get("prefill_fused_output_effective"))
                    and bool(r.get("prefill_fused_wavg_lora_effective")) == bool(row.get("prefill_fused_wavg_lora_effective"))
                    and not bool(r.get("prefill_fused_scan_output_effective"))
                    and not bool(r.get("prefill_fused_clampw_scan_effective"))
                ]
                base = max(base_candidates, key=lambda r: float(r.get("native_prefill_tokps_total") or 0.0), default=None)
                row_tokps = num(row, "native_prefill_tokps_total")
                base_tokps = num(base, "native_prefill_tokps_total")
                if row_tokps is not None and base_tokps:
                    clampw_ratios.append(float(row_tokps) / float(base_tokps))
            if clampw_ratios:
                focus.append(
                    f"native prefill raw-W clampw scan opt-in A/B ratio min={min(clampw_ratios):.3f}x "
                    f"max={max(clampw_ratios):.3f}x; promote only if state-prep+scan component and end-to-end improve"
                )
            dplr_rows = [r for r in native_prefill_scan if r.get("prefill_dplr_scan_effective")]
            if dplr_rows:
                chunks = sorted({r.get("prefill_dplr_chunk_size") for r in dplr_rows}, key=str)
                ok = sum(
                    1
                    for r in dplr_rows
                    if r.get("greedy_match") is True and r.get("decode_after_prefill_greedy_match") is True
                )
                focus.append(
                    f"native prefill DPLR/chunked scan rows present for chunk_sizes={chunks}; "
                    f"correct_cache_rows={ok}/{len(dplr_rows)}; current pure-torch path is correctness-only until affine/WY chunk kernels land"
                )
            output_rows = [
                r
                for r in native_prefill_scan
                if r.get("prefill_fused_output_effective") and not r.get("prefill_fused_scan_output_effective")
            ]
            output_ratios: list[float] = []
            for row in output_rows:
                base_candidates = [
                    r
                    for r in native_prefill_scan
                    if r.get("batch_size") == row.get("batch_size")
                    and r.get("prompt_tokens") == row.get("prompt_tokens")
                    and r.get("scan_block_m") == row.get("scan_block_m")
                    and bool(r.get("prefill_fused_state_prep_effective")) == bool(row.get("prefill_fused_state_prep_effective"))
                    and bool(r.get("prefill_fused_clampw_scan_effective")) == bool(row.get("prefill_fused_clampw_scan_effective"))
                    and not bool(r.get("prefill_fused_scan_output_effective"))
                    and not bool(r.get("prefill_fused_output_effective"))
                    and not bool(r.get("prefill_fused_wavg_lora_effective"))
                ]
                base = max(base_candidates, key=lambda r: float(r.get("native_prefill_tokps_total") or 0.0), default=None)
                row_tokps = num(row, "native_prefill_tokps_total")
                base_tokps = num(base, "native_prefill_tokps_total")
                if row_tokps is not None and base_tokps:
                    output_ratios.append(float(row_tokps) / float(base_tokps))
            if output_ratios:
                focus.append(
                    f"native prefill fused output-prep opt-in A/B ratio min={min(output_ratios):.3f}x "
                    f"max={max(output_ratios):.3f}x; keep telemetry-only unless full prefill improves"
                )
            shift_rows = [r for r in native_prefill_scan if r.get("prefill_fused_shift_mix_effective")]
            shift_ratios: list[float] = []
            for row in shift_rows:
                base_candidates = [
                    r
                    for r in native_prefill_scan
                    if r.get("batch_size") == row.get("batch_size")
                    and r.get("prompt_tokens") == row.get("prompt_tokens")
                    and r.get("scan_block_m") == row.get("scan_block_m")
                    and r.get("scan_num_warps") == row.get("scan_num_warps")
                    and bool(r.get("prefill_fused_scan_output_effective")) == bool(row.get("prefill_fused_scan_output_effective"))
                    and bool(r.get("prefill_fused_clampw_scan_effective")) == bool(row.get("prefill_fused_clampw_scan_effective"))
                    and bool(r.get("prefill_fused_state_prep_effective")) == bool(row.get("prefill_fused_state_prep_effective"))
                    and bool(r.get("prefill_fused_output_effective")) == bool(row.get("prefill_fused_output_effective"))
                    and bool(r.get("prefill_fused_wavg_lora_effective")) == bool(row.get("prefill_fused_wavg_lora_effective"))
                    and not bool(r.get("prefill_fused_shift_mix_effective"))
                ]
                base = max(base_candidates, key=lambda r: float(r.get("native_prefill_tokps_total") or 0.0), default=None)
                row_tokps = num(row, "native_prefill_tokps_total")
                base_tokps = num(base, "native_prefill_tokps_total")
                if row_tokps is not None and base_tokps:
                    shift_ratios.append(float(row_tokps) / float(base_tokps))
            if shift_ratios:
                focus.append(
                    f"native prefill fused shift-mix opt-in A/B ratio min={min(shift_ratios):.3f}x "
                    f"max={max(shift_ratios):.3f}x; promote only if full prefill wins across target bsz"
                )
            warp_rows = [
                r
                for r in native_prefill_scan
                if r.get("scan_num_warps") is not None
                and r.get("native_prefill_tokps_total") is not None
                and bool(r.get("prefill_fused_state_prep_effective"))
                and not bool(r.get("prefill_fused_scan_output_effective"))
                and not bool(r.get("prefill_fused_clampw_scan_effective"))
                and not bool(r.get("prefill_fused_shift_mix_effective"))
                and not bool(r.get("prefill_fused_output_effective"))
                and not bool(r.get("prefill_fused_wavg_lora_effective"))
            ]
            if warp_rows:
                best_warp_by_case = []
                for key in sorted({(r.get("batch_size"), r.get("prompt_tokens")) for r in warp_rows}, key=str):
                    candidates = [r for r in warp_rows if (r.get("batch_size"), r.get("prompt_tokens")) == key]
                    best = max(candidates, key=lambda r: float(r.get("native_prefill_tokps_total") or 0.0))
                    best_warp_by_case.append(
                        f"bsz={key[0]} tok={key[1]} warps={best.get('scan_num_warps')} tokps={best.get('native_prefill_tokps_total')}"
                    )
                focus.append(
                    "native prefill scan num_warps sweep present; best explicit rows "
                    + "; ".join(best_warp_by_case)
                    + "; no default promotion without repeated end-to-end win"
                )
        else:
            focus.append(f"native_prefill_scan end-to-end rows present for cases={cases}; add timing fields")
    if not native_prefill_breakdown:
        focus.append("native_prefill_breakdown rows pending for bsz=1 prefill bottleneck attribution")
    else:
        bsz1_breakdowns = [r for r in native_prefill_breakdown if int(r.get("batch_size") or 0) == 1]
        chosen = max(bsz1_breakdowns or native_prefill_breakdown, key=lambda r: int(r.get("_lineno", 0)))
        top = chosen.get("top_components") or []
        if top:
            top_name = top[0][0]
            top_ms = top[0][1]
            top_share = top[0][2]
            focus.append(
                f"native_prefill_breakdown top component for bsz={chosen.get('batch_size')} "
                f"prompt={chosen.get('prompt_tokens')}: {top_name} {top_ms}ms share={top_share}; "
                "optimize this before touching cache"
            )
            fine_rows = [r for r in native_prefill_breakdown if r.get("fine_attention_breakdown")]
            if fine_rows:
                fine_bsz1 = [r for r in fine_rows if int(r.get("batch_size") or 0) == 1]
                fine = max(fine_bsz1 or fine_rows, key=lambda r: int(r.get("_lineno", 0)))
                cm = fine.get("component_ms") or {}
                lora_sum = sum(float(cm.get(k) or 0.0) for k in ("attn_lora_w", "attn_lora_a", "attn_lora_g", "attn_lora_v_gate", "attn_lora_wavg_fused"))
                dense_rkv_sum = sum(float(cm.get(k) or 0.0) for k in ("attn_dense_r_proj", "attn_dense_k_proj", "attn_dense_v_proj"))
                if dense_rkv_sum <= 0.0:
                    dense_rkv_sum = float(cm.get("attn_dense_rkv") or 0.0)
                scan_ms = cm.get("recurrent_scan", cm.get("recurrent_scan_output_prep_fused"))
                state_ms = cm.get("attn_state_prep_fused", cm.get("attn_lora_state_prep"))
                norm_ms = cm.get("attn_norm_shift_mix")
                focus.append(
                    f"fine prefill breakdown bsz={fine.get('batch_size')} prompt={fine.get('prompt_tokens')}: "
                    f"scan={scan_ms}ms state_prep={state_ms}ms dense_rkv_sum={round(dense_rkv_sum, 4)}ms "
                    f"lora_sum={round(lora_sum, 4)}ms norm_shift_mix={norm_ms}ms; "
                    "next fusion should target scan plus norm/shift/projection/state prep, not cache"
                )
                layer_rows = [r for r in fine_rows if r.get("layer_breakdown") and r.get("top_layers_by_total")]
                if layer_rows:
                    layer_bsz1 = [r for r in layer_rows if int(r.get("batch_size") or 0) == 1]
                    layer_row = max(layer_bsz1 or layer_rows, key=lambda r: int(r.get("_lineno", 0)))
                    top_layers = layer_row.get("top_layers_by_total") or []
                    formatted_layers = "; ".join(
                        f"L{int(layer)}={float(ms):.4f}ms"
                        for layer, ms in top_layers[:5]
                    )
                    layer_components = layer_row.get("layer_top_components") or {}
                    first_layer = str(top_layers[0][0]) if top_layers else None
                    first_layer_top = layer_components.get(first_layer, []) if first_layer is not None else []
                    formatted_components = ", ".join(
                        f"{name}={float(ms):.4f}ms"
                        for name, ms in first_layer_top[:3]
                    )
                    focus.append(
                        f"layer prefill breakdown bsz={layer_row.get('batch_size')} prompt={layer_row.get('prompt_tokens')}: "
                        f"top layers {formatted_layers}; hottest layer top components {formatted_components}"
                    )
        else:
            focus.append("native_prefill_breakdown rows present but top_components missing")
    fast_prefill_ttft = [
        r for r in ttft_tpot
        if r.get("metric") == "ttft" and str(r.get("fast_prefill_env")) not in {"", "0", "false", "False", "no", "off"}
    ]
    if fast_prefill_ttft:
        best = max(fast_prefill_ttft, key=lambda r: float(r.get("prefill_tokps_p50") or 0.0))
        focus.append(
            f"HF forward TTFT fast-prefill row present: prompt={best.get('prompt_tokens')} "
            f"p50={best.get('p50_ms')} ms tokps={best.get('prefill_tokps_p50')} "
            f"backend={best.get('fast_prefill_backend_effective')} fused_scan_env={best.get('native_prefill_fused_scan_env')}"
        )
    if fused_recurrent_output_proto is None:
        focus.append("fused_recurrent_output_proto row pending")
    else:
        ro_speedup = fused_recurrent_output_proto.get("avg_speedup_vs_split")
        ro_current = fused_recurrent_output_proto.get("avg_speedup_vs_current")
        backend = fused_recurrent_output_proto.get("prototype_backend")
        out_diff = fused_recurrent_output_proto.get("split_out_max_abs_diff")
        if ro_speedup is not None and float(ro_speedup) >= 1.0:
            focus.append(
                f"fused recurrent+output prototype backend={backend} beats split recurrent/output by "
                f"{float(ro_speedup):.2f}x ({float(ro_current):.2f}x vs torch current) split_out_max_abs_diff={out_diff}; "
                "validate native_graph integration and batch sweep"
            )
        elif ro_speedup is not None:
            focus.append(
                f"fused recurrent+output prototype backend={backend} is slower than split recurrent/output "
                f"({float(ro_speedup):.2f}x); keep split kernels"
            )
    if native_graph_fused_recurrent is None:
        focus.append("native_graph fused recurrent integration row pending")
    else:
        ng_speedup = native_graph_fused_recurrent.get("speedup")
        greedy_match = native_graph_fused_recurrent.get("greedy_match")
        greedy_total = native_graph_fused_recurrent.get("greedy_total")
        if ng_speedup is not None and float(ng_speedup) >= 1.0:
            focus.append(
                f"native_graph fused recurrent integration passes greedy {greedy_match}/{greedy_total} "
                f"with speedup={float(ng_speedup):.2f}x"
            )
        elif ng_speedup is not None:
            focus.append(
                f"native_graph fused recurrent integration passes greedy {greedy_match}/{greedy_total} "
                f"but speedup={float(ng_speedup):.2f}x; keep it optional until deeper fusion improves end-to-end"
            )
    if native_graph_fused_recurrent_output is None:
        focus.append("native_graph fused recurrent+output integration row pending")
    else:
        ngro_speedup = native_graph_fused_recurrent_output.get("speedup")
        greedy_match = native_graph_fused_recurrent_output.get("greedy_match")
        greedy_total = native_graph_fused_recurrent_output.get("greedy_total")
        out_diff = native_graph_fused_recurrent_output.get("max_abs_diff_first_step")
        if ngro_speedup is not None and float(ngro_speedup) >= 1.0:
            focus.append(
                f"native_graph fused recurrent+output integration passes greedy {greedy_match}/{greedy_total} "
                f"with speedup={float(ngro_speedup):.2f}x max_abs_diff={out_diff}; default-on with env fallback while testing broader device matrix"
            )
        elif ngro_speedup is not None:
            focus.append(
                f"native_graph fused recurrent+output integration passes greedy {greedy_match}/{greedy_total} "
                f"but speedup={float(ngro_speedup):.2f}x; keep opt-in"
            )
    if native_graph_fused_recurrent_output_sweep:
        sweep_speeds = [num(r, "speedup") for r in native_graph_fused_recurrent_output_sweep]
        sweep_speeds = [v for v in sweep_speeds if v is not None]
        sweep_batches = [r.get("batch_size") for r in native_graph_fused_recurrent_output_sweep]
        greedy_ok = all(
            r.get("greedy_match") is not None
            and r.get("greedy_total") is not None
            and int(r.get("greedy_match")) == int(r.get("greedy_total"))
            for r in native_graph_fused_recurrent_output_sweep
        )
        if sweep_speeds and greedy_ok:
            focus.append(
                f"native_graph fused recurrent+output batch matrix covers bsz={sweep_batches} "
                f"with min_speedup={min(sweep_speeds):.2f}x and greedy exact"
            )
    if native_graph_fused_output is None:
        focus.append("native_graph fused output integration row pending")
    else:
        ngo_speedup = native_graph_fused_output.get("speedup")
        greedy_match = native_graph_fused_output.get("greedy_match")
        greedy_total = native_graph_fused_output.get("greedy_total")
        out_diff = native_graph_fused_output.get("max_abs_diff_first_step")
        if ngo_speedup is not None and float(ngo_speedup) >= 1.0:
            focus.append(
                f"native_graph fused output integration passes greedy {greedy_match}/{greedy_total} "
                f"with speedup={float(ngo_speedup):.2f}x max_abs_diff={out_diff}; default-on with env fallback while testing broader device matrix"
            )
        elif ngo_speedup is not None:
            focus.append(
                f"native_graph fused output integration passes greedy {greedy_match}/{greedy_total} "
                f"but speedup={float(ngo_speedup):.2f}x; keep it optional until deeper attention fusion improves end-to-end"
            )
    if native_graph_fused_output_sweep:
        sweep_speeds = [num(r, "speedup") for r in native_graph_fused_output_sweep]
        sweep_speeds = [v for v in sweep_speeds if v is not None]
        sweep_batches = [r.get("batch_size") for r in native_graph_fused_output_sweep]
        greedy_ok = all(
            r.get("greedy_match") is not None
            and r.get("greedy_total") is not None
            and int(r.get("greedy_match")) == int(r.get("greedy_total"))
            for r in native_graph_fused_output_sweep
        )
        if sweep_speeds and greedy_ok:
            focus.append(
                f"native_graph fused output batch matrix covers bsz={sweep_batches} "
                f"with min_speedup={min(sweep_speeds):.2f}x and greedy exact"
            )
    if native_graph_fused_output_project is None:
        focus.append("native_graph fused output+o_proj integration row pending")
    else:
        ngop_speedup = native_graph_fused_output_project.get("speedup")
        greedy_match = native_graph_fused_output_project.get("greedy_match")
        greedy_total = native_graph_fused_output_project.get("greedy_total")
        proj_diff = native_graph_fused_output_project.get("max_abs_diff_first_step")
        block_m = native_graph_fused_output_project.get("output_project_block_m")
        if ngop_speedup is not None and float(ngop_speedup) >= 1.0:
            focus.append(
                f"native_graph fused output+o_proj integration block_m={block_m} passes greedy {greedy_match}/{greedy_total} "
                f"with speedup={float(ngop_speedup):.2f}x max_abs_diff={proj_diff}; test batch/device matrix before default"
            )
        elif ngop_speedup is not None:
            focus.append(
                f"native_graph fused output+o_proj integration block_m={block_m} passes greedy {greedy_match}/{greedy_total} "
                f"but speedup={float(ngop_speedup):.2f}x; keep opt-in and profile kernel occupancy"
            )
    if native_graph_fused_output_project_sweep:
        sweep_speeds = [num(r, "speedup") for r in native_graph_fused_output_project_sweep]
        sweep_speeds = [v for v in sweep_speeds if v is not None]
        sweep_batches = [r.get("batch_size") for r in native_graph_fused_output_project_sweep]
        greedy_ok = all(
            r.get("greedy_match") is not None
            and r.get("greedy_total") is not None
            and int(r.get("greedy_match")) == int(r.get("greedy_total"))
            for r in native_graph_fused_output_project_sweep
        )
        if sweep_speeds and greedy_ok:
            min_speed = min(sweep_speeds)
            if min_speed >= 1.0:
                focus.append(
                    f"native_graph fused output+o_proj batch matrix covers bsz={sweep_batches} "
                    f"with min_speedup={min_speed:.2f}x and greedy exact"
                )
            else:
                focus.append(
                    f"native_graph fused output+o_proj batch matrix covers bsz={sweep_batches} "
                    f"with greedy exact but min_speedup={min_speed:.2f}x; keep opt-in, not defaultable"
                )
    if native_graph_fused_wag_lora is None:
        focus.append("native_graph fused W/A/G LoRA integration row pending")
    else:
        ngw_speedup = native_graph_fused_wag_lora.get("speedup")
        greedy_match = native_graph_fused_wag_lora.get("greedy_match")
        greedy_total = native_graph_fused_wag_lora.get("greedy_total")
        wag_diff = native_graph_fused_wag_lora.get("max_abs_diff_first_step")
        if ngw_speedup is not None and float(ngw_speedup) >= 1.0:
            focus.append(
                f"native_graph fused W/A/G LoRA latest row passes greedy {greedy_match}/{greedy_total} "
                f"with speedup={float(ngw_speedup):.2f}x max_abs_diff={wag_diff}; use sweep before any default decision"
            )
        elif ngw_speedup is not None:
            focus.append(
                f"native_graph fused W/A/G LoRA integration passes greedy {greedy_match}/{greedy_total} "
                f"but speedup={float(ngw_speedup):.2f}x; keep opt-in and tune LoRA tiles/deeper projection fusion"
            )
    if native_graph_fused_wag_lora_sweep:
        sweep_speeds = [num(r, "speedup") for r in native_graph_fused_wag_lora_sweep]
        sweep_speeds = [v for v in sweep_speeds if v is not None]
        sweep_batches = [r.get("batch_size") for r in native_graph_fused_wag_lora_sweep]
        greedy_ok = all(
            r.get("greedy_match") is not None
            and r.get("greedy_total") is not None
            and int(r.get("greedy_match")) == int(r.get("greedy_total"))
            for r in native_graph_fused_wag_lora_sweep
        )
        if sweep_speeds and greedy_ok:
            min_speed = min(sweep_speeds)
            if min_speed >= 1.0:
                focus.append(
                    f"native_graph fused W/A/G LoRA batch matrix covers bsz={sweep_batches} "
                    f"with min_speedup={min_speed:.2f}x and greedy exact"
                )
            else:
                focus.append(
                    f"native_graph fused W/A/G LoRA batch matrix covers bsz={sweep_batches} "
                    f"with greedy exact but min_speedup={min_speed:.2f}x; keep opt-in, not defaultable"
                )
    if native_graph_fused_wavg_lora is None:
        focus.append("native_graph fused W/A/G/V-gate LoRA integration row pending")
    else:
        ngwv_speedup = native_graph_fused_wavg_lora.get("speedup")
        greedy_match = native_graph_fused_wavg_lora.get("greedy_match")
        greedy_total = native_graph_fused_wavg_lora.get("greedy_total")
        wavg_diff = native_graph_fused_wavg_lora.get("max_abs_diff_first_step")
        if ngwv_speedup is not None and float(ngwv_speedup) >= 1.0:
            focus.append(
                f"native_graph fused W/A/G/V-gate LoRA latest row passes greedy {greedy_match}/{greedy_total} "
                f"with speedup={float(ngwv_speedup):.2f}x max_abs_diff={wavg_diff}; run batch sweep before default decision"
            )
        elif ngwv_speedup is not None:
            focus.append(
                f"native_graph fused W/A/G/V-gate LoRA integration passes greedy {greedy_match}/{greedy_total} "
                f"but speedup={float(ngwv_speedup):.2f}x; keep opt-in and fuse deeper with R/K/V projection"
            )
    if native_graph_fused_wavg_lora_sweep:
        sweep_speeds = [num(r, "speedup") for r in native_graph_fused_wavg_lora_sweep]
        sweep_speeds = [v for v in sweep_speeds if v is not None]
        sweep_batches = [r.get("batch_size") for r in native_graph_fused_wavg_lora_sweep]
        greedy_ok = all(
            r.get("greedy_match") is not None
            and r.get("greedy_total") is not None
            and int(r.get("greedy_match")) == int(r.get("greedy_total"))
            for r in native_graph_fused_wavg_lora_sweep
        )
        if sweep_speeds and greedy_ok:
            min_speed = min(sweep_speeds)
            if min_speed >= 1.0:
                focus.append(
                    f"native_graph fused W/A/G/V-gate LoRA batch matrix covers bsz={sweep_batches} "
                    f"with min_speedup={min_speed:.2f}x and greedy exact"
                )
            else:
                focus.append(
                    f"native_graph fused W/A/G/V-gate LoRA batch matrix covers bsz={sweep_batches} "
                    f"with greedy exact but min_speedup={min_speed:.2f}x; keep opt-in, not defaultable"
                )
    if native_graph_fused_projection is None:
        focus.append("native_graph fused projection integration row pending")
    else:
        ngp_speedup = native_graph_fused_projection.get("speedup")
        greedy_match = native_graph_fused_projection.get("greedy_match")
        greedy_total = native_graph_fused_projection.get("greedy_total")
        proj_diff = native_graph_fused_projection.get("max_abs_diff_first_step")
        if ngp_speedup is not None and float(ngp_speedup) >= 1.0:
            focus.append(
                f"native_graph fused projection integration passes greedy {greedy_match}/{greedy_total} "
                f"with speedup={float(ngp_speedup):.2f}x max_abs_diff={proj_diff}; test batch/device matrix before default"
            )
        elif ngp_speedup is not None:
            focus.append(
                f"native_graph fused projection integration passes greedy {greedy_match}/{greedy_total} "
                f"but speedup={float(ngp_speedup):.2f}x; keep opt-in and tune R/K/V+LoRA kernels"
            )
    if native_graph_fused_projection_sweep:
        sweep_speeds = [num(r, "speedup") for r in native_graph_fused_projection_sweep]
        sweep_speeds = [v for v in sweep_speeds if v is not None]
        sweep_batches = [r.get("batch_size") for r in native_graph_fused_projection_sweep]
        greedy_ok = all(
            r.get("greedy_match") is not None
            and r.get("greedy_total") is not None
            and int(r.get("greedy_match")) == int(r.get("greedy_total"))
            for r in native_graph_fused_projection_sweep
        )
        if sweep_speeds and greedy_ok:
            min_speed = min(sweep_speeds)
            variant = native_graph_fused_projection_sweep[-1].get("projection_variant")
            if min_speed >= 1.0:
                focus.append(
                    f"native_graph fused projection ({variant}) batch matrix covers bsz={sweep_batches} "
                    f"with min_speedup={min_speed:.2f}x and greedy exact"
                )
            else:
                focus.append(
                    f"native_graph fused projection ({variant}) batch matrix covers bsz={sweep_batches} "
                    f"with greedy exact but min_speedup={min_speed:.2f}x; keep opt-in, not defaultable"
                )
    if native_quant_gemv_proto is None:
        focus.append("native_quant_gemv_proto row pending")
    else:
        q_speedup = native_quant_gemv_proto.get("avg_speedup")
        q_footprint = native_quant_gemv_proto.get("sample_footprint_ratio")
        q_cos = native_quant_gemv_proto.get("min_cosine")
        if q_speedup is not None and float(q_speedup) < 1.0:
            focus.append(
                f"native int8 dequant-GEMV prototype footprint={q_footprint}x fp16 "
                f"but speed={float(q_speedup):.2f}x; optimize packed kernel before replacing bnb"
            )
        elif q_speedup is not None:
            focus.append(
                f"native int8 dequant-GEMV prototype speed={float(q_speedup):.2f}x "
                f"footprint={q_footprint}x fp16 min_cosine={q_cos}; validate model-level W8 path"
            )
    if native_quant_w4_gemv_proto is None:
        focus.append("native_quant_w4_gemv_proto row pending")
    else:
        q4_speedup = native_quant_w4_gemv_proto.get("avg_speedup")
        q4_footprint = native_quant_w4_gemv_proto.get("sample_footprint_ratio")
        q4_cos = native_quant_w4_gemv_proto.get("min_cosine")
        if q4_speedup is not None and float(q4_speedup) < 1.0:
            focus.append(
                f"native int4 dequant-GEMV prototype footprint={q4_footprint}x fp16 "
                f"but speed={float(q4_speedup):.2f}x; optimize nibble unpack/reduction before replacing bnb"
            )
        elif q4_speedup is not None:
            focus.append(
                f"native int4 dequant-GEMV prototype speed={float(q4_speedup):.2f}x "
                f"footprint={q4_footprint}x fp16 min_cosine={q4_cos}; validate model-level W4 path"
            )
    if native_quant_rkv_proto is None:
        focus.append("native_quant_rkv_proto row pending")
    else:
        fused_vs_fp16 = native_quant_rkv_proto.get("fused_speedup_vs_fp16")
        fused_vs_separate = native_quant_rkv_proto.get("fused_speedup_vs_separate_int8")
        footprint = native_quant_rkv_proto.get("sample_footprint_ratio")
        if fused_vs_separate is not None and float(fused_vs_separate) >= 1.0:
            focus.append(
                f"native int8 fused R/K/V quant projection improves separate W8 GEMVs by "
                f"{float(fused_vs_separate):.2f}x, footprint={footprint}x fp16, "
                f"vs fp16={fused_vs_fp16}x; continue fusing projection groups"
            )
        elif fused_vs_separate is not None:
            focus.append(
                f"native int8 fused R/K/V quant projection is {float(fused_vs_separate):.2f}x "
                "of separate W8 GEMVs; optimize before integrating"
            )
    if native_quant_w4_rkv_proto is None:
        focus.append("native_quant_w4_rkv_proto row pending")
    else:
        fused_vs_fp16 = native_quant_w4_rkv_proto.get("fused_speedup_vs_fp16")
        fused_vs_separate = native_quant_w4_rkv_proto.get("fused_speedup_vs_separate_int4")
        footprint = native_quant_w4_rkv_proto.get("sample_footprint_ratio")
        cosine = native_quant_w4_rkv_proto.get("min_cosine_fp16_vs_fused")
        if fused_vs_separate is not None and float(fused_vs_separate) >= 1.0:
            focus.append(
                f"native int4 fused R/K/V quant projection improves separate W4 GEMVs by "
                f"{float(fused_vs_separate):.2f}x, footprint={footprint}x fp16, "
                f"vs fp16={fused_vs_fp16}x min_cosine={cosine}; continue fusing projection groups"
            )
        elif fused_vs_separate is not None:
            focus.append(
                f"native int4 fused R/K/V quant projection is {float(fused_vs_separate):.2f}x "
                "of separate W4 GEMVs; optimize before integrating"
            )
    if native_quant_rkv_sweep:
        for row in native_quant_rkv_sweep:
            best = row.get("best_by_latency") or row.get("best_by_speedup_vs_fp16") or {}
            quant_label = "W8" if str(row.get("quantization", "")).startswith("int8") else "W4"
            speed = best.get("fused_speedup_vs_fp16")
            sep_speed = best.get("fused_speedup_vs_separate")
            bm = best.get("block_m")
            bk = best.get("block_k")
            fused_ms = best.get("avg_fused_quant_ms")
            if speed is not None and float(speed) >= 1.0:
                focus.append(
                    f"native {quant_label} R/K/V sweep best block_m={bm} block_k={bk} reaches "
                    f"{float(speed):.2f}x fp16 ({fused_ms} ms), separate={sep_speed}x; validate model-level path"
                )
            elif speed is not None:
                focus.append(
                    f"native {quant_label} R/K/V sweep best block_m={bm} block_k={bk} is "
                    f"{float(speed):.2f}x fp16 ({fused_ms} ms), separate={sep_speed}x; kernel still below fp16"
                )
    if albatross_decode_min is None:
        focus.append("fused backend target tracking needs Albatross decode ratios")
    elif albatross_decode_min < 0.55:
        focus.append(
            f"fused backend P1 pending: decode min {albatross_decode_min:.2f}x Albatross; "
            "continue fused fp16 projection/LoRA and recurrent-output kernels"
        )
    if albatross_prefill_min is not None and albatross_prefill_min < 0.60:
        focus.append(
            f"fused backend prefill P1 pending: prefill min {albatross_prefill_min:.2f}x Albatross; "
            "plan scan/chunk fused prefill path"
        )
    training_by_backend = {r.get("trainer_backend"): r for r in training_latest if r.get("status") == "pass"}
    missing_training = sorted({"trainer", "trl_sft", "trl_dpo", "trl_grpo"} - set(training_by_backend))
    if missing_training:
        focus.append(f"training smoke telemetry incomplete: missing {missing_training}")
    else:
        min_delta = min(float(r.get("max_trainable_delta") or 0.0) for r in training_by_backend.values())
        focus.append(
            "HF training telemetry passes for Trainer/SFT/DPO/GRPO "
            f"with min trainable delta {min_delta:.3g}"
        )
    for row in training_latest:
        if row.get("status") == "pass" and float(row.get("max_trainable_delta") or 0.0) <= 0.0:
            focus.append(f"training smoke did not update trainable params: {row.get('trainer_backend')}")
    deepspeed_by_stage = {int(r.get("zero_stage")): r for r in deepspeed_latest if r.get("zero_stage") is not None}
    missing_zero = sorted({2, 3} - set(deepspeed_by_stage))
    if missing_zero:
        focus.append(f"DeepSpeed ZeRO smoke telemetry incomplete: missing stages {missing_zero}")
    else:
        passed_zero = [stage for stage, row in sorted(deepspeed_by_stage.items()) if row.get("status") == "pass"]
        skipped_zero = [stage for stage, row in sorted(deepspeed_by_stage.items()) if row.get("status") == "skip"]
        if passed_zero:
            focus.append(f"DeepSpeed ZeRO smoke passes for stages {passed_zero}")
        if skipped_zero:
            reasons = {
                stage: deepspeed_by_stage[stage].get("reason")
                for stage in skipped_zero
            }
            focus.append(f"DeepSpeed ZeRO smoke skipped for stages {skipped_zero}: {reasons}")
        for stage, row in sorted(deepspeed_by_stage.items()):
            if row.get("status") == "pass" and float(row.get("max_trainable_delta") or 0.0) <= 0.0:
                focus.append(f"DeepSpeed ZeRO-{stage} smoke did not update trainable params")

    if not albatross_latest:
        focus.append("Albatross A/B rows pending")
    elif albatross_decode_comparison:
        ratios = [
            float(row["hf_vs_albatross_ratio"])
            for row in albatross_decode_comparison
            if row.get("hf_vs_albatross_ratio") is not None
        ]
        sizes = "/".join(str(row["batch_size"]) for row in albatross_decode_comparison)
        if ratios:
            focus.append(
                f"Albatross A/B decode comparison present for bsz={sizes}; "
                f"HF/Albatross ratio min={min(ratios):.2f} max={max(ratios):.2f}"
            )
        else:
            focus.append(f"Albatross A/B decode rows present for bsz={sizes}; ratio pending")
        prefill_ratios = [
            float(row["hf_vs_albatross_ratio"])
            for row in albatross_prefill_comparison
            if row.get("hf_vs_albatross_ratio") is not None
        ]
        if prefill_ratios:
            cases = "/".join(
                f"{row['batch_size']}x{row['tokens_per_sequence']}"
                for row in albatross_prefill_comparison
            )
            focus.append(
                f"Albatross A/B prefill comparison present for cases={cases}; "
                f"HF/Albatross ratio min={min(prefill_ratios):.2f} max={max(prefill_ratios):.2f}"
            )
    else:
        focus.append("Albatross rows present; add matching HF decode/prefill cases for ratios")

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
    for row in quant_best_variants:
        best_speed = row.get("best_speed") or {}
        q_ratio = row.get("decode_ratio_vs_fp16")
        footprint_ratio = row.get("footprint_ratio_vs_fp16")
        if q_ratio is not None:
            focus.append(
                f"best {row['quantization']} quant variant "
                f"policy={best_speed.get('quant_skip_policy') or 'memory'} "
                f"decode={q_ratio:.2f}x fp16 footprint={footprint_ratio if footprint_ratio is not None else 'n/a'}x"
            )
    for row in fused_quant_targets:
        if row.get("decode_status") == "GAP":
            focus.append(
                f"native fused {row['quantization']} pending: best decode "
                f"{row.get('decode_ratio_vs_fp16')}x fp16; replace generic bnb with packed dequant-GEMV"
            )
    quant_model_pass = defaultdict(set)
    for row in quant_model_latest:
        if row.get("status") == "pass":
            quant_model_pass[model_label(row)].add(row.get("quantization"))
    for label in sorted(k for k in quant_model_pass if k not in {"unknown", "0.1b"}):
        modes = sorted(str(v) for v in quant_model_pass[label] if v)
        if modes:
            focus.append(f"{label} quantization sweep rows pass for {','.join(modes)}")
    if device_map_smoke is None:
        focus.append("HF device_map multi-GPU generate smoke row pending")
    elif device_map_smoke.get("status") == "pass":
        focus.append(
            "HF device_map generate passes on "
            f"{device_map_smoke.get('device_count')} CUDA devices with split_layer={device_map_smoke.get('split_layer')}"
        )
    else:
        focus.append(f"HF device_map generate smoke did not pass: {device_map_smoke.get('status')}")

    if speculative_decode is None:
        focus.append("real-draft speculative decode benchmark row pending")
    elif speculative_decode.get("status") == "pass":
        focus.append(
            "speculative decode matches target greedy with "
            f"draft={speculative_decode.get('draft_model_name')} "
            f"acceptance={speculative_decode.get('stats_acceptance_rate')} "
            f"speedup={speculative_decode.get('speedup_vs_target_generate')} "
            f"resync_saved_tokens={speculative_decode.get('stats_resync_saved_tokens')}"
        )
    else:
        focus.append(f"speculative decode benchmark did not pass: {speculative_decode.get('status')}")

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
    if native_model_smoke is None:
        focus.append("experimental native_model smoke telemetry pending")
    elif native_model_smoke.get("status") == "pass":
        focus.append(
            "experimental native_model smoke passes with "
            f"forward_cos={native_model_smoke.get('forward_min_cos')} "
            f"generate={native_model_smoke.get('generate_token_match')}/{native_model_smoke.get('generate_token_total')} "
            f"cache={native_model_smoke.get('incremental_cache')} "
            f"backend={native_model_smoke.get('native_decode_backend')}"
        )
    else:
        focus.append(f"experimental native_model smoke did not pass: {native_model_smoke.get('status')}")
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
        "batch_sweep": [compact(r, ["_lineno", "batch_size", "decode_api", "fast_token_backend", "fast_token_backend_effective", "native_graph_fused_recurrent", "native_graph_fused_recurrent_output", "native_graph_fused_output", "native_graph_fused_output_project", "native_graph_fused_wag_lora", "native_graph_fused_wavg_lora", "native_graph_fused_projection", "decode_tokps_total", "decode_tokps_per_seq", "decode_ms_per_step", "peak_vram_mb"]) for r in batch_latest],
        "batch_sweep_default_native_graph": [compact(r, ["_lineno", "batch_size", "decode_api", "fast_token_backend", "fast_token_backend_effective", "native_graph_fused_recurrent", "native_graph_fused_recurrent_output", "native_graph_fused_output", "native_graph_fused_output_project", "native_graph_fused_wag_lora", "native_graph_fused_wavg_lora", "native_graph_fused_projection", "decode_tokps_total", "decode_tokps_per_seq", "decode_ms_per_step", "peak_vram_mb"]) for r in batch_default_latest],
        "batch_sweep_experimental_native_graph": [
            {
                **(compact(r, ["_lineno", "batch_size", "decode_api", "fast_token_backend", "fast_token_backend_effective", "native_graph_fused_recurrent", "native_graph_fused_recurrent_output", "native_graph_fused_output", "native_graph_fused_output_project", "native_graph_fused_wag_lora", "native_graph_fused_wavg_lora", "native_graph_fused_projection", "decode_tokps_total", "decode_tokps_per_seq", "decode_ms_per_step", "peak_vram_mb"]) or {}),
                "native_graph_flag_signature": native_graph_flag_signature(r),
            }
            for r in batch_experimental_latest
        ],
        "dynamic_batch": [
            compact(
                r,
                [
                    "_lineno",
                    "decode_api",
                    "fast_token_backend",
                    "fast_token_backend_effective",
                    "initial_batch_size",
                    "final_batch_size",
                    "final_cache_batch_size",
                    "cache_select_api",
                    "total_decode_tokens",
                    "reorder_count",
                    "drop_count",
                    "decode_tokps_total",
                    "decode_ms_per_token",
                    "cache_select_batch_calls",
                    "cache_native_graph_bound_selects",
                    "cache_seen_tokens",
                    "native_graph_cache_requests",
                    "native_graph_cache_hits",
                    "native_graph_cache_misses",
                    "native_graph_cache_hit_rate",
                    "native_graph_cache_batch_sizes",
                    "native_graph_copy_from_cache_calls",
                    "native_graph_copy_from_cache_fast_skips",
                    "native_graph_copy_from_cache_fast_skip_rate",
                    "native_graph_bind_cache_calls",
                    "native_graph_bind_cache_fast_skips",
                    "native_graph_bind_cache_fast_skip_rate",
                    "peak_vram_mb",
                ],
            )
            for r in dynamic_latest
        ],
        "chunked_prefill": [compact(r, ["_lineno", "prefill_mode", "batch_size", "prompt_tokens", "chunk_size", "prefill_tokps_total", "speed_ratio_vs_full", "peak_vram_mb", "peak_vram_ratio_vs_full", "max_abs_diff", "decode_max_abs_diff", "seq_length_match"]) for r in chunked_latest],
        "native_prefill_scan": [
            compact(r, ["_lineno", "status", "dtype", "device", "code_source", "native_jit_module", "effective_model_path", "batch_size", "prompt_tokens", "tokens_total", "fused_scan_requested", "scan_block_m", "scan_num_warps", "prefill_fused_scan_output_requested", "prefill_fused_scan_output_effective", "prefill_fused_clampw_scan_requested", "prefill_fused_clampw_scan_effective", "prefill_dplr_scan_requested", "prefill_dplr_scan_effective", "prefill_dplr_chunk_size", "prefill_fused_shift_mix_requested", "prefill_fused_shift_mix_effective", "prefill_fused_state_prep_requested", "prefill_fused_state_prep_effective", "prefill_fused_state_scan_requested", "prefill_fused_state_scan_effective", "prefill_fused_output_requested", "prefill_fused_output_effective", "prefill_fused_wavg_lora_requested", "prefill_fused_wavg_lora_effective", "prefill_fused_wavg_lora_max_m", "fast_token_backend_after_native_prefill", "hf_prefill_ms", "native_prefill_ms", "native_vs_hf_speedup", "hf_prefill_tokps_total", "native_prefill_tokps_total", "max_abs_diff", "min_cosine", "greedy_match", "decode_after_prefill_max_abs_diff", "decode_after_prefill_greedy_match", "peak_vram_mb"])
            for r in native_prefill_scan
        ],
        "native_prefill_breakdown": [
            compact(r, ["_lineno", "status", "dtype", "device", "model_size_label", "batch_size", "prompt_tokens", "tokens_total", "fused_scan_requested", "scan_block_m", "scan_num_warps", "fine_attention_breakdown", "layer_breakdown", "prefill_fused_scan_output_requested", "prefill_fused_scan_output_effective", "prefill_fused_clampw_scan_requested", "prefill_fused_clampw_scan_effective", "prefill_dplr_scan_requested", "prefill_dplr_scan_effective", "prefill_dplr_chunk_size", "prefill_fused_shift_mix_requested", "prefill_fused_shift_mix_effective", "prefill_fused_state_prep_requested", "prefill_fused_state_prep_effective", "prefill_fused_state_scan_requested", "prefill_fused_state_scan_effective", "prefill_fused_output_requested", "prefill_fused_output_effective", "prefill_fused_wavg_lora_requested", "prefill_fused_wavg_lora_effective", "prefill_fused_wavg_lora_max_m", "profiled_total_gpu_ms", "component_sum_ms", "profiled_tokps_total", "component_ms", "component_share", "top_components", "layer_total_ms", "top_layers_by_total", "layer_top_components", "max_abs_diff_vs_native_prefill", "greedy_match_vs_native_prefill", "peak_vram_mb"])
            for r in native_prefill_breakdown
        ],
        "decode_micro": compact(micro, ["_lineno", "fast_decode_api_name", "fast_token_layout", "fast_token_backend", "fast_token_backend_effective", "hf_forward_fixed", "hf_forward_greedy", "hf_forward_auto_fixed", "hf_forward_auto_greedy", "hf_forward_auto_backend", "fast_decode_fixed", "fast_decode_greedy", "norm_lm_head", "lm_head", "argmax", "empty_loop", "peak_vram_mb"]),
        "forward_fast_path": compact(forward_fast_path, ["_lineno", "fast_token_backend", "fast_token_layout", "reference_forward", "hf_forward_fast", "direct_fast_token", "hf_forward_fast_backend", "direct_fast_token_backend", "max_abs_diff_auto_vs_reference", "max_abs_diff_direct_vs_reference", "peak_vram_mb"]),
        "generate_fast_path": compact(generate_fast_path, ["_lineno", "fast_token_backend", "fast_token_backend_effective", "batch_size", "reference_generate", "hf_generate_fast", "speedup_vs_reference", "generated_equal", "generated_tokens_matched", "generated_tokens_total", "prompt_tokens", "max_new_tokens", "peak_vram_mb"]),
        "fast_token_warmup": compact(fast_token_warmup, ["_lineno", "fast_token_backend", "batch_sizes", "effective_backend_by_batch", "native_graph_cache_batch_sizes", "native_graph_cache_size_limit", "cleared_before", "warmup_s", "peak_vram_mb"]),
        "ttft_tpot": [
            compact(r, ["_lineno", "metric", "dtype", "device", "fast_forward_env", "fast_prefill_env", "native_prefill_fused_scan_env", "fast_prefill_backend_effective", "fast_token_backend_effective", "prompt_tokens", "decode_tokens", "batch_size", "p50_ms", "p99_ms", "mean_ms", "prefill_tokps_p50", "decode_tokps_p50", "generate_tokps", "peak_vram_mb", "status"])
            for r in ttft_tpot
        ],
        "native_graph_replay_overhead": [
            compact(r, ["_lineno", "fast_token_backend", "fast_token_backend_effective", "native_graph_fused_recurrent", "native_graph_fused_recurrent_output", "native_graph_fused_output", "native_graph_fused_output_project", "native_graph_fused_wag_lora", "native_graph_fused_wavg_lora", "native_graph_fused_projection", "batch_size", "prompt_tokens", "steps", "fixed_token", "max_abs_diff_runner_vs_api", "copy_from_cache_ms", "token_copy_ms", "graph_replay_ms", "bind_cache_ms", "argmax_ms", "copy_from_cache_calls", "copy_from_cache_fast_skips", "copy_from_cache_fast_skip_rate", "bind_cache_calls", "bind_cache_fast_skips", "bind_cache_fast_skip_rate", "manual_wall_ms_per_token", "api_ms_per_token", "manual_decode_tokps_total", "api_decode_tokps_total", "copy_share_of_manual_wall", "native_graph_cache_requests", "native_graph_cache_hits", "native_graph_cache_misses", "native_graph_cache_evictions", "native_graph_cache_hit_rate", "native_graph_cache_batch_sizes", "peak_vram_mb"])
            for r in native_graph_overhead
        ],
        "decode_components": compact(components, ["_lineno", "decode_api", "batch_size", "wall_ms_per_token", "decode_tokps_wall", "top_components", "top_layers", "peak_vram_mb"]),
        "projection_lora": compact(projection_lora, ["_lineno", "batch_size", "hidden_size", "layers", "avg_timings_ms", "avg_current_linears_lora_sum_ms", "avg_candidate_linears_lora_sum_ms", "avg_candidate_speedup", "sample_matrix_profile_summary", "fused_kernel_plan", "peak_vram_mb"]),
        "fused_projection_proto": compact(fused_projection_proto, ["_lineno", "prototype_backend", "status", "dtype", "device", "batch_size", "hidden_size", "layers", "block_m", "block_k", "steps", "avg_current_ms", "avg_prototype_ms", "avg_speedup", "max_abs_diff", "min_cosine", "layer_rows", "peak_vram_mb"]),
        "albatross_projection_layout_tune": [
            compact(r, ["_lineno", "prototype_backend", "borrowed_from", "status", "dtype", "device", "batch_size", "hidden_size", "layers", "steps", "avg_current_ms", "config_count", "best_config", "top_configs", "peak_vram_mb"])
            for r in albatross_projection_layout
        ],
        "fused_wa_lora_proto": compact(fused_wa_lora_proto, ["_lineno", "prototype_backend", "status", "dtype", "device", "batch_size", "hidden_size", "ranks", "layers", "block_m", "block_r", "block_k", "steps", "avg_current_ms", "avg_prototype_ms", "avg_speedup", "max_abs_diff", "min_cosine", "layer_rows", "peak_vram_mb"]),
        "fused_wag_lora_proto": compact(fused_wag_lora_proto, ["_lineno", "prototype_backend", "status", "dtype", "device", "batch_size", "hidden_size", "ranks", "layers", "block_m", "block_r", "block_k", "steps", "avg_current_ms", "avg_prototype_ms", "avg_speedup", "max_abs_diff", "min_cosine", "layer_rows", "peak_vram_mb"]),
        "fused_wavg_lora_proto": compact(fused_wavg_lora_proto, ["_lineno", "prototype_backend", "status", "dtype", "device", "batch_size", "hidden_size", "ranks", "layers", "block_m", "block_r", "block_k", "steps", "avg_current_ms", "avg_prototype_ms", "avg_speedup", "max_abs_diff", "min_cosine", "layer_rows", "peak_vram_mb"]),
        "fused_rkv_wag_projection_proto": compact(fused_rkv_wag_projection_proto, ["_lineno", "prototype_backend", "status", "dtype", "device", "batch_size", "sequence_length", "tokens_total", "hidden_size", "ranks", "layers", "block_m", "block_r", "block_k", "steps", "avg_current_ms", "avg_prototype_ms", "avg_speedup", "max_abs_diff", "min_cosine", "layer_rows", "peak_vram_mb"]),
        "fused_attn_output_proto": compact(fused_attn_output_proto, ["_lineno", "prototype_backend", "status", "dtype", "device", "batch_size", "hidden_size", "head_dims", "head_v_dims", "layers", "input_scale", "steps", "avg_current_ms", "avg_prototype_ms", "avg_speedup", "max_abs_diff", "output_max_abs_diff", "prep_max_abs_diff", "min_cosine", "output_min_cosine", "prep_min_cosine", "layer_rows", "peak_vram_mb"]),
        "fused_attn_output_project_proto": compact(fused_attn_output_project_proto, ["_lineno", "prototype_backend", "status", "dtype", "device", "batch_size", "hidden_size", "layers", "block_m_values", "input_scale", "steps", "avg_current_ms", "avg_prep_cublas_ms", "avg_prep_cublas_speedup", "best_fused_project", "max_abs_diff", "min_cosine", "layer_rows", "peak_vram_mb"]),
        "fused_ffn_proto": compact(fused_ffn_proto, ["_lineno", "prototype_backend", "status", "dtype", "device", "batch_size", "hidden_size", "intermediate_sizes", "layers", "block_m", "block_k", "steps", "avg_current_ms", "avg_prototype_ms", "avg_speedup", "max_abs_diff", "min_cosine", "layer_rows", "peak_vram_mb"]),
        "fused_shift_mix_proto": compact(fused_shift_mix_proto, ["_lineno", "prototype_backend", "status", "dtype", "device", "batch_size", "sequence_length", "tokens_total", "input_rank", "hidden_size", "layers", "block_size", "steps", "avg_current_ms", "avg_prototype_ms", "avg_speedup", "max_abs_diff", "min_cosine", "layer_rows", "peak_vram_mb"]),
        "fused_recurrent_proto": compact(fused_recurrent_proto, ["_lineno", "prototype_backend", "status", "dtype", "device", "batch_size", "hidden_size", "layers", "block_n", "steps", "avg_current_ms", "avg_prototype_ms", "avg_speedup", "out_max_abs_diff", "state_max_abs_diff", "out_min_cosine", "layer_rows", "peak_vram_mb"]),
        "fused_recurrent_scan_proto": [
            compact(r, ["_lineno", "prototype_backend", "status", "dtype", "device", "batch_size", "tokens", "heads", "head_dim", "block_n", "block_m", "num_warps", "chunk_size", "steps", "native_scan_ms", "native_scan_tokps_total", "fla_chunk_ms", "fla_chunk_tokps_total", "native_vs_fla_speedup", "native_vs_torch_speedup", "native_vs_torch_out_max_abs_diff", "native_vs_torch_state_max_abs_diff", "native_vs_torch_out_min_cosine", "native_vs_fla_out_min_cosine", "peak_vram_mb"])
            for r in fused_recurrent_scan_proto
        ],
        "dplr_prefill_scan_proto": [
            compact(r, ["_lineno", "status", "dtype", "device", "algorithm", "requested_algorithm", "effective_algorithm", "algorithm_family", "is_dense_affine", "detected_algorithms", "chunk_size", "triton_wy_available", "triton_wy_block_m", "triton_dense3_available", "triton_summary_block_m", "triton_prefix_block_m", "triton_apply_block_m", "B", "T", "H", "N", "warmup", "steps", "ms", "tokps", "out_max_abs_diff", "state_max_abs_diff", "out_min_cosine", "fallback_reason", "skip_reason", "error", "peak_vram_mb"])
            for r in dplr_prefill_scan_proto
        ],
        "dplr_chunk_summary_proto": [
            compact(r, ["_lineno", "status", "dtype", "device", "algorithm", "algorithm_family", "chunk_size", "triton_summary_available", "triton_summary_block_m", "B", "T", "H", "N", "warmup", "steps", "summary_shape", "ms", "tokps", "transition_max_abs_diff", "additive_max_abs_diff", "state_max_abs_diff", "skip_reason", "error", "peak_vram_mb"])
            for r in dplr_chunk_summary_proto
        ],
        "dplr_dense3_stage_proto": [
            compact(r, ["_lineno", "status", "dtype", "device", "algorithm", "algorithm_family", "stage", "chunk_size", "triton_summary_available", "triton_summary_block_m", "triton_prefix_block_m", "triton_apply_block_m", "B", "T", "H", "N", "warmup", "steps", "summary_shape", "start_states_shape", "chunk_end_shape", "ms", "tokps", "transition_max_abs_diff", "additive_max_abs_diff", "start_states_max_abs_diff", "prefix_vs_torch_summary_state_max_abs_diff", "chunk_end_max_abs_diff", "out_max_abs_diff", "state_max_abs_diff", "out_min_cosine", "skip_reason", "error", "peak_vram_mb"])
            for r in dplr_dense3_stage_proto
        ],
        "fused_recurrent_output_proto": compact(fused_recurrent_output_proto, ["_lineno", "prototype_backend", "status", "dtype", "device", "attn_mode", "fuse_norm", "batch_size", "hidden_size", "layers", "block_n", "input_scale", "steps", "avg_current_ms", "avg_split_fused_ms", "avg_fused_ms", "avg_speedup_vs_current", "avg_speedup_vs_split", "out_max_abs_diff", "state_max_abs_diff", "split_out_max_abs_diff", "split_state_max_abs_diff", "out_min_cosine", "split_out_min_cosine", "layer_rows", "peak_vram_mb"]),
        "native_graph_fused_recurrent": compact(native_graph_fused_recurrent, ["_lineno", "status", "dtype", "device", "batch_size", "prompt_tokens", "steps", "fixed_token", "baseline_effective_backend", "fused_effective_backend", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "peak_vram_mb"]),
        "native_graph_fused_recurrent_output": compact(native_graph_fused_recurrent_output, ["_lineno", "status", "dtype", "device", "attn_mode", "fuse_norm", "fast_cache", "batch_size", "prompt_tokens", "steps", "fixed_token", "fused_output_enabled", "baseline_effective_backend", "fused_effective_backend", "baseline_fused_recurrent_output", "fused_recurrent_output", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"]),
        "native_graph_fused_output": compact(native_graph_fused_output, ["_lineno", "status", "dtype", "device", "batch_size", "prompt_tokens", "steps", "fixed_token", "fused_recurrent_enabled", "baseline_effective_backend", "fused_effective_backend", "baseline_fused_output", "fused_output", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"]),
        "native_graph_fused_output_project": compact(native_graph_fused_output_project, ["_lineno", "status", "dtype", "device", "batch_size", "prompt_tokens", "steps", "fixed_token", "fused_recurrent_enabled", "fused_output_enabled", "fused_projection_enabled", "output_project_block_m", "baseline_effective_backend", "fused_effective_backend", "baseline_fused_output_project", "fused_output_project", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"]),
        "native_graph_fused_wag_lora": compact(native_graph_fused_wag_lora, ["_lineno", "status", "dtype", "device", "batch_size", "prompt_tokens", "steps", "fixed_token", "fused_recurrent_enabled", "fused_output_enabled", "fused_output_project_enabled", "block_m", "block_r", "block_k", "baseline_effective_backend", "fused_effective_backend", "baseline_fused_wag_lora", "fused_wag_lora", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"]),
        "native_graph_fused_wavg_lora": compact(native_graph_fused_wavg_lora, ["_lineno", "status", "dtype", "device", "batch_size", "prompt_tokens", "steps", "fixed_token", "fused_recurrent_enabled", "fused_recurrent_output_enabled", "fused_output_enabled", "fused_output_project_enabled", "block_m", "block_r", "block_k", "baseline_effective_backend", "fused_effective_backend", "baseline_fused_wavg_lora", "fused_wavg_lora", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"]),
        "native_graph_fused_projection": compact(native_graph_fused_projection, ["_lineno", "status", "dtype", "device", "batch_size", "prompt_tokens", "steps", "fixed_token", "projection_variant", "fused_recurrent_enabled", "fused_output_enabled", "baseline_effective_backend", "fused_effective_backend", "baseline_fused_projection", "fused_projection", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"]),
        "native_graph_fused_output_sweep": [
            compact(r, ["_lineno", "status", "dtype", "device", "batch_size", "prompt_tokens", "steps", "fixed_token", "fused_recurrent_enabled", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"])
            for r in native_graph_fused_output_sweep
        ],
        "native_graph_fused_recurrent_output_sweep": [
            compact(r, ["_lineno", "status", "dtype", "device", "attn_mode", "fuse_norm", "fast_cache", "batch_size", "prompt_tokens", "steps", "fixed_token", "fused_output_enabled", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"])
            for r in native_graph_fused_recurrent_output_sweep
        ],
        "native_graph_fused_output_project_sweep": [
            compact(r, ["_lineno", "status", "dtype", "device", "batch_size", "prompt_tokens", "steps", "fixed_token", "fused_recurrent_enabled", "fused_output_enabled", "fused_projection_enabled", "output_project_block_m", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"])
            for r in native_graph_fused_output_project_sweep
        ],
        "native_graph_fused_wag_lora_sweep": [
            compact(r, ["_lineno", "status", "dtype", "device", "batch_size", "prompt_tokens", "steps", "fixed_token", "fused_recurrent_enabled", "fused_output_enabled", "fused_output_project_enabled", "block_m", "block_r", "block_k", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"])
            for r in native_graph_fused_wag_lora_sweep
        ],
        "native_graph_fused_wavg_lora_sweep": [
            compact(r, ["_lineno", "status", "dtype", "device", "batch_size", "prompt_tokens", "steps", "fixed_token", "fused_recurrent_enabled", "fused_recurrent_output_enabled", "fused_output_enabled", "fused_output_project_enabled", "block_m", "block_r", "block_k", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"])
            for r in native_graph_fused_wavg_lora_sweep
        ],
        "native_graph_fused_projection_sweep": [
            compact(r, ["_lineno", "status", "dtype", "device", "batch_size", "prompt_tokens", "steps", "fixed_token", "projection_variant", "fused_recurrent_enabled", "fused_output_enabled", "baseline_ms_per_step", "fused_ms_per_step", "speedup", "baseline_tokps_total", "fused_tokps_total", "max_abs_diff_first_step", "min_cosine_first_step", "greedy_match", "greedy_total", "baseline_cache_stats", "fused_cache_stats", "peak_vram_mb"])
            for r in native_graph_fused_projection_sweep
        ],
        "native_quant_gemv_proto": compact(native_quant_gemv_proto, ["_lineno", "prototype_backend", "status", "quantization", "dtype", "device", "batch_size", "layers", "modules", "block_m", "block_k", "steps", "avg_current_ms", "avg_prototype_ms", "avg_speedup", "max_abs_diff", "mean_abs_diff_max", "min_cosine", "sample_fp16_weight_mb", "sample_int8_weight_mb", "sample_footprint_ratio", "layer_rows", "peak_vram_mb"]),
        "native_quant_w4_gemv_proto": compact(native_quant_w4_gemv_proto, ["_lineno", "prototype_backend", "status", "quantization", "dtype", "device", "batch_size", "layers", "modules", "block_m", "block_k", "steps", "avg_current_ms", "avg_prototype_ms", "avg_speedup", "max_abs_diff", "mean_abs_diff_max", "min_cosine", "sample_fp16_weight_mb", "sample_int4_weight_mb", "sample_footprint_ratio", "layer_rows", "peak_vram_mb"]),
        "native_quant_rkv_proto": compact(native_quant_rkv_proto, ["_lineno", "prototype_backend", "status", "quantization", "dtype", "device", "batch_size", "hidden_size", "layers", "block_m", "block_k", "steps", "avg_fp16_current_ms", "avg_separate_int8_ms", "avg_fused_int8_ms", "fused_speedup_vs_fp16", "fused_speedup_vs_separate_int8", "separate_speedup_vs_fp16", "max_abs_diff_fp16_vs_fused", "max_abs_diff_separate_vs_fused", "min_cosine_fp16_vs_fused", "min_cosine_separate_vs_fused", "sample_fp16_weight_mb", "sample_int8_weight_mb", "sample_footprint_ratio", "layer_rows", "peak_vram_mb"]),
        "native_quant_w4_rkv_proto": compact(native_quant_w4_rkv_proto, ["_lineno", "prototype_backend", "status", "quantization", "dtype", "device", "batch_size", "hidden_size", "layers", "block_m", "block_k", "steps", "avg_fp16_current_ms", "avg_separate_int4_ms", "avg_fused_int4_ms", "fused_speedup_vs_fp16", "fused_speedup_vs_separate_int4", "separate_speedup_vs_fp16", "max_abs_diff_fp16_vs_fused", "max_abs_diff_separate_vs_fused", "min_cosine_fp16_vs_fused", "min_cosine_separate_vs_fused", "sample_fp16_weight_mb", "sample_int4_weight_mb", "sample_footprint_ratio", "layer_rows", "peak_vram_mb"]),
        "native_quant_rkv_sweep": [
            compact(r, ["_lineno", "prototype_backend", "status", "quantization", "dtype", "device", "batch_size", "hidden_size", "layers", "block_m_values", "block_k_values", "block_k_unit", "warmup", "steps", "avg_fp16_baseline_ms", "best_by_speedup_vs_fp16", "best_by_latency", "sample_fp16_weight_mb", "sample_quant_weight_mb", "sample_footprint_ratio", "peak_vram_mb"])
            for r in native_quant_rkv_sweep
        ],
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
        "device_map_smoke": compact(device_map_smoke, ["_lineno", "status", "dtype", "device", "device_count", "device_map_kind", "split_layer", "num_hidden_layers", "hf_device_map_devices", "multi_cuda_device_map", "fast_forward_env", "last_fast_token_backend", "prompt_tokens", "max_new_tokens", "generated_tokens", "generated_tail", "reference_tail", "generated_equal_reference", "logits_shape", "logits_device", "logits_finite", "load_s", "generate_s", "generate_tokps", "peak_vram_mb_by_device"]),
        "speculative_decode": compact(speculative_decode, ["_lineno", "status", "dtype", "device", "target_model_name", "draft_model_name", "same_model", "prompt_tokens", "max_new_tokens", "draft_tokens", "generated_tokens", "generated_equal", "target_tail", "speculative_tail", "target_generate_s", "speculative_s", "target_generate_tokps", "speculative_tokps", "speedup_vs_target_generate", "stats_generated_tokens", "stats_proposed_tokens", "stats_accepted_tokens", "stats_corrected_tokens", "stats_resyncs", "stats_resync_tokens", "stats_full_resync_tokens", "stats_resync_saved_tokens", "stats_target_forward_calls", "stats_draft_forward_calls", "stats_acceptance_rate", "peak_vram_mb"]),
        "native_model_smoke": compact(native_model_smoke, ["_lineno", "status", "dtype", "device", "model_name", "prompt_count", "forward_min_cos", "forward_max_abs", "forward_argmax_match", "forward_argmax_total", "batch_size", "batch_prompt_tokens", "batch_forward_min_cos", "batch_forward_max_abs", "batch_forward_argmax_match", "batch_forward_argmax_total", "batch_decode_max_abs", "batch_decode_argmax_match", "batch_decode_argmax_total", "batch_cache_shape_ok", "native_decode_backend", "generate_tokens", "generate_token_match", "generate_token_total", "incremental_cache"]),
        "training_smoke": [
            compact(
                r,
                [
                    "_lineno",
                    "trainer_backend",
                    "status",
                    "train_dtype",
                    "device",
                    "attn_mode",
                    "batch_size",
                    "gradient_accumulation_steps",
                    "effective_batch_size",
                    "max_steps",
                    "train_loss",
                    "train_runtime_s",
                    "train_samples_per_second",
                    "train_steps_per_second",
                    "max_trainable_delta",
                ],
            )
            for r in training_latest
        ],
        "deepspeed_training_smoke": [
            compact(
                r,
                [
                    "_lineno",
                    "trainer_backend",
                    "zero_stage",
                    "status",
                    "reason",
                    "train_dtype",
                    "device",
                    "cuda_device_count",
                    "attn_mode",
                    "batch_size",
                    "gradient_accumulation_steps",
                    "effective_batch_size",
                    "max_steps",
                    "deepspeed_config",
                    "train_loss",
                    "train_runtime_s",
                    "train_samples_per_second",
                    "train_steps_per_second",
                    "max_trainable_delta",
                ],
            )
            for r in deepspeed_latest
        ],
        "albatross_speed": [
            compact(
                r,
                [
                    "_lineno",
                    "engine",
                    "engine_config",
                    "status",
                    "dtype",
                    "device",
                    "model_size_label",
                    "checkpoint_sha256",
                    "batch_size",
                    "tokens_per_sequence",
                    "tokens_total",
                    "iters",
                    "latency_p10_ms",
                    "latency_p50_ms",
                    "latency_p90_ms",
                    "tokps_p50",
                    "ms_per_token_p50",
                    "peak_vram_mb",
                ],
            )
            for r in albatross_latest
        ],
        "albatross_decode_comparison": albatross_decode_comparison,
        "albatross_prefill_comparison": albatross_prefill_comparison,
        "fused_backend_targets": fused_backend_targets,
        "quantization": [
            compact(
                r,
                [
                    "_lineno",
                    "model_size_label",
                    "model_name",
                    "quantization",
                    "status",
                    "prefill_tokps",
                    "decode_mode",
                    "selected_decode_path",
                    "decode_tokps",
                    "reference_decode_tokps",
                    "fast_decode_tokps",
                    "fast_decode_speedup",
                    "fast_forward_backend",
                    "fast_forward_max_abs_diff",
                    "fast_forward_same_next_token",
                    "quant_skip_policy",
                    "quant_skip_modules",
                    "module_counts",
                    "decode_ms_per_tok",
                    "model_footprint_mb",
                    "peak_vram_mb",
                    "error",
                ],
            )
            for r in quant_latest
        ],
        "quantization_variants": [
            compact(
                r,
                [
                    "_lineno",
                    "model_size_label",
                    "model_name",
                    "quantization",
                    "status",
                    "quant_skip_policy",
                    "decode_tokps",
                    "reference_decode_tokps",
                    "fast_decode_tokps",
                    "model_footprint_mb",
                    "peak_vram_mb",
                    "module_counts",
                ],
            )
            for r in quant_variant_latest
        ],
        "quantization_model_sweep": [
            compact(
                r,
                [
                    "_lineno",
                    "model_size_label",
                    "model_name",
                    "status",
                    "quantization",
                    "quant_skip_policy",
                    "prompt_tokens",
                    "decode_tokens",
                    "prefill_tokps",
                    "decode_mode",
                    "selected_decode_path",
                    "decode_tokps",
                    "reference_decode_tokps",
                    "fast_decode_tokps",
                    "fast_forward_backend",
                    "fast_forward_same_next_token",
                    "model_footprint_mb",
                    "peak_vram_mb",
                    "hidden_size",
                    "num_hidden_layers",
                    "head_dim",
                    "num_heads",
                    "error",
                ],
            )
            for r in quant_model_latest
        ],
        "quantization_best_variants": quant_best_variants,
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
    print("\n## batch_sweep_default_native_graph")
    if report["batch_sweep_default_native_graph"]:
        for row in report["batch_sweep_default_native_graph"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## batch_sweep_experimental_native_graph")
    if report["batch_sweep_experimental_native_graph"]:
        for row in report["batch_sweep_experimental_native_graph"]:
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
    print("\n## native_prefill_scan")
    if report["native_prefill_scan"]:
        for row in report["native_prefill_scan"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## native_prefill_breakdown")
    if report["native_prefill_breakdown"]:
        for row in report["native_prefill_breakdown"]:
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
    print("\n## ttft_tpot")
    if report["ttft_tpot"]:
        for row in report["ttft_tpot"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
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
    print("\n## fused_projection_proto")
    print(json.dumps(report["fused_projection_proto"], ensure_ascii=False) if report["fused_projection_proto"] else "PENDING")
    print("\n## albatross_projection_layout_tune")
    print(json.dumps(report["albatross_projection_layout_tune"], ensure_ascii=False) if report["albatross_projection_layout_tune"] else "PENDING")
    print("\n## fused_wa_lora_proto")
    print(json.dumps(report["fused_wa_lora_proto"], ensure_ascii=False) if report["fused_wa_lora_proto"] else "PENDING")
    print("\n## fused_wag_lora_proto")
    print(json.dumps(report["fused_wag_lora_proto"], ensure_ascii=False) if report["fused_wag_lora_proto"] else "PENDING")
    print("\n## fused_wavg_lora_proto")
    print(json.dumps(report["fused_wavg_lora_proto"], ensure_ascii=False) if report["fused_wavg_lora_proto"] else "PENDING")
    print("\n## fused_rkv_wag_projection_proto")
    print(json.dumps(report["fused_rkv_wag_projection_proto"], ensure_ascii=False) if report["fused_rkv_wag_projection_proto"] else "PENDING")
    print("\n## fused_attn_output_proto")
    print(json.dumps(report["fused_attn_output_proto"], ensure_ascii=False) if report["fused_attn_output_proto"] else "PENDING")
    print("\n## fused_attn_output_project_proto")
    print(json.dumps(report["fused_attn_output_project_proto"], ensure_ascii=False) if report["fused_attn_output_project_proto"] else "PENDING")
    print("\n## fused_ffn_proto")
    print(json.dumps(report["fused_ffn_proto"], ensure_ascii=False) if report["fused_ffn_proto"] else "PENDING")
    print("\n## fused_shift_mix_proto")
    print(json.dumps(report["fused_shift_mix_proto"], ensure_ascii=False) if report["fused_shift_mix_proto"] else "PENDING")
    print("\n## fused_recurrent_proto")
    print(json.dumps(report["fused_recurrent_proto"], ensure_ascii=False) if report["fused_recurrent_proto"] else "PENDING")
    print("\n## fused_recurrent_scan_proto")
    print(json.dumps(report["fused_recurrent_scan_proto"], ensure_ascii=False) if report["fused_recurrent_scan_proto"] else "PENDING")
    print("\n## dplr_prefill_scan_proto")
    if report["dplr_prefill_scan_proto"]:
        for row in report["dplr_prefill_scan_proto"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## dplr_chunk_summary_proto")
    if report["dplr_chunk_summary_proto"]:
        for row in report["dplr_chunk_summary_proto"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## dplr_dense3_stage_proto")
    if report["dplr_dense3_stage_proto"]:
        for row in report["dplr_dense3_stage_proto"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## fused_recurrent_output_proto")
    print(json.dumps(report["fused_recurrent_output_proto"], ensure_ascii=False) if report["fused_recurrent_output_proto"] else "PENDING")
    print("\n## native_graph_fused_recurrent")
    print(json.dumps(report["native_graph_fused_recurrent"], ensure_ascii=False) if report["native_graph_fused_recurrent"] else "PENDING")
    print("\n## native_graph_fused_recurrent_output")
    print(json.dumps(report["native_graph_fused_recurrent_output"], ensure_ascii=False) if report["native_graph_fused_recurrent_output"] else "PENDING")
    print("\n## native_graph_fused_output")
    print(json.dumps(report["native_graph_fused_output"], ensure_ascii=False) if report["native_graph_fused_output"] else "PENDING")
    print("\n## native_graph_fused_output_project")
    print(json.dumps(report["native_graph_fused_output_project"], ensure_ascii=False) if report["native_graph_fused_output_project"] else "PENDING")
    print("\n## native_graph_fused_wag_lora")
    print(json.dumps(report["native_graph_fused_wag_lora"], ensure_ascii=False) if report["native_graph_fused_wag_lora"] else "PENDING")
    print("\n## native_graph_fused_wavg_lora")
    print(json.dumps(report["native_graph_fused_wavg_lora"], ensure_ascii=False) if report["native_graph_fused_wavg_lora"] else "PENDING")
    print("\n## native_graph_fused_projection")
    print(json.dumps(report["native_graph_fused_projection"], ensure_ascii=False) if report["native_graph_fused_projection"] else "PENDING")
    print("\n## native_graph_fused_projection_sweep")
    if report["native_graph_fused_projection_sweep"]:
        for row in report["native_graph_fused_projection_sweep"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## native_graph_fused_output_sweep")
    if report["native_graph_fused_output_sweep"]:
        for row in report["native_graph_fused_output_sweep"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## native_graph_fused_recurrent_output_sweep")
    if report["native_graph_fused_recurrent_output_sweep"]:
        for row in report["native_graph_fused_recurrent_output_sweep"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## native_graph_fused_output_project_sweep")
    if report["native_graph_fused_output_project_sweep"]:
        for row in report["native_graph_fused_output_project_sweep"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## native_graph_fused_wag_lora_sweep")
    if report["native_graph_fused_wag_lora_sweep"]:
        for row in report["native_graph_fused_wag_lora_sweep"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## native_graph_fused_wavg_lora_sweep")
    if report["native_graph_fused_wavg_lora_sweep"]:
        for row in report["native_graph_fused_wavg_lora_sweep"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## native_quant_gemv_proto")
    print(json.dumps(report["native_quant_gemv_proto"], ensure_ascii=False) if report["native_quant_gemv_proto"] else "PENDING")
    print("\n## native_quant_w4_gemv_proto")
    print(json.dumps(report["native_quant_w4_gemv_proto"], ensure_ascii=False) if report["native_quant_w4_gemv_proto"] else "PENDING")
    print("\n## native_quant_rkv_proto")
    print(json.dumps(report["native_quant_rkv_proto"], ensure_ascii=False) if report["native_quant_rkv_proto"] else "PENDING")
    print("\n## native_quant_w4_rkv_proto")
    print(json.dumps(report["native_quant_w4_rkv_proto"], ensure_ascii=False) if report["native_quant_w4_rkv_proto"] else "PENDING")
    print("\n## native_quant_rkv_sweep")
    if report["native_quant_rkv_sweep"]:
        for row in report["native_quant_rkv_sweep"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## larger_model_smoke")
    if report["larger_model_smoke"]:
        for row in report["larger_model_smoke"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## device_map_smoke")
    print(json.dumps(report["device_map_smoke"], ensure_ascii=False) if report["device_map_smoke"] else "PENDING")
    print("\n## speculative_decode")
    print(json.dumps(report["speculative_decode"], ensure_ascii=False) if report["speculative_decode"] else "PENDING")
    print("\n## native_model_smoke")
    print(json.dumps(report["native_model_smoke"], ensure_ascii=False) if report["native_model_smoke"] else "PENDING")
    print("\n## training_smoke")
    if report.get("training_smoke"):
        for row in report["training_smoke"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## deepspeed_training_smoke")
    if report.get("deepspeed_training_smoke"):
        for row in report["deepspeed_training_smoke"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")

    print("\n## albatross_speed")
    if report.get("albatross_speed"):
        for row in report["albatross_speed"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## albatross_decode_comparison")
    if report.get("albatross_decode_comparison"):
        for row in report["albatross_decode_comparison"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## albatross_prefill_comparison")
    if report.get("albatross_prefill_comparison"):
        for row in report["albatross_prefill_comparison"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## fused_backend_targets")
    print(json.dumps(report.get("fused_backend_targets"), ensure_ascii=False))

    print("\n## quantization")
    if report["quantization"]:
        for row in report["quantization"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## quantization_best_variants")
    if report.get("quantization_best_variants"):
        for row in report["quantization_best_variants"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("PENDING")
    print("\n## quantization_model_sweep")
    if report.get("quantization_model_sweep"):
        for row in report["quantization_model_sweep"]:
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
