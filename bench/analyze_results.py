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


def best_latest_by_key(rows: Iterable[dict[str, Any]], key_fn, score_fn) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    out = []
    for key, vals in groups.items():
        vals = [v for v in vals if score_fn(v) is not None]
        if not vals:
            continue
        out.append(max(vals, key=lambda v: (float(score_fn(v)), int(v.get("_lineno", 0)))))
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


def analyze(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rows = filt(rows, device=args.device, dtype=args.dtype)
    target_decode_ratio = args.target_decode_ratio
    target_prefill_ratio = args.target_prefill_ratio
    target_memory_ratio = args.target_memory_ratio

    speed_hf = latest(rows, lambda r: r.get("axis") == "speed_mem" and r.get("backend") == "hf_adapter")
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
    batch_latest = best_latest_by_key(
        batch_rows,
        lambda r: (r.get("batch_size"), r.get("decode_api")),
        lambda r: r.get("decode_tokps_total"),
    )
    dynamic_rows = [r for r in rows if r.get("axis") == "dynamic_batch" and r.get("backend") == "hf_adapter"]
    dynamic_latest = best_latest_by_key(
        dynamic_rows,
        lambda r: r.get("decode_api"),
        lambda r: r.get("decode_tokps_total"),
    )
    micro = latest(rows, lambda r: r.get("axis") == "decode_micro" and r.get("backend") == "hf_adapter")
    components = latest(rows, lambda r: r.get("axis") == "decode_components" and r.get("backend") == "hf_adapter")

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
    if not dynamic_latest:
        focus.append("dynamic_batch rows pending")
    if micro is None:
        focus.append("decode_micro rows pending")
    if components is None:
        focus.append("decode_components rows pending")
    elif components.get("top_components"):
        top = components["top_components"][0]
        if isinstance(top, (list, tuple)) and len(top) >= 2:
            focus.append(f"largest fast-token component: {top[0]} {top[1]} ms/token")
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
            "hf": compact(speed_hf, ["_lineno", "device", "attn_mode", "fuse_norm", "fast_cache", "hf_decode_api", "prefill_tokps", "decode_tokps", "decode_ms_per_tok", "peak_vram_mb"]),
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
            "best_row": compact(best_fast, ["_lineno", "axis", "hf_decode_api", "fast_decode_api_name", "attn_mode", "decode_tokps", "decode_fast_api_greedy_tokps", "decode_fast_api_fixed_tokps", "peak_vram_mb"]),
            "decode_tokps": round(best_fast_tokps, 4) if best_fast_tokps is not None else None,
            "decode_ratio": round(fast_decode_ratio, 4) if fast_decode_ratio is not None else None,
            "decode_status": verdict_ge(fast_decode_ratio, target_decode_ratio),
        },
        "precision": {
            "latest": compact(latest_precision, ["_lineno", "axis", "dtype", "top5_match", "argmax_match", "cosine", "max_abs_diff", "mean_abs_diff", "greedy_window"]),
            "greedy_ratio": round(greedy_ratio, 4) if greedy_ratio is not None else None,
        },
        "batch_sweep": [compact(r, ["_lineno", "batch_size", "decode_api", "decode_tokps_total", "decode_tokps_per_seq", "decode_ms_per_step", "peak_vram_mb"]) for r in batch_latest],
        "dynamic_batch": [compact(r, ["_lineno", "decode_api", "initial_batch_size", "final_batch_size", "total_decode_tokens", "reorder_count", "drop_count", "decode_tokps_total", "decode_ms_per_token", "peak_vram_mb"]) for r in dynamic_latest],
        "decode_micro": compact(micro, ["_lineno", "fast_decode_api_name", "hf_forward_fixed", "hf_forward_greedy", "fast_decode_fixed", "fast_decode_greedy", "norm_lm_head", "lm_head", "argmax", "empty_loop", "peak_vram_mb"]),
        "decode_components": compact(components, ["_lineno", "decode_api", "batch_size", "wall_ms_per_token", "decode_tokps_wall", "top_components", "top_layers", "peak_vram_mb"]),
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
    print("\n## decode_micro")
    print(json.dumps(report["decode_micro"], ensure_ascii=False) if report["decode_micro"] else "PENDING")
    print("\n## decode_components")
    print(json.dumps(report["decode_components"], ensure_ascii=False) if report["decode_components"] else "PENDING")
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
