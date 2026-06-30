#!/usr/bin/env python3
# coding=utf-8
"""Compare RWKV-7 fast-token layout benchmark rows.

Rows without `fast_token_layout` are treated as the validated `3d` baseline so
older results remain comparable. The script is intentionally JSONL-only and has
no torch dependency, making it suitable for quick local and CI checks after V100
runs append new rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def match_device(row: dict[str, Any], device: str | None) -> bool:
    if not device:
        return True
    value = row.get("device")
    return value is None or device.lower() in str(value).lower()


def match_dtype(row: dict[str, Any], dtype: str | None) -> bool:
    return not dtype or row.get("dtype") == dtype


def layout(row: dict[str, Any]) -> str:
    return str(row.get("fast_token_layout") or "3d")


def fast_speed_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if not match_device(row, args.device) or not match_dtype(row, args.dtype):
            continue
        if row.get("axis") != "speed_mem" or row.get("backend") != "hf_adapter":
            continue
        if row.get("hf_decode_api") != "rwkv7_forward_token":
            continue
        out.append(row)
    return out


def fast_micro_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if not match_device(row, args.device) or not match_dtype(row, args.dtype):
            continue
        if row.get("axis") != "decode_micro" or row.get("backend") != "hf_adapter":
            continue
        if row.get("fast_decode_api_name") != "rwkv7_forward_token":
            continue
        out.append(row)
    return out


def latest_by_layout(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[layout(row)] = row
    return latest


def num(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nested_num(row: dict[str, Any] | None, key: str, nested: str) -> float | None:
    if not row or not isinstance(row.get(key), dict):
        return None
    value = row[key].get(nested)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def compact_speed(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    keys = [
        "_lineno", "fast_token_layout", "decode_tokps", "decode_ms_per_tok",
        "prefill_tokps", "peak_vram_mb", "attn_mode", "cache_type",
    ]
    out = {k: row[k] for k in keys if k in row}
    out.setdefault("fast_token_layout", layout(row))
    return out


def compact_micro(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    keys = [
        "_lineno", "fast_token_layout", "hf_forward_fixed", "fast_decode_fixed",
        "fast_decode_greedy", "lm_head", "argmax", "peak_vram_mb",
    ]
    out = {k: row[k] for k in keys if k in row}
    out.setdefault("fast_token_layout", layout(row))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    ap.add_argument("--device", default="V100", help="Case-insensitive device substring")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--baseline-layout", default="3d")
    ap.add_argument("--candidate-layout", default="2d")
    ap.add_argument("--min-speedup", type=float, default=1.0,
                    help="Minimum candidate/baseline ratio required when --require-candidate is set")
    ap.add_argument("--require-candidate", action="store_true",
                    help="Exit non-zero unless candidate rows exist and meet --min-speedup on all available ratios")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = load_rows(Path(args.results))
    speed = latest_by_layout(fast_speed_rows(rows, args))
    micro = latest_by_layout(fast_micro_rows(rows, args))
    base_speed = speed.get(args.baseline_layout)
    cand_speed = speed.get(args.candidate_layout)
    base_micro = micro.get(args.baseline_layout)
    cand_micro = micro.get(args.candidate_layout)

    report = {
        "filters": {"device": args.device, "dtype": args.dtype},
        "baseline_layout": args.baseline_layout,
        "candidate_layout": args.candidate_layout,
        "speed_mem": {
            "baseline": compact_speed(base_speed),
            "candidate": compact_speed(cand_speed),
            "candidate_vs_baseline_decode_tokps": round(ratio(num(cand_speed, "decode_tokps"), num(base_speed, "decode_tokps")), 4)
            if ratio(num(cand_speed, "decode_tokps"), num(base_speed, "decode_tokps")) is not None else None,
        },
        "decode_micro": {
            "baseline": compact_micro(base_micro),
            "candidate": compact_micro(cand_micro),
            "candidate_vs_baseline_fast_fixed_tokps": round(
                ratio(nested_num(cand_micro, "fast_decode_fixed", "tokps"), nested_num(base_micro, "fast_decode_fixed", "tokps")),
                4,
            ) if ratio(nested_num(cand_micro, "fast_decode_fixed", "tokps"), nested_num(base_micro, "fast_decode_fixed", "tokps")) is not None else None,
        },
    }

    speed_ratio = report["speed_mem"]["candidate_vs_baseline_decode_tokps"]
    micro_ratio = report["decode_micro"]["candidate_vs_baseline_fast_fixed_tokps"]
    ratios = [r for r in (speed_ratio, micro_ratio) if r is not None]
    has_candidate = cand_speed is not None or cand_micro is not None
    failures = []
    if args.require_candidate and not has_candidate:
        failures.append("candidate layout rows missing")
    if args.require_candidate and not ratios:
        failures.append("candidate/baseline ratios unavailable")
    if args.require_candidate:
        for name, value in (("speed_mem.decode_tokps", speed_ratio), ("decode_micro.fast_fixed_tokps", micro_ratio)):
            if value is not None and value < args.min_speedup:
                failures.append(f"{name} ratio {value:.4f} < {args.min_speedup:.4f}")

    report["required"] = {
        "enabled": bool(args.require_candidate),
        "min_speedup": args.min_speedup,
        "ok": not failures,
        "failures": failures,
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("# RWKV-7 fast-token layout comparison")
        print(json.dumps(report, ensure_ascii=False))
        if not ratios:
            print("PENDING: need both baseline and candidate layout rows")
        elif any(r < args.min_speedup for r in ratios):
            print("CANDIDATE: below required speedup on available rows")
        else:
            print("CANDIDATE: meets required speedup on available rows")
        if failures:
            print("FAIL: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
