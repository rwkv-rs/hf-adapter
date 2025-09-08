#!/usr/bin/env python3
# coding=utf-8
"""Summarize RWKV-7 benchmark JSONL results.

Examples:
  python bench/summarize_results.py --results bench/results.jsonl --device V100 --last 12
  python bench/summarize_results.py --axis speed_mem --require-fast-decode
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


def match(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.axis and row.get("axis") != args.axis:
        return False
    if args.backend and row.get("backend") != args.backend:
        return False
    if args.device and args.device.lower() not in str(row.get("device", "")).lower():
        return False
    if args.require_fast_decode:
        if not (row.get("hf_decode_api") == "rwkv7_forward_one" or row.get("fast_decode_api") is True):
            return False
    return True


def compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "_lineno", "axis", "backend", "device", "dtype", "attn_mode",
        "fuse_norm", "fast_cache", "cache_type", "hf_decode_api", "fast_decode_api",
        "decode_api", "batch_size", "prompt_tokens", "decode_tokens",
        "prefill_tokps", "prefill_keep1_tokps", "prefill_tokps_total",
        "prefill_tokps_per_seq", "prefill_ms",
        "decode_tokps", "decode_tokps_total", "decode_tokps_per_seq",
        "decode_ms_per_step", "decode_greedy_tokps", "decode_fixed_tokps",
        "decode_fast_api_greedy_tokps", "decode_fast_api_fixed_tokps",
        "decode_ms_per_tok", "decode_greedy_ms_per_tok", "decode_fast_api_greedy_ms_per_tok",
        "peak_vram_mb", "top5_match", "argmax_match", "cosine", "max_abs_diff",
    ]
    return {k: row[k] for k in keys if k in row}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    ap.add_argument("--axis", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--device", default=None, help="Case-insensitive substring match")
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--require-fast-decode", action="store_true")
    ap.add_argument("--json", action="store_true", help="Emit compact rows as JSON instead of text")
    args = ap.parse_args()

    rows = [r for r in load_rows(Path(args.results)) if match(r, args)]
    rows = rows[-args.last:]
    if args.json:
        print(json.dumps([compact(r) for r in rows], indent=2, ensure_ascii=False))
    else:
        for row in rows:
            print(json.dumps(compact(row), ensure_ascii=False))
    if args.require_fast_decode and not rows:
        raise SystemExit("No fast-decode rows found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
