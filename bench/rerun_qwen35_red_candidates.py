#!/usr/bin/env python3
"""Rerun slow/failing RWKV rows from a completed Qwen3.5 matrix.

The comparator keeps the last row for each role/cell, so corrected candidate
rows can be appended without deleting negative evidence. Reference failures are
never hidden and remain fail-closed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from compare_qwen35_speed_matrix import CELL_FIELDS, compare, load_rows, render_markdown


def key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in CELL_FIELDS)


def latest_candidates(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("model_role") == "candidate":
            out[key(row)] = row
    return out


def command(args: argparse.Namespace, row: dict[str, Any]) -> list[str]:
    return [
        args.python_bin,
        args.bench_script,
        "--model", str(row["model_id_or_path"]),
        "--model-kind", "rwkv",
        "--model-role", "candidate",
        "--model-pair", str(row["model_pair"]),
        "--model-size-label", str(row["model_size_label"]),
        "--benchmark-matrix", str(row.get("benchmark_matrix") or "qwen35_hf"),
        "--dtype", str(row["dtype"]),
        "--quantization", str(row["quantization"]),
        "--device", args.device,
        "--batch-size", str(row["batch_size"]),
        "--prompt-tokens", str(row["prompt_tokens"]),
        "--decode-tokens", str(row["decode_tokens"]),
        "--warmup", str(args.warmup),
        "--runs", str(args.runs),
        "--rwkv-attn-mode", args.rwkv_attn_mode,
        "--rwkv-code-source", "repo",
        "--qwen-backend", str(row.get("qwen_backend_requested") or "auto"),
        "--results", str(args.results),
    ]


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    return compare(
        load_rows(args.results),
        expected_cells=args.expected_cells,
        min_prefill_speedup=args.min_prefill_speedup,
        min_decode_speedup=args.min_decode_speedup,
        min_quant_prefill_speedup=args.min_quant_prefill_speedup,
        min_quant_decode_speedup=args.min_quant_decode_speedup,
    )


def write_summary(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--expected-cells", type=int, default=72)
    ap.add_argument("--min-prefill-speedup", type=float, default=1.05)
    ap.add_argument("--min-decode-speedup", type=float, default=1.05)
    ap.add_argument("--min-quant-prefill-speedup", type=float, default=1.0)
    ap.add_argument("--min-quant-decode-speedup", type=float, default=1.0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rwkv-attn-mode", choices=["chunk", "fused_recurrent"], default="fused_recurrent")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--bench-script", default=str(root / "bench" / "bench_cross_model_speed.py"))
    ap.add_argument("--json-output", type=Path)
    ap.add_argument("--markdown-output", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    before = summarize(args)
    candidates = latest_candidates(load_rows(args.results))
    selected: list[dict[str, Any]] = []
    for cell in before["red_cells"]:
        if cell.get("reference_status") != "pass":
            continue
        row = candidates.get(tuple(cell.get(field) for field in CELL_FIELDS))
        if row is not None:
            selected.append(row)

    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["RWKV7_NATIVE_MODEL"] = "0"
    env["RWKV7_FAST_CACHE"] = "1"
    env["RWKV7_FAST_TOKEN_BACKEND"] = "native_graph"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    failures = 0
    for index, row in enumerate(selected, 1):
        cmd = command(args, row)
        print(f"[{index}/{len(selected)}] " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, env=env, text=True)
        failures += int(proc.returncode != 0)

    after = summarize(args)
    write_summary(args, after)
    print(render_markdown(after))
    if failures:
        return 1
    return 0 if after["gates"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
