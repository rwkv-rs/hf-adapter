#!/usr/bin/env python3
# coding=utf-8
"""Summarize Blackwell fresh-process native quant benchmark JSONL as Markdown."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--only-pass", action="store_true")
    args = ap.parse_args()
    rows = load_rows(Path(args.jsonl))
    if args.only_pass:
        rows = [r for r in rows if r.get("status") == "pass"]

    print("| Model | Prompt | Decode | Bsz | Quant | Tok/s total | Ratio | Footprint MB | Footprint ratio | Peak MB | Prompt cos | Final cos | Same next | Status |")
    print("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for r in sorted(
        rows,
        key=lambda x: (
            str(x.get("model_size_label")),
            int(x.get("prompt_tokens") or 0),
            int(x.get("decode_tokens") or 0),
            int(x.get("batch_size") or 0),
            {"none": 0, "mm8": 1, "mm4": 2}.get(str(x.get("quantization")), 9),
        ),
    ):
        print(
            "| {model} | {prompt} | {decode} | {bsz} | {quant} | {tokps} | {ratio} | {foot} | {fratio} | {peak} | {pcos} | {fcos} | {same} | {status} |".format(
                model=r.get("model_size_label", "-"),
                prompt=r.get("prompt_tokens", "-"),
                decode=r.get("decode_tokens", "-"),
                bsz=r.get("batch_size", "-"),
                quant=r.get("quantization", "-"),
                tokps=fmt(r.get("decode_tokps_total"), 1),
                ratio=fmt(r.get("decode_speed_ratio_vs_fp16"), 4),
                foot=fmt(r.get("model_footprint_mb"), 1),
                fratio=fmt(r.get("footprint_ratio_vs_fp16"), 4),
                peak=fmt(r.get("peak_vram_mb"), 1),
                pcos=fmt(r.get("prompt_logits_cos_vs_fp16"), 6),
                fcos=fmt(r.get("final_logits_cos_vs_fp16"), 6),
                same=r.get("same_next_token_as_fp16", "-"),
                status=r.get("status", "-"),
            )
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("status") == "pass" and r.get("quantization") in {"mm8", "mm4"}:
            grouped[(str(r.get("model_size_label")), str(r.get("quantization")))].append(r)
    print("\n## Ratio summary")
    print("| Model | Quant | Rows | Min speed ratio | Median speed ratio | Min footprint ratio | Same-next pass |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for (model, quant), vals in sorted(grouped.items()):
        ratios = sorted(float(v["decode_speed_ratio_vs_fp16"]) for v in vals if v.get("decode_speed_ratio_vs_fp16") is not None)
        fr = sorted(float(v["footprint_ratio_vs_fp16"]) for v in vals if v.get("footprint_ratio_vs_fp16") is not None)
        same = sum(1 for v in vals if v.get("same_next_token_as_fp16") is True)
        median = ratios[len(ratios) // 2] if ratios else None
        print(f"| {model} | {quant} | {len(vals)} | {fmt(min(ratios) if ratios else None, 4)} | {fmt(median, 4)} | {fmt(min(fr) if fr else None, 4)} | {same}/{len(vals)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
