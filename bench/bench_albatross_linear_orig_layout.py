#!/usr/bin/env python3
# coding=utf-8
"""Micro-tune Albatross linear_orig_layout candidates on one GPU.

This probes the same low-level ops used by BlinkDL/Albatross v3a/v4 for weights
kept in original layout.  It is intentionally external-reference tooling: run it
on the Albatross checkout, store the compact JSON/Markdown report, and use the
result to decide which Albatross reference is the tuned speed ceiling for this
GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable


def _add_albatross(albatross_dir: str) -> None:
    path = str(Path(albatross_dir).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    os.chdir(path)


def _load_dims(torch, model_path: str) -> dict[str, int]:
    z = torch.load(model_path, map_location="cpu", mmap=True)
    h, n = z["blocks.0.att.r_k"].shape
    c = int(h) * int(n)
    v = int(z["emb.weight"].shape[0])
    f = int(z["blocks.0.ffn.key.weight"].squeeze().shape[0])
    return {"C": c, "H": int(h), "N": int(n), "V": v, "F": f}


def _candidate_specs(rows: int) -> list[tuple[str, Callable[[Any, Any, Any], Any]]]:
    # The operation callables receive (torch, x, weight_orig).
    specs: list[tuple[str, Callable[[Any, Any, Any], Any]]] = []
    specs.append(("orig", lambda torch, x, w: torch.ops.rwkv7_v3a_ops.linear_f16_orig(x, w)))
    specs.append(("wmma16", lambda torch, x, w: torch.ops.rwkv7_v3a_ops.linear_orig_wmma16_f16(x, w)))
    if rows in (1, 2):
        exact = [(128, 2, False), (128, 2, True)]
        if rows == 2:
            exact.extend([(64, 2, True), (256, 1, True)])
        for threads, out_tile, use4 in exact:
            specs.append(
                (
                    f"exact_t{threads}_o{out_tile}_u{int(use4)}",
                    lambda torch, x, w, threads=threads, out_tile=out_tile, use4=use4: torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                        x, w, threads, out_tile, use4
                    ),
                )
            )
    for workspace_mb in (0, 32, 128):
        for algo in range(7):
            specs.append(
                (
                    f"lt_ws{workspace_mb}_a{algo}",
                    lambda torch, x, w, workspace_mb=workspace_mb, algo=algo: torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x, w, workspace_mb, algo
                    ),
                )
            )
    for row_tile, out_tile in [
        (1, 2),
        (1, 4),
        (1, 8),
        (1, 16),
        (2, 2),
        (2, 4),
        (2, 8),
        (3, 2),
        (3, 4),
        (3, 8),
        (4, 2),
        (4, 4),
        (4, 8),
        (8, 2),
        (8, 4),
        (16, 1),
        (16, 2),
        (16, 4),
    ]:
        specs.append(
            (
                f"rows_r{row_tile}_o{out_tile}",
                lambda torch, x, w, row_tile=row_tile, out_tile=out_tile: torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(
                    x, w, row_tile, out_tile
                ),
            )
        )
    for threads, row_tile, out_tile in [
        (64, 1, 4),
        (64, 1, 8),
        (128, 1, 8),
        (256, 1, 1),
        (32, 4, 4),
        (64, 4, 4),
        (96, 4, 4),
        (32, 4, 8),
        (64, 4, 8),
        (32, 8, 4),
        (64, 8, 4),
        (32, 2, 4),
        (64, 2, 2),
        (64, 2, 4),
        (32, 3, 4),
        (64, 3, 4),
        (96, 3, 4),
        (32, 3, 8),
        (64, 3, 8),
    ]:
        specs.append(
            (
                f"cfg_t{threads}_r{row_tile}_o{out_tile}",
                lambda torch, x, w, threads=threads, row_tile=row_tile, out_tile=out_tile: torch.ops.rwkv7_v3a_ops.linear_orig_rows_cfg_f16(
                    x, w, threads, row_tile, out_tile
                ),
            )
        )
    return specs


def _v4_policy(group: str, rows: int, k: int) -> str:
    # Human-readable mirror of faster4_2605_cpp/src/rwkv7_fast_v4.cu for the
    # relevant 0.4B tuning buckets.  Unknown/fallback policies return "orig".
    if rows == 1:
        if group == "ffn_key":
            return "exact_t128_o2_u1" if k <= 1024 or k == 2560 else "exact_t128_o2_u0"
        if group == "att_c2c":
            return "exact_t128_o2_u1" if k < 2048 else "exact_t128_o2_u0"
        return "exact_t128_o2_u1"
    if rows == 2:
        if group == "att_c2c":
            return "exact_t64_o2_u1"
        if group == "ffn_key":
            if k == 2560:
                return "exact_t128_o2_u0"
            return "exact_t64_o2_u1" if k < 4096 else "exact_t128_o2_u0"
        if group == "head" and k == 2560:
            return "exact_t128_o2_u0"
        return "exact_t64_o2_u1"
    if group == "head":
        if k == 1024:
            if 256 <= rows < 384:
                return "orig"
            if 192 <= rows < 256:
                return "lt_ws0_a2"
            if 96 <= rows < 160:
                return "lt_ws32_a1"
        if rows >= 512:
            return "lt_ws0_a2"
        if rows >= 384:
            return "lt_ws128_a2"
        if rows >= 256:
            return "lt_ws0_a1"
        if rows >= 192:
            return "lt_ws128_a0"
        if rows >= 160:
            return "lt_ws32_a0"
        if rows >= 128:
            return "lt_ws128_a0"
        if rows >= 112:
            return "lt_ws32_a0"
        if rows >= 96:
            return "lt_ws32_a1"
        if rows >= 80:
            return "lt_ws32_a2"
        if rows >= 72:
            return "lt_ws128_a2"
        return "orig"
    if group == "att_c2c":
        if k == 1024:
            if 256 <= rows < 384:
                return "lt_ws128_a0"
            if 96 <= rows < 112:
                return "lt_ws32_a6"
        if rows >= 1024:
            return "lt_ws32_a4"
        if rows >= 768:
            return "lt_ws32_a0"
        if rows >= 512:
            return "lt_ws32_a1"
        if rows >= 384:
            return "lt_ws128_a2"
        if rows >= 256:
            return "lt_ws128_a0"
        if rows >= 192:
            return "lt_ws0_a0"
        if rows >= 160:
            return "lt_ws128_a1"
        if rows >= 128:
            return "lt_ws128_a0"
        if rows >= 112:
            return "orig"
        if rows >= 96:
            return "lt_ws0_a5"
        if rows >= 72:
            return "lt_ws32_a0"
        if rows >= 48:
            return "lt_ws32_a6"
        if rows >= 32:
            return "lt_ws0_a0"
        if rows >= 24:
            return "lt_ws0_a6"
        if rows >= 12:
            return "lt_ws0_a0"
        if rows >= 5:
            return "lt_ws0_a2"
        return "orig"
    # ffn_key / other
    if k == 1024:
        if 256 <= rows < 384:
            return "lt_ws32_a2"
        if 192 <= rows < 256:
            return "lt_ws0_a0"
        if 96 <= rows < 160:
            return "lt_ws32_a2"
    if rows >= 1024:
        return "lt_ws0_a0"
    if rows >= 768:
        return "lt_ws32_a1"
    if rows >= 512:
        return "lt_ws128_a3"
    if rows >= 384:
        return "lt_ws32_a0"
    if rows >= 256:
        return "lt_ws0_a0"
    if rows >= 192:
        return "lt_ws0_a1"
    if rows >= 160:
        return "lt_ws0_a2"
    if rows >= 128:
        return "lt_ws32_a0"
    if rows >= 112:
        return "lt_ws32_a3"
    if rows >= 96:
        return "lt_ws32_a1"
    if rows >= 72:
        return "lt_ws128_a1"
    if rows >= 64:
        return "lt_ws0_a0"
    if rows >= 48:
        return "lt_ws0_a1"
    if rows >= 12:
        return "lt_ws0_a0"
    if rows in (5, 6):
        return "lt_ws0_a1"
    return "orig"


def _bench_one(torch, label: str, fn, x, w, baseline, warmup: int, iters: int) -> dict[str, Any]:
    try:
        with torch.inference_mode():
            out = fn(torch, x, w)
            torch.cuda.synchronize()
            if tuple(out.shape) != (x.shape[0], w.shape[0]):
                raise RuntimeError(f"bad output shape {tuple(out.shape)}")
            max_abs = float((out.float() - baseline.float()).abs().max().detach().cpu())
            for _ in range(warmup):
                fn(torch, x, w)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            times: list[float] = []
            for _ in range(iters):
                start.record()
                fn(torch, x, w)
                end.record()
                torch.cuda.synchronize()
                times.append(float(start.elapsed_time(end)))
        return {"label": label, "status": "pass", "p50_ms": statistics.median(times), "min_ms": min(times), "max_abs_diff": max_abs}
    except Exception as exc:
        torch.cuda.synchronize()
        return {"label": label, "status": "fail", "error": f"{type(exc).__name__}: {str(exc)[:240]}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--albatross-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    _add_albatross(args.albatross_dir)
    import torch
    import rwkv7_fast_v3a as v3a

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    v3a.load_extensions("fp16")
    dims = _load_dims(torch, args.model)
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    cases = [
        {"case": "att_c2c_b1t1", "group": "att_c2c", "rows": 1, "K": dims["C"], "N": dims["C"]},
        {"case": "att_c2c_b64t1", "group": "att_c2c", "rows": 64, "K": dims["C"], "N": dims["C"]},
        {"case": "att_c2c_b1t512", "group": "att_c2c", "rows": 512, "K": dims["C"], "N": dims["C"]},
        {"case": "ffn_key_b1t1", "group": "ffn_key", "rows": 1, "K": dims["C"], "N": dims["F"]},
        {"case": "ffn_key_b64t1", "group": "ffn_key", "rows": 64, "K": dims["C"], "N": dims["F"]},
        {"case": "ffn_key_b1t512", "group": "ffn_key", "rows": 512, "K": dims["C"], "N": dims["F"]},
        {"case": "head_b1", "group": "head", "rows": 1, "K": dims["C"], "N": dims["V"]},
        {"case": "head_b64", "group": "head", "rows": 64, "K": dims["C"], "N": dims["V"]},
    ]
    results: list[dict[str, Any]] = []
    for case in cases:
        rows, k, n = int(case["rows"]), int(case["K"]), int(case["N"])
        print(f"tuning {case['case']} rows={rows} K={k} N={n}", flush=True)
        x = torch.randn((rows, k), device=device, dtype=torch.float16, generator=gen)
        w = torch.randn((n, k), device=device, dtype=torch.float16, generator=gen)
        baseline = torch.ops.rwkv7_v3a_ops.linear_f16_orig(x, w)
        torch.cuda.synchronize()
        cand = []
        for label, fn in _candidate_specs(rows):
            cand.append(_bench_one(torch, label, fn, x, w, baseline, args.warmup, args.iters))
        passed = [r for r in cand if r["status"] == "pass"]
        passed.sort(key=lambda r: float(r["p50_ms"]))
        current = _v4_policy(str(case["group"]), rows, k)
        current_row = next((r for r in passed if r["label"] == current), None)
        best = passed[0] if passed else None
        result = {**case, "v4_policy": current, "v4_policy_result": current_row, "best": best, "top10": passed[:10], "fail_count": len(cand) - len(passed)}
        results.append(result)
        del x, w, baseline
        torch.cuda.empty_cache()

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "model": args.model,
        "dims": dims,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# Albatross linear_orig_layout 4090 tuning", ""]
    lines.append(f"GPU: `{payload['gpu']}` sm `{payload['capability']}`")
    lines.append(f"Model: `{Path(args.model).name}`, dims `{dims}`")
    lines.append("")
    lines.append("| Case | v4 policy | v4 p50 ms | best label | best p50 ms | best/v4 | max abs diff(best) |")
    lines.append("|---|---|---:|---|---:|---:|---:|")
    for r in results:
        cur = r.get("v4_policy_result")
        best = r.get("best")
        cur_ms = float(cur["p50_ms"]) if cur else None
        best_ms = float(best["p50_ms"]) if best else None
        ratio = (cur_ms / best_ms) if (cur_ms and best_ms) else None
        lines.append(
            f"| {r['case']} | `{r['v4_policy']}` | {cur_ms if cur_ms is not None else float('nan'):.6f} | "
            f"`{best['label'] if best else 'n/a'}` | {best_ms if best_ms is not None else float('nan'):.6f} | "
            f"{ratio if ratio is not None else float('nan'):.3f}x | {float(best.get('max_abs_diff', 0.0)) if best else float('nan'):.6g} |"
        )
    lines.append("")
    lines.append("Interpretation: ratios above `1.0x` mean the current v4 policy is slower than the fastest passing candidate in this synthetic microbench bucket. Use this as per-GPU reference-tuning evidence, not as a drop-in correctness proof for full MATH500.")
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(Path(args.out_md).read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
