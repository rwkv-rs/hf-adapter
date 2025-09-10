#!/usr/bin/env python3
# coding=utf-8
"""Check RWKV-7 benchmark JSONL rows against regression or target gates.

Default mode is a *regression gate* for the current PR: it verifies that the
formal V100 benchmark axes exist and that the fast-token path keeps the measured
wins. `--target` switches to the final acceptance thresholds, currently 0.9x
official decode and <=1.1x memory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_analyzer(args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("analyze_results.py")),
        "--results", args.results,
        "--dtype", args.dtype,
        "--json",
    ]
    if args.device:
        cmd += ["--device", args.device]
    cmd += [
        "--target-prefill-ratio", str(args.target_prefill_ratio),
        "--target-decode-ratio", str(args.target_decode_ratio),
        "--target-memory-ratio", str(args.target_memory_ratio),
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


def fail(failures: list[str], msg: str) -> None:
    failures.append(msg)


def check_present(report: dict[str, Any], failures: list[str]) -> None:
    for key in ["speed_mem", "decode_breakdown", "fast_decode", "precision"]:
        if not report.get(key):
            fail(failures, f"missing report section: {key}")
    if not report.get("batch_sweep"):
        fail(failures, "missing batch_sweep rows")
    if not report.get("dynamic_batch"):
        fail(failures, "missing dynamic_batch rows")
    if not report.get("chunked_prefill"):
        fail(failures, "missing chunked_prefill rows")
    if not report.get("decode_micro"):
        fail(failures, "missing decode_micro row")
    if not report.get("forward_fast_path"):
        fail(failures, "missing forward_fast_path row")
    if not report.get("generate_fast_path"):
        fail(failures, "missing generate_fast_path row")
    if not report.get("decode_components"):
        fail(failures, "missing decode_components row")
    if not report.get("projection_lora"):
        fail(failures, "missing projection_lora row")


def index_by(rows: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    return {row.get(key): row for row in rows}


def check_common(report: dict[str, Any], failures: list[str], args: argparse.Namespace) -> None:
    precision = report.get("precision") or {}
    latest = precision.get("latest") or {}
    if latest.get("top5_match", 0) < args.min_top5_match:
        fail(failures, f"top5_match below floor: {latest.get('top5_match')} < {args.min_top5_match}")
    if latest.get("argmax_match", 0) < args.min_argmax_match:
        fail(failures, f"argmax_match below floor: {latest.get('argmax_match')} < {args.min_argmax_match}")
    if latest.get("cosine", 0) < args.min_cosine:
        fail(failures, f"cosine below floor: {latest.get('cosine')} < {args.min_cosine}")
    if precision.get("greedy_ratio") is not None and precision["greedy_ratio"] < args.min_greedy_ratio:
        fail(failures, f"greedy_ratio below floor: {precision['greedy_ratio']} < {args.min_greedy_ratio}")

    speed = report.get("speed_mem") or {}
    mem = speed.get("memory_ratio")
    if mem is None or mem > args.max_memory_ratio:
        fail(failures, f"memory_ratio exceeds floor: {mem} > {args.max_memory_ratio}")

    batch = report.get("batch_sweep") or []
    by_pair = {(r.get("batch_size"), r.get("decode_api")): r for r in batch}
    for bsz in args.required_batch_sizes:
        fwd = by_pair.get((bsz, "forward"))
        fast = by_pair.get((bsz, "rwkv7_forward_token"))
        if not fwd or not fast:
            fail(failures, f"missing batch_sweep forward/fast rows for bsz={bsz}")
            continue
        fwd_tokps = float(fwd.get("decode_tokps_total") or 0)
        fast_tokps = float(fast.get("decode_tokps_total") or 0)
        if fwd_tokps <= 0 or fast_tokps / fwd_tokps < args.min_batch_fast_speedup:
            fail(failures, f"batch bsz={bsz} fast speedup below floor: {fast_tokps}/{fwd_tokps} < {args.min_batch_fast_speedup}x")

    dynamic = index_by(report.get("dynamic_batch") or [], "decode_api")
    dyn_fwd = dynamic.get("forward")
    dyn_fast = dynamic.get("rwkv7_forward_token")
    if not dyn_fwd or not dyn_fast:
        fail(failures, "missing dynamic_batch forward or fast row")
    else:
        fwd_tokps = float(dyn_fwd.get("decode_tokps_total") or 0)
        fast_tokps = float(dyn_fast.get("decode_tokps_total") or 0)
        if fwd_tokps <= 0 or fast_tokps / fwd_tokps < args.min_dynamic_fast_speedup:
            fail(failures, f"dynamic fast speedup below floor: {fast_tokps}/{fwd_tokps} < {args.min_dynamic_fast_speedup}x")
        for row in (dyn_fwd, dyn_fast):
            if row.get("cache_select_api") is not True:
                fail(failures, f"dynamic row did not use cache select API: {row}")
            if row.get("final_cache_batch_size") is not None and row.get("final_batch_size") is not None:
                if int(row["final_cache_batch_size"]) != int(row["final_batch_size"]):
                    fail(failures, f"dynamic cache batch size mismatch: {row}")

    chunked = report.get("chunked_prefill") or []
    chunked_full = [row for row in chunked if row.get("prefill_mode") == "full"]
    chunked_parts = [row for row in chunked if row.get("prefill_mode") == "chunked"]
    if not chunked_full:
        fail(failures, "missing chunked_prefill full row")
    if not chunked_parts:
        fail(failures, "missing chunked_prefill chunked rows")
    for row in chunked_parts:
        if row.get("seq_length_match") is not True:
            fail(failures, f"chunked prefill seq length mismatch: {row}")
        for key in ("max_abs_diff", "decode_max_abs_diff"):
            val = row.get(key)
            if val is None or float(val) > args.max_chunked_prefill_diff:
                fail(failures, f"chunked prefill {key} above floor: {val} > {args.max_chunked_prefill_diff}")

    micro = report.get("decode_micro") or {}
    fast_fixed = (micro.get("fast_decode_fixed") or {}).get("tokps")
    forward_fixed = (micro.get("hf_forward_fixed") or {}).get("tokps")
    if fast_fixed is None or fast_fixed < args.min_micro_fast_tokps:
        fail(failures, f"micro fast tokps below floor: {fast_fixed} < {args.min_micro_fast_tokps}")
    if forward_fixed and fast_fixed and fast_fixed / forward_fixed < args.min_micro_fast_speedup:
        fail(failures, f"micro fast speedup below floor: {fast_fixed}/{forward_fixed} < {args.min_micro_fast_speedup}x")

    forward_fast = report.get("forward_fast_path") or {}
    ref_tokps = (forward_fast.get("reference_forward") or {}).get("tokps")
    auto_tokps = (forward_fast.get("hf_forward_fast") or {}).get("tokps")
    direct_tokps = (forward_fast.get("direct_fast_token") or {}).get("tokps")
    if ref_tokps is None or auto_tokps is None:
        fail(failures, f"forward_fast_path missing reference/auto tokps: {forward_fast}")
    elif float(ref_tokps) <= 0 or float(auto_tokps) / float(ref_tokps) < args.min_forward_fast_speedup:
        fail(failures, f"HF forward fast speedup below floor: {auto_tokps}/{ref_tokps} < {args.min_forward_fast_speedup}x")
    if direct_tokps is not None and auto_tokps is not None:
        if float(auto_tokps) / max(float(direct_tokps), 1e-9) < args.min_forward_fast_vs_direct_ratio:
            fail(failures, f"HF forward fast below direct fast-token ratio: {auto_tokps}/{direct_tokps} < {args.min_forward_fast_vs_direct_ratio}x")
    if forward_fast.get("hf_forward_fast_backend") not in {"native_graph", "native_jit", "fla"}:
        fail(failures, f"unexpected HF forward fast backend: {forward_fast.get('hf_forward_fast_backend')}")
    for key in ("max_abs_diff_auto_vs_reference", "max_abs_diff_direct_vs_reference"):
        val = forward_fast.get(key)
        if val is None or float(val) > args.max_forward_fast_diff:
            fail(failures, f"forward_fast_path {key} above floor: {val} > {args.max_forward_fast_diff}")

    generate_fast = report.get("generate_fast_path") or {}
    gen_ref_tokps = (generate_fast.get("reference_generate") or {}).get("tokps")
    gen_fast_tokps = (generate_fast.get("hf_generate_fast") or {}).get("tokps")
    if gen_ref_tokps is None or gen_fast_tokps is None:
        fail(failures, f"generate_fast_path missing reference/fast tokps: {generate_fast}")
    elif float(gen_ref_tokps) <= 0 or float(gen_fast_tokps) / float(gen_ref_tokps) < args.min_generate_fast_speedup:
        fail(failures, f"generate fast speedup below floor: {gen_fast_tokps}/{gen_ref_tokps} < {args.min_generate_fast_speedup}x")
    if generate_fast.get("generated_equal") is not True:
        fail(failures, f"generate fast path did not preserve greedy output: {generate_fast}")
    if generate_fast.get("generated_tokens_matched") is not None and generate_fast.get("generated_tokens_total") is not None:
        if int(generate_fast["generated_tokens_matched"]) != int(generate_fast["generated_tokens_total"]):
            fail(failures, f"generate token match mismatch: {generate_fast}")
    if generate_fast.get("batch_size") is None or int(generate_fast.get("batch_size") or 0) < args.min_generate_batch_size:
        fail(failures, f"generate fast-path batch size below floor: {generate_fast.get('batch_size')} < {args.min_generate_batch_size}")
    if generate_fast.get("fast_token_backend_effective") not in {"native_graph", "native_jit", "fla"}:
        fail(failures, f"unexpected generate fast backend: {generate_fast.get('fast_token_backend_effective')}")

    components = report.get("decode_components") or {}
    top_components = components.get("top_components") or []
    if not top_components:
        fail(failures, "decode_components has no top_components")
    elif top_components[0][0] != args.expected_top_component:
        fail(failures, f"unexpected top component: {top_components[0][0]} != {args.expected_top_component}")

    proj = report.get("projection_lora") or {}
    candidate_speedup = proj.get("avg_candidate_speedup")
    if candidate_speedup is None:
        fail(failures, "projection_lora missing avg_candidate_speedup")
    elif args.expect_naive_candidate_slower and float(candidate_speedup) >= 1.0:
        fail(failures, f"naive candidate unexpectedly faster/equal: {candidate_speedup}")

    if args.require_quantization:
        quant_rows = report.get("quantization") or []
        by_mode = {row.get("quantization"): row for row in quant_rows}
        base = by_mode.get("none")
        for mode in ("8bit", "4bit"):
            row = by_mode.get(mode)
            if not row:
                fail(failures, f"missing quantization row: {mode}")
                continue
            if row.get("status") != "pass":
                fail(failures, f"quantization {mode} did not pass: {row.get('status')} {row.get('error')}")
                continue
            if base and base.get("status") == "pass":
                base_mem = base.get("model_footprint_mb") or base.get("peak_vram_mb")
                q_mem = row.get("model_footprint_mb") or row.get("peak_vram_mb")
                if base_mem and q_mem:
                    max_ratio = args.max_8bit_memory_ratio if mode == "8bit" else args.max_4bit_memory_ratio
                    if float(q_mem) / float(base_mem) > max_ratio:
                        fail(failures, f"{mode} memory ratio too high: {q_mem}/{base_mem} > {max_ratio}")
                base_decode = base.get("decode_tokps")
                q_decode = row.get("decode_tokps")
                if base_decode and q_decode and float(q_decode) / float(base_decode) < args.min_quant_decode_ratio:
                    fail(failures, f"{mode} decode ratio below floor: {q_decode}/{base_decode} < {args.min_quant_decode_ratio}")


def check_regression(report: dict[str, Any], failures: list[str], args: argparse.Namespace) -> None:
    speed = report.get("speed_mem") or {}
    decode_ratio = speed.get("decode_ratio")
    fast = report.get("fast_decode") or {}
    fast_ratio = fast.get("decode_ratio")
    if decode_ratio is None or decode_ratio < args.min_regression_decode_ratio:
        fail(failures, f"speed decode_ratio below regression floor: {decode_ratio} < {args.min_regression_decode_ratio}")
    if fast_ratio is None or fast_ratio < args.min_regression_decode_ratio:
        fail(failures, f"fast decode_ratio below regression floor: {fast_ratio} < {args.min_regression_decode_ratio}")


def check_target(report: dict[str, Any], failures: list[str], args: argparse.Namespace) -> None:
    speed = report.get("speed_mem") or {}
    fast = report.get("fast_decode") or {}
    if speed.get("decode_ratio") is None or speed["decode_ratio"] < args.target_decode_ratio:
        fail(failures, f"speed decode_ratio below target: {speed.get('decode_ratio')} < {args.target_decode_ratio}")
    if fast.get("decode_ratio") is None or fast["decode_ratio"] < args.target_decode_ratio:
        fail(failures, f"fast decode_ratio below target: {fast.get('decode_ratio')} < {args.target_decode_ratio}")
    if speed.get("memory_ratio") is None or speed["memory_ratio"] > args.target_memory_ratio:
        fail(failures, f"memory_ratio above target: {speed.get('memory_ratio')} > {args.target_memory_ratio}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    ap.add_argument("--device", default="V100")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--target", action="store_true", help="Use final acceptance gates instead of current regression floors")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--target-prefill-ratio", type=float, default=0.9)
    ap.add_argument("--target-decode-ratio", type=float, default=0.9)
    ap.add_argument("--target-memory-ratio", type=float, default=1.1)
    ap.add_argument("--max-memory-ratio", type=float, default=1.1)
    ap.add_argument("--min-regression-decode-ratio", type=float, default=0.6)
    ap.add_argument("--min-top5-match", type=float, default=0.9)
    ap.add_argument("--min-argmax-match", type=float, default=1.0)
    ap.add_argument("--min-cosine", type=float, default=0.9999)
    ap.add_argument("--min-greedy-ratio", type=float, default=1.0)
    ap.add_argument("--required-batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--min-batch-fast-speedup", type=float, default=1.25)
    ap.add_argument("--min-dynamic-fast-speedup", type=float, default=1.5)
    ap.add_argument("--max-chunked-prefill-diff", type=float, default=0.2)
    ap.add_argument("--min-micro-fast-tokps", type=float, default=50.0)
    ap.add_argument("--min-micro-fast-speedup", type=float, default=1.25)
    ap.add_argument("--min-forward-fast-speedup", type=float, default=3.0)
    ap.add_argument("--min-forward-fast-vs-direct-ratio", type=float, default=0.9)
    ap.add_argument("--max-forward-fast-diff", type=float, default=0.2)
    ap.add_argument("--min-generate-fast-speedup", type=float, default=2.0)
    ap.add_argument("--min-generate-batch-size", type=int, default=2)
    ap.add_argument("--expected-top-component", default="attn_linears_lora")
    ap.add_argument("--expect-naive-candidate-slower", action="store_true", default=True)
    ap.add_argument("--require-quantization", action="store_true",
                    help="Require passing 8bit/4bit quantization benchmark rows")
    ap.add_argument("--min-quant-decode-ratio", type=float, default=1.0)
    ap.add_argument("--max-8bit-memory-ratio", type=float, default=0.85)
    ap.add_argument("--max-4bit-memory-ratio", type=float, default=0.65)
    args = ap.parse_args()

    report = run_analyzer(args)
    failures: list[str] = []
    check_present(report, failures)
    check_common(report, failures, args)
    if args.target:
        check_target(report, failures, args)
    else:
        check_regression(report, failures, args)

    output = {
        "mode": "target" if args.target else "regression",
        "device": args.device,
        "dtype": args.dtype,
        "ok": not failures,
        "failures": failures,
        "summary": {
            "decode_ratio": (report.get("speed_mem") or {}).get("decode_ratio"),
            "fast_decode_ratio": (report.get("fast_decode") or {}).get("decode_ratio"),
            "memory_ratio": (report.get("speed_mem") or {}).get("memory_ratio"),
            "next_focus": report.get("next_focus"),
        },
    }
    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(output, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
