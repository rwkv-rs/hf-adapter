#!/usr/bin/env python3
# coding=utf-8
"""One-command final MATH500 acceptance runner.

This orchestrates the benchmark shape requested for the RWKV-7 HF adapter:

1. run a short Albatross-style dynamic-batching bsz sweep;
2. pick the fastest bsz by generation token/s;
3. run the full MATH500 avg@64 evaluation with that bsz;
4. optionally compare the full summary against an Albatross summary/log;
5. optionally run uncheatable teacher-forced compression/logit alignment.

It intentionally wraps the lower-level scripts instead of replacing them so the
individual artifacts remain easy to audit and reproduce.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_int_list(raw: str) -> list[int]:
    out = []
    for part in raw.replace(",", " ").split():
        if part.strip():
            out.append(int(part))
    if not out:
        raise ValueError("empty integer list")
    return out


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} is not a JSON object")
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_logged(cmd: list[str], log_path: Path, *, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def eval_math500_cmd(args: argparse.Namespace, *, out_dir: Path, bsz: int, limit: int, rollout: int, max_new_tokens: int) -> list[str]:
    cmd = [
        sys.executable,
        "bench/eval_math500_hf.py",
        "--hf-dir",
        args.hf_dir,
        "--dataset",
        args.dataset,
        "--out-dir",
        str(out_dir),
        "--rollout",
        str(rollout),
        "--limit",
        str(limit),
        "--max-new-tokens",
        str(max_new_tokens),
        "--ctx-limit",
        str(args.ctx_limit),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--prompt-style",
        args.prompt_style,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--progress-every",
        str(args.progress_every),
        "--dynamic-batching",
        "--bsz",
        str(bsz),
        "--prefill-backend",
        args.prefill_backend,
        "--decode-backend",
        args.decode_backend,
        "--rng-mode",
        args.rng_mode,
        "--summary-speed-timing",
        args.summary_speed_timing,
    ]
    if args.add_bos:
        cmd.append("--add-bos")
    if args.defer_verification:
        cmd.extend(["--defer-verification", "--verify-workers", str(args.verify_workers)])
    if args.defer_text_decode:
        cmd.append("--defer-text-decode")
    return cmd


def summarize_bsz_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        requested_bsz = int(row.get("config", {}).get("bsz") or row.get("dynamic_bsz") or 0)
        effective_bsz = int(row.get("dynamic_bsz") or requested_bsz)
        compact.append(
            {
                "bsz": requested_bsz,
                "effective_bsz": effective_bsz,
                "generation_token_per_sec": float(row.get("generation_token_per_sec", row.get("token_per_sec", 0.0))),
                "token_per_sec": float(row.get("token_per_sec", 0.0)),
                "decode_sec": float(row.get("decode_sec", 0.0)),
                "decoded_token_events": int(row.get("decoded_token_events", 0)),
                "prefill_sec": float(row.get("prefill_sec", 0.0)),
                "verification_sec": float(row.get("verification_sec", 0.0) or 0.0),
                "pass_at_rollout_accuracy": float(row.get("pass_at_rollout_accuracy", 0.0)),
                "correct_generations": int(row.get("correct_generations", 0)),
                "summary_path": row.get("summary_path"),
            }
        )
    compact.sort(key=lambda r: r["generation_token_per_sec"], reverse=True)
    return compact


def run_bsz_sweep(args: argparse.Namespace, out_root: Path, env: dict[str, str]) -> tuple[int, list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    for bsz in parse_int_list(args.bsz_list):
        subdir = out_root / "bsz_sweep" / f"bsz_{bsz}"
        cmd = eval_math500_cmd(
            args,
            out_dir=subdir,
            bsz=bsz,
            limit=args.sweep_limit,
            rollout=args.sweep_rollout,
            max_new_tokens=args.sweep_max_new_tokens,
        )
        run_logged(cmd, subdir / "run.log", env=env)
        summary = load_json(subdir / "summary.json")
        summary["summary_path"] = str(subdir / "summary.json")
        summaries.append(summary)
    compact = summarize_bsz_sweep(summaries)
    write_json(out_root / "bsz_sweep_summary.json", compact)
    best_bsz = int(compact[0]["bsz"])
    return best_bsz, compact


def run_comparison(args: argparse.Namespace, out_root: Path, full_summary: Path, env: dict[str, str]) -> dict[str, Any] | None:
    if not args.albatross_summary:
        return None
    comparison_dir = out_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "bench/compare_math500_summaries.py",
        "--hf-summary",
        str(full_summary),
        "--albatross-summary",
        args.albatross_summary,
        "--require-compatible-shape",
        "--min-pass-at-rollout",
        str(args.min_pass_at_rollout),
        "--min-summary-speed-ratio",
        str(args.min_summary_speed_ratio),
        "--json-output",
        str(comparison_dir / "comparison.json"),
        "--text-output",
        str(comparison_dir / "comparison.txt"),
    ]
    if args.albatross_log:
        cmd.extend(["--albatross-log", args.albatross_log, "--min-decode-speed-ratio", str(args.min_decode_speed_ratio)])
    if args.fail_on_gate:
        cmd.append("--fail-on-gate")
    run_logged(cmd, comparison_dir / "compare.log", env=env)
    return load_json(comparison_dir / "comparison.json")


def run_compression(args: argparse.Namespace, out_root: Path, env: dict[str, str]) -> dict[str, Any] | None:
    if args.skip_compression:
        return None
    comp_dir = out_root / "compression_alignment"
    comp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "bench/bench_logit_compression_alignment.py",
        "--dataset",
        args.dataset,
        "--tokenizer-dir",
        args.tokenizer_dir or args.hf_dir,
        "--candidate-kind",
        args.compression_candidate_kind,
        "--candidate-hf-dir",
        args.compression_candidate_hf_dir or args.hf_dir,
        "--candidate-dtype",
        args.compression_candidate_dtype,
        "--candidate-quantization",
        args.compression_candidate_quantization,
        "--candidate-quant-policy",
        args.compression_candidate_quant_policy,
        "--candidate-quant-min-params",
        str(args.compression_candidate_quant_min_params),
        "--reference-kind",
        args.compression_reference_kind,
        "--limit",
        str(args.compression_limit),
        "--max-tokens-per-text",
        str(args.compression_max_tokens_per_text),
        "--device",
        args.device,
        "--progress-every",
        str(args.compression_progress_every),
        "--out-json",
        str(comp_dir / "compression_alignment.json"),
        "--out-md",
        str(comp_dir / "compression_alignment.md"),
    ]
    if args.add_bos:
        cmd.append("--add-bos")
    if args.compression_reference_kind == "hf":
        cmd.extend(
            [
                "--reference-hf-dir",
                args.compression_reference_hf_dir or args.hf_dir,
                "--reference-dtype",
                args.compression_reference_dtype,
                "--reference-quantization",
                args.compression_reference_quantization,
                "--reference-quant-policy",
                args.compression_reference_quant_policy,
                "--reference-quant-min-params",
                str(args.compression_reference_quant_min_params),
            ]
        )
    else:
        cmd.extend(
            [
                "--reference-albatross-dir",
                args.compression_reference_albatross_dir,
                "--reference-albatross-model",
                args.compression_reference_albatross_model,
                "--reference-albatross-module",
                args.compression_reference_albatross_module,
                "--reference-albatross-wkv",
                args.compression_reference_albatross_wkv,
                "--reference-albatross-emb",
                args.compression_reference_albatross_emb,
                "--reference-albatross-batched-rkv",
                args.compression_reference_albatross_batched_rkv,
                "--reference-albatross-cmix-sparse",
                args.compression_reference_albatross_cmix_sparse,
                "--reference-albatross-lowrank-weight",
                args.compression_reference_albatross_lowrank_weight,
                "--reference-albatross-orig-linear-groups",
                args.compression_reference_albatross_orig_linear_groups,
            ]
        )
        if args.compression_reference_chdir_albatross:
            cmd.append("--reference-chdir-albatross")
    run_logged(cmd, comp_dir / "run.log", env=env)
    return load_json(comp_dir / "compression_alignment.json")


def write_readme(out_root: Path, manifest: dict[str, Any]) -> None:
    bsz_rows = manifest.get("bsz_sweep", [])
    full = manifest.get("full_summary") or {}
    comparison = manifest.get("comparison") or {}
    compression = manifest.get("compression_alignment") or {}
    lines: list[str] = []
    lines.append("# MATH500 final acceptance benchmark")
    lines.append("")
    lines.append("This artifact follows the BlinkDL/Albatross MATH500 avg@64 evaluation shape and adds best-bsz speed selection plus uncheatable teacher-forced compression/logit alignment.")
    lines.append("")
    lines.append("## Best-bsz sweep")
    lines.append("")
    lines.append("| rank | requested bsz | effective bsz | generation tok/s | decode sec | decoded tokens | pass@rollout |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for i, row in enumerate(bsz_rows, 1):
        lines.append(
            f"| `{i}` | `{row.get('bsz')}` | `{row.get('effective_bsz', row.get('bsz'))}` | "
            f"`{float(row.get('generation_token_per_sec', 0.0)):.3f}` | "
            f"`{float(row.get('decode_sec', 0.0)):.3f}` | `{row.get('decoded_token_events')}` | "
            f"`{float(row.get('pass_at_rollout_accuracy', 0.0)):.6f}` |"
        )
    lines.append("")
    lines.append(f"Selected bsz: `{manifest.get('selected_bsz')}`.")
    lines.append("")
    if full:
        lines.append("## Full avg@64 summary")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        for key in (
            "num_tasks",
            "rollout",
            "total_generations",
            "correct_generations",
            "rollout_accuracy",
            "pass_at_rollout_accuracy",
            "generation_token_per_sec",
            "wall_token_per_sec",
            "decode_sec",
            "decoded_token_events",
        ):
            if key in full:
                lines.append(f"| `{key}` | `{full[key]}` |")
        lines.append("")
    if comparison:
        gates = comparison.get("gates", {})
        lines.append("## HF vs Albatross comparison")
        lines.append("")
        lines.append(f"Overall gate: `{'PASS' if gates.get('overall_pass') else 'FAIL'}`.")
        lines.append("")
    if compression:
        comp = compression.get("comparison", {})
        lines.append("## Uncheatable compression alignment")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| reference bits/token | `{_fmt(comp.get('reference_bits_per_token'))}` |")
        lines.append(f"| candidate bits/token | `{_fmt(comp.get('candidate_bits_per_token'))}` |")
        lines.append(f"| candidate/reference bits ratio | `{_fmt(comp.get('candidate_over_reference_bits_ratio'))}` |")
        lines.append(f"| tokens scored | `{comp.get('tokens')}` |")
        lines.append("")
        lines.append("See `compression_alignment/compression_alignment.md` for ratio vs token position.")
        lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `manifest.json`: top-level machine-readable manifest.")
    lines.append("- `bsz_sweep_summary.json`: sorted best-bsz speed rows.")
    lines.append("- `full_avg64/summary.json`: full MATH500 result when enabled.")
    lines.append("- `comparison/comparison.json`: HF-vs-Albatross gates when an Albatross summary is provided.")
    lines.append("- `compression_alignment/compression_alignment.json`: external-token compression/NLL report.")
    out_root.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer-dir", default="")
    ap.add_argument("--bsz-list", default="32 64 96 128 192")
    ap.add_argument("--sweep-limit", type=int, default=4)
    ap.add_argument("--sweep-rollout", type=int, default=64)
    ap.add_argument("--sweep-max-new-tokens", type=int, default=256)
    ap.add_argument("--skip-bsz-sweep", action="store_true")
    ap.add_argument("--fixed-bsz", type=int, default=128)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--full-limit", type=int, default=0)
    ap.add_argument("--full-rollout", type=int, default=64)
    ap.add_argument("--full-max-new-tokens", type=int, default=1500)
    ap.add_argument("--ctx-limit", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.28)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--prompt-style", choices=("fake_think", "plain"), default="fake_think")
    ap.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--progress-every", type=int, default=5000)
    ap.add_argument("--add-bos", action="store_true", default=True)
    ap.add_argument("--no-add-bos", dest="add_bos", action="store_false")
    ap.add_argument("--prefill-backend", choices=("native", "forward"), default="native")
    ap.add_argument("--decode-backend", choices=("fast_token", "forward"), default="fast_token")
    ap.add_argument("--rng-mode", choices=("global", "active_global", "per_sample"), default="global")
    ap.add_argument("--defer-verification", action="store_true", default=True)
    ap.add_argument("--no-defer-verification", dest="defer_verification", action="store_false")
    ap.add_argument("--verify-workers", type=int, default=4)
    ap.add_argument("--summary-speed-timing", choices=("wall", "generation"), default="generation")
    ap.add_argument("--defer-text-decode", action="store_true", default=True)
    ap.add_argument("--no-defer-text-decode", dest="defer_text_decode", action="store_false")
    ap.add_argument("--albatross-summary", default="")
    ap.add_argument("--albatross-log", default="")
    ap.add_argument("--min-pass-at-rollout", type=float, default=0.370)
    ap.add_argument("--min-summary-speed-ratio", type=float, default=2.0)
    ap.add_argument("--min-decode-speed-ratio", type=float, default=2.0)
    ap.add_argument("--fail-on-gate", action="store_true", default=True)
    ap.add_argument("--no-fail-on-gate", dest="fail_on_gate", action="store_false")

    ap.add_argument("--skip-compression", action="store_true")
    ap.add_argument("--compression-limit", type=int, default=128)
    ap.add_argument("--compression-max-tokens-per-text", type=int, default=1024)
    ap.add_argument("--compression-progress-every", type=int, default=25)
    ap.add_argument("--compression-reference-kind", choices=("hf", "albatross"), default="hf")
    ap.add_argument("--compression-reference-hf-dir", default="")
    ap.add_argument("--compression-reference-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--compression-reference-quantization", choices=("none", "mm8", "mm4"), default="none")
    ap.add_argument("--compression-reference-quant-policy", choices=("memory", "balanced", "speed", "dense"), default="speed")
    ap.add_argument("--compression-reference-quant-min-params", type=int, default=8_000_000)
    ap.add_argument("--compression-reference-albatross-dir", default="")
    ap.add_argument("--compression-reference-albatross-model", default="")
    ap.add_argument("--compression-reference-albatross-module", default="rwkv7_fast_v3a")
    ap.add_argument("--compression-reference-albatross-wkv", default="fp32io16")
    ap.add_argument("--compression-reference-albatross-emb", default="cpu")
    ap.add_argument("--compression-reference-albatross-batched-rkv", default="off")
    ap.add_argument("--compression-reference-albatross-cmix-sparse", default="no-fc")
    ap.add_argument("--compression-reference-albatross-lowrank-weight", default="both")
    ap.add_argument("--compression-reference-albatross-orig-linear-groups", default="att_c2c,ffn_key,head")
    ap.add_argument("--compression-reference-chdir-albatross", action="store_true")
    ap.add_argument("--compression-candidate-kind", choices=("hf", "albatross"), default="hf")
    ap.add_argument("--compression-candidate-hf-dir", default="")
    ap.add_argument("--compression-candidate-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--compression-candidate-quantization", choices=("none", "mm8", "mm4"), default="none")
    ap.add_argument("--compression-candidate-quant-policy", choices=("memory", "balanced", "speed", "dense"), default="speed")
    ap.add_argument("--compression-candidate-quant-min-params", type=int, default=8_000_000)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("RWKV7_NATIVE_MODEL", "1")
    env["PYTHONPATH"] = os.getcwd() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started = time.perf_counter()

    if args.skip_bsz_sweep:
        selected_bsz = int(args.fixed_bsz)
        bsz_sweep: list[dict[str, Any]] = []
    else:
        selected_bsz, bsz_sweep = run_bsz_sweep(args, out_root, env)

    full_summary_obj: dict[str, Any] | None = None
    full_summary_path = out_root / "full_avg64" / "summary.json"
    if not args.skip_full:
        full_dir = out_root / "full_avg64"
        cmd = eval_math500_cmd(
            args,
            out_dir=full_dir,
            bsz=selected_bsz,
            limit=args.full_limit,
            rollout=args.full_rollout,
            max_new_tokens=args.full_max_new_tokens,
        )
        run_logged(cmd, full_dir / "run.log", env=env)
        full_summary_obj = load_json(full_summary_path)

    comparison_obj = None
    if full_summary_obj is not None:
        comparison_obj = run_comparison(args, out_root, full_summary_path, env)

    compression_obj = run_compression(args, out_root, env)

    manifest = {
        "axis": "math500_final_acceptance",
        "status": "pass",
        "hf_dir": args.hf_dir,
        "dataset": args.dataset,
        "selected_bsz": selected_bsz,
        "bsz_sweep": bsz_sweep,
        "full_summary": full_summary_obj,
        "comparison": comparison_obj,
        "compression_alignment": compression_obj,
        "elapsed_sec": time.perf_counter() - started,
        "config": vars(args),
    }
    write_json(out_root / "manifest.json", manifest)
    write_readme(out_root, manifest)
    print("MATH500_FINAL_ACCEPTANCE_RESULT " + json.dumps({"selected_bsz": selected_bsz, "out_dir": str(out_root)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
