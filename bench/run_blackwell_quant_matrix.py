#!/usr/bin/env python3
# coding=utf-8
"""Fresh-process Blackwell native quant benchmark matrix.

This is a thin orchestration layer around ``bench_native_quant_e2e_decode.py``.
Each ``model x prompt x decode x batch x quantization`` row is run in a fresh
Python subprocess, which avoids the 7B-on-32GB fragmentation/OOM pattern seen
when fp16/mm8/mm4 are loaded sequentially in one process.  The fp16 subprocess
saves baseline logits/tokps to ``--baseline-dir``; later mm8/mm4 subprocesses
load that baseline and emit footprint, speed, logits-cosine, and same-token
ratios against the matching fp16 row.

Example:

  python bench/run_blackwell_quant_matrix.py \\
    --model 1.5b=/workspace/models/rwkv7_g1g_15b_hf \\
    --model 2.9b=/workspace/models/rwkv7_g1g_29b_hf \\
    --model 7.2b=/workspace/models/rwkv7_g1g_72b_hf \\
    --prompt-tokens 128 512 2048 \\
    --decode-tokens 128 512 \\
    --batch-sizes 1 2 4 8 \\
    --results bench/_runs/results_5090_blackwell_quant_matrix.jsonl \\
    --baseline-dir bench/_runs/blackwell_baselines

For boundary probes where fp16 may OOM but W4 could still fit, pass
``--allow-missing-baseline`` so quant-only rows record null fp16 ratios instead
of failing before the memory-saving evidence is captured.
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
from typing import Iterable


def parse_model_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be LABEL=/path/to/hf_dir")
    label, path = value.split("=", 1)
    label = label.strip().lower()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("--model must be LABEL=/path/to/hf_dir")
    return label, path


def existing_keys(results: Path) -> set[tuple[str, str, int, int, int, str]]:
    if not results.exists():
        return set()
    keys: set[tuple[str, str, int, int, int, str]] = set()
    for line in results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("status") not in {"pass", "fail", "skip"}:
            continue
        label = str(row.get("model_size_label") or "").lower()
        quant = str(row.get("quantization") or "")
        try:
            keys.add(
                (
                    label,
                    quant,
                    int(row.get("prompt_tokens")),
                    int(row.get("decode_tokens")),
                    int(row.get("batch_size")),
                    str(row.get("native_mm_policy") or ""),
                )
            )
        except Exception:
            continue
    return keys


def append_failure(
    path: Path,
    *,
    label: str,
    hf_dir: str,
    quantization: str,
    prompt_tokens: int,
    decode_tokens: int,
    batch_size: int,
    policy: str,
    cmd: list[str],
    proc: subprocess.CompletedProcess[str],
) -> None:
    row = {
        "axis": "native_quant_e2e_decode",
        "benchmark_matrix": "blackwell_fresh_process",
        "backend": "hf_adapter",
        "status": "fail",
        "model_size_label": label,
        "hf_model_dir": hf_dir,
        "quantization": quantization,
        "dtype": "fp16",
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "batch_size": batch_size,
        "native_mm_policy": policy,
        "returncode": proc.returncode,
        "error_tail": (proc.stderr or proc.stdout or "")[-4000:],
        "cmd": " ".join(shlex.quote(part) for part in cmd),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def quant_order(values: Iterable[str]) -> list[str]:
    order = []
    for q in ["none", *values]:
        if q not in order:
            order.append(q)
    return order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", type=parse_model_spec, required=True)
    ap.add_argument("--prompt-tokens", nargs="+", type=int, default=[128, 512, 2048])
    ap.add_argument("--decode-tokens", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--quantizations", nargs="+", choices=["none", "mm8", "mm4"], default=["none", "mm8", "mm4"])
    ap.add_argument("--min-params", type=int, default=8_000_000)
    ap.add_argument("--policy", choices=["memory", "speed"], default="speed")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--timing-repeats", type=int, default=1)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fast-token-backend", choices=["auto", "fla", "native_jit", "native_graph"], default="native_graph")
    ap.add_argument("--attn-mode", choices=["chunk", "fused_recurrent"], default="fused_recurrent")
    ap.add_argument("--results", default=str(Path(__file__).parent / "_runs/results_5090_blackwell_quant_matrix.jsonl"))
    ap.add_argument("--baseline-dir", default=str(Path(__file__).parent / "_runs/blackwell_quant_baselines"))
    ap.add_argument("--bench-script", default=str(Path(__file__).with_name("bench_native_quant_e2e_decode.py")))
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--allow-missing-baseline", action="store_true")
    ap.add_argument(
        "--paired-baseline",
        action="store_true",
        help="For quantized rows, benchmark dense and quantized paths in the same fresh process",
    )
    ap.add_argument("--max-runs", type=int, default=0, help="Debug helper: stop after N subprocess rows")
    args = ap.parse_args()

    results = Path(args.results)
    seen = existing_keys(results) if args.skip_existing else set()
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("RWKV7_NATIVE_MODEL", "1")
    env.setdefault("RWKV7_NATIVE_MODEL_JIT", "1")
    env.setdefault("RWKV_V7_ON", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    runs = 0
    failures = 0
    for label, hf_dir in args.model:
        for prompt in args.prompt_tokens:
            for decode in args.decode_tokens:
                for bsz in args.batch_sizes:
                    baseline_key = f"{label}_prompt{prompt}_decode{decode}_bsz{bsz}_{args.dtype}_{args.policy}"
                    for quant in quant_order(args.quantizations):
                        key = (label, quant, prompt, decode, bsz, args.policy)
                        if key in seen:
                            print(f"skip existing {key}", flush=True)
                            continue
                        cmd = [
                            sys.executable,
                            args.bench_script,
                            "--hf-dir",
                            hf_dir,
                            "--model-size-label",
                            label,
                            "--dtype",
                            args.dtype,
                            "--device",
                            args.device,
                            "--attn-mode",
                            args.attn_mode,
                            "--fast-cache",
                            "true",
                            "--fast-token-backend",
                            args.fast_token_backend,
                            "--single-quantization",
                            quant,
                            "--min-params",
                            str(args.min_params),
                            "--policy",
                            args.policy,
                            "--batch-size",
                            str(bsz),
                            "--prompt-tokens",
                            str(prompt),
                            "--decode-tokens",
                            str(decode),
                            "--warmup",
                            str(args.warmup),
                            "--timing-repeats",
                            str(args.timing_repeats),
                            "--baseline-dir",
                            args.baseline_dir,
                            "--baseline-key",
                            baseline_key,
                            "--results",
                            str(results),
                        ]
                        if args.allow_missing_baseline:
                            cmd.append("--allow-missing-baseline")
                        if args.paired_baseline and quant != "none":
                            cmd.append("--paired-baseline")
                        print("\n$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
                        t0 = time.time()
                        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
                        print(proc.stdout, end="", flush=True)
                        if proc.returncode != 0:
                            failures += 1
                            print(proc.stderr, end="", file=sys.stderr, flush=True)
                            append_failure(
                                results,
                                label=label,
                                hf_dir=hf_dir,
                                quantization=quant,
                                prompt_tokens=prompt,
                                decode_tokens=decode,
                                batch_size=bsz,
                                policy=args.policy,
                                cmd=cmd,
                                proc=proc,
                            )
                            if args.fail_fast:
                                return proc.returncode
                        runs += 1
                        print(f"row wall_s={time.time() - t0:.1f} failures={failures}", flush=True)
                        if args.max_runs and runs >= args.max_runs:
                            return 1 if failures else 0
    return 1 if failures and args.fail_fast else 0


if __name__ == "__main__":
    raise SystemExit(main())
