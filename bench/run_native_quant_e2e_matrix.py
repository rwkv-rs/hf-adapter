#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BaseCase:
    model_label: str
    hf_dir: str
    batch_size: int
    prompt_tokens: int
    decode_tokens: int


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=/absolute/checkpoint/path")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("model must include a non-empty label and path")
    return label, path


def expanded_cases(models: Iterable[tuple[str, str]]) -> list[BaseCase]:
    shapes = [
        *((batch, 128, 128) for batch in (1, 2, 4, 8)),
        (1, 512, 128),
        (1, 2048, 128),
        (1, 128, 512),
    ]
    return [
        BaseCase(label, hf_dir, batch, prompt, decode)
        for label, hf_dir in models
        for batch, prompt, decode in shapes
    ]


def cartesian_cases(args, models: Iterable[tuple[str, str]]) -> list[BaseCase]:
    return [
        BaseCase(label, hf_dir, batch, prompt, decode)
        for label, hf_dir in models
        for batch in args.batch_sizes
        for prompt in args.prompt_tokens
        for decode in args.decode_tokens
    ]


def case_key(case: BaseCase, quantization: str, fusion_mode: str) -> tuple:
    return (
        case.model_label,
        case.batch_size,
        case.prompt_tokens,
        case.decode_tokens,
        quantization,
        fusion_mode,
    )


def row_key(row: dict) -> tuple | None:
    if row.get("axis") != "native_quant_e2e_decode" or row.get("status") != "pass":
        return None
    return (
        str(row.get("model_size_label", "")),
        int(row.get("batch_size", 0)),
        int(row.get("prompt_tokens", 0)),
        int(row.get("decode_tokens", 0)),
        str(row.get("quantization", "")),
        (
            "deep"
            if row.get("fused_quant_ffn_down_add", False)
            else "up"
            if row.get("fused_quant_ffn", False)
            else "off"
        ),
    )


def read_completed(path: Path) -> set[tuple]:
    completed: set[tuple] = set()
    if not path.exists():
        return completed
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            key = row_key(json.loads(raw))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        if key is not None:
            completed.add(key)
    return completed


def slug(case: BaseCase, quantization: str, fusion_mode: str) -> str:
    raw = "-".join(
        [
            case.model_label,
            f"b{case.batch_size}",
            f"p{case.prompt_tokens}",
            f"d{case.decode_tokens}",
            quantization,
            fusion_mode,
        ]
    )
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in raw)
    return f"{safe}-{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


def baseline_key(case: BaseCase, args) -> str:
    return "-".join(
        [
            case.model_label,
            f"b{case.batch_size}",
            f"p{case.prompt_tokens}",
            f"d{case.decode_tokens}",
            args.dtype,
            args.fast_token_backend,
            args.policy,
            f"min{args.min_params}",
            f"warmup{args.warmup}",
            f"repeats{args.timing_repeats}",
        ]
    )


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_one(args, case: BaseCase, quantization: str, fusion_mode: str, results: Path) -> bool:
    name = slug(case, quantization, fusion_mode)
    case_dir = args.output_dir / "cases"
    log_dir = args.output_dir / "logs"
    case_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    case_result = case_dir / f"{name}.jsonl"
    log_path = log_dir / f"{name}.log"
    case_result.unlink(missing_ok=True)

    command = [
        args.python,
        str(args.benchmark_script),
        "--hf-dir",
        case.hf_dir,
        "--code-source",
        "repo",
        "--model-size-label",
        case.model_label,
        "--dtype",
        args.dtype,
        "--device",
        "cuda",
        "--fast-token-backend",
        args.fast_token_backend,
        "--single-quantization",
        quantization,
        "--min-params",
        str(args.min_params),
        "--policy",
        args.policy,
        "--batch-size",
        str(case.batch_size),
        "--prompt-tokens",
        str(case.prompt_tokens),
        "--decode-tokens",
        str(case.decode_tokens),
        "--warmup",
        str(args.warmup),
        "--timing-repeats",
        str(args.timing_repeats),
        "--baseline-dir",
        str(args.output_dir / "baselines"),
        "--baseline-key",
        baseline_key(case, args),
        "--results",
        str(case_result),
    ]
    if fusion_mode in {"up", "deep"}:
        command.append("--fused-quant-ffn")
    if fusion_mode == "deep":
        command.append("--fused-quant-ffn-down-add")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    repo = str(args.repo_root)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command=" + subprocess.list2cmdline(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=args.repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if completed.returncode == 0 and case_result.exists():
        rows = [json.loads(line) for line in case_result.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) == 1:
            append_jsonl(results, rows[0])
            return True

    failure = {
        "axis": "native_quant_e2e_matrix_attempt",
        "status": "fail",
        **asdict(case),
        "quantization": quantization,
        "fusion_mode": fusion_mode,
        "returncode": completed.returncode,
        "log": str(log_path),
    }
    append_jsonl(results, failure)
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", type=parse_model, required=True)
    ap.add_argument("--profile", choices=["expanded", "cartesian"], default="expanded")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--prompt-tokens", nargs="+", type=int, default=[128, 512, 2048])
    ap.add_argument("--decode-tokens", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--quantizations", nargs="+", choices=["mm8", "mm4"], default=["mm8", "mm4"])
    ap.add_argument("--fused-modes", nargs="+", choices=["off", "up", "deep"], default=["off", "up", "deep"])
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--fast-token-backend", choices=["native_jit", "native_graph"], default="native_graph")
    ap.add_argument("--policy", choices=["memory", "speed"], default="memory")
    ap.add_argument("--min-params", type=int, default=8_000_000)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timing-repeats", type=int, default=3)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--results", type=Path)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--benchmark-script", type=Path, default=repo_root / "bench" / "bench_native_quant_e2e_decode.py")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--list-cases", action="store_true")
    args = ap.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        ap.error("require 0 <= --shard-index < --shard-count")
    args.repo_root = repo_root
    args.output_dir = args.output_dir.resolve()
    results = (args.results or args.output_dir / f"results-shard-{args.shard_index}.jsonl").resolve()

    bases = expanded_cases(args.model) if args.profile == "expanded" else cartesian_cases(args, args.model)
    bases = [case for index, case in enumerate(bases) if index % args.shard_count == args.shard_index]
    planned = [
        (case, quantization, fusion_mode)
        for case in bases
        for quantization in ["none", *args.quantizations]
        for fusion_mode in (
            ["off"]
            if quantization == "none"
            else [mode for mode in args.fused_modes if mode != "deep" or quantization == "mm8"]
        )
    ]
    if args.list_cases:
        print(json.dumps({"base_cases": len(bases), "runs": len(planned)}, sort_keys=True))
        for case, quantization, fusion_mode in planned:
            print(json.dumps({**asdict(case), "quantization": quantization, "fusion_mode": fusion_mode}, sort_keys=True))
        return 0

    completed = read_completed(results) if args.skip_existing else set()
    failures = 0
    for index, (case, quantization, fusion_mode) in enumerate(planned, start=1):
        key = case_key(case, quantization, fusion_mode)
        if key in completed:
            print(f"[{index}/{len(planned)}] skip {slug(case, quantization, fusion_mode)}", flush=True)
            continue
        print(f"[{index}/{len(planned)}] run {slug(case, quantization, fusion_mode)}", flush=True)
        if not run_one(args, case, quantization, fusion_mode, results):
            failures += 1
    print(json.dumps({"planned": len(planned), "failures": failures, "results": str(results)}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
