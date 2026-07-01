#!/usr/bin/env python3
# coding=utf-8
"""Run or ingest Albatross RWKV-7 benchmark rows.

Albatross is the external high-performance RWKV inference-engine baseline.  This
helper keeps the comparison reproducible by turning its standard

``RESULT B=... T=... iters=... p10_ms=... p50_ms=... p90_ms=... tok_s_p50=...``

lines into the repository's ``bench/results.jsonl`` schema.  It can either run a
local Albatross checkout or parse a saved log with ``--parse-log``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CASES = "1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,2x2,4x4,8x8,16x16"
RESULT_RE = re.compile(
    r"RESULT\s+B=(?P<batch_size>\d+)\s+T=(?P<tokens_per_sequence>\d+)\s+"
    r"iters=(?P<iters>\d+)\s+"
    r"p10_ms=(?P<latency_p10_ms>[0-9.]+)\s+"
    r"p50_ms=(?P<latency_p50_ms>[0-9.]+)\s+"
    r"p90_ms=(?P<latency_p90_ms>[0-9.]+)\s+"
    r"tok_s_p50=(?P<tokps_p50>[0-9.]+)"
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_cuda_device_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def parse_result_lines(
    text: str,
    *,
    engine: str,
    dtype: str,
    device: str | None,
    model_path: str | None = None,
    model_size_label: str | None = None,
    checkpoint_sha256: str | None = None,
    engine_config: str | None = None,
    peak_vram_mb: float | None = None,
    command: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        data = match.groupdict()
        batch_size = int(data["batch_size"])
        tokens_per_sequence = int(data["tokens_per_sequence"])
        latency_p50_ms = float(data["latency_p50_ms"])
        tokps_p50 = float(data["tokps_p50"])
        row: dict[str, Any] = {
            "axis": "albatross_speed",
            "backend": "albatross",
            "engine": engine,
            "engine_config": engine_config,
            "dtype": dtype,
            "device": device,
            "model_path": model_path,
            "model_size_label": model_size_label,
            "checkpoint_sha256": checkpoint_sha256,
            "batch_size": batch_size,
            "tokens_per_sequence": tokens_per_sequence,
            "tokens_total": batch_size * tokens_per_sequence,
            "iters": int(data["iters"]),
            "latency_p10_ms": float(data["latency_p10_ms"]),
            "latency_p50_ms": latency_p50_ms,
            "latency_p90_ms": float(data["latency_p90_ms"]),
            "tokps_p50": tokps_p50,
            "ms_per_token_p50": round(1000.0 / tokps_p50, 6) if tokps_p50 > 0 else None,
            "peak_vram_mb": peak_vram_mb,
            "command": command,
            "status": "pass",
        }
        # Keep rows compact when optional fields are unknown.
        rows.append({k: v for k, v in row.items() if v is not None})
    return rows


def build_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    albatross_dir = Path(args.albatross_dir).expanduser()
    extra_args = list(args.albatross_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    if args.engine == "faster4_cpp":
        binary = Path(args.binary).expanduser() if args.binary else albatross_dir / "faster4_2605_cpp" / "bin" / "rwkv7_fast_v4"
        cmd = [
            str(binary),
            "--model",
            args.model,
            "--model-forward",
            "--cases",
            args.cases,
            "--graph-bench",
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ]
        return cmd + extra_args, binary.parent

    script = Path(args.binary).expanduser() if args.binary else albatross_dir / "faster3a_2605" / "rwkv7_fast_v3a.py"
    cmd = [
        sys.executable,
        str(script),
        "--model",
        args.model,
        "--cases",
        args.cases,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    return cmd + extra_args, script.parent


def read_parse_log(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--albatross-dir", default=os.environ.get("ALBATROSS_DIR", "/home/data/wangyue/projects/Albatross"))
    ap.add_argument("--engine", choices=["faster4_cpp", "faster3a"], default="faster4_cpp")
    ap.add_argument("--engine-config", default=None, help="Free-form Albatross config label, e.g. wkv=fp32io16")
    ap.add_argument("--binary", default=None, help="Override Albatross binary/script path")
    ap.add_argument("--model", required=True, help="Original .pth checkpoint path")
    ap.add_argument("--model-size-label", default=None)
    ap.add_argument("--checkpoint-sha256", default=None, help="Override checkpoint SHA256 when parsing logs off-host")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device-name", default=None)
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--peak-vram-mb", type=float, default=None, help="Optional externally measured peak VRAM")
    ap.add_argument("--parse-log", default=None, help="Parse an existing Albatross log instead of running it; use '-' for stdin")
    ap.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    ap.add_argument("albatross_args", nargs=argparse.REMAINDER, help="Extra args after '--' are passed to Albatross")
    args = ap.parse_args()

    command: list[str] | None = None
    if args.parse_log:
        output = read_parse_log(args.parse_log)
    else:
        command, cwd = build_command(args)
        if args.dry_run:
            print(" ".join(command))
            return 0
        completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
        output = completed.stdout + completed.stderr
        print(output, end="")
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    model_path = str(Path(args.model).expanduser())
    rows = parse_result_lines(
        output,
        engine=args.engine,
        dtype=args.dtype,
        device=args.device_name or detect_cuda_device_name(),
        model_path=model_path,
        model_size_label=args.model_size_label,
        checkpoint_sha256=args.checkpoint_sha256 or sha256_file(Path(model_path)),
        engine_config=args.engine_config,
        peak_vram_mb=args.peak_vram_mb,
        command=command,
    )
    if not rows:
        raise SystemExit("no Albatross RESULT rows found")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.results:
        append_jsonl(Path(args.results), rows)
        print(f"\nappended {len(rows)} rows -> {args.results}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
