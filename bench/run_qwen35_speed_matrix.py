#!/usr/bin/env python3
"""Run a resumable fresh-process RWKV-7 vs Qwen3.5 HF speed matrix."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CELL_FIELDS = (
    "model_pair",
    "prompt_tokens",
    "decode_tokens",
    "batch_size",
    "dtype",
    "quantization",
)


@dataclass(frozen=True)
class PairSpec:
    label: str
    rwkv_label: str
    rwkv_model: str
    qwen_label: str
    qwen_model: str


@dataclass(frozen=True)
class MatrixConfig:
    pairs: list[PairSpec]
    prompts: list[int]
    decodes: list[int]
    batch_sizes: list[int]
    quantizations: list[str]
    dtype: str


@dataclass(frozen=True)
class RunSpec:
    model_pair: str
    model_role: str
    model_kind: str
    model_size_label: str
    model: str
    prompt_tokens: int
    decode_tokens: int
    batch_size: int
    dtype: str
    quantization: str

    @property
    def cell_key(self) -> tuple[Any, ...]:
        return (
            self.model_pair,
            self.prompt_tokens,
            self.decode_tokens,
            self.batch_size,
            self.dtype,
            self.quantization,
        )

    @property
    def raw_key(self) -> tuple[Any, ...]:
        return (*self.cell_key, self.model_role)

    def raw_key_for_backend(self, qwen_backend: str, qwen_conv_backend: str = "auto") -> tuple[Any, ...]:
        backend_key = (
            qwen_backend
            if qwen_conv_backend == "auto"
            else f"{qwen_backend}+conv:{qwen_conv_backend}"
        )
        return (*self.raw_key, backend_key)


def parse_pair_spec(value: str) -> PairSpec:
    if "=" not in value or "::" not in value:
        raise argparse.ArgumentTypeError(
            "--pair must be LABEL=RWKV_MODEL::QWEN_MODEL, for example "
            "rwkv-1.5b__qwen3.5-2b=/models/rwkv-1.5b::Qwen/Qwen3.5-2B"
        )
    label, models = value.split("=", 1)
    rwkv_model, qwen_model = models.split("::", 1)
    match = re.fullmatch(r"rwkv-(.+?)__qwen3\.5-(.+)", label.strip(), flags=re.IGNORECASE)
    if not match or not rwkv_model.strip() or not qwen_model.strip():
        raise argparse.ArgumentTypeError("pair label must match rwkv-SIZE__qwen3.5-SIZE")
    return PairSpec(
        label=label.strip().lower(),
        rwkv_label=match.group(1).lower(),
        rwkv_model=rwkv_model.strip(),
        qwen_label=match.group(2).lower(),
        qwen_model=qwen_model.strip(),
    )


def build_run_specs(config: MatrixConfig) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for pair in config.pairs:
        for prompt in config.prompts:
            for decode in config.decodes:
                for batch_size in config.batch_sizes:
                    for quantization in config.quantizations:
                        specs.extend(
                            [
                                RunSpec(
                                    model_pair=pair.label,
                                    model_role="candidate",
                                    model_kind="rwkv",
                                    model_size_label=pair.rwkv_label,
                                    model=pair.rwkv_model,
                                    prompt_tokens=prompt,
                                    decode_tokens=decode,
                                    batch_size=batch_size,
                                    dtype=config.dtype,
                                    quantization=quantization,
                                ),
                                RunSpec(
                                    model_pair=pair.label,
                                    model_role="reference",
                                    model_kind="qwen35",
                                    model_size_label=pair.qwen_label,
                                    model=pair.qwen_model,
                                    prompt_tokens=prompt,
                                    decode_tokens=decode,
                                    batch_size=batch_size,
                                    dtype=config.dtype,
                                    quantization=quantization,
                                ),
                            ]
                        )
    return specs


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    qwen_backend = str(row.get("qwen_backend_requested", "auto"))
    qwen_conv_backend = str(row.get("qwen_conv_backend_requested", "auto"))
    backend_key = (
        qwen_backend
        if qwen_conv_backend == "auto"
        else f"{qwen_backend}+conv:{qwen_conv_backend}"
    )
    return tuple(row.get(field) for field in CELL_FIELDS) + (
        row.get("model_role"),
        backend_key,
    )


def existing_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys: set[tuple[Any, ...]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("axis") != "qwen35_cross_model_speed":
            continue
        if row.get("status") in {"pass", "fail", "skip"}:
            keys.add(row_key(row))
    return keys


def worker_command(args: argparse.Namespace, spec: RunSpec) -> list[str]:
    command = [
        args.python_bin,
        args.bench_script,
        "--model",
        spec.model,
        "--model-kind",
        spec.model_kind,
        "--model-role",
        spec.model_role,
        "--model-pair",
        spec.model_pair,
        "--model-size-label",
        spec.model_size_label,
        "--benchmark-matrix",
        args.benchmark_matrix,
        "--dtype",
        spec.dtype,
        "--quantization",
        spec.quantization,
        "--native-quant-min-params",
        str(args.native_quant_min_params),
        "--native-quant-policy",
        args.native_quant_policy,
        "--torchao-group-size",
        str(args.torchao_group_size),
        "--device",
        args.device,
        "--batch-size",
        str(spec.batch_size),
        "--prompt-tokens",
        str(spec.prompt_tokens),
        "--decode-tokens",
        str(spec.decode_tokens),
        "--prefill-chunk-size",
        str(args.prefill_chunk_size),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--rwkv-attn-mode",
        args.rwkv_attn_mode,
        "--rwkv-code-source",
        args.rwkv_code_source,
        "--qwen-backend",
        args.qwen_backend,
        "--qwen-conv-backend",
        args.qwen_conv_backend,
        "--results",
        str(args.results),
    ]
    if args.require_qwen_fast_path:
        command.append("--require-qwen-fast-path")
    return command


def append_orchestrator_failure(
    path: Path,
    spec: RunSpec,
    cmd: list[str],
    proc: subprocess.CompletedProcess[str],
    *,
    qwen_backend: str = "auto",
    qwen_conv_backend: str = "auto",
    benchmark_matrix: str,
) -> None:
    row = {
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": benchmark_matrix,
        "status": "fail",
        "model_pair": spec.model_pair,
        "model_role": spec.model_role,
        "model_kind": spec.model_kind,
        "model_size_label": spec.model_size_label,
        "model_id_or_path": spec.model,
        "dtype": spec.dtype,
        "quantization": spec.quantization,
        "qwen_backend_requested": qwen_backend,
        "qwen_conv_backend_requested": qwen_conv_backend,
        "batch_size": spec.batch_size,
        "prompt_tokens": spec.prompt_tokens,
        "decode_tokens": spec.decode_tokens,
        "returncode": proc.returncode,
        "error_type": "WorkerProcessError",
        "error": (proc.stderr or proc.stdout or "")[-4000:],
        "cmd": " ".join(shlex.quote(part) for part in cmd),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_matrix(config: MatrixConfig) -> None:
    if not config.pairs:
        raise ValueError("at least one --pair is required")
    for name, values in (
        ("prompt", config.prompts),
        ("decode", config.decodes),
        ("batch", config.batch_sizes),
    ):
        if not values or any(value <= 0 for value in values):
            raise ValueError(f"{name} values must be positive")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", type=parse_pair_spec, required=True)
    ap.add_argument("--prompt-tokens", nargs="+", type=int, default=[128, 512, 2048])
    ap.add_argument("--decode-tokens", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument(
        "--quantizations",
        nargs="+",
        choices=[
            "none",
            "bnb8",
            "bnb4",
            "bnb8_a8w8_head",
            "torchao_w8",
            "torchao_w4",
            "a8w8",
            "mm8",
            "mm4",
        ],
        default=["none", "bnb8", "bnb4"],
    )
    ap.add_argument("--native-quant-min-params", type=int, default=1_000_000)
    ap.add_argument("--native-quant-policy", choices=["memory", "speed"], default="memory")
    ap.add_argument("--torchao-group-size", type=int, default=128)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--benchmark-matrix", default="qwen35_hf")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prefill-chunk-size", type=int, default=0)
    ap.add_argument("--rwkv-attn-mode", choices=["chunk", "fused_recurrent"], default="fused_recurrent")
    ap.add_argument("--rwkv-code-source", choices=["repo", "model"], default="repo")
    ap.add_argument(
        "--qwen-backend",
        choices=["auto", "fla", "torch"],
        default="fla",
        help="Require verified Qwen FLA Gated DeltaNet operators by default",
    )
    ap.add_argument(
        "--qwen-conv-backend",
        choices=["auto", "causal_conv1d", "fla_triton"],
        default="auto",
    )
    ap.add_argument("--require-qwen-fast-path", action="store_true")
    ap.add_argument(
        "--model-roles",
        nargs="+",
        choices=["candidate", "reference"],
        default=["candidate", "reference"],
        help="Run one side at a time when both checkpoints cannot coexist on local storage",
    )
    ap.add_argument("--rwkv-fast-token-backend", choices=["auto", "fla", "native_jit", "native_graph"], default="native_graph")
    ap.add_argument(
        "--rwkv-bnb8-skip-policy",
        choices=["memory", "decode_rk", "decode_hot"],
        default="memory",
    )
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--bench-script", default=str(Path(__file__).with_name("bench_cross_model_speed.py")))
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0)
    return ap.parse_args()


def build_run_environment(args: argparse.Namespace, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("RWKV_V7_ON", "1")
    env.setdefault("RWKV7_FAST_CACHE", "1")
    env["RWKV7_FAST_TOKEN_BACKEND"] = args.rwkv_fast_token_backend
    # Cross-model acceptance measures the production HF wrapper. The pure
    # PyTorch native model remains an explicitly separate experimental lane.
    env["RWKV7_NATIVE_MODEL"] = "0"
    env.setdefault("RWKV7_NATIVE_MODEL_JIT", "1")
    if os.name != "nt":
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def build_worker_environment(
    base: dict[str, str],
    spec: RunSpec,
    qwen_backend: str,
    rwkv_bnb8_skip_policy: str = "memory",
) -> dict[str, str]:
    """Isolate import-time Qwen backend selection for each fresh worker."""

    env = base.copy()
    if spec.model_kind == "qwen35" and qwen_backend == "torch":
        env["RWKV7_QWEN35_FORCE_TORCH"] = "1"
    else:
        env.pop("RWKV7_QWEN35_FORCE_TORCH", None)
    if spec.model_kind == "rwkv" and spec.quantization == "bnb8":
        env["RWKV7_BNB_SKIP_POLICY"] = rwkv_bnb8_skip_policy
    else:
        env.pop("RWKV7_BNB_SKIP_POLICY", None)
    return env


def main() -> int:
    args = parse_args()
    config = MatrixConfig(
        pairs=args.pair,
        prompts=args.prompt_tokens,
        decodes=args.decode_tokens,
        batch_sizes=args.batch_sizes,
        quantizations=args.quantizations,
        dtype=args.dtype,
    )
    validate_matrix(config)
    selected_roles = set(args.model_roles)
    native_quantizations = {
        "torchao_w8",
        "torchao_w4",
        "a8w8",
        "mm8",
        "mm4",
        "bnb8_a8w8_head",
    }
    if "reference" in selected_roles and any(q in native_quantizations for q in args.quantizations):
        raise ValueError(
            "Native quant modes are RWKV candidate backends. Run candidate and reference roles separately, "
            "using the native mode for RWKV and bnb8/bnb4 for the Qwen reference."
        )
    specs = [spec for spec in build_run_specs(config) if spec.model_role in selected_roles]
    seen = existing_keys(args.results) if args.skip_existing else set()
    env = build_run_environment(args)

    executed = 0
    failures = 0
    started = time.perf_counter()
    for index, spec in enumerate(specs, 1):
        raw_key = spec.raw_key_for_backend(args.qwen_backend, args.qwen_conv_backend)
        if raw_key in seen:
            print(f"skip existing {index}/{len(specs)} {raw_key}", flush=True)
            continue
        cmd = worker_command(args, spec)
        print(f"\n[{index}/{len(specs)}] $ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
        if args.dry_run:
            executed += 1
        else:
            row_started = time.perf_counter()
            run_env = build_worker_environment(
                env,
                spec,
                args.qwen_backend,
                args.rwkv_bnb8_skip_policy,
            )
            proc = subprocess.run(cmd, text=True, capture_output=True, env=run_env)
            print(proc.stdout, end="", flush=True)
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr, flush=True)
            current_keys = existing_keys(args.results)
            if proc.returncode != 0:
                failures += 1
                if raw_key not in current_keys:
                    append_orchestrator_failure(
                        args.results,
                        spec,
                        cmd,
                        proc,
                        qwen_backend=args.qwen_backend,
                        qwen_conv_backend=args.qwen_conv_backend,
                        benchmark_matrix=args.benchmark_matrix,
                    )
                    current_keys.add(raw_key)
                if args.fail_fast:
                    return proc.returncode
            seen.update(current_keys)
            executed += 1
            print(
                f"row wall_s={time.perf_counter() - row_started:.1f} "
                f"executed={executed} failures={failures} elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
        if args.max_runs and executed >= args.max_runs:
            break
    print(
        "QWEN35_SPEED_MATRIX_RUN "
        + json.dumps(
            {
                "raw_rows_expected": len(specs),
                "comparison_cells_expected": len({spec.cell_key for spec in specs}),
                "executed": executed,
                "failures": failures,
                "results": str(args.results),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
