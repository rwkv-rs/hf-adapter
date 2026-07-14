#!/usr/bin/env python3
"""Run a reproducible Zipf-prefix MLX serving load with latency percentiles."""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AXIS = "mlx_serving_load"


def append(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def provenance() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    dirty = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "platform": platform.platform(),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "min": round(min(values), 6) if values else None,
        "p50": round(float(percentile(values, 0.50)), 6) if values else None,
        "p95": round(float(percentile(values, 0.95)), 6) if values else None,
        "p99": round(float(percentile(values, 0.99)), 6) if values else None,
        "max": round(max(values), 6) if values else None,
        "mean": round(statistics.fmean(values), 6) if values else None,
    }


def prompts(count: int) -> list[str]:
    topics = (
        "state cache",
        "dynamic batching",
        "quantized inference",
        "long context",
        "latency telemetry",
        "recurrent models",
        "memory isolation",
        "production reliability",
    )
    return [
        "User: Shared serving prefix family "
        f"{index:02d} discusses {topics[index % len(topics)]}. "
        + ("context " * (index % 9))
        + "Give one short word. Assistant:"
        for index in range(int(count))
    ]


def run_parent(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="rwkv7-mlx-load-") as temporary:
        child_results = Path(temporary) / "child.jsonl"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model",
            args.model,
            "--dtype",
            args.dtype,
            "--requests",
            str(args.requests),
            "--concurrency",
            str(args.concurrency),
            "--unique-prompts",
            str(args.unique_prompts),
            "--zipf-alpha",
            str(args.zipf_alpha),
            "--seed",
            str(args.seed),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--min-cache-hit-rate",
            str(args.min_cache_hit_rate),
            "--cache-max-bytes",
            str(args.cache_max_bytes),
            "--results",
            str(child_results),
            "--isolated-child",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        child_rows = []
        if child_results.exists():
            child_rows = [json.loads(line) for line in child_results.read_text().splitlines() if line]
        matches = [row for row in child_rows if row.get("axis") == AXIS]
        if completed.returncode or len(matches) != 1:
            row = {
                "axis": AXIS,
                "status": "fail",
                "model": Path(args.model).name,
                "reason": "isolated load child failed",
                "child_returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
                **provenance(),
            }
        else:
            row = matches[0]
            row["process_isolated"] = True
        print(json.dumps(row, ensure_ascii=False))
        append(args.results, row)
        summary = {
            "axis": AXIS + "_summary",
            "status": row["status"],
            "requests": row.get("requests"),
            "process_isolated": True,
        }
        print(json.dumps(summary, ensure_ascii=False))
        append(args.results, summary)
        return 1 if args.fail_on_gate and row["status"] != "pass" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16", "keep"])
    parser.add_argument("--requests", type=int, default=10_000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--unique-prompts", type=int, default=64)
    parser.add_argument("--zipf-alpha", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--min-cache-hit-rate", type=float, default=0.80)
    parser.add_argument("--cache-max-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--results", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--isolated-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.unique_prompts, args.max_new_tokens, args.cache_max_bytes) <= 0:
        raise ValueError("request, concurrency, prompt, token, and cache budgets must be positive")
    if args.requests < args.concurrency:
        raise ValueError("--requests must be at least --concurrency")
    if args.zipf_alpha <= 0 or not 0 <= args.min_cache_hit_rate <= 1:
        raise ValueError("Zipf alpha must be positive and hit-rate SLO must be in [0,1]")
    env = {
        "axis": AXIS + "_env",
        "status": "plan" if args.dry_run else "info",
        "model": args.model,
        "dtype": args.dtype,
        "requests": int(args.requests),
        "concurrency": int(args.concurrency),
        "unique_prompts": int(args.unique_prompts),
        "zipf_alpha": float(args.zipf_alpha),
        "seed": int(args.seed),
        "max_new_tokens": int(args.max_new_tokens),
        "min_cache_hit_rate": float(args.min_cache_hit_rate),
        **provenance(),
    }
    print(json.dumps(env, ensure_ascii=False))
    append(args.results, env)
    if args.dry_run:
        return 0
    if not args.isolated_child:
        return run_parent(args)

    import mlx.core as mx
    from transformers import AutoTokenizer

    from rwkv7_hf.mlx_bridge import mlx_memory_telemetry, reset_mlx_peak_memory
    from rwkv7_hf.mlx_cache import MLXPrefixStateCache
    from rwkv7_hf.mlx_model import load_mlx_rwkv7_model
    from rwkv7_hf.mlx_scheduler import MLXDynamicBatchScheduler

    reset_mlx_peak_memory()
    model = load_mlx_rwkv7_model(args.model, dtype=args.dtype)
    model.decode_backend = "eager"
    model.decode_norm_backend = "reference"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_values = prompts(args.unique_prompts)
    oracle = {
        prompt: model.generate_text(
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
        ).generated_ids
        for prompt in prompt_values
    }
    cache = MLXPrefixStateCache(
        model,
        max_entries=args.unique_prompts,
        max_bytes=args.cache_max_bytes,
        ttl_s=None,
        namespace="zipf-serving-load",
        tokenizer=tokenizer,
    )
    scheduler = MLXDynamicBatchScheduler(
        model,
        tokenizer,
        max_batch_size=args.concurrency,
        max_in_flight=args.concurrency,
        prefix_cache=cache,
        session_backend="auto",
        prepare_decode_policy=False,
        dtype=args.dtype,
        quantization="none",
    )
    rng = random.Random(args.seed)
    population = list(range(args.unique_prompts))
    weights = [1.0 / ((index + 1) ** args.zipf_alpha) for index in population]
    selected = rng.choices(population, weights=weights, k=args.requests)

    request_ids: list[str] = []
    memory_samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for wave_start in range(0, args.requests, args.concurrency):
        wave = selected[wave_start : wave_start + args.concurrency]
        for offset, prompt_index in enumerate(wave):
            request_id = f"load-{wave_start + offset:06d}"
            scheduler.submit(
                prompt_values[prompt_index],
                max_new_tokens=args.max_new_tokens,
                request_id=request_id,
            )
            request_ids.append(request_id)
        scheduler.run_until_idle(max_ticks=max(4, args.max_new_tokens + 2))
        completed = wave_start + len(wave)
        if completed == len(wave) or completed % 1000 == 0 or completed == args.requests:
            memory_samples.append({"completed": completed, **mlx_memory_telemetry()})
    elapsed_s = time.perf_counter() - started

    token_match = True
    state_released = True
    completed = True
    ttft: list[float] = []
    e2e: list[float] = []
    queue: list[float] = []
    reused_tokens = 0
    prompt_tokens = 0
    for request_id, prompt_index in zip(request_ids, selected, strict=True):
        request = scheduler.request(request_id)
        timing = request.telemetry()
        completed = completed and request.status == "completed"
        token_match = token_match and request.generated_ids == oracle[prompt_values[prompt_index]]
        state_released = state_released and request.session is None
        reused_tokens += int(request.prefix_tokens_reused)
        prompt_tokens += int(request.prompt_tokens)
        if timing["ttft_s"] is not None:
            ttft.append(float(timing["ttft_s"]))
        if timing["e2e_s"] is not None:
            e2e.append(float(timing["e2e_s"]))
        if timing["queue_s"] is not None:
            queue.append(float(timing["queue_s"]))

    cache_stats = cache.telemetry()
    batch_history = list(scheduler.batch_size_history)
    backend_history = list(scheduler.batch_backend_history)
    max_batch = max(batch_history, default=0)
    byte_hit_rate = reused_tokens / max(prompt_tokens, 1)
    gates = {
        "request_count": len(request_ids) == args.requests,
        "all_completed": completed and scheduler.completed_count == args.requests,
        "all_token_match": token_match,
        "all_state_released": state_released,
        "concurrency_reached": max_batch == args.concurrency,
        "true_batched": bool(backend_history)
        and all(value in {"batched", "batched_stable"} for value in backend_history),
        "request_cache_hit_rate": float(cache_stats["hit_rate"]) >= args.min_cache_hit_rate,
        "byte_cache_hit_rate": byte_hit_rate >= args.min_cache_hit_rate,
        "latency_complete": len(ttft) == len(e2e) == len(queue) == args.requests,
        "finite_latency": all(math.isfinite(value) and value >= 0 for value in [*ttft, *e2e, *queue]),
    }
    row = {
        "axis": AXIS,
        "status": "pass" if all(gates.values()) else "fail",
        "model": Path(args.model).name,
        "model_path": args.model,
        "dtype": args.dtype,
        "quantization": "none",
        "requests": int(args.requests),
        "concurrency": int(args.concurrency),
        "unique_prompts": int(args.unique_prompts),
        "zipf_alpha": float(args.zipf_alpha),
        "seed": int(args.seed),
        "max_new_tokens": int(args.max_new_tokens),
        "elapsed_s": round(elapsed_s, 6),
        "requests_per_second": round(args.requests / elapsed_s, 6),
        "generated_tokens_per_second": round(args.requests * args.max_new_tokens / elapsed_s, 6),
        "gates": gates,
        "max_observed_batch": int(max_batch),
        "batch_ticks": len(batch_history),
        "batch_sizes": sorted(set(batch_history)),
        "batch_backends": sorted(set(backend_history)),
        "cache": cache_stats,
        "request_cache_hit_rate": float(cache_stats["hit_rate"]),
        "byte_cache_hit_rate": round(byte_hit_rate, 8),
        "prompt_tokens": int(prompt_tokens),
        "prefix_tokens_reused": int(reused_tokens),
        "ttft_s": latency_summary(ttft),
        "e2e_s": latency_summary(e2e),
        "queue_s": latency_summary(queue),
        "memory_samples": memory_samples,
        **mlx_memory_telemetry(),
        **provenance(),
    }
    print(json.dumps(row, ensure_ascii=False))
    append(args.results, row)
    return 1 if args.fail_on_gate and row["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
