#!/usr/bin/env python3
"""Real-model ragged dynamic-batching and prefix-state-cache acceptance."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AXIS = "mlx_dynamic_serving"


def _sysctl(name: str) -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            text=True,
            capture_output=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def apple_device_telemetry() -> dict[str, Any]:
    """Collect dependency-free hardware identity for benchmark evidence."""

    system = platform.system()
    machine = platform.machine()
    chip = _sysctl("machdep.cpu.brand_string") if system == "Darwin" else ""
    model_identifier = _sysctl("hw.model") if system == "Darwin" else ""
    memory_raw = _sysctl("hw.memsize") if system == "Darwin" else ""
    try:
        memory_bytes = int(memory_raw) if memory_raw else None
    except ValueError:
        memory_bytes = None
    return {
        "system": system,
        "machine": machine,
        "chip": chip,
        "model_identifier": model_identifier,
        "memory_bytes": memory_bytes,
        "is_apple_silicon": system == "Darwin" and machine == "arm64" and chip.startswith("Apple "),
    }


def git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "axis": f"{AXIS}_summary",
        "status": "pass" if rows and all(row["status"] == "pass" for row in rows) else "fail",
        "models": len(rows),
        "pass_models": sum(row["status"] == "pass" for row in rows),
        "all_token_match": bool(rows) and all(row.get("all_token_match") for row in rows),
        "all_true_batched": bool(rows) and all(row.get("all_rounds_true_batched") for row in rows),
        "all_ragged_prompt_gate_pass": bool(rows) and all(row.get("ragged_prompt_gate_pass") for row in rows),
        "all_prefix_continuation_gate_pass": bool(rows)
        and all(row.get("prefix_continuation_gate_pass") for row in rows),
        "all_cache_key_isolation_pass": bool(rows)
        and all(row.get("cache_key_isolation_pass") for row in rows),
        "all_cache_eviction_gate_pass": bool(rows)
        and all(row.get("cache_eviction_gate_pass") for row in rows),
        "all_timeout_gate_pass": bool(rows) and all(row.get("timeout_gate_pass") for row in rows),
    }


def _run_isolated(args: argparse.Namespace, models: list[str]) -> int:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rwkv7-mlx-serving-") as temporary:
        for index, model in enumerate(models):
            child_results = Path(temporary) / f"model-{index}.jsonl"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--models",
                model,
                "--dtype",
                args.dtype,
                "--quantization",
                args.quantization,
                "--quant-min-params",
                str(args.quant_min_params),
                "--quant-backend",
                args.quant_backend,
                "--wkv-backend",
                args.wkv_backend,
                "--max-batch-size",
                str(args.max_batch_size),
                "--cache-max-entries",
                str(args.cache_max_entries),
                "--cache-max-bytes",
                str(args.cache_max_bytes),
                "--results",
                str(child_results),
                "--isolated-child",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            child_rows = []
            if child_results.exists():
                child_rows = [json.loads(line) for line in child_results.read_text().splitlines() if line]
            model_rows = [row for row in child_rows if row.get("axis") == AXIS]
            if result.returncode != 0 or len(model_rows) != 1:
                row = {
                    "axis": AXIS,
                    "status": "fail",
                    "model": Path(model).name,
                    "reason": "isolated dynamic serving child failed",
                    "child_returncode": result.returncode,
                    "child_stdout_tail": result.stdout[-2000:],
                    "child_stderr_tail": result.stderr[-2000:],
                }
            else:
                row = model_rows[0]
                row["process_isolated"] = True
            print(json.dumps(row, ensure_ascii=False))
            append_jsonl(args.results, row)
            rows.append(row)
    summary = _summary(rows)
    summary["process_isolated"] = True
    print(json.dumps(summary, ensure_ascii=False))
    append_jsonl(args.results, summary)
    return 1 if args.fail_on_gate and summary["status"] != "pass" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16", "keep"])
    parser.add_argument("--quantization", default="none", choices=["none", "mm8", "mm4"])
    parser.add_argument("--quant-min-params", type=int, default=8_000_000)
    parser.add_argument("--quant-backend", default="auto", choices=["auto", "metal", "affine", "reference"])
    parser.add_argument("--wkv-backend", default="auto", choices=["auto", "metal", "reference"])
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--cache-max-entries", type=int, default=4)
    parser.add_argument("--cache-max-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--results", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--isolated-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    if not models:
        raise ValueError("--models must contain at least one path")
    if min(args.max_batch_size, args.cache_max_entries, args.cache_max_bytes) <= 0:
        raise ValueError("batch/cache budgets must be positive")

    env = {
        "axis": f"{AXIS}_env",
        "status": "plan" if args.dry_run else "info",
        "models": models,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "quant_min_params": args.quant_min_params,
        "quant_backend": args.quant_backend,
        "wkv_backend": args.wkv_backend,
        "max_batch_size": args.max_batch_size,
        "cache_max_entries": args.cache_max_entries,
        "cache_max_bytes": args.cache_max_bytes,
        **apple_device_telemetry(),
        **git_provenance(),
    }
    print(json.dumps(env, ensure_ascii=False))
    append_jsonl(args.results, env)
    if args.dry_run:
        return 0
    if len(models) > 1 and not args.isolated_child:
        return _run_isolated(args, models)

    from transformers import AutoTokenizer

    import mlx.core as mx

    from rwkv7_hf.mlx_bridge import mlx_memory_telemetry, reset_mlx_peak_memory
    from rwkv7_hf.mlx_cache import MLXPrefixStateCache
    from rwkv7_hf.mlx_model import load_mlx_rwkv7_model
    from rwkv7_hf.mlx_scheduler import MLXBackpressureError, MLXDynamicBatchScheduler

    model_path = models[0]
    reset_mlx_peak_memory()
    model = load_mlx_rwkv7_model(
        model_path,
        dtype=args.dtype,
        quantization=args.quantization,
        quant_min_params=args.quant_min_params,
        quant_backend=args.quant_backend,
        wkv_backend=args.wkv_backend,
    )
    model.decode_backend = "eager"
    model.decode_norm_backend = "reference"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    cases = {
        "a": ("User: Apple MLX shared A. Assistant:", 1),
        "b": (
            "User: Apple MLX shared prompt B has deliberately more context for ragged batching. Assistant:",
            3,
        ),
        "a2": ("User: Apple MLX shared A. Assistant:", 2),
        "c": ("User: unique C. Assistant:", 4),
        "d": (
            "User: this cancelled request has a substantially different prompt length "
            "and must release state. Assistant:",
            2,
        ),
        "b2": (
            "User: Apple MLX shared prompt B has deliberately more context for ragged batching. Assistant:",
            1,
        ),
        "e": ("User: Apple MLX late arrival E uses medium context. Assistant:", 3),
        "timeout": ("User: timeout cleanup probe. Assistant:", 8),
    }
    cache = MLXPrefixStateCache(
        model,
        max_entries=args.cache_max_entries,
        max_bytes=args.cache_max_bytes,
        ttl_s=None,
        namespace="dynamic-serving-bench",
        tokenizer=tokenizer,
    )

    # Real-model chunk-boundary continuation parity.  This uses a dedicated
    # cache so the service workload's hit/eviction telemetry remains honest.
    probe_text = (
        "User: a cached RWKV state prefix must continue across this explicit token chunk boundary "
        "without changing logits or recurrent state. Assistant:"
    )
    probe_ids = [int(value) for value in tokenizer(probe_text, add_special_tokens=False).input_ids]
    if len(probe_ids) < 4:
        raise RuntimeError("prefix continuation probe requires at least four tokens")
    split = len(probe_ids) // 2
    prefix_ids = probe_ids[:split]
    suffix_ids = probe_ids[split:]
    prefix_logits, prefix_state = model.prefill([prefix_ids])
    prefix_probe_cache = MLXPrefixStateCache(
        model,
        max_entries=2,
        max_bytes=args.cache_max_bytes,
        ttl_s=None,
        namespace="dynamic-serving-prefix-probe",
        tokenizer=tokenizer,
    )
    prefix_probe_cache.put(prefix_ids, prefix_logits, prefix_state)
    prefix_hit = prefix_probe_cache.find_longest(probe_ids)
    if prefix_hit is None:
        raise RuntimeError("real-model prefix continuation probe did not hit")
    continued_logits, continued_state = model.prefill([suffix_ids], state=prefix_hit.state)
    full_logits, full_state = model.prefill([probe_ids])
    def state_arrays(state: Any) -> tuple[Any, ...]:
        return (
            state.v_first,
            *state.recurrent_state,
            *state.attn_x_prev,
            *state.ffn_x_prev,
        )
    mx.eval(continued_logits, full_logits, *state_arrays(continued_state), *state_arrays(full_state))
    prefix_logits_max_abs = float(
        mx.max(mx.abs(continued_logits.astype(mx.float32) - full_logits.astype(mx.float32)))
    )
    prefix_state_max_abs = max(
        float(mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32))))
        for left, right in zip(state_arrays(continued_state), state_arrays(full_state), strict=True)
    )
    prefix_argmax_match = (
        mx.argmax(continued_logits[:, -1, :], axis=-1).tolist()
        == mx.argmax(full_logits[:, -1, :], axis=-1).tolist()
    )
    prefix_continuation_gate = (
        not prefix_hit.exact
        and prefix_hit.prefix_tokens == split
        and int(continued_state.seen_tokens) == len(probe_ids)
        and prefix_argmax_match
        and prefix_logits_max_abs <= 0.125
        and prefix_state_max_abs <= 0.125
    )

    # Tenant/config routing must not collide, even for identical token ids.
    isolated_cache = MLXPrefixStateCache(
        model,
        max_entries=2,
        max_bytes=args.cache_max_bytes,
        ttl_s=None,
        namespace="dynamic-serving-other-tenant",
        tokenizer=tokenizer,
    )
    isolated_miss = isolated_cache.get_exact(prefix_ids) is None
    required_key_fields = {
        "namespace",
        "model_source_revision_layout",
        "tokenizer",
        "dtype",
        "quantization",
        "backend",
        "prefix_token_ids",
    }
    cache_key_isolation_pass = (
        isolated_miss
        and isolated_cache.model_fingerprint != prefix_probe_cache.model_fingerprint
        and required_key_fields.issubset(set(cache.telemetry()["key_schema"]))
        and cache.telemetry()["tokenizer_fingerprint"] is not None
    )

    # Exercise both LRU and TTL cleanup with real model states and explicit
    # post-expiration byte accounting.
    probe_clock = [100.0]
    eviction_cache = MLXPrefixStateCache(
        model,
        max_entries=1,
        max_bytes=args.cache_max_bytes,
        ttl_s=1.0,
        namespace="dynamic-serving-eviction-probe",
        tokenizer=tokenizer,
        clock=lambda: probe_clock[0],
    )
    eviction_cache.put(prefix_ids, prefix_logits, prefix_state)
    eviction_cache.put(probe_ids, full_logits, full_state)
    lru_telemetry = eviction_cache.telemetry()
    probe_clock[0] += 2.0
    ttl_telemetry = eviction_cache.telemetry()
    cache_eviction_gate_pass = (
        lru_telemetry["evictions"] == 1
        and lru_telemetry["entries"] == 1
        and lru_telemetry["bytes"] <= lru_telemetry["max_bytes"]
        and ttl_telemetry["expirations"] == 1
        and ttl_telemetry["entries"] == 0
        and ttl_telemetry["bytes"] == 0
    )

    scheduler = MLXDynamicBatchScheduler(
        model,
        tokenizer,
        max_batch_size=args.max_batch_size,
        max_in_flight=6,
        prefix_cache=cache,
        session_backend="auto",
        prepare_decode_policy=False,
        dtype=args.dtype,
        quantization=args.quantization,
    )
    expected = {
        request_id: model.generate_text(tokenizer, prompt, max_new_tokens=count).generated_ids
        for request_id, (prompt, count) in cases.items()
        if request_id not in {"d", "timeout"}
    }
    for request_id in ("a", "b", "a2", "c", "d", "b2"):
        prompt, count = cases[request_id]
        scheduler.submit(prompt, max_new_tokens=count, request_id=request_id)
    backpressure_pass = False
    try:
        scheduler.submit("overflow", max_new_tokens=1, request_id="overflow")
    except MLXBackpressureError:
        backpressure_pass = True
    scheduler.step()
    cancellation_pass = scheduler.cancel("d", reason="benchmark_cancel")
    prompt, count = cases["e"]
    scheduler.submit(prompt, max_new_tokens=count, request_id="e")
    scheduler.run_until_idle(max_ticks=64)
    timeout_prompt, timeout_count = cases["timeout"]
    scheduler.submit(
        timeout_prompt,
        max_new_tokens=timeout_count,
        request_id="timeout",
        timeout_s=1e-9,
    )
    scheduler.step()

    completed_ids = [request_id for request_id in cases if request_id not in {"d", "timeout"}]
    token_matches = {
        request_id: scheduler.request(request_id).generated_ids == expected[request_id]
        for request_id in completed_ids
    }
    seen_matches = {
        request_id: scheduler.request(request_id).final_seen_tokens
        == scheduler.request(request_id).prompt_tokens + scheduler.request(request_id).max_new_tokens
        for request_id in completed_ids
    }
    scheduler_telemetry = scheduler.telemetry()
    cache_telemetry = cache.telemetry()
    prompt_token_lengths = {
        request_id: scheduler.request(request_id).prompt_tokens for request_id in cases
    }
    ragged_prompt_gate = (
        len(set(prompt_token_lengths.values())) >= 4
        and max(prompt_token_lengths.values()) - min(prompt_token_lengths.values()) >= 4
    )
    all_true_batched = bool(scheduler.batch_backend_history) and all(
        backend in {"batched", "batched_stable"}
        for backend in scheduler.batch_backend_history
    )
    state_released = all(scheduler.request(request_id).session is None for request_id in cases)
    all_token_match = all(token_matches.values())
    all_seen_match = all(seen_matches.values())
    cache_gate = cache_telemetry["exact_hits"] >= 2 and cache_telemetry["evictions"] >= 1
    timeout_request = scheduler.request("timeout")
    timeout_gate = (
        timeout_request.status == "timed_out"
        and timeout_request.session is None
        and scheduler_telemetry["timed_out_count"] == 1
    )
    arrival_departure_gate = (
        scheduler.request("e").arrival_tick > 0
        and scheduler.request("d").status == "cancelled"
        and scheduler.request("a").status == "completed"
    )
    status = (
        "pass"
        if all_token_match
        and all_seen_match
        and all_true_batched
        and state_released
        and backpressure_pass
        and cancellation_pass
        and cache_gate
        and timeout_gate
        and ragged_prompt_gate
        and arrival_departure_gate
        and prefix_continuation_gate
        and cache_key_isolation_pass
        and cache_eviction_gate_pass
        else "fail"
    )
    row = {
        "axis": AXIS,
        "status": status,
        "model": Path(model_path).name,
        "model_path": model_path,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "request_count": len(cases),
        "completed_requests": len(completed_ids),
        "cancelled_requests": 1,
        "timed_out_requests": 1,
        "all_token_match": all_token_match,
        "all_seen_tokens_match": all_seen_match,
        "all_session_state_released": state_released,
        "all_rounds_true_batched": all_true_batched,
        "backpressure_gate_pass": backpressure_pass,
        "cancellation_gate_pass": cancellation_pass,
        "timeout_gate_pass": timeout_gate,
        "arrival_departure_gate_pass": arrival_departure_gate,
        "ragged_prompt_gate_pass": ragged_prompt_gate,
        "cache_gate_pass": cache_gate,
        "prefix_continuation_gate_pass": prefix_continuation_gate,
        "cache_key_isolation_pass": cache_key_isolation_pass,
        "cache_eviction_gate_pass": cache_eviction_gate_pass,
        "prompt_token_lengths": prompt_token_lengths,
        "prefix_continuation": {
            "prompt_tokens": len(probe_ids),
            "prefix_tokens": split,
            "suffix_tokens": len(suffix_ids),
            "longest_prefix_hit": not prefix_hit.exact,
            "argmax_match": prefix_argmax_match,
            "logits_max_abs": prefix_logits_max_abs,
            "state_max_abs": prefix_state_max_abs,
            "seen_tokens": int(continued_state.seen_tokens),
        },
        "cache_key_isolation": {
            "isolated_miss": isolated_miss,
            "fingerprints_differ": isolated_cache.model_fingerprint
            != prefix_probe_cache.model_fingerprint,
            "key_schema": cache_telemetry["key_schema"],
            "tokenizer_fingerprint_present": cache_telemetry["tokenizer_fingerprint"] is not None,
        },
        "cache_eviction": {
            "lru": lru_telemetry,
            "after_ttl": ttl_telemetry,
        },
        "token_matches": token_matches,
        "seen_matches": seen_matches,
        "scheduler": scheduler_telemetry,
        "prefix_cache": cache_telemetry,
        **git_provenance(),
        **mlx_memory_telemetry(),
    }
    print(json.dumps(row, ensure_ascii=False))
    append_jsonl(args.results, row)
    summary = _summary([row])
    print(json.dumps(summary, ensure_ascii=False))
    append_jsonl(args.results, summary)
    return 1 if args.fail_on_gate and status != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
