#!/usr/bin/env python3
# coding=utf-8
"""Apple/Qwen3.5 baseline harness for RWKV-7 MLX/CoreML work.

This script records a common JSONL schema for the first "beat Qwen3.5 on
Apple/mobile" gate.  It intentionally avoids a hard dependency on Ollama, MLX,
or CoreML so it can live in the repository and be unit-tested on CPU-only CI.

Supported live runners:

* Qwen3.5 through a local Ollama server (`/api/generate`, streaming JSONL).
* RWKV-7 through this repository's optional MLX recurrent backend.

CoreML/ANE rows use the same schema and are expected to be appended by the
follow-up CoreML runner once the export path lands.  Keeping the schema here
prevents every runner from inventing its own metric names.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

AXIS = "qwen35_apple_baseline"

DEFAULT_QWEN_MODELS = [
    "qwen3.5:0.8b-mlx",
    "qwen3.5:2b-mlx",
    "qwen3.5:4b-mlx",
    "qwen3.5:9b-mlx",
]

# Public package sizes from the Ollama qwen3.5 model page, used only as metadata.
QWEN35_PUBLIC_BASELINE = {
    "qwen3.5:0.8b-mlx": {"family": "qwen3.5", "size_class": "0.8B", "public_package_gb": 1.2},
    "qwen3.5:2b-mlx": {"family": "qwen3.5", "size_class": "2B", "public_package_gb": 3.1},
    "qwen3.5:4b-mlx": {"family": "qwen3.5", "size_class": "4B", "public_package_gb": 4.0},
    "qwen3.5:9b-mlx": {"family": "qwen3.5", "size_class": "9B", "public_package_gb": 8.9},
}

DEFAULT_PROMPT_SEED = (
    "User: Compare RWKV-7 and Qwen3.5 on Apple Silicon. "
    "Report throughput, latency, memory, state-cache behavior, and quantization stability.\n"
    "Assistant: "
)


@dataclass(frozen=True)
class PromptCase:
    name: str
    target_chars: int
    prompt: str


def parse_csv(raw: str, *, default: list[str] | None = None) -> list[str]:
    if raw is None or not str(raw).strip():
        return list(default or [])
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    values = [int(item) for item in parse_csv(raw)]
    if not values:
        raise ValueError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError(f"all integer values must be positive: {values}")
    return values


def parse_keep_alive(raw: str | int) -> str | int:
    value = str(raw).strip()
    try:
        return int(value)
    except ValueError:
        if not value:
            raise ValueError("--ollama-keep-alive must not be empty")
        return value


def make_prompt(seed: str, target_chars: int) -> str:
    target = int(target_chars)
    if target <= 0:
        raise ValueError("target_chars must be positive")
    if not seed:
        raise ValueError("prompt seed must be non-empty")
    repeats = (target + len(seed) - 1) // len(seed)
    return (seed * repeats)[:target]


def build_prompt_cases(target_chars: Iterable[int], seed: str) -> list[PromptCase]:
    cases: list[PromptCase] = []
    for chars in target_chars:
        cases.append(PromptCase(name=f"chars{int(chars)}", target_chars=int(chars), prompt=make_prompt(seed, int(chars))))
    return cases


def append_jsonl(path: str | Path | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mac_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def device_info() -> dict[str, Any]:
    row: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        row.update(
            {
                "macos_product_version": mac_command(["sw_vers", "-productVersion"]),
                "mac_hw_model": mac_command(["sysctl", "-n", "hw.model"]),
                "mac_chip": mac_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                "mac_memsize_bytes": _safe_int(mac_command(["sysctl", "-n", "hw.memsize"])),
            }
        )
    return row


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def tok_s(count: int | None, duration_ns: int | None) -> float | None:
    if count is None or duration_ns is None or duration_ns <= 0:
        return None
    return round(float(count) / (float(duration_ns) / 1_000_000_000.0), 6)


def ollama_loaded_model_telemetry(*, host: str, model: str, timeout_s: float) -> dict[str, Any]:
    """Read official `/api/ps` loaded-memory telemetry for one model."""

    request = urllib.request.Request(host.rstrip("/") + "/api/ps", method="GET")
    with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:  # noqa: S310 - local benchmark endpoint
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    for item in payload.get("models") or []:
        if str(item.get("name") or item.get("model") or "") != model:
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        return {
            # `size_vram` is the official loaded-runtime allocation, not a
            # sampled peak. Keep the name distinct from peak memory so strict
            # peak-to-peak gates cannot pass accidentally.
            "ollama_loaded_memory_bytes": _safe_int(item.get("size_vram")),
            "ollama_model_size_bytes": _safe_int(item.get("size")),
            "ollama_context_length": _safe_int(item.get("context_length")),
            "ollama_quantization_level": details.get("quantization_level"),
        }
    return {"ollama_memory_telemetry_reason": f"model {model!r} missing from /api/ps"}


def unload_ollama_model(*, host: str, model: str, timeout_s: float) -> None:
    data = json.dumps({"model": model, "stream": False, "keep_alive": 0}).encode("utf-8")
    request = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:  # noqa: S310 - local benchmark endpoint
        response.read()


def post_ollama_generate(
    *,
    host: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    timeout_s: float,
    think: bool = False,
    keep_alive: str | int = 0,
    cache_prompt: bool = False,
    capture_memory: bool = True,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    url = host.rstrip("/") + "/api/generate"
    isolate_after_row = bool(capture_memory and keep_alive == 0)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        # Thinking tokens are valid generated tokens, but they leave
        # ``response`` empty on short rows and make response-quality scoring
        # incomparable with RWKV. Keep thinking opt-in for this baseline.
        "think": bool(think),
        # Default to an isolated model lifetime. Ollama otherwise reuses a
        # completed prompt across rows and can report sub-microsecond
        # prompt_eval_duration, which is a cache-hit metric rather than prefill.
        # Keep the model alive just long enough to query `/api/ps`, then issue
        # an explicit unload below. This preserves row isolation and captures
        # the official loaded-memory value in the same run.
        "keep_alive": -1 if isolate_after_row else keep_alive,
        "cache_prompt": bool(cache_prompt),
        "options": {
            "num_predict": int(max_new_tokens),
            "temperature": float(temperature),
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    chunks: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:  # noqa: S310 - user-provided local benchmark endpoint
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunk["_client_elapsed_s"] = time.perf_counter() - t0
            chunks.append(chunk)
            if chunks[-1].get("done"):
                break
    elapsed_s = time.perf_counter() - t0
    telemetry: dict[str, Any] = {}
    if capture_memory:
        try:
            telemetry.update(ollama_loaded_model_telemetry(host=host, model=model, timeout_s=timeout_s))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            telemetry["ollama_memory_telemetry_reason"] = f"{type(exc).__name__}: {exc}"
    if isolate_after_row:
        try:
            unload_ollama_model(host=host, model=model, timeout_s=timeout_s)
        except (urllib.error.URLError, TimeoutError, OSError):
            # Do not discard a completed performance row solely because the
            # cleanup request failed; the next row will expose cache reuse in
            # prompt_eval_duration and the env row records the policy.
            telemetry["ollama_unload_failed"] = True
    return chunks, elapsed_s, telemetry


def ollama_row_from_chunks(
    *,
    model: str,
    prompt_case: PromptCase,
    max_new_tokens: int,
    chunks: list[dict[str, Any]],
    elapsed_s: float,
    store_response: bool = False,
) -> dict[str, Any]:
    final = chunks[-1] if chunks else {}
    first_response_chunk_index = None
    first_output_chunk_index = None
    first_output_s = None
    for idx, chunk in enumerate(chunks):
        if first_response_chunk_index is None and chunk.get("response"):
            first_response_chunk_index = idx
        if first_output_chunk_index is None and (chunk.get("response") or chunk.get("thinking")):
            first_output_chunk_index = idx
            first_output_s = _safe_float(chunk.get("_client_elapsed_s"))
    prompt_eval_count = _safe_int(final.get("prompt_eval_count"))
    eval_count = _safe_int(final.get("eval_count"))
    prompt_eval_duration = _safe_int(final.get("prompt_eval_duration"))
    eval_duration = _safe_int(final.get("eval_duration"))
    total_duration = _safe_int(final.get("total_duration"))
    load_duration = _safe_int(final.get("load_duration"))
    response_text = "".join(str(chunk.get("response", "")) for chunk in chunks)
    thinking_text = "".join(str(chunk.get("thinking", "")) for chunk in chunks)
    display_text = response_text or thinking_text
    load_s = float(load_duration) / 1_000_000_000.0 if load_duration is not None else 0.0
    steady_ttft_s = max(0.0, float(first_output_s) - load_s) if first_output_s is not None else None
    row = {
        "axis": AXIS,
        "status": "pass",
        "engine": "ollama",
        "runtime": "ollama_mlx",
        "model": model,
        **QWEN35_PUBLIC_BASELINE.get(model, {"family": "qwen3.5"}),
        "prompt_case": prompt_case.name,
        "prompt_target_chars": int(prompt_case.target_chars),
        "prompt_chars": len(prompt_case.prompt),
        "prompt_eval_tokens": prompt_eval_count,
        "generated_tokens": eval_count,
        "requested_generated_tokens": int(max_new_tokens),
        "wall_s": round(float(elapsed_s), 6),
        "total_duration_ns": total_duration,
        "load_duration_ns": load_duration,
        "load_s": round(load_s, 6) if load_duration is not None else None,
        "prompt_eval_duration_ns": prompt_eval_duration,
        "eval_duration_ns": eval_duration,
        "prefill_tok_s": tok_s(prompt_eval_count, prompt_eval_duration),
        "decode_tok_s": tok_s(eval_count, eval_duration),
        "first_token_s": round(float(steady_ttft_s), 6) if steady_ttft_s is not None else None,
        "ttft_s": round(float(steady_ttft_s), 6) if steady_ttft_s is not None else None,
        "cold_ttft_s": round(float(first_output_s), 6) if first_output_s is not None else None,
        "ollama_chunk_count": len(chunks),
        "first_response_chunk_index": first_response_chunk_index,
        "first_output_chunk_index": first_output_chunk_index,
        "response_preview": display_text[:160],
        "response_chars": len(response_text),
        "thinking_chars": len(thinking_text),
    }
    if store_response:
        row["response_text"] = response_text
        row["thinking_text"] = thinking_text
    return row


def run_ollama_qwen(
    *,
    host: str,
    model: str,
    prompt_case: PromptCase,
    decode_lengths: list[int],
    repeats: int,
    temperature: float,
    timeout_s: float,
    results: str,
    store_response: bool = False,
    think: bool = False,
    keep_alive: str | int = 0,
    cache_prompt: bool = False,
    capture_memory: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for max_new_tokens in decode_lengths:
        for repeat_index in range(1, repeats + 1):
            try:
                chunks, elapsed_s, runtime_telemetry = post_ollama_generate(
                    host=host,
                    model=model,
                    prompt=prompt_case.prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    think=think,
                    keep_alive=keep_alive,
                    cache_prompt=cache_prompt,
                    capture_memory=capture_memory,
                )
                row = ollama_row_from_chunks(
                    model=model,
                    prompt_case=prompt_case,
                    max_new_tokens=max_new_tokens,
                    chunks=chunks,
                    elapsed_s=elapsed_s,
                    store_response=store_response,
                )
                row["repeat_index"] = int(repeat_index)
                row["repeat"] = int(repeats)
                row.update(runtime_telemetry)
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                row = {
                    "axis": AXIS,
                    "status": "skip",
                    "engine": "ollama",
                    "runtime": "ollama_mlx",
                    "model": model,
                    **QWEN35_PUBLIC_BASELINE.get(model, {"family": "qwen3.5"}),
                    "prompt_case": prompt_case.name,
                    "prompt_target_chars": int(prompt_case.target_chars),
                    "prompt_chars": len(prompt_case.prompt),
                    "requested_generated_tokens": int(max_new_tokens),
                    "repeat_index": int(repeat_index),
                    "repeat": int(repeats),
                    "reason": f"ollama request failed: {type(exc).__name__}: {exc}",
                }
            print(json.dumps(row, ensure_ascii=False))
            append_jsonl(results, row)
            rows.append(row)
    return rows


def run_rwkv_mlx(
    *,
    model_path: str,
    prompt_case: PromptCase,
    decode_lengths: list[int],
    repeats: int,
    dtype: str,
    quantization: str,
    quant_min_params: int,
    quant_backend: str,
    wkv_backend: str,
    chunk_size: int,
    results: str,
    store_response: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from transformers import AutoTokenizer

        import mlx.core as mx

        from rwkv7_hf.mlx_bridge import mlx_memory_telemetry, reset_mlx_peak_memory
        from rwkv7_hf.mlx_model import load_mlx_rwkv7_model
    except Exception as exc:  # pragma: no cover - exercised on non-Apple CI by skip row
        for max_new_tokens in decode_lengths:
            row = {
                "axis": AXIS,
                "status": "skip",
                "engine": "rwkv7_hf",
                "runtime": "mlx",
                "model": Path(model_path).name,
                "model_path": model_path,
                "prompt_case": prompt_case.name,
                "prompt_target_chars": int(prompt_case.target_chars),
                "requested_generated_tokens": int(max_new_tokens),
                "reason": f"MLX import/load prerequisites unavailable: {type(exc).__name__}: {exc}",
            }
            print(json.dumps(row, ensure_ascii=False))
            append_jsonl(results, row)
            rows.append(row)
        return rows

    t_load = time.perf_counter()
    model = load_mlx_rwkv7_model(
        model_path,
        dtype=dtype,
        quantization=quantization,
        quant_min_params=int(quant_min_params),
        quant_backend=quant_backend,
        wkv_backend=wkv_backend,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    load_s = time.perf_counter() - t_load
    prompt_ids = [int(x) for x in tokenizer(prompt_case.prompt, add_special_tokens=False).input_ids]
    if not prompt_ids:
        raise ValueError("RWKV tokenizer produced zero prompt tokens")

    for max_new_tokens in decode_lengths:
        for repeat_index in range(1, repeats + 1):
            reset_mlx_peak_memory()
            t_prefill = time.perf_counter()
            logits, state = model.prefill([prompt_ids])
            mx.eval(logits)
            prefill_s = time.perf_counter() - t_prefill
            chunk_diff = None
            chunk_s = None
            if int(chunk_size) > 0:
                t_chunk = time.perf_counter()
                chunk_logits, chunk_state = model.chunked_prefill([prompt_ids], chunk_size=int(chunk_size))
                mx.eval(chunk_logits)
                chunk_s = time.perf_counter() - t_chunk
                chunk_diff = float(mx.max(mx.abs(logits.astype(mx.float32) - chunk_logits.astype(mx.float32))))
                if int(chunk_state.seen_tokens) != len(prompt_ids):
                    raise AssertionError(
                        f"chunked state seen_tokens={chunk_state.seen_tokens}, expected {len(prompt_ids)}"
                    )
            # Streaming-shaped greedy decode.  The first token is available as
            # soon as the prefill logits are evaluated; then each emitted token
            # is fed once to advance the recurrent state and prepare the next
            # logits.  This avoids running decode_greedy twice, which would
            # double-process the first generated token.
            t_first = time.perf_counter()
            next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
            mx.eval(next_token)
            first_s = time.perf_counter() - t_first
            generated_preview: list[int] = []
            decode_step_s = 0.0
            final_state = state
            for _ in range(int(max_new_tokens)):
                generated_preview.extend(int(x) for x in next_token.reshape(-1).tolist())
                t_step = time.perf_counter()
                logits, final_state = model.decode_step(next_token, final_state)
                mx.eval(logits)
                decode_step_s += time.perf_counter() - t_step
                next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
                mx.eval(next_token)
            decode_s = first_s + decode_step_s
            response_text = tokenizer.decode(generated_preview, skip_special_tokens=True) if store_response else ""
            telemetry = model.telemetry()
            row = {
                "axis": AXIS,
                "status": "pass",
                "engine": "rwkv7_hf",
                "runtime": "mlx",
                "model": Path(model_path).name,
                "model_path": model_path,
                "family": "rwkv7",
                "dtype": dtype,
                "quantization": quantization,
                "quant_min_params": int(quant_min_params),
                "quant_backend": quant_backend,
                "wkv_backend": wkv_backend,
                "prompt_case": prompt_case.name,
                "prompt_target_chars": int(prompt_case.target_chars),
                "prompt_chars": len(prompt_case.prompt),
                "prompt_eval_tokens": len(prompt_ids),
                "generated_tokens": int(max_new_tokens),
                "requested_generated_tokens": int(max_new_tokens),
                "repeat_index": int(repeat_index),
                "repeat": int(repeats),
                "load_s": round(float(load_s), 6),
                "prefill_s": round(float(prefill_s), 6),
                "first_token_s": round(float(first_s), 6),
                "ttft_s": round(float(prefill_s + first_s), 6),
                "cold_ttft_s": round(float(load_s + prefill_s + first_s), 6),
                "decode_s": round(float(decode_s), 6),
                "prefill_tok_s": round(float(len(prompt_ids) / prefill_s), 6) if prefill_s > 0 else None,
                "decode_tok_s": round(float(max_new_tokens / decode_s), 6) if decode_s > 0 else None,
                "generated_preview": generated_preview[:16],
                "response_chars": len(response_text) if store_response else None,
                "seen_tokens_after_generate": int(final_state.seen_tokens),
                "expected_seen_tokens": int(len(prompt_ids) + max_new_tokens),
                "quantized_linear_last_backend_counts": telemetry.get("quantized_linear_last_backend_counts"),
                "group_rkv_quant_projection": telemetry.get("group_rkv_quant_projection"),
                "group_rkv_quant_projection_mode": telemetry.get("group_rkv_quant_projection_mode"),
                "group_rkv_quant_projection_counts": telemetry.get("group_rkv_quant_projection_counts"),
                **mlx_memory_telemetry(),
            }
            if store_response:
                row["generated_token_ids"] = generated_preview
                row["response_text"] = response_text
                row["response_preview"] = response_text[:160]
            if chunk_diff is not None:
                row.update(
                    {
                        "chunk_size": int(chunk_size),
                        "chunked_prefill_s": round(float(chunk_s), 6) if chunk_s is not None else None,
                        "chunked_prefill_max_abs": round(float(chunk_diff), 8),
                    }
                )
            print(json.dumps(row, ensure_ascii=False))
            append_jsonl(results, row)
            rows.append(row)
    # ``main`` invokes this runner once per prompt case. Explicitly release the
    # model and the final lazy arrays before the next case; otherwise MLX keeps
    # both model instances live and the second case reports roughly 2x memory.
    model = None
    tokenizer = None
    logits = None
    state = None
    final_state = None
    next_token = None
    if int(chunk_size) > 0:
        chunk_logits = None
        chunk_state = None
    gc.collect()
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_rows = [row for row in rows if row.get("axis") == AXIS and row.get("status") == "pass"]
    by_engine: dict[str, list[dict[str, Any]]] = {}
    for row in pass_rows:
        key = f"{row.get('engine')}:{row.get('runtime')}"
        by_engine.setdefault(key, []).append(row)
    summary: dict[str, Any] = {
        "axis": AXIS + "_summary",
        "status": "pass" if pass_rows else "skip",
        "rows": len(rows),
        "pass_rows": len(pass_rows),
        "engines": sorted(by_engine),
    }
    for key, engine_rows in sorted(by_engine.items()):
        decode_values = [float(row["decode_tok_s"]) for row in engine_rows if row.get("decode_tok_s") is not None]
        prefill_values = [float(row["prefill_tok_s"]) for row in engine_rows if row.get("prefill_tok_s") is not None]
        ttft_values = [float(row["ttft_s"]) for row in engine_rows if row.get("ttft_s") is not None]
        prefix = key.replace(":", "_").replace("/", "_")
        summary[f"{prefix}_min_decode_tok_s"] = round(min(decode_values), 6) if decode_values else None
        summary[f"{prefix}_min_prefill_tok_s"] = round(min(prefill_values), 6) if prefill_values else None
        summary[f"{prefix}_max_ttft_s"] = round(max(ttft_values), 6) if ttft_values else None
    return summary


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object row")
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Run/normalize Apple Qwen3.5 vs RWKV-7 baseline rows.")
    ap.add_argument("--results", default="bench/results_qwen35_apple_baseline.jsonl", help="JSONL path to append rows.")
    ap.add_argument("--summarize", default="", help="Summarize an existing JSONL file and exit.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned matrix without contacting runtimes.")
    ap.add_argument("--prompt-target-chars", default="1024", help="Comma-separated prompt text sizes.")
    ap.add_argument("--prompt-seed", default=DEFAULT_PROMPT_SEED)
    ap.add_argument("--decode-lengths", default="128", help="Comma-separated max_new_tokens values.")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--store-responses", action="store_true", help="Store full generated response text/token ids for quality evaluation rows.")
    ap.add_argument("--qwen-models", default=",".join(DEFAULT_QWEN_MODELS), help="Comma-separated Ollama Qwen3.5 models; empty disables Qwen.")
    ap.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--ollama-timeout-s", type=float, default=600.0)
    ap.add_argument("--ollama-think", action="store_true", help="Enable Qwen thinking output; disabled by default for comparable response rows.")
    ap.add_argument(
        "--ollama-keep-alive",
        default="0",
        help="Ollama keep_alive value. Default 0 unloads after each row to prevent prompt-cache prefill artifacts.",
    )
    ap.add_argument("--ollama-cache-prompt", action="store_true", help="Allow Ollama/runner prompt-cache reuse across rows.")
    ap.add_argument("--no-ollama-memory", action="store_true", help="Skip official /api/ps loaded-memory telemetry.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--rwkv-mlx-models", default="", help="Comma-separated converted RWKV-7 HF model dirs; empty disables RWKV MLX.")
    ap.add_argument("--rwkv-dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    ap.add_argument("--rwkv-quantization", default="none", choices=["none", "mm8", "mm4"])
    ap.add_argument("--rwkv-quant-min-params", type=int, default=8_000_000)
    ap.add_argument("--rwkv-quant-backend", default="auto", choices=["affine", "reference", "metal", "auto"])
    ap.add_argument("--rwkv-wkv-backend", default="auto", choices=["reference", "metal", "auto"])
    ap.add_argument("--rwkv-chunk-size", type=int, default=0)
    args = ap.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.summarize:
        rows = load_jsonl(args.summarize)
        print(json.dumps(summarize_rows(rows), ensure_ascii=False))
        return 0

    prompt_cases = build_prompt_cases(parse_int_csv(args.prompt_target_chars), args.prompt_seed)
    decode_lengths = parse_int_csv(args.decode_lengths)
    qwen_models = parse_csv(args.qwen_models)
    rwkv_models = parse_csv(args.rwkv_mlx_models)
    ollama_keep_alive = parse_keep_alive(args.ollama_keep_alive)

    env = {
        "axis": AXIS + "_env",
        "status": "info",
        "qwen_models": qwen_models,
        "rwkv_mlx_models": rwkv_models,
        "prompt_cases": [{"name": case.name, "target_chars": case.target_chars} for case in prompt_cases],
        "decode_lengths": decode_lengths,
        "repeat": int(args.repeat),
        "store_responses": bool(args.store_responses),
        "ollama_host": args.ollama_host,
        "ollama_think": bool(args.ollama_think),
        "ollama_keep_alive": ollama_keep_alive,
        "ollama_cache_prompt": bool(args.ollama_cache_prompt),
        "ollama_capture_memory": not bool(args.no_ollama_memory),
        **device_info(),
    }
    print(json.dumps(env, ensure_ascii=False))
    append_jsonl(args.results, env)

    if args.dry_run:
        plan = {
            "axis": AXIS + "_plan",
            "status": "plan",
            "qwen_jobs": len(qwen_models) * len(prompt_cases) * len(decode_lengths) * int(args.repeat),
            "rwkv_mlx_jobs": len(rwkv_models) * len(prompt_cases) * len(decode_lengths) * int(args.repeat),
            "qwen_models": qwen_models,
            "rwkv_mlx_models": rwkv_models,
            "prompt_cases": [{"name": case.name, "chars": len(case.prompt)} for case in prompt_cases],
            "decode_lengths": decode_lengths,
            "store_responses": bool(args.store_responses),
        }
        print(json.dumps(plan, ensure_ascii=False))
        append_jsonl(args.results, plan)
        return 0

    rows: list[dict[str, Any]] = []
    for case in prompt_cases:
        for model in qwen_models:
            rows.extend(
                run_ollama_qwen(
                    host=args.ollama_host,
                    model=model,
                    prompt_case=case,
                    decode_lengths=decode_lengths,
                    repeats=int(args.repeat),
                    temperature=float(args.temperature),
                    timeout_s=float(args.ollama_timeout_s),
                    results=args.results,
                    store_response=bool(args.store_responses),
                    think=bool(args.ollama_think),
                    keep_alive=ollama_keep_alive,
                    cache_prompt=bool(args.ollama_cache_prompt),
                    capture_memory=not bool(args.no_ollama_memory),
                )
            )
        for model_path in rwkv_models:
            rows.extend(
                run_rwkv_mlx(
                    model_path=model_path,
                    prompt_case=case,
                    decode_lengths=decode_lengths,
                    repeats=int(args.repeat),
                    dtype=args.rwkv_dtype,
                    quantization=args.rwkv_quantization,
                    quant_min_params=int(args.rwkv_quant_min_params),
                    quant_backend=args.rwkv_quant_backend,
                    wkv_backend=args.rwkv_wkv_backend,
                    chunk_size=int(args.rwkv_chunk_size),
                    results=args.results,
                    store_response=bool(args.store_responses),
                )
            )
    summary = summarize_rows(rows)
    print(json.dumps(summary, ensure_ascii=False))
    append_jsonl(args.results, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
