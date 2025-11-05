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

DEFAULT_QWEN_MLX_VLM_MODELS = [
    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    "mlx-community/Qwen3.5-4B-MLX-4bit",
    "mlx-community/Qwen3.5-9B-MLX-4bit",
]

# Public package sizes from the Ollama qwen3.5 model page, used only as metadata.
QWEN35_PUBLIC_BASELINE = {
    "qwen3.5:0.8b-mlx": {"family": "qwen3.5", "size_class": "0.8B", "public_package_gb": 1.2},
    "qwen3.5:2b-mlx": {"family": "qwen3.5", "size_class": "2B", "public_package_gb": 3.1},
    "qwen3.5:4b-mlx": {"family": "qwen3.5", "size_class": "4B", "public_package_gb": 4.0},
    "qwen3.5:9b-mlx": {"family": "qwen3.5", "size_class": "9B", "public_package_gb": 8.9},
    "mlx-community/Qwen3.5-0.8B-MLX-4bit": {
        "family": "qwen3.5",
        "size_class": "0.8B",
        "public_package_gb": 0.63,
        "quantization": "4bit",
        "source": "huggingface_mlx_community",
    },
    "mlx-community/Qwen3.5-4B-MLX-4bit": {
        "family": "qwen3.5",
        "size_class": "4B",
        "public_package_gb": 2.4,
        "quantization": "4bit",
        "source": "huggingface_mlx_community",
    },
    "mlx-community/Qwen3.5-9B-MLX-4bit": {
        "family": "qwen3.5",
        "size_class": "9B",
        "public_package_gb": 5.7,
        "quantization": "4bit",
        "source": "huggingface_mlx_community",
    },
}


def qwen35_public_metadata(model: str) -> dict[str, Any]:
    if model in QWEN35_PUBLIC_BASELINE:
        return dict(QWEN35_PUBLIC_BASELINE[model])
    name = Path(str(model)).name.lower().replace("_", "-")
    if ("qwen3.5-0.8b" in name or "qwen35-0.8b" in name) and "mlx" in name:
        row = dict(QWEN35_PUBLIC_BASELINE["mlx-community/Qwen3.5-0.8B-MLX-4bit"])
        row["source"] = "huggingface_mlx_community_local"
        return row
    if ("qwen3.5-4b" in name or "qwen35-4b" in name) and "mlx" in name:
        row = dict(QWEN35_PUBLIC_BASELINE["mlx-community/Qwen3.5-4B-MLX-4bit"])
        row["source"] = "huggingface_mlx_community_local"
        return row
    if ("qwen3.5-9b" in name or "qwen35-9b" in name) and "mlx" in name:
        row = dict(QWEN35_PUBLIC_BASELINE["mlx-community/Qwen3.5-9B-MLX-4bit"])
        row["source"] = "huggingface_mlx_community_local"
        return row
    return {"family": "qwen3.5"}


def _mlx_vlm_add_special_tokens(model: Any, processor: Any) -> bool:
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type in ["gemma3", "gemma3n", "gemma4", "gemma4_unified"]:
        return getattr(processor, "chat_template", None) is None
    return True


def _token_only_mlx_vlm_generation(
    *,
    model: Any,
    processor: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_vlm.generate.ar import generate_step
    from mlx_vlm.utils import prepare_inputs

    reset_peak = getattr(mx, "reset_peak_memory", None)
    if callable(reset_peak):
        reset_peak()
    inputs = prepare_inputs(
        processor,
        images=None,
        audio=None,
        videos=None,
        prompts=prompt,
        image_token_index=getattr(getattr(model, "config", None), "image_token_index", None),
        add_special_tokens=_mlx_vlm_add_special_tokens(model, processor),
    )
    input_ids = inputs.get("input_ids")
    mask = inputs.get("attention_mask")
    if input_ids is None:
        raise ValueError("MLX-VLM token-only fallback produced no input_ids")

    prompt_tokens = int(input_ids.shape[-1])
    generated_ids: list[int] = []
    first_token_s = None
    t0 = time.perf_counter()
    for token, _logprobs in generate_step(
        input_ids,
        model,
        None,
        mask,
        max_tokens=int(max_new_tokens),
        temperature=float(temperature),
    ):
        if first_token_s is None:
            first_token_s = time.perf_counter() - t0
        generated_ids.append(int(token))
    wall_s = time.perf_counter() - t0
    try:
        peak_memory_gb = float(mx.get_peak_memory()) / 1e9
    except Exception:
        peak_memory_gb = None
    decode_s = max(float(wall_s) - float(first_token_s or 0.0), 0.0)
    return {
        "prompt_eval_tokens": int(prompt_tokens),
        "generated_tokens": int(len(generated_ids)),
        "wall_s": round(float(wall_s), 6),
        "first_token_s": round(float(first_token_s), 6) if first_token_s is not None else None,
        "ttft_s": round(float(first_token_s), 6) if first_token_s is not None else None,
        "decode_s": round(float(decode_s), 6),
        "prefill_tok_s": round(float(prompt_tokens) / float(first_token_s), 6) if first_token_s else None,
        "decode_tok_s": round(float(len(generated_ids)) / float(decode_s), 6) if decode_s > 0 else None,
        "mlx_vlm_chunk_count": int(len(generated_ids)),
        "mlx_peak_memory_gb": round(float(peak_memory_gb), 6) if peak_memory_gb is not None else None,
        "mlx_peak_memory_bytes": int(float(peak_memory_gb) * 1e9) if peak_memory_gb is not None else None,
        "generated_preview": generated_ids[:16],
        "response_preview": "",
        "response_chars": 0,
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


def post_ollama_generate(
    *,
    host: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    timeout_s: float,
) -> tuple[list[dict[str, Any]], float]:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
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
            chunks.append(json.loads(line))
            if chunks[-1].get("done"):
                break
    return chunks, time.perf_counter() - t0


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
    for idx, chunk in enumerate(chunks):
        if chunk.get("response"):
            first_response_chunk_index = idx
            break
    prompt_eval_count = _safe_int(final.get("prompt_eval_count"))
    eval_count = _safe_int(final.get("eval_count"))
    prompt_eval_duration = _safe_int(final.get("prompt_eval_duration"))
    eval_duration = _safe_int(final.get("eval_duration"))
    total_duration = _safe_int(final.get("total_duration"))
    load_duration = _safe_int(final.get("load_duration"))
    response_text = "".join(str(chunk.get("response", "")) for chunk in chunks)
    row = {
        "axis": AXIS,
        "status": "pass",
        "engine": "ollama",
        "runtime": "ollama_mlx",
        "model": model,
        **qwen35_public_metadata(model),
        "prompt_case": prompt_case.name,
        "prompt_target_chars": int(prompt_case.target_chars),
        "prompt_chars": len(prompt_case.prompt),
        "prompt_eval_tokens": prompt_eval_count,
        "generated_tokens": eval_count,
        "requested_generated_tokens": int(max_new_tokens),
        "wall_s": round(float(elapsed_s), 6),
        "total_duration_ns": total_duration,
        "load_duration_ns": load_duration,
        "prompt_eval_duration_ns": prompt_eval_duration,
        "eval_duration_ns": eval_duration,
        "prefill_tok_s": tok_s(prompt_eval_count, prompt_eval_duration),
        "decode_tok_s": tok_s(eval_count, eval_duration),
        "ollama_chunk_count": len(chunks),
        "first_response_chunk_index": first_response_chunk_index,
        "response_preview": response_text[:160],
        "response_chars": len(response_text),
    }
    if store_response:
        row["response_text"] = response_text
    return row


def run_ollama_qwen(
    *,
    host: str,
    model: str,
    prompt_case: PromptCase,
    decode_lengths: list[int],
    repeats: int,
    warmup_repeats: int,
    temperature: float,
    timeout_s: float,
    results: str,
    store_response: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    warmups = max(0, int(warmup_repeats))
    for max_new_tokens in decode_lengths:
        for iteration_index in range(1, warmups + repeats + 1):
            is_warmup = iteration_index <= warmups
            repeat_index = iteration_index - warmups
            try:
                chunks, elapsed_s = post_ollama_generate(
                    host=host,
                    model=model,
                    prompt=prompt_case.prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
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
                row["warmup_repeats"] = int(warmups)
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                row = {
                    "axis": AXIS,
                    "status": "skip",
                    "engine": "ollama",
                    "runtime": "ollama_mlx",
                    "model": model,
                    **qwen35_public_metadata(model),
                    "prompt_case": prompt_case.name,
                    "prompt_target_chars": int(prompt_case.target_chars),
                    "prompt_chars": len(prompt_case.prompt),
                    "requested_generated_tokens": int(max_new_tokens),
                    "repeat_index": int(repeat_index),
                    "repeat": int(repeats),
                    "warmup_repeats": int(warmups),
                    "reason": f"ollama request failed: {type(exc).__name__}: {exc}",
                }
            if is_warmup:
                continue
            print(json.dumps(row, ensure_ascii=False))
            append_jsonl(results, row)
            rows.append(row)
    return rows


def run_mlx_vlm_qwen(
    *,
    model_id: str,
    prompt_case: PromptCase,
    decode_lengths: list[int],
    repeats: int,
    warmup_repeats: int,
    temperature: float,
    results: str,
    store_response: bool = False,
    token_only: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        import mlx.core as mx
        from mlx_vlm import load, stream_generate
    except Exception as exc:  # pragma: no cover - optional Apple dependency
        for max_new_tokens in decode_lengths:
            row = {
                "axis": AXIS,
                "status": "skip",
                "engine": "mlx_vlm",
                "runtime": "mlx_vlm",
                "model": model_id,
                **qwen35_public_metadata(model_id),
                "prompt_case": prompt_case.name,
                "prompt_target_chars": int(prompt_case.target_chars),
                "prompt_chars": len(prompt_case.prompt),
                "requested_generated_tokens": int(max_new_tokens),
                "reason": f"MLX-VLM import prerequisites unavailable: {type(exc).__name__}: {exc}",
            }
            print(json.dumps(row, ensure_ascii=False))
            append_jsonl(results, row)
            rows.append(row)
        return rows

    try:
        t_load = time.perf_counter()
        model, processor = load(model_id)
        try:
            mx.eval(model.parameters())
        except Exception:
            pass
        load_s = time.perf_counter() - t_load
    except Exception as exc:
        for max_new_tokens in decode_lengths:
            row = {
                "axis": AXIS,
                "status": "skip",
                "engine": "mlx_vlm",
                "runtime": "mlx_vlm",
                "model": model_id,
                **qwen35_public_metadata(model_id),
                "prompt_case": prompt_case.name,
                "prompt_target_chars": int(prompt_case.target_chars),
                "prompt_chars": len(prompt_case.prompt),
                "requested_generated_tokens": int(max_new_tokens),
                "reason": f"MLX-VLM model load failed: {type(exc).__name__}: {exc}",
            }
            print(json.dumps(row, ensure_ascii=False))
            append_jsonl(results, row)
            rows.append(row)
        return rows

    warmups = max(0, int(warmup_repeats))
    for max_new_tokens in decode_lengths:
        for iteration_index in range(1, warmups + repeats + 1):
            is_warmup = iteration_index <= warmups
            repeat_index = iteration_index - warmups
            try:
                if token_only:
                    metrics = _token_only_mlx_vlm_generation(
                        model=model,
                        processor=processor,
                        prompt=prompt_case.prompt,
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                    )
                    row = {
                        "axis": AXIS,
                        "status": "pass",
                        "engine": "mlx_vlm",
                        "runtime": "mlx_vlm_token_only",
                        "model": model_id,
                        **qwen35_public_metadata(model_id),
                        "prompt_case": prompt_case.name,
                        "prompt_target_chars": int(prompt_case.target_chars),
                        "prompt_chars": len(prompt_case.prompt),
                        "requested_generated_tokens": int(max_new_tokens),
                        "repeat_index": int(repeat_index),
                        "repeat": int(repeats),
                        "warmup_repeats": int(warmups),
                        "load_s": round(float(load_s), 6),
                        **metrics,
                    }
                    if store_response:
                        row["response_text"] = ""
                else:
                    reset_peak = getattr(mx, "reset_peak_memory", None)
                    if callable(reset_peak):
                        reset_peak()
                    response_text = ""
                    first_token_s = None
                    last_response = None
                    chunk_count = 0
                    t0 = time.perf_counter()
                    for response in stream_generate(
                        model,
                        processor,
                        prompt_case.prompt,
                        image=None,
                        max_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        verbose=False,
                    ):
                        chunk_count += 1
                        if not getattr(response, "is_draft", False):
                            token = getattr(response, "token", None)
                            text = str(getattr(response, "text", "") or "")
                            if first_token_s is None and (token is not None or text):
                                first_token_s = time.perf_counter() - t0
                            response_text += text
                            last_response = response
                    wall_s = time.perf_counter() - t0
                    prompt_tokens = _safe_int(getattr(last_response, "prompt_tokens", None))
                    generated_tokens = _safe_int(getattr(last_response, "generation_tokens", None))
                    prompt_tps = _safe_float(getattr(last_response, "prompt_tps", None))
                    generation_tps = _safe_float(getattr(last_response, "generation_tps", None))
                    peak_memory_gb = _safe_float(getattr(last_response, "peak_memory", None))
                    if not peak_memory_gb:
                        try:
                            peak_memory_gb = float(mx.get_peak_memory()) / 1e9
                        except Exception:
                            peak_memory_gb = None
                    row = {
                        "axis": AXIS,
                        "status": "pass",
                        "engine": "mlx_vlm",
                        "runtime": "mlx_vlm",
                        "model": model_id,
                        **qwen35_public_metadata(model_id),
                        "prompt_case": prompt_case.name,
                        "prompt_target_chars": int(prompt_case.target_chars),
                        "prompt_chars": len(prompt_case.prompt),
                        "prompt_eval_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                        "requested_generated_tokens": int(max_new_tokens),
                        "repeat_index": int(repeat_index),
                        "repeat": int(repeats),
                        "warmup_repeats": int(warmups),
                        "load_s": round(float(load_s), 6),
                        "wall_s": round(float(wall_s), 6),
                        "first_token_s": round(float(first_token_s), 6) if first_token_s is not None else None,
                        "ttft_s": round(float(first_token_s), 6) if first_token_s is not None else None,
                        "prefill_tok_s": round(float(prompt_tps), 6) if prompt_tps else None,
                        "decode_tok_s": round(float(generation_tps), 6) if generation_tps else None,
                        "mlx_vlm_chunk_count": int(chunk_count),
                        "mlx_peak_memory_gb": round(float(peak_memory_gb), 6) if peak_memory_gb is not None else None,
                        "mlx_peak_memory_bytes": int(float(peak_memory_gb) * 1e9) if peak_memory_gb is not None else None,
                        "response_preview": response_text[:160],
                        "response_chars": len(response_text),
                    }
                    if store_response:
                        row["response_text"] = response_text
            except UnicodeDecodeError as exc:
                metrics = _token_only_mlx_vlm_generation(
                    model=model,
                    processor=processor,
                    prompt=prompt_case.prompt,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                )
                row = {
                    "axis": AXIS,
                    "status": "pass",
                    "engine": "mlx_vlm",
                    "runtime": "mlx_vlm_token_only",
                    "model": model_id,
                    **qwen35_public_metadata(model_id),
                    "prompt_case": prompt_case.name,
                    "prompt_target_chars": int(prompt_case.target_chars),
                    "prompt_chars": len(prompt_case.prompt),
                    "requested_generated_tokens": int(max_new_tokens),
                    "repeat_index": int(repeat_index),
                    "repeat": int(repeats),
                    "warmup_repeats": int(warmups),
                    "load_s": round(float(load_s), 6),
                    "text_decode_error": f"{type(exc).__name__}: {exc}",
                    "fallback_after_text_decode_error": True,
                    **metrics,
                }
                if store_response:
                    row["response_text"] = ""
            except Exception as exc:
                row = {
                    "axis": AXIS,
                    "status": "skip",
                    "engine": "mlx_vlm",
                    "runtime": "mlx_vlm",
                    "model": model_id,
                    **qwen35_public_metadata(model_id),
                    "prompt_case": prompt_case.name,
                    "prompt_target_chars": int(prompt_case.target_chars),
                    "prompt_chars": len(prompt_case.prompt),
                    "requested_generated_tokens": int(max_new_tokens),
                    "repeat_index": int(repeat_index),
                    "repeat": int(repeats),
                    "warmup_repeats": int(warmups),
                    "reason": f"MLX-VLM generation failed: {type(exc).__name__}: {exc}",
                }
            if is_warmup:
                continue
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
    warmup_repeats: int,
    dtype: str,
    quantization: str,
    quant_min_params: int,
    quant_rkv_min_params: int | None,
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
        quant_rkv_min_params=quant_rkv_min_params,
        quant_backend=quant_backend,
        wkv_backend=wkv_backend,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    load_s = time.perf_counter() - t_load
    prompt_ids = [int(x) for x in tokenizer(prompt_case.prompt, add_special_tokens=False).input_ids]
    if not prompt_ids:
        raise ValueError("RWKV tokenizer produced zero prompt tokens")

    warmups = max(0, int(warmup_repeats))
    for max_new_tokens in decode_lengths:
        for iteration_index in range(1, warmups + repeats + 1):
            is_warmup = iteration_index <= warmups
            repeat_index = iteration_index - warmups
            reset_counters = getattr(model, "reset_telemetry_counters", None)
            if callable(reset_counters):
                reset_counters()
            reset_mlx_peak_memory()
            t_prefill = time.perf_counter()
            logits, state = model.prefill([prompt_ids])
            # model.prefill()/forward() already synchronizes the returned logits
            # and recurrent state.  A second mx.eval(logits) here only adds a
            # redundant host/device barrier to TTFT measurement.
            prefill_logits = logits
            prefill_s = time.perf_counter() - t_prefill
            chunk_diff = None
            chunk_s = None
            chunk_telemetry = None
            chunk_memory = None
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
                next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
                mx.eval(next_token)
                decode_step_s += time.perf_counter() - t_step
            decode_s = first_s + decode_step_s
            response_text = tokenizer.decode(generated_preview, skip_special_tokens=True) if store_response else ""
            telemetry = model.telemetry()
            main_memory = mlx_memory_telemetry()
            if int(chunk_size) > 0:
                reset_counters = getattr(model, "reset_telemetry_counters", None)
                if callable(reset_counters):
                    reset_counters()
                reset_mlx_peak_memory()
                t_chunk = time.perf_counter()
                chunk_logits, chunk_state = model.chunked_prefill([prompt_ids], chunk_size=int(chunk_size))
                # chunked_prefill() also synchronizes final logits/state through
                # forward(); avoid a second barrier before measuring elapsed.
                chunk_s = time.perf_counter() - t_chunk
                chunk_diff = float(mx.max(mx.abs(prefill_logits.astype(mx.float32) - chunk_logits.astype(mx.float32))))
                if int(chunk_state.seen_tokens) != len(prompt_ids):
                    raise AssertionError(
                        f"chunked state seen_tokens={chunk_state.seen_tokens}, expected {len(prompt_ids)}"
                    )
                chunk_telemetry = model.telemetry()
                chunk_memory = mlx_memory_telemetry()
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
                "quant_rkv_min_params": quant_rkv_min_params,
                "quant_backend": quant_backend,
                "wkv_backend": wkv_backend,
                "wkv_backend_last": telemetry.get("wkv_backend_last"),
                "wkv_backend_counts": telemetry.get("wkv_backend_counts"),
                "wkv_metal_available": telemetry.get("wkv_metal_available"),
                "step_eval_interval": telemetry.get("step_eval_interval"),
                "fused_ffn_key_relu2": telemetry.get("fused_ffn_key_relu2"),
                "fused_ffn_key_relu2_counts": telemetry.get("fused_ffn_key_relu2_counts"),
                "fused_attn_mix": telemetry.get("fused_attn_mix"),
                "fused_attn_mix_counts": telemetry.get("fused_attn_mix_counts"),
                "fused_attn_mix_metal_available": telemetry.get("fused_attn_mix_metal_available"),
                "fast_layer_norm": telemetry.get("fast_layer_norm"),
                "fast_group_norm": telemetry.get("fast_group_norm"),
                "wkv_scan_prefill": telemetry.get("wkv_scan_prefill"),
                "wkv_scan_prefill_mode": telemetry.get("wkv_scan_prefill_mode"),
                "wkv_scan_prefill_min_tokens": telemetry.get("wkv_scan_prefill_min_tokens"),
                "wkv_scan_prefill_counts": telemetry.get("wkv_scan_prefill_counts"),
                "wkv_scan_prefill_reason_counts": telemetry.get("wkv_scan_prefill_reason_counts"),
                "wkv_scan_metal_available": telemetry.get("wkv_scan_metal_available"),
                "prompt_case": prompt_case.name,
                "prompt_target_chars": int(prompt_case.target_chars),
                "prompt_chars": len(prompt_case.prompt),
                "prompt_eval_tokens": len(prompt_ids),
                "generated_tokens": int(max_new_tokens),
                "requested_generated_tokens": int(max_new_tokens),
                "repeat_index": int(repeat_index),
                "repeat": int(repeats),
                "warmup_repeats": int(warmups),
                "load_s": round(float(load_s), 6),
                "prefill_s": round(float(prefill_s), 6),
                "first_token_s": round(float(first_s), 6),
                "ttft_s": round(float(prefill_s + first_s), 6),
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
                **main_memory,
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
                        "chunked_wkv_backend_counts": (chunk_telemetry or {}).get("wkv_backend_counts"),
                        "chunked_fused_ffn_key_relu2_counts": (chunk_telemetry or {}).get("fused_ffn_key_relu2_counts"),
                        "chunked_fused_attn_mix_counts": (chunk_telemetry or {}).get("fused_attn_mix_counts"),
                        "chunked_wkv_scan_prefill_counts": (chunk_telemetry or {}).get("wkv_scan_prefill_counts"),
                        "chunked_wkv_scan_prefill_reason_counts": (chunk_telemetry or {}).get("wkv_scan_prefill_reason_counts"),
                        "chunked_state_only_prefill_calls": (chunk_telemetry or {}).get("state_only_prefill_calls"),
                        "chunked_state_only_prefill_tokens": (chunk_telemetry or {}).get("state_only_prefill_tokens"),
                        "chunked_quantized_linear_last_backend_counts": (chunk_telemetry or {}).get("quantized_linear_last_backend_counts"),
                        "chunked_group_rkv_quant_projection_counts": (chunk_telemetry or {}).get("group_rkv_quant_projection_counts"),
                        "chunked_mlx_peak_memory_bytes": (chunk_memory or {}).get("mlx_peak_memory_bytes"),
                    }
                )
            if is_warmup:
                continue
            print(json.dumps(row, ensure_ascii=False))
            append_jsonl(results, row)
            rows.append(row)
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
    ap.add_argument("--warmup-repeats", type=int, default=0, help="Unrecorded warmup generations per prompt/decode length before measured repeats.")
    ap.add_argument("--store-responses", action="store_true", help="Store full generated response text/token ids for quality evaluation rows.")
    ap.add_argument("--qwen-models", default=",".join(DEFAULT_QWEN_MODELS), help="Comma-separated Ollama Qwen3.5 models; empty disables Qwen.")
    ap.add_argument(
        "--qwen-mlx-vlm-models",
        default="",
        help=(
            "Comma-separated Hugging Face MLX-VLM Qwen3.5 model ids or local dirs; "
            "empty disables this fallback baseline lane."
        ),
    )
    ap.add_argument(
        "--qwen-mlx-vlm-token-only",
        action="store_true",
        help=(
            "Benchmark MLX-VLM Qwen models by generated token ids only. This avoids text detokenizer "
            "UnicodeDecodeError failures and records speed/memory rows without response text."
        ),
    )
    ap.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--ollama-timeout-s", type=float, default=600.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--rwkv-mlx-models", default="", help="Comma-separated converted RWKV-7 HF model dirs; empty disables RWKV MLX.")
    ap.add_argument("--rwkv-dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    ap.add_argument("--rwkv-quantization", default="none", choices=["none", "mm8", "mm4"])
    ap.add_argument("--rwkv-quant-min-params", type=int, default=8_000_000)
    ap.add_argument(
        "--rwkv-quant-rkv-min-params",
        type=int,
        default=-1,
        help=(
            "Optional separate min-params threshold for attention r/k/v projection quantization. "
            "Use 0 with RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 to exercise fused quant RKV; "
            "-1 preserves --rwkv-quant-min-params."
        ),
    )
    ap.add_argument("--rwkv-quant-backend", default="auto", choices=["affine", "reference", "metal", "auto"])
    ap.add_argument("--rwkv-wkv-backend", default="auto", choices=["reference", "metal", "auto"])
    ap.add_argument("--rwkv-chunk-size", type=int, default=0)
    args = ap.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.warmup_repeats < 0:
        raise ValueError("--warmup-repeats must be non-negative")
    if args.summarize:
        rows = load_jsonl(args.summarize)
        print(json.dumps(summarize_rows(rows), ensure_ascii=False))
        return 0

    prompt_cases = build_prompt_cases(parse_int_csv(args.prompt_target_chars), args.prompt_seed)
    decode_lengths = parse_int_csv(args.decode_lengths)
    qwen_models = parse_csv(args.qwen_models)
    qwen_mlx_vlm_models = parse_csv(args.qwen_mlx_vlm_models)
    rwkv_models = parse_csv(args.rwkv_mlx_models)

    env = {
        "axis": AXIS + "_env",
        "status": "info",
        "qwen_models": qwen_models,
        "qwen_mlx_vlm_models": qwen_mlx_vlm_models,
        "qwen_mlx_vlm_token_only": bool(args.qwen_mlx_vlm_token_only),
        "rwkv_mlx_models": rwkv_models,
        "prompt_cases": [{"name": case.name, "target_chars": case.target_chars} for case in prompt_cases],
        "decode_lengths": decode_lengths,
        "repeat": int(args.repeat),
        "warmup_repeats": int(args.warmup_repeats),
        "store_responses": bool(args.store_responses),
        "ollama_host": args.ollama_host,
        "rwkv_quant_min_params": int(args.rwkv_quant_min_params),
        "rwkv_quant_rkv_min_params": None
        if int(args.rwkv_quant_rkv_min_params) < 0
        else int(args.rwkv_quant_rkv_min_params),
        "rwkv_step_eval_interval_env": os.environ.get("RWKV7_MLX_STEP_EVAL_INTERVAL", ""),
        "rwkv_fused_ffn_key_relu2_env": os.environ.get("RWKV7_MLX_FUSED_FFN_KEY_RELU2", ""),
        "rwkv_wkv_scan_prefill_env": os.environ.get("RWKV7_MLX_WKV_SCAN_PREFILL", ""),
        "rwkv_wkv_scan_prefill_min_tokens_env": os.environ.get("RWKV7_MLX_WKV_SCAN_PREFILL_MIN_TOKENS", ""),
        **device_info(),
    }
    print(json.dumps(env, ensure_ascii=False))
    append_jsonl(args.results, env)

    if args.dry_run:
        plan = {
            "axis": AXIS + "_plan",
            "status": "plan",
            "qwen_jobs": len(qwen_models) * len(prompt_cases) * len(decode_lengths) * int(args.repeat),
            "qwen_mlx_vlm_jobs": len(qwen_mlx_vlm_models)
            * len(prompt_cases)
            * len(decode_lengths)
            * int(args.repeat),
            "rwkv_mlx_jobs": len(rwkv_models) * len(prompt_cases) * len(decode_lengths) * int(args.repeat),
            "warmup_jobs": (
                len(prompt_cases)
                * len(decode_lengths)
                * int(args.warmup_repeats)
                * (len(qwen_models) + len(qwen_mlx_vlm_models) + len(rwkv_models))
            ),
            "warmup_repeats": int(args.warmup_repeats),
            "qwen_models": qwen_models,
            "qwen_mlx_vlm_models": qwen_mlx_vlm_models,
            "qwen_mlx_vlm_token_only": bool(args.qwen_mlx_vlm_token_only),
            "rwkv_mlx_models": rwkv_models,
            "prompt_cases": [{"name": case.name, "chars": len(case.prompt)} for case in prompt_cases],
            "decode_lengths": decode_lengths,
            "store_responses": bool(args.store_responses),
            "rwkv_quant_min_params": int(args.rwkv_quant_min_params),
            "rwkv_quant_rkv_min_params": None
            if int(args.rwkv_quant_rkv_min_params) < 0
            else int(args.rwkv_quant_rkv_min_params),
            "rwkv_step_eval_interval_env": os.environ.get("RWKV7_MLX_STEP_EVAL_INTERVAL", ""),
            "rwkv_fused_ffn_key_relu2_env": os.environ.get("RWKV7_MLX_FUSED_FFN_KEY_RELU2", ""),
        "rwkv_wkv_scan_prefill_env": os.environ.get("RWKV7_MLX_WKV_SCAN_PREFILL", ""),
        "rwkv_wkv_scan_prefill_min_tokens_env": os.environ.get("RWKV7_MLX_WKV_SCAN_PREFILL_MIN_TOKENS", ""),
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
                    warmup_repeats=int(args.warmup_repeats),
                    temperature=float(args.temperature),
                    timeout_s=float(args.ollama_timeout_s),
                    results=args.results,
                    store_response=bool(args.store_responses),
                )
            )
        for model in qwen_mlx_vlm_models:
            rows.extend(
                run_mlx_vlm_qwen(
                    model_id=model,
                    prompt_case=case,
                    decode_lengths=decode_lengths,
                    repeats=int(args.repeat),
                    warmup_repeats=int(args.warmup_repeats),
                    temperature=float(args.temperature),
                    results=args.results,
                    store_response=bool(args.store_responses),
                    token_only=bool(args.qwen_mlx_vlm_token_only),
                )
            )
        for model_path in rwkv_models:
            rows.extend(
                run_rwkv_mlx(
                    model_path=model_path,
                    prompt_case=case,
                    decode_lengths=decode_lengths,
                    repeats=int(args.repeat),
                    warmup_repeats=int(args.warmup_repeats),
                    dtype=args.rwkv_dtype,
                    quantization=args.rwkv_quantization,
                    quant_min_params=int(args.rwkv_quant_min_params),
                    quant_rkv_min_params=(
                        None if int(args.rwkv_quant_rkv_min_params) < 0 else int(args.rwkv_quant_rkv_min_params)
                    ),
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
