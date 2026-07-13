#!/usr/bin/env python3
"""Benchmark one RWKV-7 or official Qwen3.5 HF configuration.

The matrix orchestrator invokes this worker in a fresh process for every raw
row.  It intentionally benchmarks exact tensor shapes; model-quality evaluation
uses separate task runners.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("RWKV_V7_ON", "1")
QWEN_FORCE_TORCH = os.environ.get("RWKV7_QWEN35_FORCE_TORCH", "0").lower() in {"1", "true", "yes", "on"}
if QWEN_FORCE_TORCH:
    sys.path[:] = [path for path in sys.path if "flash-linear-attention" not in path.replace("\\", "/").lower()]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
PROMPT_SEED = (
    "RWKV and Qwen are language models evaluated with identical tensor shapes. "
    "This sentence is repeated only to build deterministic benchmark tokens. "
)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def validate_args(args: argparse.Namespace) -> None:
    for field in ("batch_size", "prompt_tokens", "decode_tokens", "runs"):
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if int(args.warmup) < 0:
        raise ValueError("--warmup must be non-negative")
    if args.model_role not in {"candidate", "reference"}:
        raise ValueError("--model-role must be candidate or reference")
    if args.model_kind not in {"rwkv", "qwen35"}:
        raise ValueError("--model-kind must be rwkv or qwen35")


def build_exact_prompt(tokenizer, prompt_tokens: int, batch_size: int, device: str) -> torch.Tensor:
    encoded = tokenizer(PROMPT_SEED * 32, return_tensors="pt", add_special_tokens=False).input_ids
    if encoded.ndim != 2 or encoded.shape[0] != 1 or encoded.shape[1] == 0:
        raise RuntimeError(f"tokenizer returned invalid input shape {tuple(encoded.shape)}")
    repeats = (prompt_tokens + int(encoded.shape[1]) - 1) // int(encoded.shape[1])
    ids = encoded.repeat(1, repeats)[:, :prompt_tokens].repeat(batch_size, 1)
    return ids.to(device) if device.startswith("cuda") else ids


def model_metadata(args: argparse.Namespace, model=None) -> dict[str, Any]:
    config = getattr(model, "config", None)
    return {
        "model_name": Path(args.model).name,
        "model_id_or_path": args.model,
        "model_size_label": args.model_size_label,
        "model_type": getattr(config, "model_type", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", getattr(config, "num_heads", None)),
        "head_dim": getattr(config, "head_dim", None),
    }


def base_row(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": args.benchmark_matrix,
        "model_pair": args.model_pair,
        "model_role": args.model_role,
        "model_kind": args.model_kind,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "qwen_backend_requested": args.qwen_backend,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        **model_metadata(args),
    }


def failure_row(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {
        **base_row(args),
        "status": "fail",
        "error_type": type(exc).__name__,
        "error": repr(exc),
    }


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_device_index(device: str) -> int:
    return int(device.split(":", 1)[1]) if ":" in device else 0


def device_map_for(device: str):
    return {"": cuda_device_index(device)} if device.startswith("cuda") else None


def device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(cuda_device_index(device))
    return device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated(cuda_device_index(device)) / 1024 / 1024, 1)


def model_footprint_mb(model) -> float:
    total = 0
    seen: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        ident = id(tensor)
        if ident in seen:
            continue
        seen.add(ident)
        total += tensor.numel() * tensor.element_size()
    return round(total / 1024 / 1024, 1)


def quantization_config(args: argparse.Namespace, dtype: torch.dtype):
    if args.quantization == "none":
        return None
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError("bitsandbytes is required for bnb8/bnb4 rows")
    from transformers import BitsAndBytesConfig

    if args.quantization == "bnb8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=False,
    )


def set_rwkv_runtime(model, args: argparse.Namespace) -> None:
    if not hasattr(model.config, "attn_mode"):
        return
    model.config.attn_mode = args.rwkv_attn_mode
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = args.rwkv_attn_mode


def prepare_rwkv_model_dir(model_path: str, code_source: str) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    if code_source == "model":
        return model_path, None
    source = Path(model_path).resolve()
    repo_code = Path(__file__).resolve().parents[1] / "rwkv7_hf"
    if not source.is_dir():
        raise ValueError("--rwkv-code-source repo requires a local converted model directory")
    temporary = tempfile.TemporaryDirectory(prefix="rwkv7_qwen35_repo_code_")
    target = Path(temporary.name)
    for item in source.iterdir():
        if item.name == "__pycache__" or item.suffix == ".py":
            continue
        link = target / item.name
        link.symlink_to(item, target_is_directory=item.is_dir())
    for py_file in repo_code.glob("*.py"):
        shutil.copy2(py_file, target / py_file.name)
    return str(target), temporary


def load_model(args: argparse.Namespace, dtype: torch.dtype, model_path: str | None = None):
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map_for(args.device),
        "low_cpu_mem_usage": True,
    }
    qconfig = quantization_config(args, dtype)
    if qconfig is not None:
        kwargs["quantization_config"] = qconfig
    if args.model_kind == "rwkv":
        kwargs["trust_remote_code"] = True
        model = AutoModelForCausalLM.from_pretrained(model_path or args.model, **kwargs).eval()
        set_rwkv_runtime(model, args)
        return model

    try:
        from transformers import Qwen3_5ForCausalLM
    except ImportError as exc:
        raise RuntimeError("installed Transformers does not provide Qwen3_5ForCausalLM") from exc
    return Qwen3_5ForCausalLM.from_pretrained(args.model, **kwargs).eval()


def last_rwkv_backend(model) -> str | None:
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


def step_function(model, model_kind: str, batch_size: int) -> tuple[Callable[..., Any], str]:
    if model_kind == "rwkv":
        fast = getattr(model, "rwkv7_forward_token", None)
        if fast is None and batch_size == 1:
            fast = getattr(model, "rwkv7_forward_one", None)
        if fast is not None:
            return lambda token, state: fast(token, past_key_values=state), "rwkv_fast_token"
    return (
        lambda token, state: model(token, past_key_values=state, use_cache=True, logits_to_keep=1),
        "module_call",
    )


def forward_prefill(model, ids):
    return model(ids, use_cache=True, logits_to_keep=1)


def timed_prefill(args: argparse.Namespace, model, ids) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = forward_prefill(model, ids)
    cuda_sync(args.device)
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(args.runs):
            cuda_sync(args.device)
            started = time.perf_counter()
            _ = forward_prefill(model, ids)
            cuda_sync(args.device)
            samples.append(time.perf_counter() - started)
    median_s = float(statistics.median(samples))
    return median_s, (args.batch_size * args.prompt_tokens) / median_s


def decode_once(args: argparse.Namespace, model, ids, step: Callable[..., Any]) -> tuple[float, Any]:
    with torch.inference_mode():
        out = forward_prefill(model, ids)
        state = out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.warmup):
            out = step(token, state)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        started = time.perf_counter()
        for _ in range(args.decode_tokens):
            out = step(token, state)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
    return time.perf_counter() - started, state


def timed_decode(args: argparse.Namespace, model, ids) -> tuple[float, str, str | None, str]:
    step, step_backend = step_function(model, args.model_kind, args.batch_size)
    samples: list[float] = []
    state = None
    for _ in range(args.runs):
        elapsed, state = decode_once(args, model, ids, step)
        samples.append(elapsed)
    return (
        float(statistics.median(samples)),
        step_backend,
        last_rwkv_backend(model),
        type(state).__name__ if state is not None else "None",
    )


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    qwen_fast_path = None
    if args.model_kind == "qwen35":
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

            qwen_fast_path = bool(is_fast_path_available)
        except Exception:
            qwen_fast_path = False
    return {
        "device": device_name(args.device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": package_version("transformers"),
        "bitsandbytes_version": package_version("bitsandbytes"),
        "fla_version": package_version("flash-linear-attention"),
        "causal_conv1d_version": package_version("causal-conv1d"),
        "qwen_fla_importable": importlib.util.find_spec("fla") is not None,
        "qwen_force_torch": QWEN_FORCE_TORCH,
        "qwen_fast_path_available": qwen_fast_path,
        "rwkv_fast_token_backend_requested": os.environ.get("RWKV7_FAST_TOKEN_BACKEND"),
    }


_QWEN35_FAST_BINDING_PREFIXES = {
    "causal_conv1d_fn": "causal_conv1d.",
    "causal_conv1d_update": "causal_conv1d.",
    "chunk_gated_delta_rule": "fla.",
    "recurrent_gated_delta_rule": "fla.",
}


def qwen35_fast_path_bindings(model) -> dict[str, Any]:
    """Verify the operators bound by live Qwen3.5 GatedDeltaNet layers.

    The Transformers module-level ``is_fast_path_available`` flag proves that
    optional packages imported, but not that a loaded layer retained those
    callables.  Inspecting the live bindings makes optimized-Qwen comparison a
    fail-closed contract rather than an availability hint.
    """

    layers = [
        module
        for module in model.modules()
        if all(hasattr(module, name) for name in _QWEN35_FAST_BINDING_PREFIXES)
    ]
    bindings: dict[str, str | None] = {}
    if layers:
        first = layers[0]
        for name in _QWEN35_FAST_BINDING_PREFIXES:
            fn = getattr(first, name, None)
            bindings[name] = getattr(fn, "__module__", None) if callable(fn) else None
    verified = bool(layers) and all(
        isinstance(bindings.get(name), str) and str(bindings[name]).startswith(prefix)
        for name, prefix in _QWEN35_FAST_BINDING_PREFIXES.items()
    )
    return {
        "verified": verified,
        "layer_count": len(layers),
        "bindings": bindings,
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dtype = DTYPES[args.dtype]
    started = time.perf_counter()
    effective_model_path = args.model
    temporary = None
    if args.model_kind == "rwkv":
        effective_model_path, temporary = prepare_rwkv_model_dir(args.model, args.rwkv_code_source)
    tokenizer = AutoTokenizer.from_pretrained(effective_model_path, trust_remote_code=args.model_kind == "rwkv")
    model = load_model(args, dtype, effective_model_path)
    if args.model_kind == "qwen35" and args.require_qwen_fast_path:
        from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

        if not bool(is_fast_path_available):
            raise RuntimeError(
                "Qwen3.5 optimized fast path is unavailable; install compatible "
                "flash-linear-attention and causal-conv1d packages"
            )
        binding_check = qwen35_fast_path_bindings(model)
        if not bool(binding_check["verified"]):
            raise RuntimeError(
                "Qwen3.5 optimized packages imported but the loaded GatedDeltaNet "
                f"layers are not bound to FLA/causal-conv1d: {binding_check}"
            )
    load_s = time.perf_counter() - started
    input_device = str(next(model.parameters()).device)
    ids = build_exact_prompt(tokenizer, args.prompt_tokens, args.batch_size, input_device)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(cuda_device_index(args.device))
    prefill_s, prefill_tokps = timed_prefill(args, model, ids)
    decode_s, step_backend, effective_backend, cache_type = timed_decode(args, model, ids)
    logits_finite = True
    with torch.inference_mode():
        check = forward_prefill(model, ids[:, : min(8, ids.shape[1])])
        logits_finite = bool(torch.isfinite(check.logits).all().item())
    if not logits_finite:
        raise RuntimeError("model produced non-finite logits")

    qwen_bindings = qwen35_fast_path_bindings(model) if args.model_kind == "qwen35" else None
    row = {
        **base_row(args),
        **model_metadata(args, model),
        **environment_metadata(args),
        "status": "pass",
        "input_device": input_device,
        "prefill_sec_median": round(prefill_s, 6),
        "prefill_tokps_total": round(prefill_tokps, 3),
        "decode_sec_median": round(decode_s, 6),
        "decode_tokps_total": round((args.batch_size * args.decode_tokens) / decode_s, 3),
        "decode_tokps_per_seq": round(args.decode_tokens / decode_s, 3),
        "decode_ms_per_step": round(1000 * decode_s / args.decode_tokens, 6),
        "step_backend": step_backend,
        "effective_backend": (
            "transformers_torch_fallback"
            if args.model_kind == "qwen35" and QWEN_FORCE_TORCH
            else (
                "fla+causal_conv1d"
                if args.model_kind == "qwen35" and qwen_bindings and qwen_bindings["verified"]
                else effective_backend or step_backend
            )
        ),
        "qwen_fast_path_verified": qwen_bindings["verified"] if qwen_bindings is not None else None,
        "qwen_fast_path_layer_count": qwen_bindings["layer_count"] if qwen_bindings is not None else None,
        "qwen_fast_path_bindings": qwen_bindings["bindings"] if qwen_bindings is not None else None,
        "cache_type": cache_type,
        "model_footprint_mb": model_footprint_mb(model),
        "peak_vram_mb": peak_mb(args.device),
        "load_s": round(load_s, 3),
        "logits_finite": logits_finite,
        "warmup": args.warmup,
        "runs": args.runs,
    }
    del model
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if temporary is not None:
        temporary.cleanup()
    return row


def append_row(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-kind", required=True, choices=["rwkv", "qwen35"])
    ap.add_argument("--model-role", required=True, choices=["candidate", "reference"])
    ap.add_argument("--model-pair", required=True)
    ap.add_argument("--model-size-label", required=True)
    ap.add_argument("--benchmark-matrix", default="qwen35_hf")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--quantization", default="none", choices=["none", "bnb8", "bnb4"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rwkv-attn-mode", choices=["chunk", "fused_recurrent"], default="fused_recurrent")
    ap.add_argument("--rwkv-code-source", choices=["repo", "model"], default="repo")
    ap.add_argument("--qwen-backend", choices=["auto", "torch"], default="auto")
    ap.add_argument("--require-qwen-fast-path", action="store_true")
    ap.add_argument("--results", default="")
    ap.add_argument("--optional", action="store_true")
    args = ap.parse_args()
    validate_args(args)
    return args


def main() -> int:
    args = parse_args()
    try:
        row = benchmark(args)
    except Exception as exc:
        row = failure_row(args, exc)
        append_row(args.results, row)
        print("QWEN35_CROSS_MODEL_SPEED_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
        if not args.optional:
            raise
        return 0
    append_row(args.results, row)
    print("QWEN35_CROSS_MODEL_SPEED_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
