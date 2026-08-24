#!/usr/bin/env python3
# coding=utf-8
"""Batch-size sweep benchmark for the RWKV-7 HF adapter.

Measures serving-style prefill and recurrent decode for multiple batch sizes.
The batched `rwkv7_forward_token` API is included when available; older adapter
builds fall back to the bsz=1 `rwkv7_forward_one` API.
"""
from __future__ import annotations

# Support direct ``python bench/<category>/<script>.py`` execution.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from bench.benchlib.paths import DEFAULT_RESULTS_PATH
from bench.benchlib.model_loader import load_tokenizer as load_hf_tokenizer
from bench.benchlib.results import append_jsonl

import argparse
import importlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from transformers import AutoModelForCausalLM

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 256
_FALSE_VALUES = {"0", "false", "False", "no", "off"}


def current_bench_case() -> str | None:
    return os.environ.get("RWKV7_BENCH_CASE")


def _model_kernel_policy(model):
    module = sys.modules.get(model.__class__.__module__)
    getter = getattr(module, "_rwkv7_kernel_policy", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    try:
        from rwkv7_hf.kernel_policy import current_kernel_policy

        return current_kernel_policy(
            device=next(model.parameters()).device,
            torch_module=torch,
        )
    except Exception:
        return None


def effective_flag(model, env_name: str, policy_attr: str, fallback: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is not None:
        return raw not in _FALSE_VALUES
    return bool(getattr(_model_kernel_policy(model), policy_attr, fallback))


def effective_wavg_lora(model, batch_size: int) -> bool:
    if not effective_flag(model, "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA", "fused_wavg_lora", False):
        return False
    policy = _model_kernel_policy(model)
    default_max = getattr(policy, "wavg_lora_bsz1_max_hidden", None)
    raw = os.environ.get("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BSZ1_MAX_HIDDEN")
    try:
        max_hidden = int(raw) if raw is not None else (0 if default_max is None else int(default_max))
    except ValueError:
        max_hidden = 0 if default_max is None else int(default_max)
    hidden = int(getattr(model.config, "hidden_size", 0))
    return not (int(batch_size) == 1 and max_hidden > 0 and hidden > max_hidden)


def effective_fused_norm_mix(model, batch_size: int) -> bool:
    """Report the shape-aware decode norm/mix route selected at runtime."""

    package = model.__class__.__module__.rsplit(".", 1)[0]
    try:
        native_jit = importlib.import_module(package + ".native_jit")
    except Exception:
        native_jit = None
    enabled = getattr(native_jit, "_native_graph_fused_norm_mix_enabled", None)
    if callable(enabled):
        try:
            return bool(
                enabled(
                    int(batch_size),
                    int(getattr(model.config, "hidden_size", 0)),
                )
            )
        except Exception:
            pass
    return effective_flag(
        model,
        "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX",
        "fused_norm_mix",
        False,
    )


def effective_fused_recurrent_raw(model, batch_size: int) -> bool:
    """Report the shape-aware raw recurrent route selected at runtime."""

    package = model.__class__.__module__.rsplit(".", 1)[0]
    try:
        native_jit = importlib.import_module(package + ".native_jit")
    except Exception:
        native_jit = None
    enabled = getattr(
        native_jit,
        "_native_graph_fused_recurrent_raw_enabled",
        None,
    )
    if callable(enabled):
        try:
            return bool(
                enabled(
                    int(batch_size),
                    int(getattr(model.config, "hidden_size", 0)),
                )
            )
        except Exception:
            pass
    return effective_flag(
        model,
        "RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW",
        "fused_recurrent_raw",
        False,
    )


def native_graph_wagv_bmm_route(model, batch_size: int) -> dict[str, Any]:
    """Report requested, selected, and captured B8 BMM state separately."""

    requested = effective_flag(
        model,
        "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM",
        "ada_wagv_bmm",
        False,
    )
    route = {
        "native_graph_ada_wagv_bmm": None,
        "native_graph_ada_wagv_bmm_requested": requested,
        "native_graph_ada_wagv_bmm_selected": None,
        "native_graph_ada_wagv_bmm_effective": None,
        "native_graph_ada_wagv_bmm_effective_layer_count": 0,
        "native_graph_ada_wagv_bmm_full_model_effective": None,
    }
    getter = getattr(model, "rwkv7_native_graph_runner_copy_stats", None)
    if not callable(getter):
        return route
    try:
        runners = getter().get("runners", [])
    except Exception:
        return route
    match = next(
        (
            row
            for row in reversed(runners)
            if int(row.get("batch_size", -1)) == int(batch_size)
        ),
        None,
    )
    if not isinstance(match, dict):
        return route
    route.update(
        {
            "native_graph_ada_wagv_bmm": bool(
                match.get("ada_wagv_bmm_effective", False)
            ),
            "native_graph_ada_wagv_bmm_requested": bool(
                match.get("ada_wagv_bmm_requested", requested)
            ),
            "native_graph_ada_wagv_bmm_selected": bool(
                match.get("ada_wagv_bmm_selected", False)
            ),
            "native_graph_ada_wagv_bmm_effective": bool(
                match.get("ada_wagv_bmm_effective", False)
            ),
            "native_graph_ada_wagv_bmm_effective_layer_count": int(
                match.get("ada_wagv_bmm_effective_layer_count", 0)
            ),
            "native_graph_ada_wagv_bmm_full_model_effective": bool(
                match.get("ada_wagv_bmm_full_model_effective", False)
            ),
        }
    )
    return route


def infer_model_size_label(hf_dir: str, explicit: str = "") -> str | None:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(hf_dir).name.lower())
    return match.group(1) if match else None


def model_metadata(args, model) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "model_name": Path(args.hf_dir).name,
        "model_size_label": infer_model_size_label(args.hf_dir, args.model_size_label),
        "hf_model_dir": args.hf_dir,
        "code_source": args.code_source,
        "hidden_size": getattr(cfg, "hidden_size", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "head_dim": getattr(cfg, "head_dim", None),
        "num_heads": getattr(cfg, "num_heads", None),
    }


def accelerator_api(device: str):
    if device.startswith("cuda"):
        return torch.cuda
    if device.startswith("musa"):
        return getattr(torch, "musa", None)
    if device.startswith("mps"):
        return getattr(torch, "mps", None)
    return None


def accelerator_device_arg(device: str):
    return torch.device(device) if device.startswith(("cuda", "musa")) else None


def device_sync(device: str) -> None:
    synchronize = getattr(accelerator_api(device), "synchronize", None)
    if callable(synchronize):
        target = accelerator_device_arg(device)
        synchronize(target) if target is not None else synchronize()


def device_name(device: str) -> str:
    getter = getattr(accelerator_api(device), "get_device_name", None)
    target = accelerator_device_arg(device)
    return str(getter(target)) if callable(getter) and target is not None else device


def peak_mb(device: str) -> float | None:
    maximum = getattr(accelerator_api(device), "max_memory_allocated", None)
    if not callable(maximum):
        return None
    target = accelerator_device_arg(device)
    value = maximum(target) if target is not None else maximum()
    return round(float(value) / 1024 / 1024, 1)


def native_graph_state_route(state) -> dict[str, Any]:
    """Report the state backend selected by a graph-bound native cache."""

    bound_runner = getattr(state, "_native_graph_bound_runner", None)
    runner = bound_runner() if callable(bound_runner) else None
    if runner is None:
        return {}
    return {
        "native_graph_state_dtype": str(getattr(runner, "state_dtype", "unknown")),
        "native_graph_triton_fp16_state": bool(
            getattr(runner, "triton_fp16_state", False)
        ),
        "native_graph_native_fp16_recurrent": bool(
            getattr(runner, "fp16_recurrent", False)
        ),
    }


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def timed(fn, device: str, runs: int) -> float:
    device_sync(device)
    t0 = time.time()
    # Prefill is an inference benchmark.  Keep the measured calls in the same
    # no-grad mode as warmup; otherwise the adapter correctly rejects its
    # inference-only native prefill route and the row silently measures a
    # different, much slower training-capable path.
    with torch.inference_mode():
        for _ in range(runs):
            fn()
    device_sync(device)
    return (time.time() - t0) / runs


def load_model(args, dtype):
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    os.environ["RWKV7_FAST_TOKEN_BACKEND"] = args.fast_token_backend
    model_cls = (
        NativeRWKV7ForCausalLM if args.code_source == "repo" else AutoModelForCausalLM
    )
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device if args.device.startswith("cuda") else None,
    }
    if args.code_source == "model":
        load_kwargs["trust_remote_code"] = True
    model = model_cls.from_pretrained(
        args.hf_dir,
        **load_kwargs,
    ).eval()
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")
    set_attn_mode(model, args.attn_mode)
    if not args.device.startswith("cuda"):
        model.to(args.device)
    return model


def load_tokenizer(args):
    if args.code_source == "repo":
        from rwkv7_hf.tokenization_rwkv7 import RWKV7Tokenizer

        return RWKV7Tokenizer.from_pretrained(args.hf_dir)
    return load_hf_tokenizer(args.hf_dir)


def last_fast_token_backend(model):
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


def last_fast_prefill_backend(model):
    for name in (
        "rwkv7_last_fast_prefill_backend",
        "rwkv7_native_model_last_prefill_backend",
    ):
        getter = getattr(model, name, None)
        if callable(getter):
            value = getter()
            if value is not None:
                return value
    return getattr(
        model,
        "_rwkv7_last_fast_prefill_backend",
        getattr(model, "_rwkv7_native_model_last_prefill_backend", None),
    )


def native_prefill_route(model, batch_size: int, prompt_tokens: int) -> dict[str, Any]:
    """Report the effective compiled-prefill route for one measured shape."""

    backend = last_fast_prefill_backend(model)
    route: dict[str, Any] = {
        "prefill_backend_effective": backend,
        "prefill_graph_requested": effective_flag(
            model, "RWKV7_NATIVE_PREFILL_GRAPH", "prefill_graph", False
        ),
        "prefill_graph_effective": backend == "native_prefill_graph",
        "prefill_fused_scan_requested": effective_flag(
            model,
            "RWKV7_NATIVE_PREFILL_FUSED_SCAN",
            "fused_prefill_scan",
            False,
        ),
        "prefill_sequence_ffn_effective": bool(
            getattr(model, "_rwkv7_native_prefill_sequence_ffn_effective", False)
        ),
        "prefill_fp16_accum_ffn_key_effective": bool(
            getattr(
                model,
                "_rwkv7_native_prefill_fp16_accum_ffn_key_effective",
                False,
            )
        ),
        "prefill_global_fp16_accum_effective": bool(
            getattr(
                model,
                "_rwkv7_native_prefill_global_fp16_accum_effective",
                False,
            )
        ),
    }
    method = getattr(model, "rwkv7_prefill_native", None)
    fn = getattr(method, "__func__", method)
    native_jit = getattr(fn, "__globals__", {}).get("native_jit")
    if native_jit is None:
        try:
            package = model.__class__.__module__.rsplit(".", 1)[0]
            native_jit = importlib.import_module(package + ".native_jit")
        except Exception:
            native_jit = None
    fused_scan = getattr(native_jit, "_native_prefill_fused_scan_enabled", None)
    if callable(fused_scan):
        try:
            route["prefill_fused_scan_effective"] = bool(
                fused_scan(
                    int(batch_size),
                    int(prompt_tokens),
                    int(model.config.hidden_size),
                    int(model.config.num_hidden_layers),
                )
            )
        except Exception:
            route["prefill_fused_scan_effective"] = None
    else:
        route["prefill_fused_scan_effective"] = None
    return route


def musa_wkv_route(model) -> dict[str, Any]:
    if not str(next(model.parameters()).device).startswith("musa"):
        return {}
    package = model.__class__.__module__.rsplit(".", 1)[0]
    wkv = importlib.import_module(package + ".musa_wkv")
    fused = importlib.import_module(package + ".musa_fused")
    return {
        "musa_wkv_mode": wkv._mode(),
        "musa_wkv_available": bool(wkv.musa_wkv_available(next(model.parameters()).device)),
        "musa_wkv_module_loaded": wkv._MODULE is not None,
        "musa_wkv_module_error": repr(wkv._MODULE_ERROR),
        "musa_attn_shift_mix_enabled": os.environ.get("RWKV7_MUSA_ATTN_SHIFT_MIX", "0").strip().lower() in {"1", "true", "yes", "on"},
        "musa_attn_shift_mix_module_loaded": fused._MODULE is not None,
        "musa_attn_shift_mix_module_error": repr(fused._MODULE_ERROR),
        "musa_attn_shift_mix_calls": int(fused._CALLS),
    }


@contextmanager
def reference_forward_env():
    old = os.environ.get("RWKV7_FAST_FORWARD")
    os.environ["RWKV7_FAST_FORWARD"] = "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RWKV7_FAST_FORWARD", None)
        else:
            os.environ["RWKV7_FAST_FORWARD"] = old


def encode(tok, prompt_tokens: int, bsz: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    ids = ids.repeat(bsz, 1)
    return ids.to(device)


def bench_one(args, tok, model, bsz: int) -> list[dict[str, Any]]:
    route_before = musa_wkv_route(model)
    calls_before = int(route_before.get("musa_attn_shift_mix_calls", 0))
    api = accelerator_api(args.device)
    empty_cache = getattr(api, "empty_cache", None)
    reset_peak = getattr(api, "reset_peak_memory_stats", None)
    target = accelerator_device_arg(args.device)
    if callable(empty_cache):
        empty_cache()
    if callable(reset_peak):
        reset_peak(target) if target is not None else reset_peak()
    ids = encode(tok, args.prompt_tokens, bsz, args.device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(ids, use_cache=True, logits_to_keep=args.hf_logits_to_keep)
        prefill_dt = timed(
            lambda: model(ids, use_cache=True, logits_to_keep=args.hf_logits_to_keep),
            args.device,
            args.runs,
        )
    prefill_route = native_prefill_route(model, bsz, int(ids.shape[1]))

    with torch.inference_mode():
        out = model(ids[:, :8], use_cache=True, logits_to_keep=args.hf_logits_to_keep)
        state = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.warmup):
            with reference_forward_env():
                out = model(nxt, past_key_values=state, use_cache=True, logits_to_keep=args.hf_logits_to_keep)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
        device_sync(args.device)
        t0 = time.time()
        for _ in range(args.decode_tokens):
            with reference_forward_env():
                out = model(nxt, past_key_values=state, use_cache=True, logits_to_keep=args.hf_logits_to_keep)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
        device_sync(args.device)
        decode_dt = time.time() - t0

    forward_route = musa_wkv_route(model)
    forward_route["musa_attn_shift_mix_calls_delta"] = int(
        forward_route.get("musa_attn_shift_mix_calls", 0)
    ) - calls_before
    rows = [{
        "axis": "batch_sweep",
        "backend": "hf_adapter",
        "bench_case": current_bench_case(),
        "decode_api": "forward",
        "dtype": args.dtype,
        "device": device_name(args.device),
        **model_metadata(args, model),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in {"0", "false", "False", "no", "off"},
        "cache_type": type(state).__name__ if state is not None else None,
        "batch_size": bsz,
        "prompt_tokens": int(ids.shape[1]),
        "decode_tokens": args.decode_tokens,
        "prefill_tokps_total": round((bsz * int(ids.shape[1])) / prefill_dt, 1),
        "prefill_tokps_per_seq": round(int(ids.shape[1]) / prefill_dt, 1),
        "prefill_ms": round(1000 * prefill_dt, 2),
        "decode_tokps_total": round((bsz * args.decode_tokens) / decode_dt, 1),
        "decode_tokps_per_seq": round(args.decode_tokens / decode_dt, 1),
        "decode_ms_per_step": round(1000 * decode_dt / args.decode_tokens, 2),
        "peak_vram_mb": peak_mb(args.device),
        **prefill_route,
        **native_graph_state_route(state),
        **forward_route,
    }]

    fast_fn = getattr(model, "rwkv7_forward_token", None)
    fast_name = "rwkv7_forward_token" if fast_fn is not None else None
    if fast_fn is None and bsz == 1:
        fast_fn = getattr(model, "rwkv7_forward_one", None)
        fast_name = "rwkv7_forward_one" if fast_fn is not None else None

    if args.fast_decode_api != "false" and fast_fn is not None:
        requested_backend = os.environ.get("RWKV7_FAST_TOKEN_BACKEND", "auto")
        with torch.inference_mode():
            out = model(ids[:, :8], use_cache=True, logits_to_keep=args.hf_logits_to_keep)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
            for _ in range(args.warmup):
                out = fast_fn(nxt, past_key_values=state)
                state = out.past_key_values
                nxt = out.logits[:, -1:].argmax(dim=-1)
            device_sync(args.device)
            t0 = time.time()
            for _ in range(args.decode_tokens):
                out = fast_fn(nxt, past_key_values=state)
                state = out.past_key_values
                nxt = out.logits[:, -1:].argmax(dim=-1)
            device_sync(args.device)
            fast_dt = time.time() - t0
        fast_route = musa_wkv_route(model)
        state_route = native_graph_state_route(state)
        fast_route["musa_attn_shift_mix_calls_delta"] = int(
            fast_route.get("musa_attn_shift_mix_calls", 0)
        ) - int(forward_route.get("musa_attn_shift_mix_calls", 0))
        bmm_route = native_graph_wagv_bmm_route(model, bsz)
        rows.append({**rows[0],
            "decode_api": fast_name,
            "fast_token_backend": requested_backend,
            "fast_token_backend_effective": last_fast_token_backend(model) or requested_backend,
            "native_graph_fused_recurrent": os.environ.get("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT", "0") not in {"0", "false", "False", "no", "off"},
            "native_graph_fused_recurrent_output": os.environ.get("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT", "1") not in {"0", "false", "False", "no", "off"},
            "native_graph_fused_recurrent_raw": effective_fused_recurrent_raw(model, bsz),
            "native_graph_fused_output": os.environ.get("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT", "1") not in {"0", "false", "False", "no", "off"},
            "native_graph_fused_output_project": os.environ.get("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT", "0") not in {"0", "false", "False", "no", "off"},
            "native_graph_fused_wag_lora": os.environ.get("RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA", "0") not in {"0", "false", "False", "no", "off"},
            "native_graph_fused_wavg_lora": effective_wavg_lora(model, bsz),
            "native_graph_fused_projection": os.environ.get("RWKV7_NATIVE_GRAPH_FUSED_PROJECTION", "0") not in {"0", "false", "False", "no", "off"},
            "native_graph_fused_norm_mix": effective_fused_norm_mix(model, bsz),
            "native_graph_sm70_linear": effective_flag(model, "RWKV7_NATIVE_GRAPH_SM70_LINEAR", "sm70_linear", False),
            "native_graph_ada_linear": effective_flag(model, "RWKV7_NATIVE_GRAPH_ADA_LINEAR", "ada_linear", False),
            "native_graph_ada_wagv_lora": effective_flag(model, "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA", "ada_wagv_lora", False),
            **bmm_route,
            "native_graph_ada_sparse_ffn": effective_flag(model, "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN", "ada_sparse_ffn", False),
            "native_graph_ada_sparse_ffn_max_rows": int(os.environ.get(
                "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_MAX_ROWS",
                str(getattr(_model_kernel_policy(model), "ada_sparse_ffn_max_rows", 19)),
            )),
            "native_graph_ada_sparse_ffn_inplace": effective_flag(
                model,
                "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_INPLACE",
                "ada_sparse_ffn_inplace",
                False,
            ),
            "decode_tokps_total": round((bsz * args.decode_tokens) / fast_dt, 1),
            "decode_tokps_per_seq": round(args.decode_tokens / fast_dt, 1),
            "decode_ms_per_step": round(1000 * fast_dt / args.decode_tokens, 2),
            "cache_type": type(state).__name__ if state is not None else None,
            "peak_vram_mb": peak_mb(args.device),
            **state_route,
            **fast_route,
        })
    elif args.fast_decode_api == "true":
        raise ValueError("Loaded model does not expose a fast one-token decode API for this batch size")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--model-size-label", default="", help="Optional size label such as 0.4b; inferred from --hf-dir when omitted")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--code-source",
        choices=("model", "repo"),
        default="model",
        help="load checkpoint-bundled remote code or the current repository implementation",
    )
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-decode-api", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-token-backend", choices=["auto", "fla", "native_jit", "native_graph"], default="auto",
                    help="Fast-token backend; native_graph captures one CUDA graph per fixed batch size")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--hf-logits-to-keep", type=int, default=1)
    ap.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    tok = load_tokenizer(args)
    model = load_model(args, dtype)
    all_rows: list[dict[str, Any]] = []
    for bsz in args.batch_sizes:
        print(f"\n===== batch_size={bsz} =====", flush=True)
        rows = bench_one(args, tok, model, bsz)
        all_rows.extend(rows)
        for row in rows:
            print(json.dumps(row, indent=2), flush=True)
    if args.results:
        for row in all_rows:
            append_jsonl(args.results, row)
        print(f"\nappended {len(all_rows)} rows -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
