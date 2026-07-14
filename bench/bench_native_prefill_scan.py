#!/usr/bin/env python3
# coding=utf-8
"""Benchmark the native layer-wise prefill path and optional fused scan.

Rows from this script are the end-to-end prefill counterpart to
`bench_fused_recurrent_scan.py`: the recurrent scan kernel is useful only if it
survives full-layer projection/output/FFN overhead and produces a cache that the
native_graph decode path can continue from.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rwkv7_hf import native_jit


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 256


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CODE_DIR = REPO_ROOT / "rwkv7_hf"


def prepare_model_dir(model_path: str, *, code_source: str) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    """Return a model directory, optionally overlaying repo remote-code files.

    Local HF checkpoints usually carry their own ``modeling_rwkv7.py`` and
    ``native_jit.py`` next to the weights.  That is correct for release
    validation, but it hides in-repo experiments such as ``dplr_prefill.py`` and
    ``dplr_prefill_triton.py`` from ``trust_remote_code=True``.  ``code_source=
    repo`` creates a temporary checkpoint directory that symlinks the original
    non-code files and copies the current repo's ``rwkv7_hf/*.py`` files to the
    checkpoint root, so the benchmark measures the current worktree code.
    """

    if code_source == "model":
        return model_path, None
    if code_source != "repo":
        raise ValueError(f"code_source must be 'model' or 'repo', got {code_source!r}")

    src = Path(model_path).resolve()
    if not src.is_dir():
        raise ValueError(f"--model must be a local directory for --code-source repo; got {model_path!r}")
    if not REPO_CODE_DIR.is_dir():
        raise ValueError(f"repo code directory not found: {REPO_CODE_DIR}")

    # Keep the staging directory on the checkpoint volume so Windows can use
    # hardlinks when developer-mode symlink privileges are unavailable.
    tmp = tempfile.TemporaryDirectory(prefix="rwkv7_repo_code_model_", dir=src.parent)
    dst = Path(tmp.name)
    for item in src.iterdir():
        if item.name == "__pycache__" or item.suffix == ".py":
            continue
        target = dst / item.name
        try:
            target.symlink_to(item, target_is_directory=item.is_dir())
        except OSError:
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                os.link(item, target)
    for py_file in REPO_CODE_DIR.glob("*.py"):
        shutil.copy2(py_file, dst / py_file.name)
    return str(dst), tmp


def model_native_jit_module(model) -> Any:
    """Return the native_jit module actually used by the loaded HF model."""

    method = getattr(model, "rwkv7_prefill_native", None)
    fn = getattr(method, "__func__", method)
    globals_dict = getattr(fn, "__globals__", {})
    return globals_dict.get("native_jit", native_jit)


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def wall_ms(fn: Callable[[], Any], device: str) -> float:
    cuda_sync(device)
    t0 = time.perf_counter()
    fn()
    cuda_sync(device)
    return (time.perf_counter() - t0) * 1000.0


def measured_ms(fn: Callable[[], Any], device: str, timing: str) -> float:
    """Match Albatross' CUDA-event graph timing while retaining wall fallback."""

    if timing != "cuda-event" or not device.startswith("cuda"):
        return wall_ms(fn, device)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    fn()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end))


@contextmanager
def env_override(**values: str | None):
    """Temporarily set capture-affecting variables without leaking cases."""

    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def build_ids(tok, batch_size: int, prompt_tokens: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    if int(ids.shape[1]) < prompt_tokens:
        raise ValueError(f"seed produced only {ids.shape[1]} tokens, need {prompt_tokens}")
    return ids.repeat(batch_size, 1).to(device)


def median(vals: list[float]) -> float:
    vals = sorted(vals)
    return vals[len(vals) // 2]


def infer_model_size_label(model_path: str) -> str | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", str(model_path).lower())
    return f"{match.group(1)}b" if match else None


def scan_block_m(model, batch_size: int | None = None) -> int | None:
    raw = os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    try:
        head_dim = int(model._rwkv7_native_jit_packs()[0][2])
        return model_native_jit_module(model)._native_prefill_scan_block_m(head_dim, batch_size)
    except Exception:
        return None


def scan_num_warps(model, block_m: int | None) -> int | None:
    raw = os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    try:
        head_dim = int(model._rwkv7_native_jit_packs()[0][2])
        return model_native_jit_module(model)._native_prefill_scan_num_warps(head_dim, block_m)
    except Exception:
        return None


def cosine_min(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=-1).min().detach().cpu())


def _tensor_payload_bytes(tensor, seen: set[int]) -> int:
    ident = id(tensor)
    if ident in seen:
        return 0
    seen.add(ident)
    flatten = getattr(tensor, "__tensor_flatten__", None)
    if callable(flatten) and type(tensor) not in {torch.Tensor, torch.nn.Parameter}:
        try:
            payload = sum(
                _tensor_payload_bytes(getattr(tensor, name), seen)
                for name in flatten()[0]
                if isinstance(getattr(tensor, name), torch.Tensor)
            )
            if payload:
                return payload
        except Exception:
            pass
    return int(tensor.numel()) * int(tensor.element_size())


def model_payload_mb(model) -> float:
    seen: set[int] = set()
    total = sum(_tensor_payload_bytes(tensor, seen) for tensor in list(model.parameters()) + list(model.buffers()))
    return round(total / 1024 / 1024, 1)


def run_case(args: argparse.Namespace, tok, model, batch_size: int, prompt_tokens: int) -> dict[str, Any]:
    nj = model_native_jit_module(model)
    quantizer = getattr(model, "hf_quantizer", None)
    quant_config = getattr(quantizer, "quantization_config", None)
    if quant_config is None:
        quant_config = getattr(getattr(model, "config", None), "quantization_config", None)
    quant_get = (
        quant_config.get
        if isinstance(quant_config, dict)
        else lambda name, default=None: getattr(quant_config, name, default)
    )
    ids = build_ids(tok, batch_size, prompt_tokens, args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def hf_reference_call():
        return model(ids, use_cache=True, logits_to_keep=1, return_dict=True)

    def native_direct_reference_call():
        return model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)

    def candidate_call():
        # Match Albatross graph timing by entering the dedicated native prefill
        # helper directly; policy still selects/caches the fixed-shape runner.
        return model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)

    reference_call = hf_reference_call if args.reference_backend == "hf" else native_direct_reference_call
    reference_env = (
        {"RWKV7_FAST_PREFILL": "0"}
        if args.reference_backend == "hf"
        else {
            "RWKV7_FAST_PREFILL": "1",
            "RWKV7_NATIVE_PREFILL_GRAPH": "0",
            "RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX": "0",
        }
    )
    candidate_env = {"RWKV7_FAST_PREFILL": "1"}

    with torch.inference_mode():
        with env_override(**reference_env):
            ref = reference_call()
        with env_override(**candidate_env):
            native = candidate_call()
        ref_logits = ref.logits[:, -1, :].detach()
        native_logits = native.logits[:, -1, :].detach()
        max_abs = float((ref_logits.float() - native_logits.float()).abs().max().detach().cpu())
        min_cos = cosine_min(ref_logits, native_logits)
        greedy_match = bool(torch.equal(ref_logits.argmax(dim=-1).detach().cpu(), native_logits.argmax(dim=-1).detach().cpu()))

        next_token = ref_logits.argmax(dim=-1, keepdim=True)
        reference_decode_backend = "fla" if args.reference_backend == "hf" else "native_jit"
        candidate_decode_backend = os.environ.get("RWKV7_FAST_TOKEN_BACKEND", "auto")
        with env_override(RWKV7_FAST_TOKEN_BACKEND=reference_decode_backend):
            ref_next = model.rwkv7_forward_token(
                next_token,
                past_key_values=ref.past_key_values,
                return_dict=True,
            )
            reference_decode_backend_effective = getattr(model, "rwkv7_last_fast_token_backend", lambda: None)()
        with env_override(RWKV7_FAST_TOKEN_BACKEND=candidate_decode_backend):
            native_next = model.rwkv7_forward_token(
                next_token,
                past_key_values=native.past_key_values,
                return_dict=True,
            )
        decode_max_abs = float((ref_next.logits[:, -1].float() - native_next.logits[:, -1].float()).abs().max().detach().cpu())
        decode_min_cosine = cosine_min(ref_next.logits[:, -1], native_next.logits[:, -1])
        decode_greedy_match = bool(torch.equal(ref_next.logits[:, -1].argmax(dim=-1).detach().cpu(), native_next.logits[:, -1].argmax(dim=-1).detach().cpu()))
        decode_backend = getattr(model, "rwkv7_last_fast_token_backend", lambda: None)()

    with torch.inference_mode():
        with env_override(**reference_env):
            for _ in range(args.warmup):
                reference_call()
        with env_override(**candidate_env):
            for _ in range(args.warmup):
                candidate_call()

    ref_times: list[float] = []
    native_times: list[float] = []
    with torch.inference_mode():
        with env_override(**reference_env):
            for _ in range(args.steps):
                ref_times.append(measured_ms(reference_call, args.device, args.timing))
        with env_override(**candidate_env):
            for _ in range(args.steps):
                native_times.append(measured_ms(candidate_call, args.device, args.timing))

    ref_ms = median(ref_times)
    native_ms = median(native_times)
    peak = None
    if args.device.startswith("cuda"):
        peak = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    scan_m = scan_block_m(model, batch_size)
    row = {
        "axis": "native_prefill_scan",
        "backend": "hf_adapter",
        "bench_case": os.environ.get("RWKV7_BENCH_CASE"),
        "status": (
            "pass"
            if greedy_match
            and decode_greedy_match
            and min_cos >= args.min_cosine
            and decode_min_cosine >= args.min_cosine
            else "fail"
        ),
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "model_path": args.model,
        "effective_model_path": getattr(args, "effective_model_path", args.model),
        "code_source": args.code_source,
        "reference_backend": args.reference_backend,
        "timing": args.timing,
        "native_jit_module": getattr(nj, "__name__", str(nj)),
        "model_size_label": infer_model_size_label(args.model),
        "quantization": args.quantization,
        "quant_policy": args.quant_policy,
        "bnb_skip_policy_requested": (
            os.environ.get("RWKV7_BNB_SKIP_POLICY") if args.quantization.startswith("bnb") else None
        ),
        "bnb_skip_policy": (
            getattr(model, "_rwkv7_bnb_skip_policy", None) if args.quantization.startswith("bnb") else None
        ),
        "bnb_int8_threshold_requested": (
            float(os.environ["RWKV7_BNB_INT8_THRESHOLD"])
            if args.quantization == "bnb8" and "RWKV7_BNB_INT8_THRESHOLD" in os.environ
            else None
        ),
        "bnb_int8_threshold": (
            float(quant_get("llm_int8_threshold", 6.0))
            if args.quantization == "bnb8"
            else None
        ),
        "native_bnb8_direct_requested": os.environ.get("RWKV7_NATIVE_BNB8_DIRECT"),
        "native_bnb8_relu_quant_requested": os.environ.get("RWKV7_NATIVE_BNB8_RELU_QUANT"),
        "native_bnb8_rkv_mix_quant_requested": os.environ.get(
            "RWKV7_NATIVE_BNB8_RKV_MIX_QUANT"
        ),
        "native_bnb8_ffn_mix_quant_requested": os.environ.get(
            "RWKV7_NATIVE_BNB8_FFN_MIX_QUANT"
        ),
        "native_bnb8_direct_effective": bool(
            nj._native_bnb8_policy_flag("RWKV7_NATIVE_BNB8_DIRECT", "native_bnb8_direct")
        ) if args.quantization == "bnb8" else None,
        "native_bnb8_relu_quant_effective": bool(
            nj._native_bnb8_policy_flag("RWKV7_NATIVE_BNB8_RELU_QUANT", "native_bnb8_relu_quant")
        ) if args.quantization == "bnb8" else None,
        "native_bnb8_rkv_mix_quant_effective": bool(
            nj._native_bnb8_policy_flag("RWKV7_NATIVE_BNB8_RKV_MIX_QUANT", "native_bnb8_rkv_mix_quant")
        ) if args.quantization == "bnb8" else None,
        "native_bnb8_ffn_mix_quant_effective": bool(
            nj._native_bnb8_policy_flag("RWKV7_NATIVE_BNB8_FFN_MIX_QUANT", "native_bnb8_ffn_mix_quant")
        ) if args.quantization == "bnb8" else None,
        "native_bnb8_attn_mix_block_effective": int(
            nj._native_bnb8_policy_block(
                "RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK", "native_bnb8_attn_mix_block", 1024
            )
        ) if args.quantization == "bnb8" else None,
        "native_bnb8_ffn_mix_block_effective": int(
            nj._native_bnb8_policy_block(
                "RWKV7_NATIVE_BNB8_FFN_MIX_BLOCK", "native_bnb8_ffn_mix_block", 1024
            )
        ) if args.quantization == "bnb8" else None,
        "quantized_modules": int(getattr(model, "_rwkv7_native_mm_replaced_modules", 0)),
        "model_payload_mb": model_payload_mb(model),
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "tokens_total": batch_size * prompt_tokens,
        "prefill_graph_requested": (
            os.environ.get("RWKV7_NATIVE_PREFILL_GRAPH", "").lower() not in {"", "0", "false", "no", "off"}
            or getattr(model, "_rwkv7_last_fast_prefill_backend", None) == "native_prefill_graph"
        ),
        "prefill_backend_effective": getattr(model, "_rwkv7_last_fast_prefill_backend", None),
        "prefill_graph_effective": getattr(model, "_rwkv7_last_fast_prefill_backend", None) == "native_prefill_graph",
        "fused_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SCAN", "0") not in {"0", "false", "False", "no", "off"},
        "fused_scan_effective": nj._native_prefill_fused_scan_enabled(),
        "scan_block_m": scan_m,
        "scan_num_warps": scan_num_warps(model, scan_m),
        "prefill_fused_scan_output_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_scan_output_effective": nj._native_prefill_fused_scan_output_enabled(),
        "prefill_fused_clampw_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_clampw_scan_effective": nj._native_prefill_fused_clampw_scan_enabled(),
        "prefill_dplr_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_DPLR_SCAN", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_dplr_scan_effective": (
            getattr(nj, "_native_prefill_dplr_scan_enabled", lambda: False)()
            and not nj._native_prefill_fused_scan_enabled()
            and not nj._native_prefill_fused_scan_output_enabled()
        ),
        "prefill_dplr_algorithm": os.environ.get("RWKV7_DPLR_PREFILL_ALGORITHM"),
        "prefill_dplr_chunk_size": getattr(nj, "_native_prefill_dplr_chunk_size", lambda: None)(),
        "prefill_dplr_triton_block_m": os.environ.get("RWKV7_DPLR_TRITON_BLOCK_M"),
        "prefill_dplr_triton_num_warps": os.environ.get("RWKV7_DPLR_TRITON_NUM_WARPS"),
        "prefill_dplr_compact_block_n": os.environ.get("RWKV7_DPLR_TRITON_COMPACT_BLOCK_N"),
        "prefill_dplr_compact_block_r": os.environ.get("RWKV7_DPLR_TRITON_COMPACT_BLOCK_R"),
        "prefill_dplr_compact_prefix_block_m": os.environ.get("RWKV7_DPLR_TRITON_COMPACT_PREFIX_BLOCK_M"),
        "prefill_fused_shift_mix_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_shift_mix_effective": nj._native_prefill_fused_shift_mix_enabled(),
        "prefill_fused_state_prep_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_prep_effective": nj._native_prefill_fused_state_prep_enabled(),
        "prefill_fused_state_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_scan_effective": getattr(nj, "_native_prefill_fused_state_scan_enabled", lambda _batch_size=None: False)(batch_size),
        "prefill_fused_state_scan_max_batch": getattr(nj, "_native_prefill_fused_state_scan_max_batch", lambda: None)(),
        "prefill_state_prep_w_dtype": nj._native_prefill_state_prep_w_dtype(),
        "prefill_fused_output_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_OUTPUT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_output_effective": nj._native_prefill_fused_output_enabled(),
        "prefill_fused_output_project_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_OUTPUT_PROJECT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_output_project_effective": getattr(nj, "_native_prefill_fused_output_project_enabled", lambda: False)(),
        "prefill_fused_output_project_block_m": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_OUTPUT_PROJECT_BLOCK_M"),
        "prefill_fused_wavg_lora_requested": nj._native_prefill_fused_wavg_lora_requested(),
        "prefill_fused_wavg_lora_effective": nj._native_prefill_fused_wavg_lora_enabled(batch_size * prompt_tokens),
        "prefill_fused_wavg_lora_max_m": nj._native_prefill_fused_wavg_lora_max_m(),
        "prefill_fused_wavg_lora_block_m": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_BLOCK_M"),
        "prefill_fused_wavg_lora_block_r": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_BLOCK_R"),
        "prefill_fused_wavg_lora_block_k": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_BLOCK_K"),
        "fast_token_backend_after_native_prefill": decode_backend,
        "reference_decode_backend": reference_decode_backend_effective,
        "candidate_decode_backend": decode_backend,
        "reference_prefill_ms": round(ref_ms, 4),
        "hf_prefill_ms": round(ref_ms, 4) if args.reference_backend == "hf" else None,
        "native_prefill_ms": round(native_ms, 4),
        "native_vs_reference_speedup": round(ref_ms / native_ms, 4) if native_ms > 0 else None,
        "native_vs_hf_speedup": round(ref_ms / native_ms, 4) if native_ms > 0 and args.reference_backend == "hf" else None,
        "reference_prefill_tokps_total": round(1000.0 * batch_size * prompt_tokens / ref_ms, 1) if ref_ms > 0 else None,
        "hf_prefill_tokps_total": round(1000.0 * batch_size * prompt_tokens / ref_ms, 1) if ref_ms > 0 and args.reference_backend == "hf" else None,
        "native_prefill_tokps_total": round(1000.0 * batch_size * prompt_tokens / native_ms, 1) if native_ms > 0 else None,
        "max_abs_diff": round(max_abs, 6),
        "min_cosine": round(min_cos, 8),
        "greedy_match": greedy_match,
        "decode_after_prefill_max_abs_diff": round(decode_max_abs, 6),
        "decode_after_prefill_min_cosine": round(decode_min_cosine, 8),
        "decode_after_prefill_greedy_match": decode_greedy_match,
        "min_cosine_gate": args.min_cosine,
        "peak_vram_mb": peak,
    }
    return row


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def append_row(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=DTYPES, default="fp16")
    ap.add_argument("--batch-sizes", default="1,4")
    ap.add_argument("--prompt-tokens", default="128")
    ap.add_argument("--fused-scan", choices=["auto", "true", "false"], default="auto")
    ap.add_argument(
        "--reference-backend",
        choices=["hf", "native-direct"],
        default="hf",
        help="HF/FLA reference, or independent direct-native reference for cards where FLA cannot compile",
    )
    ap.add_argument(
        "--quantization",
        choices=["none", "bnb8", "bnb4", "a8w8", "mm4", "torchao_w8", "torchao_w4"],
        default="none",
    )
    ap.add_argument("--quant-policy", choices=["memory", "speed"], default="speed")
    ap.add_argument("--quant-min-params", type=int, default=1)
    ap.add_argument("--code-source", choices=["model", "repo"], default="model", help="load trust_remote_code from checkpoint files or overlay current repo rwkv7_hf/*.py")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--timing", choices=["cuda-event", "wall"], default="cuda-event")
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    if args.fused_scan != "auto":
        os.environ["RWKV7_NATIVE_PREFILL_FUSED_SCAN"] = "1" if args.fused_scan == "true" else "0"

    effective_model_path, tmp_model_dir = prepare_model_dir(args.model, code_source=args.code_source)
    args.effective_model_path = effective_model_path
    try:
        tok = AutoTokenizer.from_pretrained(effective_model_path, trust_remote_code=True)
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": DTYPES[args.dtype],
            "device_map": args.device if args.device.startswith("cuda") else None,
        }
        if args.quantization == "bnb8":
            threshold = float(os.environ.get("RWKV7_BNB_INT8_THRESHOLD", "6.0"))
            if threshold < 0.0:
                raise ValueError("RWKV7_BNB_INT8_THRESHOLD must be non-negative")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=threshold,
            )
        elif args.quantization == "bnb4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=DTYPES[args.dtype],
                bnb_4bit_use_double_quant=False,
            )
        model = AutoModelForCausalLM.from_pretrained(effective_model_path, **load_kwargs).eval()
        if args.quantization == "a8w8":
            from rwkv7_hf.native_quant_a8w8 import quantize_model_a8w8

            quantize_model_a8w8(model, min_params=args.quant_min_params, policy=args.quant_policy)
        elif args.quantization == "mm4":
            from rwkv7_hf.native_quant_mm4 import quantize_model_mm4

            quantize_model_mm4(model, min_params=args.quant_min_params, policy=args.quant_policy)
        elif args.quantization in {"torchao_w8", "torchao_w4"}:
            from rwkv7_hf.native_quant_torchao import quantize_model_torchao

            quantize_model_torchao(
                model,
                args.quantization,
                min_params=args.quant_min_params,
                policy=args.quant_policy,
            )
        for bsz in parse_ints(args.batch_sizes):
            for prompt_tokens in parse_ints(args.prompt_tokens):
                row = run_case(args, tok, model, bsz, prompt_tokens)
                print(json.dumps(row, ensure_ascii=False))
                append_row(args.results, row)
    finally:
        if tmp_model_dir is not None:
            tmp_model_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
