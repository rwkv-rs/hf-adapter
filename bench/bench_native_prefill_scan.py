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
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    tmp = tempfile.TemporaryDirectory(prefix="rwkv7_repo_code_model_")
    dst = Path(tmp.name)
    for item in src.iterdir():
        if item.name == "__pycache__" or item.suffix == ".py":
            continue
        target = dst / item.name
        if item.is_dir():
            target.symlink_to(item, target_is_directory=True)
        else:
            target.symlink_to(item)
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


def scan_block_m(model) -> int | None:
    raw = os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    try:
        head_dim = int(model._rwkv7_native_jit_packs()[0][2])
        return model_native_jit_module(model)._native_prefill_scan_block_m(head_dim)
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


def scan_num_stages(model) -> int | None:
    raw = os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_NUM_STAGES")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    try:
        return model_native_jit_module(model)._native_prefill_scan_num_stages()
    except Exception:
        return None


def cosine_min(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=-1).min().detach().cpu())


def run_case(args: argparse.Namespace, tok, model, batch_size: int, prompt_tokens: int) -> dict[str, Any]:
    nj = model_native_jit_module(model)
    ids = build_ids(tok, batch_size, prompt_tokens, args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # FLA/HF reference prefill.
    with torch.inference_mode():
        ref = model(ids, use_cache=True, logits_to_keep=1, return_dict=True)
        native = model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)
        ref_logits = ref.logits[:, -1, :].detach()
        native_logits = native.logits[:, -1, :].detach()
        max_abs = float((ref_logits.float() - native_logits.float()).abs().max().detach().cpu())
        min_cos = cosine_min(ref_logits, native_logits)
        greedy_match = bool(torch.equal(ref_logits.argmax(dim=-1).detach().cpu(), native_logits.argmax(dim=-1).detach().cpu()))

        next_token = ref_logits.argmax(dim=-1, keepdim=True)
        ref_next = model(next_token, past_key_values=ref.past_key_values, use_cache=True, logits_to_keep=1, return_dict=True)
        native_next = model.rwkv7_forward_token(next_token, past_key_values=native.past_key_values, return_dict=True)
        decode_max_abs = float((ref_next.logits[:, -1].float() - native_next.logits[:, -1].float()).abs().max().detach().cpu())
        decode_greedy_match = bool(torch.equal(ref_next.logits[:, -1].argmax(dim=-1).detach().cpu(), native_next.logits[:, -1].argmax(dim=-1).detach().cpu()))
        decode_backend = getattr(model, "rwkv7_last_fast_token_backend", lambda: None)()

    for _ in range(args.warmup):
        with torch.inference_mode():
            model(ids, use_cache=True, logits_to_keep=1, return_dict=True)
            model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)

    ref_times: list[float] = []
    native_times: list[float] = []
    with torch.inference_mode():
        for _ in range(args.steps):
            ref_times.append(wall_ms(lambda: model(ids, use_cache=True, logits_to_keep=1, return_dict=True), args.device))
        for _ in range(args.steps):
            native_times.append(wall_ms(lambda: model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True), args.device))

    ref_ms = median(ref_times)
    native_ms = median(native_times)
    peak = None
    if args.device.startswith("cuda"):
        peak = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    scan_m = scan_block_m(model)
    shift_wavg_blocks = getattr(nj, "_native_prefill_fused_shift_wavg_lora_blocks", lambda: (None, None, None))()
    shift_wavg_warps = getattr(nj, "_native_prefill_fused_shift_wavg_lora_warps", lambda: (None, None))()
    rkv_head_dim = None
    try:
        rkv_pack = nj._ensure_rkv_pack(model._rwkv7_native_jit_packs()[0])
        rkv_head_dim = int(rkv_pack[2])
        rkv_hidden = int(rkv_pack[1]) * rkv_head_dim
        rkv_weight = rkv_pack[-1]
        prefill_rkv_bmm_effective = getattr(nj, "_native_prefill_rkv_bmm_enabled", lambda *_args: False)(
            batch_size * prompt_tokens,
            rkv_hidden,
            rkv_weight,
        )
    except Exception:
        prefill_rkv_bmm_effective = False
    cuda_state_scan_effective = getattr(nj, "_native_prefill_cuda_state_scan_enabled", lambda: False)()
    cuda_state_scan_lanes = getattr(nj, "_native_prefill_cuda_state_scan_lanes_per_row", lambda: 1)()
    cuda_state_scan_precompute = getattr(nj, "_native_prefill_cuda_state_scan_precompute_enabled", lambda: False)()
    cuda_state_scan_precompute_mode = getattr(nj, "_native_prefill_cuda_state_scan_precompute_mode", lambda: "none")()
    cuda_state_scan_reuse_precompute = getattr(
        nj, "_native_prefill_cuda_state_scan_reuse_precompute_enabled", lambda: False
    )()
    cuda_state_scan_inplace_kv = getattr(
        nj, "_native_prefill_cuda_state_scan_inplace_kv_enabled", lambda: False
    )()
    cuda_state_scan_inplace_kka = getattr(
        nj, "_native_prefill_cuda_state_scan_inplace_kka_enabled", lambda: False
    )()
    cuda_state_scan_rows_per_block = getattr(nj, "_native_prefill_cuda_state_scan_rows_per_block", lambda: 1)()
    cuda_state_scan_schedule = getattr(nj, "_native_prefill_cuda_state_scan_schedule", lambda: "default")()
    cuda_state_scan_w_precomputed = getattr(
        nj, "_native_prefill_fused_shift_wavg_lora_w_decay_enabled", lambda _rows: False
    )(batch_size * prompt_tokens)
    fused_state_scan_raw_output_effective = getattr(
        nj, "_native_prefill_fused_state_scan_raw_output_enabled", lambda: False
    )()
    cuda_state_scan_raw_nokv_effective = bool(
        fused_state_scan_raw_output_effective
        and cuda_state_scan_effective
        and rkv_head_dim == 64
        and scan_m == 64
        and args.dtype == "fp16"
        and int(cuda_state_scan_lanes) == 64
        and not cuda_state_scan_precompute
        and cuda_state_scan_precompute_mode == "none"
        and cuda_state_scan_schedule == "warp_specialized"
        and not cuda_state_scan_w_precomputed
    )
    return {
        "axis": "native_prefill_scan",
        "backend": "hf_adapter",
        "bench_case": os.environ.get("RWKV7_BENCH_CASE"),
        "status": "pass" if greedy_match and decode_greedy_match else "fail",
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "model_path": args.model,
        "effective_model_path": getattr(args, "effective_model_path", args.model),
        "code_source": args.code_source,
        "native_jit_module": getattr(nj, "__name__", str(nj)),
        "model_size_label": infer_model_size_label(args.model),
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "tokens_total": batch_size * prompt_tokens,
        "fused_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SCAN", "0") not in {"0", "false", "False", "no", "off"},
        "scan_block_m": scan_m,
        "scan_num_warps": scan_num_warps(model, scan_m),
        "scan_num_stages": scan_num_stages(model),
        "scan_algebraic_output": getattr(nj, "_native_prefill_scan_algebraic_output_enabled", lambda: False)(),
        "scan_nomask64": getattr(nj, "_native_prefill_scan_nomask64_enabled", lambda: False)(),
        "scan_precompute_w": getattr(nj, "_native_prefill_scan_precompute_w_enabled", lambda: False)(),
        "scan_precompute_w_dtype": getattr(nj, "_native_prefill_scan_precompute_w_dtype", lambda: "fp32")(),
        "prefill_rkv_bmm_requested": getattr(nj, "_native_prefill_rkv_bmm_requested", lambda: False)(),
        "prefill_rkv_bmm_effective": bool(prefill_rkv_bmm_effective),
        "prefill_rkv_bmm_max_rows": getattr(nj, "_native_prefill_rkv_bmm_max_rows", lambda: None)(),
        "prefill_cuda_state_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_cuda_state_scan_effective": cuda_state_scan_effective,
        "prefill_cuda_state_scan_sk_requested": os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SK", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_cuda_state_scan_sk_effective": getattr(nj, "_native_prefill_cuda_state_scan_sk_enabled", lambda: False)(),
        "prefill_cuda_state_scan_lanes": cuda_state_scan_lanes,
        "prefill_cuda_state_scan_precompute": cuda_state_scan_precompute,
        "prefill_cuda_state_scan_precompute_mode": cuda_state_scan_precompute_mode,
        "prefill_cuda_state_scan_reuse_precompute": cuda_state_scan_reuse_precompute,
        "prefill_cuda_state_scan_reuse_precompute_effective": bool(
            cuda_state_scan_effective
            and cuda_state_scan_reuse_precompute
            and cuda_state_scan_precompute
            and cuda_state_scan_precompute_mode == "wk_half"
        ),
        "prefill_cuda_state_scan_inplace_kv": cuda_state_scan_inplace_kv,
        "prefill_cuda_state_scan_inplace_kv_effective": bool(
            cuda_state_scan_effective
            and cuda_state_scan_inplace_kv
            and cuda_state_scan_precompute
            and not cuda_state_scan_w_precomputed
        ),
        "prefill_cuda_state_scan_inplace_kka": cuda_state_scan_inplace_kka,
        "prefill_cuda_state_scan_inplace_kka_effective": bool(
            cuda_state_scan_effective
            and cuda_state_scan_inplace_kka
            and cuda_state_scan_precompute
            and cuda_state_scan_precompute_mode == "wk_half"
            and not cuda_state_scan_w_precomputed
        ),
        "prefill_cuda_state_scan_rows_per_block": cuda_state_scan_rows_per_block,
        "prefill_cuda_state_scan_schedule": cuda_state_scan_schedule,
        "prefill_cuda_state_scan_w_precomputed": cuda_state_scan_w_precomputed,
        "prefill_cuda_state_scan_raw_nokv_effective": cuda_state_scan_raw_nokv_effective,
        "prefill_cuda_state_scan_write_kv": (
            (not cuda_state_scan_raw_nokv_effective) if cuda_state_scan_effective else None
        ),
        "prefill_fused_scan_output_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_scan_output_effective": nj._native_prefill_fused_scan_output_enabled(),
        "prefill_fused_state_scan_output_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_OUTPUT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_scan_output_effective": getattr(nj, "_native_prefill_fused_state_scan_output_enabled", lambda: False)(),
        "prefill_fused_state_scan_correction_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_CORRECTION", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_scan_correction_effective": getattr(nj, "_native_prefill_fused_state_scan_correction_enabled", lambda: False)(),
        "prefill_fused_state_scan_raw_output_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_RAW_OUTPUT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_scan_raw_output_effective": fused_state_scan_raw_output_effective,
        "prefill_fused_state_scan_sk_output_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_SK_OUTPUT", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_scan_sk_output_effective": getattr(nj, "_native_prefill_fused_state_scan_sk_output_enabled", lambda: False)(),
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
        "prefill_dplr_compact_start_state_dtype": os.environ.get(
            "RWKV7_DPLR_TRITON_COMPACT_START_STATE_DTYPE", "fp32"
        ),
        "prefill_dplr_compact_output_only": os.environ.get("RWKV7_DPLR_TRITON_COMPACT_OUTPUT_ONLY", "0").lower()
        not in {"0", "false", "no", "off"},
        "prefill_dplr_compact_recompute_starts": os.environ.get(
            "RWKV7_DPLR_TRITON_COMPACT_RECOMPUTE_STARTS", "0"
        ).lower()
        not in {"0", "false", "no", "off"},
        "prefill_dplr_compact_prefix_shared": os.environ.get(
            "RWKV7_DPLR_TRITON_COMPACT_PREFIX_SHARED", "0"
        ).lower()
        not in {"0", "false", "no", "off"},
        "prefill_dplr_compact_prefix_shared_group_size": int(
            os.environ.get("RWKV7_DPLR_TRITON_COMPACT_PREFIX_SHARED_GROUP_SIZE", "0")
        ),
        "prefill_fused_shift_mix_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_shift_mix_effective": nj._native_prefill_fused_shift_mix_enabled(),
        "prefill_fused_norm_mix_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_NORM_MIX", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_norm_mix_effective": getattr(nj, "_native_prefill_fused_norm_mix_enabled", lambda: False)(),
        "prefill_fused_state_prep_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_prep_effective": nj._native_prefill_fused_state_prep_enabled(),
        "prefill_fused_state_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN", "0").lower() not in {"0", "false", "no", "off"},
        "prefill_fused_state_scan_effective": getattr(nj, "_native_prefill_fused_state_scan_enabled", lambda: False)(),
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
        "prefill_fused_shift_wavg_lora_requested": getattr(nj, "_native_prefill_fused_shift_wavg_lora_requested", lambda: False)(),
        "prefill_fused_shift_wavg_lora_effective": getattr(nj, "_native_prefill_fused_shift_wavg_lora_enabled", lambda _rows: False)(batch_size * prompt_tokens),
        "prefill_fused_shift_wavg_lora_block_m": shift_wavg_blocks[0],
        "prefill_fused_shift_wavg_lora_block_r": shift_wavg_blocks[1],
        "prefill_fused_shift_wavg_lora_block_k": shift_wavg_blocks[2],
        "prefill_fused_shift_wavg_lora_down_warps": shift_wavg_warps[0],
        "prefill_fused_shift_wavg_lora_up_warps": shift_wavg_warps[1],
        "prefill_fused_shift_wavg_lora_lean_down": getattr(nj, "_native_prefill_fused_shift_wavg_lora_lean_down_requested", lambda: False)(),
        "prefill_fused_shift_wavg_lora_lean_up": getattr(nj, "_native_prefill_fused_shift_wavg_lora_lean_up_requested", lambda: False)(),
        "prefill_fused_shift_wavg_lora_g_mid_output_requested": getattr(nj, "_native_prefill_fused_shift_wavg_lora_g_mid_output_requested", lambda: False)(),
        "prefill_fused_shift_wavg_lora_g_mid_output_effective": getattr(nj, "_native_prefill_fused_shift_wavg_lora_g_mid_output_enabled", lambda: False)(),
        "prefill_fused_shift_wavg_lora_w_decay_requested": getattr(nj, "_native_prefill_fused_shift_wavg_lora_w_decay_requested", lambda: False)(),
        "prefill_fused_shift_wavg_lora_w_decay_effective": getattr(nj, "_native_prefill_fused_shift_wavg_lora_w_decay_enabled", lambda _rows: False)(batch_size * prompt_tokens),
        "prefill_fused_shift_wavg_lora_a_sigmoid_requested": getattr(nj, "_native_prefill_fused_shift_wavg_lora_a_sigmoid_requested", lambda: False)(),
        "prefill_fused_shift_wavg_lora_a_sigmoid_effective": getattr(nj, "_native_prefill_fused_shift_wavg_lora_a_sigmoid_enabled", lambda _rows: False)(batch_size * prompt_tokens),
        "prefill_fused_shift_wavg_lora_prev_cache_requested": getattr(nj, "_native_prefill_fused_shift_wavg_lora_prev_cache_requested", lambda: False)(),
        "prefill_fused_shift_wavg_lora_prev_cache_effective": getattr(nj, "_native_prefill_fused_shift_wavg_lora_prev_cache_enabled", lambda _rows: False)(batch_size * prompt_tokens),
        "prefill_ffn_fused_act_requested": getattr(nj, "_native_prefill_ffn_fused_act_requested", lambda: False)(),
        "prefill_ffn_fused_act_effective": getattr(nj, "_native_prefill_ffn_fused_act_enabled", lambda: False)(),
        "prefill_ffn_fused_act_mode": getattr(nj, "_native_prefill_ffn_fused_act_mode", lambda: "triton")(),
        "prefill_ffn_fused_act_block_size": getattr(nj, "_native_prefill_ffn_fused_act_block_size", lambda: None)(),
        "prefill_ffn_fused_norm_shift_requested": getattr(nj, "_native_prefill_ffn_fused_norm_shift_requested", lambda: False)(),
        "prefill_ffn_fused_norm_shift_effective": getattr(nj, "_native_prefill_ffn_fused_norm_shift_enabled", lambda: False)(),
        "prefill_ffn_fused_norm_shift_block_h": getattr(nj, "_native_prefill_ffn_fused_norm_shift_block_h", lambda: None)(),
        "prefill_ffn_fused_norm_shift_mode": getattr(nj, "_native_prefill_ffn_fused_norm_shift_mode", lambda: "recompute")(),
        "prefill_fused_projection_requested": getattr(nj, "_native_prefill_fused_projection_requested", lambda: False)(),
        "prefill_fused_projection_effective": getattr(nj, "_native_prefill_fused_projection_enabled", lambda _rows: False)(batch_size * prompt_tokens),
        "prefill_fused_projection_max_m": getattr(nj, "_native_prefill_fused_projection_max_m", lambda: None)(),
        "prefill_fused_projection_block_m": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_PROJECTION_BLOCK_M"),
        "prefill_fused_projection_block_r": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_PROJECTION_BLOCK_R"),
        "prefill_fused_projection_block_k": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_PROJECTION_BLOCK_K"),
        "fast_token_backend_after_native_prefill": decode_backend,
        "hf_prefill_ms": round(ref_ms, 4),
        "native_prefill_ms": round(native_ms, 4),
        "native_vs_hf_speedup": round(ref_ms / native_ms, 4) if native_ms > 0 else None,
        "hf_prefill_tokps_total": round(1000.0 * batch_size * prompt_tokens / ref_ms, 1) if ref_ms > 0 else None,
        "native_prefill_tokps_total": round(1000.0 * batch_size * prompt_tokens / native_ms, 1) if native_ms > 0 else None,
        "max_abs_diff": round(max_abs, 6),
        "min_cosine": round(min_cos, 8),
        "greedy_match": greedy_match,
        "decode_after_prefill_max_abs_diff": round(decode_max_abs, 6),
        "decode_after_prefill_greedy_match": decode_greedy_match,
        "peak_vram_mb": peak,
    }


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
    ap.add_argument("--code-source", choices=["model", "repo"], default="model", help="load trust_remote_code from checkpoint files or overlay current repo rwkv7_hf/*.py")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    if args.fused_scan != "auto":
        os.environ["RWKV7_NATIVE_PREFILL_FUSED_SCAN"] = "1" if args.fused_scan == "true" else "0"

    effective_model_path, tmp_model_dir = prepare_model_dir(args.model, code_source=args.code_source)
    args.effective_model_path = effective_model_path
    try:
        tok = AutoTokenizer.from_pretrained(effective_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            effective_model_path,
            trust_remote_code=True,
            torch_dtype=DTYPES[args.dtype],
            device_map=args.device if args.device.startswith("cuda") else None,
        ).eval()
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
