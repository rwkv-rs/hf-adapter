#!/usr/bin/env python3
# coding=utf-8
"""Apple Silicon MLX recurrent RWKV-7 smoke.

This is stronger than the tensor bridge smoke: it verifies a full MLX recurrent
reference path, state-cache select/reorder behavior, chunked prefill, and
optional converted-model greedy decode.  The backend is correctness-first; the
future production step is replacing the inner recurrent update with fused
MLX/Metal kernels.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from rwkv7_hf.mlx_bridge import mlx_available, require_mlx, torch_tensor_to_mlx
from rwkv7_hf.mlx_model import MLXGenerationSession, MLXGenerationSessionBatch, MLXRWKV7Model


from apple_silicon_utils import (
    darwin_sysctl,
    apple_memory_gb,
    emit,
    infer_model_size_label,
    is_apple_silicon,
    package_version,
)


def max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def prompt_or_tokens(args: argparse.Namespace) -> tuple[list[int], str]:
    """Resolve real-model smoke input from --prompt or --tokens."""

    if args.prompt:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        encoded = tokenizer(args.prompt, add_special_tokens=False)
        tokens = [int(tok) for tok in encoded.input_ids]
        if not tokens:
            raise ValueError("--prompt produced no token ids")
        return tokens, "prompt"
    tokens = [int(tok) for tok in args.tokens.split(",") if tok.strip()]
    if not tokens:
        raise ValueError("--tokens must contain at least one token id")
    return tokens, "tokens"


def tiny_torch_model_to_mlx() -> tuple[Any, MLXRWKV7Model, dict[str, Any]]:
    import torch

    from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM

    torch.manual_seed(1234)
    cfg = NativeRWKV7Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        head_dim=4,
        num_heads=4,
        intermediate_size=32,
        decay_low_rank_dim=8,
        gate_low_rank_dim=8,
        a_low_rank_dim=8,
        v_low_rank_dim=8,
        use_cache=True,
        norm_eps=1e-5,
    )
    torch_model = NativeRWKV7ForCausalLM(cfg).eval()
    arrays = {name: torch_tensor_to_mlx(value, dtype="fp32") for name, value in torch_model.state_dict().items()}
    mlx_model = MLXRWKV7Model.from_arrays(cfg.to_dict(), arrays)
    return torch_model, mlx_model, cfg.to_dict()


class TinyTokenizer:
    def __call__(self, prompt: str, *, add_special_tokens: bool = False):
        class Encoded:
            input_ids = [1, 2, 3, 4]

        return Encoded()

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return ",".join(str(int(x)) for x in ids)


def run_tiny_recurrent_parity(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    mx = require_mlx()
    torch_model, mlx_model, cfg = tiny_torch_model_to_mlx()
    ids_np = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)
    ids = torch.tensor(ids_np, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        torch_logits = torch_model(ids, use_cache=True).logits.detach().cpu().float().numpy()
    mlx_logits, mlx_state = mlx_model.forward(ids_np, collect_all=True)
    mx.eval(mlx_logits)
    elapsed = time.perf_counter() - t0
    diff = max_abs(mlx_logits, torch_logits)
    torch_argmax = torch_logits[:, -1, :].argmax(axis=-1).tolist()
    mlx_argmax = np.asarray(mx.argmax(mlx_logits[:, -1, :], axis=-1)).astype(int).tolist()
    # MLX and torch use different low-level matmul / normalization kernels on
    # Apple GPU.  Keep this tight enough to catch layout/formula bugs while
    # allowing expected fp32 backend drift.
    assert diff < 5e-3, f"tiny MLX/Torch recurrent parity max_abs={diff}"
    assert torch_argmax == mlx_argmax
    return {
        "axis": "apple_silicon_mlx_recurrent_tiny_parity",
        "status": "pass",
        "batch_size": int(ids_np.shape[0]),
        "seq_len": int(ids_np.shape[1]),
        "vocab_size": int(cfg["vocab_size"]),
        "hidden_size": int(cfg["hidden_size"]),
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "max_abs": round(diff, 8),
        "argmax_match": True,
        "seen_tokens": int(mlx_state.seen_tokens),
        "elapsed_s": round(elapsed, 6),
    }


def run_tiny_state_cache(args: argparse.Namespace) -> dict[str, Any]:
    mx = require_mlx()
    _, mlx_model, _ = tiny_torch_model_to_mlx()
    ids_np = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)
    full_last, full_state = mlx_model.prefill(ids_np)
    chunk_last, chunk_state = mlx_model.chunked_prefill(ids_np, chunk_size=2)
    mx.eval(full_last, chunk_last)
    chunk_diff = max_abs(full_last, chunk_last)
    assert chunk_diff < 1e-5, f"chunked/full prefill mismatch: {chunk_diff}"

    next_tokens = mx.argmax(full_last[:, -1, :], axis=-1).astype(mx.int32)
    full_decode, _ = mlx_model.decode_step(next_tokens, full_state.clone())
    selected_state = chunk_state.select_batch([1])
    selected_decode, _ = mlx_model.decode_step(next_tokens[1:2], selected_state)
    mx.eval(full_decode, selected_decode)
    select_diff = max_abs(full_decode[1:2], selected_decode)
    assert select_diff < 5e-3, f"selected-state decode mismatch: {select_diff}"
    return {
        "axis": "apple_silicon_mlx_state_cache_tiny",
        "status": "pass",
        "batch_size": 2,
        "seq_len": 4,
        "chunk_size": 2,
        "chunked_prefill_max_abs": round(chunk_diff, 8),
        "select_batch_decode_max_abs": round(select_diff, 8),
        "seen_tokens": int(chunk_state.seen_tokens),
    }


def run_tiny_generation_session(args: argparse.Namespace) -> dict[str, Any]:
    _, mlx_model, _ = tiny_torch_model_to_mlx()
    tokenizer = TinyTokenizer()
    session = MLXGenerationSession.from_prompt(mlx_model, tokenizer, "tiny")
    first = session.decode(2)
    second = session.decode(2)
    one_shot = mlx_model.generate_text(tokenizer, "tiny", max_new_tokens=4)
    assert session.generated_ids == one_shot.generated_ids
    assert session.text == one_shot.text
    assert int(session.state.seen_tokens) == len(session.prompt_ids) + 4
    assert first.generated_tokens == 2
    assert second.generated_tokens == 2
    return {
        "axis": "apple_silicon_mlx_session_tiny",
        "status": "pass",
        "prompt_tokens": len(session.prompt_ids),
        "generated_tokens": session.generated_tokens,
        "step_sizes": [first.generated_tokens, second.generated_tokens],
        "session_one_shot_token_match": True,
        "session_one_shot_text_match": True,
        "seen_tokens_after_generate": int(session.state.seen_tokens),
        "generated_preview": session.generated_ids[:8],
    }


def run_tiny_generation_session_batch(args: argparse.Namespace) -> dict[str, Any]:
    _, mlx_model, _ = tiny_torch_model_to_mlx()
    tokenizer = TinyTokenizer()
    prompts = ["tiny-a", "tiny-b"]
    try:
        MLXGenerationSessionBatch.from_prompts(mlx_model, tokenizer, "tiny-not-a-list")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("single-string prompts must be rejected")
    guard_batch = MLXGenerationSessionBatch.from_prompts(mlx_model, tokenizer, prompts)
    guard_seen = [int(session.state.seen_tokens) for session in guard_batch.sessions]
    guard_generated = [int(session.generated_tokens) for session in guard_batch.sessions]
    try:
        guard_batch.decode_round([1, -1])
    except ValueError:
        pass
    else:
        raise AssertionError("negative token counts must be rejected")
    assert [int(session.state.seen_tokens) for session in guard_batch.sessions] == guard_seen
    assert [int(session.generated_tokens) for session in guard_batch.sessions] == guard_generated
    batch = MLXGenerationSessionBatch.from_prompts(mlx_model, tokenizer, prompts)
    first = batch.decode_round(1)
    second = batch.decode_round(2)
    total_new = 3
    for session in batch.sessions:
        one_shot = mlx_model.generate_text(tokenizer, session.prompt, max_new_tokens=total_new)
        assert session.generated_ids == one_shot.generated_ids
        assert session.text == one_shot.text
        assert int(session.state.seen_tokens) == len(session.prompt_ids) + total_new
    return {
        "axis": "apple_silicon_mlx_session_batch_tiny",
        "status": "pass",
        "session_count": batch.batch_size,
        "rounds": [1, 2],
        "generated_tokens": [session.generated_tokens for session in batch.sessions],
        "seen_tokens": [int(session.state.seen_tokens) for session in batch.sessions],
        "all_session_one_shot_token_match": True,
        "all_session_one_shot_text_match": True,
        "generated_previews": [session.generated_ids[:8] for session in batch.sessions],
    }


def run_real_model_smoke(args: argparse.Namespace) -> dict[str, Any]:
    mx = require_mlx()
    tokens, prompt_source = prompt_or_tokens(args)
    t0 = time.perf_counter()
    model = MLXRWKV7Model.from_hf(args.model, dtype=args.dtype)
    load_s = time.perf_counter() - t0
    ids = np.array([tokens], dtype=np.int64)
    t_prefill = time.perf_counter()
    full_last, full_state = model.prefill(ids)
    mx.eval(full_last)
    prefill_s = time.perf_counter() - t_prefill
    t_chunk = time.perf_counter()
    chunk_last, chunk_state = model.chunked_prefill(ids, chunk_size=max(1, int(args.chunk_size)))
    mx.eval(chunk_last)
    chunk_s = time.perf_counter() - t_chunk
    t_decode = time.perf_counter()
    generated, gen_state = model.decode_greedy(full_last, full_state.clone(), max_new_tokens=int(args.max_new_tokens))
    mx.eval(full_last, chunk_last, generated)
    decode_s = time.perf_counter() - t_decode
    run_s = prefill_s + decode_s
    diff = max_abs(full_last, chunk_last)
    assert diff < float(args.chunk_tolerance), f"real-model chunked/full MLX mismatch: {diff}"
    assert int(generated.shape[1]) == int(args.max_new_tokens)
    assert int(full_state.seen_tokens) == len(tokens)
    assert int(chunk_state.seen_tokens) == len(tokens)
    assert int(gen_state.seen_tokens) == len(tokens) + int(args.max_new_tokens)
    telemetry = model.telemetry()
    row = {
        "axis": "apple_silicon_mlx_recurrent_model_smoke",
        "status": "pass",
        "model": Path(args.model).name,
        "model_size_label": infer_model_size_label(args.model, args.model_size_label),
        "dtype": args.dtype,
        "prompt_source": prompt_source,
        "prompt_preview": args.prompt[:40] if args.prompt else "",
        "prompt_tokens": len(tokens),
        "generated_tokens": int(args.max_new_tokens),
        "chunk_size": int(args.chunk_size),
        "chunked_prefill_max_abs": round(diff, 8),
        "generated_shape": [int(x) for x in generated.shape],
        "generated_preview": np.asarray(generated).astype(int).reshape(-1).tolist()[:8],
        "seen_tokens_after_prefill": int(full_state.seen_tokens),
        "seen_tokens_after_generate": int(gen_state.seen_tokens),
        "load_s": round(load_s, 4),
        "prefill_s": round(prefill_s, 4),
        "decode_s": round(decode_s, 4),
        "chunked_prefill_s": round(chunk_s, 4),
        "elapsed_s": round(run_s, 4),
        "prefill_tok_s": round(len(tokens) / prefill_s, 4) if prefill_s > 0 else None,
        "decode_tok_s": round(int(args.max_new_tokens) / decode_s, 4) if decode_s > 0 else None,
        "tensor_count": telemetry["tensor_count"],
        "total_params": telemetry["total_params"],
        "total_bytes": telemetry["total_bytes"],
    }
    if args.dynamic_batch:
        batch_ids = np.array([tokens, list(reversed(tokens))], dtype=np.int64)
        batch_last, batch_state = model.prefill(batch_ids)
        next_tokens = mx.argmax(batch_last[:, -1, :], axis=-1).astype(mx.int32)
        batch_decode, _ = model.decode_step(next_tokens, batch_state.clone())
        selected_state = batch_state.select_batch([1])
        selected_decode, selected_after = model.decode_step(next_tokens[1:2], selected_state)
        mx.eval(batch_decode, selected_decode)
        dynamic_diff = max_abs(batch_decode[1:2], selected_decode)
        dynamic_argmax_match = (
            np.asarray(mx.argmax(batch_decode[1:2, -1, :], axis=-1)).astype(int).tolist()
            == np.asarray(mx.argmax(selected_decode[:, -1, :], axis=-1)).astype(int).tolist()
        )
        assert dynamic_diff < float(args.dynamic_tolerance), f"real-model dynamic select mismatch: {dynamic_diff}"
        assert dynamic_argmax_match
        row.update(
            {
                "dynamic_batch": True,
                "dynamic_batch_size": 2,
                "dynamic_select_decode_max_abs": round(dynamic_diff, 8),
                "dynamic_select_argmax_match": True,
                "dynamic_selected_seen_tokens": int(selected_after.seen_tokens),
            }
        )
    if args.compare_torch:
        import torch
        from transformers import AutoModelForCausalLM

        torch_dtype = torch.float32 if args.dtype == "fp32" else torch.float16
        old_native = os.environ.get("RWKV7_NATIVE_MODEL")
        old_jit = os.environ.get("RWKV7_NATIVE_MODEL_JIT")
        os.environ["RWKV7_NATIVE_MODEL"] = "1"
        os.environ["RWKV7_NATIVE_MODEL_JIT"] = "0"
        try:
            torch_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                dtype=torch_dtype,
                device_map="cpu",
            ).eval()
        finally:
            if old_native is None:
                os.environ.pop("RWKV7_NATIVE_MODEL", None)
            else:
                os.environ["RWKV7_NATIVE_MODEL"] = old_native
            if old_jit is None:
                os.environ.pop("RWKV7_NATIVE_MODEL_JIT", None)
            else:
                os.environ["RWKV7_NATIVE_MODEL_JIT"] = old_jit
        with torch.no_grad():
            torch_logits = torch_model(torch.tensor(ids, dtype=torch.long), use_cache=True).logits.detach().cpu().float().numpy()
        torch_diff = max_abs(full_last[:, -1:, :], torch_logits[:, -1:, :])
        mlx_argmax = np.asarray(mx.argmax(full_last[:, -1, :], axis=-1)).astype(int).tolist()
        torch_argmax = torch_logits[:, -1, :].argmax(axis=-1).astype(int).tolist()
        assert torch_diff < float(args.torch_compare_tolerance), (
            f"real-model MLX/Torch parity max_abs={torch_diff} "
            f"(tolerance={args.torch_compare_tolerance})"
        )
        assert mlx_argmax == torch_argmax
        row.update(
            {
                "torch_compare": True,
                "torch_compare_dtype": str(torch_dtype).replace("torch.", ""),
                "torch_compare_max_abs": round(torch_diff, 8),
                "torch_argmax_match": True,
            }
        )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="Optional converted RWKV-7 HF model dir for full MLX recurrent smoke.")
    ap.add_argument("--model-size-label", default="")
    ap.add_argument("--dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    ap.add_argument("--tokens", default="1,2,3,4", help="Comma-separated token ids for converted-model smoke.")
    ap.add_argument("--prompt", default="", help="Optional text prompt; overrides --tokens for converted-model smoke.")
    ap.add_argument("--chunk-size", type=int, default=2)
    ap.add_argument("--chunk-tolerance", type=float, default=1e-2)
    ap.add_argument("--max-new-tokens", type=int, default=1)
    ap.add_argument("--dynamic-batch", action="store_true", help="Validate real-model state select after a 2-row prefill.")
    ap.add_argument("--dynamic-tolerance", type=float, default=0.1)
    ap.add_argument("--compare-torch", action="store_true", help="Also compare converted-model final logits with HF native PyTorch on CPU.")
    ap.add_argument("--torch-compare-tolerance", type=float, default=0.2)
    ap.add_argument("--results", default="")
    ap.add_argument("--require-apple", action="store_true")
    ap.add_argument("--require-mlx", action="store_true")
    ap.add_argument("--skip-tiny", action="store_true")
    args = ap.parse_args()

    if not is_apple_silicon():
        row = {
            "axis": "apple_silicon_mlx_recurrent_smoke",
            "status": "skip",
            "reason": "not Darwin/arm64",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": Path(args.model).name if args.model else "",
        }
        emit(args.results, row)
        if args.require_apple:
            raise SystemExit(2)
        return 0

    if not mlx_available():
        row = {
            "axis": "apple_silicon_mlx_recurrent_smoke",
            "status": "skip",
            "reason": "mlx not installed",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": Path(args.model).name if args.model else "",
        }
        emit(args.results, row)
        if args.require_mlx:
            raise SystemExit(2)
        return 0

    import mlx.core as mx

    header = {
        "axis": "apple_silicon_mlx_recurrent_env",
        "status": "info",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": darwin_sysctl("machdep.cpu.brand_string"),
        "memory_gb": apple_memory_gb(),
        "mlx": package_version("mlx"),
        "mlx_default_device": str(mx.default_device()),
        "dtype": args.dtype,
        "model": Path(args.model).name if args.model else "",
    }
    emit(args.results, header)

    if not args.skip_tiny:
        emit(args.results, run_tiny_recurrent_parity(args))
        emit(args.results, run_tiny_state_cache(args))
        emit(args.results, run_tiny_generation_session(args))
        emit(args.results, run_tiny_generation_session_batch(args))
    if args.model:
        emit(args.results, run_real_model_smoke(args))

    print("APPLE SILICON MLX RECURRENT SMOKE PASS")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("RWKV7_NATIVE_MODEL_JIT", "0")
    raise SystemExit(main())
