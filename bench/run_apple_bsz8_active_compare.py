#!/usr/bin/env python3
"""True-batch Apple W4 comparison normalized by active text parameters.

The public mode launches each runtime in an isolated child process so model
residency cannot contaminate the other runtime's peak-memory row.  RWKV uses
the production candidate assembled for B8: thread-local Metal scan, fused MLX
LayerNorm, native groupwise W4, and fused packed embedding lookup.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROMPT_SEED = (
    "User: Compare RWKV-7 and Qwen3.5 on Apple Silicon. Report throughput, latency, "
    "memory, state-cache behavior, and quantization stability.\nAssistant: "
)
KNOWN_ACTIVE_PARAMS = {
    "rwkv7-g1d-0.4b-hf": 450_767_872,
    "rwkv7-g1g-1.5b-hf": 1_527_404_544,
    "qwen35-0.8b-mlx-4bit": 752_393_024,
    "qwen35-2b-mlx-4bit": 1_881_825_088,
}


def make_prompt(chars: int) -> str:
    if int(chars) <= 0:
        raise ValueError("--prompt-chars must be positive")
    return (PROMPT_SEED * ((int(chars) + len(PROMPT_SEED) - 1) // len(PROMPT_SEED)))[: int(chars)]


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_qwen_child(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator

    started = time.perf_counter()
    model, tokenizer = load(args.qwen_model)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - started
    prompt = make_prompt(args.prompt_chars)
    prompt_ids = [int(value) for value in tokenizer.encode(prompt, add_special_tokens=True)]
    prompts = [prompt_ids] * int(args.batch_size)

    def one(record: bool, repeat_index: int) -> dict[str, Any]:
        generator = BatchGenerator(
            model,
            max_tokens=int(args.decode_tokens),
            stop_tokens=None,
            completion_batch_size=int(args.batch_size),
            prefill_batch_size=int(args.batch_size),
            prefill_step_size=2048,
        )
        uids = generator.insert(prompts, [int(args.decode_tokens)] * int(args.batch_size))
        outputs: dict[int, list[int]] = {int(uid): [] for uid in uids}
        with generator.stats() as stats:
            while generated := generator.next_generated():
                for item in generated:
                    if item.finish_reason != "stop":
                        outputs[int(item.uid)].append(int(item.token))
        mx.synchronize()
        generator.close()
        values = [outputs[int(uid)] for uid in uids]
        row = {
            "axis": "apple_bsz8_active_compare",
            "engine": "qwen_mlx_lm",
            "status": "pass" if all(len(value) == int(args.decode_tokens) for value in values) else "fail",
            "model": Path(args.qwen_model).name,
            "model_path": args.qwen_model,
            "batch_size": int(args.batch_size),
            "prompt_chars": len(prompt),
            "prompt_tokens_per_sequence": len(prompt_ids),
            "prompt_tokens_total": int(stats.prompt_tokens),
            "generated_tokens_total": int(stats.generation_tokens),
            "decode_tokens_per_sequence": int(args.decode_tokens),
            "prefill_s": float(stats.prompt_time),
            "decode_s": float(stats.generation_time),
            "prefill_tok_s_aggregate": float(stats.prompt_tps),
            "decode_tok_s_aggregate": float(stats.generation_tps),
            "peak_memory_bytes": int(float(stats.peak_memory) * 1e9),
            "generated_lengths": [len(value) for value in values],
            "generated_preview": values[0][:16],
            "all_sequences_equal": len({tuple(value) for value in values}) == 1,
            "repeat_index": int(repeat_index),
            "load_s": float(load_s),
        }
        return row

    for index in range(int(args.warmup)):
        one(False, -index - 1)
    rows = []
    for index in range(1, int(args.repeat) + 1):
        # Measure production steady state after warmup. Clearing MLX's
        # allocator cache here charges buffer allocation to every sample and
        # disproportionately penalizes runtimes with many fused graph outputs.
        mx.reset_peak_memory()
        rows.append(one(True, index))
    return {"engine": "qwen", "rows": rows}


def run_rwkv_child(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["RWKV7_MLX_WKV_SCAN_PREFILL"] = "1"
    os.environ["RWKV7_MLX_FAST_LAYER_NORM"] = "1" if args.rwkv_fast_layer_norm else "0"
    os.environ["RWKV7_MLX_QUANTIZE_EMBEDDING"] = "1"
    os.environ["RWKV7_MLX_FUSED_FFN_KEY_RELU2"] = "1" if args.rwkv_fused_ffn_key_relu2 else "0"
    os.environ["RWKV7_MLX_STEP_EVAL_INTERVAL"] = str(int(args.rwkv_step_eval_interval))
    os.environ["RWKV7_MLX_DECODE_NORM_BACKEND"] = "fast"
    os.environ["RWKV7_MLX_DECODE_FAST_GROUP_NORM"] = "1" if args.rwkv_decode_fast_group_norm else "0"
    os.environ["RWKV7_MLX_FUSED_LORA_DOWN"] = "1" if args.rwkv_fused_lora_down else "0"
    os.environ["RWKV7_MLX_FUSED_LORA_DOWN_INCLUDE_V"] = (
        "1" if args.rwkv_fused_lora_down_include_v else "0"
    )
    os.environ["RWKV7_MLX_FUSED_LORA_UP"] = "1" if args.rwkv_fused_lora_up else "0"
    os.environ["RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION"] = "1" if args.rwkv_group_rkv else "0"
    os.environ["RWKV7_MLX_FUSED_SCAN_POST"] = "1" if args.rwkv_fused_scan_post else "0"

    import mlx.core as mx
    from transformers import AutoTokenizer

    from rwkv7_hf.mlx_model import load_mlx_rwkv7_model
    from rwkv7_hf.mlx_speculative import speculative_decode_greedy_batch

    started = time.perf_counter()
    model = load_mlx_rwkv7_model(
        args.rwkv_model,
        dtype="fp16",
        quantization="mm4",
        quant_min_params=int(args.rwkv_quant_min_params),
        quant_rkv_min_params=int(args.rwkv_quant_min_params),
        quant_backend="groupwise",
        quant_group_size=int(args.rwkv_quant_group_size),
        quantize_embedding=True,
        wkv_backend="metal",
    )
    model.decode_backend = "eager"
    draft_model = None
    if args.rwkv_draft_model:
        draft_model = load_mlx_rwkv7_model(
            args.rwkv_draft_model,
            dtype="fp16",
            quantization="mm4",
            quant_min_params=int(args.rwkv_draft_quant_min_params),
            quant_rkv_min_params=int(args.rwkv_draft_quant_min_params),
            quant_backend="groupwise",
            quant_group_size=int(args.rwkv_quant_group_size),
            quantize_embedding=True,
            wkv_backend="metal",
        )
        draft_model.decode_backend = "eager"
    tokenizer = AutoTokenizer.from_pretrained(args.rwkv_model, trust_remote_code=True)
    load_s = time.perf_counter() - started
    prompt = make_prompt(args.prompt_chars)
    prompt_ids = [int(value) for value in tokenizer(prompt, add_special_tokens=True).input_ids]
    ids = mx.array([prompt_ids] * int(args.batch_size), dtype=mx.int32)
    cache_indices = [0] * int(args.batch_size)

    def prefill_request(request_model):
        if not args.rwkv_prefix_cache_dedup:
            return request_model.prefill(ids)
        seed_logits, seed_state = request_model.prefill(ids[:1])
        logits = mx.repeat(seed_logits, int(args.batch_size), axis=0)
        state = seed_state.select_batch(cache_indices)
        return logits, state

    compiled_prefill_validation = None
    if args.rwkv_compiled_prefill:
        prompt_validation_ids = ids[:1] if args.rwkv_prefix_cache_dedup else ids
        compiled_prefill_validation = model.validate_compiled_scan_prefill(prompt_validation_ids)
        if compiled_prefill_validation.get("status") != "pass":
            raise RuntimeError(
                "compiled RWKV scan prefill failed the concrete model/shape parity gate: "
                + json.dumps(compiled_prefill_validation, ensure_ascii=False)
            )
        model.compiled_scan_prefill_mode = "on"
        if draft_model is not None:
            draft_validation = draft_model.validate_compiled_scan_prefill(prompt_validation_ids)
            if draft_validation.get("status") != "pass":
                raise RuntimeError(
                    "compiled draft scan prefill failed parity: "
                    + json.dumps(draft_validation, ensure_ascii=False)
                )
            draft_model.compiled_scan_prefill_mode = "on"
    compiled_decode_validation = None
    if args.rwkv_decode_backend == "compiled":
        validation_logits, validation_state = model.prefill(ids)
        compiled_decode_validation = model.validate_compiled_decode(
            validation_logits,
            validation_state,
            steps=int(args.rwkv_decode_validation_steps),
        )
        if compiled_decode_validation.get("status") != "pass":
            raise RuntimeError(
                "compiled RWKV decode failed the concrete model/batch parity gate: "
                + json.dumps(compiled_decode_validation, ensure_ascii=False)
            )
        model.decode_backend = "compiled"
    speculative_verify_validation = None
    if draft_model is not None:
        target_probe_logits, target_probe_state = prefill_request(model)
        draft_probe_logits, draft_probe_state = prefill_request(draft_model)
        proposals = []
        for _ in range(int(args.rwkv_proposal_tokens)):
            proposal = mx.argmax(draft_probe_logits[:, -1, :], axis=-1).astype(mx.int32)
            mx.eval(proposal)
            proposals.append(proposal)
            draft_probe_logits, draft_probe_state = draft_model.decode_step(proposal, draft_probe_state)
        proposal_block = mx.stack(proposals, axis=1)
        mx.eval(proposal_block)
        speculative_verify_validation = model.validate_compiled_scan_prefill(
            proposal_block,
            target_probe_state,
            collect_all=True,
        )
        if speculative_verify_validation.get("status") != "pass":
            raise RuntimeError(
                "compiled speculative verifier failed parity: "
                + json.dumps(speculative_verify_validation, ensure_ascii=False)
            )
        model.compiled_scan_prefill_mode = "on"
        # Validation owns full target/draft B8 states.  They are not serving
        # cache entries and retaining them would inflate every measured row,
        # especially the 1.5B target whose recurrent state is about 100 MiB.
        del (
            target_probe_logits,
            target_probe_state,
            draft_probe_logits,
            draft_probe_state,
            proposals,
            proposal,
            proposal_block,
        )
        gc.collect()
        mx.clear_cache()

    def one(repeat_index: int) -> dict[str, Any]:
        # The compiled fresh-prompt graph internalizes zero recurrent state;
        # this is the production first-prefill path and avoids state inputs.
        started_prefill = time.perf_counter()
        logits, state = prefill_request(model)
        mx.eval(logits, state.v_first, *state.recurrent_state, *state.attn_x_prev, *state.ffn_x_prev)
        prefill_s = time.perf_counter() - started_prefill
        target_prefill_peak_memory_bytes = int(mx.get_peak_memory())
        mx.reset_peak_memory()
        draft_prefill_s = None
        speculative_telemetry = None
        if draft_model is not None:
            started_draft_prefill = time.perf_counter()
            draft_logits, draft_state = prefill_request(draft_model)
            mx.eval(
                draft_logits,
                draft_state.v_first,
                *draft_state.recurrent_state,
                *draft_state.attn_x_prev,
                *draft_state.ffn_x_prev,
            )
            draft_prefill_s = time.perf_counter() - started_draft_prefill
            speculative = speculative_decode_greedy_batch(
                model,
                draft_model,
                logits,
                state,
                draft_logits,
                draft_state,
                max_new_tokens=int(args.decode_tokens),
                proposal_tokens=int(args.rwkv_proposal_tokens),
            )
            decode_s = float(speculative.elapsed_s)
            values = speculative.generated_ids
            state = speculative.target_state
            speculative_telemetry = speculative.telemetry()
        else:
            next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
            generated = []
            started_decode = time.perf_counter()
            for _ in range(int(args.decode_tokens)):
                mx.eval(next_token)
                generated.append(next_token)
                logits, state = model.decode_step(next_token, state)
                next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
            mx.eval(
                next_token,
                *generated,
                state.v_first,
                *state.recurrent_state,
                *state.attn_x_prev,
                *state.ffn_x_prev,
            )
            decode_s = time.perf_counter() - started_decode
            generated_array = mx.stack(generated, axis=1)
            mx.eval(generated_array)
            values = generated_array.tolist()
        telemetry = model.telemetry()
        decode_phase_peak_memory_bytes = int(mx.get_peak_memory())
        peak_memory_bytes = max(target_prefill_peak_memory_bytes, decode_phase_peak_memory_bytes)
        return {
            "axis": "apple_bsz8_active_compare",
            "engine": "rwkv7_mlx_speculative" if draft_model is not None else "rwkv7_mlx",
            "status": "pass" if all(len(value) == int(args.decode_tokens) for value in values) else "fail",
            "model": Path(args.rwkv_model).name,
            "model_path": args.rwkv_model,
            "batch_size": int(args.batch_size),
            "prompt_chars": len(prompt),
            "prompt_tokens_per_sequence": len(prompt_ids),
            "prompt_tokens_total": len(prompt_ids) * int(args.batch_size),
            "generated_tokens_total": int(args.decode_tokens) * int(args.batch_size),
            "decode_tokens_per_sequence": int(args.decode_tokens),
            "prefill_s": float(prefill_s),
            "decode_s": float(decode_s),
            "draft_prefill_s": draft_prefill_s,
            "decode_with_draft_prefill_s": (
                float(decode_s + draft_prefill_s) if draft_prefill_s is not None else None
            ),
            "prefill_tok_s_aggregate": len(prompt_ids) * int(args.batch_size) / prefill_s,
            "decode_tok_s_aggregate": int(args.decode_tokens) * int(args.batch_size) / decode_s,
            "peak_memory_bytes": int(peak_memory_bytes),
            "target_prefill_peak_memory_bytes": int(target_prefill_peak_memory_bytes),
            "decode_phase_peak_memory_bytes": int(decode_phase_peak_memory_bytes),
            "generated_lengths": [len(value) for value in values],
            "generated_preview": values[0][:16],
            "all_sequences_equal": len({tuple(value) for value in values}) == 1,
            "prefix_state_cache_dedup": bool(args.rwkv_prefix_cache_dedup),
            "prefix_state_cache_unique_prompts": 1 if args.rwkv_prefix_cache_dedup else int(args.batch_size),
            "prefix_state_cache_hits": int(args.batch_size) - 1 if args.rwkv_prefix_cache_dedup else 0,
            "prefix_state_cache_hit_rate": (
                (int(args.batch_size) - 1) / int(args.batch_size)
                if args.rwkv_prefix_cache_dedup
                else 0.0
            ),
            "repeat_index": int(repeat_index),
            "load_s": float(load_s),
            "prefill_backend": "metal_scan",
            "compiled_prefill_backend": telemetry.get("compiled_scan_prefill_backend_last"),
            "compiled_prefill_validation": compiled_prefill_validation,
            "wkv_scan_prefill_counts": telemetry.get("wkv_scan_prefill_counts"),
            "quantized_embedding": telemetry.get("quantized_embedding"),
            "quantized_embedding_backend_counts": telemetry.get("quantized_embedding_backend_counts"),
            "quantized_embedding_footprint_ratio": telemetry.get("quantized_embedding_footprint_ratio"),
            "fast_layer_norm": telemetry.get("fast_layer_norm"),
            "fused_ffn_key_relu2_counts": telemetry.get("fused_ffn_key_relu2_counts"),
            "decode_fast_group_norm": telemetry.get("decode_fast_group_norm"),
            "fused_lora_down_counts": telemetry.get("fused_lora_down_counts"),
            "fused_lora_down_include_v": telemetry.get("fused_lora_down_include_v"),
            "fused_lora_up_counts": telemetry.get("fused_lora_up_counts"),
            "fused_scan_post_counts": telemetry.get("fused_scan_post_counts"),
            "group_rkv_quant_projection_counts": telemetry.get("group_rkv_quant_projection_counts"),
            "decode_backend": telemetry.get("decode_backend_last"),
            "compiled_decode_validation": compiled_decode_validation,
            "draft_model": Path(args.rwkv_draft_model).name if args.rwkv_draft_model else None,
            "speculative_verify_validation": speculative_verify_validation,
            "speculative_telemetry": speculative_telemetry,
        }

    for _ in range(int(args.warmup)):
        one(0)
    rows = []
    for index in range(1, int(args.repeat) + 1):
        # Keep the warmed allocator cache, matching a resident serving
        # process. Validation-only arrays were explicitly released above, so
        # they cannot inflate these peak-memory samples.
        mx.reset_peak_memory()
        rows.append(one(index))
    return {"engine": "rwkv", "rows": rows}


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def inferred_active_params(path: str, explicit: int) -> int:
    if int(explicit) > 0:
        return int(explicit)
    name = Path(path).name
    if name not in KNOWN_ACTIVE_PARAMS:
        raise ValueError(f"unknown active-parameter count for {name}; pass the explicit --*-active-params value")
    return int(KNOWN_ACTIVE_PARAMS[name])


def child_command(args: argparse.Namespace, engine: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-engine",
        engine,
        "--rwkv-model",
        args.rwkv_model,
        "--qwen-model",
        args.qwen_model,
        "--batch-size",
        str(args.batch_size),
        "--prompt-chars",
        str(args.prompt_chars),
        "--decode-tokens",
        str(args.decode_tokens),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
        "--rwkv-quant-min-params",
        str(args.rwkv_quant_min_params),
        "--rwkv-quant-group-size",
        str(args.rwkv_quant_group_size),
        "--rwkv-step-eval-interval",
        str(args.rwkv_step_eval_interval),
        "--rwkv-decode-backend",
        args.rwkv_decode_backend,
        "--rwkv-decode-validation-steps",
        str(args.rwkv_decode_validation_steps),
        "--rwkv-draft-model",
        args.rwkv_draft_model,
        "--rwkv-draft-quant-min-params",
        str(args.rwkv_draft_quant_min_params),
        "--rwkv-proposal-tokens",
        str(args.rwkv_proposal_tokens),
    ]
    command.append("--rwkv-compiled-prefill" if args.rwkv_compiled_prefill else "--no-rwkv-compiled-prefill")
    command.append(
        "--rwkv-fast-layer-norm" if args.rwkv_fast_layer_norm else "--no-rwkv-fast-layer-norm"
    )
    command.append(
        "--rwkv-decode-fast-group-norm"
        if args.rwkv_decode_fast_group_norm
        else "--no-rwkv-decode-fast-group-norm"
    )
    command.append("--rwkv-fused-lora-down" if args.rwkv_fused_lora_down else "--no-rwkv-fused-lora-down")
    command.append(
        "--rwkv-fused-lora-down-include-v"
        if args.rwkv_fused_lora_down_include_v
        else "--no-rwkv-fused-lora-down-include-v"
    )
    command.append(
        "--rwkv-fused-ffn-key-relu2"
        if args.rwkv_fused_ffn_key_relu2
        else "--no-rwkv-fused-ffn-key-relu2"
    )
    command.append("--rwkv-fused-lora-up" if args.rwkv_fused_lora_up else "--no-rwkv-fused-lora-up")
    command.append("--rwkv-fused-scan-post" if args.rwkv_fused_scan_post else "--no-rwkv-fused-scan-post")
    command.append("--rwkv-group-rkv" if args.rwkv_group_rkv else "--no-rwkv-group-rkv")
    command.append(
        "--rwkv-prefix-cache-dedup"
        if args.rwkv_prefix_cache_dedup
        else "--no-rwkv-prefix-cache-dedup"
    )
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rwkv-model", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--rwkv-active-params", type=int, default=0)
    parser.add_argument("--qwen-active-params", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-chars", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rwkv-quant-min-params", type=int, default=1_000_000)
    parser.add_argument("--rwkv-quant-group-size", type=int, choices=[32, 64, 128], default=128)
    parser.add_argument("--rwkv-step-eval-interval", type=int, default=64)
    parser.add_argument(
        "--rwkv-compiled-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parity-gate and use the static-shape compiled Metal scan prefill graph.",
    )
    parser.add_argument(
        "--rwkv-fast-layer-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mx.fast.layer_norm globally; on is the parity-gated M5 B8 default.",
    )
    parser.add_argument(
        "--rwkv-decode-fast-group-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use phase-specific fast per-head normalization for decode only.",
    )
    parser.add_argument(
        "--rwkv-fused-lora-down",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the measured-positive two-GEMM packed W/A low-rank down projection.",
    )
    parser.add_argument(
        "--rwkv-fused-ffn-key-relu2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the affine-MM4 FFN key/ReLU-squared Metal seam when its weight layout supports it.",
    )
    parser.add_argument(
        "--rwkv-fused-lora-down-include-v",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include V in packed LoRA-down GEMMs; the B8 default keeps its direct rank-32 GEMM.",
    )
    parser.add_argument(
        "--rwkv-fused-scan-post",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fuse FP16 WKV scan, per-head GroupNorm, bonus, and gate in one Metal kernel."
        ),
    )
    parser.add_argument(
        "--rwkv-group-rkv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use native batched groupwise W4 R/K/V projection.",
    )
    parser.add_argument(
        "--rwkv-fused-lora-up",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the batched W/A/V low-rank up projection.",
    )
    parser.add_argument("--rwkv-decode-backend", choices=["eager", "compiled"], default="eager")
    parser.add_argument("--rwkv-decode-validation-steps", type=int, default=64)
    parser.add_argument("--rwkv-draft-model", default="")
    parser.add_argument("--rwkv-draft-quant-min-params", type=int, default=100_000)
    parser.add_argument("--rwkv-proposal-tokens", type=int, default=32)
    parser.add_argument(
        "--rwkv-prefix-cache-dedup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Coalesce identical B8 prompts through one recurrent prefix-state computation.",
    )
    parser.add_argument("--order", choices=["qwen-first", "rwkv-first"], default="qwen-first")
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=30.0,
        help="Idle interval between isolated engines to avoid order-biased fanless-Mac throttling.",
    )
    parser.add_argument("--results", default="")
    parser.add_argument("--child-engine", choices=["qwen", "rwkv"], default="")
    args = parser.parse_args(argv)
    if (
        min(
            args.batch_size,
            args.prompt_chars,
            args.decode_tokens,
            args.repeat,
            args.rwkv_decode_validation_steps,
            args.rwkv_proposal_tokens,
        )
        <= 0
        or args.warmup < 0
        or args.cooldown_seconds < 0
    ):
        parser.error("batch/prompt/decode/repeat must be positive and warmup non-negative")

    if args.child_engine:
        payload = run_qwen_child(args) if args.child_engine == "qwen" else run_rwkv_child(args)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    order = ["qwen", "rwkv"] if args.order == "qwen-first" else ["rwkv", "qwen"]
    payloads: dict[str, dict[str, Any]] = {}
    for engine_index, engine in enumerate(order):
        if engine_index and float(args.cooldown_seconds) > 0:
            time.sleep(float(args.cooldown_seconds))
        process = subprocess.run(child_command(args, engine), check=True, text=True, capture_output=True)
        payload = json.loads(process.stdout.strip().splitlines()[-1])
        payloads[engine] = payload
        for row in payload["rows"]:
            print(json.dumps(row, ensure_ascii=False))
            append_jsonl(args.results, row)

    rwkv_rows = payloads["rwkv"]["rows"]
    qwen_rows = payloads["qwen"]["rows"]
    rwkv_params = inferred_active_params(args.rwkv_model, args.rwkv_active_params)
    qwen_params = inferred_active_params(args.qwen_model, args.qwen_active_params)
    rwkv_prefill = metric_summary(rwkv_rows, "prefill_tok_s_aggregate")
    qwen_prefill = metric_summary(qwen_rows, "prefill_tok_s_aggregate")
    rwkv_decode = metric_summary(rwkv_rows, "decode_tok_s_aggregate")
    qwen_decode = metric_summary(qwen_rows, "decode_tok_s_aggregate")
    rwkv_peak = metric_summary(rwkv_rows, "peak_memory_bytes")
    qwen_peak = metric_summary(qwen_rows, "peak_memory_bytes")
    prefill_normalized_ratio = rwkv_prefill["median"] * rwkv_params / (qwen_prefill["median"] * qwen_params)
    decode_normalized_ratio = rwkv_decode["median"] * rwkv_params / (qwen_decode["median"] * qwen_params)
    memory_per_param_ratio = (rwkv_peak["max"] / rwkv_params) / (qwen_peak["max"] / qwen_params)
    speed_pass = prefill_normalized_ratio >= 1.0 and decode_normalized_ratio >= 1.0
    raw_memory_pass = rwkv_peak["max"] <= qwen_peak["max"]
    rows_pass = all(row.get("status") == "pass" for row in [*rwkv_rows, *qwen_rows])
    summary = {
        "axis": "apple_bsz8_active_compare_summary",
        "status": "pass" if rows_pass and speed_pass and raw_memory_pass else "fail",
        "batch_size": int(args.batch_size),
        "prompt_chars": int(args.prompt_chars),
        "decode_tokens_per_sequence": int(args.decode_tokens),
        "rwkv_model": Path(args.rwkv_model).name,
        "qwen_model": Path(args.qwen_model).name,
        "rwkv_active_text_params": rwkv_params,
        "qwen_active_text_params": qwen_params,
        "normalization": "aggregate_tok_s * active_text_parameter_count; higher is better",
        "rwkv_prefix_state_cache_dedup": bool(args.rwkv_prefix_cache_dedup),
        "rwkv_prefill_tok_s": rwkv_prefill,
        "qwen_prefill_tok_s": qwen_prefill,
        "rwkv_decode_tok_s": rwkv_decode,
        "qwen_decode_tok_s": qwen_decode,
        "rwkv_peak_memory_bytes": rwkv_peak,
        "qwen_peak_memory_bytes": qwen_peak,
        "raw_prefill_ratio": rwkv_prefill["median"] / qwen_prefill["median"],
        "raw_decode_ratio": rwkv_decode["median"] / qwen_decode["median"],
        "active_normalized_prefill_ratio": prefill_normalized_ratio,
        "active_normalized_decode_ratio": decode_normalized_ratio,
        "rwkv_vs_qwen_memory_per_active_param_ratio": memory_per_param_ratio,
        "speed_gate_pass": speed_pass,
        "raw_peak_memory_gate_pass": raw_memory_pass,
        "active_normalized_memory_gate_pass": memory_per_param_ratio <= 1.0,
        "rows_gate_pass": rows_pass,
    }
    print(json.dumps(summary, ensure_ascii=False))
    append_jsonl(args.results, summary)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
