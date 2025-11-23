#!/usr/bin/env python3
"""Validate the Apple B8 W4 candidate against FP16, generic scan, and cache paths."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any


PROMPT_SEED = (
    "User: Compare RWKV-7 and Qwen3.5 on Apple Silicon. Report throughput, latency, "
    "memory, state-cache behavior, and quantization stability.\nAssistant: "
)


def make_prompt(chars: int) -> str:
    count = int(chars)
    if count <= 0:
        raise ValueError("--prompt-chars must be positive")
    return (PROMPT_SEED * ((count + len(PROMPT_SEED) - 1) // len(PROMPT_SEED)))[:count]


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def flat_state(state) -> list[Any]:
    return [state.v_first, *state.recurrent_state, *state.attn_x_prev, *state.ffn_x_prev]


def state_max_abs(mx, left, right) -> float:
    return max(
        float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        for a, b in zip(flat_state(left), flat_state(right), strict=True)
    )


def configure(args: argparse.Namespace, *, fused_scan_post: bool) -> None:
    os.environ["RWKV7_MLX_WKV_SCAN_PREFILL"] = "1"
    os.environ["RWKV7_MLX_FAST_LAYER_NORM"] = "1"
    os.environ["RWKV7_MLX_QUANTIZE_EMBEDDING"] = "1"
    os.environ["RWKV7_MLX_FUSED_LORA_DOWN"] = "1" if args.fused_lora_down else "0"
    os.environ["RWKV7_MLX_FUSED_LORA_DOWN_INCLUDE_G"] = "0"
    os.environ["RWKV7_MLX_FUSED_LORA_DOWN_INCLUDE_V"] = "0"
    os.environ["RWKV7_MLX_FUSED_LORA_UP"] = "0"
    os.environ["RWKV7_MLX_FUSED_FFN_KEY_RELU2"] = "1" if args.fused_ffn_key_relu2 else "0"
    os.environ["RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION"] = "0"
    os.environ["RWKV7_MLX_FUSED_SCAN_POST"] = "1" if fused_scan_post else "0"
    os.environ["RWKV7_MLX_FUSED_SCAN_PREP_POST"] = (
        "1" if fused_scan_post and args.fused_scan_prep_post else "0"
    )
    os.environ["RWKV7_MLX_FUSED_SEQUENCE_MIX"] = "1" if args.fused_sequence_mix else "0"
    os.environ["RWKV7_MLX_FUSED_ADD_LAYER_NORM"] = "1" if args.fused_add_layer_norm else "0"
    os.environ["RWKV7_MLX_FUSED_SQUARE_QMM"] = "1" if args.fused_square_qmm else "0"
    os.environ["RWKV7_MLX_FLATTEN_WIDE_GROUPWISE_PREFILL"] = "1"
    os.environ["RWKV7_MLX_DECODE_FAST_GROUP_NORM"] = "1"
    os.environ["RWKV7_MLX_STEP_EVAL_INTERVAL"] = "64"


def load_model(args: argparse.Namespace, *, quantization: str, fused_scan_post: bool):
    configure(args, fused_scan_post=fused_scan_post)
    from rwkv7_hf.mlx_model import load_mlx_rwkv7_model

    quantized = quantization == "mm4"
    return load_mlx_rwkv7_model(
        args.model,
        dtype="fp16",
        quantization=quantization,
        quant_min_params=int(args.quant_min_params),
        quant_rkv_min_params=int(args.quant_min_params),
        quant_backend="groupwise",
        quant_group_size=int(args.quant_group_size),
        quantize_embedding=quantized,
        wkv_backend="metal",
    )


def mixed_prompt_ids(prompt_ids: list[int], *, batch_size: int, unique_prompts: int, vocab_size: int):
    """Build equal-length deterministic prefixes and their B8 reuse mapping."""

    unique = []
    for index in range(int(unique_prompts)):
        row = list(prompt_ids)
        if index:
            position = max(0, len(row) - 2 - (index % max(1, min(8, len(row)))))
            row[position] = 1 + ((int(row[position]) + 7919 * index) % max(1, int(vocab_size) - 1))
        unique.append(row)
    mapping = [index % int(unique_prompts) for index in range(int(batch_size))]
    return unique, mapping


def run_greedy(mx, model, ids, tokens: int):
    logits, prefill_state = model.prefill(ids)
    generated, final_state = model.decode_greedy(logits, prefill_state.clone(), max_new_tokens=int(tokens))
    mx.eval(logits, generated, *flat_state(prefill_state), *flat_state(final_state))
    return logits, prefill_state, generated, final_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-chars", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--quant-min-params", type=int, default=1_000_000)
    parser.add_argument("--quant-group-size", type=int, choices=[32, 64, 128], default=128)
    parser.add_argument("--fused-lora-down", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-ffn-key-relu2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare-fused-post", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fused-scan-prep-post",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fused-sequence-mix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-add-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-square-qmm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare-prefix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix-unique-prompts", type=int, default=2)
    parser.add_argument("--draft-model", default="")
    parser.add_argument("--draft-quant-min-params", type=int, default=100_000)
    parser.add_argument(
        "--compare-speculative-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--results", default="")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.prompt_chars, args.decode_tokens) <= 0:
        parser.error("batch, prompt chars, and decode tokens must be positive")
    if not 1 <= int(args.prefix_unique_prompts) <= int(args.batch_size):
        parser.error("--prefix-unique-prompts must be in [1, batch-size]")
    if args.compare_speculative_mismatch and not args.draft_model:
        parser.error("--compare-speculative-mismatch requires --draft-model")

    import mlx.core as mx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = make_prompt(args.prompt_chars)
    prompt_ids = [int(value) for value in tokenizer(prompt, add_special_tokens=True).input_ids]
    ids = mx.array([prompt_ids] * int(args.batch_size), dtype=mx.int32)
    result: dict[str, Any] = {
        "axis": "apple_bsz8_w4_fidelity",
        "model": Path(args.model).name,
        "batch_size": int(args.batch_size),
        "prompt_chars": len(prompt),
        "prompt_tokens_per_sequence": len(prompt_ids),
        "decode_tokens_per_sequence": int(args.decode_tokens),
        "quant_group_size": int(args.quant_group_size),
        "fused_lora_down": bool(args.fused_lora_down),
        "fused_ffn_key_relu2": bool(args.fused_ffn_key_relu2),
        "fused_scan_prep_post": bool(args.fused_scan_prep_post),
        "fused_sequence_mix": bool(args.fused_sequence_mix),
        "fused_add_layer_norm": bool(args.fused_add_layer_norm),
        "fused_square_qmm": bool(args.fused_square_qmm),
    }
    gates: list[bool] = []

    fp16_logits = fp16_tokens = None
    if args.compare_fp16:
        fp16 = load_model(args, quantization="none", fused_scan_post=True)
        fp16_logits, _, fp16_tokens, _ = run_greedy(mx, fp16, ids, args.decode_tokens)
        del fp16
        gc.collect()
        mx.clear_cache()

    optimized = load_model(args, quantization="mm4", fused_scan_post=True)
    compiled_validation = optimized.validate_compiled_scan_prefill(ids)
    optimized.compiled_scan_prefill_mode = "auto"
    opt_logits, opt_prefill_state, opt_tokens, opt_final_state = run_greedy(
        mx, optimized, ids, args.decode_tokens
    )
    result["compiled_prefill_validation"] = compiled_validation
    gates.append(compiled_validation.get("status") == "pass")
    if args.compare_fp16:
        fp16_exact = fp16_tokens.tolist() == opt_tokens.tolist()
        result["w4_vs_fp16_greedy_exact"] = fp16_exact
        result["w4_vs_fp16_token_match_rate"] = sum(
            int(a == b)
            for left, right in zip(fp16_tokens.tolist(), opt_tokens.tolist(), strict=True)
            for a, b in zip(left, right, strict=True)
        ) / (int(args.batch_size) * int(args.decode_tokens))
        result["w4_vs_fp16_prefill_logits_max_abs"] = float(
            mx.max(mx.abs(fp16_logits.astype(mx.float32) - opt_logits.astype(mx.float32)))
        )
        gates.append(fp16_exact)

    if args.compare_fused_post:
        generic = load_model(args, quantization="mm4", fused_scan_post=False)
        generic_logits, generic_prefill_state, generic_tokens, generic_final_state = run_greedy(
            mx, generic, ids, args.decode_tokens
        )
        fused_exact = generic_tokens.tolist() == opt_tokens.tolist()
        fused_logits_diff = float(
            mx.max(mx.abs(generic_logits.astype(mx.float32) - opt_logits.astype(mx.float32)))
        )
        fused_prefill_state_diff = state_max_abs(mx, generic_prefill_state, opt_prefill_state)
        fused_final_state_diff = state_max_abs(mx, generic_final_state, opt_final_state)
        result.update(
            {
                "fused_post_greedy_exact": fused_exact,
                "fused_post_prefill_logits_max_abs": fused_logits_diff,
                "fused_post_prefill_state_max_abs": fused_prefill_state_diff,
                "fused_post_final_state_max_abs": fused_final_state_diff,
            }
        )
        gates.append(
            fused_exact
            and fused_logits_diff <= 0.25
            and fused_prefill_state_diff <= 0.5
            and fused_final_state_diff <= 0.5
        )
        del generic
        gc.collect()
        mx.clear_cache()

    if args.compare_prefix_cache:
        unique_ids, cache_indices = mixed_prompt_ids(
            prompt_ids,
            batch_size=int(args.batch_size),
            unique_prompts=int(args.prefix_unique_prompts),
            vocab_size=int(optimized.vocab_size),
        )
        unique_array = mx.array(unique_ids, dtype=mx.int32)
        mixed_ids = mx.take(unique_array, mx.array(cache_indices, dtype=mx.int32), axis=0)
        mixed_logits, mixed_prefill_state, mixed_tokens, mixed_final_state = run_greedy(
            mx,
            optimized,
            mixed_ids,
            args.decode_tokens,
        )
        seed_logits, seed_state = optimized.prefill(unique_array)
        cache_logits = mx.take(seed_logits, mx.array(cache_indices, dtype=mx.int32), axis=0)
        cache_state = seed_state.select_batch(cache_indices)
        cache_tokens, cache_final_state = optimized.decode_greedy(
            cache_logits,
            cache_state.clone(),
            max_new_tokens=int(args.decode_tokens),
        )
        mx.eval(cache_logits, cache_tokens, *flat_state(cache_state), *flat_state(cache_final_state))
        cache_exact = cache_tokens.tolist() == mixed_tokens.tolist()
        cache_logits_diff = float(
            mx.max(mx.abs(cache_logits.astype(mx.float32) - mixed_logits.astype(mx.float32)))
        )
        cache_prefill_state_diff = state_max_abs(mx, cache_state, mixed_prefill_state)
        cache_final_state_diff = state_max_abs(mx, cache_final_state, mixed_final_state)
        permutation = list(reversed(range(int(args.batch_size))))
        reordered_logits = mx.take(cache_logits, mx.array(permutation, dtype=mx.int32), axis=0)
        reordered_state = cache_state.reorder_cache(permutation)
        compacted_state = cache_state.compact(permutation)
        reorder_compact_state_diff = state_max_abs(mx, reordered_state, compacted_state)
        reordered_tokens, reordered_final_state = optimized.decode_greedy(
            reordered_logits,
            reordered_state.clone(),
            max_new_tokens=int(args.decode_tokens),
        )
        expected_reordered_tokens = mx.take(
            mixed_tokens,
            mx.array(permutation, dtype=mx.int32),
            axis=0,
        )
        expected_reordered_final = mixed_final_state.select_batch(permutation)
        mx.eval(reordered_tokens, expected_reordered_tokens)
        reorder_exact = reordered_tokens.tolist() == expected_reordered_tokens.tolist()
        reorder_final_state_diff = state_max_abs(
            mx,
            reordered_final_state,
            expected_reordered_final,
        )
        result.update(
            {
                "prefix_cache_unique_prompts": int(args.prefix_unique_prompts),
                "prefix_cache_hits": int(args.batch_size) - int(args.prefix_unique_prompts),
                "prefix_cache_hit_rate": (
                    int(args.batch_size) - int(args.prefix_unique_prompts)
                ) / int(args.batch_size),
                "prefix_cache_greedy_exact": cache_exact,
                "prefix_cache_prefill_logits_max_abs": cache_logits_diff,
                "prefix_cache_prefill_state_max_abs": cache_prefill_state_diff,
                "prefix_cache_final_state_max_abs": cache_final_state_diff,
                "prefix_cache_reorder_compact_state_max_abs": reorder_compact_state_diff,
                "prefix_cache_reorder_greedy_exact": reorder_exact,
                "prefix_cache_reorder_final_state_max_abs": reorder_final_state_diff,
            }
        )
        gates.append(
            cache_exact
            and cache_logits_diff <= 0.25
            and cache_prefill_state_diff <= 0.5
            and cache_final_state_diff <= 0.5
            and reorder_compact_state_diff == 0.0
            and reorder_exact
            and reorder_final_state_diff <= 0.5
        )

    if args.compare_speculative_mismatch:
        from rwkv7_hf.mlx_model import load_mlx_rwkv7_model
        from rwkv7_hf.mlx_speculative import speculative_decode_greedy_batch

        draft = load_mlx_rwkv7_model(
            args.draft_model,
            dtype="fp16",
            quantization="mm4",
            quant_min_params=int(args.draft_quant_min_params),
            quant_rkv_min_params=int(args.draft_quant_min_params),
            quant_backend="groupwise",
            quant_group_size=int(args.quant_group_size),
            quantize_embedding=True,
            wkv_backend="metal",
        )
        mismatch_found = False
        search_attempt = 0
        target_logits = target_state = draft_logits = draft_state = None
        # Prefer the mixed real prompt. If its first greedy token happens to
        # agree, deterministically search short valid-token prefixes until the
        # real small draft and target disagree on at least one B8 row.
        candidates = [mixed_ids if args.compare_prefix_cache else ids]
        for attempt in range(1, 5):
            candidates.append(
                mx.array(
                    [
                        [
                            1
                            + (
                                (17 + row * 997 + pos * 7919 + attempt * 104729)
                                % (optimized.vocab_size - 1)
                            )
                            for pos in range(32)
                        ]
                        for row in range(int(args.batch_size))
                    ],
                    dtype=mx.int32,
                )
            )
        for search_attempt, candidate_ids in enumerate(candidates):
            target_logits, target_state = optimized.prefill(candidate_ids)
            draft_logits, draft_state = draft.prefill(candidate_ids)
            target_choice = mx.argmax(target_logits[:, -1, :], axis=-1)
            draft_choice = mx.argmax(draft_logits[:, -1, :], axis=-1)
            mx.eval(target_choice, draft_choice)
            if target_choice.tolist() != draft_choice.tolist():
                mismatch_found = True
                break
        if not mismatch_found:
            raise RuntimeError("real target/draft mismatch search exhausted without a disagreement")
        greedy_tokens, _ = optimized.decode_greedy(
            target_logits,
            target_state.clone(),
            max_new_tokens=int(args.decode_tokens),
        )
        speculative = speculative_decode_greedy_batch(
            optimized,
            draft,
            target_logits,
            target_state.clone(),
            draft_logits,
            draft_state.clone(),
            max_new_tokens=int(args.decode_tokens),
            proposal_tokens=min(8, max(2, int(args.decode_tokens))),
        )
        mx.eval(greedy_tokens, *flat_state(speculative.target_state))
        speculative_exact = speculative.generated_ids == greedy_tokens.tolist()
        telemetry = speculative.telemetry()
        result.update(
            {
                "speculative_mismatch_search_attempt": int(search_attempt),
                "speculative_mismatch_found": bool(mismatch_found),
                "speculative_greedy_exact": bool(speculative_exact),
                "speculative_acceptance_rate": telemetry["acceptance_rate"],
                "speculative_target_replay_calls": telemetry["target_replay_calls"],
                "speculative_telemetry": telemetry,
            }
        )
        gates.append(
            mismatch_found
            and speculative_exact
            and float(telemetry["acceptance_rate"]) < 1.0
            and int(telemetry["target_replay_calls"]) > 0
        )

    telemetry = optimized.telemetry()
    result["quantized_embedding_footprint_ratio"] = telemetry.get("quantized_embedding_footprint_ratio")
    result["fused_scan_prep_post_counts"] = telemetry.get("fused_scan_prep_post_counts")
    result["fused_ffn_key_relu2_counts"] = telemetry.get("fused_ffn_key_relu2_counts")
    result["fused_sequence_mix_counts"] = telemetry.get("fused_sequence_mix_counts")
    result["fused_add_layer_norm_counts"] = telemetry.get("fused_add_layer_norm_counts")
    result["quantized_linear_footprint_ratio"] = (
        telemetry["quantized_linear_bytes"] / telemetry["quantized_dense_equivalent_bytes"]
    )
    result["generated_preview"] = opt_tokens[0, :16].tolist()
    result["status"] = "pass" if all(gates) else "fail"
    print(json.dumps(result, ensure_ascii=False))
    append_jsonl(args.results, result)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
