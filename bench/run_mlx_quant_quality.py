#!/usr/bin/env python3
"""Measure Apple MLX W8/W4 quality against the same model in FP16.

The existing Apple speed gate compares RWKV with Qwen, while compiled-decode
validation compares two executions of the *same quantized weights*. Neither
measurement answers whether quantization itself preserved model quality. This
runner adds a deterministic teacher-forced NLL/perplexity gate, top-1 agreement
over every scored corpus token, and multi-prompt greedy telemetry.

``q4_k_m`` is an RWKV/MLX mixed W8/W4 profile inspired by llama.cpp's mixed
precision K-quants. It is not labelled equivalent to GGUF Q4_K_M unless a
future run includes the same checkpoint/corpus through both implementations.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AXIS = "mlx_quant_quality"

DEFAULT_CORPUS = """
The quick brown fox jumps over the lazy dog. RWKV recurrent models process
tokens with a compact state rather than a transformer KV cache. A production
inference server must measure latency, throughput, memory, correctness, cache
hit rate, and reliability under concurrent load.

In mathematics, the derivative of x squared is two x. The integral of one over
x is the natural logarithm of the absolute value of x plus a constant. A prime
number has exactly two positive integer divisors, one and itself.

Python code can define a function with def, process a list with a loop, handle
an exception with try and except, and return a result. Tests should check both
normal inputs and boundary conditions instead of only executing one example.

Beijing is the capital of China, while Shanghai is an important economic
center. Water freezes near zero degrees Celsius at standard atmospheric
pressure, and the Earth travels around the Sun once per year.

机器学习模型的量化需要同时考虑速度、内存占用和精度，不能只看单个短提示的结果。
生产级验收还需要覆盖不同批大小、长上下文、并发请求、缓存隔离以及持续运行稳定性。
""".strip()

DEFAULT_GREEDY_PROMPTS = (
    "User: Explain recurrent state caching in one paragraph.\nAssistant:",
    "User: What is the derivative of x squared?\nAssistant:",
    "User: Write a short Python function that adds two integers.\nAssistant:",
    "用户：用一句话说明模型量化为什么需要精度测试。\n助手：",
)


def append_jsonl(path: str | Path | None, rows: Iterable[dict[str, Any]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def token_agreement(left: list[int], right: list[int]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(int(a) == int(b) for a, b in zip(left, right, strict=True)) / len(left)


def gate_thresholds(bits: int, args: argparse.Namespace) -> dict[str, float]:
    if int(bits) == 8:
        return {
            "max_nll_delta": float(args.max_nll_delta_w8),
            "max_perplexity_ratio": float(args.max_perplexity_ratio_w8),
            "min_teacher_top1_agreement": float(args.min_teacher_top1_agreement_w8),
        }
    return {
        "max_nll_delta": float(args.max_nll_delta_w4),
        "max_perplexity_ratio": float(args.max_perplexity_ratio_w4),
        "min_teacher_top1_agreement": float(args.min_teacher_top1_agreement_w4),
    }


def quality_gate(
    *,
    nll_delta: float,
    perplexity_ratio: float,
    teacher_top1_agreement: float,
    thresholds: dict[str, float],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not math.isfinite(nll_delta) or nll_delta > thresholds["max_nll_delta"]:
        reasons.append("nll_delta")
    if not math.isfinite(perplexity_ratio) or perplexity_ratio > thresholds["max_perplexity_ratio"]:
        reasons.append("perplexity_ratio")
    if (
        not math.isfinite(teacher_top1_agreement)
        or teacher_top1_agreement < thresholds["min_teacher_top1_agreement"]
    ):
        reasons.append("teacher_top1_agreement")
    return ("pass" if not reasons else "fail"), reasons


def load_corpus(args: argparse.Namespace) -> str:
    if args.corpus_file:
        return Path(args.corpus_file).read_text(encoding="utf-8")
    return DEFAULT_CORPUS


def load_prompts(args: argparse.Namespace) -> list[str]:
    if not args.greedy_prompts_file:
        return list(DEFAULT_GREEDY_PROMPTS)
    raw = json.loads(Path(args.greedy_prompts_file).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
        raise ValueError("--greedy-prompts-file must contain a non-empty JSON string list")
    return list(raw)


def evaluate_model(
    model: Any,
    tokenizer: Any,
    corpus_ids: list[int],
    prompts: list[str],
    *,
    chunk_tokens: int,
    greedy_tokens: int,
) -> dict[str, Any]:
    import mlx.core as mx

    state = None
    nll_sum = 0.0
    scored_tokens = 0
    top1: list[int] = []
    for start in range(0, len(corpus_ids) - 1, int(chunk_tokens)):
        inputs = corpus_ids[start : min(start + int(chunk_tokens), len(corpus_ids) - 1)]
        targets_list = corpus_ids[start + 1 : start + 1 + len(inputs)]
        if not inputs:
            continue
        logits, state = model.forward([inputs], state=state, collect_all=True)
        targets = mx.array([targets_list], dtype=mx.int32)
        target_logits = mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        token_nll = mx.logsumexp(logits.astype(mx.float32), axis=-1) - target_logits.astype(mx.float32)
        predicted = mx.argmax(logits, axis=-1).astype(mx.int32)
        mx.eval(token_nll, predicted)
        nll_sum += float(mx.sum(token_nll))
        scored_tokens += len(inputs)
        top1.extend(int(value) for value in predicted.tolist()[0])

    if scored_tokens <= 0:
        raise ValueError("quality corpus must tokenize to at least two tokens")
    mean_nll = nll_sum / scored_tokens
    greedy: list[list[int]] = []
    for prompt in prompts:
        prompt_ids = [
            int(value)
            for value in tokenizer(prompt, add_special_tokens=False).input_ids
        ]
        generated, _ = model.generate_greedy([prompt_ids], max_new_tokens=int(greedy_tokens))
        greedy.append([int(value) for value in generated.tolist()[0]])

    dense_bytes = sum(int(value.nbytes) for value in model.arrays.values())
    quant_bytes = int(getattr(model, "quantized_linear_bytes", 0))
    telemetry = model.telemetry()
    return {
        "mean_nll": float(mean_nll),
        "perplexity": float(math.exp(mean_nll)),
        "scored_tokens": int(scored_tokens),
        "teacher_top1": top1,
        "greedy": greedy,
        "weight_storage_bytes": int(dense_bytes + quant_bytes),
        "quantized_linear_count": int(telemetry.get("quantized_linear_count", 0)),
        "quantized_linear_bits_histogram": telemetry.get("quantized_linear_bits_histogram", {}),
        "quantized_footprint_ratio": telemetry.get("quantized_footprint_ratio"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="Comma-separated converted RWKV-7 HF directories")
    parser.add_argument("--quantizations", default="mm8,mm4")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16", "keep"])
    parser.add_argument("--quant-backend", default="groupwise", choices=["groupwise", "affine", "metal", "auto", "reference"])
    parser.add_argument("--quant-min-params", type=int, default=1_000_000)
    parser.add_argument("--quant-rkv-min-params", type=int, default=-1)
    parser.add_argument("--quant-group-size", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument("--w4-profile", default="q4_k_m", choices=["uniform", "q4_k_m"])
    parser.add_argument("--wkv-backend", default="metal", choices=["reference", "metal", "auto"])
    parser.add_argument("--corpus-file", default="")
    parser.add_argument("--max-corpus-tokens", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--greedy-prompts-file", default="")
    parser.add_argument("--greedy-tokens", type=int, default=32)
    parser.add_argument("--max-nll-delta-w8", type=float, default=0.02)
    parser.add_argument("--max-perplexity-ratio-w8", type=float, default=1.02)
    parser.add_argument("--min-teacher-top1-agreement-w8", type=float, default=0.95)
    parser.add_argument("--max-nll-delta-w4", type=float, default=0.08)
    parser.add_argument("--max-perplexity-ratio-w4", type=float, default=1.09)
    parser.add_argument("--min-teacher-top1-agreement-w4", type=float, default=0.80)
    parser.add_argument("--results", default="")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = parse_csv(args.models)
    quantizations = parse_csv(args.quantizations)
    if not models:
        raise ValueError("--models must not be empty")
    if not quantizations or any(value not in {"mm8", "mm4"} for value in quantizations):
        raise ValueError("--quantizations must contain mm8 and/or mm4")
    if min(args.quant_min_params, args.max_corpus_tokens, args.chunk_tokens, args.greedy_tokens) <= 0:
        raise ValueError("quant threshold, corpus/chunk tokens, and greedy tokens must be positive")

    env = {
        "axis": AXIS + "_env",
        "status": "plan" if args.dry_run else "info",
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "models": models,
        "quantizations": quantizations,
        "quant_backend": args.quant_backend,
        "quant_min_params": int(args.quant_min_params),
        "quant_rkv_min_params": int(args.quant_rkv_min_params),
        "quant_group_size": int(args.quant_group_size),
        "w4_profile": args.w4_profile,
        "max_corpus_tokens": int(args.max_corpus_tokens),
        "chunk_tokens": int(args.chunk_tokens),
        "greedy_tokens": int(args.greedy_tokens),
    }
    print(json.dumps(env, ensure_ascii=False))
    append_jsonl(args.results, [env])
    if args.dry_run:
        return 0

    import mlx.core as mx
    from transformers import AutoTokenizer

    from rwkv7_hf.mlx_model import load_mlx_rwkv7_model

    corpus = load_corpus(args)
    prompts = load_prompts(args)
    output_rows: list[dict[str, Any]] = []
    any_failed = False
    for model_path in models:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        corpus_ids = [int(value) for value in tokenizer(corpus, add_special_tokens=False).input_ids]
        corpus_ids = corpus_ids[: int(args.max_corpus_tokens) + 1]

        baseline_model = load_mlx_rwkv7_model(
            model_path,
            dtype=args.dtype,
            wkv_backend=args.wkv_backend,
        )
        baseline_model.prefill_backend = "auto"
        baseline = evaluate_model(
            baseline_model,
            tokenizer,
            corpus_ids,
            prompts,
            chunk_tokens=args.chunk_tokens,
            greedy_tokens=args.greedy_tokens,
        )
        baseline_row = {
            "axis": AXIS,
            "status": "pass",
            "model": Path(model_path).name,
            "model_path": model_path,
            "dtype": args.dtype,
            "quantization": "none",
            "quant_profile": "none",
            "mean_nll": round(baseline["mean_nll"], 8),
            "perplexity": round(baseline["perplexity"], 8),
            "scored_tokens": baseline["scored_tokens"],
            "greedy_prompts": len(prompts),
            "greedy_tokens_per_prompt": int(args.greedy_tokens),
            "weight_storage_bytes": baseline["weight_storage_bytes"],
        }
        print(json.dumps(baseline_row, ensure_ascii=False))
        output_rows.append(baseline_row)
        baseline_top1 = list(baseline["teacher_top1"])
        baseline_greedy = list(baseline["greedy"])
        baseline_storage = int(baseline["weight_storage_bytes"])
        del baseline_model
        gc.collect()
        mx.clear_cache()

        for quantization in quantizations:
            bits = 8 if quantization == "mm8" else 4
            profile = args.w4_profile if bits == 4 else "uniform"
            model = load_mlx_rwkv7_model(
                model_path,
                dtype=args.dtype,
                quantization=quantization,
                quant_min_params=args.quant_min_params,
                quant_rkv_min_params=(None if args.quant_rkv_min_params < 0 else args.quant_rkv_min_params),
                quant_backend=args.quant_backend,
                quant_profile=profile,
                quant_group_size=args.quant_group_size,
                wkv_backend=args.wkv_backend,
            )
            model.prefill_backend = "auto"
            measured = evaluate_model(
                model,
                tokenizer,
                corpus_ids,
                prompts,
                chunk_tokens=args.chunk_tokens,
                greedy_tokens=args.greedy_tokens,
            )
            teacher_agreement = token_agreement(baseline_top1, measured["teacher_top1"])
            prompt_exact = 0
            greedy_agreements: list[float] = []
            greedy_prefix_ratios: list[float] = []
            for reference, candidate in zip(baseline_greedy, measured["greedy"], strict=True):
                prompt_exact += int(reference == candidate)
                greedy_agreements.append(token_agreement(reference, candidate))
                greedy_prefix_ratios.append(
                    common_prefix_length(reference, candidate) / max(len(reference), 1)
                )
            nll_delta = float(measured["mean_nll"] - baseline["mean_nll"])
            perplexity_ratio = float(measured["perplexity"] / baseline["perplexity"])
            thresholds = gate_thresholds(bits, args)
            status, failed_gates = quality_gate(
                nll_delta=nll_delta,
                perplexity_ratio=perplexity_ratio,
                teacher_top1_agreement=teacher_agreement,
                thresholds=thresholds,
            )
            any_failed = any_failed or status != "pass"
            row = {
                "axis": AXIS,
                "status": status,
                "model": Path(model_path).name,
                "model_path": model_path,
                "dtype": args.dtype,
                "quantization": quantization,
                "quant_bits": bits,
                "quant_backend": args.quant_backend,
                "quant_profile": profile,
                "quant_group_size": int(args.quant_group_size),
                "quant_min_params": int(args.quant_min_params),
                "mean_nll": round(measured["mean_nll"], 8),
                "fp16_mean_nll": round(baseline["mean_nll"], 8),
                "nll_delta": round(nll_delta, 8),
                "perplexity": round(measured["perplexity"], 8),
                "fp16_perplexity": round(baseline["perplexity"], 8),
                "perplexity_ratio": round(perplexity_ratio, 8),
                "scored_tokens": measured["scored_tokens"],
                "teacher_top1_agreement": round(teacher_agreement, 8),
                "greedy_exact_prompts": int(prompt_exact),
                "greedy_prompt_count": len(prompts),
                "greedy_mean_token_agreement": round(sum(greedy_agreements) / len(greedy_agreements), 8),
                "greedy_min_prefix_ratio": round(min(greedy_prefix_ratios), 8),
                "weight_storage_bytes": measured["weight_storage_bytes"],
                "fp16_weight_storage_bytes": baseline_storage,
                "weight_storage_ratio": round(measured["weight_storage_bytes"] / baseline_storage, 8),
                "quantized_linear_count": measured["quantized_linear_count"],
                "quantized_linear_bits_histogram": measured["quantized_linear_bits_histogram"],
                "quantized_footprint_ratio": measured["quantized_footprint_ratio"],
                "thresholds": thresholds,
                "failed_gates": failed_gates,
            }
            print(json.dumps(row, ensure_ascii=False))
            output_rows.append(row)
            del model
            gc.collect()
            mx.clear_cache()

    quant_rows = [row for row in output_rows if row.get("quantization") in {"mm8", "mm4"}]
    summary = {
        "axis": AXIS + "_summary",
        "status": "fail" if any_failed or not quant_rows else "pass",
        "models": len(models),
        "quant_rows": len(quant_rows),
        "pass_rows": sum(row.get("status") == "pass" for row in quant_rows),
        "fail_rows": sum(row.get("status") == "fail" for row in quant_rows),
        "note": "q4_k_m is an RWKV/MLX mixed W8/W4 proxy; GGUF Q4_K_M equivalence requires a same-checkpoint external baseline",
    }
    print(json.dumps(summary, ensure_ascii=False))
    append_jsonl(args.results, [*output_rows, summary])
    return 1 if args.fail_on_gate and summary["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
