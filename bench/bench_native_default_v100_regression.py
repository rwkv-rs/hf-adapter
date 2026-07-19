#!/usr/bin/env python3
"""Fail-closed Native-default prefill/cached-decode regression probe.

The existing V100 production artifacts were recorded through the historical HF
wrapper.  This probe loads ``NativeRWKV7ForCausalLM`` directly so a default
backend migration cannot accidentally reuse the wrapper while claiming Native
coverage.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM


SEED = "The quick brown fox jumps over the lazy dog. " * 512


@contextmanager
def backend(name: str):
    previous = os.environ.get("RWKV7_NATIVE_MODEL_BACKEND")
    os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RWKV7_NATIVE_MODEL_BACKEND", None)
        else:
            os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = previous


def encode(tokenizer, *, batch_size: int, tokens: int, device: str) -> torch.Tensor:
    ids = tokenizer(SEED, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] < tokens:
        raise RuntimeError(f"seed produced {ids.shape[1]} tokens, need {tokens}")
    return ids[:, :tokens].repeat(batch_size, 1).to(device)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    values = F.cosine_similarity(left.float(), right.float(), dim=-1)
    return float(values.min().item())


def timed_prefill(model, ids: torch.Tensor, *, warmup: int, repeats: int) -> tuple[float, str]:
    with backend("native_graph"), torch.inference_mode():
        for _ in range(warmup):
            output = model(ids, use_cache=True, logits_to_keep=1)
            del output
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            output = model(ids, use_cache=True, logits_to_keep=1)
            end.record()
            end.synchronize()
            elapsed.append(float(begin.elapsed_time(end)))
            del output
        effective = model.rwkv7_native_model_last_prefill_backend()
    return float(statistics.median(elapsed)), str(effective)


def correctness(model, ids: torch.Tensor) -> dict[str, object]:
    with torch.inference_mode(), backend("eager"):
        reference = model(ids, use_cache=True, logits_to_keep=1)
    with torch.inference_mode(), backend("native_graph"):
        candidate = model(ids, use_cache=True, logits_to_keep=1)
    reference_logits = reference.logits[:, -1]
    candidate_logits = candidate.logits[:, -1]
    token = reference_logits.argmax(dim=-1)
    with torch.inference_mode(), backend("native_graph"):
        reference_next, _ = model.rwkv7_forward_token(
            token,
            past_key_values=reference.past_key_values,
            return_dict=False,
            copy_logits=True,
        )
        candidate_next, _ = model.rwkv7_forward_token(
            token,
            past_key_values=candidate.past_key_values,
            return_dict=False,
            copy_logits=True,
        )
    reference_next = reference_next[:, -1]
    candidate_next = candidate_next[:, -1]
    return {
        "prompt_min_cosine": cosine(reference_logits, candidate_logits),
        "prompt_max_abs": float((reference_logits.float() - candidate_logits.float()).abs().max().item()),
        "prompt_top1_equal": bool(
            torch.equal(reference_logits.argmax(dim=-1), candidate_logits.argmax(dim=-1))
        ),
        "continuation_min_cosine": cosine(reference_next, candidate_next),
        "continuation_max_abs": float(
            (reference_next.float() - candidate_next.float()).abs().max().item()
        ),
        "continuation_top1_equal": bool(
            torch.equal(reference_next.argmax(dim=-1), candidate_next.argmax(dim=-1))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--correctness-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.hf_dir,
        torch_dtype=torch.float16,
        device_map=args.device,
    ).eval()

    rows = []
    passed = True
    for batch_size in args.batch_sizes:
        correctness_ids = encode(
            tokenizer,
            batch_size=batch_size,
            tokens=min(args.correctness_tokens, args.prompt_tokens),
            device=args.device,
        )
        metrics = correctness(model, correctness_ids)
        timed_ids = encode(
            tokenizer,
            batch_size=batch_size,
            tokens=args.prompt_tokens,
            device=args.device,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        median_ms, effective = timed_prefill(
            model,
            timed_ids,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        graph_stats_fn = getattr(
            model, "rwkv7_native_prefill_graph_cache_stats", None
        )
        graph_stats = graph_stats_fn() if callable(graph_stats_fn) else None
        row_passed = bool(
            effective in {"native_prefill", "native_prefill_graph"}
            and metrics["prompt_top1_equal"]
            and metrics["continuation_top1_equal"]
            and metrics["prompt_min_cosine"] >= args.min_cosine
            and metrics["continuation_min_cosine"] >= args.min_cosine
        )
        row = {
            "axis": "native_default_v100_regression",
            "status": "pass" if row_passed else "fail",
            "model": args.model_label,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "batch_size": batch_size,
            "prompt_tokens": args.prompt_tokens,
            "correctness_tokens": min(args.correctness_tokens, args.prompt_tokens),
            "prefill_backend": effective,
            "prefill_graph_cache_stats": graph_stats,
            "prefill_ms": round(median_ms, 4),
            "prefill_tokps": round(1000.0 * batch_size * args.prompt_tokens / median_ms, 2),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
            **metrics,
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
        passed = passed and row_passed

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
