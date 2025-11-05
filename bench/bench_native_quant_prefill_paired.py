#!/usr/bin/env python3
"""Paired-process V100 prefill gate for repository-native W8/W4 heads.

Dense and quantized CUDA graphs are captured and measured in the same process,
on the same model instance and token batch. This removes cross-process clock and
allocator variance from the small (<2%) head-only prefill delta.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_native_prefill_scan import (
    build_ids,
    env_override,
    measured_ms,
    median,
    model_payload_mb,
    prepare_model_dir,
)

_GRAPH_CACHE_ATTRS = (
    "_rwkv7_native_jit_pack_cache",
    "_rwkv7_native_graph_pack_cache",
    "_rwkv7_native_graph_runner_cache",
    "_rwkv7_native_prefill_graph_runner_cache",
    "_rwkv7_native_prefill_graph_hot_runner",
)


def clear_graph_caches(model) -> None:
    for name in _GRAPH_CACHE_ATTRS:
        if hasattr(model, name):
            delattr(model, name)


def graph_state(model):
    return {
        name: getattr(model, name)
        for name in _GRAPH_CACHE_ATTRS
        if hasattr(model, name)
    }


def restore_graph_state(model, state) -> None:
    clear_graph_caches(model)
    for name, value in state.items():
        setattr(model, name, value)


def warm_capture(model, ids, *, warmup: int):
    def call():
        return model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)

    with torch.inference_mode(), env_override(RWKV7_FAST_PREFILL="1"):
        output = call()
        for _ in range(warmup):
            output = call()
    return graph_state(model), output.logits[:, -1].detach()


def measure_pair(
    model, ids, dense_head, quant_head, dense_state, quant_state, *, steps: int
):
    def call():
        return model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)

    dense_times, quant_times = [], []
    with torch.inference_mode(), env_override(RWKV7_FAST_PREFILL="1"):
        for step in range(steps):
            # Alternate order to cancel clock/thermal drift while retaining
            # independently captured graph-runner caches for both heads.
            order = (
                (dense_head, dense_state, dense_times),
                (quant_head, quant_state, quant_times),
            )
            if step & 1:
                order = tuple(reversed(order))
            for head, state, times in order:
                model.lm_head = head
                restore_graph_state(model, state)
                times.append(measured_ms(call, "cuda", "cuda-event"))
    return median(dense_times), median(quant_times)


def append(path: str, row: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-size-label", required=True)
    ap.add_argument("--quantization", choices=["a8w8", "mm4"], required=True)
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=31)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    effective, tmp = prepare_model_dir(args.model, code_source="repo")
    try:
        tok = AutoTokenizer.from_pretrained(effective, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            effective,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        ).eval()
        dense_head = model.lm_head
        dense_payload = model_payload_mb(model)
        if args.quantization == "a8w8":
            from rwkv7_hf.native_quant_a8w8 import A8W8Linear

            quant_head = A8W8Linear(dense_head)
        else:
            from rwkv7_hf.native_quant_mm4 import MM4Linear

            quant_head = MM4Linear(dense_head, fused=True)
        model.lm_head = quant_head
        quant_payload = model_payload_mb(model)
        model.lm_head = dense_head

        for batch in [int(x) for x in args.batch_sizes.split(",") if x.strip()]:
            ids = build_ids(tok, batch, args.prompt_tokens, "cuda")
            model.lm_head = dense_head
            clear_graph_caches(model)
            dense_state, dense_logits = warm_capture(model, ids, warmup=args.warmup)
            model.lm_head = quant_head
            clear_graph_caches(model)
            quant_state, quant_logits = warm_capture(model, ids, warmup=args.warmup)
            dense_ms, quant_ms = measure_pair(
                model,
                ids,
                dense_head,
                quant_head,
                dense_state,
                quant_state,
                steps=args.steps,
            )
            cosine = float(
                F.cosine_similarity(
                    dense_logits.float(), quant_logits.float(), dim=-1
                ).min()
            )
            greedy = bool(torch.equal(dense_logits.argmax(-1), quant_logits.argmax(-1)))
            row = {
                "axis": "native_quant_prefill_paired",
                "status": "pass" if greedy else "fail",
                "device": torch.cuda.get_device_name(),
                "model_size_label": args.model_size_label,
                "quantization": args.quantization,
                "batch_size": batch,
                "prompt_tokens": args.prompt_tokens,
                "dense_prefill_ms": round(dense_ms, 4),
                "quant_prefill_ms": round(quant_ms, 4),
                "quant_speed_ratio_vs_fp16": round(dense_ms / quant_ms, 4),
                "dense_payload_mb": dense_payload,
                "quant_payload_mb": quant_payload,
                "payload_ratio_vs_fp16": round(quant_payload / dense_payload, 4),
                "logits_min_cosine": round(cosine, 8),
                "same_next_token_as_fp16": greedy,
                "timing": "paired_same_process_cuda_event_median",
                "warmup": args.warmup,
                "steps": args.steps,
            }
            print(json.dumps(row), flush=True)
            append(args.results, row)
    finally:
        if tmp is not None:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
