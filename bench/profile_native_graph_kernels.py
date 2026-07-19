#!/usr/bin/env python3
# coding=utf-8
"""Profile CUDA kernels launched by one Native HF decode graph."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM


PROMPT = "User: Profile recurrent decode.\n\nAssistant:" * 16


def event_device_time_us(event: Any) -> float:
    return float(
        getattr(event, "device_time_total", None)
        or getattr(event, "cuda_time_total", None)
        or getattr(event, "self_device_time_total", None)
        or getattr(event, "self_cuda_time_total", None)
        or 0.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    ids = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
    ids = ids[:, : args.prompt_tokens].repeat(args.batch_size, 1).cuda()
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.hf_dir,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()
    with torch.inference_mode():
        output = model(ids, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        token = output.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.warmup):
            output = model.rwkv7_forward_token(token, cache, copy_logits=False)
            cache = output.past_key_values
            token = output.logits[:, -1:].argmax(dim=-1)
        runner = cache._native_graph_bound_runner()
        if runner is None or runner.graph is None:
            raise RuntimeError("native graph runner is not active")
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.steps):
                runner.graph.replay()
            torch.cuda.synchronize()

    if args.trace:
        trace = Path(args.trace)
        trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace))

    aggregates: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "device_time_total_us": 0.0}
    )
    for event in prof.events():
        duration = event_device_time_us(event)
        if duration <= 0:
            continue
        item = aggregates[str(event.name)]
        item["count"] += 1
        item["device_time_total_us"] += duration
    rows = []
    for name, item in aggregates.items():
        count = int(item["count"])
        total = float(item["device_time_total_us"])
        rows.append(
            {
                "name": name,
                "count": count,
                "device_time_total_us": total,
                "device_time_avg_us": total / max(count, 1),
                "device_time_per_decode_us": total / max(args.steps, 1),
            }
        )
    rows.sort(key=lambda row: row["device_time_total_us"], reverse=True)
    report = {
        "axis": "native_graph_kernel_profile",
        "device": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "top_events": rows[:100],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
