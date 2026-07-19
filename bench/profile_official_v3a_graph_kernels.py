#!/usr/bin/env python3
# coding=utf-8
"""Profile the pinned official v3a CUDA Graph with the Native profiler schema."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from scripts.compare_official_native_inference import verify_official_source


def event_device_time_us(event: Any) -> float:
    return float(
        getattr(event, "device_time_total", None)
        or getattr(event, "self_device_time_total", None)
        or 0.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", required=True)
    parser.add_argument("--official-model", required=True)
    parser.add_argument("--official-commit", required=True)
    parser.add_argument("--official-source-manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", default="")
    args = parser.parse_args()

    verification = verify_official_source(
        args.official_dir,
        expected_commit=args.official_commit,
        manifest_path=args.official_source_manifest,
    )
    os.environ.setdefault("RWKV_V7_ON", "1")
    sys.path.insert(0, args.official_dir)
    module = importlib.import_module("rwkv7_fast_v3a")
    module.MODEL_PATH = args.official_model
    module.WKV_MODE = "fp16"
    module.EMB_DEVICE = "gpu"
    module.RKV_MODE = "off"
    module.CMIX_SPARSE = "no-fc"
    module.LOWRANK_WEIGHT = "both"
    module.ORIG_LINEAR_GROUPS = module.parse_orig_linear_groups(
        "att_c2c,ffn_key,head"
    )
    os.chdir(args.official_dir)
    torch.set_grad_enabled(False)
    module.load_extensions(module.WKV_MODE)
    model = module.RWKV7()
    batch_size = args.batch_size
    tokens = torch.arange(batch_size, dtype=torch.long, device="cuda").view(batch_size, 1)
    state = model.zero_state(batch_size)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model.forward(tokens, state)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        model.forward(tokens, state)
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            graph.replay()
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
        "axis": "official_v3a_graph_kernel_profile",
        "device": torch.cuda.get_device_name(),
        "batch_size": batch_size,
        "steps": args.steps,
        "source_verification": verification,
        "top_events": rows[:100],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
