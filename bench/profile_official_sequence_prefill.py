#!/usr/bin/env python3
# coding=utf-8
"""CUDA-op profile for the pinned official RWKV v3a sequence path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from compare_official_native_inference import load_official  # noqa: E402
from compare_official_native_prefill import prompt_ids  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--official-dir", required=True)
    ap.add_argument("--official-model", required=True)
    ap.add_argument("--official-source-manifest", required=True)
    ap.add_argument("--official-module", default="rwkv7_fast_v3a")
    ap.add_argument(
        "--official-commit",
        default="cc57df475465c6cacd42ecd4f2f05a588ee5473b",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trace")
    args = ap.parse_args()

    ids = prompt_ids(args).to(args.device)
    model, revision, verification = load_official(args)

    def run_prefill():
        state = model.zero_state(args.batch_size)
        return model.forward(ids, state)

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_prefill()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            output = run_prefill()
        torch.cuda.synchronize()

    rows = []
    for event in sorted(
        prof.key_averages(),
        key=lambda item: float(item.self_device_time_total),
        reverse=True,
    )[: args.top]:
        rows.append(
            {
                "name": event.key,
                "self_cuda_ms": float(event.self_device_time_total) / 1000.0,
                "cuda_total_ms": float(event.device_time_total) / 1000.0,
                "cpu_total_ms": float(event.cpu_time_total) / 1000.0,
                "calls": int(event.count),
            }
        )
    result = {
        "axis": "official_sequence_prefill_profile",
        "device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "official_commit": revision,
        "source_verification": verification,
        "logits_finite": bool(torch.isfinite(output).all()),
        "top_ops": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.trace:
        prof.export_chrome_trace(args.trace)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
