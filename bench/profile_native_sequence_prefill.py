#!/usr/bin/env python3
# coding=utf-8
"""CUDA-op profile for the opt-in Native FP16-state sequence prefill path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trace")
    args = ap.parse_args()

    from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.hf_dir,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()
    vocab = int(model.config.vocab_size)
    ids = torch.arange(
        args.batch_size * args.prompt_tokens,
        device="cuda",
        dtype=torch.long,
    ).view(args.batch_size, args.prompt_tokens)
    ids = (ids * 1103515245 + 12345) % vocab

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(ids, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            output = model(ids, use_cache=True, logits_to_keep=1)
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
        "axis": "native_sequence_prefill_profile",
        "device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "prefill_backend": model.rwkv7_native_model_last_prefill_backend(),
        "fp16_recurrent_effective": bool(
            getattr(model, "_rwkv7_native_prefill_fp16_recurrent_effective", False)
        ),
        "stacked_rkv_effective": bool(
            getattr(model, "_rwkv7_native_prefill_stacked_rkv_effective", False)
        ),
        "sequence_ffn_effective": bool(
            getattr(model, "_rwkv7_native_prefill_sequence_ffn_effective", False)
        ),
        "logits_finite": bool(torch.isfinite(output.logits).all()),
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
