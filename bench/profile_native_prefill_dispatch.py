#!/usr/bin/env python3
# coding=utf-8
"""Measure graph, runner, and public-HF dispatch overhead for Native prefill."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


def measure(call, repeats: int) -> dict:
    cuda_ms = []
    cpu_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        _result = call()
        end.record()
        end.synchronize()
        cpu_ms.append((time.perf_counter() - wall) * 1000.0)
        cuda_ms.append(float(start.elapsed_time(end)))
        del _result
    return {
        "cuda_ms": cuda_ms,
        "cuda_median_ms": statistics.median(cuda_ms),
        "cpu_ms": cpu_ms,
        "cpu_median_ms": statistics.median(cpu_ms),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.hf_dir,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()
    ids = torch.arange(
        args.batch_size * args.prompt_tokens,
        device="cuda",
        dtype=torch.long,
    ).view(args.batch_size, args.prompt_tokens)
    ids = (ids * 1103515245 + 12345) % int(model.config.vocab_size)

    with torch.inference_mode():
        _warm = model(ids, use_cache=True, logits_to_keep=1)
        runner = model._rwkv7_native_prefill_graph_hot_runner
        del _warm
        torch.cuda.synchronize()

        rows = {
            "graph_replay": measure(runner.graph.replay, args.repeats),
            "runner_replay": measure(
                lambda: runner.replay(ids, seen_tokens=args.prompt_tokens),
                args.repeats,
            ),
            "hf_forward": measure(
                lambda: model(ids, use_cache=True, logits_to_keep=1),
                args.repeats,
            ),
        }

    result = {
        "axis": "native_prefill_dispatch_profile",
        "device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "backend": model.rwkv7_native_model_last_prefill_backend(),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
