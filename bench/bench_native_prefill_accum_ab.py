#!/usr/bin/env python3
# coding=utf-8
"""Paired same-process A/B for scoped native-Prefill FP16 accumulation.

The generic native-Prefill benchmark also times an independent recurrent
oracle. That is useful for kernel development, but unnecessarily expensive
when screening many exact model/batch/prompt shapes. This benchmark loads one
checkpoint, captures each accumulation route in both forward and reverse
order, and compares prompt plus cache-handoff decode logits with the local
FP32-accumulation control.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("fp16", "bf16", "fp32"), default="fp16"
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument(
        "--prompt-tokens", type=int, nargs="+", default=[128, 512, 2048]
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Chunk long prompts through rwkv7_prefill_chunks; 0 disables chunking.",
    )
    parser.add_argument(
        "--orders", choices=("forward", "reverse", "both"), default="both"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--code-source", choices=("model", "repo"), default="repo")
    parser.add_argument("--results", default="")
    return parser


# ``--help`` should not pay the Torch/Transformers import cost. This also keeps
# the documented direct-script entrypoint responsive on cold Windows hosts.
if __name__ == "__main__" and any(
    arg in {"-h", "--help"} for arg in sys.argv[1:]
):
    build_parser().parse_args()


REPO_ROOT = Path(__file__).resolve().parents[1]
# Direct ``python bench/<script>.py`` execution puts ``bench/`` ahead of the
# repository root.  That can resolve ``bench`` to ``bench/bench.py`` instead
# of the package, even when PYTHONPATH already contains the root.  Reinsert the
# root at position zero so the benchmark is reproducible from its documented
# CLI entrypoint as well as through pytest/module imports.
try:
    sys.path.remove(str(REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

try:
    from bench.bench_native_prefill_scan import (
        DTYPES,
        build_ids,
        infer_model_size_label,
        prepare_model_dir,
    )
except ModuleNotFoundError:  # Direct ``python bench/...`` execution.
    from bench_native_prefill_scan import (
        DTYPES,
        build_ids,
        infer_model_size_label,
        prepare_model_dir,
    )


MODES = ("off", "global", "block")


def mode_flags(mode: str) -> tuple[str, str]:
    if mode == "off":
        return "0", "0"
    if mode == "global":
        return "1", "0"
    if mode == "block":
        return "0", "1"
    raise ValueError(f"unsupported accumulation mode: {mode!r}")


def route_effective_matches(mode: str, global_effective: bool, block_effective: bool) -> bool:
    expected_global, expected_block = mode_flags(mode)
    return global_effective == (expected_global == "1") and block_effective == (
        expected_block == "1"
    )


def sweep_orders(selection: str) -> tuple[tuple[str, ...], ...]:
    forward = MODES
    reverse = tuple(reversed(MODES))
    if selection == "forward":
        return (forward,)
    if selection == "reverse":
        return (reverse,)
    if selection == "both":
        return (forward, reverse)
    raise ValueError(f"unsupported order selection: {selection!r}")


def model_shape_spec(
    hidden_size: int,
    num_layers: int,
    batch_sizes: list[int],
    prompt_tokens: list[int],
) -> str:
    return " ".join(
        f"{int(hidden_size)}x{int(num_layers)}x{int(batch)}x{int(tokens)}"
        for batch in batch_sizes
        for tokens in prompt_tokens
    )


def _cosine_min(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.float(), right.float(), dim=-1).min())


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def _cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _prefill_call(model, ids: torch.Tensor, chunk_size: int):
    if chunk_size > 0 and int(ids.shape[1]) > chunk_size:
        return model.rwkv7_prefill_chunks(
            ids, chunk_size=chunk_size, logits_to_keep=1
        )
    return model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)


def _timed_call(
    model,
    ids: torch.Tensor,
    device: str,
    chunk_size: int,
) -> tuple[Any, float]:
    if device.startswith("cuda"):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = _prefill_call(model, ids, chunk_size)
        end.record()
        end.synchronize()
        return output, float(begin.elapsed_time(end))
    started = time.perf_counter()
    output = _prefill_call(model, ids, chunk_size)
    return output, (time.perf_counter() - started) * 1000.0


def _capture_mode(
    model,
    ids: torch.Tensor,
    *,
    mode: str,
    device: str,
    warmup: int,
    steps: int,
    chunk_size: int,
) -> dict[str, Any]:
    global_flag, block_flag = mode_flags(mode)
    previous = {
        "RWKV7_NATIVE_PREFILL_GLOBAL_FP16_ACCUM": os.environ.get(
            "RWKV7_NATIVE_PREFILL_GLOBAL_FP16_ACCUM"
        ),
        "RWKV7_NATIVE_PREFILL_BLOCK_FP16_ACCUM": os.environ.get(
            "RWKV7_NATIVE_PREFILL_BLOCK_FP16_ACCUM"
        ),
    }
    try:
        os.environ["RWKV7_NATIVE_PREFILL_GLOBAL_FP16_ACCUM"] = global_flag
        os.environ["RWKV7_NATIVE_PREFILL_BLOCK_FP16_ACCUM"] = block_flag
        model.rwkv7_clear_native_prefill_graph_cache()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range(warmup):
                _prefill_call(model, ids, chunk_size)
            _cuda_sync(device)
            times: list[float] = []
            output = None
            for _ in range(steps):
                output, elapsed = _timed_call(model, ids, device, chunk_size)
                times.append(elapsed)
            assert output is not None
            prompt_logits = output.logits[:, -1].detach().float().cpu().clone()
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            next_output = model.rwkv7_forward_token(
                next_token,
                past_key_values=output.past_key_values,
                return_dict=True,
            )
            decode_logits = next_output.logits[:, -1].detach().float().cpu().clone()
        peak_vram_mb = None
        if device.startswith("cuda"):
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        return {
            "prompt_logits": prompt_logits,
            "decode_logits": decode_logits,
            "global_effective": bool(
                getattr(
                    model,
                    "_rwkv7_native_prefill_global_fp16_accum_effective",
                    False,
                )
            ),
            "block_effective": bool(
                getattr(
                    model,
                    "_rwkv7_native_prefill_block_fp16_accum_effective",
                    False,
                )
            ),
            "prefill_ms_median": statistics.median(times),
            "peak_vram_mb": peak_vram_mb,
        }
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def append_row(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = build_parser().parse_args()

    effective_path, temporary_dir = prepare_model_dir(
        args.model,
        code_source=args.code_source,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(effective_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            effective_path,
            trust_remote_code=True,
            dtype=DTYPES[args.dtype],
            device_map=args.device if args.device.startswith("cuda") else None,
        ).eval()
        hidden_size = int(model.config.hidden_size)
        num_layers = int(model.config.num_hidden_layers)
        shapes = model_shape_spec(
            hidden_size,
            num_layers,
            args.batch_sizes,
            args.prompt_tokens,
        )
        os.environ["RWKV7_FAST_PREFILL"] = "1"
        os.environ["RWKV7_NATIVE_PREFILL_GRAPH"] = "1"
        os.environ["RWKV7_FAST_TOKEN_BACKEND"] = "native_graph"
        os.environ["RWKV7_NATIVE_PREFILL_GLOBAL_FP16_ACCUM_MODEL_SHAPES"] = shapes
        os.environ["RWKV7_NATIVE_PREFILL_BLOCK_FP16_ACCUM_MODEL_SHAPES"] = shapes

        passed_all = True
        for batch_size in args.batch_sizes:
            for prompt_tokens in args.prompt_tokens:
                ids = build_ids(tokenizer, batch_size, prompt_tokens, args.device)
                for order_index, order in enumerate(sweep_orders(args.orders)):
                    captures = {
                        mode: _capture_mode(
                            model,
                            ids,
                            mode=mode,
                            device=args.device,
                            warmup=args.warmup,
                            steps=args.steps,
                            chunk_size=args.chunk_size,
                        )
                        for mode in order
                    }
                    reference = captures["off"]
                    for mode in order:
                        capture = captures[mode]
                        prompt_cosine = _cosine_min(
                            reference["prompt_logits"], capture["prompt_logits"]
                        )
                        decode_cosine = _cosine_min(
                            reference["decode_logits"], capture["decode_logits"]
                        )
                        prompt_greedy = bool(
                            torch.equal(
                                reference["prompt_logits"].argmax(dim=-1),
                                capture["prompt_logits"].argmax(dim=-1),
                            )
                        )
                        decode_greedy = bool(
                            torch.equal(
                                reference["decode_logits"].argmax(dim=-1),
                                capture["decode_logits"].argmax(dim=-1),
                            )
                        )
                        route_ok = route_effective_matches(
                            mode,
                            capture["global_effective"],
                            capture["block_effective"],
                        )
                        passed = bool(
                            route_ok
                            and prompt_cosine >= args.min_cosine
                            and decode_cosine >= args.min_cosine
                            and prompt_greedy
                            and decode_greedy
                        )
                        passed_all = passed_all and passed
                        prefill_ms = float(capture["prefill_ms_median"])
                        baseline_ms = float(reference["prefill_ms_median"])
                        row = {
                            "axis": "native_prefill_accum_same_process_ab",
                            "status": "pass" if passed else "fail",
                            "device": (
                                torch.cuda.get_device_name(0)
                                if args.device.startswith("cuda")
                                else args.device
                            ),
                            "gpu_arch": (
                                f"sm_{torch.cuda.get_device_capability(0)[0]}"
                                f"{torch.cuda.get_device_capability(0)[1]}"
                                if args.device.startswith("cuda")
                                else None
                            ),
                            "dtype": args.dtype,
                            "model": args.model,
                            "model_size_label": infer_model_size_label(args.model),
                            "hidden_size": hidden_size,
                            "num_hidden_layers": num_layers,
                            "batch_size": batch_size,
                            "prompt_tokens": prompt_tokens,
                            "chunk_size": args.chunk_size,
                            "order_index": order_index,
                            "order": list(order),
                            "mode": mode,
                            "global_effective": capture["global_effective"],
                            "block_effective": capture["block_effective"],
                            "route_effective_match": route_ok,
                            "prefill_ms_median": round(prefill_ms, 5),
                            "prefill_tokps_total": round(
                                1000.0 * batch_size * prompt_tokens / prefill_ms,
                                1,
                            ),
                            "speedup_vs_off": round(baseline_ms / prefill_ms, 6),
                            "prompt_min_cosine": round(prompt_cosine, 8),
                            "prompt_max_abs_diff": round(
                                _max_abs(
                                    reference["prompt_logits"],
                                    capture["prompt_logits"],
                                ),
                                6,
                            ),
                            "prompt_greedy_match": prompt_greedy,
                            "decode_min_cosine": round(decode_cosine, 8),
                            "decode_max_abs_diff": round(
                                _max_abs(
                                    reference["decode_logits"],
                                    capture["decode_logits"],
                                ),
                                6,
                            ),
                            "decode_greedy_match": decode_greedy,
                            "peak_vram_mb": capture["peak_vram_mb"],
                            "warmup": args.warmup,
                            "steps": args.steps,
                            "min_cosine_gate": args.min_cosine,
                        }
                        print(json.dumps(row, ensure_ascii=False), flush=True)
                        append_row(args.results, row)
        return 0 if passed_all else 2
    finally:
        if temporary_dir is not None:
            temporary_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
