#!/usr/bin/env python3
"""Same-process correctness/performance A/B for exact self-chunk Prefill routes."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("fp16", "bf16", "fp32"), default="fp16"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--stacked-rkv", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--code-source", choices=("model", "repo"), default="repo")
    parser.add_argument("--results", default="")
    return parser


if __name__ == "__main__" and any(
    arg in {"-h", "--help"} for arg in sys.argv[1:]
):
    build_parser().parse_args()


REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

try:
    from bench.bench_native_prefill_scan import (  # noqa: E402
        DTYPES,
        build_ids,
        prepare_model_dir,
    )
except ModuleNotFoundError:  # Direct ``python bench/...`` execution.
    from bench_native_prefill_scan import (  # type: ignore[no-redef]  # noqa: E402
        DTYPES,
        build_ids,
        prepare_model_dir,
    )


ROUTE_ENV = (
    "RWKV7_NATIVE_PREFILL_SELF_CHUNK",
    "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MIN_TOKENS",
    "RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE",
    "RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BV",
    "RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BC",
    "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MODEL_SHAPES",
    "RWKV7_NATIVE_PREFILL_STACKED_RKV",
    "RWKV7_NATIVE_PREFILL_STACKED_RKV_MIN_ROWS",
    "RWKV7_NATIVE_PREFILL_STACKED_RKV_MAX_ROWS",
    "RWKV7_NATIVE_PREFILL_STACKED_RKV_MODEL_SHAPES",
)


def route_environment(
    *,
    candidate: bool,
    chunk_size: int,
    stacked_rkv: bool,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    prompt_tokens: int,
) -> dict[str, str]:
    shape = f"{hidden_size}x{num_layers}x{batch_size}x{prompt_tokens}"
    enabled = bool(candidate)
    return {
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK": "1" if enabled else "0",
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MIN_TOKENS": str(prompt_tokens),
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE": str(chunk_size),
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BV": str(chunk_size),
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BC": str(chunk_size),
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MODEL_SHAPES": shape,
        "RWKV7_NATIVE_PREFILL_STACKED_RKV": (
            "1" if enabled and stacked_rkv else "0"
        ),
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_MIN_ROWS": str(batch_size),
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_MAX_ROWS": str(batch_size),
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_MODEL_SHAPES": shape,
    }


def _capture(
    model,
    ids: torch.Tensor,
    *,
    candidate: bool,
    chunk_size: int,
    stacked_rkv: bool,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    config = model.config
    env = route_environment(
        candidate=candidate,
        chunk_size=chunk_size,
        stacked_rkv=stacked_rkv,
        hidden_size=int(config.hidden_size),
        num_layers=int(config.num_hidden_layers),
        batch_size=int(ids.shape[0]),
        prompt_tokens=int(ids.shape[1]),
    )
    previous = {name: os.environ.get(name) for name in ROUTE_ENV}
    try:
        os.environ.update(env)
        model.rwkv7_clear_native_prefill_graph_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range(warmup):
                model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)
            torch.cuda.synchronize()
            timings: list[float] = []
            output = None
            for _ in range(steps):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                output = model.rwkv7_prefill_native(
                    ids, logits_to_keep=1, return_dict=True
                )
                end.record()
                end.synchronize()
                timings.append(float(begin.elapsed_time(end)))
            assert output is not None
            prompt_logits = output.logits[:, -1].detach().float().cpu().clone()
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            decode = model.rwkv7_forward_token(
                next_token,
                past_key_values=output.past_key_values,
                return_dict=True,
            )
            decode_logits = decode.logits[:, -1].detach().float().cpu().clone()
        return {
            "prompt_logits": prompt_logits,
            "decode_logits": decode_logits,
            "prefill_ms": statistics.median(timings),
            "peak_vram_mb": round(
                torch.cuda.max_memory_allocated() / 1024 / 1024, 1
            ),
            "self_chunk_effective": bool(
                getattr(model, "_rwkv7_native_prefill_self_chunk_effective", False)
            ),
            "stacked_rkv_effective": bool(
                getattr(model, "_rwkv7_native_prefill_stacked_rkv_effective", False)
            ),
            "block_fp16_accum_effective": bool(
                getattr(
                    model,
                    "_rwkv7_native_prefill_block_fp16_accum_effective",
                    False,
                )
            ),
        }
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.float(), right.float(), dim=-1).min())


def _append(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = build_parser().parse_args()

    effective_path, temporary_dir = prepare_model_dir(
        args.model, code_source=args.code_source
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            effective_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            effective_path,
            trust_remote_code=True,
            dtype=DTYPES[args.dtype],
            device_map=args.device,
        ).eval()
        ids = build_ids(
            tokenizer, args.batch_size, args.prompt_tokens, args.device
        )
        shape = (
            f"{int(model.config.hidden_size)}x{int(model.config.num_hidden_layers)}"
            f"x{args.batch_size}x{args.prompt_tokens}"
        )
        os.environ["RWKV7_FAST_PREFILL"] = "1"
        os.environ["RWKV7_NATIVE_PREFILL_GRAPH"] = "1"
        os.environ["RWKV7_FAST_TOKEN_BACKEND"] = "native_graph"
        os.environ["RWKV7_NATIVE_PREFILL_GLOBAL_FP16_ACCUM"] = "0"
        os.environ["RWKV7_NATIVE_PREFILL_BLOCK_FP16_ACCUM"] = "1"
        os.environ["RWKV7_NATIVE_PREFILL_BLOCK_FP16_ACCUM_MODEL_SHAPES"] = shape

        passed_all = True
        for order_index, order in enumerate(
            ((False, True), (True, False)), start=1
        ):
            captures = {
                candidate: _capture(
                    model,
                    ids,
                    candidate=candidate,
                    chunk_size=args.chunk_size,
                    stacked_rkv=args.stacked_rkv,
                    warmup=args.warmup,
                    steps=args.steps,
                )
                for candidate in order
            }
            control = captures[False]
            candidate = captures[True]
            prompt_cosine = _cosine(
                control["prompt_logits"], candidate["prompt_logits"]
            )
            decode_cosine = _cosine(
                control["decode_logits"], candidate["decode_logits"]
            )
            prompt_greedy = bool(
                torch.equal(
                    control["prompt_logits"].argmax(dim=-1),
                    candidate["prompt_logits"].argmax(dim=-1),
                )
            )
            decode_greedy = bool(
                torch.equal(
                    control["decode_logits"].argmax(dim=-1),
                    candidate["decode_logits"].argmax(dim=-1),
                )
            )
            route_ok = bool(
                not control["self_chunk_effective"]
                and candidate["self_chunk_effective"]
                and candidate["stacked_rkv_effective"] == args.stacked_rkv
                and control["block_fp16_accum_effective"]
                and candidate["block_fp16_accum_effective"]
            )
            passed = bool(
                route_ok
                and prompt_cosine >= args.min_cosine
                and decode_cosine >= args.min_cosine
                and prompt_greedy
                and decode_greedy
            )
            passed_all = passed_all and passed
            row = {
                "axis": "native_prefill_self_chunk_same_process_ab",
                "status": "pass" if passed else "fail",
                "device": torch.cuda.get_device_name(),
                "dtype": args.dtype,
                "hidden_size": int(model.config.hidden_size),
                "num_hidden_layers": int(model.config.num_hidden_layers),
                "batch_size": args.batch_size,
                "prompt_tokens": args.prompt_tokens,
                "chunk_size": args.chunk_size,
                "stacked_rkv_requested": args.stacked_rkv,
                "order_index": order_index,
                "candidate_first": order[0],
                "control_ms": round(float(control["prefill_ms"]), 6),
                "candidate_ms": round(float(candidate["prefill_ms"]), 6),
                "speedup": round(
                    float(control["prefill_ms"]) / float(candidate["prefill_ms"]),
                    6,
                ),
                "prompt_cosine": prompt_cosine,
                "decode_cosine": decode_cosine,
                "prompt_greedy_match": prompt_greedy,
                "decode_greedy_match": decode_greedy,
                "control_peak_vram_mb": control["peak_vram_mb"],
                "candidate_peak_vram_mb": candidate["peak_vram_mb"],
                "control_self_chunk_effective": control["self_chunk_effective"],
                "candidate_self_chunk_effective": candidate[
                    "self_chunk_effective"
                ],
                "candidate_stacked_rkv_effective": candidate[
                    "stacked_rkv_effective"
                ],
                "block_fp16_accum_effective": candidate[
                    "block_fp16_accum_effective"
                ],
                "min_cosine_gate": args.min_cosine,
            }
            _append(args.results, row)
            print(json.dumps(row, ensure_ascii=False))
        return 0 if passed_all else 1
    finally:
        if temporary_dir is not None:
            temporary_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
