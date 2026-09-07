# coding=utf-8
"""Compare scoped FP16 GEMM accumulation with the FP32-accumulation oracle."""
from __future__ import annotations

# Support direct ``python bench/<category>/<script>.py`` execution.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.runners.bench_native_prefill_scan import (
    DTYPES,
    build_ids,
    cosine_min,
    env_override,
    infer_model_size_label,
    prepare_model_dir,
)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def _run_prefill(
    model,
    ids: torch.Tensor,
    *,
    fp16_accum: bool,
    chunk_size: int,
) -> dict[str, Any]:
    model.rwkv7_clear_native_prefill_graph_cache()
    with env_override(
        RWKV7_NATIVE_PREFILL_GLOBAL_FP16_ACCUM="1" if fp16_accum else "0",
    ):
        with torch.inference_mode():
            if chunk_size > 0:
                output = model.rwkv7_prefill_chunks(
                    ids,
                    chunk_size=chunk_size,
                    logits_to_keep=1,
                    return_dict=True,
                )
            else:
                output = model.rwkv7_prefill_native(
                    ids,
                    logits_to_keep=1,
                    return_dict=True,
                )
            prompt_logits = output.logits[:, -1].detach().float().cpu().clone()
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            next_output = model.rwkv7_forward_token(
                next_token,
                past_key_values=output.past_key_values,
                return_dict=True,
            )
            decode_logits = next_output.logits[:, -1].detach().float().cpu().clone()
            effective = bool(
                getattr(
                    model,
                    "_rwkv7_native_prefill_global_fp16_accum_effective",
                    False,
                )
            )
    return {
        "prompt_logits": prompt_logits,
        "decode_logits": decode_logits,
        "effective": effective,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=DTYPES, default="fp16")
    parser.add_argument("--batch-size", type=int, nargs="+", required=True)
    parser.add_argument("--prompt-tokens", type=int, nargs="+", required=True)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--code-source", choices=("model", "repo"), default="repo")
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--results", default="")
    args = parser.parse_args()

    effective_path, temporary_dir = prepare_model_dir(
        args.model,
        code_source=args.code_source,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            effective_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            effective_path,
            trust_remote_code=True,
            dtype=DTYPES[args.dtype],
            device_map=args.device if args.device.startswith("cuda") else None,
        ).eval()
        passed_all = True
        for batch_size in args.batch_size:
            for prompt_tokens in args.prompt_tokens:
                ids = build_ids(tokenizer, batch_size, prompt_tokens, args.device)
                reference = _run_prefill(
                    model,
                    ids,
                    fp16_accum=False,
                    chunk_size=args.chunk_size,
                )
                candidate = _run_prefill(
                    model,
                    ids,
                    fp16_accum=True,
                    chunk_size=args.chunk_size,
                )

                prompt_cosine = cosine_min(
                    reference["prompt_logits"], candidate["prompt_logits"]
                )
                decode_cosine = cosine_min(
                    reference["decode_logits"], candidate["decode_logits"]
                )
                prompt_greedy = bool(
                    torch.equal(
                        reference["prompt_logits"].argmax(dim=-1),
                        candidate["prompt_logits"].argmax(dim=-1),
                    )
                )
                decode_greedy = bool(
                    torch.equal(
                        reference["decode_logits"].argmax(dim=-1),
                        candidate["decode_logits"].argmax(dim=-1),
                    )
                )
                passed = bool(
                    not reference["effective"]
                    and candidate["effective"]
                    and prompt_cosine >= args.min_cosine
                    and decode_cosine >= args.min_cosine
                    and prompt_greedy
                    and decode_greedy
                )
                passed_all = passed_all and passed
                row = {
                    "axis": "native_prefill_global_fp16_accum_correctness",
                    "status": "pass" if passed else "fail",
                    "reference": "native_prefill_fp32_accum",
                    "candidate": "native_prefill_fp16_accum",
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
                    "batch_size": batch_size,
                    "prompt_tokens": prompt_tokens,
                    "chunk_size": args.chunk_size,
                    "reference_effective": reference["effective"],
                    "candidate_effective": candidate["effective"],
                    "min_cosine": round(prompt_cosine, 8),
                    "max_abs_diff": round(
                        _max_abs(
                            reference["prompt_logits"], candidate["prompt_logits"]
                        ),
                        6,
                    ),
                    "greedy_match": prompt_greedy,
                    "decode_after_prefill_min_cosine": round(decode_cosine, 8),
                    "decode_after_prefill_max_abs_diff": round(
                        _max_abs(
                            reference["decode_logits"], candidate["decode_logits"]
                        ),
                        6,
                    ),
                    "decode_after_prefill_greedy_match": decode_greedy,
                    "min_cosine_gate": args.min_cosine,
                }
                print(json.dumps(row, ensure_ascii=False))
                if args.results:
                    output = Path(args.results)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return 0 if passed_all else 2
    finally:
        if temporary_dir is not None:
            temporary_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
