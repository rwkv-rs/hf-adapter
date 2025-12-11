#!/usr/bin/env python3
# coding=utf-8
"""Compare an opt-in Native CUDA-graph route with conservative Native decode."""
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from rwkv7_hf.ada_lora import ada_wagv_lora_available, ada_wagv_lora_build_error
from rwkv7_hf.ada_sparse_ffn import ada_sparse_ffn_available, ada_sparse_ffn_build_error
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

SEED = "User: Summarize recurrent neural networks and cache reuse.\n\nAssistant:" * 16

BASELINE_ENV = {
    "RWKV7_NATIVE_MODEL_BACKEND": "native_graph",
    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN": "0",
    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_FP32_ACCUM": "0",
    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_OFFICIAL_BOUNDARY": "0",
    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_UP": "0",
    "RWKV7_NATIVE_GRAPH_ADA_WAG_LORA": "0",
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA": "0",
    "RWKV7_NATIVE_GRAPH_ADA_LINEAR": "0",
    "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX": "1",
    "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS": "8",
    "RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW": "1",
    "RWKV7_NATIVE_GRAPH_RKV_POLICY": "vkwr_auto",
}

CANDIDATE_ENV = {
    "RWKV7_NATIVE_MODEL_BACKEND": "native_graph",
    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN": "1",
    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_FP32_ACCUM": "1",
    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_OFFICIAL_BOUNDARY": "0",
    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_MAX_ROWS": "19",
    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_UP": "1",
    "RWKV7_NATIVE_GRAPH_ADA_WAG_LORA": "1",
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA": "1",
    "RWKV7_NATIVE_GRAPH_ADA_LINEAR": "1",
    "RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROWS": "1",
    "RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROLES": "hidden,ffn_up,ffn_down",
    "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX": "1",
    "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS": "8",
    "RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW": "1",
    "RWKV7_NATIVE_GRAPH_RKV_POLICY": "manual",
}

MANAGED_ENV = frozenset(BASELINE_ENV) | frozenset(CANDIDATE_ENV)


@contextmanager
def decode_environment(values: dict[str, str]):
    previous = {name: os.environ.get(name) for name in MANAGED_ENV}
    for name in MANAGED_ENV:
        os.environ.pop(name, None)
    os.environ.update(values)
    try:
        yield
    finally:
        for name in MANAGED_ENV:
            value = previous[name]
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def encode(tokenizer, prompt_tokens: int, batch_size: int, device: str) -> torch.Tensor:
    ids = tokenizer(SEED, return_tensors="pt", add_special_tokens=False).input_ids
    ids = ids[:, :prompt_tokens].repeat(batch_size, 1)
    return ids.to(device)


def decode_trace(
    model: NativeRWKV7ForCausalLM,
    prompt: torch.Tensor,
    *,
    steps: int,
    environment: dict[str, str],
    teacher_tokens: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    with decode_environment(environment), torch.inference_mode():
        model.rwkv7_clear_native_graph_cache()
        prefill = model(prompt, use_cache=True, logits_to_keep=1)
        state = prefill.past_key_values
        token = prefill.logits[:, -1].argmax(dim=-1)
        inputs: list[torch.Tensor] = []
        logits_trace: list[torch.Tensor] = []
        for step in range(steps):
            if teacher_tokens is not None:
                token = teacher_tokens[step].to(prompt.device)
            inputs.append(token.detach().cpu())
            logits, state = model.rwkv7_forward_token(
                token,
                past_key_values=state,
                return_dict=False,
                copy_logits=True,
            )
            last_logits = logits[:, -1].float()
            logits_trace.append(last_logits.detach().cpu())
            token = last_logits.argmax(dim=-1)
    return inputs, logits_trace


def extension_status(device: str) -> dict[str, dict[str, Any]]:
    sparse_active = ada_sparse_ffn_available(device, build=True)
    lora_active = ada_wagv_lora_available(device, build=True)
    return {
        "ada_sparse_ffn": {
            "active": bool(sparse_active),
            "error": ada_sparse_ffn_build_error(),
        },
        "ada_lora": {
            "active": bool(lora_active),
            "error": ada_wagv_lora_build_error(),
        },
    }


def compare_traces(
    baseline: list[torch.Tensor],
    candidate: list[torch.Tensor],
) -> dict[str, Any]:
    cosine_values: list[float] = []
    max_abs_diff = 0.0
    top1_matches = 0
    top1_total = 0
    finite = True
    for reference, observed in zip(baseline, candidate):
        cosine = F.cosine_similarity(reference, observed, dim=-1)
        cosine_values.extend(float(value) for value in cosine.tolist())
        max_abs_diff = max(max_abs_diff, float((reference - observed).abs().max().item()))
        top1_matches += int((reference.argmax(dim=-1) == observed.argmax(dim=-1)).sum().item())
        top1_total += int(reference.shape[0])
        finite = finite and bool(torch.isfinite(observed).all().item())
    return {
        "min_logits_cosine": min(cosine_values),
        "mean_logits_cosine": sum(cosine_values) / len(cosine_values),
        "max_logits_abs_diff": max_abs_diff,
        "top1_matches": top1_matches,
        "top1_total": top1_total,
        "top1_match_rate": top1_matches / top1_total,
        "logits_finite": finite,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--min-cosine", type=float, default=0.9999)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    if args.steps <= 0 or args.prompt_tokens <= 0:
        raise ValueError("steps and prompt-tokens must be positive")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    extensions = extension_status(args.device)
    inactive = {name: item for name, item in extensions.items() if not item["active"]}
    if inactive:
        raise RuntimeError(
            "candidate CUDA extensions are inactive; refusing fallback alignment: "
            + json.dumps(inactive, ensure_ascii=False)
        )

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.hf_dir,
        torch_dtype=dtype,
        device_map=args.device,
    ).eval()

    rows = []
    passed = True
    for batch_size in args.batch_sizes:
        prompt = encode(tokenizer, args.prompt_tokens, batch_size, args.device)
        teacher_tokens, baseline_logits = decode_trace(
            model,
            prompt,
            steps=args.steps,
            environment=BASELINE_ENV,
        )
        _, candidate_logits = decode_trace(
            model,
            prompt,
            steps=args.steps,
            environment=CANDIDATE_ENV,
            teacher_tokens=teacher_tokens,
        )
        metrics = compare_traces(baseline_logits, candidate_logits)
        row = {
            "axis": "native_model_decode_alignment",
            "device": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
            "prompt_tokens": args.prompt_tokens,
            "steps": args.steps,
            **metrics,
            "baseline_backend": "native_graph_conservative_vkwr_auto",
            "candidate_backend": "native_graph_fp32_sparse_manual_rkv_wagv_wag",
            "requested_extensions": extensions,
        }
        row_passed = bool(
            row["logits_finite"]
            and row["min_logits_cosine"] >= args.min_cosine
            and row["top1_match_rate"] == 1.0
        )
        row["status"] = "pass" if row_passed else "fail"
        passed = passed and row_passed
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    if args.results:
        output = Path(args.results)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended {len(rows)} rows -> {output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
