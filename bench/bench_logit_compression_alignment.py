#!/usr/bin/env python3
# coding=utf-8
"""Uncheatable logit-alignment benchmark using compression/NLL.

This is stricter than max-diff / cosine / greedy checks.  It scores the
probability assigned by two inference paths to fixed, external target tokens
and reports:

* bits/token for the reference and candidate paths;
* candidate-vs-reference compression ratio;
* compression ratio vs token position bins.

The target tokens come from input text / JSONL fields, not from either model
path, so the metric cannot be gamed by matching only sampled generations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
DEFAULT_TEXT = (
    "User: Solve the following math problem and put the final answer in a box. "
    "If x + 2y = 7 and 3x - y = 4, find x. "
    "Assistant: We need solve the linear system carefully."
)


FALSE_VALUES = {"0", "false", "False", "no", "off"}


@contextmanager
def env_override(**updates: str):
    old = {k: os.environ.get(k) for k in updates}
    for key, value in updates.items():
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def infer_model_size_label(hf_dir: str, explicit: str = "") -> str | None:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(hf_dir).name.lower())
    return match.group(1) if match else None


def load_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        raw = Path(args.text_file).read_text(encoding="utf-8")
        if args.text_file_split_lines:
            texts.extend(line.strip() for line in raw.splitlines() if line.strip())
        else:
            texts.append(raw)
    if args.jsonl:
        with Path(args.jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                value = str(item[args.jsonl_field])
                if args.jsonl_prefix:
                    value = args.jsonl_prefix + value
                if args.jsonl_suffix:
                    value = value + args.jsonl_suffix
                texts.append(value)
                if args.limit > 0 and len(texts) >= args.limit:
                    break
    if not texts:
        texts = [DEFAULT_TEXT]
    if args.limit > 0:
        texts = texts[: args.limit]
    return texts


def encode_text(tokenizer, text: str, args: argparse.Namespace) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if args.add_bos:
        bos = torch.tensor([[args.bos_token_id]], dtype=ids.dtype)
        ids = torch.cat([bos, ids], dim=1)
    if args.max_tokens > 0:
        ids = ids[:, : args.max_tokens]
    if int(ids.shape[1]) < 2:
        raise ValueError("need at least two tokens to score next-token compression")
    return ids


def forward_reference(model, ids: torch.Tensor):
    with env_override(RWKV7_FAST_PREFILL="0", RWKV7_FAST_FORWARD="0"):
        return model(ids, use_cache=True, logits_to_keep=int(ids.shape[1]), return_dict=True)


def forward_candidate(model, ids: torch.Tensor, candidate: str):
    if candidate == "native_prefill":
        return model.rwkv7_prefill_native(ids, logits_to_keep=int(ids.shape[1]), return_dict=True)
    if candidate == "forward_fast_prefill":
        with env_override(RWKV7_FAST_PREFILL="1"):
            return model(ids, use_cache=True, logits_to_keep=int(ids.shape[1]), return_dict=True)
    if candidate == "forward":
        return model(ids, use_cache=True, logits_to_keep=int(ids.shape[1]), return_dict=True)
    raise ValueError(f"unknown candidate path: {candidate}")


def nll_bits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    nll = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return nll / math.log(2.0)


def iter_position_bins(num_positions: int, bin_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, num_positions, bin_size):
        yield start, min(num_positions, start + bin_size)


def score_one(model, tokenizer, text: str, args: argparse.Namespace, device: str) -> dict[str, Any]:
    ids = encode_text(tokenizer, text, args).to(device)
    with torch.inference_mode():
        ref = forward_reference(model, ids)
        cand = forward_candidate(model, ids, args.candidate)
    ref_logits = ref.logits[:, :-1, :]
    cand_logits = cand.logits[:, :-1, :]
    targets = ids[:, 1:]
    if ref_logits.shape[:2] != targets.shape or cand_logits.shape[:2] != targets.shape:
        raise RuntimeError(
            f"logit/target shape mismatch ref={tuple(ref_logits.shape)} "
            f"cand={tuple(cand_logits.shape)} targets={tuple(targets.shape)}"
        )

    ref_bits = nll_bits(ref_logits, targets).view_as(targets)
    cand_bits = nll_bits(cand_logits, targets).view_as(targets)
    vocab_bits = math.log2(float(ref_logits.shape[-1]))
    diff = (ref_logits.float() - cand_logits.float()).detach()
    ref_argmax = ref_logits.argmax(dim=-1)
    cand_argmax = cand_logits.argmax(dim=-1)

    bins = []
    flat_ref = ref_bits.reshape(-1)
    flat_cand = cand_bits.reshape(-1)
    # Current benchmark processes one sequence per row, so positions are direct.
    for start, end in iter_position_bins(int(targets.shape[1]), args.position_bin_size):
        r = ref_bits[:, start:end].reshape(-1)
        c = cand_bits[:, start:end].reshape(-1)
        r_sum = float(r.sum().detach().cpu())
        c_sum = float(c.sum().detach().cpu())
        count = int(r.numel())
        bins.append(
            {
                "start_pos": start + 1,
                "end_pos": end,
                "token_count": count,
                "ref_bits_per_token": r_sum / max(count, 1),
                "candidate_bits_per_token": c_sum / max(count, 1),
                "candidate_vs_ref_bits_ratio": c_sum / max(r_sum, 1e-12),
                "ref_compression_ratio_vs_uniform": r_sum / max(count * vocab_bits, 1e-12),
                "candidate_compression_ratio_vs_uniform": c_sum / max(count * vocab_bits, 1e-12),
            }
        )

    ref_sum = float(flat_ref.sum().detach().cpu())
    cand_sum = float(flat_cand.sum().detach().cpu())
    count = int(flat_ref.numel())
    return {
        "token_count": count,
        "input_tokens": int(ids.shape[1]),
        "ref_bits_sum": ref_sum,
        "candidate_bits_sum": cand_sum,
        "ref_bits_per_token": ref_sum / max(count, 1),
        "candidate_bits_per_token": cand_sum / max(count, 1),
        "candidate_vs_ref_bits_ratio": cand_sum / max(ref_sum, 1e-12),
        "ref_compression_ratio_vs_uniform": ref_sum / max(count * vocab_bits, 1e-12),
        "candidate_compression_ratio_vs_uniform": cand_sum / max(count * vocab_bits, 1e-12),
        "max_abs_diff": float(diff.abs().max().detach().cpu()),
        "mean_abs_diff": float(diff.abs().mean().detach().cpu()),
        "argmax_match": int((ref_argmax == cand_argmax).sum().detach().cpu()),
        "argmax_total": int(ref_argmax.numel()),
        "position_bins": bins,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = sum(int(r["token_count"]) for r in rows)
    ref_sum = sum(float(r["ref_bits_sum"]) for r in rows)
    cand_sum = sum(float(r["candidate_bits_sum"]) for r in rows)
    argmax_match = sum(int(r["argmax_match"]) for r in rows)
    argmax_total = sum(int(r["argmax_total"]) for r in rows)
    return {
        "sample_count": len(rows),
        "token_count": total_tokens,
        "ref_bits_sum": ref_sum,
        "candidate_bits_sum": cand_sum,
        "ref_bits_per_token": ref_sum / max(total_tokens, 1),
        "candidate_bits_per_token": cand_sum / max(total_tokens, 1),
        "candidate_vs_ref_bits_ratio": cand_sum / max(ref_sum, 1e-12),
        "argmax_match": argmax_match,
        "argmax_total": argmax_total,
        "argmax_match_rate": argmax_match / max(argmax_total, 1),
        "max_abs_diff": max(float(r["max_abs_diff"]) for r in rows) if rows else None,
        "mean_abs_diff_mean": sum(float(r["mean_abs_diff"]) for r in rows) / max(len(rows), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--model-size-label", default="")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--candidate", choices=["native_prefill", "forward_fast_prefill", "forward"], default="native_prefill")
    ap.add_argument("--text", default="")
    ap.add_argument("--text-file", default="")
    ap.add_argument("--text-file-split-lines", action="store_true")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--jsonl-field", default="problem")
    ap.add_argument("--jsonl-prefix", default="")
    ap.add_argument("--jsonl-suffix", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--add-bos", action="store_true")
    ap.add_argument("--bos-token-id", type=int, default=0)
    ap.add_argument("--position-bin-size", type=int, default=32)
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device if device.startswith("cuda") else None,
    ).eval()
    texts = load_texts(args)
    sample_rows = [score_one(model, tokenizer, text, args, device) for text in texts]
    summary = {
        "axis": "logit_compression_alignment",
        "backend": "hf_adapter",
        "candidate": args.candidate,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
        "model_name": Path(args.hf_dir).name,
        "model_size_label": infer_model_size_label(args.hf_dir, args.model_size_label),
        "hf_model_dir": args.hf_dir,
        "max_tokens": args.max_tokens,
        "position_bin_size": args.position_bin_size,
        **aggregate(sample_rows),
        "samples": sample_rows,
        "status": "pass" if sample_rows else "fail",
    }
    print(json.dumps(summary, ensure_ascii=False))
    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
