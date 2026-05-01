#!/usr/bin/env python3
"""Compare isolated RWKV-7 reference and native-prefill correctness probes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the minimum per-batch-row cosine, never a batch-flattened value."""

    if left.shape != right.shape or left.numel() == 0:
        raise ValueError(
            f"logits shape mismatch or empty tensor: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    left32 = left.float()
    right32 = right.float()
    if not bool(torch.isfinite(left32).all() and torch.isfinite(right32).all()):
        raise ValueError("logits contain NaN or Inf")
    if left32.dim() == 1:
        left32 = left32.unsqueeze(0)
        right32 = right32.unsqueeze(0)
    else:
        left32 = left32.reshape(left32.shape[0], -1)
        right32 = right32.reshape(right32.shape[0], -1)
    values = torch.nn.functional.cosine_similarity(left32, right32, dim=-1)
    result = float(values.min().item())
    if not math.isfinite(result):
        raise ValueError("logits cosine is not finite")
    return result


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _logits_metrics(left: Any, right: Any) -> dict[str, Any]:
    shape_match = bool(
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.shape == right.shape
        and left.numel() > 0
    )
    if not shape_match:
        return {
            "shape_match": False,
            "finite": False,
            "min_row_cosine": None,
            "max_abs": None,
        }
    left32 = left.float()
    right32 = right.float()
    finite = bool(torch.isfinite(left32).all() and torch.isfinite(right32).all())
    if not finite:
        return {
            "shape_match": True,
            "finite": False,
            "min_row_cosine": None,
            "max_abs": None,
        }
    return {
        "shape_match": True,
        "finite": True,
        "min_row_cosine": cosine(left, right),
        "max_abs": max_abs(left, right),
    }


def compare(
    reference: dict[str, Any], native: dict[str, Any], min_cosine: float
) -> dict[str, Any]:
    reference_inputs = reference.get("input_ids")
    native_inputs = native.get("input_ids")
    inputs_match = bool(
        isinstance(reference_inputs, torch.Tensor)
        and isinstance(native_inputs, torch.Tensor)
        and reference_inputs.shape == native_inputs.shape
        and torch.equal(reference_inputs, native_inputs)
    )
    reference_greedy = reference.get("greedy_tokens")
    native_greedy = native.get("greedy_tokens")
    greedy_match = bool(
        isinstance(reference_greedy, torch.Tensor)
        and isinstance(native_greedy, torch.Tensor)
        and reference_greedy.shape == native_greedy.shape
        and torch.equal(reference_greedy, native_greedy)
    )
    prompt = _logits_metrics(
        reference.get("prompt_logits"), native.get("prompt_logits")
    )
    final = _logits_metrics(reference.get("final_logits"), native.get("final_logits"))
    logits_gate = bool(
        prompt["shape_match"]
        and prompt["finite"]
        and final["shape_match"]
        and final["finite"]
        and prompt["min_row_cosine"] >= min_cosine
        and final["min_row_cosine"] >= min_cosine
    )
    input_ids = native_inputs
    greedy_tokens = native_greedy
    probe_batch_size = (
        int(input_ids.shape[0])
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() >= 1
        else 0
    )
    probe_tokens = (
        int(greedy_tokens.shape[0])
        if isinstance(greedy_tokens, torch.Tensor) and greedy_tokens.dim() >= 1
        else 0
    )
    distinct_batch_prompts = bool(
        isinstance(input_ids, torch.Tensor)
        and input_ids.dim() >= 2
        and probe_batch_size > 1
        and torch.unique(input_ids.reshape(probe_batch_size, -1), dim=0).shape[0]
        == probe_batch_size
    )
    reference_decode_finite = reference.get("decode_logits_finite_by_batch")
    native_decode_finite = native.get("decode_logits_finite_by_batch")
    decode_finite_shape_match = bool(
        isinstance(reference_decode_finite, torch.Tensor)
        and isinstance(native_decode_finite, torch.Tensor)
        and tuple(reference_decode_finite.shape) == (probe_batch_size,)
        and reference_decode_finite.shape == native_decode_finite.shape
    )
    reference_decode_all_finite = bool(
        decode_finite_shape_match and reference_decode_finite.bool().all().item()
    )
    native_decode_all_finite = bool(
        decode_finite_shape_match and native_decode_finite.bool().all().item()
    )
    passed = bool(
        inputs_match
        and greedy_match
        and logits_gate
        and reference_decode_all_finite
        and native_decode_all_finite
    )
    return {
        "axis": "rwkv7_native_prefill_correctness",
        "status": "pass" if passed else "fail",
        "min_cosine_required": min_cosine,
        "input_ids_match": inputs_match,
        "greedy_tokens_match": greedy_match,
        "greedy_tokens": (
            native_greedy.tolist()
            if isinstance(native_greedy, torch.Tensor)
            else []
        ),
        "probe_batch_size": probe_batch_size,
        "probe_tokens": probe_tokens,
        "distinct_batch_prompts": distinct_batch_prompts,
        "decode_finite_shape_match": decode_finite_shape_match,
        "reference_decode_logits_all_finite": reference_decode_all_finite,
        "native_decode_logits_all_finite": native_decode_all_finite,
        "prompt_logits_shape_match": prompt["shape_match"],
        "prompt_logits_finite": prompt["finite"],
        "prompt_logits_cosine": prompt["min_row_cosine"],
        "prompt_logits_max_abs": prompt["max_abs"],
        "final_logits_shape_match": final["shape_match"],
        "final_logits_finite": final["finite"],
        "final_logits_cosine": final["min_row_cosine"],
        "final_logits_max_abs": final["max_abs"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-probe", required=True)
    parser.add_argument("--native-probe", required=True)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--required-batch-size", type=int)
    parser.add_argument("--required-probe-tokens", type=int)
    parser.add_argument("--require-distinct-batch-prompts", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    reference = torch.load(args.reference_probe, map_location="cpu", weights_only=True)
    native = torch.load(args.native_probe, map_location="cpu", weights_only=True)
    result = compare(reference, native, args.min_cosine)
    contract_errors = []
    if (
        args.required_batch_size is not None
        and result["probe_batch_size"] != args.required_batch_size
    ):
        contract_errors.append(
            f"probe_batch_size={result['probe_batch_size']}, expected {args.required_batch_size}"
        )
    if (
        args.required_probe_tokens is not None
        and result["probe_tokens"] != args.required_probe_tokens
    ):
        contract_errors.append(
            f"probe_tokens={result['probe_tokens']}, expected {args.required_probe_tokens}"
        )
    if args.require_distinct_batch_prompts and not result["distinct_batch_prompts"]:
        contract_errors.append("batch prompts are not all distinct")
    result["contract_errors"] = contract_errors
    if contract_errors:
        result["status"] = "fail"
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return int(args.fail_on_gate and result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
