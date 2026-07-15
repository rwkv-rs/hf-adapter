#!/usr/bin/env python3
"""Compare isolated RWKV-7 reference and native-prefill correctness probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.float().reshape(-1), right.float().reshape(-1), dim=0
        ).item()
    )


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def compare(reference: dict[str, Any], native: dict[str, Any], min_cosine: float) -> dict[str, Any]:
    inputs_match = bool(torch.equal(reference["input_ids"], native["input_ids"]))
    greedy_match = bool(torch.equal(reference["greedy_tokens"], native["greedy_tokens"]))
    prompt_cosine = cosine(reference["prompt_logits"], native["prompt_logits"])
    final_cosine = cosine(reference["final_logits"], native["final_logits"])
    passed = inputs_match and greedy_match and min(prompt_cosine, final_cosine) >= min_cosine
    return {
        "axis": "rwkv7_native_prefill_correctness",
        "status": "pass" if passed else "fail",
        "min_cosine_required": min_cosine,
        "input_ids_match": inputs_match,
        "greedy_tokens_match": greedy_match,
        "greedy_tokens": native["greedy_tokens"].tolist(),
        "prompt_logits_cosine": prompt_cosine,
        "prompt_logits_max_abs": max_abs(reference["prompt_logits"], native["prompt_logits"]),
        "final_logits_cosine": final_cosine,
        "final_logits_max_abs": max_abs(reference["final_logits"], native["final_logits"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-probe", required=True)
    parser.add_argument("--native-probe", required=True)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--output", default="")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    reference = torch.load(args.reference_probe, map_location="cpu", weights_only=True)
    native = torch.load(args.native_probe, map_location="cpu", weights_only=True)
    result = compare(reference, native, args.min_cosine)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return int(args.fail_on_gate and result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
