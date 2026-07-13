#!/usr/bin/env python3
"""Compare isolated Qwen3.5 FLA and Torch correctness probes."""
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


def compare(fla: dict[str, Any], torch_ref: dict[str, Any], min_cosine: float) -> dict[str, Any]:
    inputs_match = bool(torch.equal(fla["input_ids"], torch_ref["input_ids"]))
    greedy_match = bool(torch.equal(fla["greedy_tokens"], torch_ref["greedy_tokens"]))
    prompt_cosine = cosine(fla["prompt_logits"], torch_ref["prompt_logits"])
    final_cosine = cosine(fla["final_logits"], torch_ref["final_logits"])
    passed = inputs_match and greedy_match and min(prompt_cosine, final_cosine) >= min_cosine
    return {
        "axis": "qwen35_fla_torch_probe",
        "status": "pass" if passed else "fail",
        "min_cosine_required": min_cosine,
        "input_ids_match": inputs_match,
        "greedy_tokens_match": greedy_match,
        "greedy_tokens": fla["greedy_tokens"].tolist(),
        "prompt_logits_cosine": prompt_cosine,
        "prompt_logits_max_abs": max_abs(fla["prompt_logits"], torch_ref["prompt_logits"]),
        "final_logits_cosine": final_cosine,
        "final_logits_max_abs": max_abs(fla["final_logits"], torch_ref["final_logits"]),
        "fla_backend_requested": fla.get("qwen_backend_requested"),
        "torch_backend_requested": torch_ref.get("qwen_backend_requested"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fla-probe", required=True)
    parser.add_argument("--torch-probe", required=True)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--output", default="")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    fla = torch.load(args.fla_probe, map_location="cpu", weights_only=True)
    torch_ref = torch.load(args.torch_probe, map_location="cpu", weights_only=True)
    result = compare(fla, torch_ref, args.min_cosine)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return int(args.fail_on_gate and result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
