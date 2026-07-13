#!/usr/bin/env python3
"""Compare current MM4 and groupwise W4 on exact safetensors weights."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rwkv7_hf.native_quant_mm4 import dequantize_mm4, quantize_mm4
from rwkv7_hf.native_quant_mm4_groupwise import (
    dequantize_groupwise_mm4,
    groupwise_mm4_storage_bytes,
    quantize_groupwise_mm4,
)
from rwkv7_hf.native_quant_mm8 import dequantize_mm8, quantize_mm8


DEFAULT_WEIGHTS = (
    "model.layers.0.ffn.key.weight",
    "model.layers.0.ffn.value.weight",
    "lm_head.weight",
)


def tensor_bytes(*tensors) -> int:
    return sum(int(t.numel()) * int(t.element_size()) for t in tensors)


def stable_flat_metrics(reference, candidate, *, chunk_elements: int = 4_194_304):
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    dot = 0.0
    ref_sq = 0.0
    cand_sq = 0.0
    error_sq = 0.0
    for start in range(0, ref.numel(), chunk_elements):
        stop = min(start + chunk_elements, ref.numel())
        ref_chunk = ref[start:stop].double()
        cand_chunk = cand[start:stop].double()
        dot += torch.dot(ref_chunk, cand_chunk).item()
        ref_sq += torch.dot(ref_chunk, ref_chunk).item()
        cand_sq += torch.dot(cand_chunk, cand_chunk).item()
        error = cand_chunk - ref_chunk
        error_sq += torch.dot(error, error).item()
    cosine = dot / max(math.sqrt(ref_sq * cand_sq), 1e-30)
    nrmse = math.sqrt(error_sq / max(ref_sq, 1e-30))
    return cosine, nrmse


def metrics(reference, candidate, x, sample_columns: int) -> dict:
    flat_cos, nrmse = stable_flat_metrics(reference, candidate)
    columns = min(int(sample_columns), int(reference.shape[1]))
    index = torch.linspace(0, reference.shape[1] - 1, columns).round().long()
    ref_y = x @ reference[:, index].float()
    cand_y = x @ candidate[:, index].float()
    output_cos = F.cosine_similarity(ref_y.flatten(), cand_y.flatten(), dim=0).item()
    return {
        "weight_cosine": round(flat_cos, 9),
        "weight_nrmse": round(nrmse, 9),
        "output_cosine": round(output_cos, 9),
    }


def emit(path: Path | None, row: dict) -> None:
    print(json.dumps(row, sort_keys=True), flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weight", action="append", default=[])
    parser.add_argument("--group-size", action="append", type=int, default=[])
    parser.add_argument("--activation-rows", type=int, default=2)
    parser.add_argument("--sample-columns", type=int, default=4096)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    weights = tuple(args.weight) or DEFAULT_WEIGHTS
    group_sizes = tuple(args.group_size) or (32, 64, 128)
    torch.manual_seed(20260713)

    with safe_open(str(args.model), framework="pt", device="cpu") as handle:
        for name in weights:
            dense_linear = handle.get_tensor(name)
            weight = dense_linear.t().contiguous()
            x = torch.randn(args.activation_rows, weight.shape[0], dtype=torch.float32)
            dense_bytes = tensor_bytes(dense_linear)

            current = quantize_mm4(weight)
            current_dense = dequantize_mm4(
                current[0], current[1], current[2], current[3], current[4], current[5],
                out_dtype=weight.dtype,
            )
            emit(
                args.results,
                {
                    "axis": "native_mm4_quality_oracle",
                    "status": "pass",
                    "weight": name,
                    "shape": list(dense_linear.shape),
                    "format": "mm4_affine",
                    "storage_bytes": tensor_bytes(*current[:5]),
                    "storage_ratio": round(tensor_bytes(*current[:5]) / dense_bytes, 6),
                    **metrics(weight, current_dense, x, args.sample_columns),
                },
            )
            del current, current_dense

            mm8 = quantize_mm8(weight)
            mm8_dense = dequantize_mm8(*mm8, out_dtype=weight.dtype)
            emit(
                args.results,
                {
                    "axis": "native_mm4_quality_oracle",
                    "status": "pass",
                    "weight": name,
                    "shape": list(dense_linear.shape),
                    "format": "mm8_affine",
                    "storage_bytes": tensor_bytes(*mm8),
                    "storage_ratio": round(tensor_bytes(*mm8) / dense_bytes, 6),
                    **metrics(weight, mm8_dense, x, args.sample_columns),
                },
            )
            del mm8, mm8_dense

            for group_size in group_sizes:
                grouped = quantize_groupwise_mm4(weight, group_size=group_size)
                grouped_dense = dequantize_groupwise_mm4(
                    grouped[0], grouped[1], grouped[2], grouped[3], grouped[4], grouped[7],
                    out_dtype=weight.dtype,
                )
                storage = groupwise_mm4_storage_bytes(grouped[0], grouped[1], grouped[2])
                emit(
                    args.results,
                    {
                        "axis": "native_mm4_quality_oracle",
                        "status": "pass",
                        "weight": name,
                        "shape": list(dense_linear.shape),
                        "format": "mm4_groupwise",
                        "group_size": group_size,
                        "storage_bytes": storage,
                        "storage_ratio": round(storage / dense_bytes, 6),
                        **metrics(weight, grouped_dense, x, args.sample_columns),
                    },
                )
                del grouped, grouped_dense
            del dense_linear, weight, x
            gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
