#!/usr/bin/env python3
"""Fail-closed gate for the canonical V100 production-close evidence bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ("0.1b", "0.4b", "1.5b")
BATCHES = (1, 2, 4, 8)
ALBATROSS_DECODE = {
    "0.1b": {1: 788.19, 2: 1504.72, 4: 2612.88, 8: 3611.88},
    "0.4b": {1: 469.16, 2: 810.76, 4: 1281.16, 8: 1565.95},
    "1.5b": {1: 239.12, 2: 415.21, 4: 594.33, 8: 860.64},
}
ALBATROSS_PREFILL = {
    "0.1b": {1: 39323.63, 2: 71382.68, 4: 109051.25, 8: 153368.36},
    "0.4b": {1: 18462.45, 2: 31264.66, 4: 45953.77, 8: 59046.69},
    "1.5b": {1: 11911.85, 2: 16332.13, 4: 20141.39, 8: 21807.28},
}


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exact_matrix(got, *, quantized=False):
    expected = {
        (m, b, q) if quantized else (m, b)
        for m in MODELS
        for b in BATCHES
        for q in (("a8w8", "mm4") if quantized else (None,))
    }
    return got == expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).with_name("v100_production_close_20260711"),
    )
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()
    dense_d = rows(args.dir / "dense_decode.jsonl")
    dense_p = rows(args.dir / "dense_prefill.jsonl")
    quant_d = rows(args.dir / "quant_decode_acceptance.jsonl")
    quant_p = rows(args.dir / "quant_prefill_acceptance.jsonl")
    chunked = rows(args.dir / "chunked_prefill_serving.jsonl")
    dynamic = rows(args.dir / "dynamic_batch_serving.jsonl")
    device_map = rows(args.dir / "device_map_2gpu.jsonl")
    training = rows(args.dir / "training_regression.jsonl")
    zero2 = rows(args.dir / "zero2_resume_regression.jsonl")
    zero3 = rows(args.dir / "zero3_resume_regression.jsonl")

    assert exact_matrix({(x["model_size_label"], x["batch_size"]) for x in dense_d})
    assert exact_matrix({(x["model_size_label"], x["batch_size"]) for x in dense_p})
    assert exact_matrix(
        {(x["model_size_label"], x["batch_size"], x["quantization"]) for x in quant_d},
        quantized=True,
    )
    assert exact_matrix(
        {(x["model_size_label"], x["batch_size"], x["quantization"]) for x in quant_p},
        quantized=True,
    )

    dense_decode_ratios = [
        x["api_decode_tokps_total"]
        / ALBATROSS_DECODE[x["model_size_label"]][x["batch_size"]]
        for x in dense_d
    ]
    dense_prefill_ratios = [
        x["native_prefill_tokps_total"]
        / ALBATROSS_PREFILL[x["model_size_label"]][x["batch_size"]]
        for x in dense_p
    ]
    assert min(dense_decode_ratios) >= 0.90
    assert min(dense_prefill_ratios) >= 0.90
    assert all(x["fast_token_backend_effective"] == "native_graph" for x in dense_d)
    assert all(x["native_graph_cache_hit_rate"] >= 0.99 for x in dense_d)
    assert all(
        x["prefill_graph_effective"]
        and x["greedy_match"]
        and x["decode_after_prefill_greedy_match"]
        for x in dense_p
    )

    assert all(x["decode_speed_ratio_vs_fp16"] >= 1.0 for x in quant_d)
    assert all(x["footprint_ratio_vs_fp16"] < 1.0 for x in quant_d)
    assert all(x["same_next_token_as_fp16"] for x in quant_d)
    # Prefill deltas are below timer/clock variance because only logits_to_keep=1
    # is quantized. Treat +/-1% as production-equivalent; canonical rows are
    # measured as interleaved same-process CUDA-event medians (101 samples).
    assert all(x["quant_speed_ratio_vs_fp16"] >= 0.99 for x in quant_p)
    assert all(x["payload_ratio_vs_fp16"] < 1.0 for x in quant_p)
    assert all(x["same_next_token_as_fp16"] for x in quant_p)

    chunk512 = next(x for x in chunked if x.get("chunk_size") == 512)
    assert chunk512["speed_ratio_vs_full"] >= 0.80
    assert chunk512["peak_vram_ratio_vs_full"] <= 0.85
    assert chunk512["max_abs_diff"] <= 0.1 and chunk512["decode_max_abs_diff"] <= 0.1
    assert (
        len(dynamic) == 1
        and dynamic[0]["fast_token_backend_effective"] == "native_graph"
    )
    assert dynamic[0]["native_graph_cache_hit_rate"] >= 0.99
    assert dynamic[0]["final_batch_size"] == 2 and dynamic[0]["drop_count"] > 0
    assert len(device_map) == 1 and device_map[0]["status"] == "pass"
    assert (
        device_map[0]["multi_cuda_device_map"]
        and device_map[0]["generated_equal_reference"]
    )
    assert {x["trainer_backend"] for x in training} == {"trainer", "trl_sft"}
    assert all(x["status"] == "pass" and x["max_trainable_delta"] > 0 for x in training)
    for stage, stage_rows in ((2, zero2), (3, zero3)):
        assert len(stage_rows) == 2
        assert all(
            x["status"] == "pass" and x["zero_stage"] == stage for x in stage_rows
        )
        assert all(
            x["distributed_world_size"] == 2 and x["global_step"] == 2
            for x in stage_rows
        )
        assert all(
            x["first_max_trainable_delta"] > 0 and x["resume_max_trainable_delta"] > 0
            for x in stage_rows
        )

    summary = {
        "status": "pass",
        "dense_decode_rows": len(dense_d),
        "dense_prefill_rows": len(dense_p),
        "quant_decode_rows": len(quant_d),
        "quant_prefill_rows": len(quant_p),
        "dense_decode_ratio_min": round(min(dense_decode_ratios), 4),
        "dense_decode_ratio_max": round(max(dense_decode_ratios), 4),
        "dense_prefill_ratio_min": round(min(dense_prefill_ratios), 4),
        "dense_prefill_ratio_max": round(max(dense_prefill_ratios), 4),
        "quant_decode_ratio_min": round(
            min(x["decode_speed_ratio_vs_fp16"] for x in quant_d), 4
        ),
        "quant_decode_ratio_max": round(
            max(x["decode_speed_ratio_vs_fp16"] for x in quant_d), 4
        ),
        "quant_prefill_ratio_min": round(
            min(x["quant_speed_ratio_vs_fp16"] for x in quant_p), 4
        ),
        "quant_prefill_ratio_max": round(
            max(x["quant_speed_ratio_vs_fp16"] for x in quant_p), 4
        ),
        "quant_payload_ratio_min": round(
            min(x["payload_ratio_vs_fp16"] for x in quant_p), 4
        ),
        "quant_payload_ratio_max": round(
            max(x["payload_ratio_vs_fp16"] for x in quant_p), 4
        ),
        "chunk512_speed_ratio_vs_full": chunk512["speed_ratio_vs_full"],
        "chunk512_vram_ratio_vs_full": chunk512["peak_vram_ratio_vs_full"],
        "dynamic_batch_cache_hit_rate": dynamic[0]["native_graph_cache_hit_rate"],
        "device_map_2gpu_status": device_map[0]["status"],
        "training_backends": sorted(x["trainer_backend"] for x in training),
        "zero_resume_stages": [2, 3],
    }
    encoded = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    print(encoded, end="")
    if args.summary:
        args.summary.write_text(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
