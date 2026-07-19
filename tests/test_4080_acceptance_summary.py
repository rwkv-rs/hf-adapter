from __future__ import annotations

import json
from pathlib import Path

from bench.summarize_4080_qwen35_acceptance import summarize


SHAPES = [
    (8, prompt, decode)
    for prompt in (128, 512, 2048)
    for decode in (128, 512)
]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def create_passing_artifact(root: Path) -> None:
    dense = []
    for batch, prompt, decode in SHAPES:
        common = {
            "batch_size": batch,
            "prompt_tokens": prompt,
            "decode_tokens": decode,
            "status": "pass",
            "logits_finite": True,
        }
        dense.append(
            {
                **common,
                "model_role": "candidate",
                "effective_backend": "native_graph",
                "prefill_effective_backend": "native_prefill_graph",
                "prefill_tokps_total": 110.0,
                "decode_tokps_total": 150.0,
                "decode_tokps_per_active_billion": 120.0,
                "model_footprint_mb": 100.0,
            }
        )
        dense.append(
            {
                **common,
                "model_role": "reference",
                "qwen_backend_requested": "fla",
                "qwen_full_fused_contract_pass": True,
                "qwen_fast_path_verified": True,
                "prefill_tokps_total": 100.0,
                "decode_tokps_total": 100.0,
                "decode_tokps_per_active_billion": 80.0,
                "model_footprint_mb": 120.0,
            }
        )
    write_rows(root / "dense.jsonl", dense)

    memory = []
    paired = []
    for quantization in ("bnb8", "bnb4"):
        for batch, prompt, decode in SHAPES:
            memory.append(
                {
                    "batch_size": batch,
                    "prompt_tokens": prompt,
                    "decode_tokens": decode,
                    "quantization": quantization,
                    "status": "pass",
                    "logits_finite": True,
                    "model_footprint_mb": 60.0,
                }
            )
    for quantization in ("a8w8", "torchao_w4"):
        for batch, prompt, decode in SHAPES:
            paired.append(
                {
                    "batch_size": batch,
                    "prompt_tokens": prompt,
                    "decode_tokens": decode,
                    "quantization": quantization,
                    "status": "pass",
                    "paired_baseline": True,
                    "same_greedy_tokens_as_fp16": True,
                    "prefill_speed_ratio_vs_fp16": 1.01,
                    "decode_speed_ratio_vs_fp16": 1.02,
                    "baseline_prefill_tokps_total": 100.0,
                    "baseline_decode_tokps_total": 100.0,
                    "prefill_tokps_total": 101.0,
                    "decode_tokps_total": 102.0,
                    "footprint_ratio_vs_fp16": 0.9,
                    "prompt_logits_cos_vs_fp16": 0.9999,
                    "final_logits_cos_vs_fp16": 0.9998,
                }
            )
    write_rows(root / "memory.jsonl", memory)
    write_rows(root / "paired_quant.jsonl", paired)


def test_4080_summary_passes_complete_exact_matrix(tmp_path: Path) -> None:
    create_passing_artifact(tmp_path)
    report = summarize(tmp_path)
    assert report["status"] == "pass"
    assert report["coverage"] == {
        "dense_candidate_rows": 6,
        "qwen_reference_rows": 6,
        "memory_rows": 12,
        "paired_quant_rows": 12,
    }
    assert report["dense_vs_qwen"]["qwen_full_fla_rows"] == 6


def test_4080_summary_fails_one_slow_quant_cell(tmp_path: Path) -> None:
    create_passing_artifact(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "paired_quant.jsonl").read_text().splitlines()]
    rows[0]["decode_speed_ratio_vs_fp16"] = 0.99
    rows[0]["decode_tokps_total"] = 99.0
    write_rows(tmp_path / "paired_quant.jsonl", rows)
    report = summarize(tmp_path)
    assert report["status"] == "fail"
    assert any("a8w8" in error and "total/decode speed ratios" in error for error in report["errors"])


def test_4080_summary_reports_but_does_not_gate_prefill_alone(tmp_path: Path) -> None:
    create_passing_artifact(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "paired_quant.jsonl").read_text().splitlines()]
    row = rows[2]
    row["prefill_speed_ratio_vs_fp16"] = 0.999
    row["prefill_tokps_total"] = 99.9
    row["decode_speed_ratio_vs_fp16"] = 1.05
    row["decode_tokps_total"] = 105.0
    write_rows(tmp_path / "paired_quant.jsonl", rows)
    report = summarize(tmp_path)
    assert report["status"] == "pass"
    assert report["paired_speed_routes"]["a8w8"]["prefill_speed_ratio_vs_fp16"]["min"] == 0.999
    assert report["paired_speed_routes"]["a8w8"]["total_speed_ratio_vs_fp16"]["min"] > 1.0
