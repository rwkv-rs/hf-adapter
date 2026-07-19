from __future__ import annotations

import json
from pathlib import Path

from bench.summarize_4080_qwen35_acceptance import summarize


def shapes(batch_size: int) -> list[tuple[int, int, int]]:
    return [
        (batch_size, prompt, decode)
        for prompt in (128, 512, 2048)
        for decode in (128, 512)
    ]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def create_passing_artifact(root: Path, *, batch_size: int = 8) -> None:
    dense = []
    for batch, prompt, decode in shapes(batch_size):
        common = {
            "batch_size": batch,
            "prompt_tokens": prompt,
            "decode_tokens": decode,
            "device": "NVIDIA GeForce RTX 4080",
            "dtype": "fp16",
            "model_pair": "rwkv-1.5b__qwen3.5-2b",
            "status": "pass",
            "logits_finite": True,
        }
        dense.append(
            {
                **common,
                "model_role": "candidate",
                "model_kind": "rwkv",
                "model_size_label": "1.5b",
                "effective_backend": "native_graph",
                "prefill_effective_backend": "native_prefill_graph",
                "prefill_tokps_total": 110.0,
                "decode_tokps_total": 150.0,
                "decode_tokps_per_active_billion": 160.0,
                "model_footprint_mb": 100.0,
            }
        )
        dense.append(
            {
                **common,
                "model_role": "reference",
                "model_kind": "qwen35",
                "model_size_label": "2b",
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
        for batch, prompt, decode in shapes(batch_size):
            memory.append(
                {
                    "batch_size": batch,
                    "prompt_tokens": prompt,
                    "decode_tokens": decode,
                    "device": "NVIDIA GeForce RTX 4080",
                    "dtype": "fp16",
                    "model_pair": "rwkv-1.5b__qwen3.5-2b",
                    "model_kind": "rwkv",
                    "model_size_label": "1.5b",
                    "quantization": quantization,
                    "status": "pass",
                    "logits_finite": True,
                    "model_footprint_mb": 60.0,
                }
            )
    for quantization in ("a8w8", "torchao_w4"):
        for batch, prompt, decode in shapes(batch_size):
            paired.append(
                {
                    "batch_size": batch,
                    "prompt_tokens": prompt,
                    "decode_tokens": decode,
                    "device": "NVIDIA GeForce RTX 4080",
                    "dtype": "fp16",
                    "model_size_label": "1.5b",
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


def test_4080_summary_supports_exact_b1_matrix(tmp_path: Path) -> None:
    create_passing_artifact(tmp_path, batch_size=1)
    report = summarize(tmp_path, batch_size=1)
    assert report["status"] == "pass"
    assert report["axis"] == "rtx4080_qwen35_bsz1_acceptance"
    assert report["scope"]["batch_size"] == 1
    assert report["gates"]["min_active_work_decode_ratio"] == 1.0


def test_4080_summary_fails_mismatched_pair_contract(tmp_path: Path) -> None:
    create_passing_artifact(tmp_path)
    dense_path = tmp_path / "dense.jsonl"
    rows = [json.loads(line) for line in dense_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["model_pair"] = "rwkv-0.4b__qwen3.5-0.8b"
    write_rows(dense_path, rows)

    report = summarize(tmp_path)

    assert report["status"] == "fail"
    assert any("wrong model-pair label" in error for error in report["errors"])


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


def test_4080_summary_accepts_native_chunk_continuation(tmp_path: Path) -> None:
    create_passing_artifact(tmp_path)
    dense_path = tmp_path / "dense.jsonl"
    rows = [json.loads(line) for line in dense_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("model_role") == "candidate" and row.get("prompt_tokens") == 2048:
            row["prefill_chunk_size"] = 512
            row["prefill_effective_backend"] = "native_prefill_continuation"
    write_rows(dense_path, rows)

    report = summarize(tmp_path)

    assert report["status"] == "pass"


def test_4080_summary_fails_active_work_decode_gate(tmp_path: Path) -> None:
    create_passing_artifact(tmp_path)
    dense_path = tmp_path / "dense.jsonl"
    rows = [json.loads(line) for line in dense_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("model_role") == "candidate":
            row["decode_tokps_per_active_billion"] = 130.0
    write_rows(dense_path, rows)

    report = summarize(tmp_path)

    assert report["status"] == "fail"
    assert any("active-work decode ratio" in error for error in report["errors"])
