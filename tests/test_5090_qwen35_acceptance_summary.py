from __future__ import annotations

import json
from pathlib import Path

from bench.summarize_5090_qwen35_acceptance import BATCH_SIZES, PAIR_DIRS, summarize


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_pair(root: Path, label: str, batch_size: int) -> None:
    root.mkdir(parents=True)
    for name in (
        "pipeline_exit_code.txt",
        "matrix_failures.txt",
        "compose_exit_code.txt",
        "compare_memory_exit_code.txt",
        "compare_speed_exit_code.txt",
        "compare_active_work_exit_code.txt",
        "correctness-failures.txt",
    ):
        (root / name).write_text("0\n", encoding="utf-8")
    for name in (
        "full-fla-vs-transformers-conv-oracle.json",
        "rwkv-prefill-correctness-none.json",
        "rwkv-prefill-correctness-bnb8.json",
        "rwkv-prefill-correctness-bnb4.json",
    ):
        _write_json(
            root / name,
            {
                "status": "pass",
                "greedy_tokens_match": True,
                "prompt_logits_cosine": 1.0,
                "final_logits_cosine": 1.0,
            },
        )
    family = {
        "cells": 6,
        "min_prefill_speedup": 1.1,
        "min_decode_speedup": 2.0,
        "min_total_speedup_vs_dense": 1.0,
        "max_footprint_ratio_vs_dense": 0.9,
        "max_peak_vram_ratio_vs_dense": 0.9,
    }
    common_summary = {
        "coverage": {"expected_cells": 18, "joined_cells": 18},
        "red_cells": [],
        "gates": {"overall_pass": True},
        "speed_by_quantization": {"none": family, "w8": family, "w4": family},
    }
    _write_json(root / "summary_speed.json", common_summary)
    _write_json(root / "summary_memory.json", common_summary)
    _write_json(
        root / "summary_active_work.json",
        {
            "coverage": {"expected_cells": 6, "joined_cells": 6},
            "red_cells": [],
            "gates": {"overall_pass": True},
            "active_parameter_work": {"passing_cells": 6, "total_cells": 6},
            "speed_by_quantization": {"none": family},
        },
    )
    _write_json(
        root / "route_manifest.json",
        {"status": "pass", "failures": 0, "decisions": [{} for _ in range(12)]},
    )

    rows = []
    for quantization in ("none", "mm8", "mm4"):
        for prompt_tokens in (128, 512, 2048):
            for decode_tokens in (128, 512):
                exact = (
                    batch_size == 8
                    and label == "rwkv-1.5b__qwen3.5-2b"
                    and quantization == "none"
                    and prompt_tokens == 512
                )
                rows.append(
                    {
                        "model_pair": label,
                        "model_role": "candidate",
                        "status": "pass",
                        "logits_finite": True,
                        "quantization": quantization,
                        "batch_size": batch_size,
                        "prompt_tokens": prompt_tokens,
                        "decode_tokens": decode_tokens,
                        "prefill_backend_effective": "native_prefill_graph",
                        "rwkv_prefill_clampw_scan_effective": exact,
                        "rwkv_prefill_stacked_rkv_effective": exact,
                        "rwkv_prefill_sequence_ffn_effective": exact,
                    }
                )
                rows.append(
                    {
                        "model_pair": label,
                        "model_role": "reference",
                        "status": "pass",
                        "logits_finite": True,
                        "batch_size": batch_size,
                        "prompt_tokens": prompt_tokens,
                        "decode_tokens": decode_tokens,
                        "qwen_backend_requested": "fla",
                        "qwen_conv_backend_effective": "fla_triton",
                        "qwen_full_fused_contract_pass": True,
                        "qwen_fast_path_verified": True,
                    }
                )
    (root / "combined_auto.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_complete_5090_summary_passes(tmp_path: Path) -> None:
    for batch_size in BATCH_SIZES:
        for label, directory in PAIR_DIRS.items():
            _write_pair(tmp_path / f"b{batch_size}" / directory, label, batch_size)
    report = summarize(tmp_path)
    assert report["status"] == "pass"
    assert report["coverage"] == {
        "batch_pairs": 8,
        "expected_batch_pairs": 8,
        "candidate_rows": 144,
        "reference_rows": 144,
        "qwen_full_fla_rows": 144,
    }
    assert report["errors"] == []


def test_5090_summary_fails_on_missing_full_fla_row(tmp_path: Path) -> None:
    for batch_size in BATCH_SIZES:
        for label, directory in PAIR_DIRS.items():
            _write_pair(tmp_path / f"b{batch_size}" / directory, label, batch_size)
    path = tmp_path / "b8" / "pair_7.2b_9b" / "combined_auto.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    report = summarize(tmp_path)
    assert report["status"] == "fail"
    assert any("18 candidate + 18 reference" in error for error in report["errors"])


def test_5090_summary_accepts_explicit_partial_pair_set(tmp_path: Path) -> None:
    selected = tuple(list(PAIR_DIRS)[:3])
    for batch_size in BATCH_SIZES:
        for label in selected:
            _write_pair(
                tmp_path / f"b{batch_size}" / PAIR_DIRS[label],
                label,
                batch_size,
            )
    report = summarize(tmp_path, selected)
    assert report["status"] == "pass"
    assert report["matrix_complete"] is False
    assert report["coverage"] == {
        "batch_pairs": 6,
        "expected_batch_pairs": 6,
        "candidate_rows": 108,
        "reference_rows": 108,
        "qwen_full_fla_rows": 108,
    }
