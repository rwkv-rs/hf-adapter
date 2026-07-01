#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from bench.bench_albatross import parse_result_lines
from bench.compare_fast_token_layouts import fast_micro_rows, fast_speed_rows, latest_by_layout, load_rows, nested_num, num, ratio


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def assert_training_smoke_survives_inference_dtype_filter(tmpdir: Path) -> None:
    """Guard mixed inference/training benchmark reports.

    The stable V100 training smoke rows currently use train_dtype/dtype=fp32,
    while most serving gap reports are requested with --dtype fp16.  Training
    compatibility is an HF deliverable, so analyzer output must keep those rows
    visible instead of silently hiding Trainer/SFT/DPO/GRPO evidence behind the
    inference dtype filter.
    """

    training_rows = [
        {
            "axis": "training_smoke",
            "backend": "hf_adapter",
            "trainer_backend": backend,
            "status": "pass",
            "dtype": "fp32",
            "train_dtype": "fp32",
            "device": "Tesla V100-PCIE-32GB",
            "attn_mode": "fused_recurrent",
            "batch_size": 2,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": 2 * grad_accum,
            "max_steps": 1,
            "train_loss": 0.5,
            "train_runtime_s": 1.0,
            "train_samples_per_second": 1.0,
            "train_steps_per_second": 1.0,
            "max_trainable_delta": 1e-4,
        }
        for backend, grad_accum in (
            ("trainer", 2),
            ("trl_sft", 2),
            ("trl_dpo", 1),
            ("trl_grpo", 1),
        )
    ]
    path = tmpdir / "training_results.jsonl"
    write_jsonl(path, training_rows)

    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    backends = {row["trainer_backend"] for row in report["training_smoke"]}
    assert backends == {"trainer", "trl_sft", "trl_dpo", "trl_grpo"}
    assert all(row["max_trainable_delta"] > 0 for row in report["training_smoke"])
    assert any(
        "HF training telemetry passes for Trainer/SFT/DPO/GRPO" in item
        for item in report["next_focus"]
    )
    assert not any("training smoke telemetry incomplete" in item for item in report["next_focus"])


def assert_albatross_rows_are_parsed_and_compared(tmpdir: Path) -> None:
    sample = "\n".join(
        [
            "warmup complete",
            "RESULT B=1 T=1 iters=10 p10_ms=7.0000 p50_ms=8.0000 p90_ms=9.0000 tok_s_p50=125.00",
            "RESULT B=2 T=1 iters=10 p10_ms=8.0000 p50_ms=10.0000 p90_ms=12.0000 tok_s_p50=200.00",
        ]
    )
    albatross_rows = parse_result_lines(
        sample,
        engine="faster4_cpp",
        dtype="fp16",
        device="Tesla V100-PCIE-32GB",
        model_path="/models/rwkv7-g1d-0.1b.pth",
        model_size_label="0.1B",
        checkpoint_sha256="abc123",
    )
    assert len(albatross_rows) == 2
    assert albatross_rows[0]["axis"] == "albatross_speed"
    assert albatross_rows[0]["batch_size"] == 1
    assert albatross_rows[0]["tokens_per_sequence"] == 1
    assert albatross_rows[0]["tokps_p50"] == 125.0

    hf_rows = [
        {
            "axis": "batch_sweep",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "decode_api": "rwkv7_forward_token",
            "fast_token_backend_effective": "native_graph",
            "decode_tokps_total": 250.0,
            "prompt_tokens": 512,
            "prefill_tokps_total": 12000.0,
        },
        {
            "axis": "batch_sweep",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 2,
            "decode_api": "rwkv7_forward_token",
            "fast_token_backend_effective": "native_graph",
            "decode_tokps_total": 400.0,
            "prompt_tokens": 512,
            "prefill_tokps_total": 24000.0,
        },
    ]
    path = tmpdir / "albatross_results.jsonl"
    write_jsonl(path, hf_rows + albatross_rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    comparisons = report["albatross_decode_comparison"]
    assert [row["batch_size"] for row in comparisons] == [1, 2]
    assert [row["hf_vs_albatross_ratio"] for row in comparisons] == [2.0, 2.0]
    assert any("Albatross A/B decode comparison present" in item for item in report["next_focus"])


def assert_quantization_best_variants_are_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "quantization",
            "backend": "hf_adapter",
            "quantization": "none",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "decode_tokps": 200.0,
            "model_footprint_mb": 400.0,
            "peak_vram_mb": 600.0,
        },
        {
            "axis": "quantization",
            "backend": "hf_adapter",
            "quantization": "4bit",
            "quant_skip_policy": "memory",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "decode_tokps": 40.0,
            "model_footprint_mb": 240.0,
            "peak_vram_mb": 300.0,
        },
        {
            "axis": "quantization",
            "backend": "hf_adapter",
            "quantization": "4bit",
            "quant_skip_policy": "decode_hot",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "decode_tokps": 80.0,
            "model_footprint_mb": 280.0,
            "peak_vram_mb": 340.0,
        },
        {
            "axis": "quantization",
            "backend": "hf_adapter",
            "quantization": "8bit",
            "quant_skip_policy": "memory",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "decode_tokps": 50.0,
            "model_footprint_mb": 300.0,
            "peak_vram_mb": 360.0,
        },
    ]
    path = tmpdir / "quant_results.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    best_by_mode = {row["quantization"]: row for row in report["quantization_best_variants"]}
    assert best_by_mode["4bit"]["best_speed"]["quant_skip_policy"] == "decode_hot"
    assert best_by_mode["4bit"]["best_memory"]["quant_skip_policy"] == "memory"
    assert best_by_mode["4bit"]["decode_ratio_vs_fp16"] == 0.4
    assert best_by_mode["4bit"]["footprint_ratio_vs_fp16"] == 0.7
    assert best_by_mode["8bit"]["decode_ratio_vs_fp16"] == 0.25
    assert any("best 4bit quant variant policy=decode_hot" in item for item in report["next_focus"])


def assert_fused_backend_targets_are_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "batch_sweep",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "decode_api": "rwkv7_forward_token",
            "fast_token_backend_effective": "native_graph",
            "decode_tokps_total": 50.0,
            "prompt_tokens": 512,
            "prefill_tokps_total": 500.0,
        },
        {
            "axis": "albatross_speed",
            "backend": "albatross",
            "engine": "faster3a",
            "engine_config": "wkv=fp32io16",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "model_size_label": "0.1b",
            "batch_size": 1,
            "tokens_per_sequence": 1,
            "tokps_p50": 100.0,
        },
        {
            "axis": "albatross_speed",
            "backend": "albatross",
            "engine": "faster3a",
            "engine_config": "wkv=fp32io16",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "model_size_label": "0.1b",
            "batch_size": 1,
            "tokens_per_sequence": 512,
            "tokps_p50": 1000.0,
        },
        {
            "axis": "quantization",
            "backend": "hf_adapter",
            "quantization": "none",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "decode_tokps": 100.0,
            "model_footprint_mb": 400.0,
        },
        {
            "axis": "quantization",
            "backend": "hf_adapter",
            "quantization": "8bit",
            "quant_skip_policy": "decode_hot",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "decode_tokps": 90.0,
            "model_footprint_mb": 280.0,
        },
        {
            "axis": "quantization",
            "backend": "hf_adapter",
            "quantization": "4bit",
            "quant_skip_policy": "decode_hot",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "decode_tokps": 120.0,
            "model_footprint_mb": 200.0,
        },
    ]
    path = tmpdir / "fused_backend_targets.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    targets = report["fused_backend_targets"]
    assert targets["phase"] == "rwkv7_hf_fused_backend"
    assert targets["albatross_decode"]["current_ratio_min"] == 0.5
    assert targets["albatross_decode"]["p1_status"] == "GAP"
    assert targets["albatross_prefill"]["current_ratio_min"] == 0.5
    assert targets["albatross_prefill"]["p1_status"] == "GAP"
    quant = {row["quantization"]: row for row in targets["quantization"]}
    assert quant["8bit"]["decode_status"] == "GAP"
    assert quant["8bit"]["footprint_status"] == "PASS"
    assert quant["4bit"]["decode_status"] == "PASS"
    assert quant["4bit"]["footprint_status"] == "PASS"
    assert any("fused backend P1 pending" in item for item in report["next_focus"])
    assert any("native fused 8bit pending" in item for item in report["next_focus"])


def assert_projection_kernel_plan_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "projection_lora",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "hidden_size": 768,
            "layers": [0, 1, 11],
            "avg_timings_ms": {
                "rkv_current": 0.8,
                "rkv_bmm_candidate": 1.2,
                "wa_lora_current": 0.3,
                "wa_lora_bmm_candidate": 0.4,
            },
            "avg_current_linears_lora_sum_ms": 1.6,
            "avg_candidate_linears_lora_sum_ms": 2.1,
            "avg_candidate_speedup": 0.7619,
            "sample_matrix_profile_summary": {
                "attn_rkv_dense": {
                    "matrix_count": 3,
                    "params": 1769472,
                    "flops_per_token": 3538944,
                    "fp16_weight_mb": 3.375,
                }
            },
            "fused_kernel_plan": {
                "first_fused_fp16_target": {
                    "group": "attn_time_mix_linears_lora",
                    "members": ["r_proj", "k_proj", "v_proj", "w_lora", "a_lora", "g_lora"],
                    "current_ms": 1.6,
                    "naive_candidate_ms": 2.1,
                    "naive_candidate_speedup": 0.7619,
                },
                "fused_groups": [
                    {
                        "name": "attn_rkv_dense",
                        "members": ["r_proj", "k_proj", "v_proj"],
                        "current_ms": 0.8,
                        "naive_candidate_ms": 1.2,
                    }
                ],
                "native_quant_candidates": [
                    {"name": "ffn_key_value", "status": "planned_not_measured_by_projection_lora"}
                ],
            },
        }
    ]
    path = tmpdir / "projection_kernel_plan.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    plan = report["projection_lora"]["fused_kernel_plan"]
    assert plan["first_fused_fp16_target"]["group"] == "attn_time_mix_linears_lora"
    assert report["projection_lora"]["sample_matrix_profile_summary"]["attn_rkv_dense"]["matrix_count"] == 3
    assert any("fused projection first target: attn_time_mix_linears_lora" in item for item in report["next_focus"])


def assert_fused_projection_proto_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "fused_projection_proto",
            "backend": "hf_adapter",
            "prototype_backend": "triton_rkv_gemv",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "hidden_size": 768,
            "layers": [0, 1, 11],
            "block_m": 16,
            "block_k": 64,
            "steps": 128,
            "avg_current_ms": 0.09,
            "avg_prototype_ms": 0.12,
            "avg_speedup": 0.75,
            "max_abs_diff": 0.001953125,
            "min_cosine": 0.9999997,
            "layer_rows": [
                {"layer_idx": 0, "current_ms": 0.1, "prototype_ms": 0.13, "speedup": 0.7692},
            ],
        }
    ]
    path = tmpdir / "fused_projection_proto.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["fused_projection_proto"]["prototype_backend"] == "triton_rkv_gemv"
    assert report["fused_projection_proto"]["avg_speedup"] == 0.75
    assert any("fused R/K/V projection prototype backend=triton_rkv_gemv is slower" in item for item in report["next_focus"])


def assert_fused_wa_lora_proto_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "fused_wa_lora_proto",
            "backend": "hf_adapter",
            "prototype_backend": "triton_fused_wa_lora",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "hidden_size": 768,
            "ranks": [64],
            "layers": [0],
            "block_m": 64,
            "block_r": 64,
            "block_k": 64,
            "steps": 128,
            "avg_current_ms": 0.145,
            "avg_prototype_ms": 0.169,
            "avg_speedup": 0.858,
            "max_abs_diff": 0.015625,
            "min_cosine": 0.9999998,
            "layer_rows": [
                {"layer_idx": 0, "rank": 64, "current_ms": 0.145, "prototype_ms": 0.169, "speedup": 0.858},
            ],
        }
    ]
    path = tmpdir / "fused_wa_lora_proto.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["fused_wa_lora_proto"]["prototype_backend"] == "triton_fused_wa_lora"
    assert report["fused_wa_lora_proto"]["avg_speedup"] == 0.858
    assert any("fused W/A LoRA prototype backend=triton_fused_wa_lora is slower" in item for item in report["next_focus"])


def assert_fused_wag_lora_proto_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "fused_wag_lora_proto",
            "backend": "hf_adapter",
            "prototype_backend": "triton_fused_wag_lora",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "hidden_size": 768,
            "ranks": [{"w": 64, "a": 64, "g": 128}],
            "layers": [0],
            "block_m": 64,
            "block_r": 64,
            "block_k": 64,
            "steps": 128,
            "avg_current_ms": 0.221,
            "avg_prototype_ms": 0.214,
            "avg_speedup": 1.033,
            "max_abs_diff": 0.015625,
            "min_cosine": 0.9999998,
            "layer_rows": [
                {
                    "layer_idx": 0,
                    "ranks": {"w": 64, "a": 64, "g": 128},
                    "current_ms": 0.221,
                    "prototype_ms": 0.214,
                    "speedup": 1.033,
                },
            ],
        }
    ]
    path = tmpdir / "fused_wag_lora_proto.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["fused_wag_lora_proto"]["prototype_backend"] == "triton_fused_wag_lora"
    assert report["fused_wag_lora_proto"]["avg_speedup"] == 1.033
    assert any("fused W/A/G LoRA prototype backend=triton_fused_wag_lora speedup=1.03x" in item for item in report["next_focus"])


def assert_fused_rkv_wag_projection_proto_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "fused_rkv_wag_projection_proto",
            "backend": "hf_adapter",
            "prototype_backend": "triton_rkv_wag_down_plus_wag_up",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "hidden_size": 768,
            "ranks": [{"w": 64, "a": 64, "g": 128}],
            "layers": [0],
            "block_m": 64,
            "block_r": 64,
            "block_k": 64,
            "steps": 512,
            "avg_current_ms": 0.314,
            "avg_prototype_ms": 0.311,
            "avg_speedup": 1.01,
            "max_abs_diff": 0.015625,
            "min_cosine": 0.9999998,
            "layer_rows": [
                {
                    "layer_idx": 0,
                    "ranks": {"w": 64, "a": 64, "g": 128},
                    "current_ms": 0.314,
                    "prototype_ms": 0.311,
                    "speedup": 1.01,
                },
            ],
        }
    ]
    path = tmpdir / "fused_rkv_wag_projection_proto.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["fused_rkv_wag_projection_proto"]["prototype_backend"] == "triton_rkv_wag_down_plus_wag_up"
    assert report["fused_rkv_wag_projection_proto"]["avg_speedup"] == 1.01
    assert any("fused R/K/V + W/A/G projection prototype backend=triton_rkv_wag_down_plus_wag_up speedup=1.01x" in item for item in report["next_focus"])


def assert_fused_shift_mix_proto_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "fused_shift_mix_proto",
            "backend": "hf_adapter",
            "prototype_backend": "triton_attn_shift_mix",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "input_rank": 2,
            "hidden_size": 768,
            "layers": [0],
            "block_size": 256,
            "steps": 512,
            "avg_current_ms": 0.12,
            "avg_prototype_ms": 0.16,
            "avg_speedup": 0.75,
            "max_abs_diff": 0.0,
            "min_cosine": 0.9999999,
            "layer_rows": [
                {"layer_idx": 0, "current_ms": 0.12, "prototype_ms": 0.16, "speedup": 0.75},
            ],
        }
    ]
    path = tmpdir / "fused_shift_mix_proto.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["fused_shift_mix_proto"]["prototype_backend"] == "triton_attn_shift_mix"
    assert report["fused_shift_mix_proto"]["avg_speedup"] == 0.75
    assert any("fused attention shift-mix prototype backend=triton_attn_shift_mix is slower" in item for item in report["next_focus"])


def assert_fused_recurrent_proto_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "fused_recurrent_proto",
            "backend": "hf_adapter",
            "prototype_backend": "triton_rank1_recurrent",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "hidden_size": 768,
            "layers": [0],
            "block_n": 64,
            "steps": 256,
            "avg_current_ms": 0.22,
            "avg_prototype_ms": 0.08,
            "avg_speedup": 2.75,
            "out_max_abs_diff": 0.0234375,
            "state_max_abs_diff": 0.0039,
            "out_min_cosine": 0.9999997,
            "layer_rows": [
                {
                    "layer_idx": 0,
                    "num_heads": 12,
                    "head_dim": 64,
                    "current_ms": 0.22,
                    "prototype_ms": 0.08,
                    "speedup": 2.75,
                },
            ],
        }
    ]
    path = tmpdir / "fused_recurrent_proto.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["fused_recurrent_proto"]["prototype_backend"] == "triton_rank1_recurrent"
    assert report["fused_recurrent_proto"]["avg_speedup"] == 2.75
    assert any("fused recurrent prototype backend=triton_rank1_recurrent speedup=2.75x" in item for item in report["next_focus"])


def assert_native_graph_fused_recurrent_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "native_graph_fused_recurrent",
            "backend": "hf_adapter",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "prompt_tokens": 64,
            "steps": 32,
            "fixed_token": True,
            "baseline_effective_backend": "native_graph",
            "fused_effective_backend": "native_graph",
            "baseline_ms_per_step": 4.3,
            "fused_ms_per_step": 4.1,
            "speedup": 1.05,
            "baseline_tokps_total": 232.0,
            "fused_tokps_total": 244.0,
            "max_abs_diff_first_step": 0.0,
            "min_cosine_first_step": 1.0,
            "greedy_match": 32,
            "greedy_total": 32,
        }
    ]
    path = tmpdir / "native_graph_fused_recurrent.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["native_graph_fused_recurrent"]["speedup"] == 1.05
    assert any("native_graph fused recurrent integration passes greedy 32/32" in item for item in report["next_focus"])


def assert_native_quant_gemv_proto_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "native_quant_gemv_proto",
            "backend": "hf_adapter",
            "prototype_backend": "triton_int8_rowwise_gemv",
            "status": "pass",
            "quantization": "int8_rowwise",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "layers": [0],
            "modules": ["attn.r_proj", "ffn.key"],
            "block_m": 16,
            "block_k": 64,
            "steps": 128,
            "avg_current_ms": 0.02,
            "avg_prototype_ms": 0.05,
            "avg_speedup": 0.4,
            "max_abs_diff": 0.04,
            "mean_abs_diff_max": 0.004,
            "min_cosine": 0.9999,
            "sample_fp16_weight_mb": 5.625,
            "sample_int8_weight_mb": 2.82715,
            "sample_footprint_ratio": 0.5026,
            "layer_rows": [
                {
                    "layer_idx": 0,
                    "module": "attn.r_proj",
                    "shape": [768, 768],
                    "current_ms": 0.02,
                    "prototype_ms": 0.05,
                    "speedup": 0.4,
                    "footprint_ratio": 0.5026,
                },
            ],
        }
    ]
    path = tmpdir / "native_quant_gemv_proto.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["native_quant_gemv_proto"]["prototype_backend"] == "triton_int8_rowwise_gemv"
    assert report["native_quant_gemv_proto"]["sample_footprint_ratio"] == 0.5026
    assert any("native int8 dequant-GEMV prototype footprint=0.5026x fp16" in item for item in report["next_focus"])


def assert_native_quant_w4_gemv_proto_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "native_quant_w4_gemv_proto",
            "backend": "hf_adapter",
            "prototype_backend": "triton_int4_rowwise_gemv",
            "status": "pass",
            "quantization": "int4_rowwise",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "layers": [0],
            "modules": ["attn.r_proj", "ffn.key"],
            "block_m": 16,
            "block_k": 64,
            "steps": 128,
            "avg_current_ms": 0.02,
            "avg_prototype_ms": 0.08,
            "avg_speedup": 0.25,
            "max_abs_diff": 0.12,
            "mean_abs_diff_max": 0.02,
            "min_cosine": 0.997,
            "sample_fp16_weight_mb": 5.625,
            "sample_int4_weight_mb": 1.424,
            "sample_footprint_ratio": 0.2532,
            "layer_rows": [
                {
                    "layer_idx": 0,
                    "module": "attn.r_proj",
                    "shape": [768, 768],
                    "current_ms": 0.02,
                    "prototype_ms": 0.08,
                    "speedup": 0.25,
                    "footprint_ratio": 0.2532,
                },
            ],
        }
    ]
    path = tmpdir / "native_quant_w4_gemv_proto.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["native_quant_w4_gemv_proto"]["prototype_backend"] == "triton_int4_rowwise_gemv"
    assert report["native_quant_w4_gemv_proto"]["sample_footprint_ratio"] == 0.2532
    assert any("native int4 dequant-GEMV prototype footprint=0.2532x fp16" in item for item in report["next_focus"])


def assert_native_quant_rkv_proto_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "native_quant_rkv_proto",
            "backend": "hf_adapter",
            "prototype_backend": "triton_int8_fused_rkv_gemv",
            "status": "pass",
            "quantization": "int8_rowwise_fused_rkv",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "hidden_size": 768,
            "layers": [0],
            "block_m": 16,
            "block_k": 64,
            "steps": 128,
            "avg_fp16_current_ms": 0.09,
            "avg_separate_int8_ms": 0.16,
            "avg_fused_int8_ms": 0.11,
            "fused_speedup_vs_fp16": 0.8182,
            "fused_speedup_vs_separate_int8": 1.4545,
            "separate_speedup_vs_fp16": 0.5625,
            "max_abs_diff_fp16_vs_fused": 0.04,
            "max_abs_diff_separate_vs_fused": 0.001,
            "min_cosine_fp16_vs_fused": 0.9999,
            "min_cosine_separate_vs_fused": 1.0,
            "sample_fp16_weight_mb": 3.375,
            "sample_int8_weight_mb": 1.69629,
            "sample_footprint_ratio": 0.5026,
            "layer_rows": [
                {
                    "layer_idx": 0,
                    "fp16_current_ms": 0.09,
                    "separate_int8_ms": 0.16,
                    "fused_int8_ms": 0.11,
                    "fused_speedup_vs_separate_int8": 1.4545,
                },
            ],
        }
    ]
    path = tmpdir / "native_quant_rkv_proto.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["native_quant_rkv_proto"]["prototype_backend"] == "triton_int8_fused_rkv_gemv"
    assert report["native_quant_rkv_proto"]["fused_speedup_vs_separate_int8"] == 1.4545
    assert any("native int8 fused R/K/V quant projection improves separate W8 GEMVs" in item for item in report["next_focus"])


def assert_native_quant_w4_rkv_proto_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "native_quant_w4_rkv_proto",
            "backend": "hf_adapter",
            "prototype_backend": "triton_int4_fused_rkv_gemv",
            "status": "pass",
            "quantization": "int4_rowwise_fused_rkv",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "batch_size": 1,
            "hidden_size": 768,
            "layers": [0],
            "block_m": 16,
            "block_k": 64,
            "steps": 128,
            "avg_fp16_current_ms": 0.09,
            "avg_separate_int4_ms": 0.16,
            "avg_fused_int4_ms": 0.11,
            "fused_speedup_vs_fp16": 0.8182,
            "fused_speedup_vs_separate_int4": 1.4545,
            "separate_speedup_vs_fp16": 0.5625,
            "max_abs_diff_fp16_vs_fused": 0.4,
            "max_abs_diff_separate_vs_fused": 0.001,
            "min_cosine_fp16_vs_fused": 0.98,
            "min_cosine_separate_vs_fused": 1.0,
            "sample_fp16_weight_mb": 3.375,
            "sample_int4_weight_mb": 0.85254,
            "sample_footprint_ratio": 0.2526,
            "layer_rows": [
                {
                    "layer_idx": 0,
                    "fp16_current_ms": 0.09,
                    "separate_int4_ms": 0.16,
                    "fused_int4_ms": 0.11,
                    "fused_speedup_vs_separate_int4": 1.4545,
                },
            ],
        }
    ]
    path = tmpdir / "native_quant_w4_rkv_proto.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["native_quant_w4_rkv_proto"]["prototype_backend"] == "triton_int4_fused_rkv_gemv"
    assert report["native_quant_w4_rkv_proto"]["fused_speedup_vs_separate_int4"] == 1.4545
    assert any("native int4 fused R/K/V quant projection improves separate W4 GEMVs" in item for item in report["next_focus"])


def assert_quantization_model_sweep_does_not_override_canonical(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "quantization",
            "backend": "hf_adapter",
            "model_size_label": "0.1b",
            "model_name": "rwkv7-g1d-0.1b-hf",
            "quantization": "none",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "decode_tokps": 200.0,
            "model_footprint_mb": 400.0,
            "peak_vram_mb": 600.0,
        },
        {
            "axis": "quantization",
            "backend": "hf_adapter",
            "model_size_label": "0.1b",
            "model_name": "rwkv7-g1d-0.1b-hf",
            "quantization": "4bit",
            "quant_skip_policy": "memory",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "decode_tokps": 40.0,
            "model_footprint_mb": 240.0,
            "peak_vram_mb": 300.0,
        },
        {
            "axis": "quantization",
            "backend": "hf_adapter",
            "model_size_label": "0.4b",
            "model_name": "rwkv7-g1d-0.4b-hf",
            "quantization": "none",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "decode_tokps": 100.0,
            "model_footprint_mb": 900.0,
            "peak_vram_mb": 1100.0,
        },
        {
            "axis": "quantization",
            "backend": "hf_adapter",
            "model_size_label": "0.4b",
            "model_name": "rwkv7-g1d-0.4b-hf",
            "quantization": "4bit",
            "quant_skip_policy": "memory",
            "status": "pass",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "decode_tokps": 25.0,
            "model_footprint_mb": 500.0,
            "peak_vram_mb": 700.0,
        },
    ]
    path = tmpdir / "quant_model_sweep.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    canonical = {row["quantization"]: row for row in report["quantization"]}
    assert canonical["none"]["model_name"] == "rwkv7-g1d-0.1b-hf"
    assert canonical["4bit"]["model_name"] == "rwkv7-g1d-0.1b-hf"
    sweep = {
        (row.get("model_size_label"), row.get("quantization")): row
        for row in report["quantization_model_sweep"]
    }
    assert sweep[("0.4b", "4bit")]["model_name"] == "rwkv7-g1d-0.4b-hf"
    assert any("0.4b quantization sweep rows pass" in item for item in report["next_focus"])


def assert_native_model_smoke_is_reported(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "native_model_smoke",
            "backend": "hf_native_model",
            "status": "pass",
            "dtype": "fp32",
            "device": "Tesla V100-PCIE-32GB",
            "model_name": "rwkv7-g1d-0.1b-hf",
            "prompt_count": 3,
            "forward_min_cos": 1.0,
            "forward_max_abs": 0.000038,
            "forward_argmax_match": 3,
            "forward_argmax_total": 3,
            "batch_size": 3,
            "batch_prompt_tokens": 16,
            "batch_forward_min_cos": 0.999999,
            "batch_forward_max_abs": 0.000027,
            "batch_forward_argmax_match": 3,
            "batch_forward_argmax_total": 3,
            "batch_decode_max_abs": 0.000019,
            "batch_decode_argmax_match": 3,
            "batch_decode_argmax_total": 3,
            "batch_cache_shape_ok": True,
            "native_decode_backend": "native_jit",
            "generate_tokens": 16,
            "generate_token_match": 16,
            "generate_token_total": 16,
            "incremental_cache": True,
        }
    ]
    path = tmpdir / "native_model_results.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["native_model_smoke"]["status"] == "pass"
    assert report["native_model_smoke"]["native_decode_backend"] == "native_jit"
    assert report["native_model_smoke"]["generate_token_match"] == 16
    assert any("experimental native_model smoke passes" in item for item in report["next_focus"])


def assert_deepspeed_smoke_survives_inference_dtype_filter(tmpdir: Path) -> None:
    rows = [
        {
            "axis": "deepspeed_training_smoke",
            "backend": "hf_adapter",
            "trainer_backend": f"trainer_zero{stage}",
            "zero_stage": stage,
            "status": "pass",
            "dtype": "fp32",
            "train_dtype": "fp32",
            "device": "Tesla V100-PCIE-32GB",
            "cuda_device_count": 2,
            "attn_mode": "fused_recurrent",
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 1,
            "max_steps": 1,
            "deepspeed_config": f"configs/deepspeed/zero{stage}.json",
            "train_loss": 1.0,
            "train_runtime_s": 1.0,
            "train_samples_per_second": 1.0,
            "train_steps_per_second": 1.0,
            "max_trainable_delta": 1e-4,
        }
        for stage in (2, 3)
    ]
    path = tmpdir / "deepspeed_results.jsonl"
    write_jsonl(path, rows)
    analyzed = subprocess.run(
        [
            sys.executable,
            "bench/analyze_results.py",
            "--results",
            str(path),
            "--device",
            "V100",
            "--dtype",
            "fp16",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    report = json.loads(analyzed.stdout)
    stages = {int(row["zero_stage"]) for row in report["deepspeed_training_smoke"]}
    assert stages == {2, 3}
    assert all(row["status"] == "pass" for row in report["deepspeed_training_smoke"])
    assert any("DeepSpeed ZeRO smoke passes for stages [2, 3]" in item for item in report["next_focus"])


def main() -> int:
    rows = [
        {
            "axis": "speed_mem",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "hf_decode_api": "rwkv7_forward_token",
            "decode_tokps": 60.0,
        },
        {
            "axis": "speed_mem",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "hf_decode_api": "rwkv7_forward_token",
            "fast_token_layout": "2d",
            "decode_tokps": 66.0,
        },
        {
            "axis": "decode_micro",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "fast_decode_api_name": "rwkv7_forward_token",
            "fast_decode_fixed": {"tokps": 59.0},
        },
        {
            "axis": "decode_micro",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "fast_decode_api_name": "rwkv7_forward_token",
            "fast_token_layout": "2d",
            "fast_decode_fixed": {"tokps": 64.9},
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        path = tmpdir / "results.jsonl"
        write_jsonl(path, rows)
        loaded = load_rows(path)
        passed = subprocess.run(
            [
                sys.executable,
                "bench/compare_fast_token_layouts.py",
                "--results",
                str(path),
                "--device",
                "V100",
                "--dtype",
                "fp16",
                "--require-candidate",
                "--min-speedup",
                "1.0",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        assert passed.returncode == 0, passed.stdout + passed.stderr

        missing_path = Path(td) / "missing.jsonl"
        write_jsonl(missing_path, rows[:1])
        failed = subprocess.run(
            [
                sys.executable,
                "bench/compare_fast_token_layouts.py",
                "--results",
                str(missing_path),
                "--device",
                "V100",
                "--dtype",
                "fp16",
                "--require-candidate",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        assert failed.returncode != 0
        assert "candidate layout rows missing" in failed.stdout
        assert_training_smoke_survives_inference_dtype_filter(tmpdir)
        assert_albatross_rows_are_parsed_and_compared(tmpdir)
        assert_quantization_best_variants_are_reported(tmpdir)
        assert_fused_backend_targets_are_reported(tmpdir)
        assert_projection_kernel_plan_is_reported(tmpdir)
        assert_fused_projection_proto_is_reported(tmpdir)
        assert_fused_wa_lora_proto_is_reported(tmpdir)
        assert_fused_wag_lora_proto_is_reported(tmpdir)
        assert_fused_rkv_wag_projection_proto_is_reported(tmpdir)
        assert_fused_shift_mix_proto_is_reported(tmpdir)
        assert_fused_recurrent_proto_is_reported(tmpdir)
        assert_native_graph_fused_recurrent_is_reported(tmpdir)
        assert_native_quant_gemv_proto_is_reported(tmpdir)
        assert_native_quant_w4_gemv_proto_is_reported(tmpdir)
        assert_native_quant_rkv_proto_is_reported(tmpdir)
        assert_native_quant_w4_rkv_proto_is_reported(tmpdir)
        assert_quantization_model_sweep_does_not_override_canonical(tmpdir)
        assert_native_model_smoke_is_reported(tmpdir)
        assert_deepspeed_smoke_survives_inference_dtype_filter(tmpdir)
    args = argparse.Namespace(device="V100", dtype="fp16")
    speeds = latest_by_layout(fast_speed_rows(loaded, args))
    micros = latest_by_layout(fast_micro_rows(loaded, args))
    assert speeds["3d"]["_lineno"] == 1
    assert speeds["2d"]["_lineno"] == 2
    assert round(ratio(num(speeds["2d"], "decode_tokps"), num(speeds["3d"], "decode_tokps")), 4) == 1.1
    assert round(ratio(nested_num(micros["2d"], "fast_decode_fixed", "tokps"), nested_num(micros["3d"], "fast_decode_fixed", "tokps")), 4) == 1.1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
