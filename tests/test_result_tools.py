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
