#!/usr/bin/env python3
# coding=utf-8
"""Static guards for Apple Silicon / no-FLA packaging and docs."""
from __future__ import annotations

import stat
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fla_is_optional_dependency() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    deps_match = re.search(r"(?ms)^dependencies\s*=\s*\[(.*?)^\]", text)
    assert deps_match is not None
    deps_block = deps_match.group(1)
    assert "flash-linear-attention" not in deps_block
    assert "fla = [\"flash-linear-attention\"]" in text
    assert "cuda = [\"flash-linear-attention\"" in text


def test_mlx_extra_is_apple_optional_dependency() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    deps_match = re.search(r"(?ms)^dependencies\s*=\s*\[(.*?)^\]", text)
    assert deps_match is not None
    deps_block = deps_match.group(1)
    assert "mlx" not in deps_block
    assert 'mlx = ["mlx; platform_system == \'Darwin\' and platform_machine == \'arm64\'"]' in text


def test_apple_smoke_script_static() -> None:
    script = ROOT / "scripts/run_apple_silicon_smoke.sh"
    assert script.exists()
    assert script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=True)
    text = script.read_text(encoding="utf-8")
    assert "RWKV7_NATIVE_MODEL" in text
    assert "PYTORCH_ENABLE_MPS_FALLBACK" in text
    assert "tests/test_apple_silicon_smoke.py" in text
    assert "MODEL_SIZE_LABEL" in text
    assert "SKIP_TINY" in text
    assert "--model-size-label" in text
    assert "--skip-tiny" in text
    train_script = ROOT / "scripts/run_apple_silicon_training_smoke.sh"
    assert train_script.exists()
    assert train_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(train_script)], cwd=ROOT, check=True)
    trainer_script = ROOT / "scripts/run_apple_silicon_trainer_smoke.sh"
    assert trainer_script.exists()
    assert trainer_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(trainer_script)], cwd=ROOT, check=True)
    model_train_script = ROOT / "scripts/run_apple_silicon_model_training_smoke.sh"
    assert model_train_script.exists()
    assert model_train_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(model_train_script)], cwd=ROOT, check=True)
    trl_sft_script = ROOT / "scripts/run_apple_silicon_model_trl_sft_smoke.sh"
    assert trl_sft_script.exists()
    assert trl_sft_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(trl_sft_script)], cwd=ROOT, check=True)
    rl_script = ROOT / "scripts/run_apple_silicon_model_rl_smoke.sh"
    assert rl_script.exists()
    assert rl_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(rl_script)], cwd=ROOT, check=True)
    sweep_script = ROOT / "scripts/run_apple_silicon_model_sweep.sh"
    assert sweep_script.exists()
    assert sweep_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(sweep_script)], cwd=ROOT, check=True)
    quant_script = ROOT / "scripts/run_apple_silicon_quant_smoke.sh"
    assert quant_script.exists()
    assert quant_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(quant_script)], cwd=ROOT, check=True)
    mlx_script = ROOT / "scripts/run_apple_silicon_mlx_smoke.sh"
    assert mlx_script.exists()
    assert mlx_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(mlx_script)], cwd=ROOT, check=True)
    mlx_model_script = ROOT / "scripts/run_apple_silicon_mlx_model_smoke.sh"
    assert mlx_model_script.exists()
    assert mlx_model_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(mlx_model_script)], cwd=ROOT, check=True)
    mlx_session_wrapper = ROOT / "scripts/run_apple_silicon_mlx_session_smoke.sh"
    assert mlx_session_wrapper.exists()
    assert mlx_session_wrapper.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(mlx_session_wrapper)], cwd=ROOT, check=True)
    mlx_session_wrapper_text = mlx_session_wrapper.read_text(encoding="utf-8")
    assert "QUANT_BACKEND" in mlx_session_wrapper_text
    assert "WKV_BACKEND" in mlx_session_wrapper_text
    assert "--wkv-backend" in mlx_session_wrapper_text
    mlx_session_batch_wrapper = ROOT / "scripts/run_apple_silicon_mlx_session_batch_smoke.sh"
    assert mlx_session_batch_wrapper.exists()
    assert mlx_session_batch_wrapper.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(mlx_session_batch_wrapper)], cwd=ROOT, check=True)
    mlx_session_batch_text = mlx_session_batch_wrapper.read_text(encoding="utf-8")
    assert "SESSION_COUNT" in mlx_session_batch_text
    assert "PROMPTS_FILE" in mlx_session_batch_text
    assert "EXTRA_PROMPTS" in mlx_session_batch_text
    assert "PROMPT_H" in mlx_session_batch_text
    assert "QUANT_BACKEND" in mlx_session_batch_text
    assert "WKV_BACKEND" in mlx_session_batch_text
    assert "SESSION_BACKEND" in mlx_session_batch_text
    assert "COMPARE_SESSION_BACKEND" in mlx_session_batch_text
    assert "COMPARE_ONLY" in mlx_session_batch_text
    assert "REQUIRE_SESSION_BACKEND_MATCH" in mlx_session_batch_text
    assert "TRACE_MISMATCH_LOGITS" in mlx_session_batch_text
    assert "MISMATCH_TOPK" in mlx_session_batch_text
    assert "--wkv-backend" in mlx_session_batch_text
    assert "--session-backend" in mlx_session_batch_text
    assert "--compare-session-backend" in mlx_session_batch_text
    assert "--compare-only" in mlx_session_batch_text
    assert "--require-session-backend-match" in mlx_session_batch_text
    assert "--trace-mismatch-logits" in mlx_session_batch_text
    assert "--mismatch-topk" in mlx_session_batch_text
    mlx_session_batch_wrapper = ROOT / "scripts/run_apple_silicon_mlx_session_batch_smoke.sh"
    assert mlx_session_batch_wrapper.exists()
    assert mlx_session_batch_wrapper.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(mlx_session_batch_wrapper)], cwd=ROOT, check=True)
    mlx_sweep_wrapper = ROOT / "scripts/run_apple_silicon_mlx_generation_sweep.sh"
    assert mlx_sweep_wrapper.exists()
    assert mlx_sweep_wrapper.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(mlx_sweep_wrapper)], cwd=ROOT, check=True)
    qwen_acceptance_wrapper = ROOT / "scripts/run_qwen35_apple_acceptance.sh"
    assert qwen_acceptance_wrapper.exists()
    assert qwen_acceptance_wrapper.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(qwen_acceptance_wrapper)], cwd=ROOT, check=True)
    qwen_acceptance_text = qwen_acceptance_wrapper.read_text(encoding="utf-8")
    assert "bench/run_qwen35_apple_baseline.py" in qwen_acceptance_text
    assert "bench/compare_qwen35_apple_baseline.py" in qwen_acceptance_text
    assert "COMPARE_DIAGNOSTICS" in qwen_acceptance_text
    assert "--diagnostics" in qwen_acceptance_text
    assert "bench/run_coreml_apple_baseline.py" in qwen_acceptance_text or "COREML_RUNTIME_MANIFESTS" in qwen_acceptance_text
    assert "bench/score_qwen35_quality.py" in qwen_acceptance_text or "QUALITY_RUBRIC" in qwen_acceptance_text
    assert "scripts/export_rwkv7_coreml.py" in qwen_acceptance_text
    assert "PULL_QWEN" in qwen_acceptance_text
    assert "RWKV_MLX_MODELS" in qwen_acceptance_text
    assert "RWKV_PREFILL_EVAL_INTERVAL" in qwen_acceptance_text
    assert "COREML_EXPORT_MODELS" in qwen_acceptance_text
    mlx_prefill_eval_bench = ROOT / "scripts/mlx_prefill_eval_interval_bench.py"
    assert mlx_prefill_eval_bench.exists()
    assert mlx_prefill_eval_bench.stat().st_mode & stat.S_IXUSR
    mlx_dplr_prefill_bench = ROOT / "scripts/mlx_dplr_prefill_bench.py"
    assert mlx_dplr_prefill_bench.exists()
    assert mlx_dplr_prefill_bench.stat().st_mode & stat.S_IXUSR
    convert_mlx_script = ROOT / "scripts/convert_hf_to_mlx.py"
    assert convert_mlx_script.exists()
    assert convert_mlx_script.stat().st_mode & stat.S_IXUSR
    mlx_generate_script = ROOT / "scripts/mlx_generate.py"
    assert mlx_generate_script.exists()
    assert mlx_generate_script.stat().st_mode & stat.S_IXUSR
    mlx_session_script = ROOT / "scripts/mlx_session_smoke.py"
    assert mlx_session_script.exists()
    assert mlx_session_script.stat().st_mode & stat.S_IXUSR
    mlx_session_text = mlx_session_script.read_text(encoding="utf-8")
    assert 'choices=["affine", "reference", "metal", "auto"]' in mlx_session_text
    assert 'choices=["reference", "metal", "auto"]' in mlx_session_text
    assert "--quant-rkv-min-params" in mlx_session_text
    mlx_session_batch_script = ROOT / "scripts/mlx_session_batch_smoke.py"
    assert mlx_session_batch_script.exists()
    assert mlx_session_batch_script.stat().st_mode & stat.S_IXUSR
    mlx_session_batch_text = mlx_session_batch_script.read_text(encoding="utf-8")
    assert 'choices=["affine", "reference", "metal", "auto"]' in mlx_session_batch_text
    assert 'choices=["reference", "metal", "auto"]' in mlx_session_batch_text
    assert 'choices=["sequential", "batched", "batched_stable", "auto"]' in mlx_session_batch_text
    assert 'choices=["none", "sequential", "batched", "batched_stable", "auto"]' in mlx_session_batch_text
    assert '"axis": "mlx_session_batch_backend_compare"' in mlx_session_batch_text
    assert '"backend_compare_status": "match" if strict_match else "mismatch"' in mlx_session_batch_text
    assert '"strict_match": bool(strict_match)' in mlx_session_batch_text
    assert '"mismatch_logit_trace": mismatch_logit_trace' in mlx_session_batch_text
    assert "trace_backend_mismatch_logits" in mlx_session_batch_text
    assert "all_left_one_shot_match" in mlx_session_batch_text
    assert "all_right_one_shot_match" in mlx_session_batch_text
    assert "if require_match and not strict_match" in mlx_session_batch_text
    assert '"quantization": args.quantization' in mlx_session_batch_text
    assert "--quant-rkv-min-params" in mlx_session_batch_text
    mlx_generation_sweep_text = (ROOT / "scripts/mlx_generation_sweep.py").read_text(encoding="utf-8")
    assert '"quantized_linear_last_backend_counts": telemetry.get("quantized_linear_last_backend_counts")' in mlx_generation_sweep_text
    assert '"quant_backend": args.quant_backend' in mlx_session_batch_text
    assert '"quantized_linear_last_backend_counts": telemetry.get("quantized_linear_last_backend_counts")' in mlx_session_batch_text
    assert "model_quant_runtime_telemetry" in mlx_generation_sweep_text
    assert "group_rkv_quant_projection_mode" in mlx_generation_sweep_text
    assert "group_rkv_quant_projection_counts" in mlx_generation_sweep_text
    assert "model_quant_runtime_telemetry" in mlx_session_batch_text
    assert "group_rkv_quant_projection_mode" in mlx_session_batch_text
    assert "group_rkv_quant_projection_counts" in mlx_session_batch_text
    assert '"session_backend": args.session_backend' in mlx_session_batch_text
    assert '"min_round_decode_tok_s": min_round_decode_tok_s(rows)' in mlx_session_batch_text
    assert '"round_backend_reasons": sorted(' in mlx_session_batch_text
    model_text = (ROOT / "rwkv7_hf/mlx_model.py").read_text(encoding="utf-8")
    quant_text = (ROOT / "rwkv7_hf/mlx_quant.py").read_text(encoding="utf-8")
    assert "auto_metal_max_rows" in quant_text
    assert "mm8_group_matmul_metal" in quant_text
    assert "mm4_group_matmul_metal" in quant_text
    assert "mm8_group_matmul_metal_inputs" in quant_text
    assert "mm4_group_matmul_metal_inputs" in quant_text
    assert "mm8_triple_matmul_metal_inputs" in quant_text
    assert "mm4_triple_matmul_metal_inputs" in quant_text
    assert "pack_mlx_mm8_group" in quant_text
    assert 'RWKV7_MLX_QUANT_AUTO_W4_METAL_MAX_ROWS", 4096' in quant_text
    assert "@lru_cache(maxsize=1)" in quant_text
    assert "batched_stable" in model_text
    assert "equal_positive_round_stable_argmax_tol" in model_text
    assert "auto_mm8_metal_batch_exactness_guard" in model_text
    assert "RWKV7_MLX_SESSION_AUTO_W8_STABLE" in model_text
    assert "RWKV7_MLX_SESSION_AUTO_W4_STABLE" in model_text
    assert "auto_mm4_metal_batch_exactness_guard" in model_text
    assert "RWKV7_MLX_SESSION_STABLE_ARGMAX_TOLERANCE" in model_text
    assert "--stable-argmax-tolerance" in mlx_session_batch_text
    assert "--stable-argmax-mode" in mlx_session_batch_text
    assert "RWKV7_MLX_SESSION_STABLE_ARGMAX_MODE" in model_text
    assert "round_stable_repair_counts" in model_text
    assert '"max_round_stable_repair_count"' in mlx_session_batch_text
    assert "RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION" in model_text
    assert "RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION_MODE" in model_text
    assert "group_rkv_quant_projection_mode" in model_text
    assert "group_rkv_quant_projection_counts" in model_text
    assert "quantized_linear_rkv_min_params" in model_text
    assert "RWKV7_MLX_STEP_EVAL_INTERVAL" in model_text
    assert "step_eval_interval" in mlx_generation_sweep_text
    assert "auto_metal_max_rows" in model_text
    quant_bench_script = ROOT / "scripts/mlx_quant_projection_bench.py"
    assert quant_bench_script.exists()
    assert quant_bench_script.stat().st_mode & stat.S_IXUSR
    coreml_export_script = ROOT / "scripts/export_rwkv7_coreml.py"
    assert coreml_export_script.exists()
    assert coreml_export_script.stat().st_mode & stat.S_IXUSR
    coreml_export_text = coreml_export_script.read_text(encoding="utf-8")
    assert "rwkv7_coreml_export" in coreml_export_text
    assert "full-logits" in coreml_export_text
    assert "wkv-coreml" in coreml_export_text
    assert "lut4" in coreml_export_text
    quant_bench_text = quant_bench_script.read_text(encoding="utf-8")
    assert "mlx_quant_projection_bench" in quant_bench_text
    assert "speedup_vs_dense" in quant_bench_text
    assert "max_abs_vs_quant_reference" in quant_bench_text
    assert "mlx_quant_group_projection_bench" in quant_bench_text
    assert "speedup_vs_separate_metal" in quant_bench_text
    assert "_uses_w8_metal_projection" in model_text
    assert "round_backend_reasons" in model_text
    mlx_session_batch_script = ROOT / "scripts/mlx_session_batch_smoke.py"
    assert mlx_session_batch_script.exists()
    assert mlx_session_batch_script.stat().st_mode & stat.S_IXUSR
    mlx_sweep_script = ROOT / "scripts/mlx_generation_sweep.py"
    assert mlx_sweep_script.exists()
    assert mlx_sweep_script.stat().st_mode & stat.S_IXUSR


def test_apple_doc_links_entry_points() -> None:
    doc = ROOT / "docs/hardware/APPLE_SILICON.md"
    text = doc.read_text(encoding="utf-8")
    assert "scripts/run_apple_silicon_smoke.sh" in text
    assert "tests/test_apple_silicon_smoke.py" in text
    assert "scripts/run_apple_silicon_trainer_smoke.sh" in text
    assert "tests/test_apple_silicon_trainer_smoke.py" in text
    assert "scripts/run_apple_silicon_model_training_smoke.sh" in text
    assert "scripts/run_apple_silicon_model_trl_sft_smoke.sh" in text
    assert "scripts/run_apple_silicon_model_rl_smoke.sh" in text
    assert "scripts/run_apple_silicon_model_sweep.sh" in text
    assert "scripts/run_apple_silicon_quant_smoke.sh" in text
    assert "scripts/run_apple_silicon_mlx_smoke.sh" in text
    assert "scripts/run_apple_silicon_mlx_model_smoke.sh" in text
    assert "scripts/run_apple_silicon_mlx_session_smoke.sh" in text
    assert "scripts/run_apple_silicon_mlx_session_batch_smoke.sh" in text
    assert "scripts/run_apple_silicon_mlx_generation_sweep.sh" in text
    assert "scripts/convert_hf_to_mlx.py" in text
    assert "scripts/mlx_generate.py" in text
    assert "scripts/mlx_session_smoke.py" in text
    assert "scripts/mlx_session_batch_smoke.py" in text
    assert "scripts/mlx_generation_sweep.py" in text
    assert "scripts/mlx_quant_projection_bench.py" in text
    assert "scripts/export_rwkv7_coreml.py" in text
    assert "bench/run_coreml_apple_baseline.py" in text
    assert "bench/score_qwen35_quality.py" in text
    assert "qwen35_apple_baseline_gap_diagnostic" in text
    assert "scripts/run_qwen35_apple_acceptance.sh" in text
    assert "tests/test_apple_silicon_model_training_smoke.py" in text
    assert "tests/test_apple_silicon_model_sweep.py" in text
    assert "tests/test_apple_silicon_quant_smoke.py" in text
    assert "tests/test_apple_silicon_mlx_smoke.py" in text
    assert "tests/test_apple_silicon_mlx_model_smoke.py" in text
    train_script = ROOT / "scripts/run_apple_silicon_training_smoke.sh"
    assert train_script.exists()
    assert train_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(train_script)], cwd=ROOT, check=True)
    trainer_script = ROOT / "scripts/run_apple_silicon_trainer_smoke.sh"
    assert trainer_script.exists()
    assert trainer_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(trainer_script)], cwd=ROOT, check=True)
    model_train_script = ROOT / "scripts/run_apple_silicon_model_training_smoke.sh"
    assert model_train_script.exists()
    assert model_train_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(model_train_script)], cwd=ROOT, check=True)
    trl_sft_script = ROOT / "scripts/run_apple_silicon_model_trl_sft_smoke.sh"
    assert trl_sft_script.exists()
    assert trl_sft_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(trl_sft_script)], cwd=ROOT, check=True)
    rl_script = ROOT / "scripts/run_apple_silicon_model_rl_smoke.sh"
    assert rl_script.exists()
    assert rl_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(rl_script)], cwd=ROOT, check=True)
    sweep_script = ROOT / "scripts/run_apple_silicon_model_sweep.sh"
    assert sweep_script.exists()
    assert sweep_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(sweep_script)], cwd=ROOT, check=True)
    quant_script = ROOT / "scripts/run_apple_silicon_quant_smoke.sh"
    assert quant_script.exists()
    assert quant_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(quant_script)], cwd=ROOT, check=True)
    mlx_script = ROOT / "scripts/run_apple_silicon_mlx_smoke.sh"
    assert mlx_script.exists()
    assert mlx_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(mlx_script)], cwd=ROOT, check=True)
    mlx_model_script = ROOT / "scripts/run_apple_silicon_mlx_model_smoke.sh"
    assert mlx_model_script.exists()
    assert mlx_model_script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(mlx_model_script)], cwd=ROOT, check=True)
    mlx_session_wrapper = ROOT / "scripts/run_apple_silicon_mlx_session_smoke.sh"
    assert mlx_session_wrapper.exists()
    assert mlx_session_wrapper.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(mlx_session_wrapper)], cwd=ROOT, check=True)
    mlx_sweep_wrapper = ROOT / "scripts/run_apple_silicon_mlx_generation_sweep.sh"
    assert mlx_sweep_wrapper.exists()
    assert mlx_sweep_wrapper.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(mlx_sweep_wrapper)], cwd=ROOT, check=True)
    qwen_acceptance_wrapper = ROOT / "scripts/run_qwen35_apple_acceptance.sh"
    assert qwen_acceptance_wrapper.exists()
    assert qwen_acceptance_wrapper.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(qwen_acceptance_wrapper)], cwd=ROOT, check=True)
    qwen_acceptance_text = qwen_acceptance_wrapper.read_text(encoding="utf-8")
    assert "bench/run_qwen35_apple_baseline.py" in qwen_acceptance_text
    assert "bench/compare_qwen35_apple_baseline.py" in qwen_acceptance_text
    assert "COMPARE_DIAGNOSTICS" in qwen_acceptance_text
    assert "--diagnostics" in qwen_acceptance_text
    assert "bench/run_coreml_apple_baseline.py" in qwen_acceptance_text or "COREML_RUNTIME_MANIFESTS" in qwen_acceptance_text
    assert "bench/score_qwen35_quality.py" in qwen_acceptance_text or "QUALITY_RUBRIC" in qwen_acceptance_text
    assert "scripts/export_rwkv7_coreml.py" in qwen_acceptance_text
    assert "PULL_QWEN" in qwen_acceptance_text
    assert "RWKV_MLX_MODELS" in qwen_acceptance_text
    assert "RWKV_PREFILL_EVAL_INTERVAL" in qwen_acceptance_text
    assert "COREML_EXPORT_MODELS" in qwen_acceptance_text
    mlx_prefill_eval_bench = ROOT / "scripts/mlx_prefill_eval_interval_bench.py"
    assert mlx_prefill_eval_bench.exists()
    assert mlx_prefill_eval_bench.stat().st_mode & stat.S_IXUSR
    mlx_dplr_prefill_bench = ROOT / "scripts/mlx_dplr_prefill_bench.py"
    assert mlx_dplr_prefill_bench.exists()
    assert mlx_dplr_prefill_bench.stat().st_mode & stat.S_IXUSR
    convert_mlx_script = ROOT / "scripts/convert_hf_to_mlx.py"
    assert convert_mlx_script.exists()
    assert convert_mlx_script.stat().st_mode & stat.S_IXUSR
    mlx_generate_script = ROOT / "scripts/mlx_generate.py"
    assert mlx_generate_script.exists()
    assert mlx_generate_script.stat().st_mode & stat.S_IXUSR
    mlx_session_script = ROOT / "scripts/mlx_session_smoke.py"
    assert mlx_session_script.exists()
    assert mlx_session_script.stat().st_mode & stat.S_IXUSR
    mlx_sweep_script = ROOT / "scripts/mlx_generation_sweep.py"
    assert mlx_sweep_script.exists()
    assert mlx_sweep_script.stat().st_mode & stat.S_IXUSR
    assert "RafaelUI" in text
    assert "RWKV7_NATIVE_MODEL=1" in text
    assert "rwkv7-g1d-0.4b-hf" in text
    assert "SKIP_TINY=1" in text
    assert "MLX" in text
    assert "Metal" in text


def main() -> int:
    test_fla_is_optional_dependency()
    test_mlx_extra_is_apple_optional_dependency()
    test_apple_smoke_script_static()
    test_apple_doc_links_entry_points()
    print("APPLE SILICON PACKAGING PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
