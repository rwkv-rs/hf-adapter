#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/_hf_script_common.sh",
    "scripts/run_hf_acceptance.sh",
    "scripts/run_hardware_smoke.sh",
    "scripts/run_hf_training_matrix.sh",
    "scripts/run_zero_training_smoke.sh",
    "scripts/run_math500_acceptance.sh",
    "scripts/run_apple_silicon_smoke.sh",
    "scripts/run_apple_silicon_training_smoke.sh",
    "scripts/run_apple_silicon_trainer_smoke.sh",
    "scripts/run_apple_silicon_model_training_smoke.sh",
    "scripts/run_apple_silicon_model_trl_sft_smoke.sh",
    "scripts/run_apple_silicon_model_rl_smoke.sh",
    "scripts/run_apple_silicon_model_sweep.sh",
    "scripts/run_apple_silicon_quant_smoke.sh",
    "scripts/run_apple_silicon_mlx_smoke.sh",
    "scripts/run_apple_silicon_mlx_model_smoke.sh",
    "scripts/run_apple_silicon_mlx_session_smoke.sh",
    "scripts/run_apple_silicon_mlx_session_batch_smoke.sh",
    "scripts/run_apple_silicon_mlx_generation_sweep.sh",
]
BENCH_RUNNERS = [
    "bench/run_a6000_hf_validation.sh",
    "bench/run_v100_qwen35_speed_matrix.sh",
    "bench/run_3090_qwen35_speed_matrix.sh",
    "bench/run_3090_qwen35_pair.sh",
    "bench/run_4080_qwen35_pair_acceptance.sh",
    "bench/run_t4_hf_validation.sh",
]


def run_bash(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash is required for acceptance script tests")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        # A login shell may reset cwd to $HOME (notably in root-owned GPU
        # containers), which makes every repository-relative script vanish.
        # cwd already provides the isolated repository context we need.
        [bash, "-c", script],
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def assert_ok(proc: subprocess.CompletedProcess[str]) -> None:
    if proc.returncode != 0:
        raise AssertionError(
            f"command failed with {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def test_shell_syntax_and_executable_bits() -> None:
    for rel in SCRIPTS + BENCH_RUNNERS:
        path = ROOT / rel
        assert path.exists(), rel
        assert path.stat().st_mode & stat.S_IXUSR, f"{rel} should be executable"
        proc = run_bash(f"bash -n {rel}")
        assert_ok(proc)


def test_a6000_validation_runner_contract() -> None:
    text = (ROOT / "bench/run_a6000_hf_validation.sh").read_text(encoding="utf-8")
    assert "PYTHON_BIN=\"${PYTHON_BIN:-/home/zhiyuanzhou/draft/venv/bin/python}\"" in text
    assert "MODEL_ROOT=\"${MODEL_ROOT:-/home/zhiyuanzhou/rwkv_models}\"" in text
    assert "A6000_SINGLE_VISIBLE_DEVICES=\"${A6000_SINGLE_VISIBLE_DEVICES:-2}\"" in text
    assert "A6000_MULTI_VISIBLE_DEVICES=\"${A6000_MULTI_VISIBLE_DEVICES:-2,3}\"" in text
    assert "rwkv_vocab_v20230424.txt" in text
    assert "rwkv7-g1d-0.1b-20260129-ctx8192.pth" in text
    assert "rwkv7-g1d-0.4b-20260210-ctx8192.pth" in text
    assert "rwkv7-g1g-1.5b-20260526-ctx8192.pth" in text
    assert "rwkv7-g1g-2.9b-20260526-ctx8192.pth" in text
    assert "rwkv7-g1g-7.2b-20260523-ctx8192.pth" in text
    assert "scripts/convert_rwkv7_to_hf.py" in text
    assert "bench/bench_larger_model_smoke.py" in text
    assert "bench/bench_batch_sweep.py" in text
    assert "bench/bench_quantization.py" in text
    assert "bench/bench_native_mm_quant_decode.py" in text
    assert "scripts/run_hf_training_matrix.sh" in text
    assert "scripts/run_zero_training_smoke.sh" in text
    assert "tests/test_deepspeed_resume_smoke.py" in text
    assert "scripts/print_env.sh" in text
    assert "RESULTS=\"${RESULTS:-bench/results.jsonl}\"" in text
    assert "VALIDATION_MODEL_LABELS=\"${VALIDATION_MODEL_LABELS:-0.4b 1.5b 2.9b 7.2b}\"" in text
    assert "TRAIN_MODEL_LABELS=\"${TRAIN_MODEL_LABELS:-0.4b 1.5b 2.9b}\"" in text
    assert "ZERO_MODEL_LABELS=\"${ZERO_MODEL_LABELS:-0.4b 1.5b 2.9b}\"" in text
    assert "--model-size-label" in text
    assert "a6000_hf_validation_$(date" not in text
    assert "OUT_DIR=" not in text


def test_t4_validation_runner_contract() -> None:
    text = (ROOT / "bench/run_t4_hf_validation.sh").read_text(encoding="utf-8")
    assert 'REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"' in text
    assert 'export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' in text
    assert 'cd "${REPO_ROOT}"' in text
    assert 'MODEL_ROOT="${MODEL_ROOT:-/opt/models}"' in text
    assert 'MATRIX_MODE="${MATRIX_MODE:-short}"' in text
    assert 'FULL_MODEL_LABELS="${FULL_MODEL_LABELS:-0.1b 0.4b 1.5b 2.9b}"' in text
    assert "rwkv7-g1d-0.1b-hf" in text
    assert "rwkv7-g1d-0.4b-hf" in text
    assert "rwkv7-g1g-1.5b-hf" in text
    assert "rwkv7-g1g-2.9b-hf" in text
    assert 'RUN_TRAINER_RESUME="${RUN_TRAINER_RESUME:-0}"' in text
    assert 'RUN_TRL="${RUN_TRL:-0}"' in text
    assert 'TRAIN_DTYPE="${TRAIN_DTYPE:-auto}"' in text
    assert 'TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-auto}"' in text
    assert 'PEFT_MAX_LOGIT_DIFF="${PEFT_MAX_LOGIT_DIFF:-auto}"' in text
    assert 'model_train_dtype()' in text
    assert 'model_train_max_length()' in text
    assert 'model_peft_max_logit_diff()' in text
    assert 'TRAIN_DTYPE="${model_training_dtype}"' in text
    assert 'TRAIN_MAX_LENGTH="${model_training_length}"' in text
    assert 'PEFT_MAX_LOGIT_DIFF="${model_peft_diff}"' in text
    assert '--max-steps 6 --batch-size 1 --length "${TRAIN_MAX_LENGTH}"' in text
    assert '--max-logit-diff "${PEFT_MAX_LOGIT_DIFF}"' in text
    assert 'RUN_DEEPSPEED="${RUN_DEEPSPEED:-0}"' in text
    assert 'RUN_LONG_PREFILL="${RUN_LONG_PREFILL:-0}"' in text
    assert "tests/test_native_trainer_resume_smoke.py" in text
    assert "tests/test_native_sft_smoke.py" in text
    assert "tests/test_native_dpo_smoke.py" in text
    assert "tests/test_native_grpo_smoke.py" in text
    assert "tests/test_deepspeed_training_smoke.py" in text
    assert "tests/test_deepspeed_resume_smoke.py" in text
    assert 'model_batch_sizes "${model_label}"' in text
    assert 'model_prefill_batch_sizes "${model_label}"' in text
    assert 'model_dynamic_batch_size "${model_label}"' in text
    assert 'MODEL_OUT_DIR="${OUT_DIR}/${model_label}"' in text
    assert 'EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-Tesla T4}"' in text
    assert 'EXPECTED_GPU_CAPABILITY="${EXPECTED_GPU_CAPABILITY:-7.5}"' in text
    assert 'TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"' in text
    assert 'TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"' in text
    assert 'ALLOW_GPU_MISMATCH="${ALLOW_GPU_MISMATCH:-0}"' in text
    assert "tests/smoke_hf_generate.py" in text
    assert "tests/test_hf_api_contract.py" in text
    assert "tests/test_batch_cache.py" in text
    assert "tests/test_dynamic_batch_cache.py" in text
    assert "tests/test_chunked_prefill.py" in text
    assert "tests/test_native_trainer_smoke.py" in text
    assert "tests/test_native_peft_save_load_merge.py" in text
    assert "bench/bench_batch_sweep.py" in text
    assert "bench/bench_native_graph_overhead.py" in text
    assert "bench/bench_chunked_prefill.py" in text
    assert "bench/bench_native_graph_fused_output.py" in text
    assert "bench/bench_native_graph_fused_recurrent_output.py" in text
    assert "bench/bench_quantization.py" in text
    assert "bench/bench_native_mm_quant_decode.py" in text
    assert "bench/bench_native_quant_e2e_decode.py" in text
    assert 'BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"' in text
    assert 'PREFILL_BATCH_SIZES="${PREFILL_BATCH_SIZES:-1,2,4,8}"' in text
    assert "RWKV7_NATIVE_MODEL=1" in text
    assert '--device "${EXPECTED_GPU_NAME}"' in text


def test_acceptance_requires_model() -> None:
    proc = run_bash("bash scripts/run_hf_acceptance.sh")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "MODEL is required" in proc.stderr


def test_hardware_wrapper_requires_model() -> None:
    proc = run_bash("bash scripts/run_hardware_smoke.sh")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "MODEL is required" in proc.stderr


def test_apple_silicon_smoke_requires_model() -> None:
    proc = run_bash("bash scripts/run_apple_silicon_smoke.sh")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "MODEL is required" in proc.stderr


def test_apple_silicon_model_training_requires_model() -> None:
    proc = run_bash("bash scripts/run_apple_silicon_model_training_smoke.sh")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "MODEL is required" in proc.stderr


def test_apple_silicon_model_trl_sft_requires_model() -> None:
    proc = run_bash("bash scripts/run_apple_silicon_model_trl_sft_smoke.sh")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "MODEL is required" in proc.stderr


def test_apple_silicon_model_rl_requires_model() -> None:
    proc = run_bash("bash scripts/run_apple_silicon_model_rl_smoke.sh")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "MODEL is required" in proc.stderr


def test_apple_silicon_model_sweep_requires_model() -> None:
    proc = run_bash("bash scripts/run_apple_silicon_model_sweep.sh")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "MODEL is required" in proc.stderr


def test_apple_silicon_mlx_session_requires_model() -> None:
    proc = run_bash("bash scripts/run_apple_silicon_mlx_session_smoke.sh")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "MODEL is required" in proc.stderr


def test_apple_silicon_mlx_session_batch_requires_model() -> None:
    proc = run_bash("bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "MODEL is required" in proc.stderr


def test_apple_silicon_mlx_generation_sweep_requires_model() -> None:
    proc = run_bash("bash scripts/run_apple_silicon_mlx_generation_sweep.sh")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "MODEL is required" in proc.stderr


def test_math500_acceptance_defaults_are_final_benchmark() -> None:
    text = (ROOT / "scripts/run_math500_acceptance.sh").read_text(encoding="utf-8")
    assert 'BSZ="${BSZ:-128}"' in text
    assert 'SEED="${SEED:-43}"' in text
    assert 'DEFER_VERIFICATION="${DEFER_VERIFICATION:-1}"' in text
    assert 'SUMMARY_SPEED_TIMING="${SUMMARY_SPEED_TIMING:-generation}"' in text
    assert 'DEFER_TEXT_DECODE="${DEFER_TEXT_DECODE:-1}"' in text
    assert 'ACCEPTANCE_MIN_PASS_AT_ROLLOUT="${ACCEPTANCE_MIN_PASS_AT_ROLLOUT:-0.370}"' in text
    assert 'ACCEPTANCE_MIN_SUMMARY_SPEED_RATIO="${ACCEPTANCE_MIN_SUMMARY_SPEED_RATIO:-2.0}"' in text


def test_blackwell_matrix_supports_paired_baselines() -> None:
    matrix = (ROOT / "bench/run_blackwell_quant_matrix.py").read_text(encoding="utf-8")
    decode = (ROOT / "bench/bench_native_quant_e2e_decode.py").read_text(encoding="utf-8")
    assert '"--paired-baseline"' in matrix
    assert 'cmd.append("--paired-baseline")' in matrix
    assert '"--timing-repeats"' in matrix
    assert '"--paired-baseline"' in decode
    assert '"--timing-repeats"' in decode
    assert '"--quantize-before-device"' in decode
    assert "quantization != \"none\" and args.paired_baseline" in decode


def test_official_prefill_matrix_forwards_low_memory_runtime() -> None:
    matrix = (ROOT / "bench/run_official_native_prefill_matrix.py").read_text(
        encoding="utf-8"
    )
    assert '"--official-emb"' in matrix
    assert '"--official-lowrank-weight"' in matrix
    assert '"--official-orig-linear-groups"' in matrix
    assert '"--native-torch-extensions-dir"' in matrix
    assert '"--official-torch-extensions-dir"' in matrix
    assert '"--official-self-envelope-dir"' in matrix
    assert 'env_overrides=' in matrix


def test_converter_exposes_low_memory_path() -> None:
    converter = (ROOT / "scripts/convert_rwkv7_to_hf.py").read_text(encoding="utf-8")
    assert '"--low-memory"' in converter
    assert 'with torch.device("meta")' in converter
    assert "save_torch_state_dict(" in converter


def test_common_pythonpath_separator_linux() -> None:
    proc = run_bash(
        "export PYTHONPATH=/tmp/existing; source scripts/_hf_script_common.sh >/dev/null; "
        "test \"$PYTHONPATH\" = \"$PWD:/tmp/existing\""
    )
    assert_ok(proc)


def test_common_pythonpath_separator_windows_msys() -> None:
    proc = run_bash(
        "export OSTYPE=msys MSYSTEM=MINGW64 PYTHONPATH='D:/existing'; "
        "source scripts/_hf_script_common.sh >/dev/null; "
        "test \"$PYTHONPATH\" = \"$PWD;D:/existing\""
    )
    assert_ok(proc)


def main() -> int:
    test_shell_syntax_and_executable_bits()
    test_acceptance_requires_model()
    test_hardware_wrapper_requires_model()
    test_apple_silicon_smoke_requires_model()
    test_apple_silicon_model_training_requires_model()
    test_apple_silicon_model_trl_sft_requires_model()
    test_apple_silicon_model_rl_requires_model()
    test_apple_silicon_model_sweep_requires_model()
    test_apple_silicon_mlx_session_requires_model()
    test_apple_silicon_mlx_session_batch_requires_model()
    test_apple_silicon_mlx_generation_sweep_requires_model()
    test_math500_acceptance_defaults_are_final_benchmark()
    test_blackwell_matrix_supports_paired_baselines()
    test_converter_exposes_low_memory_path()
    test_common_pythonpath_separator_linux()
    test_common_pythonpath_separator_windows_msys()
    print("ACCEPTANCE SCRIPTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
