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
