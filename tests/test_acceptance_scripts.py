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
]


def run_bash(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash is required for acceptance script tests")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        [bash, "-lc", script],
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
    for rel in SCRIPTS:
        path = ROOT / rel
        assert path.exists(), rel
        assert path.stat().st_mode & stat.S_IXUSR, f"{rel} should be executable"
        proc = run_bash(f"bash -n {rel}")
        assert_ok(proc)


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


def test_math500_acceptance_defaults_are_final_benchmark() -> None:
    text = (ROOT / "scripts/run_math500_acceptance.sh").read_text(encoding="utf-8")
    assert 'BSZ="${BSZ:-128}"' in text
    assert 'SEED="${SEED:-43}"' in text
    assert 'DEFER_VERIFICATION="${DEFER_VERIFICATION:-1}"' in text
    assert 'SUMMARY_SPEED_TIMING="${SUMMARY_SPEED_TIMING:-generation}"' in text
    assert 'DEFER_TEXT_DECODE="${DEFER_TEXT_DECODE:-1}"' in text
    assert 'ACCEPTANCE_MIN_PASS_AT_ROLLOUT="${ACCEPTANCE_MIN_PASS_AT_ROLLOUT:-0.370}"' in text
    assert 'ACCEPTANCE_MIN_SUMMARY_SPEED_RATIO="${ACCEPTANCE_MIN_SUMMARY_SPEED_RATIO:-2.0}"' in text


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
    test_math500_acceptance_defaults_are_final_benchmark()
    test_common_pythonpath_separator_linux()
    test_common_pythonpath_separator_windows_msys()
    print("ACCEPTANCE SCRIPTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
