from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from examples.cpu_tiny_demo import build_parser, run_demo


def test_cpu_tiny_demo_defaults_are_small_and_safe() -> None:
    args = build_parser().parse_args([])
    assert args.mode == "all"
    assert args.steps == 12
    assert args.batch_size == 4
    assert args.length == 16
    assert args.threads <= 4


def test_cpu_tiny_demo_trains_generates_and_round_trips(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--steps",
            "4",
            "--threads",
            "1",
            "--output-dir",
            str(tmp_path / "checkpoint"),
        ]
    )
    result = run_demo(args)

    assert result["status"] == "pass"
    assert result["backend"] == "native_eager"
    assert result["device"] == "cpu"
    assert result["training"]["final_loss"] < result["training"]["initial_loss"]
    assert result["training"]["max_grad_l1"] > 0
    assert result["training"]["parameter_changed_l1"] > 0
    assert len(result["inference_before_training"]["generated_token_ids"]) == 6
    assert len(result["inference_after_training"]["generated_token_ids"]) == 6
    assert result["save_reload"]["max_abs"] == 0.0
    assert (tmp_path / "checkpoint" / "config.json").exists()
    assert (tmp_path / "checkpoint" / "model.safetensors").exists()


def test_cpu_tiny_demo_restores_backend_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RWKV7_NATIVE_MODEL_BACKEND", "native_jit")
    args = build_parser().parse_args(
        [
            "--mode",
            "infer",
            "--threads",
            "1",
            "--max-new-tokens",
            "1",
            "--output-dir",
            str(tmp_path / "checkpoint"),
        ]
    )
    result = run_demo(args)

    assert result["backend"] == "native_eager"
    assert os.environ["RWKV7_NATIVE_MODEL_BACKEND"] == "native_jit"


def test_cpu_tiny_demo_executable_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "cpu_tiny_demo.py"),
            "--mode",
            "all",
            "--steps",
            "2",
            "--threads",
            "1",
            "--max-new-tokens",
            "2",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    for marker in (
        "CPU INFERENCE PASS",
        "CPU TRAINING PASS",
        "CPU SAVE/RELOAD PASS",
        "CPU DEMO PASS",
        "CPU_DEMO_RESULT=",
    ):
        assert marker in proc.stdout
