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
    assert "tests/test_apple_silicon_model_training_smoke.py" in text
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
    assert "RafaelUI" in text
    assert "RWKV7_NATIVE_MODEL=1" in text
    assert "rwkv7-g1d-0.4b-hf" in text
    assert "SKIP_TINY=1" in text
    assert "MLX" in text
    assert "Metal" in text


def main() -> int:
    test_fla_is_optional_dependency()
    test_apple_smoke_script_static()
    test_apple_doc_links_entry_points()
    print("APPLE SILICON PACKAGING PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
