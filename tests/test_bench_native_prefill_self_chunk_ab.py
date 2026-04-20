from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from bench.bench_native_prefill_self_chunk_ab import route_environment


def test_route_environment_keeps_control_off() -> None:
    env = route_environment(
        candidate=False,
        chunk_size=16,
        stacked_rkv=True,
        hidden_size=2048,
        num_layers=24,
        batch_size=1,
        prompt_tokens=2048,
    )
    assert env["RWKV7_NATIVE_PREFILL_SELF_CHUNK"] == "0"
    assert env["RWKV7_NATIVE_PREFILL_STACKED_RKV"] == "0"


def test_route_environment_selects_exact_candidate_shape() -> None:
    env = route_environment(
        candidate=True,
        chunk_size=16,
        stacked_rkv=True,
        hidden_size=2048,
        num_layers=24,
        batch_size=1,
        prompt_tokens=2048,
    )
    assert env["RWKV7_NATIVE_PREFILL_SELF_CHUNK"] == "1"
    assert env["RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE"] == "16"
    assert env["RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BV"] == "16"
    assert env["RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BC"] == "16"
    assert env["RWKV7_NATIVE_PREFILL_STACKED_RKV"] == "1"
    assert (
        env["RWKV7_NATIVE_PREFILL_SELF_CHUNK_MODEL_SHAPES"]
        == "2048x24x1x2048"
    )
    assert (
        env["RWKV7_NATIVE_PREFILL_STACKED_RKV_MODEL_SHAPES"]
        == "2048x24x1x2048"
    )


def test_direct_script_help_avoids_gpu_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "bench" / "bench_native_prefill_self_chunk_ab.py"),
            "--help",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Same-process correctness/performance A/B" in completed.stdout
