from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from bench.bench_native_prefill_accum_ab import (
    _prefill_call,
    mode_flags,
    model_shape_spec,
    route_effective_matches,
    sweep_orders,
)


class _PrefillRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rwkv7_prefill_chunks(self, ids, *, chunk_size: int, logits_to_keep: int):
        self.calls.append(("chunks", chunk_size))
        assert logits_to_keep == 1
        return "chunked"

    def rwkv7_prefill_native(self, ids, *, logits_to_keep: int, return_dict: bool):
        self.calls.append(("native", int(ids.shape[1])))
        assert logits_to_keep == 1
        assert return_dict
        return "native"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("off", ("0", "0")),
        ("global", ("1", "0")),
        ("block", ("0", "1")),
    ],
)
def test_mode_flags(mode: str, expected: tuple[str, str]) -> None:
    assert mode_flags(mode) == expected


def test_mode_flags_reject_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported accumulation mode"):
        mode_flags("unknown")


def test_sweep_orders_support_forward_reverse_and_both() -> None:
    assert sweep_orders("forward") == (("off", "global", "block"),)
    assert sweep_orders("reverse") == (("block", "global", "off"),)
    assert sweep_orders("both") == (
        ("off", "global", "block"),
        ("block", "global", "off"),
    )


def test_route_effective_match_is_exact() -> None:
    assert route_effective_matches("off", False, False)
    assert route_effective_matches("global", True, False)
    assert route_effective_matches("block", False, True)
    assert not route_effective_matches("global", True, True)
    assert not route_effective_matches("block", False, False)


def test_model_shape_spec_is_deterministic() -> None:
    assert model_shape_spec(1024, 24, [1, 8], [128, 512]) == (
        "1024x24x1x128 1024x24x1x512 "
        "1024x24x8x128 1024x24x8x512"
    )


def test_prefill_call_matches_final_matrix_chunking() -> None:
    import torch

    model = _PrefillRecorder()
    ids = torch.zeros((1, 2048), dtype=torch.long)
    assert _prefill_call(model, ids, 512) == "chunked"
    assert _prefill_call(model, ids[:, :128], 512) == "native"
    assert model.calls == [("chunks", 512), ("native", 128)]


def test_direct_script_entrypoint_resolves_bench_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "bench" / "bench_native_prefill_accum_ab.py"),
            "--help",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Paired same-process A/B" in completed.stdout
