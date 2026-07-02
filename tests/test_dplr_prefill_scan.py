#!/usr/bin/env python3
# coding=utf-8
"""Tests for the pure torch RWKV-7 DPLR/chunked prefill reference."""
from __future__ import annotations

import os
import sys

try:
    import torch
except Exception:  # pragma: no cover - local lightweight environments
    torch = None  # type: ignore[assignment]

if "pytest" in sys.modules:  # pragma: no cover - pytest collection metadata
    import pytest

    pytestmark = pytest.mark.skipif(torch is None, reason="torch unavailable")


CHUNK_SIZES = (1, 2, 3, 8)
ALGORITHMS = ("sequential", "affine", "lowrank", "wy")


def _skip_if_no_torch() -> bool:
    if torch is not None:
        return False
    try:
        import pytest
    except Exception:  # pragma: no cover - direct script mode without pytest
        return True
    pytest.skip("torch unavailable")
    return True


def _make_inputs(*, flat: bool, dtype=None):
    assert torch is not None
    if dtype is None:
        dtype = torch.float32
    torch.manual_seed(7007)
    B, T, H, N = 2, 7, 2, 4
    r = torch.randn(B, T, H, N, dtype=dtype) * 0.2
    w = torch.sigmoid(torch.randn(B, T, H, N, dtype=dtype))
    k = torch.randn(B, T, H, N, dtype=dtype) * 0.2
    v = torch.randn(B, T, H, N, dtype=dtype) * 0.2
    kk = torch.randn(B, T, H, N, dtype=dtype) * 0.2
    a = torch.randn(B, T, H, N, dtype=dtype) * 0.2
    state = torch.randn(B, H, N, N, dtype=dtype) * 0.2
    if flat:
        r = r.reshape(B, T, H * N)
        w = w.reshape(B, T, H * N)
        k = k.reshape(B, T, H * N)
        v = v.reshape(B, T, H * N)
        kk = kk.reshape(B, T, H * N)
        a = a.reshape(B, T, H * N)
    return r, w, k, v, kk, a, state


def _assert_matches_reference(
    *,
    flat: bool,
    chunk_size: int,
    algorithm: str = "sequential",
    force_fallback: bool = False,
    dtype=None,
    atol: float = 2e-6,
    rtol: float = 2e-6,
) -> None:
    assert torch is not None
    from rwkv7_hf.dplr_prefill import dplr_chunk_scan
    from rwkv7_hf.fused_recurrent_update import torch_recurrent_scan

    r, w, k, v, kk, a, state = _make_inputs(flat=flat, dtype=dtype)
    with torch.no_grad():
        ref_out, ref_state = torch_recurrent_scan(r, w, k, v, kk, a, state)
        got_out, got_state = dplr_chunk_scan(
            r,
            w,
            k,
            v,
            kk,
            a,
            state,
            chunk_size=chunk_size,
            force_fallback=force_fallback,
            algorithm=algorithm,
        )
    assert got_out.shape == ref_out.shape == r.shape
    assert got_state.shape == ref_state.shape == state.shape
    assert got_out.dtype == r.dtype
    assert got_state.dtype == state.dtype
    assert torch.allclose(got_out, ref_out, atol=atol, rtol=rtol), (got_out - ref_out).abs().max()
    assert torch.allclose(got_state, ref_state, atol=atol, rtol=rtol), (got_state - ref_state).abs().max()


def test_dplr_chunk_scan_bthn_matches_torch_recurrent_scan() -> None:
    if _skip_if_no_torch():
        return
    for algorithm in ALGORITHMS:
        for chunk_size in CHUNK_SIZES:
            _assert_matches_reference(flat=False, chunk_size=chunk_size, algorithm=algorithm)


def test_dplr_chunk_scan_flat_matches_torch_recurrent_scan() -> None:
    if _skip_if_no_torch():
        return
    for algorithm in ALGORITHMS:
        for chunk_size in CHUNK_SIZES:
            _assert_matches_reference(flat=True, chunk_size=chunk_size, algorithm=algorithm)


def test_dplr_chunk_scan_force_fallback_matches_reference() -> None:
    if _skip_if_no_torch():
        return
    _assert_matches_reference(flat=False, chunk_size=3, force_fallback=True)


def test_dplr_chunk_scan_lowrank_fp32_matches_torch_recurrent_scan() -> None:
    if _skip_if_no_torch():
        return
    for algorithm in ("lowrank", "wy"):
        for flat in (False, True):
            for chunk_size in CHUNK_SIZES:
                _assert_matches_reference(
                    flat=flat,
                    chunk_size=chunk_size,
                    algorithm=algorithm,
                    dtype=torch.float32,
                )


def test_dplr_chunk_scan_env_algorithm_lowrank_matches_reference() -> None:
    if _skip_if_no_torch():
        return
    old = os.environ.get("RWKV7_DPLR_PREFILL_ALGORITHM")
    os.environ["RWKV7_DPLR_PREFILL_ALGORITHM"] = "wy"
    try:
        _assert_matches_reference(flat=False, chunk_size=2, algorithm=None)
    finally:
        if old is None:
            os.environ.pop("RWKV7_DPLR_PREFILL_ALGORITHM", None)
        else:
            os.environ["RWKV7_DPLR_PREFILL_ALGORITHM"] = old


def test_lowrank_chunk_summary_metadata_shapes() -> None:
    if _skip_if_no_torch():
        return
    from rwkv7_hf.dplr_prefill import lowrank_chunk_summary

    r, w, k, v, kk, a, _state = _make_inputs(flat=False, dtype=torch.float32)
    summary = lowrank_chunk_summary(w[:, :3], k[:, :3], v[:, :3], kk[:, :3], a[:, :3])
    assert summary["algorithm"] == "lowrank-wy"
    assert summary["length"] == 3
    assert summary["rank"] == 3
    assert len(summary["prefix"]) == 3
    assert summary["transition_diag"].shape == (2, 2, 4)
    assert summary["transition_left"].shape == (2, 2, 4, 3)
    assert summary["transition_right"].shape == (2, 2, 4, 3)
    assert summary["additive_left"].shape == (2, 2, 4, 3)
    assert summary["additive_right"].shape == (2, 2, 4, 3)


def main() -> int:
    if torch is None:
        print("SKIP dplr prefill scan test: torch unavailable")
        return 0
    test_dplr_chunk_scan_bthn_matches_torch_recurrent_scan()
    test_dplr_chunk_scan_flat_matches_torch_recurrent_scan()
    test_dplr_chunk_scan_force_fallback_matches_reference()
    test_dplr_chunk_scan_lowrank_fp32_matches_torch_recurrent_scan()
    test_dplr_chunk_scan_env_algorithm_lowrank_matches_reference()
    test_lowrank_chunk_summary_metadata_shapes()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
