#!/usr/bin/env python3
# coding=utf-8
"""Tests for Triton DPLR/WY prefill prototypes."""
from __future__ import annotations

import sys

try:
    import torch
except Exception:  # pragma: no cover - local lightweight environments
    torch = None  # type: ignore[assignment]

if "pytest" in sys.modules:  # pragma: no cover - pytest collection metadata
    import pytest

    pytestmark = pytest.mark.skipif(torch is None, reason="torch unavailable")


def _skip_if_no_torch() -> bool:
    if torch is not None:
        return False
    try:
        import pytest
    except Exception:  # pragma: no cover
        return True
    pytest.skip("torch unavailable")
    return True


def _make_inputs(device="cpu", dtype=None):
    assert torch is not None
    if dtype is None:
        dtype = torch.float32
    torch.manual_seed(7007)
    B, T, H, N = 1, 8, 2, 4
    shape = (B, T, H, N)
    r = torch.randn(shape, device=device, dtype=dtype) * 0.2
    w = torch.sigmoid(torch.randn(shape, device=device, dtype=dtype))
    k = torch.randn(shape, device=device, dtype=dtype) * 0.2
    v = torch.randn(shape, device=device, dtype=dtype) * 0.2
    kk = torch.randn(shape, device=device, dtype=dtype) * 0.2
    a = torch.randn(shape, device=device, dtype=dtype) * 0.2
    state = torch.randn(B, H, N, N, device=device, dtype=torch.float32) * 0.2
    return r, w, k, v, kk, a, state


def _apply_dense_summaries(state, summary):
    cur = state.float()
    transition = summary["transition"]
    additive = summary["additive"]
    for chunk in range(int(transition.shape[1])):
        cur = cur @ transition[:, chunk] + additive[:, chunk]
    return cur


def test_dense_chunk_summary_torch_final_state_matches_recurrent_scan() -> None:
    if _skip_if_no_torch():
        return
    from rwkv7_hf.dplr_prefill_triton import (
        dplr_compact_wy_apply_summaries_torch,
        dplr_compact_wy_chunk_summary_torch,
        dplr_compact_wy_summary_to_dense,
        dplr_dense_chunk_apply_torch,
        dplr_dense_chunk_summary_torch,
        dplr_dense_prefix_combine_torch,
        dplr_dense_three_stage_triton,
    )
    from rwkv7_hf.fused_recurrent_update import torch_recurrent_scan

    r, w, k, v, kk, a, state = _make_inputs(device="cpu", dtype=torch.float32)
    summary = dplr_dense_chunk_summary_torch(w, k, v, kk, a, chunk_size=4)
    ref_out, ref_state = torch_recurrent_scan(r, w, k, v, kk, a, state)
    got_state = _apply_dense_summaries(state, summary)
    assert torch.allclose(got_state, ref_state, atol=2e-6, rtol=2e-6), (got_state - ref_state).abs().max()

    compact = dplr_compact_wy_chunk_summary_torch(w, k, v, kk, a, chunk_size=4)
    dense_from_compact = dplr_compact_wy_summary_to_dense(compact)
    assert compact["transition_diag"].shape == (1, 2, 2, 4)
    assert compact["transition_left"].shape == (1, 2, 2, 4, 4)
    assert compact["transition_right"].shape == (1, 2, 2, 4, 4)
    assert compact["additive_left"].shape == (1, 2, 2, 4, 4)
    assert compact["additive_right"].shape == (1, 2, 2, 4, 4)
    assert torch.allclose(dense_from_compact["transition"], summary["transition"], atol=2e-6, rtol=2e-6), (
        dense_from_compact["transition"] - summary["transition"]
    ).abs().max()
    assert torch.allclose(dense_from_compact["additive"], summary["additive"], atol=2e-6, rtol=2e-6), (
        dense_from_compact["additive"] - summary["additive"]
    ).abs().max()
    compact_state = dplr_compact_wy_apply_summaries_torch(state, compact)
    assert torch.allclose(compact_state, ref_state, atol=2e-6, rtol=2e-6), (compact_state - ref_state).abs().max()

    start_states, prefix_final = dplr_dense_prefix_combine_torch(state, summary["transition"], summary["additive"])
    assert torch.allclose(start_states[:, 0], state.float(), atol=0, rtol=0)
    assert torch.allclose(prefix_final, ref_state, atol=2e-6, rtol=2e-6), (prefix_final - ref_state).abs().max()

    got_out, chunk_ends = dplr_dense_chunk_apply_torch(r, w, k, v, kk, a, start_states, chunk_size=4)
    assert torch.allclose(got_out, ref_out, atol=2e-6, rtol=2e-6), (got_out - ref_out).abs().max()
    assert torch.allclose(chunk_ends[:, -1], ref_state, atol=2e-6, rtol=2e-6), (chunk_ends[:, -1] - ref_state).abs().max()

    dense3_out, dense3_state = dplr_dense_three_stage_triton(
        r, w, k, v, kk, a, state, chunk_size=4, force_fallback=True
    )
    assert torch.allclose(dense3_out, ref_out, atol=2e-6, rtol=2e-6), (dense3_out - ref_out).abs().max()
    assert torch.allclose(dense3_state, ref_state, atol=2e-6, rtol=2e-6), (dense3_state - ref_state).abs().max()


def test_dense_chunk_summary_triton_matches_torch_cuda() -> None:
    if _skip_if_no_torch():
        return
    if not torch.cuda.is_available():
        try:
            import pytest
        except Exception:  # pragma: no cover
            return
        pytest.skip("cuda unavailable")
    from rwkv7_hf.dplr_prefill_triton import (
        dplr_dense_chunk_apply_torch,
        dplr_dense_chunk_apply_triton,
        dplr_dense_chunk_summary_torch,
        dplr_dense_chunk_summary_triton,
        dplr_dense_chunk_summary_triton_available,
        dplr_dense_prefix_combine_torch,
        dplr_dense_prefix_combine_triton,
        dplr_dense_three_stage_triton,
    )
    from rwkv7_hf.fused_recurrent_update import torch_recurrent_scan

    if not dplr_dense_chunk_summary_triton_available():
        try:
            import pytest
        except Exception:  # pragma: no cover
            return
        pytest.skip("triton summary unavailable")

    r, w, k, v, kk, a, state = _make_inputs(device="cuda", dtype=torch.float32)
    ref_out, ref_state = torch_recurrent_scan(r, w, k, v, kk, a, state)
    ref = dplr_dense_chunk_summary_torch(w, k, v, kk, a, chunk_size=4)
    got = dplr_dense_chunk_summary_triton(w, k, v, kk, a, chunk_size=4, block_m=2)
    assert torch.allclose(got["transition"], ref["transition"], atol=2e-6, rtol=2e-6), (
        got["transition"] - ref["transition"]
    ).abs().max()
    assert torch.allclose(got["additive"], ref["additive"], atol=2e-6, rtol=2e-6), (
        got["additive"] - ref["additive"]
    ).abs().max()

    starts_ref, prefix_final_ref = dplr_dense_prefix_combine_torch(state, ref["transition"], ref["additive"])
    starts_got, prefix_final_got = dplr_dense_prefix_combine_triton(
        state, got["transition"], got["additive"], block_m=2
    )
    assert torch.allclose(starts_got, starts_ref, atol=2e-6, rtol=2e-6), (starts_got - starts_ref).abs().max()
    assert torch.allclose(prefix_final_got, prefix_final_ref, atol=2e-6, rtol=2e-6), (
        prefix_final_got - prefix_final_ref
    ).abs().max()
    assert torch.allclose(prefix_final_got, ref_state, atol=2e-6, rtol=2e-6), (prefix_final_got - ref_state).abs().max()

    out_ref, ends_ref = dplr_dense_chunk_apply_torch(r, w, k, v, kk, a, starts_ref, chunk_size=4)
    out_got, ends_got = dplr_dense_chunk_apply_triton(
        r, w, k, v, kk, a, starts_got, chunk_size=4, block_m=2
    )
    assert torch.allclose(out_got, out_ref, atol=2e-6, rtol=2e-6), (out_got - out_ref).abs().max()
    assert torch.allclose(ends_got, ends_ref, atol=2e-6, rtol=2e-6), (ends_got - ends_ref).abs().max()

    dense3_out, dense3_state = dplr_dense_three_stage_triton(r, w, k, v, kk, a, state, chunk_size=4)
    assert torch.allclose(dense3_out, ref_out, atol=2e-6, rtol=2e-6), (dense3_out - ref_out).abs().max()
    assert torch.allclose(dense3_state, ref_state, atol=2e-6, rtol=2e-6), (dense3_state - ref_state).abs().max()


def main() -> int:
    if torch is None:
        print("SKIP dplr triton tests: torch unavailable")
        return 0
    test_dense_chunk_summary_torch_final_state_matches_recurrent_scan()
    if torch.cuda.is_available():
        test_dense_chunk_summary_triton_matches_torch_cuda()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
