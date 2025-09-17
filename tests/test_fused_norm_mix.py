#!/usr/bin/env python3
# coding=utf-8
"""CPU correctness coverage for the isolated norm + time-mix6 helper."""
from __future__ import annotations

import os

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - lightweight local envs
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _skip_without_torch() -> bool:
    if torch is None or F is None:
        if "PYTEST_CURRENT_TEST" in os.environ:
            import pytest

            pytest.skip("torch unavailable")
        print("SKIP fused_norm_mix tests: torch unavailable")
        return True
    return False


def _randn(*shape: int):
    return torch.randn(*shape, dtype=torch.float32)


def _mix_ref(h, prev_h, mixes):
    delta = prev_h - h
    return tuple(torch.addcmul(h, delta, mix.view(1, 1, -1)) for mix in mixes)


def _assert_output_close(out, residual_ref, h_ref, mixes_ref) -> None:
    assert out.backend == "torch"
    assert torch.allclose(out.residual, residual_ref, atol=0.0, rtol=0.0)
    assert torch.allclose(out.h, h_ref, atol=0.0, rtol=0.0)
    for got, ref in zip(out.mix_tuple(), mixes_ref, strict=True):
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=0.0, rtol=0.0), (got - ref).abs().max()


def test_matches_native_prefill_norm_shift_mix_expression_with_and_without_pre_norm() -> None:
    if _skip_without_torch():
        return
    from rwkv7_hf.fused_norm_mix import fused_attn_norm_shift_mix

    torch.manual_seed(101)
    batch, tokens, hidden = 2, 4, 8
    x = _randn(batch, tokens, hidden) * 0.2
    pre_w = 1.0 + _randn(hidden) * 0.01
    pre_b = _randn(hidden) * 0.01
    norm_w = 1.0 + _randn(hidden) * 0.01
    norm_b = _randn(hidden) * 0.01
    mixes = tuple(torch.rand(hidden, dtype=torch.float32) for _ in range(6))
    cached_prev_h = _randn(batch, hidden) * 0.05

    for has_pre_norm in (False, True):
        residual_ref = F.layer_norm(x, (hidden,), pre_w, pre_b, 1e-5) if has_pre_norm else x
        h_ref = F.layer_norm(residual_ref, (hidden,), norm_w, norm_b, 1e-5)
        prev_h = torch.cat([cached_prev_h.view(batch, 1, hidden), h_ref[:, :-1, :]], dim=1)
        mixes_ref = _mix_ref(h_ref, prev_h, mixes)

        out = fused_attn_norm_shift_mix(
            x,
            prev_h,
            *mixes,
            pre_norm_weight=pre_w,
            pre_norm_bias=pre_b,
            norm_weight=norm_w,
            norm_bias=norm_b,
            has_pre_norm=has_pre_norm,
        )
        _assert_output_close(out, residual_ref, h_ref, mixes_ref)


def test_layer_norm_absent_is_identity_before_time_mix() -> None:
    if _skip_without_torch():
        return
    from rwkv7_hf.fused_norm_mix import fused_attn_norm_shift_mix

    torch.manual_seed(202)
    batch, tokens, hidden = 3, 2, 6
    x = _randn(batch, tokens, hidden)
    prev_x = _randn(batch, tokens, hidden)
    mixes = tuple(torch.rand(hidden, dtype=torch.float32) for _ in range(6))

    out = fused_attn_norm_shift_mix(x, prev_x, *mixes, has_pre_norm=False)
    mixes_ref = _mix_ref(x, prev_x, mixes)
    _assert_output_close(out, x, x, mixes_ref)


def test_decode_rank2_shape_is_supported_for_telemetry_probes() -> None:
    if _skip_without_torch():
        return
    from rwkv7_hf.fused_norm_mix import fused_attn_norm_shift_mix

    torch.manual_seed(303)
    batch, hidden = 2, 5
    x = _randn(batch, hidden)
    prev_x = _randn(batch, hidden)
    norm_w = 1.0 + _randn(hidden) * 0.01
    norm_b = _randn(hidden) * 0.01
    mixes = tuple(torch.rand(hidden, dtype=torch.float32) for _ in range(6))

    h_ref = F.layer_norm(x, (hidden,), norm_w, norm_b, 1e-5)
    delta = prev_x - h_ref
    mixes_ref = tuple(torch.addcmul(h_ref, delta, mix.view(1, -1)) for mix in mixes)
    out = fused_attn_norm_shift_mix(x, prev_x, *mixes, norm_weight=norm_w, norm_bias=norm_b)
    _assert_output_close(out, x, h_ref, mixes_ref)


def main() -> int:
    if _skip_without_torch():
        return 0
    test_matches_native_prefill_norm_shift_mix_expression_with_and_without_pre_norm()
    test_layer_norm_absent_is_identity_before_time_mix()
    test_decode_rank2_shape_is_supported_for_telemetry_probes()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
