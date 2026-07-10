#!/usr/bin/env python3
from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover - lightweight local environments
    torch = None  # type: ignore[assignment]

from rwkv7_hf.fused_decode_norm_mix import (
    fused_attn_norm_mix6_decode,
    fused_ffn_add_norm_mix_decode,
)


def test_fallback_matches_reference() -> None:
    if torch is None:
        print("SKIP fused decode norm-mix: torch unavailable")
        return
    torch.manual_seed(321)
    batch, hidden = 3, 16
    x = torch.randn(batch, hidden)
    previous = torch.randn_like(x)
    weight = torch.randn(hidden)
    bias = torch.randn(hidden)
    mixes = [torch.randn(hidden) for _ in range(6)]
    h = torch.nn.functional.layer_norm(x, (hidden,), weight, bias, 1e-5)
    expected = [h + (previous - h) * mix for mix in mixes]
    state = previous.clone()
    got = fused_attn_norm_mix6_decode(
        x, state, weight, bias, *mixes, force_fallback=True
    )
    assert torch.equal(state, h)
    for actual, reference in zip(got, expected):
        assert torch.equal(actual, reference)

    attn_out = torch.randn_like(x)
    ffn_previous = torch.randn_like(x)
    ffn_mix = torch.randn(hidden)
    residual_ref = x + attn_out
    ffn_h = torch.nn.functional.layer_norm(residual_ref, (hidden,), weight, bias, 1e-5)
    mixed_ref = ffn_h + (ffn_previous - ffn_h) * ffn_mix
    ffn_state = ffn_previous.clone()
    residual, mixed = fused_ffn_add_norm_mix_decode(
        x, attn_out, ffn_state, weight, bias, ffn_mix, force_fallback=True
    )
    assert torch.equal(residual, residual_ref)
    assert torch.equal(mixed, mixed_ref)
    assert torch.equal(ffn_state, ffn_h)


def test_cuda_matches_fallback() -> None:
    if torch is None or not torch.cuda.is_available():
        print("SKIP fused decode norm-mix CUDA test")
        return
    torch.manual_seed(654)
    # Cover the native widths used from the small checkpoints through the
    # 7B-class model. Batch>1 needs only one representative compile shape.
    for batch, hidden in ((1, 768), (4, 768), (1, 2048), (1, 4096)):
        device, dtype = "cuda", torch.float16
        x = torch.randn(batch, hidden, device=device, dtype=dtype)
        previous = torch.randn_like(x)
        weight = torch.randn(hidden, device=device, dtype=dtype)
        bias = torch.randn(hidden, device=device, dtype=dtype)
        mixes = [torch.randn(hidden, device=device, dtype=dtype) for _ in range(6)]
        ref_state = previous.clone()
        reference = fused_attn_norm_mix6_decode(
            x, ref_state, weight, bias, *mixes, force_fallback=True
        )
        state = previous.clone()
        actual = fused_attn_norm_mix6_decode(x, state, weight, bias, *mixes)
        torch.cuda.synchronize()
        assert torch.allclose(state.float(), ref_state.float(), atol=2e-3, rtol=2e-3)
        for got, ref in zip(actual, reference):
            assert torch.allclose(got.float(), ref.float(), atol=2e-2, rtol=3e-3)
            assert float(torch.nn.functional.cosine_similarity(got.float(), ref.float(), dim=-1).min()) >= 0.9999

        attn_out = torch.randn_like(x)
        ffn_previous = torch.randn_like(x)
        ffn_mix = torch.randn(hidden, device=device, dtype=dtype)
        ref_ffn_state = ffn_previous.clone()
        ref_residual, ref_mixed = fused_ffn_add_norm_mix_decode(
            x, attn_out, ref_ffn_state, weight, bias, ffn_mix, force_fallback=True
        )
        ffn_state = ffn_previous.clone()
        residual, mixed = fused_ffn_add_norm_mix_decode(
            x, attn_out, ffn_state, weight, bias, ffn_mix
        )
        torch.cuda.synchronize()
        assert torch.allclose(residual.float(), ref_residual.float(), atol=2e-3, rtol=2e-3)
        assert torch.allclose(mixed.float(), ref_mixed.float(), atol=2e-2, rtol=3e-3)
        assert float(torch.nn.functional.cosine_similarity(mixed.float(), ref_mixed.float(), dim=-1).min()) >= 0.9999
        assert torch.allclose(ffn_state.float(), ref_ffn_state.float(), atol=4e-3, rtol=2e-3)


def main() -> int:
    test_fallback_matches_reference()
    test_cuda_matches_fallback()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
