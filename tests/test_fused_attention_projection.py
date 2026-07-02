#!/usr/bin/env python3
# coding=utf-8
"""Correctness smoke for optional fused attention projection helpers."""
from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover - lightweight local envs
    torch = None  # type: ignore[assignment]


def test_fused_rkv_wavg_projection_matches_fallback() -> None:
    from rwkv7_hf.fused_attention_projection import fused_rkv_wavg_projection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(11)
    batch, hidden, rank = 2, 32, 8
    inputs = [torch.randn(batch, hidden, device=device, dtype=dtype) * 0.1 for _ in range(6)]
    dense = [torch.randn(hidden, hidden, device=device, dtype=dtype) * 0.01 for _ in range(3)]
    down = [torch.randn(rank, hidden, device=device, dtype=dtype) * 0.01 for _ in range(4)]
    up = [torch.randn(hidden, rank, device=device, dtype=dtype) * 0.01 for _ in range(4)]
    bias = [torch.randn(hidden, device=device, dtype=dtype) * 0.01 for _ in range(4)]
    ref = fused_rkv_wavg_projection(*inputs, *dense, *down, *up, *bias, force_fallback=True)
    got = fused_rkv_wavg_projection(*inputs, *dense, *down, *up, *bias, block_m=32, block_r=16, block_k=32)
    for ref_tensor, got_tensor in zip(ref, got, strict=True):
        assert ref_tensor.shape == got_tensor.shape
        assert torch.allclose(ref_tensor.float(), got_tensor.float(), atol=2e-4, rtol=2e-4)


def main() -> int:
    if torch is None:
        print("SKIP fused attention projection test: torch unavailable")
        return 0
    test_fused_rkv_wavg_projection_matches_fallback()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
