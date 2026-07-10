#!/usr/bin/env python3
from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover - lightweight local environments
    torch = None  # type: ignore[assignment]

from rwkv7_hf.fused_recurrent_update import (
    fused_recurrent_scan_state_prep,
    fused_recurrent_scan_state_prep_available,
)


def _inputs(*, has_v_gate: bool):
    torch.manual_seed(123 + int(has_v_gate))
    batch, tokens, heads, head_dim = 1, 16, 2, 64
    shape = (batch, tokens, heads, head_dim)
    device = "cuda"
    dtype = torch.float16
    r = torch.randn(shape, device=device, dtype=dtype) * 0.05
    w = torch.randn_like(r) * 0.1
    k = torch.randn_like(r) * 0.05
    v = torch.randn_like(r) * 0.05
    a = torch.sigmoid(torch.randn_like(r))
    state = torch.randn(batch, heads, head_dim, head_dim, device=device, dtype=torch.float32) * 0.01
    k_k = torch.randn(heads * head_dim, device=device, dtype=dtype) * 0.1
    k_a = torch.randn(heads * head_dim, device=device, dtype=dtype) * 0.1
    gate = {}
    if has_v_gate:
        gate = {
            "v_first": torch.randn_like(v) * 0.05,
            "v_gate": torch.sigmoid(torch.randn_like(v)),
        }
    return (r, w, k, v, a, state, k_k, k_a), gate


def test_split_state_scan_matches_full_head_and_torch() -> None:
    if torch is None or not torch.cuda.is_available() or not fused_recurrent_scan_state_prep_available():
        print("SKIP split state-scan CUDA test")
        return

    for has_v_gate in (False, True):
        args, gate = _inputs(has_v_gate=has_v_gate)
        full = fused_recurrent_scan_state_prep(
            *args,
            block_n=64,
            block_m=64,
            num_warps=8,
            **gate,
        )
        split = fused_recurrent_scan_state_prep(
            *args,
            block_n=64,
            block_m=16,
            num_warps=4,
            **gate,
        )
        reference = fused_recurrent_scan_state_prep(
            *args,
            block_n=64,
            block_m=16,
            num_warps=4,
            force_fallback=True,
            **gate,
        )
        torch.cuda.synchronize()
        for split_tensor, full_tensor, reference_tensor in zip(split, full, reference):
            assert torch.allclose(split_tensor.float(), full_tensor.float(), atol=2e-4, rtol=2e-4)
            assert torch.allclose(split_tensor.float(), reference_tensor.float(), atol=2e-4, rtol=2e-4)


def main() -> int:
    test_split_state_scan_matches_full_head_and_torch()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
