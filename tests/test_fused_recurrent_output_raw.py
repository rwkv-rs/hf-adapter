#!/usr/bin/env python3
from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from rwkv7_hf.fused_recurrent_update import fused_recurrent_output_prepare_raw


def run_case(device: str, dtype, batch: int, heads: int, head_dim: int) -> None:
    hidden = heads * head_dim
    values = [
        torch.randn(batch, heads, head_dim, device=device, dtype=dtype) * 0.2
        for _ in range(5)
    ]
    r, w_raw, k_raw, v, g = values
    a = torch.sigmoid(torch.randn_like(r))
    state = torch.randn(batch, heads, head_dim, head_dim, device=device, dtype=torch.float32) * 0.1
    k_k = torch.randn(heads, head_dim, device=device, dtype=dtype)
    k_a = torch.randn(heads, head_dim, device=device, dtype=dtype) * 0.1
    r_k = torch.randn(heads, head_dim, device=device, dtype=dtype) * 0.1
    norm_w = torch.randn(hidden, device=device, dtype=dtype)
    norm_b = torch.randn(hidden, device=device, dtype=dtype)
    args = (r, w_raw, k_raw, v, a, state, g, k_k, k_a, r_k, norm_w, norm_b)
    ref_out, ref_state = fused_recurrent_output_prepare_raw(
        *args, eps=head_dim * 1e-5, block_n=head_dim, force_fallback=True
    )
    out, new_state = fused_recurrent_output_prepare_raw(
        *args, eps=head_dim * 1e-5, block_n=head_dim
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    assert float(F.cosine_similarity(out.float().reshape(batch, -1), ref_out.float().reshape(batch, -1), dim=-1).min()) >= 0.9999
    assert torch.allclose(new_state, ref_state, atol=5e-4, rtol=3e-3)


def main() -> int:
    if torch is None or F is None:
        print("SKIP raw recurrent output: torch unavailable")
        return 0
    torch.manual_seed(707)
    run_case("cpu", torch.float32, 2, 2, 8)
    if torch.cuda.is_available():
        run_case("cuda", torch.float16, 1, 4, 64)
        run_case("cuda", torch.float16, 2, 4, 64)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
