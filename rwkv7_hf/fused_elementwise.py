# coding=utf-8
"""Small graph-safe Triton elementwise kernels used by native inference."""
from __future__ import annotations

try:  # pragma: no cover - optional on CPU-only installs
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised on CUDA/Triton hosts
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


_HAS_TRITON = triton is not None and tl is not None


if _HAS_TRITON:

    @triton.jit
    def _relu_square_kernel(x_ptr, out_ptr, count: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        values = tl.maximum(values, 0.0)
        tl.store(out_ptr + offsets, values * values, mask=mask)


def fused_relu_square_available() -> bool:
    return bool(_HAS_TRITON and torch is not None)


def fused_relu_square(x, *, block: int = 512):
    """Return ``relu(x) ** 2`` using one graph-safe elementwise launch."""

    if torch is None:
        raise RuntimeError("fused_relu_square requires torch")
    if not fused_relu_square_available() or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        return torch.relu(x) ** 2
    source = x.contiguous()
    out = torch.empty_like(source)
    count = int(source.numel())
    _relu_square_kernel[(triton.cdiv(count, int(block)),)](
        source,
        out,
        count,
        BLOCK=int(block),
        num_warps=4,
    )
    return out
