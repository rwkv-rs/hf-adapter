# coding=utf-8
"""Official-rwkv-style int8 (fp16i8) weight quantization + dequant matmul.

Ported from BlinkDL/rwkv `model.py` (`torch_mm8_seq` / the `cuda_mm8_*` path):
a per-row + per-column affine int8 scheme with two scales (rx, ry) and two
offsets (mx, my). The official fast path fuses the dequant into a CUDA matmul
(``torch.ops.rwkv.mm8_seq``); this module starts from the readable PyTorch
reference (the dequant formula), so correctness can be validated before the
fused-Triton speed kernel is added.

Quantization of a weight ``W: [N, M]`` (used as ``y = x @ W``)::

    w = W.float()
    my = amin(w, dim=1)        # [N,1]  per-row offset
    w = w - my
    mx = amin(w, dim=0)        # [M]    per-col offset
    w = w - mx
    rx = amax(w, dim=0)        # [M]    per-col scale
    w = w / rx
    ry = amax(w, dim=1)        # [N,1]  per-row scale
    w = w / ry
    w_u8 = clip(floor(w * 256), 0, 255).to(uint8)
    rx_stored = rx / 16
    ry_stored = ry / 16

Dequantization (the inverse, with a +0.5 rounding center)::

    W_approx = (w_u8 + 0.5) * ry_stored * rx_stored + my + mx

For an ``nn.Linear`` whose ``weight`` is ``[out, in]`` (so ``F.linear(x, w) =
x @ w.T``), quantize ``weight.t().contiguous()`` (i.e. ``W = weight.T`` with
``N = in``, ``M = out``) and call :func:`mm8_linear`.
"""
from __future__ import annotations

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .native_quant_policy import normalize_native_mm_policy, should_quantize_linear


def quantize_mm8(weight):
    """Quantize ``weight: [N, M]`` to the official rwkv fp16i8 format.

    Returns ``(w_u8, mx, rx, my, ry)`` where ``w_u8`` is ``uint8 [N, M]`` and
    ``mx, rx`` are ``[M]`` / ``my, ry`` are ``[N, 1]``, in ``weight.dtype``.
    """
    if torch is None:
        raise RuntimeError("quantize_mm8 requires torch")
    w = weight.float()
    n, m = w.shape
    eps = 1e-8
    # Order mirrors the official code: if N > M subtract row-min first, else
    # column-min first. The math is symmetric; this just keeps ranges tame.
    if n > m:
        my = w.amin(dim=1, keepdim=True)
        w = w - my
        mx = w.amin(dim=0)
        w = w - mx
    else:
        mx = w.amin(dim=0)
        w = w - mx
        my = w.amin(dim=1, keepdim=True)
        w = w - my
    rx = w.amax(dim=0).clamp(min=eps)
    w = w / rx
    ry = w.amax(dim=1, keepdim=True).clamp(min=eps)
    w = w / ry
    w_u8 = torch.clamp(torch.floor(w * 256.0), 0, 255).to(torch.uint8)
    out_dtype = weight.dtype
    return (
        w_u8,
        mx.to(out_dtype),
        (rx / 16.0).to(out_dtype),
        my.to(out_dtype),
        (ry / 16.0).to(out_dtype),
    )


def dequantize_mm8(w_u8, mx, rx, my, ry, out_dtype=None):
    """Materialize the dequantized weight ``[N, M]`` (reference, not fused)."""
    if torch is None:
        raise RuntimeError("dequantize_mm8 requires torch")
    dtype = out_dtype if out_dtype is not None else mx.dtype
    return (w_u8.to(dtype) + 0.5) * ry * rx + my + mx


def mm8_matmul(x, w_u8, mx, rx, my, ry):
    """``y = x @ dequant(W)`` for ``x: [..., N]``, returns ``[..., M]``.

    Reference path: materialize the full dequantized weight, then matmul.
    Equivalent to the official ``torch_mm8_seq`` / ``torch_mm8_one``.
    """
    if torch is None:
        raise RuntimeError("mm8_matmul requires torch")
    deq = dequantize_mm8(w_u8, mx, rx, my, ry, out_dtype=x.dtype)
    return x @ deq


def mm8_linear(x, weight_u8, mx, rx, my, ry):
    """Drop-in for ``F.linear(x, weight)`` with pre-quantized ``weight``.

    ``weight`` must have been quantized via ``quantize_mm8(weight.t().contiguous())``
    so that ``W = weight.T`` has ``N = in_features``, ``M = out_features``.
    """
    return mm8_matmul(x, weight_u8, mx, rx, my, ry)


# --------------------------------------------------------------------------- #
# Fused Triton dequant-matmul (the speed path; mirrors official cuda_mm8_*).
# The reference (:func:`mm8_matmul`) materializes the full dequantized weight,
# which costs VRAM + a dense fp16 matmul. This kernel reads uint8 + scales and
# dequantizes in registers, so it never materializes the fp16 weight.
# --------------------------------------------------------------------------- #

try:  # pragma: no cover
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

_HAS_TRITON = triton is not None and tl is not None


if _HAS_TRITON:

    @triton.jit
    def _mm8_gemv_kernel(
        x_ptr, w_ptr, mx_ptr, rx_ptr, my_ptr, ry_ptr, y_ptr,
        N, M,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """y[m] = sum_n x[n] * ((w[n,m]+0.5)*ry[n]*rx[m] + my[n] + mx[m])."""
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        rx_m = tl.load(rx_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)  # [BLOCK_M]
        mx_m = tl.load(mx_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)  # [BLOCK_M]
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_N)
        for n_start in range(0, N, BLOCK_N):
            n = n_start + offs_n
            mask_n = n < N
            x = tl.load(x_ptr + n, mask=mask_n, other=0.0).to(tl.float32)        # [BLOCK_N]
            ry_n = tl.load(ry_ptr + n, mask=mask_n, other=0.0).to(tl.float32)    # [BLOCK_N]
            my_n = tl.load(my_ptr + n, mask=mask_n, other=0.0).to(tl.float32)    # [BLOCK_N]
            w_addr = n.to(tl.int64)[:, None] * M + offs_m.to(tl.int64)[None, :]
            w_mask = mask_n[:, None] & mask_m[None, :]
            w = tl.load(w_ptr + w_addr, mask=w_mask, other=0.0).to(tl.float32)   # [BLOCK_N, BLOCK_M]
            deq = (w + 0.5) * ry_n[:, None] * rx_m[None, :] + my_n[:, None] + mx_m[None, :]
            acc += tl.sum(x[:, None] * deq, axis=0)
        tl.store(y_ptr + offs_m, acc, mask=mask_m)

    @triton.jit
    def _mm8_batched_gemv_kernel(
        x_ptr, w_ptr, mx_ptr, rx_ptr, my_ptr, ry_ptr, y_ptr,
        B, N, M,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Batched decode GEMV with one launch for all active rows.

        The previous path launched one kernel per row through ``torch.stack``
        and fell back to materializing the complete fp16 weight above four
        rows.  A two-dimensional launch keeps every row independent while
        removing Python/launch serialization and the large-batch dequantize
        fallback.
        """
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        rx_m = tl.load(rx_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        mx_m = tl.load(mx_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_N)
        for n_start in range(0, N, BLOCK_N):
            n = n_start + offs_n
            mask_n = n < N
            x = tl.load(x_ptr + pid_b * N + n, mask=mask_n, other=0.0).to(tl.float32)
            ry_n = tl.load(ry_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
            my_n = tl.load(my_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
            w_addr = n.to(tl.int64)[:, None] * M + offs_m.to(tl.int64)[None, :]
            w_mask = mask_n[:, None] & mask_m[None, :]
            w = tl.load(w_ptr + w_addr, mask=w_mask, other=0.0).to(tl.float32)
            deq = (w + 0.5) * ry_n[:, None] * rx_m[None, :] + my_n[:, None] + mx_m[None, :]
            acc += tl.sum(x[:, None] * deq, axis=0)
        tl.store(y_ptr + pid_b * M + offs_m, acc, mask=mask_m)

    @triton.jit
    def _mm8_batched_dot_kernel(
        x_ptr, w_ptr, mx_ptr, rx_ptr, my_ptr, ry_ptr, y_ptr,
        B, N, M,
        BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Tensor-core batched decode kernel with algebraic affine dequant.

        Expanding the affine format lets the large ``x @ uint8_weight`` term
        use ``tl.dot`` while the row/column offsets are applied as two small
        reductions.  Unlike row-wise GEMV, every weight tile is read once for
        the complete decode batch.
        """
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_b = offs_b < B
        mask_m = offs_m < M
        acc = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)
        sum_x = tl.zeros((BLOCK_B,), dtype=tl.float32)
        sum_x_my = tl.zeros((BLOCK_B,), dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_N)
        for n0 in range(0, N, BLOCK_N):
            n = n0 + offs_n
            mask_n = n < N
            x = tl.load(
                x_ptr + offs_b[:, None] * N + n[None, :],
                mask=mask_b[:, None] & mask_n[None, :], other=0.0,
            ).to(tl.float32)
            ry_n = tl.load(ry_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
            my_n = tl.load(my_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
            w = tl.load(
                w_ptr + n[:, None].to(tl.int64) * M + offs_m[None, :].to(tl.int64),
                mask=mask_n[:, None] & mask_m[None, :], other=0.0,
            ).to(tl.float32)
            acc += tl.dot((x * ry_n[None, :]).to(tl.float16), (w + 0.5).to(tl.float16))
            sum_x += tl.sum(x, axis=1)
            sum_x_my += tl.sum(x * my_n[None, :], axis=1)
        rx_m = tl.load(rx_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        mx_m = tl.load(mx_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        out = acc * rx_m[None, :] + sum_x_my[:, None] + sum_x[:, None] * mx_m[None, :]
        tl.store(
            y_ptr + offs_b[:, None] * M + offs_m[None, :], out,
            mask=mask_b[:, None] & mask_m[None, :],
        )


def mm8_gemv_available(device=None) -> bool:
    """Return whether the fused Triton GEMV path can run on ``device``.

    Importing Triton is not enough: CPU-only CI or CUDA-hidden processes can
    import Triton but still have no active CUDA driver, which would fail at
    launch time. Keep the fused path CUDA-only and let callers fall back to the
    reference dequant+matmul path elsewhere.
    """
    if not (_HAS_TRITON and torch is not None and torch.cuda.is_available()):
        return False
    if device is None:
        return True
    dev = torch.device(device)
    return dev.type == "cuda"


def _as_1d(t):
    return t.reshape(-1) if t is not None else None


def _mm8_decode_blocks(x, block_m, block_n):
    if block_m is not None and block_n is not None:
        return int(block_m), int(block_n)
    blackwell = bool(x.is_cuda and torch.cuda.get_device_capability(x.device)[0] >= 12)
    return int(block_m or (128 if blackwell else 64)), int(block_n or (128 if blackwell else 64))


def mm8_gemv_triton(x, w_u8, mx, rx, my, ry, *, block_m=None, block_n=None):
    """Fused int8 dequant GEMV: ``x: [N]`` -> ``[M]`` (single vector, decode path)."""
    if not (x.is_cuda and mm8_gemv_available(x.device)):
        raise RuntimeError("mm8_gemv_triton requires triton + torch + CUDA input")
    block_m, block_n = _mm8_decode_blocks(x, block_m, block_n)
    n, m = w_u8.shape
    y = torch.empty(m, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, block_m),)
    _mm8_gemv_kernel[grid](
        x, w_u8, _as_1d(mx), _as_1d(rx), _as_1d(my), _as_1d(ry), y,
        n, m, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4,
    )
    return y


def mm8_batched_gemv_triton(x, w_u8, mx, rx, my, ry, *, block_m=None, block_n=None):
    """Fused int8 dequant GEMV for a small contiguous ``[B, N]`` decode batch."""
    if not (x.is_cuda and x.dim() == 2 and mm8_gemv_available(x.device)):
        raise RuntimeError("mm8_batched_gemv_triton requires a 2-D CUDA input")
    block_m, block_n = _mm8_decode_blocks(x, block_m, block_n)
    x = x.contiguous()
    b, n = x.shape
    wn, m = w_u8.shape
    if int(n) != int(wn):
        raise ValueError(f"input width {n} does not match quantized weight width {wn}")
    y = torch.empty((b, m), device=x.device, dtype=x.dtype)
    grid = (b, triton.cdiv(m, block_m))
    _mm8_batched_gemv_kernel[grid](
        x, w_u8, _as_1d(mx), _as_1d(rx), _as_1d(my), _as_1d(ry), y,
        b, n, m, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4,
    )
    return y


def mm8_batched_dot_triton(
    x, w_u8, mx, rx, my, ry, *, block_b=16, block_m=128, block_n=64,
):
    """Tensor-core fp16 W8 path for decode batches large enough to reuse weights."""
    if not (x.is_cuda and x.dim() == 2 and mm8_gemv_available(x.device)):
        raise RuntimeError("mm8_batched_dot_triton requires a 2-D CUDA input")
    if x.dtype != torch.float16:
        raise TypeError("mm8_batched_dot_triton currently requires fp16 input")
    x = x.contiguous()
    b, n = x.shape
    wn, m = w_u8.shape
    if int(n) != int(wn):
        raise ValueError(f"input width {n} does not match quantized weight width {wn}")
    y = torch.empty((b, m), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(b, block_b), triton.cdiv(m, block_m))
    _mm8_batched_dot_kernel[grid](
        x, w_u8, _as_1d(mx), _as_1d(rx), _as_1d(my), _as_1d(ry), y,
        b, n, m, BLOCK_B=block_b, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=8,
    )
    return y


def mm8_matmul_triton(x, w_u8, mx, rx, my, ry, *, max_gemv_rows: int | None = None):
    """Fused int8 dequant matmul with safe fallbacks.

    ``x: [N]`` uses the fused GEMV decode path. On compute capability 12+, small
    ``[B, N]`` decode batches use one two-dimensional Triton launch; older architectures
    retain their previously validated dispatch. Prefill / large-batch inputs
    fall back to the reference path that materializes once for one PyTorch GEMM.
    """
    if not (x.is_cuda and mm8_gemv_available(x.device)):
        return mm8_matmul(x, w_u8, mx, rx, my, ry)
    if x.dim() == 1:
        return mm8_gemv_triton(x, w_u8, mx, rx, my, ry)
    if x.dim() != 2:
        return mm8_matmul(x, w_u8, mx, rx, my, ry)
    if int(x.shape[0]) == 1:
        return mm8_gemv_triton(x[0], w_u8, mx, rx, my, ry).unsqueeze(0)
    blackwell = torch.cuda.get_device_capability(x.device)[0] >= 12
    row_limit = int(max_gemv_rows) if max_gemv_rows is not None else (16 if blackwell else 4)
    if int(x.shape[0]) > row_limit:
        return mm8_matmul(x, w_u8, mx, rx, my, ry)
    if blackwell and int(x.shape[0]) >= 4 and x.dtype == torch.float16:
        return mm8_batched_dot_triton(x, w_u8, mx, rx, my, ry)
    if blackwell:
        return mm8_batched_gemv_triton(x, w_u8, mx, rx, my, ry)
    # Preserve the measured pre-sm120 route until every older family has
    # an exact-card batched-kernel A/B artifact.
    return torch.stack([mm8_gemv_triton(row, w_u8, mx, rx, my, ry) for row in x], dim=0)


if _HAS_TRITON:

    @triton.jit
    def _mm8_gemv_sk_kernel(
        x_ptr, w_ptr, mx_ptr, rx_ptr, my_ptr, ry_ptr, y_ptr,
        N, M,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Split-K GEMV: grid = (m_tiles, n_chunks); atomic_add reduction.

        Mirrors the official kernel_mm_one_fp16i8 layout (split the N reduction
        across blocks, reduce with atomicAdd) so large layers get enough
        parallelism -- the naive single-program-full-N kernel can be
        parallelism-starved on older accelerators.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        rx_m = tl.load(rx_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        mx_m = tl.load(mx_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        ry_n = tl.load(ry_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        my_n = tl.load(my_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        w_addr = offs_n.to(tl.int64)[:, None] * M + offs_m.to(tl.int64)[None, :]
        w = tl.load(w_ptr + w_addr, mask=mask_n[:, None] & mask_m[None, :], other=0).to(tl.float32)
        deq = (w + 0.5) * ry_n[:, None] * rx_m[None, :] + my_n[:, None] + mx_m[None, :]
        acc = tl.sum(x[:, None] * deq, axis=0)  # [BLOCK_M]
        tl.atomic_add(y_ptr + offs_m, acc, mask=mask_m)


def mm8_gemv_triton_sk(x, w_u8, mx, rx, my, ry, *, block_m=64, block_n=128):
    """Split-K fused int8 dequant GEMV (more parallelism than :func:`mm8_gemv_triton`)."""
    if not (x.is_cuda and mm8_gemv_available(x.device)):
        raise RuntimeError("mm8_gemv_triton_sk requires triton + torch + CUDA input")
    n, m = w_u8.shape
    y = torch.zeros(m, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _mm8_gemv_sk_kernel[grid](
        x, w_u8, _as_1d(mx), _as_1d(rx), _as_1d(my), _as_1d(ry), y,
        n, m, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4,
    )
    return y.to(x.dtype)



# --------------------------------------------------------------------------- #
# Model integration: an int8 (mm8) nn.Linear drop-in + policy-gated replacement.
# The memory policy keeps the historical size gate. The speed policy only swaps
# lm_head, avoiding the cached-decode slowdown from replacing every FFN Linear.
# --------------------------------------------------------------------------- #

class MM8Linear(torch.nn.Module):
    """Drop-in for ``nn.Linear`` storing int8 (mm8) weights + dequant on forward."""

    def __init__(self, linear, *, fused=True):
        super().__init__()
        wu8, mx, rx, my, ry = quantize_mm8(linear.weight.data.t().contiguous())
        self.in_features, self.out_features = linear.weight.shape[1], linear.weight.shape[0]
        self.register_buffer("w_u8", wu8)   # uint8 [in, out]
        self.register_buffer("mx", mx)      # [out]
        self.register_buffer("rx", rx)      # [out]
        self.register_buffer("my", my)      # [in, 1]
        self.register_buffer("ry", ry)      # [in, 1]
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None
        self.fused = bool(fused)

    def forward(self, x):
        if x.dim() == 1:
            if self.fused and x.is_cuda and mm8_gemv_available(x.device):
                y = mm8_gemv_triton(x, self.w_u8, self.mx, self.rx, self.my, self.ry)
            else:
                y = mm8_matmul(x, self.w_u8, self.mx, self.rx, self.my, self.ry)
            if self.bias is not None:
                y = y + self.bias
            return y
        leading = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        if self.fused and x2.is_cuda and mm8_gemv_available(x2.device):
            y = mm8_matmul_triton(x2, self.w_u8, self.mx, self.rx, self.my, self.ry)
        else:
            y = mm8_matmul(x2, self.w_u8, self.mx, self.rx, self.my, self.ry)
        y = y.reshape(*leading, self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, mm8(fused={self.fused})"


def quantize_model_mm8(
    model,
    *,
    min_params: int = 8_000_000,
    fused: bool = True,
    policy: str = "memory",
) -> int:
    """Swap eligible ``nn.Linear`` modules for :class:`MM8Linear`.

    ``policy="memory"`` quantizes every Linear with ``weight.numel() >=
    min_params``. ``policy="speed"`` quantizes only ``lm_head`` after the same
    size gate, keeping per-layer FFN/recurrent decode dense. Set
    ``fused=False`` to force the portable reference path. Returns the number of
    modules replaced.
    """
    if torch is None:
        raise RuntimeError("quantize_model_mm8 requires torch")
    policy = normalize_native_mm_policy(policy)
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and should_quantize_linear(
            name,
            int(mod.weight.numel()),
            min_params=min_params,
            policy=policy,
        ):
            targets.append(name)
    for full_name in targets:
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, MM8Linear(getattr(parent, attr), fused=fused))
    return len(targets)
