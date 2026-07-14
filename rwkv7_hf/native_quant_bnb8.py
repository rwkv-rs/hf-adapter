# coding=utf-8
"""Inference-only fused activation preparation for bitsandbytes W8 GEMM.

The RWKV FFN down projection consumes ``relu(up) ** 2``.  A normal BnB W8
module first materialises that fp16 tensor and then scans it again to compute
row-wise activation scales and int8 values.  The kernels here derive the BnB
activation scales and int8 matrix directly from the pre-activation, avoiding
the intermediate fp16 write/read while retaining BnB's packed weight and
``int8_scaled_mm`` operator.
"""
from __future__ import annotations

try:  # pragma: no cover - optional on CPU-only installs
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised on CUDA/Triton hosts
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    libdevice = None  # type: ignore[assignment]


_HAS_TRITON = triton is not None and tl is not None


if _HAS_TRITON:

    @triton.jit
    def _relu_square_row_scale_kernel(
        x_ptr,
        scale_ptr,
        cols: tl.constexpr,
        BLOCK: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        maximum = 0.0
        for start in range(0, cols, BLOCK):
            col = start + offsets
            values = tl.load(x_ptr + row * cols + col, mask=col < cols, other=0.0).to(tl.float32)
            values = tl.maximum(values, 0.0)
            squared = values * values
            squared = squared.to(tl.bfloat16).to(tl.float32) if IS_BF16 else squared.to(tl.float16).to(tl.float32)
            maximum = tl.maximum(maximum, tl.max(squared, axis=0))
        tl.store(scale_ptr + row, maximum)

    @triton.jit
    def _relu_square_row_quant_kernel(
        x_ptr,
        scale_ptr,
        quant_ptr,
        cols: tl.constexpr,
        BLOCK: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        col = block * BLOCK + tl.arange(0, BLOCK)
        mask = col < cols
        values = tl.load(x_ptr + row * cols + col, mask=mask, other=0.0).to(tl.float32)
        values = tl.maximum(values, 0.0)
        values = values * values
        values = values.to(tl.bfloat16).to(tl.float32) if IS_BF16 else values.to(tl.float16).to(tl.float32)
        scale = tl.load(scale_ptr + row).to(tl.float32)
        # BnB uses symmetric per-row int8 activation quantization.  FFN
        # ReLU² inputs are non-negative, so floor(v + 0.5) implements the same
        # nearest-integer result, including IEEE ties-to-even.
        multiplier = tl.where(scale > 0.0, 127.0 / scale, 0.0)
        quantized = tl.minimum(127.0, libdevice.rint(values * multiplier)).to(tl.int8)
        tl.store(quant_ptr + row * cols + col, quantized, mask=mask)

    @triton.jit
    def _attn_mix4_kernel(
        x_ptr,
        initial_ptr,
        xw_mix_ptr,
        xv_mix_ptr,
        xa_mix_ptr,
        xg_mix_ptr,
        out_w_ptr,
        out_v_ptr,
        out_a_ptr,
        out_g_ptr,
        next_ptr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        total: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        columns = offsets % hidden
        rows = offsets // hidden
        token_ids = rows % tokens
        batch_ids = rows // tokens
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sequence_prev = tl.load(
            x_ptr + offsets - hidden,
            mask=mask & (token_ids > 0),
            other=0.0,
        )
        initial_prev = tl.load(
            initial_ptr + batch_ids * hidden + columns,
            mask=mask & (token_ids == 0),
            other=0.0,
        )
        delta = sequence_prev + initial_prev - values
        mix_w = tl.load(xw_mix_ptr + columns, mask=mask, other=0.0)
        mix_v = tl.load(xv_mix_ptr + columns, mask=mask, other=0.0)
        mix_a = tl.load(xa_mix_ptr + columns, mask=mask, other=0.0)
        mix_g = tl.load(xg_mix_ptr + columns, mask=mask, other=0.0)
        tl.store(out_w_ptr + offsets, values + delta * mix_w, mask=mask)
        tl.store(out_v_ptr + offsets, values + delta * mix_v, mask=mask)
        tl.store(out_a_ptr + offsets, values + delta * mix_a, mask=mask)
        tl.store(out_g_ptr + offsets, values + delta * mix_g, mask=mask)
        tl.store(
            next_ptr + batch_ids * hidden + columns,
            values,
            mask=mask & (token_ids == tokens - 1),
        )

    @triton.jit
    def _rkv_mix_row_scale_kernel(
        x_ptr,
        initial_ptr,
        xr_mix_ptr,
        xk_mix_ptr,
        xv_mix_ptr,
        scale_ptr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        row = tl.program_id(0)
        token_id = row % tokens
        batch_id = row // tokens
        offsets = tl.arange(0, BLOCK)
        max_r = 0.0
        max_k = 0.0
        max_v = 0.0
        for start in range(0, hidden, BLOCK):
            columns = start + offsets
            mask = columns < hidden
            values = tl.load(x_ptr + row * hidden + columns, mask=mask, other=0.0)
            sequence_prev = tl.load(
                x_ptr + (row - 1) * hidden + columns,
                mask=mask & (token_id > 0),
                other=0.0,
            )
            initial_prev = tl.load(
                initial_ptr + batch_id * hidden + columns,
                mask=mask & (token_id == 0),
                other=0.0,
            )
            delta = sequence_prev + initial_prev - values
            mixed_r = values + delta * tl.load(xr_mix_ptr + columns, mask=mask, other=0.0)
            mixed_k = values + delta * tl.load(xk_mix_ptr + columns, mask=mask, other=0.0)
            mixed_v = values + delta * tl.load(xv_mix_ptr + columns, mask=mask, other=0.0)
            if IS_BF16:
                mixed_r = mixed_r.to(tl.bfloat16).to(tl.float32)
                mixed_k = mixed_k.to(tl.bfloat16).to(tl.float32)
                mixed_v = mixed_v.to(tl.bfloat16).to(tl.float32)
            else:
                mixed_r = mixed_r.to(tl.float16).to(tl.float32)
                mixed_k = mixed_k.to(tl.float16).to(tl.float32)
                mixed_v = mixed_v.to(tl.float16).to(tl.float32)
            max_r = tl.maximum(max_r, tl.max(tl.abs(mixed_r), axis=0))
            max_k = tl.maximum(max_k, tl.max(tl.abs(mixed_k), axis=0))
            max_v = tl.maximum(max_v, tl.max(tl.abs(mixed_v), axis=0))
        rows = tl.num_programs(0)
        tl.store(scale_ptr + row, max_r)
        tl.store(scale_ptr + rows + row, max_k)
        tl.store(scale_ptr + 2 * rows + row, max_v)

    @triton.jit
    def _rkv_mix_row_quant_kernel(
        x_ptr,
        initial_ptr,
        xr_mix_ptr,
        xk_mix_ptr,
        xv_mix_ptr,
        scale_ptr,
        quant_ptr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        rows = tl.num_programs(0)
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        mask = columns < hidden
        token_id = row % tokens
        batch_id = row // tokens
        values = tl.load(x_ptr + row * hidden + columns, mask=mask, other=0.0)
        sequence_prev = tl.load(
            x_ptr + (row - 1) * hidden + columns,
            mask=mask & (token_id > 0),
            other=0.0,
        )
        initial_prev = tl.load(
            initial_ptr + batch_id * hidden + columns,
            mask=mask & (token_id == 0),
            other=0.0,
        )
        delta = sequence_prev + initial_prev - values
        mixed_r = values + delta * tl.load(xr_mix_ptr + columns, mask=mask, other=0.0)
        mixed_k = values + delta * tl.load(xk_mix_ptr + columns, mask=mask, other=0.0)
        mixed_v = values + delta * tl.load(xv_mix_ptr + columns, mask=mask, other=0.0)
        if IS_BF16:
            mixed_r = mixed_r.to(tl.bfloat16).to(tl.float32)
            mixed_k = mixed_k.to(tl.bfloat16).to(tl.float32)
            mixed_v = mixed_v.to(tl.bfloat16).to(tl.float32)
        else:
            mixed_r = mixed_r.to(tl.float16).to(tl.float32)
            mixed_k = mixed_k.to(tl.float16).to(tl.float32)
            mixed_v = mixed_v.to(tl.float16).to(tl.float32)
        row_offset = row * hidden + columns
        matrix_stride = rows * hidden
        scale_r = tl.load(scale_ptr + row).to(tl.float32)
        scale_k = tl.load(scale_ptr + rows + row).to(tl.float32)
        scale_v = tl.load(scale_ptr + 2 * rows + row).to(tl.float32)
        quant_r = libdevice.rint(mixed_r * tl.where(scale_r > 0.0, 127.0 / scale_r, 0.0))
        quant_k = libdevice.rint(mixed_k * tl.where(scale_k > 0.0, 127.0 / scale_k, 0.0))
        quant_v = libdevice.rint(mixed_v * tl.where(scale_v > 0.0, 127.0 / scale_v, 0.0))
        quant_r = tl.maximum(tl.minimum(quant_r, 127.0), -127.0).to(tl.int8)
        quant_k = tl.maximum(tl.minimum(quant_k, 127.0), -127.0).to(tl.int8)
        quant_v = tl.maximum(tl.minimum(quant_v, 127.0), -127.0).to(tl.int8)
        tl.store(quant_ptr + row_offset, quant_r, mask=mask)
        tl.store(quant_ptr + matrix_stride + row_offset, quant_k, mask=mask)
        tl.store(quant_ptr + 2 * matrix_stride + row_offset, quant_v, mask=mask)

    @triton.jit
    def _ffn_mix_row_scale_kernel(
        x_ptr,
        initial_ptr,
        mix_ptr,
        scale_ptr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        row = tl.program_id(0)
        token_id = row % tokens
        batch_id = row // tokens
        offsets = tl.arange(0, BLOCK)
        maximum = 0.0
        for start in range(0, hidden, BLOCK):
            columns = start + offsets
            mask = columns < hidden
            values = tl.load(x_ptr + row * hidden + columns, mask=mask, other=0.0)
            sequence_prev = tl.load(
                x_ptr + (row - 1) * hidden + columns,
                mask=mask & (token_id > 0),
                other=0.0,
            )
            initial_prev = tl.load(
                initial_ptr + batch_id * hidden + columns,
                mask=mask & (token_id == 0),
                other=0.0,
            )
            mixed = values + (sequence_prev + initial_prev - values) * tl.load(
                mix_ptr + columns,
                mask=mask,
                other=0.0,
            )
            mixed = mixed.to(tl.bfloat16).to(tl.float32) if IS_BF16 else mixed.to(tl.float16).to(tl.float32)
            maximum = tl.maximum(maximum, tl.max(tl.abs(mixed), axis=0))
        tl.store(scale_ptr + row, maximum)

    @triton.jit
    def _ffn_mix_row_quant_kernel(
        x_ptr,
        initial_ptr,
        mix_ptr,
        scale_ptr,
        quant_ptr,
        next_ptr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        token_id = row % tokens
        batch_id = row // tokens
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        mask = columns < hidden
        values = tl.load(x_ptr + row * hidden + columns, mask=mask, other=0.0)
        sequence_prev = tl.load(
            x_ptr + (row - 1) * hidden + columns,
            mask=mask & (token_id > 0),
            other=0.0,
        )
        initial_prev = tl.load(
            initial_ptr + batch_id * hidden + columns,
            mask=mask & (token_id == 0),
            other=0.0,
        )
        mixed = values + (sequence_prev + initial_prev - values) * tl.load(
            mix_ptr + columns,
            mask=mask,
            other=0.0,
        )
        mixed = mixed.to(tl.bfloat16).to(tl.float32) if IS_BF16 else mixed.to(tl.float16).to(tl.float32)
        scale = tl.load(scale_ptr + row).to(tl.float32)
        multiplier = tl.where(scale > 0.0, 127.0 / scale, 0.0)
        quantized = libdevice.rint(mixed * multiplier)
        quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0).to(tl.int8)
        tl.store(quant_ptr + row * hidden + columns, quantized, mask=mask)
        tl.store(
            next_ptr + batch_id * hidden + columns,
            values,
            mask=mask & (token_id == tokens - 1),
        )


def fused_bnb8_relu_square_quant_available() -> bool:
    return bool(_HAS_TRITON and torch is not None)


def fused_bnb8_relu_square_quant(x, *, block: int = 1024):
    """Return BnB-compatible ``(int8_values, fp32_row_scales)``.

    ``x`` is the FFN pre-activation and may have any leading dimensions.  The
    final dimension is quantized independently for each flattened row.
    """

    if torch is None:
        raise RuntimeError("fused_bnb8_relu_square_quant requires torch")
    if (
        not fused_bnb8_relu_square_quant_available()
        or not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16)
        or x.dim() < 2
    ):
        raise RuntimeError("fused BnB8 ReLU-square quantization is unavailable for this input")
    source = x.contiguous()
    cols = int(source.shape[-1])
    rows = int(source.numel() // cols)
    block = min(max(256, int(block)), 4096)
    if block & (block - 1):
        raise ValueError("block must be a power of two")
    quantized = torch.empty((rows, cols), device=source.device, dtype=torch.int8)
    scales = torch.empty((rows,), device=source.device, dtype=torch.float32)
    _relu_square_row_scale_kernel[(rows,)](
        source,
        scales,
        cols=cols,
        BLOCK=block,
        IS_BF16=source.dtype == torch.bfloat16,
        num_warps=8,
    )
    _relu_square_row_quant_kernel[(rows, triton.cdiv(cols, block))](
        source,
        scales,
        quantized,
        cols=cols,
        BLOCK=block,
        IS_BF16=source.dtype == torch.bfloat16,
        num_warps=4,
    )
    return quantized, scales


def fused_bnb8_attn_sequence_mix_quant(
    x,
    initial_prev,
    r_mix,
    w_mix,
    k_mix,
    v_mix,
    a_mix,
    g_mix,
    *,
    block: int = 1024,
    mix_workspace=None,
    quant_workspace=None,
    scale_workspace=None,
):
    """Shift-mix a sequence and directly quantize its R/K/V streams.

    W/V/A/G remain materialized because the low-rank state/gate branches also
    consume them.  R and K never hit global memory in fp16, and V is quantized
    while its source sequence is already resident, reducing the six-stream
    shift-mix plus three independent BnB quant scans to one combined route.
    """

    if torch is None or not fused_bnb8_relu_square_quant_available():
        raise RuntimeError("fused BnB8 R/K/V mix quantization is unavailable")
    if x.dim() != 3 or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("fused BnB8 R/K/V mix quantization requires CUDA [B,T,C] fp16/bf16 input")
    batch, tokens, hidden = map(int, x.shape)
    source = x.contiguous()
    initial = initial_prev.reshape(batch, hidden).contiguous()
    mixes = tuple(item.reshape(-1).contiguous() for item in (r_mix, w_mix, k_mix, v_mix, a_mix, g_mix))
    if any(int(item.numel()) != hidden or item.dtype != x.dtype for item in mixes):
        raise ValueError("all attention mix vectors must match the hidden size and dtype")
    expected_mix = (4, batch, tokens, hidden)
    if (
        mix_workspace is None
        or tuple(mix_workspace.shape) != expected_mix
        or mix_workspace.dtype != x.dtype
        or mix_workspace.device != x.device
        or not mix_workspace.is_contiguous()
    ):
        mix_workspace = torch.empty(expected_mix, device=x.device, dtype=x.dtype)
    rows = batch * tokens
    expected_quant = (3, rows, hidden)
    if (
        quant_workspace is None
        or tuple(quant_workspace.shape) != expected_quant
        or quant_workspace.dtype != torch.int8
        or quant_workspace.device != x.device
        or not quant_workspace.is_contiguous()
    ):
        quant_workspace = torch.empty(expected_quant, device=x.device, dtype=torch.int8)
    expected_scale = (3, rows)
    if (
        scale_workspace is None
        or tuple(scale_workspace.shape) != expected_scale
        or scale_workspace.dtype != torch.float32
        or scale_workspace.device != x.device
        or not scale_workspace.is_contiguous()
    ):
        scale_workspace = torch.empty(expected_scale, device=x.device, dtype=torch.float32)
    next_state = torch.empty((batch, hidden), device=x.device, dtype=x.dtype)
    total = int(source.numel())
    mix_w, mix_v, mix_a, mix_g = mix_workspace.unbind(0)
    _attn_mix4_kernel[(triton.cdiv(total, 256),)](
        source,
        initial,
        mixes[1],
        mixes[3],
        mixes[4],
        mixes[5],
        mix_w,
        mix_v,
        mix_a,
        mix_g,
        next_state,
        hidden=hidden,
        tokens=tokens,
        total=total,
        BLOCK=256,
        num_warps=4,
    )
    block = min(max(256, int(block)), 4096)
    if block & (block - 1):
        raise ValueError("block must be a power of two")
    _rkv_mix_row_scale_kernel[(rows,)](
        source,
        initial,
        mixes[0],
        mixes[2],
        mixes[3],
        scale_workspace,
        hidden=hidden,
        tokens=tokens,
        BLOCK=block,
        IS_BF16=x.dtype == torch.bfloat16,
        num_warps=8,
    )
    _rkv_mix_row_quant_kernel[(rows, triton.cdiv(hidden, block))](
        source,
        initial,
        mixes[0],
        mixes[2],
        mixes[3],
        scale_workspace,
        quant_workspace,
        hidden=hidden,
        tokens=tokens,
        BLOCK=block,
        IS_BF16=x.dtype == torch.bfloat16,
        num_warps=4,
    )
    return (
        quant_workspace[0], scale_workspace[0],
        quant_workspace[1], scale_workspace[1],
        quant_workspace[2], scale_workspace[2],
        mix_w, mix_v, mix_a, mix_g, next_state,
        mix_workspace, quant_workspace, scale_workspace,
    )


def fused_bnb8_ffn_sequence_mix_quant(
    x,
    initial_prev,
    mix,
    *,
    block: int = 1024,
    quant_workspace=None,
    scale_workspace=None,
):
    """Directly row-quantize the sequence FFN shift-mix stream."""

    if torch is None or not fused_bnb8_relu_square_quant_available():
        raise RuntimeError("fused BnB8 FFN mix quantization is unavailable")
    if x.dim() != 3 or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("fused BnB8 FFN mix quantization requires CUDA [B,T,C] fp16/bf16 input")
    batch, tokens, hidden = map(int, x.shape)
    rows = batch * tokens
    source = x.contiguous()
    initial = initial_prev.reshape(batch, hidden).contiguous()
    mix = mix.reshape(-1).contiguous()
    if int(mix.numel()) != hidden or mix.dtype != x.dtype:
        raise ValueError("FFN mix vector must match hidden size and dtype")
    if (
        quant_workspace is None
        or tuple(quant_workspace.shape) != (rows, hidden)
        or quant_workspace.dtype != torch.int8
        or quant_workspace.device != x.device
        or not quant_workspace.is_contiguous()
    ):
        quant_workspace = torch.empty((rows, hidden), device=x.device, dtype=torch.int8)
    if (
        scale_workspace is None
        or tuple(scale_workspace.shape) != (rows,)
        or scale_workspace.dtype != torch.float32
        or scale_workspace.device != x.device
        or not scale_workspace.is_contiguous()
    ):
        scale_workspace = torch.empty((rows,), device=x.device, dtype=torch.float32)
    next_state = torch.empty((batch, hidden), device=x.device, dtype=x.dtype)
    block = min(max(256, int(block)), 4096)
    if block & (block - 1):
        raise ValueError("block must be a power of two")
    _ffn_mix_row_scale_kernel[(rows,)](
        source,
        initial,
        mix,
        scale_workspace,
        hidden=hidden,
        tokens=tokens,
        BLOCK=block,
        IS_BF16=x.dtype == torch.bfloat16,
        num_warps=8,
    )
    _ffn_mix_row_quant_kernel[(rows, triton.cdiv(hidden, block))](
        source,
        initial,
        mix,
        scale_workspace,
        quant_workspace,
        next_state,
        hidden=hidden,
        tokens=tokens,
        BLOCK=block,
        IS_BF16=x.dtype == torch.bfloat16,
        num_warps=4,
    )
    return quant_workspace, scale_workspace, next_state
