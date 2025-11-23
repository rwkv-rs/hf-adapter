# coding=utf-8
"""MLX packed W8/W4 affine dequant-matmul helpers for RWKV-7.

This is the Apple-side sibling of :mod:`rwkv7_hf.native_quant_mm8` and
:mod:`rwkv7_hf.native_quant_mm4`.  It intentionally exposes two execution
styles:

``reference``
    Materialize the approximate dense dequantized matrix, then call MLX matmul.
    This is useful for tests and formula validation.

``affine``
    Compute the same affine quantized matmul without materializing the fp16/fp32
    dequantized weight matrix.  It is still written in portable MLX ops rather
    than a custom Metal kernel, but it is the stable speed-path seam that a
    future fused Metal W8/W4 projection kernel can replace.

``metal``
    Use an optional custom MLX/Metal kernel that fuses dequantization and the
    projection dot product.  Quantized weights are stored in a Metal-friendly
    transposed packed layout so each output column reads contiguous bytes.  This
    is the first Apple W8/W4 fused-kernel seam; it remains opt-in while the
    production speed path is tuned across model sizes and Apple GPUs.

``groupwise``
    Use MLX's native packed groupwise affine weights and fused
    ``quantized_matmul``.  This is the production Apple W8/W4 route: it avoids
    dense dequantized weights, reduces resident model memory, and reaches the
    optimized MLX GPU kernels used by current Apple inference stacks.

Weights use the same layout as the native Torch/CUDA helpers: quantize
``W: [N, M]`` used as ``y = x @ W``.  For an HF/torch Linear weight shaped
``[out, in]``, pass ``weight.T`` so ``N=in`` and ``M=out``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
from typing import Any, Sequence

from .mlx_bridge import mlx_array_nbytes, mlx_available, require_mlx


_EPS = 1e-8


def _mx():
    return require_mlx()


def _weight_dtype(weight: Any):
    return getattr(weight, "dtype", None)


@lru_cache(maxsize=1)
def metal_quant_available() -> bool:
    """Return whether MLX custom Metal kernels are available for quant matmul."""

    if not mlx_available():
        return False
    try:
        mx = require_mlx()
        return bool(hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel"))
    except Exception:
        return False


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _select_auto_backend(bits: int, rows: int) -> str:
    """Choose a safe MLX quant projection backend for ``backend="auto"``.

    W4/Metal is batch-exact in the current session and long prompt/decode gates,
    and the real generation rows favor the fused path over the affine fallback,
    so auto uses Metal for normal prefill/decode row counts and only falls back
    to affine above the configurable row limit.

    W8/Metal still has a known session-batch exactness gap.  Keep W8 auto on
    the affine path by default; developers can opt into row-1 Metal while
    investigating with ``RWKV7_MLX_QUANT_AUTO_W8_METAL_MAX_ROWS=1``.
    """

    if not metal_quant_available():
        return "affine"
    if int(bits) == 4:
        max_rows = _env_int("RWKV7_MLX_QUANT_AUTO_W4_METAL_MAX_ROWS", 4096)
    elif int(bits) == 8:
        max_rows = _env_int("RWKV7_MLX_QUANT_AUTO_W8_METAL_MAX_ROWS", 0)
    else:
        max_rows = 0
    return "metal" if max_rows > 0 and int(rows) <= max_rows else "affine"


def _affine_minmax(weight: Any):
    """Return ``(w_norm, mx, rx, my, ry)`` for the RWKV affine quantizer."""

    mx = _mx()
    w = weight.astype(mx.float32)
    n, m = int(w.shape[0]), int(w.shape[1])
    if n > m:
        my = mx.min(w, axis=1, keepdims=True)
        w = w - my
        mx_col = mx.min(w, axis=0)
        w = w - mx_col
    else:
        mx_col = mx.min(w, axis=0)
        w = w - mx_col
        my = mx.min(w, axis=1, keepdims=True)
        w = w - my
    rx = mx.maximum(mx.max(w, axis=0), _EPS)
    w = w / rx
    ry = mx.maximum(mx.max(w, axis=1, keepdims=True), _EPS)
    w = w / ry
    return w, mx_col, rx, my, ry


@dataclass
class MLXMM8Weight:
    """Packed int8 affine weight for ``y = x @ W``."""

    w_u8: Any | None
    mx: Any
    rx: Any
    my: Any
    ry: Any
    n: int
    m: int
    dense_dtype: Any
    w_u8_t: Any | None = None

    @property
    def bits(self) -> int:
        return 8

    @property
    def storage_bytes(self) -> int:
        return sum(
            mlx_array_nbytes(x)
            for x in (self.w_u8, self.w_u8_t, self.mx, self.rx, self.my, self.ry)
            if x is not None
        )


@dataclass
class MLXMM8GroupWeight:
    """Prepacked grouped MM8 weights for one-launch Metal projection groups."""

    q_t: Any
    mx: Any
    rx: Any
    my: Any
    ry: Any
    n: int
    m: int
    groups: int
    dense_dtype: Any

    @property
    def bits(self) -> int:
        return 8

    @property
    def storage_bytes(self) -> int:
        return sum(mlx_array_nbytes(x) for x in (self.q_t, self.mx, self.rx, self.my, self.ry))


@dataclass
class MLXMM4Weight:
    """Packed int4 affine weight for ``y = x @ W``."""

    packed: Any | None
    mx: Any
    rx_s: Any
    my: Any
    ry_s: Any
    n: int
    m_orig: int
    m_padded: int
    dense_dtype: Any
    packed_t: Any | None = None

    @property
    def bits(self) -> int:
        return 4

    @property
    def storage_bytes(self) -> int:
        return sum(
            mlx_array_nbytes(x)
            for x in (self.packed, self.packed_t, self.mx, self.rx_s, self.my, self.ry_s)
            if x is not None
        )


@dataclass
class MLXMM4GroupWeight:
    """Prepacked grouped MM4 weights for one-launch Metal projection groups."""

    packed_t: Any
    mx: Any
    rx_s: Any
    my: Any
    ry_s: Any
    n: int
    m_orig: int
    m_padded: int
    groups: int
    dense_dtype: Any

    @property
    def bits(self) -> int:
        return 4

    @property
    def storage_bytes(self) -> int:
        return sum(mlx_array_nbytes(x) for x in (self.packed_t, self.mx, self.rx_s, self.my, self.ry_s))


@dataclass
class MLXGroupwiseWeight:
    """MLX-native groupwise affine weight used by ``mx.quantized_matmul``."""

    w_q: Any
    scales: Any
    biases: Any
    n: int
    m: int
    quant_bits: int
    group_size: int
    dense_dtype: Any

    @property
    def bits(self) -> int:
        return int(self.quant_bits)

    @property
    def storage_bytes(self) -> int:
        return sum(mlx_array_nbytes(x) for x in (self.w_q, self.scales, self.biases))


def quantize_mlx_groupwise_linear(
    dense_weight: Any,
    *,
    bits: int,
    group_size: int = 64,
) -> MLXGroupwiseWeight:
    """Quantize ``Linear.weight [out, in]`` in MLX's fused native layout."""

    mx = _mx()
    out_features, in_features = int(dense_weight.shape[0]), int(dense_weight.shape[1])
    if in_features % int(group_size):
        raise ValueError(
            f"groupwise quantization requires in_features divisible by {group_size}; got {in_features}"
        )
    w_q, scales, biases = mx.quantize(
        dense_weight,
        group_size=int(group_size),
        bits=int(bits),
        mode="affine",
    )
    mx.eval(w_q, scales, biases)
    return MLXGroupwiseWeight(
        w_q=w_q,
        scales=scales,
        biases=biases,
        n=in_features,
        m=out_features,
        quant_bits=int(bits),
        group_size=int(group_size),
        dense_dtype=dense_weight.dtype,
    )


@lru_cache(maxsize=1)
def _metal_groupwise_embedding_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are unavailable")
    source = r'''
        uint index = thread_position_in_grid.x;
        uint rows = uint(dims[0]);
        uint vocab = uint(dims[1]);
        uint hidden = uint(dims[2]);
        uint groups = uint(dims[3]);
        uint total = rows * hidden;
        if (index >= total) {
            return;
        }

        uint row = index / hidden;
        uint col = index - row * hidden;
        uint token = uint(ids[row]);
        if (token >= vocab) {
            out[index] = 0.0f;
            return;
        }
        constexpr uint pack_factor = 32 / BITS;
        constexpr uint mask = (1u << BITS) - 1u;
        uint packed_cols = hidden / pack_factor;
        uint word = weight[token * packed_cols + col / pack_factor];
        uint q = (word >> ((col % pack_factor) * BITS)) & mask;
        uint group = col / GROUP_SIZE;
        uint affine_index = token * groups + group;
        out[index] = float(q) * float(scales[affine_index])
                   + float(biases[affine_index]);
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_groupwise_embedding",
        input_names=["weight", "scales", "biases", "ids", "dims"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def groupwise_embedding(
    token_ids: Any,
    weight: MLXGroupwiseWeight,
    *,
    backend: str = "auto",
) -> tuple[Any, str]:
    """Gather and dequantize native MLX W4/W8 embedding rows.

    The Metal route performs packed lookup and affine dequantization in one
    launch.  This avoids the three indexed gathers plus generic dequant graph
    that otherwise dominate batched recurrent decode.
    """

    mx = _mx()
    choice = (backend or "auto").lower().strip()
    if choice not in {"auto", "metal", "reference"}:
        raise ValueError("groupwise embedding backend must be auto, metal, or reference")
    use_metal = choice == "metal" or (choice == "auto" and metal_quant_available())
    ids = token_ids.astype(mx.int32)
    if not use_metal:
        out = mx.dequantize(
            weight.w_q[ids],
            scales=weight.scales[ids],
            biases=weight.biases[ids],
            group_size=int(weight.group_size),
            bits=int(weight.bits),
            mode="affine",
            dtype=weight.dense_dtype,
        )
        return out, "reference"
    rows = int(ids.size)
    dims = mx.array([rows, int(weight.w_q.shape[0]), int(weight.n), int(weight.scales.shape[1])], dtype=mx.uint32)
    (out,) = _metal_groupwise_embedding_kernel()(
        inputs=[weight.w_q, weight.scales, weight.biases, ids.reshape(-1), dims],
        template=[("BITS", int(weight.bits)), ("GROUP_SIZE", int(weight.group_size))],
        grid=(rows * int(weight.n), 1, 1),
        threadgroup=(min(256, max(1, int(weight.n))), 1, 1),
        output_shapes=[(*ids.shape, int(weight.n))],
        output_dtypes=[weight.dense_dtype],
    )
    return out, "metal"


@dataclass
class MLXQuantizedLinear:
    """Quantized MLX Linear weight with reference, affine, Metal, and auto backends."""

    weight: MLXMM8Weight | MLXMM4Weight | MLXGroupwiseWeight
    backend: str = "affine"
    auto_metal_max_rows: int = 0
    last_backend: str | None = None
    backend_counts: dict[str, int] = field(
        default_factory=lambda: {"reference": 0, "affine": 0, "metal": 0, "groupwise": 0}
    )

    @property
    def bits(self) -> int:
        return int(self.weight.bits)

    @property
    def in_features(self) -> int:
        return int(self.weight.n)

    @property
    def out_features(self) -> int:
        if isinstance(self.weight, MLXMM4Weight):
            return int(self.weight.m_orig)
        return int(self.weight.m)

    @property
    def storage_bytes(self) -> int:
        return int(self.weight.storage_bytes)

    @classmethod
    def from_linear_weight(
        cls,
        dense_weight: Any,
        *,
        bits: int,
        backend: str = "affine",
        group_size: int = 64,
    ) -> "MLXQuantizedLinear":
        """Quantize an MLX Linear ``weight [out, in]`` for ``linear(x, weight)``."""

        backend = (backend or "affine").lower().strip()
        if backend not in {"reference", "affine", "metal", "auto", "groupwise"}:
            raise ValueError(
                f"unsupported MLX quant backend {backend!r}; expected reference, affine, metal, auto, or groupwise"
            )
        if backend == "groupwise":
            return cls(
                quantize_mlx_groupwise_linear(
                    dense_weight,
                    bits=int(bits),
                    group_size=int(group_size),
                ),
                backend=backend,
            )
        auto_metal_max_rows = 0
        if backend == "auto" and metal_quant_available():
            if int(bits) == 4:
                auto_metal_max_rows = _env_int("RWKV7_MLX_QUANT_AUTO_W4_METAL_MAX_ROWS", 4096)
            elif int(bits) == 8:
                auto_metal_max_rows = _env_int("RWKV7_MLX_QUANT_AUTO_W8_METAL_MAX_ROWS", 0)
        layout = "metal" if backend == "metal" or auto_metal_max_rows > 0 else "standard"
        if bits == 8:
            return cls(quantize_mlx_mm8(dense_weight.T, layout=layout), backend=backend, auto_metal_max_rows=auto_metal_max_rows)
        if bits == 4:
            return cls(quantize_mlx_mm4(dense_weight.T, layout=layout), backend=backend, auto_metal_max_rows=auto_metal_max_rows)
        raise ValueError(f"unsupported MLX quant bits {bits}; expected 8 or 4")

    def _selected_backend(self, x: Any) -> str:
        backend = (self.backend or "affine").lower().strip()
        if backend == "auto":
            rows = int(x.reshape(-1, self.in_features).shape[0])
            return "metal" if self.auto_metal_max_rows > 0 and rows <= self.auto_metal_max_rows else "affine"
        if backend in {"reference", "affine", "metal", "groupwise"}:
            return backend
        raise ValueError(
            f"unsupported MLX quant backend {self.backend!r}; expected reference, affine, metal, auto, or groupwise"
        )

    def __call__(self, x: Any) -> Any:
        backend = self._selected_backend(x)
        if isinstance(self.weight, MLXGroupwiseWeight):
            mx = _mx()
            y = mx.quantized_matmul(
                x,
                self.weight.w_q,
                scales=self.weight.scales,
                biases=self.weight.biases,
                transpose=True,
                group_size=int(self.weight.group_size),
                bits=int(self.weight.bits),
                mode="affine",
            )
        elif isinstance(self.weight, MLXMM8Weight):
            y = mm8_matmul_mlx(x, self.weight, backend=backend)
        else:
            y = mm4_matmul_mlx(x, self.weight, backend=backend)
        self.last_backend = backend
        self.backend_counts[backend] = int(self.backend_counts.get(backend, 0)) + 1
        return y

    def relu2(self, x: Any) -> Any:
        """Run ``relu(linear(x)) ** 2`` with a fused MM4/Metal fast path."""

        mx = _mx()
        backend = self._selected_backend(x)
        if isinstance(self.weight, MLXMM4Weight) and backend == "metal":
            y = mm4_matmul_relu2_metal(x, self.weight)
            self.last_backend = backend
            self.backend_counts[backend] = int(self.backend_counts.get(backend, 0)) + 1
            return y
        y = self(x)
        y = mx.maximum(y, 0)
        return y * y

    def telemetry(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "backend": self.backend,
            "auto_metal_max_rows": int(self.auto_metal_max_rows),
            "last_backend": self.last_backend,
            "backend_counts": dict(self.backend_counts),
            "in_features": self.in_features,
            "out_features": self.out_features,
            "storage_bytes": self.storage_bytes,
        }


def _mm8_u8(q: MLXMM8Weight) -> Any:
    mx = _mx()
    if q.w_u8 is not None:
        return q.w_u8
    if q.w_u8_t is None:
        raise ValueError("MLXMM8Weight has neither standard nor transposed uint8 storage")
    return mx.transpose(q.w_u8_t)


def _mm8_u8_t(q: MLXMM8Weight) -> Any:
    mx = _mx()
    if q.w_u8_t is not None:
        return q.w_u8_t
    if q.w_u8 is None:
        raise ValueError("MLXMM8Weight has neither standard nor transposed uint8 storage")
    return mx.contiguous(mx.transpose(q.w_u8))


def quantize_mlx_mm8(weight: Any, *, layout: str = "standard") -> MLXMM8Weight:
    """Quantize ``weight [N, M]`` into the RWKV affine int8 layout."""

    mx = _mx()
    layout = (layout or "standard").lower().strip()
    if layout not in {"standard", "metal"}:
        raise ValueError(f"unsupported MLX mm8 layout {layout!r}; expected standard or metal")
    dense_dtype = _weight_dtype(weight)
    w, mx_col, rx, my, ry = _affine_minmax(weight)
    w_u8 = mx.clip(mx.floor(w * 256.0), 0, 255).astype(mx.uint8)
    w_u8_t = mx.contiguous(mx.transpose(w_u8)) if layout == "metal" else None
    out_dtype = dense_dtype or mx.float16
    q = MLXMM8Weight(
        w_u8=None if layout == "metal" else w_u8,
        mx=mx_col.astype(out_dtype),
        rx=(rx / 16.0).astype(out_dtype),
        my=my.astype(out_dtype),
        ry=(ry / 16.0).astype(out_dtype),
        n=int(weight.shape[0]),
        m=int(weight.shape[1]),
        dense_dtype=out_dtype,
        w_u8_t=w_u8_t,
    )
    mx.eval(*[x for x in (q.w_u8, q.w_u8_t, q.mx, q.rx, q.my, q.ry) if x is not None])
    return q


def pack_mlx_mm8_group(weights: Sequence[MLXMM8Weight]) -> MLXMM8GroupWeight:
    """Prepack equal-shaped MM8 weights for grouped Metal projection."""

    mx = _mx()
    qs = list(weights)
    if not qs:
        raise ValueError("pack_mlx_mm8_group requires at least one weight")
    n, m = int(qs[0].n), int(qs[0].m)
    if any(int(q.n) != n or int(q.m) != m for q in qs):
        raise ValueError("all grouped MM8 weights must have the same [N, M] shape")
    out = MLXMM8GroupWeight(
        q_t=mx.stack([_mm8_u8_t(q) for q in qs], axis=0),
        mx=mx.stack([q.mx.reshape(m) for q in qs], axis=0),
        rx=mx.stack([q.rx.reshape(m) for q in qs], axis=0),
        my=mx.stack([q.my.reshape(n) for q in qs], axis=0),
        ry=mx.stack([q.ry.reshape(n) for q in qs], axis=0),
        n=n,
        m=m,
        groups=len(qs),
        dense_dtype=qs[0].dense_dtype,
    )
    mx.eval(out.q_t, out.mx, out.rx, out.my, out.ry)
    return out


def dequantize_mlx_mm8(q: MLXMM8Weight, *, out_dtype: Any | None = None) -> Any:
    mx = _mx()
    dtype = out_dtype or q.dense_dtype
    return (_mm8_u8(q).astype(dtype) + 0.5) * q.ry * q.rx + q.my + q.mx


@lru_cache(maxsize=1)
def _metal_mm8_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint R = uint(dims[0]);
        uint N = uint(dims[1]);
        uint M = uint(dims[2]);
        uint total = R * M;
        if (row_id >= total) {
            return;
        }

        uint r_id = row_id / M;
        uint m_id = row_id - r_id * M;
        uint x_base = r_id * N;
        uint q_base = m_id * N;

        float rx_m = float(rx[m_id]);
        float mx_m = float(mx_col[m_id]);
        float acc = 0.0f;
        for (uint n = 0; n < N; ++n) {
            float xv = float(x[x_base + n]);
            float qv = float(q_t[q_base + n]) + 0.5f;
            float deq = qv * float(ry[n]) * rx_m + float(my[n]) + mx_m;
            acc += xv * deq;
        }
        out[row_id] = acc;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_mm8_affine_matmul",
        input_names=["x", "q_t", "mx_col", "rx", "my", "ry", "dims"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def mm8_matmul_metal(x: Any, q: MLXMM8Weight) -> Any:
    """Run fused Metal ``x @ dequant(q)`` for MM8 affine weights."""

    mx = _mx()
    x2 = x.reshape(-1, q.n)
    rows = int(x2.shape[0])
    dims = mx.array([rows, q.n, q.m], dtype=mx.uint32)
    out = _metal_mm8_kernel()(
        inputs=[
            x2,
            _mm8_u8_t(q),
            q.mx.reshape(q.m),
            q.rx.reshape(q.m),
            q.my.reshape(q.n),
            q.ry.reshape(q.n),
            dims,
        ],
        grid=(rows * q.m, 1, 1),
        threadgroup=(min(256, max(1, q.m)), 1, 1),
        output_shapes=[(rows, q.m)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*x.shape[:-1], q.m)


@lru_cache(maxsize=1)
def _metal_mm8_group_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint R = uint(dims[0]);
        uint N = uint(dims[1]);
        uint M = uint(dims[2]);
        uint G = uint(dims[3]);
        uint total = G * R * M;
        if (row_id >= total) {
            return;
        }

        uint m_id = row_id % M;
        uint tmp = row_id / M;
        uint r_id = tmp % R;
        uint g_id = tmp / R;
        uint x_base = r_id * N;
        uint q_base = (g_id * M + m_id) * N;
        uint col_base = g_id * M + m_id;
        uint row_base = g_id * N;

        float rx_m = float(rx[col_base]);
        float mx_m = float(mx_col[col_base]);
        float acc = 0.0f;
        for (uint n = 0; n < N; ++n) {
            float xv = float(x[x_base + n]);
            float qv = float(q_t[q_base + n]) + 0.5f;
            float deq = qv * float(ry[row_base + n]) * rx_m + float(my[row_base + n]) + mx_m;
            acc += xv * deq;
        }
        out[row_id] = acc;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_mm8_affine_group_matmul",
        input_names=["x", "q_t", "mx_col", "rx", "my", "ry", "dims"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def mm8_group_matmul_metal(x: Any, weights: Sequence[MLXMM8Weight] | MLXMM8GroupWeight) -> Any:
    """Run grouped fused Metal MM8 projections for equal-shaped weights.

    Returns ``[groups, *x.shape[:-1], out_features]``. This experimental
    launch-fusion seam targets decode-hot groups such as R/K/V projections; it
    does not change the default single-projection path.
    """

    mx = _mx()
    group = weights if isinstance(weights, MLXMM8GroupWeight) else pack_mlx_mm8_group(weights)
    n, m = int(group.n), int(group.m)
    x2 = x.reshape(-1, n)
    rows = int(x2.shape[0])
    groups = int(group.groups)
    dims = mx.array([rows, n, m, groups], dtype=mx.uint32)
    out = _metal_mm8_group_kernel()(
        inputs=[x2, group.q_t, group.mx, group.rx, group.my, group.ry, dims],
        grid=(groups * rows * m, 1, 1),
        threadgroup=(min(256, max(1, m)), 1, 1),
        output_shapes=[(groups, rows, m)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(groups, *x.shape[:-1], m)


@lru_cache(maxsize=1)
def _metal_mm8_group_inputs_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint R = uint(dims[0]);
        uint N = uint(dims[1]);
        uint M = uint(dims[2]);
        uint G = uint(dims[3]);
        uint total = G * R * M;
        if (row_id >= total) {
            return;
        }

        uint m_id = row_id % M;
        uint tmp = row_id / M;
        uint r_id = tmp % R;
        uint g_id = tmp / R;
        uint x_base = (g_id * R + r_id) * N;
        uint q_base = (g_id * M + m_id) * N;
        uint col_base = g_id * M + m_id;
        uint row_base = g_id * N;

        float rx_m = float(rx[col_base]);
        float mx_m = float(mx_col[col_base]);
        float acc = 0.0f;
        for (uint n = 0; n < N; ++n) {
            float xv = float(x_group[x_base + n]);
            float qv = float(q_t[q_base + n]) + 0.5f;
            float deq = qv * float(ry[row_base + n]) * rx_m + float(my[row_base + n]) + mx_m;
            acc += xv * deq;
        }
        out[row_id] = acc;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_mm8_affine_group_inputs_matmul",
        input_names=["x_group", "q_t", "mx_col", "rx", "my", "ry", "dims"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def mm8_group_matmul_metal_inputs(x_group: Any, weights: Sequence[MLXMM8Weight] | MLXMM8GroupWeight) -> Any:
    """Run grouped MM8 Metal projections with one input tensor per group.

    ``x_group`` must be shaped ``[groups, *batch_shape, in_features]``. Returns
    ``[groups, *batch_shape, out_features]``.
    """

    mx = _mx()
    group = weights if isinstance(weights, MLXMM8GroupWeight) else pack_mlx_mm8_group(weights)
    groups, n, m = int(group.groups), int(group.n), int(group.m)
    if int(x_group.shape[0]) != groups or int(x_group.shape[-1]) != n:
        raise ValueError(
            f"x_group must be [groups, ..., {n}] with groups={groups}; got {tuple(x_group.shape)}"
        )
    x2 = x_group.reshape(groups, -1, n)
    rows = int(x2.shape[1])
    dims = mx.array([rows, n, m, groups], dtype=mx.uint32)
    out = _metal_mm8_group_inputs_kernel()(
        inputs=[x2, group.q_t, group.mx, group.rx, group.my, group.ry, dims],
        grid=(groups * rows * m, 1, 1),
        threadgroup=(min(256, max(1, m)), 1, 1),
        output_shapes=[(groups, rows, m)],
        output_dtypes=[x_group.dtype],
    )[0]
    return out.reshape(groups, *x_group.shape[1:-1], m)


def _as_mm8_triple(weights: Sequence[MLXMM8Weight]) -> tuple[MLXMM8Weight, MLXMM8Weight, MLXMM8Weight]:
    qs = tuple(weights)
    if len(qs) != 3:
        raise ValueError(f"expected exactly three MM8 weights, got {len(qs)}")
    n, m = int(qs[0].n), int(qs[0].m)
    if any(int(q.n) != n or int(q.m) != m for q in qs):
        raise ValueError("all triple MM8 weights must have the same [N, M] shape")
    return qs  # type: ignore[return-value]


@lru_cache(maxsize=1)
def _metal_mm8_triple_inputs_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint R = uint(dims[0]);
        uint N = uint(dims[1]);
        uint M = uint(dims[2]);
        uint total = 3 * R * M;
        if (row_id >= total) {
            return;
        }

        uint m_id = row_id % M;
        uint tmp = row_id / M;
        uint r_id = tmp % R;
        uint g_id = tmp / R;
        uint x_base = r_id * N;
        uint q_base = m_id * N;

        float acc = 0.0f;
        if (g_id == 0) {
            float rx_m = float(rx0[m_id]);
            float mx_m = float(mx0[m_id]);
            for (uint n = 0; n < N; ++n) {
                float xv = float(x0[x_base + n]);
                float qv = float(q0_t[q_base + n]) + 0.5f;
                float deq = qv * float(ry0[n]) * rx_m + float(my0[n]) + mx_m;
                acc += xv * deq;
            }
        } else if (g_id == 1) {
            float rx_m = float(rx1[m_id]);
            float mx_m = float(mx1[m_id]);
            for (uint n = 0; n < N; ++n) {
                float xv = float(x1[x_base + n]);
                float qv = float(q1_t[q_base + n]) + 0.5f;
                float deq = qv * float(ry1[n]) * rx_m + float(my1[n]) + mx_m;
                acc += xv * deq;
            }
        } else {
            float rx_m = float(rx2[m_id]);
            float mx_m = float(mx2[m_id]);
            for (uint n = 0; n < N; ++n) {
                float xv = float(x2[x_base + n]);
                float qv = float(q2_t[q_base + n]) + 0.5f;
                float deq = qv * float(ry2[n]) * rx_m + float(my2[n]) + mx_m;
                acc += xv * deq;
            }
        }
        out[row_id] = acc;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_mm8_affine_triple_inputs_matmul",
        input_names=[
            "x0",
            "x1",
            "x2",
            "q0_t",
            "q1_t",
            "q2_t",
            "mx0",
            "rx0",
            "my0",
            "ry0",
            "mx1",
            "rx1",
            "my1",
            "ry1",
            "mx2",
            "rx2",
            "my2",
            "ry2",
            "dims",
        ],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def mm8_triple_matmul_metal_inputs(x0: Any, x1: Any, x2: Any, weights: Sequence[MLXMM8Weight]) -> Any:
    """Run one MM8 Metal launch for three distinct inputs and weights.

    Unlike :func:`mm8_group_matmul_metal_inputs`, this direct triple path does
    not stack/copy quantized weights into an additional grouped cache. It is the
    lower-memory R/K/V projection seam used by the MLX model integration.
    """

    mx = _mx()
    q0, q1, q2 = _as_mm8_triple(weights)
    n, m = int(q0.n), int(q0.m)
    if tuple(x0.shape) != tuple(x1.shape) or tuple(x0.shape) != tuple(x2.shape):
        raise ValueError(f"triple MM8 inputs must have identical shapes, got {x0.shape}, {x1.shape}, {x2.shape}")
    if int(x0.shape[-1]) != n:
        raise ValueError(f"triple MM8 inputs must end with {n}; got {tuple(x0.shape)}")
    x0_2 = x0.reshape(-1, n)
    x1_2 = x1.reshape(-1, n)
    x2_2 = x2.reshape(-1, n)
    rows = int(x0_2.shape[0])
    dims = mx.array([rows, n, m], dtype=mx.uint32)
    out = _metal_mm8_triple_inputs_kernel()(
        inputs=[
            x0_2,
            x1_2,
            x2_2,
            _mm8_u8_t(q0),
            _mm8_u8_t(q1),
            _mm8_u8_t(q2),
            q0.mx.reshape(m),
            q0.rx.reshape(m),
            q0.my.reshape(n),
            q0.ry.reshape(n),
            q1.mx.reshape(m),
            q1.rx.reshape(m),
            q1.my.reshape(n),
            q1.ry.reshape(n),
            q2.mx.reshape(m),
            q2.rx.reshape(m),
            q2.my.reshape(n),
            q2.ry.reshape(n),
            dims,
        ],
        grid=(3 * rows * m, 1, 1),
        threadgroup=(min(256, max(1, m)), 1, 1),
        output_shapes=[(3, rows, m)],
        output_dtypes=[x0.dtype],
    )[0]
    return out.reshape(3, *x0.shape[:-1], m)


def mm8_matmul_mlx(x: Any, q: MLXMM8Weight, *, backend: str = "affine") -> Any:
    """Run ``x @ dequant(q)`` with a reference, affine, Metal, or auto MLX backend."""

    mx = _mx()
    backend = (backend or "affine").lower().strip()
    if backend == "auto":
        backend = _select_auto_backend(8, int(x.reshape(-1, q.n).shape[0]))
    if backend == "reference":
        return x @ dequantize_mlx_mm8(q, out_dtype=x.dtype)
    if backend == "metal":
        return mm8_matmul_metal(x, q)
    if backend != "affine":
        raise ValueError(f"unsupported MLX mm8 backend {backend!r}")
    x2 = x.reshape(-1, q.n).astype(mx.float32)
    qf = _mm8_u8(q).astype(mx.float32) + 0.5
    # y = (x*ry) @ q_u8 * rx + (x @ my) + sum(x)*mx
    term_q = (x2 * q.ry.reshape(1, q.n)) @ qf
    term_q = term_q * q.rx.reshape(1, q.m)
    term_my = x2 @ q.my.reshape(q.n, 1)
    term_mx = mx.sum(x2, axis=-1, keepdims=True) * q.mx.reshape(1, q.m)
    y = term_q + term_my + term_mx
    return y.astype(x.dtype).reshape(*x.shape[:-1], q.m)


def _mm4_packed(q: MLXMM4Weight) -> Any:
    mx = _mx()
    if q.packed is not None:
        return q.packed
    if q.packed_t is None:
        raise ValueError("MLXMM4Weight has neither standard nor transposed packed storage")
    return mx.transpose(q.packed_t)


def _mm4_packed_t(q: MLXMM4Weight) -> Any:
    mx = _mx()
    if q.packed_t is not None:
        return q.packed_t
    if q.packed is None:
        raise ValueError("MLXMM4Weight has neither standard nor transposed packed storage")
    return mx.contiguous(mx.transpose(q.packed))


def quantize_mlx_mm4(weight: Any, *, layout: str = "standard") -> MLXMM4Weight:
    """Quantize ``weight [N, M]`` into packed affine int4 layout."""

    mx = _mx()
    layout = (layout or "standard").lower().strip()
    if layout not in {"standard", "metal"}:
        raise ValueError(f"unsupported MLX mm4 layout {layout!r}; expected standard or metal")
    dense_dtype = _weight_dtype(weight)
    w = weight
    n, m_orig = int(w.shape[0]), int(w.shape[1])
    if m_orig % 2:
        pad = mx.zeros((n, 1), dtype=w.dtype)
        w = mx.concatenate([w, pad], axis=1)
    m_padded = int(w.shape[1])
    w_norm, mx_col, rx, my, ry = _affine_minmax(w)
    u4 = mx.clip(mx.floor(w_norm * 16.0), 0, 15).astype(mx.uint8)
    lo = u4[:, 0::2]
    hi = u4[:, 1::2]
    packed = mx.bitwise_or(lo, mx.left_shift(hi, 4)).astype(mx.uint8)
    packed_t = mx.contiguous(mx.transpose(packed)) if layout == "metal" else None
    out_dtype = dense_dtype or mx.float16
    q = MLXMM4Weight(
        packed=None if layout == "metal" else packed,
        mx=mx_col.astype(out_dtype),
        rx_s=(rx / 4.0).astype(out_dtype),
        my=my.astype(out_dtype),
        ry_s=(ry / 4.0).astype(out_dtype),
        n=n,
        m_orig=m_orig,
        m_padded=m_padded,
        dense_dtype=out_dtype,
        packed_t=packed_t,
    )
    mx.eval(*[x for x in (q.packed, q.packed_t, q.mx, q.rx_s, q.my, q.ry_s) if x is not None])
    return q


def pack_mlx_mm4_group(weights: Sequence[MLXMM4Weight]) -> MLXMM4GroupWeight:
    """Prepack equal-shaped MM4 weights for grouped Metal projection."""

    mx = _mx()
    qs = list(weights)
    if not qs:
        raise ValueError("pack_mlx_mm4_group requires at least one weight")
    n, m_orig, m_padded = int(qs[0].n), int(qs[0].m_orig), int(qs[0].m_padded)
    if any(int(q.n) != n or int(q.m_orig) != m_orig or int(q.m_padded) != m_padded for q in qs):
        raise ValueError("all grouped MM4 weights must have the same [N, M] shape")
    out = MLXMM4GroupWeight(
        packed_t=mx.stack([_mm4_packed_t(q) for q in qs], axis=0),
        mx=mx.stack([q.mx.reshape(m_padded) for q in qs], axis=0),
        rx_s=mx.stack([q.rx_s.reshape(m_padded) for q in qs], axis=0),
        my=mx.stack([q.my.reshape(n) for q in qs], axis=0),
        ry_s=mx.stack([q.ry_s.reshape(n) for q in qs], axis=0),
        n=n,
        m_orig=m_orig,
        m_padded=m_padded,
        groups=len(qs),
        dense_dtype=qs[0].dense_dtype,
    )
    mx.eval(out.packed_t, out.mx, out.rx_s, out.my, out.ry_s)
    return out


def unpack_mlx_mm4(q: MLXMM4Weight, *, out_dtype: Any | None = None) -> Any:
    """Unpack int4 nibbles into a dense uint/float matrix ``[N, M_padded]``."""

    mx = _mx()
    dtype = out_dtype or q.dense_dtype
    packed = _mm4_packed(q)
    lo = mx.bitwise_and(packed, 0x0F).astype(dtype)
    hi = mx.bitwise_and(mx.right_shift(packed, 4), 0x0F).astype(dtype)
    # Stack pairs then flatten the pair dimension: [N, M/2, 2] -> [N, M].
    return mx.stack([lo, hi], axis=-1).reshape(q.n, q.m_padded)


def dequantize_mlx_mm4(q: MLXMM4Weight, *, out_dtype: Any | None = None) -> Any:
    dtype = out_dtype or q.dense_dtype
    u4 = unpack_mlx_mm4(q, out_dtype=dtype)
    deq = (u4 + 0.5) * q.ry_s * q.rx_s + q.my + q.mx
    return deq[:, : q.m_orig]


@lru_cache(maxsize=1)
def _metal_mm4_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint R = uint(dims[0]);
        uint N = uint(dims[1]);
        uint M = uint(dims[2]);
        uint total = R * M;
        if (row_id >= total) {
            return;
        }

        uint r_id = row_id / M;
        uint m_id = row_id - r_id * M;
        uint packed_col = m_id >> 1;
        bool high = (m_id & 1) != 0;
        uint x_base = r_id * N;
        uint q_base = packed_col * N;

        float rx_m = float(rx_s[m_id]);
        float mx_m = float(mx_col[m_id]);
        float acc = 0.0f;
        for (uint n = 0; n < N; ++n) {
            uint byte_v = uint(packed_t[q_base + n]);
            uint q_u4 = high ? ((byte_v >> 4) & 0x0Fu) : (byte_v & 0x0Fu);
            float xv = float(x[x_base + n]);
            float qv = float(q_u4) + 0.5f;
            float deq = qv * float(ry_s[n]) * rx_m + float(my[n]) + mx_m;
            acc += xv * deq;
        }
        out[row_id] = acc;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_mm4_affine_matmul",
        input_names=["x", "packed_t", "mx_col", "rx_s", "my", "ry_s", "dims"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def mm4_matmul_metal(x: Any, q: MLXMM4Weight) -> Any:
    """Run fused Metal ``x @ dequant(q)`` for packed MM4 affine weights."""

    mx = _mx()
    x2 = x.reshape(-1, q.n)
    rows = int(x2.shape[0])
    dims = mx.array([rows, q.n, q.m_orig], dtype=mx.uint32)
    out = _metal_mm4_kernel()(
        inputs=[
            x2,
            _mm4_packed_t(q),
            q.mx.reshape(q.m_padded),
            q.rx_s.reshape(q.m_padded),
            q.my.reshape(q.n),
            q.ry_s.reshape(q.n),
            dims,
        ],
        grid=(rows * q.m_orig, 1, 1),
        threadgroup=(min(256, max(1, q.m_orig)), 1, 1),
        output_shapes=[(rows, q.m_orig)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*x.shape[:-1], q.m_orig)


@lru_cache(maxsize=1)
def _metal_mm4_relu2_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint R = uint(dims[0]);
        uint N = uint(dims[1]);
        uint M = uint(dims[2]);
        uint total = R * M;
        if (row_id >= total) {
            return;
        }

        uint r_id = row_id / M;
        uint m_id = row_id - r_id * M;
        uint packed_col = m_id >> 1;
        bool high = (m_id & 1) != 0;
        uint x_base = r_id * N;
        uint q_base = packed_col * N;

        float rx_m = float(rx_s[m_id]);
        float mx_m = float(mx_col[m_id]);
        float acc = 0.0f;
        for (uint n = 0; n < N; ++n) {
            uint byte_v = uint(packed_t[q_base + n]);
            uint q_u4 = high ? ((byte_v >> 4) & 0x0Fu) : (byte_v & 0x0Fu);
            float xv = float(x[x_base + n]);
            float qv = float(q_u4) + 0.5f;
            float deq = qv * float(ry_s[n]) * rx_m + float(my[n]) + mx_m;
            acc += xv * deq;
        }
        float relu = acc > 0.0f ? acc : 0.0f;
        out[row_id] = relu * relu;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_mm4_affine_matmul_relu2",
        input_names=["x", "packed_t", "mx_col", "rx_s", "my", "ry_s", "dims"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def mm4_matmul_relu2_metal(x: Any, q: MLXMM4Weight) -> Any:
    """Run fused Metal ``relu(x @ dequant(q)) ** 2`` for FFN key projections."""

    mx = _mx()
    x2 = x.reshape(-1, q.n)
    rows = int(x2.shape[0])
    dims = mx.array([rows, q.n, q.m_orig], dtype=mx.uint32)
    out = _metal_mm4_relu2_kernel()(
        inputs=[
            x2,
            _mm4_packed_t(q),
            q.mx.reshape(q.m_padded),
            q.rx_s.reshape(q.m_padded),
            q.my.reshape(q.n),
            q.ry_s.reshape(q.n),
            dims,
        ],
        grid=(rows * q.m_orig, 1, 1),
        threadgroup=(min(256, max(1, q.m_orig)), 1, 1),
        output_shapes=[(rows, q.m_orig)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*x.shape[:-1], q.m_orig)


@lru_cache(maxsize=1)
def _metal_mm4_group_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint R = uint(dims[0]);
        uint N = uint(dims[1]);
        uint M = uint(dims[2]);
        uint M_PAD = uint(dims[3]);
        uint G = uint(dims[4]);
        uint PACKED_COLS = M_PAD >> 1;
        uint total = G * R * M;
        if (row_id >= total) {
            return;
        }

        uint m_id = row_id % M;
        uint tmp = row_id / M;
        uint r_id = tmp % R;
        uint g_id = tmp / R;
        uint packed_col = m_id >> 1;
        bool high = (m_id & 1) != 0;
        uint x_base = r_id * N;
        uint q_base = (g_id * PACKED_COLS + packed_col) * N;
        uint col_base = g_id * M_PAD + m_id;
        uint row_base = g_id * N;

        float rx_m = float(rx_s[col_base]);
        float mx_m = float(mx_col[col_base]);
        float acc = 0.0f;
        for (uint n = 0; n < N; ++n) {
            uint byte_v = uint(packed_t[q_base + n]);
            uint q_u4 = high ? ((byte_v >> 4) & 0x0Fu) : (byte_v & 0x0Fu);
            float xv = float(x[x_base + n]);
            float qv = float(q_u4) + 0.5f;
            float deq = qv * float(ry_s[row_base + n]) * rx_m + float(my[row_base + n]) + mx_m;
            acc += xv * deq;
        }
        out[row_id] = acc;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_mm4_affine_group_matmul",
        input_names=["x", "packed_t", "mx_col", "rx_s", "my", "ry_s", "dims"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def mm4_group_matmul_metal(x: Any, weights: Sequence[MLXMM4Weight] | MLXMM4GroupWeight) -> Any:
    """Run grouped fused Metal MM4 projections for equal-shaped weights.

    Returns ``[groups, *x.shape[:-1], out_features]``. This experimental
    launch-fusion seam targets decode-hot groups such as R/K/V projections; it
    does not change the default single-projection path.
    """

    mx = _mx()
    group = weights if isinstance(weights, MLXMM4GroupWeight) else pack_mlx_mm4_group(weights)
    n, m_orig, m_padded = int(group.n), int(group.m_orig), int(group.m_padded)
    x2 = x.reshape(-1, n)
    rows = int(x2.shape[0])
    groups = int(group.groups)
    dims = mx.array([rows, n, m_orig, m_padded, groups], dtype=mx.uint32)
    out = _metal_mm4_group_kernel()(
        inputs=[x2, group.packed_t, group.mx, group.rx_s, group.my, group.ry_s, dims],
        grid=(groups * rows * m_orig, 1, 1),
        threadgroup=(min(256, max(1, m_orig)), 1, 1),
        output_shapes=[(groups, rows, m_orig)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(groups, *x.shape[:-1], m_orig)


@lru_cache(maxsize=1)
def _metal_mm4_group_inputs_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint R = uint(dims[0]);
        uint N = uint(dims[1]);
        uint M = uint(dims[2]);
        uint M_PAD = uint(dims[3]);
        uint G = uint(dims[4]);
        uint PACKED_COLS = M_PAD >> 1;
        uint total = G * R * M;
        if (row_id >= total) {
            return;
        }

        uint m_id = row_id % M;
        uint tmp = row_id / M;
        uint r_id = tmp % R;
        uint g_id = tmp / R;
        uint packed_col = m_id >> 1;
        bool high = (m_id & 1) != 0;
        uint x_base = (g_id * R + r_id) * N;
        uint q_base = (g_id * PACKED_COLS + packed_col) * N;
        uint col_base = g_id * M_PAD + m_id;
        uint row_base = g_id * N;

        float rx_m = float(rx_s[col_base]);
        float mx_m = float(mx_col[col_base]);
        float acc = 0.0f;
        for (uint n = 0; n < N; ++n) {
            uint byte_v = uint(packed_t[q_base + n]);
            uint q_u4 = high ? ((byte_v >> 4) & 0x0Fu) : (byte_v & 0x0Fu);
            float xv = float(x_group[x_base + n]);
            float qv = float(q_u4) + 0.5f;
            float deq = qv * float(ry_s[row_base + n]) * rx_m + float(my[row_base + n]) + mx_m;
            acc += xv * deq;
        }
        out[row_id] = acc;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_mm4_affine_group_inputs_matmul",
        input_names=["x_group", "packed_t", "mx_col", "rx_s", "my", "ry_s", "dims"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def mm4_group_matmul_metal_inputs(x_group: Any, weights: Sequence[MLXMM4Weight] | MLXMM4GroupWeight) -> Any:
    """Run grouped MM4 Metal projections with one input tensor per group.

    ``x_group`` must be shaped ``[groups, *batch_shape, in_features]``. Returns
    ``[groups, *batch_shape, out_features]``.
    """

    mx = _mx()
    group = weights if isinstance(weights, MLXMM4GroupWeight) else pack_mlx_mm4_group(weights)
    groups, n, m_orig = int(group.groups), int(group.n), int(group.m_orig)
    if int(x_group.shape[0]) != groups or int(x_group.shape[-1]) != n:
        raise ValueError(
            f"x_group must be [groups, ..., {n}] with groups={groups}; got {tuple(x_group.shape)}"
        )
    x2 = x_group.reshape(groups, -1, n)
    rows = int(x2.shape[1])
    dims = mx.array([rows, n, m_orig, int(group.m_padded), groups], dtype=mx.uint32)
    out = _metal_mm4_group_inputs_kernel()(
        inputs=[x2, group.packed_t, group.mx, group.rx_s, group.my, group.ry_s, dims],
        grid=(groups * rows * m_orig, 1, 1),
        threadgroup=(min(256, max(1, m_orig)), 1, 1),
        output_shapes=[(groups, rows, m_orig)],
        output_dtypes=[x_group.dtype],
    )[0]
    return out.reshape(groups, *x_group.shape[1:-1], m_orig)


def _as_mm4_triple(weights: Sequence[MLXMM4Weight]) -> tuple[MLXMM4Weight, MLXMM4Weight, MLXMM4Weight]:
    qs = tuple(weights)
    if len(qs) != 3:
        raise ValueError(f"expected exactly three MM4 weights, got {len(qs)}")
    n, m_orig, m_padded = int(qs[0].n), int(qs[0].m_orig), int(qs[0].m_padded)
    if any(int(q.n) != n or int(q.m_orig) != m_orig or int(q.m_padded) != m_padded for q in qs):
        raise ValueError("all triple MM4 weights must have the same [N, M] shape")
    return qs  # type: ignore[return-value]


@lru_cache(maxsize=1)
def _metal_mm4_triple_inputs_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint R = uint(dims[0]);
        uint N = uint(dims[1]);
        uint M = uint(dims[2]);
        uint total = 3 * R * M;
        if (row_id >= total) {
            return;
        }

        uint m_id = row_id % M;
        uint tmp = row_id / M;
        uint r_id = tmp % R;
        uint g_id = tmp / R;
        uint packed_col = m_id >> 1;
        bool high = (m_id & 1) != 0;
        uint x_base = r_id * N;
        uint q_base = packed_col * N;

        float acc = 0.0f;
        if (g_id == 0) {
            float rx_m = float(rx0[m_id]);
            float mx_m = float(mx0[m_id]);
            for (uint n = 0; n < N; ++n) {
                uint byte_v = uint(p0_t[q_base + n]);
                uint q_u4 = high ? ((byte_v >> 4) & 0x0Fu) : (byte_v & 0x0Fu);
                float xv = float(x0[x_base + n]);
                float qv = float(q_u4) + 0.5f;
                float deq = qv * float(ry0[n]) * rx_m + float(my0[n]) + mx_m;
                acc += xv * deq;
            }
        } else if (g_id == 1) {
            float rx_m = float(rx1[m_id]);
            float mx_m = float(mx1[m_id]);
            for (uint n = 0; n < N; ++n) {
                uint byte_v = uint(p1_t[q_base + n]);
                uint q_u4 = high ? ((byte_v >> 4) & 0x0Fu) : (byte_v & 0x0Fu);
                float xv = float(x1[x_base + n]);
                float qv = float(q_u4) + 0.5f;
                float deq = qv * float(ry1[n]) * rx_m + float(my1[n]) + mx_m;
                acc += xv * deq;
            }
        } else {
            float rx_m = float(rx2[m_id]);
            float mx_m = float(mx2[m_id]);
            for (uint n = 0; n < N; ++n) {
                uint byte_v = uint(p2_t[q_base + n]);
                uint q_u4 = high ? ((byte_v >> 4) & 0x0Fu) : (byte_v & 0x0Fu);
                float xv = float(x2[x_base + n]);
                float qv = float(q_u4) + 0.5f;
                float deq = qv * float(ry2[n]) * rx_m + float(my2[n]) + mx_m;
                acc += xv * deq;
            }
        }
        out[row_id] = acc;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_mm4_affine_triple_inputs_matmul",
        input_names=[
            "x0",
            "x1",
            "x2",
            "p0_t",
            "p1_t",
            "p2_t",
            "mx0",
            "rx0",
            "my0",
            "ry0",
            "mx1",
            "rx1",
            "my1",
            "ry1",
            "mx2",
            "rx2",
            "my2",
            "ry2",
            "dims",
        ],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def mm4_triple_matmul_metal_inputs(x0: Any, x1: Any, x2: Any, weights: Sequence[MLXMM4Weight]) -> Any:
    """Run one MM4 Metal launch for three distinct inputs and weights.

    This avoids the extra grouped packed-weight copies used by the generic
    grouped path, while preserving one-launch R/K/V projection fusion.
    """

    mx = _mx()
    q0, q1, q2 = _as_mm4_triple(weights)
    n, m_orig = int(q0.n), int(q0.m_orig)
    if tuple(x0.shape) != tuple(x1.shape) or tuple(x0.shape) != tuple(x2.shape):
        raise ValueError(f"triple MM4 inputs must have identical shapes, got {x0.shape}, {x1.shape}, {x2.shape}")
    if int(x0.shape[-1]) != n:
        raise ValueError(f"triple MM4 inputs must end with {n}; got {tuple(x0.shape)}")
    x0_2 = x0.reshape(-1, n)
    x1_2 = x1.reshape(-1, n)
    x2_2 = x2.reshape(-1, n)
    rows = int(x0_2.shape[0])
    dims = mx.array([rows, n, m_orig], dtype=mx.uint32)
    out = _metal_mm4_triple_inputs_kernel()(
        inputs=[
            x0_2,
            x1_2,
            x2_2,
            _mm4_packed_t(q0),
            _mm4_packed_t(q1),
            _mm4_packed_t(q2),
            q0.mx.reshape(q0.m_padded),
            q0.rx_s.reshape(q0.m_padded),
            q0.my.reshape(n),
            q0.ry_s.reshape(n),
            q1.mx.reshape(q1.m_padded),
            q1.rx_s.reshape(q1.m_padded),
            q1.my.reshape(n),
            q1.ry_s.reshape(n),
            q2.mx.reshape(q2.m_padded),
            q2.rx_s.reshape(q2.m_padded),
            q2.my.reshape(n),
            q2.ry_s.reshape(n),
            dims,
        ],
        grid=(3 * rows * m_orig, 1, 1),
        threadgroup=(min(256, max(1, m_orig)), 1, 1),
        output_shapes=[(3, rows, m_orig)],
        output_dtypes=[x0.dtype],
    )[0]
    return out.reshape(3, *x0.shape[:-1], m_orig)


def mm4_matmul_mlx(x: Any, q: MLXMM4Weight, *, backend: str = "affine") -> Any:
    """Run ``x @ dequant(q)`` with a reference, affine, Metal, or auto MLX backend."""

    mx = _mx()
    backend = (backend or "affine").lower().strip()
    if backend == "auto":
        backend = _select_auto_backend(4, int(x.reshape(-1, q.n).shape[0]))
    if backend == "reference":
        return x @ dequantize_mlx_mm4(q, out_dtype=x.dtype)
    if backend == "metal":
        return mm4_matmul_metal(x, q)
    if backend != "affine":
        raise ValueError(f"unsupported MLX mm4 backend {backend!r}")
    x2 = x.reshape(-1, q.n).astype(mx.float32)
    u4 = unpack_mlx_mm4(q, out_dtype=mx.float32) + 0.5
    term_q = (x2 * q.ry_s.reshape(1, q.n)) @ u4
    term_q = term_q * q.rx_s.reshape(1, q.m_padded)
    term_my = x2 @ q.my.reshape(q.n, 1)
    term_mx = mx.sum(x2, axis=-1, keepdims=True) * q.mx.reshape(1, q.m_padded)
    y = term_q + term_my + term_mx
    return y[:, : q.m_orig].astype(x.dtype).reshape(*x.shape[:-1], q.m_orig)
