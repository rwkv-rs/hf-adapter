# coding=utf-8
"""FP8 E4M3 per-tensor weight quantization + online activation quantization (W8A8).

Mirrors the structure of :mod:`native_quant_mm8` but targets hardware FP8
tensor cores instead of the affine int8 (fp16i8) scheme. Weights are stored in
``torch.float8_e4m3fn`` and activations are quantized on the fly each forward,
so this is a symmetric W8A8 path: ``y = (x_fp8 @ w_fp8.T) * x_scale * w_scale``.

The fast path fuses the dequant + matmul + scale into a single
``torch._scaled_mm`` call, which dispatches to FP8 tensor cores on Hopper
(sm90) and Blackwell (sm100/120) -- and the underlying e4m3 datatype is also
available on Ada Lovelace (sm89). Everything else (CPU, older GPUs, or torch
builds without ``_scaled_mm``) falls back to a portable dequantize + dense
matmul, so the module remains importable and runnable everywhere.

Module-selection policies (shared with the other native quant backends):

* ``"memory"`` -- quantize every ``nn.Linear`` whose ``weight.numel() >=
  min_params`` (the historical size gate).
* ``"speed"``  -- quantize only ``lm_head`` after the same size gate, keeping
  the cached decode path dense through the recurrent/FFN layers.
"""
from __future__ import annotations

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .native_quant_policy import normalize_native_mm_policy, should_quantize_linear

try:
    from .kernel_policy import current_kernel_policy
except Exception:  # pragma: no cover - remote-code fallback
    current_kernel_policy = None  # type: ignore[assignment]


# Maximum finite value representable in the E4M3 floating-point format
# (exponent bias 7, 3 mantissa bits). Used as the symmetric quantization bound.
FP8_E4M3_MAX = 448.0


def fp8_available(device=None) -> bool:
    """Return whether hardware FP8 (E4M3) tensor cores can run on ``device``.

    Requires CUDA, the ``torch._scaled_mm`` operator, the
    ``torch.float8_e4m3fn`` dtype, and a compute capability of at least 8.9
    (Ada Lovelace / Hopper / Blackwell). CPU-only machines and older GPUs fall
    back to the portable dequantize + matmul path.
    """
    if torch is None or not torch.cuda.is_available():
        return False
    if not hasattr(torch, "_scaled_mm"):
        return False
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    dev = torch.device("cuda" if device is None else device)
    if dev.type != "cuda":
        return False
    try:
        major, minor = torch.cuda.get_device_capability(dev)
    except Exception:  # pragma: no cover - defensive driver/property failure
        return False
    return (major * 10 + minor) >= 89


def quantize_fp8(weight, per_channel=False):
    """Quantize ``weight: [N, K]`` to FP8 E4M3.

    Returns ``(w_fp8, scale)`` where ``w_fp8`` is ``float8_e4m3fn [N, K]``:

    * ``per_channel=False`` -> ``scale`` is a scalar ``float32`` (per-tensor),
      ``scale = max(|W|) / 448.0``.
    * ``per_channel=True``  -> ``scale`` is ``[N]`` ``float32``
      (per-output-channel), ``scale[n] = max(|W[n, :]|) / 448.0``.

    Values are clamped to ``[-448, 448]`` before the FP8 cast so that overflow
    saturates to a finite ``float8_e4m3fn`` value rather than ``NaN``.
    """
    if torch is None:
        raise RuntimeError("quantize_fp8 requires torch")
    w = weight.float()
    if per_channel:
        amax = w.abs().amax(dim=1)  # [N]
        scale = (amax / FP8_E4M3_MAX).clamp(min=1e-12)
        w_q = w / scale.reshape(-1, 1)
    else:
        amax = w.abs().max()
        scale = (amax / FP8_E4M3_MAX).clamp(min=1e-12)
        w_q = w / scale
    w_q = w_q.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return w_q, scale.to(torch.float32)


# --------------------------------------------------------------------------- #
# Model integration: an FP8 (E4M3) nn.Linear drop-in + policy-gated replacement.
# The memory policy keeps the historical size gate. The speed policy only swaps
# lm_head. When _scaled_mm is unavailable the forward dequantizes on the fly and
# runs a dense matmul, so the module is importable on CPU-only machines.
# --------------------------------------------------------------------------- #

class FP8Linear(torch.nn.Module):
    """Drop-in for ``nn.Linear`` storing FP8 E4M3 weights + W8A8 forward."""

    def __init__(self, linear, *, per_channel=False):
        super().__init__()
        self.in_features, self.out_features = (
            linear.weight.shape[1],
            linear.weight.shape[0],
        )
        self.per_channel = bool(per_channel)
        w_fp8, scale = quantize_fp8(linear.weight.data, per_channel=self.per_channel)
        self.register_buffer("w_fp8", w_fp8)   # float8_e4m3fn [out, in]
        self.register_buffer("scale", scale)   # scalar float32 (or [out] if per_channel)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        if torch is None:
            raise RuntimeError("FP8Linear requires torch")
        orig_dtype = x.dtype
        leading = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        # torch._scaled_mm targets a bf16 accumulator; align fp16 inputs to bf16.
        if x2.dtype == torch.float16:
            x2 = x2.to(torch.bfloat16)

        can_scaled_mm = (
            hasattr(torch, "_scaled_mm")
            and x2.is_cuda
            and fp8_available(x2.device)
        )
        if can_scaled_mm:
            # Online (per-tensor) activation quantization.
            amax_x = x2.abs().max()
            x_scale = (amax_x / FP8_E4M3_MAX).clamp(min=1e-12)
            x_fp8 = (x2 / x_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(
                torch.float8_e4m3fn
            )
            M = x2.shape[0]
            if self.per_channel:
                scale_a = x_scale.reshape(1).expand(M, 1).contiguous()
                scale_b = self.scale.reshape(1, -1).contiguous()
            else:
                scale_a = x_scale.reshape(1)
                scale_b = self.scale.reshape(1)
            out = torch._scaled_mm(
                x_fp8,
                self.w_fp8.t(),
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
        else:
            # Portable fallback: dequantize the weight and run a dense matmul.
            # Used on CPU, older GPUs, or torch builds without _scaled_mm.
            if self.per_channel:
                w_fp16 = self.w_fp8.to(torch.float16) * self.scale.reshape(-1, 1)
            else:
                w_fp16 = self.w_fp8.to(torch.float16) * self.scale
            out = x2.to(torch.float16) @ w_fp16.t()
            out = out.to(torch.bfloat16)

        out = out.reshape(*leading, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out

    def rwkv7_forward_into(self, x, out):
        result = self.forward(x)
        out.copy_(result)
        return out

    def extra_repr(self):
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"fp8(per_channel={self.per_channel})"
        )


def _clear_quant_caches(model) -> None:
    """Drop the JIT/graph caches invalidated by swapping in quantized linears.

    Mirrors the cache list cleared by :func:`native_quant_mm8.quantize_model_mm8`
    so that any pre-built CUDA graph / JIT pack that captured the dense linears
    is rebuilt against the new FP8 modules.
    """
    for cache_attr in (
        "_rwkv7_native_jit_pack_cache",
        "_rwkv7_native_graph_pack_cache",
        "_rwkv7_native_graph_runner_cache",
        "_rwkv7_native_prefill_graph_runner_cache",
        "_rwkv7_native_prefill_graph_hot_runner",
        "_rwkv7_native_model_jit_pack_cache",
    ):
        if hasattr(model, cache_attr):
            delattr(model, cache_attr)


def quantize_model_fp8(
    model,
    *,
    min_params: int = 8_000_000,
    policy: str = "memory",
    per_channel: bool = False,
) -> int:
    """Swap eligible ``nn.Linear`` modules for :class:`FP8Linear`.

    ``policy="memory"`` quantizes every Linear with ``weight.numel() >=
    min_params`` (the historical size gate). ``policy="speed"`` quantizes only
    ``lm_head`` after the same size gate, keeping per-layer FFN/recurrent
    decode dense. Set ``per_channel=True`` for per-output-channel weight scales.
    Returns the number of modules replaced.
    """
    if torch is None:
        raise RuntimeError("quantize_model_fp8 requires torch")

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
        setattr(
            parent,
            attr,
            FP8Linear(getattr(parent, attr), per_channel=per_channel),
        )

    setattr(model, "_rwkv7_native_mm_quantization", "fp8")
    setattr(model, "_rwkv7_native_mm_replaced_modules", len(targets))
    setattr(model, "_rwkv7_native_mm_kernel", "fp8_e4m3_scaled_mm")
    setattr(
        model,
        "_rwkv7_native_mm_block_replaced_modules",
        sum(name.startswith("model.layers.") for name in targets),
    )
    _clear_quant_caches(model)
    return len(targets)
