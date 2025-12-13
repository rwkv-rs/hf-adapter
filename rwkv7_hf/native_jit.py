# coding=utf-8
"""TorchScript-native RWKV-7 decode. The ENTIRE per-layer block (LayerNorms +
TMix_one + CMix_one) is fused into one torch.jit.script function, so per token
there is only ~1 C++ call per layer + embedding/head. Math ports the official
RWKV_x070 TMix_one/CMix_one (bit-exact vs FLA, see native.py).

Run: python -m rwkv7_hf.native_jit <hf_dir>
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional Triton prefill acceleration
    from .fused_elementwise import fused_relu_square, fused_relu_square_available
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_elementwise import fused_relu_square, fused_relu_square_available
    except Exception:
        fused_relu_square = None  # type: ignore[assignment]
        fused_relu_square_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional sequence FFN tensor-core path
    from .fused_ffn import fused_sequence_ffn, fused_sequence_ffn_available
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_ffn import fused_sequence_ffn, fused_sequence_ffn_available
    except Exception:
        fused_sequence_ffn = None  # type: ignore[assignment]
        fused_sequence_ffn_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional BnB W8 FFN activation fusion
    from .native_quant_bnb8 import (
        fused_bnb8_attn_sequence_mix_quant,
        fused_bnb8_ffn_sequence_mix_quant,
        fused_bnb8_relu_square_quant,
        fused_bnb8_relu_square_quant_available,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from native_quant_bnb8 import (
            fused_bnb8_attn_sequence_mix_quant,
            fused_bnb8_ffn_sequence_mix_quant,
            fused_bnb8_relu_square_quant,
            fused_bnb8_relu_square_quant_available,
        )
    except Exception:
        fused_bnb8_attn_sequence_mix_quant = None  # type: ignore[assignment]
        fused_bnb8_ffn_sequence_mix_quant = None  # type: ignore[assignment]
        fused_bnb8_relu_square_quant = None  # type: ignore[assignment]
        fused_bnb8_relu_square_quant_available = None  # type: ignore[assignment]


def _linear_module(module, x: torch.Tensor) -> torch.Tensor:
    """Linear call that also supports native MM8/MM4Linear lm_head modules."""
    if type(module) is torch.nn.Linear:
        return F.linear(x, module.weight, module.bias)
    return module(x)


def _graph_linear_operand(module):
    """Return a dense weight when possible, otherwise retain the quant module.

    Native CUDA-graph decode historically packed bare ``nn.Linear.weight``
    tensors. Packed MM8/MM4 and BnB modules must remain live callables, letting
    graph capture record their quantized GEMV without materialising a second
    fp16 copy of the model.
    """

    if type(module) is torch.nn.Linear and type(module.weight) is torch.nn.Parameter:
        return module.weight
    return module


def _graph_linear_is_dense(operand) -> bool:
    return isinstance(operand, torch.Tensor)


def _graph_linear_shape(operand) -> tuple[int, int]:
    if _graph_linear_is_dense(operand):
        return int(operand.shape[0]), int(operand.shape[1])
    return int(operand.out_features), int(operand.in_features)


def _native_graph_sparse_ffn_low_memory_pack_enabled() -> bool:
    return env_flag("RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_LOW_MEMORY_PACK", False)


def _native_graph_relayout_ffn_value_weight(module):
    """Store an FFN down weight in the sparse kernel's transposed layout.

    The exposed parameter keeps its original ``[hidden, ffn]`` shape, so
    ``F.linear`` and state-dict names remain compatible. Its backing storage is
    contiguous as ``[ffn, hidden]``, allowing the sparse decode kernel to reuse
    the same bytes instead of allocating a second full-size model copy.
    """

    if type(module) is not torch.nn.Linear or module.bias is not None:
        raise TypeError("low-memory sparse FFN packing requires a bias-free nn.Linear")
    if getattr(module, "_rwkv7_sparse_low_memory_layout", False):
        return module.weight
    weight = module.weight
    if weight.dim() != 2:
        raise ValueError("low-memory sparse FFN packing requires a rank-2 weight")
    packed = weight.detach().transpose(0, 1).contiguous()
    module.weight = torch.nn.Parameter(
        packed.transpose(0, 1), requires_grad=bool(weight.requires_grad)
    )
    module._rwkv7_sparse_low_memory_layout = True
    return module.weight


def _native_bnb8_policy_flag(env_name: str, policy_name: str) -> bool:
    try:
        default = bool(getattr(_kernel_policy(), policy_name, False))
    except Exception:
        default = False
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _native_bnb8_policy_block(env_name: str, policy_name: str, fallback: int) -> int:
    try:
        default = int(getattr(_kernel_policy(), policy_name, fallback))
    except Exception:
        default = int(fallback)
    raw = os.environ.get(env_name)
    value = default if raw is None else int(raw)
    if value not in {256, 512, 1024, 2048, 4096}:
        raise ValueError(f"{env_name} must be 256, 512, 1024, 2048, or 4096")
    return value


def _native_bnb8_direct_enabled() -> bool:
    """Use the inference-only BnB W8 operator path without autograd wrappers.

    ``Linear8bitLt.forward`` enters a custom ``autograd.Function`` for every
    projection even under ``torch.inference_mode``.  Native prefill invokes six
    large projections per layer, so that Python/autograd dispatch is visible at
    serving-sized row counts.  The direct path below executes the exact two BnB
    operators used by the threshold-zero fast path while preserving the module
    fallback for training, gradients, outliers, and non-BnB quantizers.
    """

    return _native_bnb8_policy_flag("RWKV7_NATIVE_BNB8_DIRECT", "native_bnb8_direct")


def _is_bnb8_linear(operand) -> bool:
    cls = type(operand)
    return bool(
        cls.__name__ == "Linear8bitLt"
        and cls.__module__.startswith("bitsandbytes.")
        and hasattr(operand, "state")
        and hasattr(operand, "weight")
    )


def _bnb8_direct_linear(x: torch.Tensor, operand) -> torch.Tensor | None:
    """Run BnB's threshold-zero inference operators directly, if eligible."""

    if (
        not _native_bnb8_direct_enabled()
        or not _is_bnb8_linear(operand)
        or torch.is_grad_enabled()
        or bool(getattr(operand, "training", False))
    ):
        return None
    state = operand.state
    if float(getattr(state, "threshold", 0.0) or 0.0) != 0.0:
        return None
    if getattr(state, "CB", None) is None:
        if getattr(operand.weight, "CB", None) is None:
            return None
        operand.init_8bit_state()
    cb = getattr(state, "CB", None)
    scb = getattr(state, "SCB", None)
    if cb is None or scb is None:
        return None

    input_shape = tuple(x.shape)
    rows = x.reshape(-1, input_shape[-1]) if x.dim() != 2 else x
    quantized, scales, _ = torch.ops.bitsandbytes.int8_vectorwise_quant.default(
        rows.to(torch.float16),
        0.0,
    )
    bias = getattr(operand, "bias", None)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    out = torch.ops.bitsandbytes.int8_scaled_mm.default(
        quantized,
        cb,
        scales,
        scb,
        bias=bias,
        dtype=x.dtype,
    )
    if x.dim() != 2:
        out = out.reshape(*input_shape[:-1], int(operand.out_features))
    return out


def _bnb8_direct_relu_square_linear(x: torch.Tensor, operand) -> torch.Tensor | None:
    """Fuse RWKV FFN ReLU² preparation into BnB W8 activation quantization."""

    enabled = _native_bnb8_policy_flag(
        "RWKV7_NATIVE_BNB8_RELU_QUANT",
        "native_bnb8_relu_quant",
    )
    if (
        not enabled
        or not _native_bnb8_direct_enabled()
        or not _is_bnb8_linear(operand)
        or torch.is_grad_enabled()
        or bool(getattr(operand, "training", False))
        or fused_bnb8_relu_square_quant is None
        or fused_bnb8_relu_square_quant_available is None
        or not fused_bnb8_relu_square_quant_available()
    ):
        return None
    state = operand.state
    if float(getattr(state, "threshold", 0.0) or 0.0) != 0.0:
        return None
    if getattr(state, "CB", None) is None:
        if getattr(operand.weight, "CB", None) is None:
            return None
        operand.init_8bit_state()
    cb = getattr(state, "CB", None)
    scb = getattr(state, "SCB", None)
    if cb is None or scb is None:
        return None
    quantized, scales = fused_bnb8_relu_square_quant(x)
    bias = getattr(operand, "bias", None)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    out = torch.ops.bitsandbytes.int8_scaled_mm.default(
        quantized,
        cb,
        scales,
        scb,
        bias=bias,
        dtype=x.dtype,
    )
    input_shape = tuple(x.shape)
    if x.dim() != 2:
        out = out.reshape(*input_shape[:-1], int(operand.out_features))
    return out


def _bnb8_prequant_linear(quantized, scales, operand, *, dtype, output_shape):
    """Apply a BnB W8 matrix to already row-quantized activations."""

    if not _is_bnb8_linear(operand):
        raise TypeError("prequantized BnB dispatch requires Linear8bitLt")
    state = operand.state
    if float(getattr(state, "threshold", 0.0) or 0.0) != 0.0:
        raise ValueError("prequantized BnB dispatch requires threshold=0")
    if getattr(state, "CB", None) is None:
        operand.init_8bit_state()
    bias = getattr(operand, "bias", None)
    if bias is not None and bias.dtype != dtype:
        bias = bias.to(dtype)
    out = torch.ops.bitsandbytes.int8_scaled_mm.default(
        quantized,
        state.CB,
        scales,
        state.SCB,
        bias=bias,
        dtype=dtype,
    )
    return out.reshape(*output_shape, int(operand.out_features))


def _bnb8_rkv_mix_quant_enabled(*operands) -> bool:
    if (
        not _native_bnb8_policy_flag(
            "RWKV7_NATIVE_BNB8_RKV_MIX_QUANT",
            "native_bnb8_rkv_mix_quant",
        )
        or not _native_bnb8_direct_enabled()
        or fused_bnb8_attn_sequence_mix_quant is None
        or fused_bnb8_relu_square_quant_available is None
        or not fused_bnb8_relu_square_quant_available()
        or torch.is_grad_enabled()
    ):
        return False
    for operand in operands:
        if not _is_bnb8_linear(operand) or bool(getattr(operand, "training", False)):
            return False
        if float(getattr(operand.state, "threshold", 0.0) or 0.0) != 0.0:
            return False
    return True


def _bnb8_ffn_mix_quant_enabled(operand) -> bool:
    return bool(
        _native_bnb8_policy_flag(
            "RWKV7_NATIVE_BNB8_FFN_MIX_QUANT",
            "native_bnb8_ffn_mix_quant",
        )
        and _native_bnb8_direct_enabled()
        and fused_bnb8_ffn_sequence_mix_quant is not None
        and fused_bnb8_relu_square_quant_available is not None
        and fused_bnb8_relu_square_quant_available()
        and not torch.is_grad_enabled()
        and _is_bnb8_linear(operand)
        and not bool(getattr(operand, "training", False))
        and float(getattr(operand.state, "threshold", 0.0) or 0.0) == 0.0
    )


def _graph_linear_call(x: torch.Tensor, operand) -> torch.Tensor:
    if _graph_linear_is_dense(operand):
        return F.linear(x, operand)
    direct = _bnb8_direct_linear(x, operand)
    if direct is not None:
        return direct
    # bitsandbytes W8 accepts only rank-2/3 activations, whereas the scalar
    # native-graph runner deliberately keeps hidden state rank-1. Preserve the
    # runner ABI while presenting a supported matrix shape to quant modules.
    if x.dim() == 1:
        return operand(x.unsqueeze(0)).squeeze(0)
    return operand(x)


def _native_prefill_linear(
    x: torch.Tensor,
    operand,
    bias=None,
    *,
    allow_fp16_accumulation: bool = False,
) -> torch.Tensor:
    """Sequence linear supporting dense and HF/native quantized operands."""

    if _graph_linear_is_dense(operand):
        matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
        can_select_accumulation = bool(
            allow_fp16_accumulation
            and x.is_cuda
            and x.dtype == torch.float16
            and matmul is not None
            and hasattr(matmul, "allow_fp16_accumulation")
        )
        if not can_select_accumulation:
            return F.linear(x, operand, bias)
        previous = bool(matmul.allow_fp16_accumulation)
        if not previous:
            matmul.allow_fp16_accumulation = True
        try:
            return F.linear(x, operand, bias)
        finally:
            if not previous:
                matmul.allow_fp16_accumulation = False
    direct = _bnb8_direct_linear(x, operand)
    if direct is not None:
        return direct
    # Quantized modules retain and apply their own bias. Explicit packed biases
    # belong only to dense low-rank operands.
    return operand(x)


def _graph_linears_are_dense(*operands) -> bool:
    return all(_graph_linear_is_dense(item) for item in operands)


def _graph_linear_call_with_explicit_bias(x: torch.Tensor, operand, bias) -> torch.Tensor:
    """Apply a packed linear whose module form already owns ``bias``."""

    y = _graph_linear_call(x, operand)
    if _graph_linear_is_dense(operand) and bias is not None:
        y = y + bias
    return y


def _lm_head(model, x: torch.Tensor) -> torch.Tensor:
    return _linear_module(model.lm_head, x)

try:  # pragma: no cover - optional in older converted model dirs
    from .kernel_policy import current_kernel_policy, env_blocks, env_flag, env_int
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from kernel_policy import current_kernel_policy, env_blocks, env_flag, env_int
    except Exception:
        current_kernel_policy = None  # type: ignore[assignment]

        def env_flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return bool(default)
            return raw.strip().lower() not in {"0", "false", "no", "off"}

        def env_int(name: str, default: int, *, lower: int = 1, upper: int | None = None) -> int:
            try:
                value = int(os.environ.get(name, str(default)).strip())
            except Exception:
                value = default
            value = max(lower, value)
            return min(value, upper) if upper is not None else value

        def env_blocks(names: tuple[str, str, str], defaults: tuple[int, int, int], uppers: tuple[int, int, int]) -> tuple[int, int, int]:
            return (
                env_int(names[0], defaults[0], lower=1, upper=uppers[0]),
                env_int(names[1], defaults[1], lower=1, upper=uppers[1]),
                env_int(names[2], defaults[2], lower=1, upper=uppers[2]),
            )

try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_recurrent_update import (
        fused_recurrent_output_prepare,
        fused_recurrent_output_prepare_raw,
        fused_recurrent_output_prepare_available,
        fused_recurrent_scan,
        fused_recurrent_scan_available,
        fused_recurrent_scan_clampw,
        fused_recurrent_scan_clampw_available,
        fused_recurrent_scan_state_prep,
        fused_recurrent_scan_state_prep_available,
        fused_recurrent_scan_output_prepare,
        fused_recurrent_scan_output_prepare_available,
        fused_recurrent_update,
        fused_recurrent_update_available,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_recurrent_update import (
            fused_recurrent_output_prepare,
            fused_recurrent_output_prepare_raw,
            fused_recurrent_output_prepare_available,
            fused_recurrent_scan,
            fused_recurrent_scan_available,
            fused_recurrent_scan_clampw,
            fused_recurrent_scan_clampw_available,
            fused_recurrent_scan_state_prep,
            fused_recurrent_scan_state_prep_available,
            fused_recurrent_scan_output_prepare,
            fused_recurrent_scan_output_prepare_available,
            fused_recurrent_update,
            fused_recurrent_update_available,
        )
    except Exception:
        fused_recurrent_output_prepare = None  # type: ignore[assignment]
        fused_recurrent_output_prepare_raw = None  # type: ignore[assignment]
        fused_recurrent_output_prepare_available = None  # type: ignore[assignment]
        fused_recurrent_scan = None  # type: ignore[assignment]
        fused_recurrent_scan_available = None  # type: ignore[assignment]
        fused_recurrent_scan_clampw = None  # type: ignore[assignment]
        fused_recurrent_scan_clampw_available = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep_available = None  # type: ignore[assignment]
        fused_recurrent_scan_output_prepare = None  # type: ignore[assignment]
        fused_recurrent_scan_output_prepare_available = None  # type: ignore[assignment]
        fused_recurrent_update = None  # type: ignore[assignment]
        fused_recurrent_update_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional pure-torch DPLR/chunked prefill prototype
    from .dplr_prefill import dplr_chunk_scan
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from dplr_prefill import dplr_chunk_scan
    except Exception:
        dplr_chunk_scan = None  # type: ignore[assignment]

try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_output import (
        fused_attn_output_prepare,
        fused_attn_output_prepare_available,
        fused_attn_output_project,
        fused_attn_output_project_available,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_output import (
            fused_attn_output_prepare,
            fused_attn_output_prepare_available,
            fused_attn_output_project,
            fused_attn_output_project_available,
        )
    except Exception:
        fused_attn_output_prepare = None  # type: ignore[assignment]
        fused_attn_output_prepare_available = None  # type: ignore[assignment]
        fused_attn_output_project = None  # type: ignore[assignment]
        fused_attn_output_project_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_attention_projection import (
        fused_rkv_wag_projection,
        fused_rkv_wag_projection_available,
        fused_rkv_wavg_projection,
        fused_rkv_wavg_projection_available,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_attention_projection import (
            fused_rkv_wag_projection,
            fused_rkv_wag_projection_available,
            fused_rkv_wavg_projection,
            fused_rkv_wavg_projection_available,
        )
    except Exception:
        fused_rkv_wag_projection = None  # type: ignore[assignment]
        fused_rkv_wag_projection_available = None  # type: ignore[assignment]
        fused_rkv_wavg_projection = None  # type: ignore[assignment]
        fused_rkv_wavg_projection_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional sm_70 grouped low-rank path
    from .sm70_wagv import sm70_orig_linear, sm70_orig_rkv, sm70_wagv_lora, sm70_wagv_lora_available
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from sm70_wagv import sm70_orig_linear, sm70_orig_rkv, sm70_wagv_lora, sm70_wagv_lora_available
    except Exception:
        sm70_orig_linear = None  # type: ignore[assignment]
        sm70_orig_rkv = None  # type: ignore[assignment]
        sm70_wagv_lora = None  # type: ignore[assignment]
        sm70_wagv_lora_available = None  # type: ignore[assignment]


try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_lora import (
        fused_wag_lora,
        fused_wag_lora_available,
        fused_wavg_lora,
        fused_wavg_lora_available,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_lora import (
            fused_wag_lora,
            fused_wag_lora_available,
            fused_wavg_lora,
            fused_wavg_lora_available,
        )
    except Exception:
        fused_wag_lora = None  # type: ignore[assignment]
        fused_wag_lora_available = None  # type: ignore[assignment]
        fused_wavg_lora = None  # type: ignore[assignment]
        fused_wavg_lora_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_prefill import (
        fused_prefill_kv_kk_prep,
        fused_prefill_kv_kk_prep_available,
        fused_prefill_state_prep,
        fused_prefill_state_prep_available,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_prefill import (
            fused_prefill_kv_kk_prep,
            fused_prefill_kv_kk_prep_available,
            fused_prefill_state_prep,
            fused_prefill_state_prep_available,
        )
    except Exception:
        fused_prefill_kv_kk_prep = None  # type: ignore[assignment]
        fused_prefill_kv_kk_prep_available = None  # type: ignore[assignment]
        fused_prefill_state_prep = None  # type: ignore[assignment]
        fused_prefill_state_prep_available = None  # type: ignore[assignment]

try:  # pragma: no cover - vendored FLA-independent chunk forward
    from .self_chunk_rwkv7 import self_chunk_rwkv7, self_chunk_rwkv7_available
except Exception:  # pragma: no cover
    try:
        from self_chunk_rwkv7 import self_chunk_rwkv7, self_chunk_rwkv7_available
    except Exception:
        self_chunk_rwkv7 = None  # type: ignore[assignment]
        self_chunk_rwkv7_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_time_mix import (
        fused_attn_sequence_shift_mix,
        fused_attn_shift_mix,
        fused_attn_shift_mix_available,
        fused_ffn_sequence_shift_mix,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_time_mix import (
            fused_attn_sequence_shift_mix,
            fused_attn_shift_mix,
            fused_attn_shift_mix_available,
            fused_ffn_sequence_shift_mix,
        )
    except Exception:
        fused_attn_sequence_shift_mix = None  # type: ignore[assignment]
        fused_attn_shift_mix = None  # type: ignore[assignment]
        fused_attn_shift_mix_available = None  # type: ignore[assignment]
        fused_ffn_sequence_shift_mix = None  # type: ignore[assignment]

try:  # pragma: no cover - optional decode-only norm/mix fast path
    from .fused_decode_norm_mix import (
        fused_attn_norm_mix6_decode,
        fused_decode_norm_mix_available,
        fused_ffn_add_norm_mix_decode,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_decode_norm_mix import (
            fused_attn_norm_mix6_decode,
            fused_decode_norm_mix_available,
            fused_ffn_add_norm_mix_decode,
        )
    except Exception:
        fused_attn_norm_mix6_decode = None  # type: ignore[assignment]
        fused_decode_norm_mix_available = None  # type: ignore[assignment]
        fused_ffn_add_norm_mix_decode = None  # type: ignore[assignment]

try:  # pragma: no cover - optional sm_70 small-row fp16 linear
    from .sm70_linear import (
        sm70_linear,
        sm70_linear_should_use,
        sm70_linear_threads,
        sm70_ffn_down_add,
        sm70_ffn_down_add_should_use,
        sm70_ffn_up_relu2,
        sm70_ffn_up_relu2_should_use,
        sm70_rkv,
        sm70_rkv_should_use,
        sm70_rkv_threads,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from sm70_linear import (
            sm70_linear,
            sm70_linear_should_use,
            sm70_linear_threads,
            sm70_ffn_down_add,
            sm70_ffn_down_add_should_use,
            sm70_ffn_up_relu2,
            sm70_ffn_up_relu2_should_use,
            sm70_rkv,
            sm70_rkv_should_use,
            sm70_rkv_threads,
        )
    except Exception:
        sm70_linear = None  # type: ignore[assignment]
        sm70_linear_should_use = None  # type: ignore[assignment]
        sm70_linear_threads = None  # type: ignore[assignment]
        sm70_ffn_down_add = None  # type: ignore[assignment]
        sm70_ffn_down_add_should_use = None  # type: ignore[assignment]
        sm70_ffn_up_relu2 = None  # type: ignore[assignment]
        sm70_ffn_up_relu2_should_use = None  # type: ignore[assignment]
        sm70_rkv = None  # type: ignore[assignment]
        sm70_rkv_should_use = None  # type: ignore[assignment]
        sm70_rkv_threads = None  # type: ignore[assignment]

try:  # pragma: no cover - optional sm_89 sparse FFN contraction
    from .ada_sparse_ffn import (
        ada_ffn_up,
        ada_linear,
        ada_linear_should_use,
        ada_sparse_ffn_down_add,
        ada_sparse_ffn_pack_weight,
        ada_sparse_ffn_prepare_deterministic_scratch,
        ada_sparse_ffn_prepare_fp32_scratch,
        ada_sparse_ffn_should_use,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from ada_sparse_ffn import (
            ada_ffn_up,
            ada_linear,
            ada_linear_should_use,
            ada_sparse_ffn_down_add,
            ada_sparse_ffn_pack_weight,
            ada_sparse_ffn_prepare_deterministic_scratch,
            ada_sparse_ffn_prepare_fp32_scratch,
            ada_sparse_ffn_should_use,
        )
    except Exception:
        ada_ffn_up = None  # type: ignore[assignment]
        ada_linear = None  # type: ignore[assignment]
        ada_linear_should_use = None  # type: ignore[assignment]
        ada_sparse_ffn_down_add = None  # type: ignore[assignment]
        ada_sparse_ffn_pack_weight = None  # type: ignore[assignment]
        ada_sparse_ffn_prepare_deterministic_scratch = None  # type: ignore[assignment]
        ada_sparse_ffn_prepare_fp32_scratch = None  # type: ignore[assignment]
        ada_sparse_ffn_should_use = None  # type: ignore[assignment]

try:  # pragma: no cover - optional sm_89/sm_120 grouped W/A/G/V LoRA
    from .ada_lora import (
        ada_wag_lora,
        ada_wagv_lora,
        ada_wagv_lora_available,
        ada_wagv_lora_should_use,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from ada_lora import (
            ada_wag_lora,
            ada_wagv_lora,
            ada_wagv_lora_available,
            ada_wagv_lora_should_use,
        )
    except Exception:
        ada_wag_lora = None  # type: ignore[assignment]
        ada_wagv_lora = None  # type: ignore[assignment]
        ada_wagv_lora_available = None  # type: ignore[assignment]
        ada_wagv_lora_should_use = None  # type: ignore[assignment]

try:  # pragma: no cover - optional exact-shape FP16 recurrent state
    from .native_wkv_fp16 import (
        native_fp16_recurrent_output_prepare_raw,
        native_fp16_sequence,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from native_wkv_fp16 import (
            native_fp16_recurrent_output_prepare_raw,
            native_fp16_sequence,
        )
    except Exception:
        native_fp16_recurrent_output_prepare_raw = None  # type: ignore[assignment]
        native_fp16_sequence = None  # type: ignore[assignment]

try:  # pragma: no cover - optional exact official-order Blackwell norm/mix
    from .blackwell_norm_mix import (
        blackwell_ffn_add_norm_mix,
        blackwell_norm_mix_should_use,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from blackwell_norm_mix import (
            blackwell_ffn_add_norm_mix,
            blackwell_norm_mix_should_use,
        )
    except Exception:
        blackwell_ffn_add_norm_mix = None  # type: ignore[assignment]
        blackwell_norm_mix_should_use = None  # type: ignore[assignment]


_FALSE_VALUES = {"0", "false", "False", "no", "off"}


def _kernel_policy():
    if current_kernel_policy is None:
        return None
    try:
        return current_kernel_policy(torch_module=torch)
    except Exception:
        return None


def _native_graph_fused_recurrent_enabled() -> bool:
    """Runtime switch for the experimental native-graph recurrent Triton path."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT", bool(getattr(policy, "fused_recurrent", False))):
        return False
    if fused_recurrent_update is None or fused_recurrent_update_available is None:
        return False
    try:
        return bool(fused_recurrent_update_available())
    except Exception:
        return False


def _native_prefill_fused_scan_enabled() -> bool:
    """Runtime switch for the experimental native prefill recurrent scan."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_SCAN", bool(getattr(policy, "fused_prefill_scan", False))):
        return False
    if fused_recurrent_scan is None or fused_recurrent_scan_available is None:
        return False
    try:
        return bool(fused_recurrent_scan_available())
    except Exception:
        return False


def _native_prefill_fp16_recurrent_requested() -> bool:
    """Select official-precision sequence recurrence without changing defaults."""

    return env_flag("RWKV7_NATIVE_PREFILL_FP16_RECURRENT", False)


def _native_prefill_fp16_recurrent_enabled(state: torch.Tensor) -> bool:
    return bool(
        _native_prefill_fp16_recurrent_requested()
        and native_fp16_sequence is not None
        and state.dtype == torch.float16
        and int(state.shape[-1]) == 64
    )


def _native_prefill_self_chunk_enabled(
    tokens: int,
    head_dim: int,
    batch_size: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Select the vendored sequence-parallel DPLR forward for long prompts."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK",
        bool(getattr(policy, "fused_prefill_self_chunk", False)),
    ):
        return False
    min_tokens = env_int(
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MIN_TOKENS",
        int(getattr(policy, "prefill_self_chunk_min_tokens", 1024)),
        lower=16,
    )
    raw_model_shapes = os.environ.get("RWKV7_NATIVE_PREFILL_SELF_CHUNK_MODEL_SHAPES")
    if raw_model_shapes is None:
        model_shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(policy, "prefill_self_chunk_model_shapes", ())
            if len(shape) == 4
        }
    else:
        model_shapes = set()
        try:
            for item in raw_model_shapes.replace(",", " ").split():
                values = tuple(int(value) for value in item.lower().split("x"))
                if len(values) != 4 or any(value <= 0 for value in values):
                    raise ValueError
                model_shapes.add(values)
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MODEL_SHAPES must contain HxLxBxT tuples"
            ) from exc
    exact_model_shape = (
        (int(hidden_size), int(num_layers), int(batch_size), int(tokens))
        if None not in (hidden_size, num_layers, batch_size)
        else None
    )
    if (
        int(tokens) < min_tokens
        and exact_model_shape not in model_shapes
    ) or int(tokens) % 16 or int(head_dim) != 64:
        return False
    if self_chunk_rwkv7 is None or self_chunk_rwkv7_available is None:
        return False
    try:
        return bool(self_chunk_rwkv7_available())
    except Exception:
        return False


def _native_prefill_self_chunk_size(batch_size: int, tokens: int | None = None) -> int:
    """Return the exact-card sequence chunk size."""

    policy = _kernel_policy()
    default = int(getattr(policy, "prefill_self_chunk_size", 16))
    if tokens is not None:
        for policy_batch, policy_tokens, policy_size in getattr(
            policy,
            "prefill_self_chunk_shape_sizes",
            (),
        ):
            if (int(batch_size), int(tokens)) == (int(policy_batch), int(policy_tokens)):
                default = int(policy_size)
                break
    chunk_size = env_int(
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE",
        default,
        lower=16,
        upper=64,
    )
    if chunk_size not in {16, 32, 64}:
        raise ValueError("RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE must be 16, 32, or 64")
    return chunk_size


def _native_prefill_self_chunk_h_tiles(
    batch_size: int,
    tokens: int,
) -> tuple[int, int] | None:
    """Return an exact-shape tile override from the centralized card policy."""

    policy = _kernel_policy()
    for policy_batch, policy_tokens, policy_bv, policy_bc in getattr(
        policy,
        "prefill_self_chunk_h_tile_shapes",
        (),
    ):
        if (int(batch_size), int(tokens)) == (int(policy_batch), int(policy_tokens)):
            return int(policy_bv), int(policy_bc)
    return None


def _native_prefill_self_chunk_safe_gate() -> bool:
    """Select the numerically conservative tensor-core intra-chunk kernel."""

    return env_flag("RWKV7_NATIVE_PREFILL_SELF_CHUNK_SAFE_GATE", True)


def _native_prefill_dplr_scan_enabled() -> bool:
    """Runtime switch for the correctness-first DPLR/chunked prefill scan."""

    if not env_flag("RWKV7_NATIVE_PREFILL_DPLR_SCAN", False):
        return False
    return dplr_chunk_scan is not None


def _native_prefill_fused_residual_gemm_enabled() -> bool:
    """Use GEMM beta=1 epilogues for the two residual projections."""

    policy = _kernel_policy()
    return env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_RESIDUAL_GEMM",
        bool(getattr(policy, "fused_prefill_residual_gemm", False)),
    )


def _native_prefill_linear_add_residual(x, weight, residual):
    """Compute ``residual + linear(x, weight)`` with one GEMM output write."""

    hidden = int(weight.shape[0])
    out = residual.reshape(-1, hidden)
    out.addmm_(
        x.reshape(-1, int(weight.shape[1])),
        weight.t(),
    )
    return out.view_as(residual)


def _native_prefill_project_residual(x, operand, residual):
    """Use GEMM beta=1 for dense weights and a safe add for quant modules."""

    if _graph_linear_is_dense(operand) and _native_prefill_fused_residual_gemm_enabled():
        return _native_prefill_linear_add_residual(x, operand, residual)
    return residual + _native_prefill_linear(x, operand)


def _native_prefill_dplr_chunk_size() -> int:
    """Chunk length for the pure-torch DPLR/chunked prefill reference path."""

    return env_int("RWKV7_NATIVE_PREFILL_DPLR_CHUNK_SIZE", 64, lower=1, upper=4096)


def _native_prefill_fused_clampw_scan_enabled(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Runtime switch for raw-W clampw native prefill recurrent scan."""

    env_name = "RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN"
    raw_enabled = os.environ.get(env_name)
    if raw_enabled is not None:
        if not env_flag(env_name, False):
            return False
    else:
        policy = _kernel_policy()
        exact_model_shape = (
            (int(hidden_size), int(num_layers), int(batch_size), int(prompt_tokens))
            if None not in (hidden_size, num_layers, batch_size, prompt_tokens)
            else None
        )
        model_shapes = {
            tuple(int(v) for v in shape)
            for shape in getattr(policy, "prefill_clampw_scan_model_shapes", ())
            if len(shape) == 4
        }
        if not (
            bool(getattr(policy, "fused_prefill_clampw_scan", False))
            or exact_model_shape in model_shapes
        ):
            return False
    if not _native_prefill_fused_scan_enabled():
        return False
    if fused_recurrent_scan_clampw is None or fused_recurrent_scan_clampw_available is None:
        return False
    try:
        return bool(fused_recurrent_scan_clampw_available())
    except Exception:
        return False


def _native_prefill_fused_scan_output_enabled() -> bool:
    """Runtime switch for fused prefill scan plus attention output prep."""

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT", False):
        return False
    if fused_recurrent_scan_output_prepare is None or fused_recurrent_scan_output_prepare_available is None:
        return False
    try:
        return bool(fused_recurrent_scan_output_prepare_available())
    except Exception:
        return False


def _native_prefill_default_scan_block_m(
    head_dim: int,
    batch_size: int | None = None,
    tokens: int | None = None,
    hidden_size: int | None = None,
) -> int:
    """Architecture-aware default row tile for optional prefill scans."""

    head_dim = int(head_dim)
    policy = _kernel_policy()
    if hidden_size is not None and batch_size is not None and tokens is not None:
        for policy_hidden, policy_batch, policy_tokens, policy_block_m in getattr(
            policy,
            "prefill_scan_block_m_model_shapes",
            (),
        ):
            if (int(hidden_size), int(batch_size), int(tokens)) == (
                int(policy_hidden),
                int(policy_batch),
                int(policy_tokens),
            ):
                return int(policy_block_m)
    if batch_size is not None and tokens is not None:
        for policy_batch, policy_tokens, policy_block_m in getattr(
            policy,
            "prefill_scan_block_m_shapes",
            (),
        ):
            if (int(batch_size), int(tokens)) == (int(policy_batch), int(policy_tokens)):
                return int(policy_block_m)
    policy_value = getattr(policy, "prefill_scan_block_m", None)
    if policy_value is not None:
        if batch_size is not None and int(batch_size) >= 4:
            batch_value = getattr(policy, "prefill_scan_block_m_b4", None)
            if batch_value is not None:
                return int(batch_value)
        if batch_size is not None and int(batch_size) >= 2:
            batch_value = getattr(policy, "prefill_scan_block_m_b2", None)
            if batch_value is not None:
                return int(batch_value)
        return int(policy_value)
    if head_dim == 64 and torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception:
            major, minor = 0, 0
        if (int(major), int(minor)) == (7, 0):
            return 32 if batch_size is not None and int(batch_size) >= 4 else 16
        if (int(major), int(minor)) == (8, 9):
            try:
                name = str(torch.cuda.get_device_name()).lower()
            except Exception:
                name = ""
            if "4090" in name:
                if (
                    batch_size is not None
                    and int(batch_size) >= 8
                    and tokens is not None
                    and int(tokens) == 128
                ):
                    return 32
                if (
                    batch_size is not None
                    and int(batch_size) >= 8
                    and tokens is not None
                    and int(tokens) >= 512
                    and hidden_size is not None
                    and int(hidden_size) == 2048
                ):
                    return 32
                return 8 if batch_size is not None and int(batch_size) >= 2 else 4
        if int(major) >= 12:
            batch_size = 1 if batch_size is None else int(batch_size)
            if batch_size <= 1:
                return 8
            if batch_size <= 2:
                return 16
            if batch_size <= 4:
                return 32
            return 64
    return head_dim


def _native_prefill_scan_block_m(
    head_dim: int,
    batch_size: int | None = None,
    tokens: int | None = None,
    hidden_size: int | None = None,
) -> int:
    """Row tile for optional recurrent scans; explicit env always wins."""

    return env_int(
        "RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M",
        _native_prefill_default_scan_block_m(head_dim, batch_size, tokens, hidden_size),
        lower=1,
        upper=int(head_dim),
    )


def _native_prefill_scan_num_warps(head_dim: int, block_m: int | None = None) -> int:
    """Triton warp count for the optional native prefill recurrent scan."""

    if block_m is None:
        block_m = _native_prefill_scan_block_m(head_dim)
    policy = _kernel_policy()
    policy_value = getattr(policy, "prefill_scan_num_warps", None)
    if policy_value is not None:
        default = int(policy_value)
    else:
        is_blackwell = False
        if torch.cuda.is_available():
            try:
                major, _minor = torch.cuda.get_device_capability()
                is_blackwell = int(major) >= 12
            except Exception:
                pass
        if is_blackwell and int(head_dim) == 64:
            default = 4 if int(block_m) >= 64 else 1
        else:
            default = 4 if int(block_m) < int(head_dim) else 8
    value = env_int("RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS", default, lower=1, upper=8)
    if value not in {1, 2, 4, 8}:
        raise ValueError(f"RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS must be one of 1, 2, 4, or 8; got {value}")
    return value


def _native_prefill_model_shape_selected(
    env_name: str,
    policy_name: str,
    batch_size: int | None,
    prompt_tokens: int | None,
    hidden_size: int | None,
    num_layers: int | None,
) -> bool:
    """Apply an optional exact model-shape restriction to a prefill route."""

    policy = _kernel_policy()
    raw = os.environ.get(env_name)
    if raw is None:
        shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(policy, policy_name, ())
            if len(shape) == 4
        }
    else:
        shapes = set()
        try:
            for item in raw.replace(",", " ").split():
                values = tuple(int(value) for value in item.lower().split("x"))
                if len(values) != 4 or any(value <= 0 for value in values):
                    raise ValueError
                shapes.add(values)
        except ValueError as exc:
            raise ValueError(f"{env_name} must contain HxLxBxT tuples") from exc
    if not shapes:
        return True
    if None in (batch_size, prompt_tokens, hidden_size, num_layers):
        return False
    return (
        int(hidden_size),
        int(num_layers),
        int(batch_size),
        int(prompt_tokens),
    ) in shapes


def _native_prefill_fused_shift_mix_enabled(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Runtime switch for prefill attention shift-mix fusion telemetry."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX",
        bool(getattr(policy, "fused_prefill_shift_mix", False)),
    ):
        return False
    if not _native_prefill_model_shape_selected(
        "RWKV7_NATIVE_PREFILL_SHIFT_MIX_MODEL_SHAPES",
        "prefill_shift_mix_model_shapes",
        batch_size,
        prompt_tokens,
        hidden_size,
        num_layers,
    ):
        return False
    if fused_attn_shift_mix is None or fused_attn_shift_mix_available is None:
        return False
    try:
        return bool(fused_attn_shift_mix_available())
    except Exception:
        return False


def _native_prefill_fused_state_prep_enabled(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Runtime switch for the native prefill state-prep fusion probe."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP",
        bool(getattr(policy, "fused_prefill_state_prep", False)),
    ):
        return False
    if not _native_prefill_model_shape_selected(
        "RWKV7_NATIVE_PREFILL_STATE_PREP_MODEL_SHAPES",
        "prefill_state_prep_model_shapes",
        batch_size,
        prompt_tokens,
        hidden_size,
        num_layers,
    ):
        return False
    if fused_prefill_state_prep is None or fused_prefill_state_prep_available is None:
        return False
    try:
        return bool(fused_prefill_state_prep_available())
    except Exception:
        return False


def _native_prefill_state_prep_layers(
    batch_size: int,
    prompt_tokens: int,
    hidden_size: int,
    num_layers: int,
) -> set[int] | None:
    """Return selected state-prep layers, or ``None`` for every layer."""

    specific = f"RWKV7_NATIVE_PREFILL_STATE_PREP_LAYERS_B{batch_size}_T{prompt_tokens}"
    raw = os.environ.get(
        specific,
        os.environ.get("RWKV7_NATIVE_PREFILL_STATE_PREP_LAYERS"),
    )
    if raw is not None:
        selected: set[int] = set()
        try:
            for item in raw.replace(" ", ",").split(","):
                item = item.strip()
                if not item:
                    continue
                if "-" in item:
                    start, end = (int(value) for value in item.split("-", 1))
                    if start < 0 or end < start:
                        raise ValueError
                    selected.update(range(start, end + 1))
                else:
                    value = int(item)
                    if value < 0:
                        raise ValueError
                    selected.add(value)
        except ValueError as exc:
            raise ValueError(
                f"{specific} must contain non-negative layers or inclusive ranges"
            ) from exc
        return {value for value in selected if value < int(num_layers)}

    target = (
        int(hidden_size),
        int(num_layers),
        int(batch_size),
        int(prompt_tokens),
    )
    for profile in getattr(
        _kernel_policy(), "prefill_state_prep_layer_counts", ()
    ):
        if len(profile) == 5 and tuple(int(value) for value in profile[:4]) == target:
            count = min(max(int(profile[4]), 0), int(num_layers))
            return set(range(count))
    return None


def _native_prefill_fused_state_scan_max_batch() -> int | None:
    """Optional batch ceiling for the fused state-prep scan route."""

    raw = os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_MAX_BATCH")
    if raw is None or not raw.strip():
        policy = _kernel_policy()
        value = getattr(policy, "fused_prefill_state_scan_max_batch", None)
        return None if value is None else int(value)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_MAX_BATCH must be an integer") from exc
    if value < 1:
        raise ValueError("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_MAX_BATCH must be >= 1")
    return value


def _native_prefill_fused_state_scan_enabled(batch_size: int | None = None) -> bool:
    """Runtime switch for the fused state-prep plus scan probe."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN",
        bool(getattr(policy, "fused_prefill_state_scan", False)),
    ):
        return False
    max_batch = _native_prefill_fused_state_scan_max_batch()
    if max_batch is not None and batch_size is not None and int(batch_size) > max_batch:
        return False
    if fused_recurrent_scan_state_prep is None or fused_recurrent_scan_state_prep_available is None:
        return False
    try:
        return bool(fused_recurrent_scan_state_prep_available())
    except Exception:
        return False


def _native_prefill_state_prep_w_dtype() -> str:
    """Output dtype policy for fused native-prefill W decay.

    ``fp32`` preserves the historical torch expression
    ``exp(... w.float())``. ``input`` stores the decay in the model dtype to
    reduce bandwidth into the split-row scan; it is opt-in until end-to-end
    rows prove correctness and speed for a card/model.
    """

    raw = os.environ.get("RWKV7_NATIVE_PREFILL_STATE_PREP_W_DTYPE", "fp32").strip().lower()
    aliases = {
        "fp32": "fp32",
        "float32": "fp32",
        "f32": "fp32",
        "input": "input",
        "model": "input",
        "same": "input",
        "fp16": "input",
        "bf16": "input",
    }
    if raw not in aliases:
        raise ValueError(
            "RWKV7_NATIVE_PREFILL_STATE_PREP_W_DTYPE must be 'fp32' or 'input' "
            f"(aliases: same/model/fp16/bf16); got {raw!r}"
        )
    return aliases[raw]


def _native_prefill_fused_output_enabled() -> bool:
    """Runtime switch for native prefill output-prep fusion.

    This reuses the profitable decode fused-output-prep kernel, but keeps the
    prefill path explicit until end-to-end prompt rows prove it helps each
    card/model shape.
    """

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_OUTPUT",
        bool(getattr(policy, "fused_prefill_output", False)),
    ):
        return False
    if fused_attn_output_prepare is None or fused_attn_output_prepare_available is None:
        return False
    try:
        return bool(fused_attn_output_prepare_available())
    except Exception:
        return False


def _native_prefill_fused_output_project_enabled() -> bool:
    """Runtime switch for native prefill output-prep plus ``o_proj`` fusion.

    This is an opt-in experiment for the bsz=1 prompt-prefill gap.  The kernel
    is intentionally disabled by default because it recomputes the prepared
    attention output inside the projection tile; exact-card benchmark rows must
    prove it beats the cuBLAS ``o_proj`` path before it becomes a default.
    """

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_OUTPUT_PROJECT", False):
        return False
    if fused_attn_output_project is None or fused_attn_output_project_available is None:
        return False
    try:
        return bool(fused_attn_output_project_available())
    except Exception:
        return False


def _native_prefill_fused_output_project_block_m() -> int:
    """Output tile for the native prefill fused output-project experiment."""

    default = env_int("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT_BLOCK_M", 16, lower=1, upper=128)
    return env_int("RWKV7_NATIVE_PREFILL_FUSED_OUTPUT_PROJECT_BLOCK_M", default, lower=1, upper=128)


def _native_prefill_fused_wavg_lora_requested() -> bool:
    """Return whether the prefill W/A/G/V-gate LoRA fusion probe is requested."""

    return env_flag("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA", False)


def _native_prefill_fused_wavg_lora_max_m() -> int:
    """Maximum flattened rows for prefill WAVG LoRA before falling back.

    Initial card-local probes were profitable for `B*T=512` but slower for
    `B*T=2048`, so the opt-in path defaults to the small-prefill shape until an
    exact-card sweep proves a larger tile.
    """

    return env_int("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_MAX_M", 1024, lower=1, upper=1 << 30)


def _native_prefill_fused_wavg_lora_enabled(total_rows: int) -> bool:
    """Runtime switch for the native prefill W/A/G/V-gate LoRA fusion probe."""

    if not _native_prefill_fused_wavg_lora_requested():
        return False
    if int(total_rows) > _native_prefill_fused_wavg_lora_max_m():
        return False
    if fused_wavg_lora is None or fused_wavg_lora_available is None:
        return False
    try:
        return bool(fused_wavg_lora_available())
    except Exception:
        return False


def _native_prefill_fused_wavg_lora_blocks() -> tuple[int, int, int]:
    """Return ``(block_m, block_r, block_k)`` for prefill WAVG LoRA."""

    vals = []
    for name, fallback, default, upper in (
        ("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_BLOCK_M", "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_M", 64, 128),
        ("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_BLOCK_R", "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_R", 64, 128),
        ("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_BLOCK_K", "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_K", 64, 256),
    ):
        raw = os.environ.get(name, os.environ.get(fallback))
        if raw is None:
            vals.append(env_int(name, int(default), lower=1, upper=upper))
        else:
            try:
                val = int(str(raw).strip())
            except ValueError:
                val = int(default)
            vals.append(min(max(1, val), upper))
    return vals[0], vals[1], vals[2]


def _native_prefill_fused_sequence_ffn_enabled(
    total_rows: int,
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Enable the tensor-core sequence FFN only for measured prefill shapes."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_SEQUENCE_FFN",
        bool(getattr(policy, "fused_prefill_sequence_ffn", False)),
    ):
        return False
    min_rows = env_int(
        "RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_MIN_ROWS",
        int(getattr(policy, "prefill_sequence_ffn_min_rows", 128)),
        lower=1,
    )
    policy_max = getattr(policy, "prefill_sequence_ffn_max_rows", None)
    max_rows = env_int(
        "RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_MAX_ROWS",
        (1 << 30) if policy_max is None else int(policy_max),
        lower=1,
    )
    raw_extra = os.environ.get("RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_EXTRA_ROWS")
    if raw_extra is None:
        extra_rows = {int(v) for v in getattr(policy, "prefill_sequence_ffn_extra_rows", ())}
    else:
        try:
            extra_rows = {int(v) for v in raw_extra.replace(",", " ").split() if int(v) > 0}
        except ValueError as exc:
            raise ValueError("RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_EXTRA_ROWS must contain integers") from exc
    raw_model_shapes = os.environ.get("RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_MODEL_SHAPES")
    if raw_model_shapes is None:
        model_shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(policy, "prefill_sequence_ffn_model_shapes", ())
            if len(shape) == 4
        }
    else:
        model_shapes = set()
        try:
            for item in raw_model_shapes.replace(",", " ").split():
                values = tuple(int(value) for value in item.lower().split("x"))
                if len(values) != 4 or any(value <= 0 for value in values):
                    raise ValueError
                model_shapes.add(values)
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_MODEL_SHAPES must contain HxLxBxT tuples"
            ) from exc
    exact_model_shape = (
        (int(hidden_size), int(num_layers), int(batch_size), int(prompt_tokens))
        if None not in (hidden_size, num_layers, batch_size, prompt_tokens)
        else None
    )
    if not (
        min_rows <= int(total_rows) <= max_rows
        or int(total_rows) in extra_rows
        or exact_model_shape in model_shapes
    ):
        return False
    if fused_sequence_ffn is None or fused_sequence_ffn_available is None:
        return False
    try:
        return bool(fused_sequence_ffn_available())
    except Exception:
        return False


def _native_prefill_fp16_accum_ffn_key_enabled(
    batch_size: int,
    prompt_tokens: int,
    hidden_size: int,
    num_layers: int,
    dtype: torch.dtype,
) -> bool:
    """Select reduced-precision accumulation only for measured FFN-key shapes."""

    if dtype != torch.float16:
        return False
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is None or not hasattr(matmul, "allow_fp16_accumulation"):
        return False
    policy = _kernel_policy()
    raw_shapes = os.environ.get(
        "RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY_MODEL_SHAPES"
    )
    if raw_shapes is None:
        model_shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(
                policy,
                "prefill_fp16_accum_ffn_key_model_shapes",
                (),
            )
            if len(shape) == 4
        }
    else:
        model_shapes = set()
        try:
            for item in raw_shapes.replace(",", " ").split():
                values = tuple(int(value) for value in item.lower().split("x"))
                if len(values) != 4 or any(value <= 0 for value in values):
                    raise ValueError
                model_shapes.add(values)
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY_MODEL_SHAPES must "
                "contain HxLxBxT tuples"
            ) from exc
    exact_shape = (
        int(hidden_size),
        int(num_layers),
        int(batch_size),
        int(prompt_tokens),
    )
    selected = exact_shape in model_shapes
    return bool(
        selected
        and env_flag(
            "RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY",
            selected,
        )
    )


def _native_prefill_fp16_accum_ffn_key_layers(
    batch_size: int,
    prompt_tokens: int,
    hidden_size: int,
    num_layers: int,
) -> set[int] | None:
    """Return selected FFN-key accumulation layers, or ``None`` for all."""

    specific = (
        "RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY_LAYERS_"
        f"B{batch_size}_T{prompt_tokens}"
    )
    raw = os.environ.get(
        specific,
        os.environ.get("RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY_LAYERS"),
    )
    if raw is not None:
        selected: set[int] = set()
        try:
            for item in raw.replace(" ", ",").split(","):
                item = item.strip()
                if not item:
                    continue
                if "-" in item:
                    start, end = (int(value) for value in item.split("-", 1))
                    if start < 0 or end < start:
                        raise ValueError
                    selected.update(range(start, end + 1))
                else:
                    value = int(item)
                    if value < 0:
                        raise ValueError
                    selected.add(value)
        except ValueError as exc:
            raise ValueError(
                f"{specific} must contain non-negative layers or inclusive ranges"
            ) from exc
        return {value for value in selected if value < int(num_layers)}

    target = (
        int(hidden_size),
        int(num_layers),
        int(batch_size),
        int(prompt_tokens),
    )
    for profile in getattr(
        _kernel_policy(), "prefill_fp16_accum_ffn_key_layer_counts", ()
    ):
        if len(profile) == 5 and tuple(int(value) for value in profile[:4]) == target:
            count = min(max(int(profile[4]), 0), int(num_layers))
            return set(range(count))
    return None


def _native_prefill_stacked_rkv_enabled(
    total_rows: int,
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Shape gate for lazy packed strided-batched R/K/V GEMM."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_STACKED_RKV",
        bool(getattr(policy, "fused_prefill_stacked_rkv", False)),
    ):
        return False
    min_rows = env_int(
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_MIN_ROWS",
        int(getattr(policy, "prefill_stacked_rkv_min_rows", 128)),
        lower=1,
    )
    policy_max = getattr(policy, "prefill_stacked_rkv_max_rows", None)
    max_rows = env_int(
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_MAX_ROWS",
        (1 << 30) if policy_max is None else int(policy_max),
        lower=1,
    )
    raw_extra = os.environ.get("RWKV7_NATIVE_PREFILL_STACKED_RKV_EXTRA_ROWS")
    if raw_extra is None:
        extra_rows = {int(v) for v in getattr(policy, "prefill_stacked_rkv_extra_rows", ())}
    else:
        try:
            extra_rows = {int(v) for v in raw_extra.replace(",", " ").split() if int(v) > 0}
        except ValueError as exc:
            raise ValueError("RWKV7_NATIVE_PREFILL_STACKED_RKV_EXTRA_ROWS must contain integers") from exc
    raw_shapes = os.environ.get("RWKV7_NATIVE_PREFILL_STACKED_RKV_SHAPES")
    if raw_shapes is None:
        shapes = {
            (int(shape[0]), int(shape[1]))
            for shape in getattr(policy, "prefill_stacked_rkv_shapes", ())
            if len(shape) == 2
        }
    else:
        shapes = set()
        try:
            for item in raw_shapes.replace(",", " ").split():
                left, right = item.lower().split("x", 1)
                shape = (int(left), int(right))
                if shape[0] <= 0 or shape[1] <= 0:
                    raise ValueError
                shapes.add(shape)
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_PREFILL_STACKED_RKV_SHAPES must contain BxT pairs"
            ) from exc
    exact_shape = (
        (int(batch_size), int(prompt_tokens))
        if batch_size is not None and prompt_tokens is not None
        else None
    )
    raw_model_shapes = os.environ.get("RWKV7_NATIVE_PREFILL_STACKED_RKV_MODEL_SHAPES")
    if raw_model_shapes is None:
        model_shapes = {
            tuple(int(v) for v in shape)
            for shape in getattr(policy, "prefill_stacked_rkv_model_shapes", ())
            if len(shape) == 4
        }
    else:
        model_shapes = set()
        try:
            for item in raw_model_shapes.replace(",", " ").split():
                values = tuple(int(v) for v in item.lower().split("x"))
                if len(values) != 4 or any(v <= 0 for v in values):
                    raise ValueError
                model_shapes.add(values)
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_PREFILL_STACKED_RKV_MODEL_SHAPES must contain HxLxBxT tuples"
            ) from exc
    exact_model_shape = (
        (int(hidden_size), int(num_layers), int(batch_size), int(prompt_tokens))
        if None not in (hidden_size, num_layers, batch_size, prompt_tokens)
        else None
    )
    return bool(
        min_rows <= int(total_rows) <= max_rows
        or int(total_rows) in extra_rows
        or exact_shape in shapes
        or exact_model_shape in model_shapes
    )


def _native_prefill_stacked_rkv_weights(model, packs) -> list[torch.Tensor]:
    """Lazily pack transposed dense R/K/V weights for one bmm per layer.

    The cache is an ordinary Python attribute (not a parameter/buffer), so it
    never changes checkpoints.  Weight data pointers and tensor versions form
    the key, which makes adapter merges or in-place edits rebuild safely.
    """

    signatures = []
    for p in packs:
        rw, kw, vw = p[20], p[21], p[22]
        if not all(isinstance(weight, torch.Tensor) and weight.dim() == 2 for weight in (rw, kw, vw)):
            return []
        signatures.append(
            tuple((int(weight.data_ptr()), int(getattr(weight, "_version", 0))) for weight in (rw, kw, vw))
        )
    key = tuple(signatures)
    cached = getattr(model, "_rwkv7_native_prefill_stacked_rkv_cache", None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == key:
        return cached[1]
    packed = [torch.stack((p[20].t(), p[21].t(), p[22].t()), dim=0).contiguous() for p in packs]
    setattr(model, "_rwkv7_native_prefill_stacked_rkv_cache", (key, packed))
    return packed


def _native_prefill_sequence_ffn_blocks(total_rows: int | None = None) -> tuple[int, int, int, int, int]:
    """Return measured ``(BM, BN, key-BK, value-BK, group-M)`` tiles."""

    policy = _kernel_policy()
    large_min = int(getattr(policy, "prefill_sequence_ffn_large_min_rows", 1024))
    use_large = total_rows is not None and int(total_rows) >= large_min
    defaults = tuple(
        getattr(
            policy,
            "prefill_sequence_ffn_large_blocks" if use_large else "prefill_sequence_ffn_blocks",
            (128, 128, 32, 64, 8),
        )
    )
    names = (
        f"RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_{'LARGE_' if use_large else ''}BLOCK_M",
        f"RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_{'LARGE_' if use_large else ''}BLOCK_N",
        f"RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_{'LARGE_' if use_large else ''}KEY_BLOCK_K",
        f"RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_{'LARGE_' if use_large else ''}VALUE_BLOCK_K",
        f"RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_{'LARGE_' if use_large else ''}GROUP_M",
    )
    return tuple(env_int(name, int(default), lower=1, upper=256) for name, default in zip(names, defaults))  # type: ignore[return-value]


def _native_prefill_sequence_ffn_launch() -> tuple[int, int]:
    """Return measured ``(num_stages, num_warps)`` launch settings."""

    policy = _kernel_policy()
    stages = env_int(
        "RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_NUM_STAGES",
        int(getattr(policy, "prefill_sequence_ffn_num_stages", 3)),
        lower=1,
        upper=5,
    )
    warps = env_int(
        "RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_NUM_WARPS",
        int(getattr(policy, "prefill_sequence_ffn_num_warps", 4)),
        lower=1,
        upper=8,
    )
    if warps not in {1, 2, 4, 8}:
        raise ValueError("RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_NUM_WARPS must be 1, 2, 4, or 8")
    return stages, warps


def _native_graph_fused_recurrent_output_enabled() -> bool:
    """Runtime switch for fused recurrent update plus output-prep."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT", bool(getattr(policy, "fused_recurrent_output", True))):
        return False
    if fused_recurrent_output_prepare is None or fused_recurrent_output_prepare_available is None:
        return False
    try:
        return bool(fused_recurrent_output_prepare_available())
    except Exception:
        return False


def _native_graph_fused_recurrent_raw_enabled() -> bool:
    """Fold W decay and K/KK preparation into recurrent output fusion."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW", bool(getattr(policy, "fused_recurrent_raw", False))):
        return False
    return bool(fused_recurrent_output_prepare_raw is not None and _native_graph_fused_recurrent_output_enabled())


def _native_graph_fp16_recurrent_enabled(state: torch.Tensor, elapsed) -> bool:
    return bool(
        env_flag("RWKV7_NATIVE_GRAPH_FP16_RECURRENT", False)
        and native_fp16_recurrent_output_prepare_raw is not None
        and elapsed is not None
        and state.dtype == torch.float16
        and int(state.shape[-1]) == 64
    )


def _native_graph_fused_output_enabled() -> bool:
    """Runtime switch for the experimental native-graph output-prep Triton path."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT", bool(getattr(policy, "fused_output", True))):
        return False
    if fused_attn_output_prepare is None or fused_attn_output_prepare_available is None:
        return False
    try:
        return bool(fused_attn_output_prepare_available())
    except Exception:
        return False


def _native_graph_fused_output_project_enabled() -> bool:
    """Runtime switch for fused output-prep plus ``o_proj`` in native_graph."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT", bool(getattr(policy, "fused_output_project", False))):
        return False
    if fused_attn_output_project is None or fused_attn_output_project_available is None:
        return False
    try:
        return bool(fused_attn_output_project_available())
    except Exception:
        return False


def _native_graph_fused_output_project_block_m() -> int:
    """Output-projection row tile used by the prototype fused output-project kernel."""

    policy = _kernel_policy()
    default = int(getattr(policy, "output_project_block_m", 16))
    return env_int("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT_BLOCK_M", default, lower=1, upper=128)


def _native_graph_fused_projection_enabled() -> bool:
    """Runtime switch for the experimental native-graph R/K/V + W/A/G projection path."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_PROJECTION", bool(getattr(policy, "fused_projection", False))):
        return False
    if fused_rkv_wavg_projection is None or fused_rkv_wavg_projection_available is None:
        return False
    try:
        return bool(fused_rkv_wavg_projection_available())
    except Exception:
        return False


def _native_graph_fused_wag_lora_enabled() -> bool:
    """Runtime switch for the native-graph W/A/G LoRA-only fusion probe."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA", bool(getattr(policy, "fused_wag_lora", False))):
        return False
    if fused_wag_lora is None or fused_wag_lora_available is None:
        return False
    try:
        return bool(fused_wag_lora_available())
    except Exception:
        return False


def _native_graph_sm70_wagv_lora_enabled(rows: int, hidden_size: int) -> bool:
    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_GRAPH_SM70_WAGV_LORA",
        bool(getattr(policy, "sm70_wagv_lora", False)),
    ):
        return False
    if int(rows) < 1 or int(rows) > 4 or int(hidden_size) < 1024:
        return False
    if sm70_wagv_lora is None or sm70_wagv_lora_available is None:
        return False
    try:
        return bool(sm70_wagv_lora_available())
    except Exception:
        return False


def _native_graph_fused_wavg_lora_enabled(rows: int, hidden_size: int) -> bool:
    """Runtime switch for the native-graph W/A/G/V-gate LoRA fusion probe."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA", bool(getattr(policy, "fused_wavg_lora", False))):
        return False
    default_max = getattr(policy, "wavg_lora_bsz1_max_hidden", None)
    bsz1_max_hidden = env_int(
        "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BSZ1_MAX_HIDDEN",
        0 if default_max is None else int(default_max),
        lower=0,
    )
    if int(rows) == 1 and bsz1_max_hidden > 0 and int(hidden_size) > bsz1_max_hidden:
        return False
    if fused_wavg_lora is None or fused_wavg_lora_available is None:
        return False
    try:
        return bool(fused_wavg_lora_available())
    except Exception:
        return False


def _native_graph_fused_norm_mix_enabled() -> bool:
    """Runtime switch for decode layer-norm/residual/time-mix fusion."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX",
        bool(getattr(policy, "fused_norm_mix", False)),
    ):
        return False
    if (
        fused_attn_norm_mix6_decode is None
        or fused_ffn_add_norm_mix_decode is None
        or fused_decode_norm_mix_available is None
    ):
        return False
    try:
        return bool(fused_decode_norm_mix_available())
    except Exception:
        return False


def _native_graph_fused_norm_mix_num_warps() -> int:
    policy = _kernel_policy()
    default = int(getattr(policy, "norm_mix_num_warps", 4))
    value = env_int("RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS", default, lower=1, upper=8)
    if value not in {1, 2, 4, 8}:
        raise ValueError(f"RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS must be one of 1, 2, 4, or 8; got {value}")
    return value


def _native_graph_blackwell_norm_mix_enabled(
    residual, attention, previous, *, layer_index: int
) -> bool:
    if not env_flag("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX", False):
        return False
    batch_size = int(residual.shape[0]) if residual.ndim > 1 else 1
    selected = os.environ.get(
        f"RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX_LAYERS_B{batch_size}",
        os.environ.get("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX_LAYERS", ""),
    ).strip()
    if selected:
        try:
            layers = {int(value.strip()) for value in selected.split(",") if value.strip()}
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX_LAYERS must be comma-separated integers"
            ) from exc
        if int(layer_index) not in layers:
            return False
    if blackwell_ffn_add_norm_mix is None or blackwell_norm_mix_should_use is None:
        return False
    try:
        return bool(blackwell_norm_mix_should_use(residual, attention, previous))
    except Exception:
        return False


def _native_graph_sm70_linear_enabled() -> bool:
    """Whether measured sm_70 small-row linear routes may be captured."""

    policy = _kernel_policy()
    return bool(
        env_flag("RWKV7_NATIVE_GRAPH_SM70_LINEAR", bool(getattr(policy, "sm70_linear", False)))
        and sm70_linear is not None
        and sm70_linear_should_use is not None
        and sm70_linear_threads is not None
    )


def _native_graph_ada_sparse_ffn_enabled() -> bool:
    """Whether the measured sm_89 sparse FFN route may be captured."""

    policy = _kernel_policy()
    return bool(
        env_flag(
            "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN",
            bool(getattr(policy, "ada_sparse_ffn", False)),
        )
        and ada_sparse_ffn_down_add is not None
        and ada_ffn_up is not None
        and ada_sparse_ffn_should_use is not None
    )


def _native_graph_ada_linear_enabled() -> bool:
    """Whether measured no-copy sm_89 exact-row linears may be captured."""

    policy = _kernel_policy()
    return bool(
        env_flag(
            "RWKV7_NATIVE_GRAPH_ADA_LINEAR",
            bool(getattr(policy, "ada_linear", False)),
        )
        and ada_linear is not None
        and ada_linear_should_use is not None
    )


def _native_graph_ada_linear_should_route(rows: int, role: str) -> bool:
    """Shape/role gate; row 1 remains a probe while measured row 2 is default."""

    if not _native_graph_ada_linear_enabled():
        return False
    policy = _kernel_policy()
    raw_rows = os.environ.get(
        "RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROWS",
        str(getattr(policy, "ada_linear_rows", "2 4")),
    )
    try:
        enabled_rows = {int(item) for item in raw_rows.replace(",", " ").split()}
    except ValueError:
        enabled_rows = {2}
    raw_roles = os.environ.get("RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROLES")
    if raw_roles is None:
        raw_roles = "hidden" if int(rows) == 4 else "hidden,ffn_up,ffn_down"
    enabled_roles = {item.strip().lower() for item in raw_roles.replace(",", " ").split() if item.strip()}
    return int(rows) in enabled_rows and role.lower() in enabled_roles


def _native_graph_ada_wagv_lora_enabled(rows: int, hidden_size: int, max_rank: int) -> bool:
    """Whether the no-copy sm_89 grouped low-rank route may be captured."""

    policy = _kernel_policy()
    return bool(
        env_flag(
            "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA",
            bool(getattr(policy, "ada_wagv_lora", False)),
        )
        and ada_wagv_lora is not None
        and ada_wagv_lora_should_use is not None
        and ada_wagv_lora_should_use(int(rows), int(hidden_size), int(max_rank))
    )


def _native_graph_ada_wag_lora_enabled() -> bool:
    """Whether the exact-card W/A/G-only low-rank route may be captured."""

    if not env_flag("RWKV7_NATIVE_GRAPH_ADA_WAG_LORA", False):
        return False
    if ada_wag_lora is None or ada_wagv_lora_available is None:
        return False
    try:
        return bool(ada_wagv_lora_available())
    except Exception:
        return False


def _native_graph_linear_dispatch(x: torch.Tensor, weight, *, role: str) -> torch.Tensor:
    """Dispatch dense or native-quantized linears during graph capture."""

    rows = 1 if x.dim() == 1 else int(x.shape[0])
    outputs, inputs = _graph_linear_shape(weight)
    if (
        _graph_linear_is_dense(weight)
        and role != "head"
        and _native_graph_ada_linear_should_route(rows, role)
        and ada_linear_should_use(rows, outputs, inputs)
    ):
        return ada_linear(x, weight)
    if not _graph_linear_is_dense(weight):
        return _graph_linear_call(x, weight)
    if (
        sm70_orig_linear is not None
        and role == "hidden"
        and rows in {2, 4}
        and outputs == inputs
        and inputs >= 2048
    ):
        return sm70_orig_linear(x, weight)
    if not _native_graph_sm70_linear_enabled():
        return F.linear(x, weight)
    if not sm70_linear_should_use(rows, outputs, inputs, role=role):
        return F.linear(x, weight)
    threads = sm70_linear_threads(rows, outputs, inputs, role=role)
    return sm70_linear(x, weight, threads=threads)


def _native_graph_ffn_up_relu2_dispatch(x: torch.Tensor, weight) -> torch.Tensor:
    rows = 1 if x.dim() == 1 else int(x.shape[0])
    outputs, inputs = _graph_linear_shape(weight)
    if (
        _graph_linear_is_dense(weight)
        and _native_graph_ada_linear_should_route(rows, "ffn_up")
        and ada_linear_should_use(rows, outputs, inputs)
    ):
        return torch.relu(ada_linear(x, weight)) ** 2
    fused_quant = getattr(weight, "rwkv7_forward_relu2", None)
    if not _graph_linear_is_dense(weight) and callable(fused_quant):
        return fused_quant(x)
    if not _graph_linear_is_dense(weight):
        fused = getattr(weight, "rwkv7_forward_relu2", None)
        if bool(getattr(weight, "fused_relu2", False)) and callable(fused):
            return fused(x)
        return torch.relu(_graph_linear_call(x, weight)) ** 2
    if (
        not _native_graph_sm70_linear_enabled()
        or sm70_ffn_up_relu2 is None
        or sm70_ffn_up_relu2_should_use is None
    ):
        return torch.relu(F.linear(x, weight)) ** 2
    if not sm70_ffn_up_relu2_should_use(rows, outputs, inputs):
        return torch.relu(F.linear(x, weight)) ** 2
    threads = sm70_linear_threads(rows, outputs, inputs, role="ffn_up")
    return sm70_ffn_up_relu2(x, weight, threads=threads)


def _native_graph_ffn_down_add_dispatch(
    x: torch.Tensor,
    weight,
    residual: torch.Tensor,
) -> torch.Tensor:
    rows = 1 if x.dim() == 1 else int(x.shape[0])
    outputs, inputs = _graph_linear_shape(weight)
    if (
        _graph_linear_is_dense(weight)
        and _native_graph_ada_linear_should_route(rows, "ffn_down")
        and ada_linear_should_use(rows, outputs, inputs)
    ):
        return residual + ada_linear(x, weight)
    fused_quant = getattr(weight, "rwkv7_forward_add", None)
    if not _graph_linear_is_dense(weight) and callable(fused_quant):
        return fused_quant(x, residual)
    if not _graph_linear_is_dense(weight):
        return residual + _graph_linear_call(x, weight)
    if (
        not _native_graph_sm70_linear_enabled()
        or sm70_ffn_down_add is None
        or sm70_ffn_down_add_should_use is None
    ):
        return residual + F.linear(x, weight)
    if not sm70_ffn_down_add_should_use(rows, outputs, inputs):
        return residual + F.linear(x, weight)
    threads = sm70_linear_threads(rows, outputs, inputs, role="ffn_down")
    return sm70_ffn_down_add(x, weight, residual, threads=threads)


def _native_graph_ffn_dispatch(
    x: torch.Tensor,
    up_weight,
    down_weight,
    residual: torch.Tensor,
    *,
    sparse_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route the complete FFN boundary so sparse kernels can avoid ReLU² IO."""

    rows = 1 if x.dim() == 1 else int(x.shape[0])
    outputs, inputs = _graph_linear_shape(down_weight)
    if (
        _graph_linear_is_dense(up_weight)
        and _graph_linear_is_dense(down_weight)
        and _native_graph_ada_sparse_ffn_enabled()
        and rows <= env_int(
            "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_MAX_ROWS",
            int(getattr(_kernel_policy(), "ada_sparse_ffn_max_rows", 19)),
            lower=1,
            upper=19,
        )
        and ada_sparse_ffn_should_use(rows, outputs, inputs)
    ):
        if env_flag(
            "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_UP",
            bool(getattr(_kernel_policy(), "ada_sparse_ffn_up", True)),
        ):
            preact = ada_ffn_up(x, up_weight)
        else:
            preact = F.linear(x, up_weight)
        target = residual if env_flag(
            "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_INPLACE",
            bool(getattr(_kernel_policy(), "ada_sparse_ffn_inplace", False)),
        ) else sparse_out
        return ada_sparse_ffn_down_add(preact, down_weight, residual, out=target)
    hidden = _native_graph_ffn_up_relu2_dispatch(x, up_weight)
    return _native_graph_ffn_down_add_dispatch(hidden, down_weight, residual)


def prewarm_ada_sparse_ffn(packs, rows: int = 1) -> int:
    """Pack sparse FFN down weights before CUDA graph capture.

    Creating the transposed weights during capture places them in a graph
    private pool. Prepacking on the normal stream gives each enabled batch
    shape a stable read-only operand before its independent graph is captured.
    """

    max_rows = env_int(
        "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_MAX_ROWS",
        int(getattr(_kernel_policy(), "ada_sparse_ffn_max_rows", 19)),
        lower=1,
        upper=19,
    )
    if (
        not _native_graph_ada_sparse_ffn_enabled()
        or ada_sparse_ffn_pack_weight is None
        or int(rows) > max_rows
    ):
        return 0
    packed = 0
    for operands in packs:
        down_weight = operands[-2]
        if not _graph_linear_is_dense(down_weight):
            continue
        outputs, inputs = _graph_linear_shape(down_weight)
        if not ada_sparse_ffn_should_use(1, outputs, inputs):
            continue
        ada_sparse_ffn_pack_weight(down_weight, cache_tag=int(rows))
        if (
            env_flag("RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_FP32_ACCUM", False)
            and ada_sparse_ffn_prepare_fp32_scratch is not None
        ):
            ada_sparse_ffn_prepare_fp32_scratch(down_weight, int(rows))
        elif (
            os.environ.get(
                "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_DETERMINISTIC_SPLITS", "0"
            ).strip()
            == "4"
            and int(rows) >= 8
            and ada_sparse_ffn_prepare_deterministic_scratch is not None
        ):
            ada_sparse_ffn_prepare_deterministic_scratch(down_weight, int(rows))
        packed += 1
    return packed


def _native_graph_rkv_policy() -> str:
    """Return the optional VKWR-inspired R/K/V projection dispatch policy.

    VKWR stacks the receptance/key/value matrices and uses a grouped batched
    projection for selected small-row decode cases.  Keep the HF adapter's
    historical three-``F.linear`` path by default and enable the stacked path
    only through ``RWKV7_NATIVE_GRAPH_RKV_POLICY=vkwr_auto`` while collecting
    telemetry.
    """

    policy = _kernel_policy()
    default = str(getattr(policy, "rkv_policy", "manual"))
    raw = os.environ.get("RWKV7_NATIVE_GRAPH_RKV_POLICY", default).strip().lower()
    if raw in {"", "manual", "explicit", "env"}:
        return "manual"
    if raw in {"0", "false", "no", "off", "disabled"}:
        return "off"
    if raw in {"vkwr", "vkwr_auto", "auto", "stacked", "bmm"}:
        return "vkwr_auto"
    return "manual"


def _native_graph_int_env(name: str, default: int, *, lo: int = 1, hi: int | None = None) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _native_graph_vkwr_rkv_dispatch(rows: int, hidden_size: int) -> bool:
    """VKWR-style row gate for stacked R/K/V native-graph decode.

    VKWR's automatic RKV path is used for one-row decode and medium tiny-row
    batches (roughly 4..64 rows), but not for rows 2/3.  Mirroring that rule
    avoids forcing a grouped path into shapes where three cuBLAS calls can be
    competitive or faster.
    """

    if _native_graph_rkv_policy() != "vkwr_auto":
        return False
    if rows <= 0 or hidden_size <= 0:
        return False
    min_hidden = _native_graph_int_env("RWKV7_NATIVE_GRAPH_RKV_MIN_HIDDEN", 1, lo=1)
    max_rows = _native_graph_int_env("RWKV7_NATIVE_GRAPH_RKV_MAX_ROWS", 64, lo=1, hi=4096)
    if hidden_size < min_hidden:
        return False
    return rows == 1 or (4 <= rows <= max_rows)


def _native_graph_rkv_project(
    xr: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    Rw,
    Kw,
    Vw,
    RKVw: torch.Tensor,
    rows: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project R/K/V with either separate linears or VKWR-style stacked bmm."""

    dense_rkv = all(_graph_linear_is_dense(item) for item in (Rw, Kw, Vw))
    output_size = int(_graph_linear_shape(Rw)[0])
    if dense_rkv and _native_graph_vkwr_rkv_dispatch(int(rows), int(hidden_size)) and RKVw.numel() != 0:
        shared_storage = False
        try:
            row_values = int(rows) * int(hidden_size)
            shared_storage = bool(
                xr.is_contiguous()
                and xk.is_contiguous()
                and xv.is_contiguous()
                and xr.untyped_storage().data_ptr() == xk.untyped_storage().data_ptr()
                and xr.untyped_storage().data_ptr() == xv.untyped_storage().data_ptr()
                and int(xk.storage_offset()) == int(xr.storage_offset()) + row_values
                and int(xv.storage_offset()) == int(xr.storage_offset()) + 2 * row_values
            )
        except Exception:
            shared_storage = False
        if shared_storage:
            flat = xr.as_strided(
                (3, int(rows), int(hidden_size)),
                (int(rows) * int(hidden_size), int(hidden_size), 1),
            )
        elif xr.dim() == 1:
            flat = torch.stack(
                (
                    xr.reshape(1, hidden_size),
                    xk.reshape(1, hidden_size),
                    xv.reshape(1, hidden_size),
                ),
                dim=0,
            )
        else:
            flat = torch.stack(
                (
                    xr.reshape(rows, hidden_size),
                    xk.reshape(rows, hidden_size),
                    xv.reshape(rows, hidden_size),
                ),
                dim=0,
            )
        rkv = torch.bmm(flat, RKVw)
        if xr.dim() == 1:
            return rkv[0, 0], rkv[1, 0], rkv[2, 0]
        return rkv[0], rkv[1], rkv[2]
    if (
        dense_rkv
        and output_size == int(hidden_size)
        and sm70_orig_rkv is not None
        and int(rows) in {2, 4}
        and int(hidden_size) >= 2048
    ):
        return sm70_orig_rkv(xr, xk, xv, Rw, Kw, Vw)
    if (
        dense_rkv
        and output_size == int(hidden_size)
        and _native_graph_sm70_linear_enabled()
        and sm70_rkv is not None
        and sm70_rkv_should_use is not None
        and sm70_rkv_threads is not None
        and sm70_rkv_should_use(int(rows), int(hidden_size))
    ):
        threads = sm70_rkv_threads(int(rows), int(hidden_size))
        return sm70_rkv(xr, xk, xv, Rw, Kw, Vw, threads=threads)
    return (
        _native_graph_linear_dispatch(xr, Rw, role="hidden"),
        _native_graph_linear_dispatch(xk, Kw, role="hidden"),
        _native_graph_linear_dispatch(xv, Vw, role="hidden"),
    )


def _native_graph_fused_wag_lora_blocks() -> tuple[int, int, int]:
    """Return ``(block_m, block_r, block_k)`` for the W/A/G LoRA probe."""

    policy = _kernel_policy()
    defaults = tuple(getattr(policy, "wag_lora_blocks", (64, 64, 64)))
    return env_blocks(
        ("RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_M", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_R", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_K"),
        defaults,  # type: ignore[arg-type]
        (128, 128, 256),
    )


def _native_graph_fused_wavg_lora_blocks() -> tuple[int, int, int]:
    """Return ``(block_m, block_r, block_k)`` for the W/A/G/V-gate probe."""

    policy = _kernel_policy()
    defaults = tuple(getattr(policy, "wavg_lora_blocks", (64, 64, 64)))
    vals = []
    for name, fallback, default, upper in (
        ("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_M", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_M", defaults[0], 128),
        ("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_R", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_R", defaults[1], 128),
        ("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_K", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_K", defaults[2], 256),
    ):
        raw = os.environ.get(name, os.environ.get(fallback))
        if raw is None:
            vals.append(env_int(name, int(default), lower=1, upper=upper))
        else:
            try:
                val = int(str(raw).strip())
            except ValueError:
                val = int(default)
            vals.append(min(max(1, val), upper))
    return vals[0], vals[1], vals[2]


def _native_graph_fused_wavg_lora_num_warps() -> int:
    policy = _kernel_policy()
    default = int(getattr(policy, "wavg_lora_num_warps", 4))
    value = env_int("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_NUM_WARPS", default, lower=1, upper=8)
    if value not in {1, 2, 4, 8}:
        raise ValueError(
            "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_NUM_WARPS must be one of 1, 2, 4, or 8; "
            f"got {value}"
        )
    return value


def _recurrent_update_unbatched(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    H: int,
    N: int,
):
    if _native_graph_fused_recurrent_enabled():
        out, new_state = fused_recurrent_update(
            r.view(1, H, N),
            w.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            kk.view(1, H, N),
            a.view(1, H, N),
            state.view(1, H, N, N),
            block_n=N,
        )
        return out.reshape(H * N), new_state.reshape(H, N, N)
    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    new_state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = new_state.to(r.dtype) @ r.view(H, N, 1)
    return out.view(H * N), new_state


def _recurrent_update_batched(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    B: int,
    H: int,
    N: int,
):
    if _native_graph_fused_recurrent_enabled():
        out, new_state = fused_recurrent_update(
            r.view(B, H, N),
            w.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            kk.view(B, H, N),
            a.view(B, H, N),
            state,
            block_n=N,
        )
        return out.reshape(B, H * N), new_state
    vk = v.view(B, H, N, 1) @ k.view(B, H, 1, N)
    ab = (-kk).view(B, H, N, 1) @ (kk * a).view(B, H, 1, N)
    new_state = state * w.view(B, H, 1, N) + state @ ab.float() + vk.float()
    out = new_state.to(r.dtype) @ r.view(B, H, N, 1)
    return out.view(B, H * N), new_state


@torch.jit.script
def block_step(x: torch.Tensor, xpa: torch.Tensor, xpf: torch.Tensor,
               v_first: torch.Tensor, state: torch.Tensor,
               layer_id: int, H: int, N: int, eps: float, has_pre: int,
               pre_w: torch.Tensor, pre_b: torch.Tensor,
               an_w: torch.Tensor, an_b: torch.Tensor,
               fn_w: torch.Tensor, fn_b: torch.Tensor,
               x_r: torch.Tensor, x_w: torch.Tensor, x_k: torch.Tensor,
               x_v: torch.Tensor, x_a: torch.Tensor, x_g: torch.Tensor,
               k_k: torch.Tensor, k_a: torch.Tensor, r_k: torch.Tensor,
               Rw: torch.Tensor, Kw: torch.Tensor, Vw: torch.Tensor, Ow: torch.Tensor,
               w1: torch.Tensor, w2: torch.Tensor, w0: torch.Tensor,
               a1: torch.Tensor, a2: torch.Tensor, a0: torch.Tensor,
               v1: torch.Tensor, v2: torch.Tensor, v0: torch.Tensor,
               g1: torch.Tensor, g2: torch.Tensor,
               gn_w: torch.Tensor, gn_b: torch.Tensor,
               fx_k: torch.Tensor, fK: torch.Tensor, fV: torch.Tensor,
               RKVw: torch.Tensor):
    D = int(an_w.numel())
    A = H * N
    # --- block wiring (fuse_norm=False) ---
    if has_pre == 1:
        residual = F.layer_norm(x, [D], pre_w, pre_b, 1e-5)
    else:
        residual = x
    h = F.layer_norm(residual, [D], an_w, an_b, 1e-5)

    # --- TMix_one ---
    xx = xpa - h
    xpa = h
    xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
    xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    r = F.linear(xr, Rw)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
    k = F.linear(xk, Kw)
    v = F.linear(xv, Vw)
    a = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(A)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
    w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(h.dtype) @ r.view(H, N, 1)
    out = out.view(A)
    out = F.group_norm(out.view(1, A), H, gn_w, gn_b, eps).view(A)
    sk = (r.view(H, N) * k.view(H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(H, N)).view(A)
    out = F.linear(out * g, Ow)
    x = residual + out

    # --- CMix_one ---
    residual = x
    h2 = F.layer_norm(x, [D], fn_w, fn_b, 1e-5)
    fxx = xpf - h2
    xpf = h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    x = residual + F.linear(fk, fV)
    return x, xpa, xpf, v_first, state


@torch.jit.script
def block_step_batched(x: torch.Tensor, xpa: torch.Tensor, xpf: torch.Tensor,
                       v_first: torch.Tensor, state: torch.Tensor,
                       layer_id: int, H: int, N: int, eps: float, has_pre: int,
                       pre_w: torch.Tensor, pre_b: torch.Tensor,
                       an_w: torch.Tensor, an_b: torch.Tensor,
                       fn_w: torch.Tensor, fn_b: torch.Tensor,
                       x_r: torch.Tensor, x_w: torch.Tensor, x_k: torch.Tensor,
                       x_v: torch.Tensor, x_a: torch.Tensor, x_g: torch.Tensor,
                       k_k: torch.Tensor, k_a: torch.Tensor, r_k: torch.Tensor,
                       Rw: torch.Tensor, Kw: torch.Tensor, Vw: torch.Tensor, Ow: torch.Tensor,
                       w1: torch.Tensor, w2: torch.Tensor, w0: torch.Tensor,
                       a1: torch.Tensor, a2: torch.Tensor, a0: torch.Tensor,
                       v1: torch.Tensor, v2: torch.Tensor, v0: torch.Tensor,
                       g1: torch.Tensor, g2: torch.Tensor,
                       gn_w: torch.Tensor, gn_b: torch.Tensor,
               fx_k: torch.Tensor, fK: torch.Tensor, fV: torch.Tensor,
               RKVw: torch.Tensor):
    # Batched variant of block_step. Shapes:
    # x/xpa/xpf:[B,D], v_first:[B,A], state:[B,H,N,N].
    B = x.shape[0]
    D = int(an_w.numel())
    A = H * N
    if has_pre == 1:
        residual = F.layer_norm(x, [D], pre_w, pre_b, 1e-5)
    else:
        residual = x
    h = F.layer_norm(residual, [D], an_w, an_b, 1e-5)

    xx = xpa - h
    xpa = h
    xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
    xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    r = F.linear(xr, Rw)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
    k = F.linear(xk, Kw)
    v = F.linear(xv, Vw)
    a = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, A)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
    w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
    vk = v.view(B, H, N, 1) @ k.view(B, H, 1, N)
    ab = (-kk).view(B, H, N, 1) @ (kk * a).view(B, H, 1, N)
    state = state * w.view(B, H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(h.dtype) @ r.view(B, H, N, 1)
    out = out.view(B, A)
    out = F.group_norm(out, H, gn_w, gn_b, eps).view(B, A)
    sk = (r.view(B, H, N) * k.view(B, H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(B, H, N)).view(B, A)
    out = F.linear(out * g, Ow)
    x = residual + out

    residual = x
    h2 = F.layer_norm(x, [D], fn_w, fn_b, 1e-5)
    fxx = xpf - h2
    xpf = h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    x = residual + F.linear(fk, fV)
    return x, xpa, xpf, v_first, state


def extract(model):
    layers = model.model.layers
    H = layers[0].attn.num_heads
    N = layers[0].attn.head_dim
    eps = float(N * 1e-5)
    packs = []
    hidden = int(layers[0].attn.hidden_size)
    attention_hidden = int(getattr(layers[0].attn, "attention_hidden_size", H * N))
    dense_ref = model.model.embeddings.weight
    stack_rkv = _native_graph_rkv_policy() == "vkwr_auto"
    for i, layer in enumerate(layers):
        a = layer.attn
        ref = a.w_lora.lora[0].weight
        vl = getattr(a, "v_lora", None)
        v1 = vl.lora[0].weight if vl is not None else torch.zeros(1, ref.shape[1], device=ref.device, dtype=ref.dtype)
        v2 = vl.lora[2].weight if vl is not None else torch.zeros(attention_hidden, 1, device=ref.device, dtype=ref.dtype)
        v0 = vl.lora[2].bias if vl is not None else torch.zeros(attention_hidden, device=ref.device, dtype=ref.dtype)
        if hasattr(layer, "pre_norm"):
            pre_w, pre_b, has_pre = layer.pre_norm.weight, layer.pre_norm.bias, 1
        else:
            pre_w = torch.zeros(hidden, device=ref.device, dtype=ref.dtype)
            pre_b = torch.zeros(hidden, device=ref.device, dtype=ref.dtype)
            has_pre = 0
        packs.append((
            i, H, N, eps, has_pre,
            pre_w, pre_b, layer.attn_norm.weight, layer.attn_norm.bias,
            layer.ffn_norm.weight, layer.ffn_norm.bias,
            a.x_r.reshape(-1), a.x_w.reshape(-1), a.x_k.reshape(-1),
            a.x_v.reshape(-1), a.x_a.reshape(-1), a.x_g.reshape(-1),
            a.k_k, a.k_a, a.r_k,
            a.r_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight,
            a.w_lora.lora[0].weight, a.w_lora.lora[2].weight, a.w_lora.lora[2].bias,
            a.a_lora.lora[0].weight, a.a_lora.lora[2].weight, a.a_lora.lora[2].bias,
            v1, v2, v0,
            a.g_lora.lora[0].weight, a.g_lora.lora[2].weight,
            a.g_norm.weight, a.g_norm.bias,
            layer.ffn.x_k, layer.ffn.key.weight, layer.ffn.value.weight,
            torch.stack((a.r_proj.weight.t(), a.k_proj.weight.t(), a.v_proj.weight.t())).contiguous()
            if stack_rkv
            else dense_ref.new_empty((0,)),
        ))
    return packs, H, N, eps


def extract_graph(model):
    """Pack CUDA-graph operands while preserving MM8/MM4 modules.

    Dense models keep the exact historical tensor tuple. Quantized projection
    modules are retained as callable operands and are consumed by the eager
    graph-capture dispatchers below. This function is intentionally separate
    from :func:`extract`: TorchScript decode still requires tensor-only packs.
    """

    layers = model.model.layers
    H = layers[0].attn.num_heads
    N = layers[0].attn.head_dim
    eps = float(N * 1e-5)
    packs = []
    hidden = int(layers[0].attn.hidden_size)
    attention_hidden = int(getattr(layers[0].attn, "attention_hidden_size", H * N))
    stack_rkv = _native_graph_rkv_policy() == "vkwr_auto"
    embed_ref = model.model.embeddings.weight
    for i, layer in enumerate(layers):
        if _native_graph_sparse_ffn_low_memory_pack_enabled():
            if model.training or torch.is_grad_enabled():
                raise RuntimeError(
                    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_LOW_MEMORY_PACK is inference-only"
                )
            value_weight = layer.ffn.value.weight
            if value_weight.device.type != "cuda" or value_weight.dtype != torch.float16:
                raise RuntimeError(
                    "low-memory sparse FFN packing requires CUDA fp16 value weights"
                )
            _native_graph_relayout_ffn_value_weight(layer.ffn.value)
        a = layer.attn
        vl = getattr(a, "v_lora", None)
        if vl is not None:
            v1 = _graph_linear_operand(vl.lora[0])
            v2 = _graph_linear_operand(vl.lora[2])
            v0 = vl.lora[2].bias
        else:
            v1 = torch.zeros(1, hidden, device=embed_ref.device, dtype=embed_ref.dtype)
            v2 = torch.zeros(attention_hidden, 1, device=embed_ref.device, dtype=embed_ref.dtype)
            v0 = torch.zeros(attention_hidden, device=embed_ref.device, dtype=embed_ref.dtype)
        if hasattr(layer, "pre_norm"):
            pre_w, pre_b, has_pre = layer.pre_norm.weight, layer.pre_norm.bias, 1
        else:
            pre_w = torch.zeros(hidden, device=embed_ref.device, dtype=embed_ref.dtype)
            pre_b = torch.zeros(hidden, device=embed_ref.device, dtype=embed_ref.dtype)
            has_pre = 0

        r_op = _graph_linear_operand(a.r_proj)
        k_op = _graph_linear_operand(a.k_proj)
        v_op = _graph_linear_operand(a.v_proj)
        if stack_rkv and all(_graph_linear_is_dense(item) for item in (r_op, k_op, v_op)):
            stacked_rkv = torch.stack((r_op.t(), k_op.t(), v_op.t())).contiguous()
        else:
            stacked_rkv = embed_ref.new_empty((0,))

        packs.append((
            i, H, N, eps, has_pre,
            pre_w, pre_b, layer.attn_norm.weight, layer.attn_norm.bias,
            layer.ffn_norm.weight, layer.ffn_norm.bias,
            a.x_r.reshape(-1), a.x_w.reshape(-1), a.x_k.reshape(-1),
            a.x_v.reshape(-1), a.x_a.reshape(-1), a.x_g.reshape(-1),
            a.k_k, a.k_a, a.r_k,
            r_op, k_op, v_op, _graph_linear_operand(a.o_proj),
            _graph_linear_operand(a.w_lora.lora[0]),
            _graph_linear_operand(a.w_lora.lora[2]),
            a.w_lora.lora[2].bias,
            _graph_linear_operand(a.a_lora.lora[0]),
            _graph_linear_operand(a.a_lora.lora[2]),
            a.a_lora.lora[2].bias,
            v1, v2, v0,
            _graph_linear_operand(a.g_lora.lora[0]),
            _graph_linear_operand(a.g_lora.lora[2]),
            a.g_norm.weight, a.g_norm.bias,
            layer.ffn.x_k,
            _graph_linear_operand(layer.ffn.key),
            _graph_linear_operand(layer.ffn.value),
            stacked_rkv,
        ))
    return packs, H, N, eps


def _init(model, device, dtype):
    layers = model.model.layers
    n = len(layers)
    H = layers[0].attn.num_heads
    N = layers[0].attn.head_dim
    hid = layers[0].attn.hidden_size
    attention_hidden = getattr(layers[0].attn, "attention_hidden_size", H * N)
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(n)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    v_first = torch.zeros(attention_hidden, device=device, dtype=dtype)
    return state, xpa, xpf, v_first


def _init_batched_from_packs(
    packs,
    batch_size: int,
    device,
    dtype,
    *,
    state_dtype=None,
):
    n = len(packs)
    H = int(packs[0][1])
    N = int(packs[0][2])
    hid = int(packs[0][7].numel())
    if state_dtype is None:
        state_dtype = torch.float32
    state = [torch.zeros(batch_size, H, N, N, device=device, dtype=state_dtype) for _ in range(n)]
    xpa = [torch.zeros(batch_size, hid, device=device, dtype=dtype) for _ in range(n)]
    xpf = [torch.zeros(batch_size, hid, device=device, dtype=dtype) for _ in range(n)]
    return state, xpa, xpf


def step(model, x, state, xpa, xpf, v_first, packs):
    for p in packs:
        x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]] = block_step(x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]], *p)
    return x, state, xpa, xpf, v_first


def step_batched(model, x, state, xpa, xpf, v_first, packs):
    """Batched TorchScript block-step decode for native_model caches.

    Shapes mirror ``rwkv7_hf.native._step_token_batched``: x/xpa/xpf are
    ``[B,D]``, v_first is ``[B,A]``, and recurrent state is
    ``[B,H,N,N]`` per layer.
    Keeping this helper in native_jit lets the experimental FLA-free model use
    the same reduced-dispatch H2 decode idea without importing the wrapper.
    """
    for p in packs:
        x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]] = block_step_batched(
            x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]], *p
        )
    return x, state, xpa, xpf, v_first


def _native_prefill_scan(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    B: int,
    T: int,
    H: int,
    N: int,
    *,
    w_is_raw: bool = False,
    w_is_log: bool = False,
    use_self_chunk: bool | None = None,
    num_layers: int | None = None,
):
    """Run the recurrent prefill scan, using Triton only when explicitly enabled."""

    if w_is_raw and _native_prefill_fused_clampw_scan_enabled(B, T, H * N, num_layers):
        scan_block_m = _native_prefill_scan_block_m(N, B, T, H * N)
        out, new_state = fused_recurrent_scan_clampw(
            r.view(B, T, H, N),
            w.view(B, T, H, N),
            k.view(B, T, H, N),
            v.view(B, T, H, N),
            kk.view(B, T, H, N),
            a.view(B, T, H, N),
            state,
            block_n=N,
            block_m=scan_block_m,
            num_warps=_native_prefill_scan_num_warps(N, scan_block_m),
        )
        return out.reshape(B, T, H * N), new_state

    if w_is_raw:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))

    if use_self_chunk is None:
        use_self_chunk = _native_prefill_self_chunk_enabled(T, N)
    if use_self_chunk:
        chunk_size = _native_prefill_self_chunk_size(B, T)
        if T % chunk_size:
            chunk_size = 16
        out, new_state = self_chunk_rwkv7(
            r.view(B, T, H, N),
            w.view(B, T, H, N),
            k.view(B, T, H, N),
            v.view(B, T, H, N),
            kk.view(B, T, H, N),
            a.view(B, T, H, N),
            state,
            chunk_size=chunk_size,
            w_is_log=w_is_log,
            safe_gate=_native_prefill_self_chunk_safe_gate(),
            h_tiles=_native_prefill_self_chunk_h_tiles(B, T),
        )
        return out.reshape(B, T, H * N), new_state

    if w_is_log:
        w = torch.exp(w.float())

    if _native_prefill_fused_scan_enabled():
        scan_block_m = _native_prefill_scan_block_m(N, B, T, H * N)
        out, new_state = fused_recurrent_scan(
            r.view(B, T, H, N),
            w.view(B, T, H, N),
            k.view(B, T, H, N),
            v.view(B, T, H, N),
            kk.view(B, T, H, N),
            a.view(B, T, H, N),
            state,
            block_n=N,
            block_m=scan_block_m,
            num_warps=_native_prefill_scan_num_warps(N, scan_block_m),
        )
        return out.reshape(B, T, H * N), new_state

    if _native_prefill_dplr_scan_enabled() and T > 1:
        out, new_state = dplr_chunk_scan(
            r.view(B, T, H, N),
            w.view(B, T, H, N),
            k.view(B, T, H, N),
            v.view(B, T, H, N),
            kk.view(B, T, H, N),
            a.view(B, T, H, N),
            state,
            chunk_size=_native_prefill_dplr_chunk_size(),
        )
        return out.reshape(B, T, H * N), new_state

    cur_state = state
    outs = []
    for t in range(T):
        out, cur_state = _recurrent_update_batched(
            r[:, t],
            w[:, t],
            k[:, t],
            v[:, t],
            kk[:, t],
            a[:, t],
            cur_state,
            B,
            H,
            N,
        )
        outs.append(out)
    return torch.stack(outs, dim=1), cur_state


def prefill(
    model,
    ids,
    packs,
    *,
    state=None,
    xpa=None,
    xpf=None,
    logits_to_keep: int | None = 1,
    fp16_elapsed=None,
):
    """Layer-wise native RWKV-7 prefill over a full prompt.

    This is the first production-facing bridge for the fused recurrent scan
    prototype: it computes every layer over `[batch, tokens]` using vectorized
    projections and an optional fused recurrent scan instead of repeatedly
    calling the one-token decode path.  Returned state uses the native layout
    `[B,H,N,N]`; callers that expose HF/FLA cache state should transpose the
    final two dimensions, matching the native-graph decode runner.
    """

    base = model.model
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2:
        raise ValueError("native_jit.prefill expects ids shaped [batch, tokens]")
    B = int(ids.shape[0])
    T = int(ids.shape[1])
    if T <= 0:
        raise ValueError("native_jit.prefill requires at least one token")
    H = int(packs[0][1])
    N = int(packs[0][2])
    attention_hidden = H * N
    residual_hidden = int(packs[0][7].numel())
    dtype = base.embeddings.weight.dtype
    use_fp16_recurrent_requested = bool(
        _native_prefill_fp16_recurrent_requested()
        and native_fp16_sequence is not None
        and dtype == torch.float16
        and N == 64
    )
    state_dtype = torch.float16 if use_fp16_recurrent_requested else torch.float32
    if state is None or xpa is None or xpf is None:
        state, xpa, xpf = _init_batched_from_packs(
            packs,
            B,
            ids.device,
            dtype,
            state_dtype=state_dtype,
        )
    else:
        state = [s.to(device=ids.device, dtype=state_dtype).contiguous() for s in state]
        xpa = [s.to(device=ids.device, dtype=dtype).contiguous() for s in xpa]
        xpf = [s.to(device=ids.device, dtype=dtype).contiguous() for s in xpf]
    if use_fp16_recurrent_requested:
        if fp16_elapsed is None:
            fp16_elapsed = torch.zeros(B, device=ids.device, dtype=torch.int32)
        elif not (
            fp16_elapsed.is_cuda
            and fp16_elapsed.device == ids.device
            and fp16_elapsed.dtype == torch.int32
            and fp16_elapsed.is_contiguous()
            and int(fp16_elapsed.numel()) == B
        ):
            raise ValueError("fp16_elapsed must be contiguous CUDA int32 [batch]")

    x = F.embedding(ids, base.embeddings.weight).reshape(B, T, residual_hidden)
    v_first_seq = torch.zeros(B, T, attention_hidden, device=ids.device, dtype=dtype)
    use_clampw_scan_requested = not use_fp16_recurrent_requested and _native_prefill_fused_clampw_scan_enabled(
        B,
        T,
        attention_hidden,
        len(packs),
    )
    clampw_scan_used = False
    use_prefill_sequence_ffn = _native_prefill_fused_sequence_ffn_enabled(
        B * T,
        B,
        T,
        residual_hidden,
        len(packs),
    )
    sequence_ffn_blocks = _native_prefill_sequence_ffn_blocks(B * T) if use_prefill_sequence_ffn else None
    sequence_ffn_launch = _native_prefill_sequence_ffn_launch() if use_prefill_sequence_ffn else None
    sequence_ffn_workspace = None
    sequence_attn_mix_workspace = None
    bnb8_attn_mix_workspace = None
    bnb8_attn_quant_workspace = None
    bnb8_attn_scale_workspace = None
    sequence_ffn_mix_workspace = None
    bnb8_ffn_quant_workspace = None
    bnb8_ffn_scale_workspace = None
    self_chunk_used = False
    sequence_ffn_used = False
    use_fp16_accum_ffn_key = _native_prefill_fp16_accum_ffn_key_enabled(
        B,
        T,
        residual_hidden,
        len(packs),
        dtype,
    )
    fp16_accum_ffn_key_layers = (
        _native_prefill_fp16_accum_ffn_key_layers(
            B,
            T,
            residual_hidden,
            len(packs),
        )
        if use_fp16_accum_ffn_key
        else set()
    )
    fp16_accum_ffn_key_used = False
    use_prefill_shift_mix = _native_prefill_fused_shift_mix_enabled(
        B, T, residual_hidden, len(packs)
    )
    use_prefill_state_prep = _native_prefill_fused_state_prep_enabled(
        B, T, residual_hidden, len(packs)
    )
    prefill_state_prep_layers = (
        _native_prefill_state_prep_layers(B, T, residual_hidden, len(packs))
        if use_prefill_state_prep
        else set()
    )
    capture_layer_outputs = env_flag(
        "RWKV7_NATIVE_PREFILL_CAPTURE_LAYER_OUTPUTS",
        False,
    )
    layer_outputs = [] if capture_layer_outputs else None
    stacked_rkv_weights = (
        _native_prefill_stacked_rkv_weights(model, packs)
        if _native_prefill_stacked_rkv_enabled(B * T, B, T, residual_hidden, len(packs))
        else None
    )
    stacked_rkv_used = False
    wavg_lora_used = False

    for p in packs:
        (i, H, N, eps, has_pre,
         pre_w, pre_b, an_w, an_b, fn_w, fn_b,
         x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
         Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
         gn_w, gn_b, fx_k, fK, fV, _RKVw) = p
        layer_idx = int(i)
        H = int(H)
        N = int(N)
        attention_hidden = H * N
        residual_hidden = int(an_w.numel())
        use_fp16_recurrent = _native_prefill_fp16_recurrent_enabled(state[layer_idx])
        use_layer_state_prep = bool(
            use_prefill_state_prep
            and (
                prefill_state_prep_layers is None
                or layer_idx in prefill_state_prep_layers
            )
        )
        residual = F.layer_norm(x, [residual_hidden], pre_w, pre_b, 1e-5) if int(has_pre) == 1 else x
        h = F.layer_norm(residual, [residual_hidden], an_w, an_b, 1e-5)
        defer_state_sigmoid = bool(
            not use_fp16_recurrent
            and use_layer_state_prep
            and not _native_prefill_fused_state_scan_enabled(B)
            and not use_clampw_scan_requested
        )
        state_sigmoid_is_raw = False
        use_sequence_attn_mix = use_prefill_shift_mix and fused_attn_sequence_shift_mix is not None
        v_gate = None
        prequantized_rkv = None
        use_bnb8_rkv_mix = bool(
            use_sequence_attn_mix and _bnb8_rkv_mix_quant_enabled(Rw, Kw, Vw)
        )
        if use_bnb8_rkv_mix:
            (
                qr, sr, qk, sk, qv, sv,
                xw, xv, xa, xg, next_xpa,
                bnb8_attn_mix_workspace,
                bnb8_attn_quant_workspace,
                bnb8_attn_scale_workspace,
            ) = fused_bnb8_attn_sequence_mix_quant(
                h,
                xpa[layer_idx],
                x_r,
                x_w,
                x_k,
                x_v,
                x_a,
                x_g,
                mix_workspace=bnb8_attn_mix_workspace,
                quant_workspace=bnb8_attn_quant_workspace,
                scale_workspace=bnb8_attn_scale_workspace,
                block=_native_bnb8_policy_block(
                    "RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK",
                    "native_bnb8_attn_mix_block",
                    1024,
                ),
            )
            prequantized_rkv = (qr, sr, qk, sk, qv, sv)
            xr = xk = None
        elif use_sequence_attn_mix:
            if sequence_attn_mix_workspace is None:
                sequence_attn_mix_workspace = torch.empty(
                    (6, B, T, residual_hidden), device=h.device, dtype=h.dtype
                )
            xr, xw, xk, xv, xa, xg, next_xpa = fused_attn_sequence_shift_mix(
                h,
                xpa[layer_idx],
                x_r,
                x_w,
                x_k,
                x_v,
                x_a,
                x_g,
                workspace=sequence_attn_mix_workspace,
            )
        else:
            prev_h = torch.cat([xpa[layer_idx].view(B, 1, residual_hidden), h[:, :-1, :]], dim=1)
            xx = prev_h - h
            xr = h + xx * x_r.view(1, 1, residual_hidden)
            xw = h + xx * x_w.view(1, 1, residual_hidden)
            xk = h + xx * x_k.view(1, 1, residual_hidden)
            xv = h + xx * x_v.view(1, 1, residual_hidden)
            xa = h + xx * x_a.view(1, 1, residual_hidden)
            xg = h + xx * x_g.view(1, 1, residual_hidden)

        use_stacked_rkv = False
        if prequantized_rkv is not None:
            qr, sr, qk, sk, qv, sv = prequantized_rkv
            r = _bnb8_prequant_linear(qr, sr, Rw, dtype=h.dtype, output_shape=(B, T))
            k = _bnb8_prequant_linear(qk, sk, Kw, dtype=h.dtype, output_shape=(B, T))
            v = _bnb8_prequant_linear(qv, sv, Vw, dtype=h.dtype, output_shape=(B, T))
        elif stacked_rkv_weights:
            row_values = B * T * residual_hidden
            use_stacked_rkv = bool(
                xr.is_contiguous()
                and xk.is_contiguous()
                and xv.is_contiguous()
                and xr.untyped_storage().data_ptr() == xk.untyped_storage().data_ptr()
                and xr.untyped_storage().data_ptr() == xv.untyped_storage().data_ptr()
                and int(xk.storage_offset()) == int(xr.storage_offset()) + row_values
                and int(xv.storage_offset()) == int(xr.storage_offset()) + 2 * row_values
            )
        if prequantized_rkv is not None:
            pass
        elif use_stacked_rkv:
            stacked_rkv_used = True
            rkv_inputs = xr.as_strided(
                (3, B * T, residual_hidden),
                (B * T * residual_hidden, residual_hidden, 1),
            )
            rkv = torch.bmm(rkv_inputs, stacked_rkv_weights[layer_idx])
            r = rkv[0].view(B, T, attention_hidden)
            k = rkv[1].view(B, T, attention_hidden)
            v = rkv[2].view(B, T, attention_hidden)
        else:
            r = _native_prefill_linear(xr, Rw)
            k = _native_prefill_linear(xk, Kw)
            v = _native_prefill_linear(xv, Vw)
        use_prefill_wavg_lora = bool(
            (not use_fp16_recurrent or T > 16)
            and layer_idx > 0
            and _native_prefill_fused_wavg_lora_enabled(B * T)
            and _graph_linears_are_dense(w1, w2, a1, a2, g1, g2, v1, v2)
        )
        if use_prefill_wavg_lora:
            wavg_lora_used = True
            block_m, block_r, block_k = _native_prefill_fused_wavg_lora_blocks()
            w, a, g, v_gate = fused_wavg_lora(
                xw.reshape(B * T, residual_hidden),
                xa.reshape(B * T, residual_hidden),
                xg.reshape(B * T, residual_hidden),
                xv.reshape(B * T, residual_hidden),
                w1,
                a1,
                g1,
                v1,
                w2,
                a2,
                g2,
                v2,
                w0,
                a0,
                None,
                v0,
                block_m=block_m,
                block_r=block_r,
                block_k=block_k,
            )
            w = w.view(B, T, attention_hidden)
            a = torch.sigmoid(a.view(B, T, attention_hidden))
            g = g.view(B, T, attention_hidden)
            v_gate = v_gate.view(B, T, attention_hidden)
        else:
            w_mid = _native_prefill_linear(xw, w1)
            w_mid.tanh_()
            if use_fp16_recurrent and T <= 16:
                w = _native_prefill_linear(w_mid, w2)
                fp16_w0 = w0.reshape(-1).contiguous()
            else:
                w = _native_prefill_linear(w_mid, w2, w0)
                fp16_w0 = None
            a_mid = _native_prefill_linear(xa, a1)
            a = _native_prefill_linear(a_mid, a2, a0)
            if not defer_state_sigmoid:
                a.sigmoid_()
            else:
                state_sigmoid_is_raw = True
            g_mid = _native_prefill_linear(xg, g1)
            g_mid.sigmoid_()
            g = _native_prefill_linear(g_mid, g2)
            if layer_idx != 0:
                v_mid = _native_prefill_linear(xv, v1)
                v_gate = _native_prefill_linear(v_mid, v2, v0)
                if not defer_state_sigmoid:
                    v_gate.sigmoid_()
        use_fused_scan_output = bool(
            not use_fp16_recurrent and _native_prefill_fused_scan_output_enabled()
        )
        use_self_chunk = _native_prefill_self_chunk_enabled(
            T,
            N,
            B,
            attention_hidden,
            len(packs),
        ) and not use_fused_scan_output and not use_fp16_recurrent
        self_chunk_used = bool(self_chunk_used or use_self_chunk)
        self_chunk_w_is_log = False
        use_clampw_scan = use_clampw_scan_requested and not use_fused_scan_output
        use_fused_state_scan = bool(
            not use_fp16_recurrent
            and _native_prefill_fused_state_scan_enabled(B)
            and not use_fused_scan_output
        )
        if use_clampw_scan and use_layer_state_prep and fused_prefill_kv_kk_prep is None:
            use_clampw_scan = False
        state_scan_done = False
        if use_fused_state_scan:
            scan_block_m = _native_prefill_scan_block_m(N, B, T, H * N)
            scan_num_warps = _native_prefill_scan_num_warps(N, scan_block_m)
            if layer_idx == 0:
                out, new_state, k, v = fused_recurrent_scan_state_prep(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    block_n=N,
                    block_m=scan_block_m,
                    num_warps=scan_num_warps,
                )
                v_first_seq = v.reshape(B, T, attention_hidden)
            else:
                out, new_state, k, v = fused_recurrent_scan_state_prep(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    v_first=v_first_seq.view(B, T, H, N),
                    v_gate=v_gate.view(B, T, H, N),
                    block_n=N,
                    block_m=scan_block_m,
                    num_warps=scan_num_warps,
                )
            out = out.reshape(B, T, attention_hidden)
            k = k.reshape(B, T, attention_hidden)
            v = v.reshape(B, T, attention_hidden)
            state_scan_done = True
        elif use_layer_state_prep and use_fp16_recurrent:
            if fused_prefill_kv_kk_prep is None:
                raise RuntimeError(
                    "RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP requires the fused K/V/KK prep kernel"
                )
            if layer_idx == 0:
                k, v, kk = fused_prefill_kv_kk_prep(
                    k,
                    v,
                    a,
                    k_k,
                    k_a,
                    num_heads=H,
                    head_dim=N,
                )
                v_first_seq = v
            else:
                k, v, kk = fused_prefill_kv_kk_prep(
                    k,
                    v,
                    a,
                    k_k,
                    k_a,
                    v_first=v_first_seq,
                    v_gate=v_gate,
                    num_heads=H,
                    head_dim=N,
                )
        elif use_layer_state_prep and not use_fp16_recurrent:
            self_chunk_w_is_log = bool(use_self_chunk and not use_clampw_scan)
            if use_clampw_scan:
                if layer_idx == 0:
                    k, v, kk = fused_prefill_kv_kk_prep(
                        k,
                        v,
                        a,
                        k_k,
                        k_a,
                        num_heads=H,
                        head_dim=N,
                    )
                    v_first_seq = v
                else:
                    k, v, kk = fused_prefill_kv_kk_prep(
                        k,
                        v,
                        a,
                        k_k,
                        k_a,
                        v_first=v_first_seq,
                        v_gate=v_gate,
                        num_heads=H,
                        head_dim=N,
                    )
            elif layer_idx == 0:
                w, k, v, kk = fused_prefill_state_prep(
                    w,
                    k,
                    v,
                    a,
                    k_k,
                    k_a,
                    num_heads=H,
                    head_dim=N,
                    w_out_dtype=_native_prefill_state_prep_w_dtype(),
                    w_transform="log_decay" if use_self_chunk else "decay",
                    a_is_raw=state_sigmoid_is_raw,
                    v_gate_is_raw=state_sigmoid_is_raw,
                )
                v_first_seq = v
            else:
                w, k, v, kk = fused_prefill_state_prep(
                    w,
                    k,
                    v,
                    a,
                    k_k,
                    k_a,
                    v_first=v_first_seq,
                    v_gate=v_gate,
                    num_heads=H,
                    head_dim=N,
                    w_out_dtype=_native_prefill_state_prep_w_dtype(),
                    w_transform="log_decay" if use_self_chunk else "decay",
                    a_is_raw=state_sigmoid_is_raw,
                    v_gate_is_raw=state_sigmoid_is_raw,
                )
        else:
            kk = F.normalize(
                (k * k_k.view(1, 1, attention_hidden)).view(B, T, H, N),
                dim=-1,
                p=2.0,
            ).view(B, T, attention_hidden)
            k = k * (1 + (a - 1) * k_a.view(1, 1, attention_hidden))
            if layer_idx == 0:
                v_first_seq = v
            else:
                v = v + (v_first_seq - v) * v_gate
            if not use_clampw_scan and not use_fp16_recurrent:
                w = torch.exp(-0.606531 * torch.sigmoid(w.float()))

        if use_fp16_recurrent:
            assert fp16_elapsed is not None
            out = native_fp16_sequence(
                r.view(B, T, H, N).contiguous(),
                w.view(B, T, H, N).contiguous(),
                k.view(B, T, H, N).contiguous(),
                v.view(B, T, H, N).contiguous(),
                (-kk).view(B, T, H, N).contiguous(),
                (kk * a).view(B, T, H, N).contiguous(),
                state[layer_idx],
                fp16_elapsed,
                w0=fp16_w0,
            ).reshape(B, T, attention_hidden)
            new_state = state[layer_idx]
            state_scan_done = True
        elif use_fused_scan_output:
            out, new_state = fused_recurrent_scan_output_prepare(
                r.view(B, T, H, N),
                w.view(B, T, H, N),
                k.view(B, T, H, N),
                v.view(B, T, H, N),
                kk.view(B, T, H, N),
                a.view(B, T, H, N),
                state[layer_idx],
                g.view(B, T, H, N),
                r_k,
                gn_w,
                gn_b,
                eps=eps,
                block_n=N,
            )
            out = out.reshape(B, T, attention_hidden)
        elif not state_scan_done:
            clampw_scan_used = bool(clampw_scan_used or use_clampw_scan)
            out, new_state = _native_prefill_scan(
                r, w, k, v, kk, a, state[layer_idx], B, T, H, N,
                w_is_raw=use_clampw_scan,
                w_is_log=self_chunk_w_is_log,
                use_self_chunk=use_self_chunk,
                num_layers=len(packs),
            )
        out_projected = False
        if use_fused_scan_output:
            pass
        elif _native_prefill_fused_output_project_enabled() and _graph_linear_is_dense(Ow):
            out = fused_attn_output_project(
                out.reshape(B * T, attention_hidden),
                r.reshape(B * T, H, N),
                k.reshape(B * T, H, N),
                v.reshape(B * T, H, N),
                g.reshape(B * T, attention_hidden),
                r_k,
                gn_w,
                gn_b,
                Ow,
                None,
                num_heads=H,
                head_dim=N,
                head_v_dim=N,
                eps=eps,
                block_m=_native_prefill_fused_output_project_block_m(),
            ).view(B, T, residual_hidden)
            out_projected = True
        elif _native_prefill_fused_output_enabled():
            out = fused_attn_output_prepare(
                out.reshape(B * T, attention_hidden),
                r.reshape(B * T, H, N),
                k.reshape(B * T, H, N),
                v.reshape(B * T, H, N),
                g.reshape(B * T, attention_hidden),
                r_k,
                gn_w,
                gn_b,
                num_heads=H,
                head_dim=N,
                head_v_dim=N,
                eps=eps,
            ).view(B, T, attention_hidden)
        else:
            out = F.group_norm(
                out.reshape(B * T, attention_hidden), H, gn_w, gn_b, eps
            ).view(B, T, attention_hidden)
            sk = (r.view(B, T, H, N) * k.view(B, T, H, N) * r_k.view(1, 1, H, N)).sum(dim=-1, keepdim=True)
            out = (
                out + (sk * v.view(B, T, H, N)).view(B, T, attention_hidden)
            ) * g
        if not out_projected:
            x = _native_prefill_project_residual(out, Ow, residual)
        else:
            x = residual + out
        xpa[layer_idx] = (
            next_xpa
            if use_sequence_attn_mix
            else h[:, -1, :].contiguous()
        )
        state[layer_idx] = new_state.contiguous()

        residual = x
        h2 = F.layer_norm(x, [residual_hidden], fn_w, fn_b, 1e-5)
        use_layer_sequence_ffn = bool(
            use_prefill_sequence_ffn and _graph_linears_are_dense(fK, fV)
        )
        if use_layer_sequence_ffn:
            sequence_ffn_used = True
            assert sequence_ffn_blocks is not None
            assert sequence_ffn_launch is not None
            if sequence_ffn_workspace is None:
                sequence_ffn_workspace = (
                    torch.empty((B * T, residual_hidden), device=h2.device, dtype=h2.dtype),
                    torch.empty((B * T, int(fK.shape[0])), device=h2.device, dtype=h2.dtype),
                )
            ffn_out, next_xpf = fused_sequence_ffn(
                h2,
                xpf[layer_idx],
                fx_k,
                fK,
                fV,
                block_m=sequence_ffn_blocks[0],
                block_n=sequence_ffn_blocks[1],
                key_block_k=sequence_ffn_blocks[2],
                value_block_k=sequence_ffn_blocks[3],
                group_m=sequence_ffn_blocks[4],
                num_stages=sequence_ffn_launch[0],
                num_warps=sequence_ffn_launch[1],
                workspace=sequence_ffn_workspace,
            )
            x = residual + ffn_out
        else:
            ffn_up_prequantized = False
            if use_prefill_shift_mix and _bnb8_ffn_mix_quant_enabled(fK):
                (
                    qfk,
                    sfk,
                    next_xpf,
                ) = fused_bnb8_ffn_sequence_mix_quant(
                    h2,
                    xpf[layer_idx],
                    fx_k,
                    quant_workspace=bnb8_ffn_quant_workspace,
                    scale_workspace=bnb8_ffn_scale_workspace,
                    block=_native_bnb8_policy_block(
                        "RWKV7_NATIVE_BNB8_FFN_MIX_BLOCK",
                        "native_bnb8_ffn_mix_block",
                        1024,
                    ),
                )
                bnb8_ffn_quant_workspace = qfk
                bnb8_ffn_scale_workspace = sfk
                fk = _bnb8_prequant_linear(qfk, sfk, fK, dtype=h2.dtype, output_shape=(B, T))
                ffn_up_prequantized = True
            elif use_prefill_shift_mix and fused_ffn_sequence_shift_mix is not None:
                if sequence_ffn_mix_workspace is None:
                    sequence_ffn_mix_workspace = torch.empty_like(h2)
                fk, next_xpf = fused_ffn_sequence_shift_mix(
                    h2,
                    xpf[layer_idx],
                    fx_k,
                    workspace=sequence_ffn_mix_workspace,
                )
            else:
                prev_h2 = torch.cat(
                    [xpf[layer_idx].view(B, 1, residual_hidden), h2[:, :-1, :]],
                    dim=1,
                )
                fxx = prev_h2 - h2
                fk = h2 + fxx * fx_k.view(1, 1, residual_hidden)
                next_xpf = h2[:, -1, :].contiguous()
            fused_up_relu2 = False
            if not ffn_up_prequantized:
                fused = getattr(fK, "rwkv7_forward_relu2", None)
                fused_up_relu2 = bool(
                    getattr(fK, "fused_relu2", False) and callable(fused)
                )
                if fused_up_relu2:
                    fk = fused(fk)
                else:
                    fp16_accum_ffn_key_layer = bool(
                        use_fp16_accum_ffn_key and _graph_linear_is_dense(fK)
                        and (
                            fp16_accum_ffn_key_layers is None
                            or layer_idx in fp16_accum_ffn_key_layers
                        )
                    )
                    fp16_accum_ffn_key_used = bool(
                        fp16_accum_ffn_key_used or fp16_accum_ffn_key_layer
                    )
                    fk = _native_prefill_linear(
                        fk,
                        fK,
                        allow_fp16_accumulation=fp16_accum_ffn_key_layer,
                    )
            fused_bnb8_ffn = (
                None
                if fused_up_relu2
                else _bnb8_direct_relu_square_linear(fk, fV)
            )
            if fused_bnb8_ffn is not None:
                x = residual + fused_bnb8_ffn
            elif fused_up_relu2:
                x = _native_prefill_project_residual(fk, fV, residual)
            elif (
                use_prefill_shift_mix
                and fused_relu_square is not None
                and fused_relu_square_available is not None
                and fused_relu_square_available()
            ):
                fk = fused_relu_square(fk)
                x = _native_prefill_project_residual(fk, fV, residual)
            else:
                fk = torch.relu(fk) ** 2
                x = _native_prefill_project_residual(fk, fV, residual)
        xpf[layer_idx] = next_xpf
        if layer_outputs is not None:
            layer_outputs.append(x[:, -1, :].detach().clone())

    keep = T if logits_to_keep is None or int(logits_to_keep) <= 0 else min(int(logits_to_keep), T)
    # Recurrent/shift state is already complete. Final norm is consumed only
    # by the language head, so serving requests that ask for the last logits
    # must not normalize and materialize the entire prompt sequence.
    x_for_logits = x if keep == T else x[:, -keep:, :]
    x_for_logits = F.layer_norm(
        x_for_logits,
        [residual_hidden],
        base.norm.weight,
        base.norm.bias,
        1e-5,
    )
    logits = _lm_head(model, x_for_logits)
    setattr(
        model,
        "_rwkv7_native_prefill_clampw_scan_effective",
        bool(clampw_scan_used),
    )
    setattr(model, "_rwkv7_native_prefill_stacked_rkv_effective", bool(stacked_rkv_used))
    setattr(model, "_rwkv7_native_prefill_wavg_lora_effective", bool(wavg_lora_used))
    setattr(model, "_rwkv7_native_prefill_self_chunk_effective", bool(self_chunk_used))
    setattr(model, "_rwkv7_native_prefill_sequence_ffn_effective", bool(sequence_ffn_used))
    setattr(
        model,
        "_rwkv7_native_prefill_fp16_accum_ffn_key_effective",
        bool(fp16_accum_ffn_key_used),
    )
    setattr(
        model,
        "_rwkv7_native_prefill_fp16_recurrent_effective",
        bool(use_fp16_recurrent_requested),
    )
    setattr(model, "_rwkv7_native_prefill_layer_outputs", layer_outputs)
    return logits, state, xpa, xpf


def forward(model, ids, packs):
    base = model.model
    H, N = packs[0][1], packs[0][2]
    state, xpa, xpf, v_first = _init(model, ids.device, base.embeddings.weight.dtype)
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], base.embeddings.weight).reshape(-1)
        x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
    x = F.layer_norm(x, [H * N], base.norm.weight, base.norm.bias, 1e-5)
    return _lm_head(model, x)


def decode_speed(model, ids, packs, n=128):
    import time
    base = model.model
    H, N = packs[0][1], packs[0][2]
    state, xpa, xpf, v_first = _init(model, ids.device, base.embeddings.weight.dtype)
    emb = base.embeddings.weight
    head = model.lm_head
    norm_w = base.norm.weight
    norm_b = base.norm.bias
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
    nx = _linear_module(head, F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5)).argmax()
    with torch.no_grad():
        for _ in range(5):
            x = F.embedding(nx.reshape(1, 1), emb).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = _linear_module(head, F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5)).argmax()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            x = F.embedding(nx.reshape(1, 1), emb).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = _linear_module(head, F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5)).argmax()
        torch.cuda.synchronize(); dt = time.time() - t0
    return n / dt


def _block_ip(
    x,
    state,
    xpa,
    xpf,
    v_first,
    p,
    sparse_ffn_out=None,
    fp16_elapsed=None,
    fp16_advance_elapsed=False,
):
    """In-place (eager) block step for CUDA-graph capture: state/xpa/xpf/v_first
    are fixed buffers updated in place. Same math as block_step."""
    (i, H, N, eps, has_pre,
     pre_w, pre_b, an_w, an_b, fn_w, fn_b,
     x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
     Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
     gn_w, gn_b, fx_k, fK, fV, RKVw) = p
    D = int(an_w.numel())
    A = int(H * N)
    equal_width = D == A
    residual = F.layer_norm(x, [D], pre_w, pre_b, 1e-5) if has_pre else x
    use_fused_norm_mix = _native_graph_fused_norm_mix_enabled()
    if use_fused_norm_mix:
        stack_rkv = _native_graph_vkwr_rkv_dispatch(1, D) and RKVw.numel() != 0
        xr, xw, xk, xv, xa, xg = fused_attn_norm_mix6_decode(
            residual,
            xpa,
            an_w,
            an_b,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            num_warps=_native_graph_fused_norm_mix_num_warps(),
            stack_rkv=stack_rkv,
        )
    else:
        h = F.layer_norm(residual, [D], an_w, an_b, 1e-5)
        xx = xpa - h
        xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
        xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    v_gate = None
    v_mixed = False
    lora_dense = _graph_linears_are_dense(w1, w2, a1, a2, v1, v2, g1, g2)
    if _native_graph_fused_projection_enabled() and lora_dense and _graph_linears_are_dense(Rw, Kw, Vw):
        r, k, v, w, a, g, v_gate = fused_rkv_wavg_projection(
            xr.view(1, D),
            xk.view(1, D),
            xv.view(1, D),
            xw.view(1, D),
            xa.view(1, D),
            xg.view(1, D),
            Rw,
            Kw,
            Vw,
            w1,
            a1,
            g1,
            v1,
            w2,
            a2,
            g2,
            v2,
            w0,
            a0,
            None,
            v0,
        )
        r = r.view(A)
        k = k.view(A)
        v = v.view(A)
        w = w.view(A)
        a = torch.sigmoid(a.view(A))
        g = g.view(A)
        v_gate = torch.sigmoid(v_gate.view(A))
    elif equal_width and i > 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        1,
        D,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0], _graph_linear_shape(v1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        w, a, g, v = ada_wagv_lora(
            xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2,
            w0, a0, v0, v, v_first, sigmoid_a=True,
        )
        v_mixed = True
    elif equal_width and i == 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        1,
        D,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        w, a, g, _unused_v = ada_wagv_lora(
            xw, xa, xg, xg, w1, a1, g1, g1, w2, a2, g2, g2,
            w0, a0, a0, v, v, sigmoid_a=True, compute_v=False,
        )
    elif equal_width and lora_dense and _native_graph_ada_wag_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        w, a, g = ada_wag_lora(
            xw, xa, xg, w1, a1, g1, w2, a2, g2, w0, a0,
        )
        a = torch.sigmoid(a)
    elif equal_width and i > 0 and lora_dense and _native_graph_sm70_wagv_lora_enabled(1, D):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        w, a, g, v = sm70_wagv_lora(
            xw.view(1, D), xa.view(1, D), xg.view(1, D), xv.view(1, D),
            w1, a1, g1, v1, w2, a2, g2, v2, w0, a0, v0,
            v.view(1, A), v_first.view(1, A),
        )
        w = w.view(A); a = torch.sigmoid(a.view(A)); g = g.view(A); v = v.view(A)
        v_mixed = True
    elif lora_dense and _native_graph_fused_wavg_lora_enabled(1, D):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        if i == 0:
            w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
            a = a0 + F.linear(F.linear(xa, a1), a2)
            g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
        else:
            block_m, block_r, block_k = _native_graph_fused_wavg_lora_blocks()
            w, a, g, v_gate = fused_wavg_lora(
                xw.view(1, D),
                xa.view(1, D),
                xg.view(1, D),
                xv.view(1, D),
                w1,
                a1,
                g1,
                v1,
                w2,
                a2,
                g2,
                v2,
                w0,
                a0,
                None,
                v0,
                block_m=block_m,
                block_r=block_r,
                block_k=block_k,
                num_warps=_native_graph_fused_wavg_lora_num_warps(),
            )
            w = w.view(A)
            a = a.view(A)
            g = g.view(A)
            v_gate = v_gate.view(A)
        a = torch.sigmoid(a)
    elif lora_dense and _native_graph_fused_wag_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        block_m, block_r, block_k = _native_graph_fused_wag_lora_blocks()
        w, a, g = fused_wag_lora(
            xw.view(1, D),
            xa.view(1, D),
            xg.view(1, D),
            w1,
            a1,
            g1,
            w2,
            a2,
            g2,
            w0,
            a0,
            None,
            block_m=block_m,
            block_r=block_r,
            block_k=block_k,
        )
        w = w.view(A)
        a = torch.sigmoid(a.view(A))
        g = g.view(A)
    else:
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        w = _graph_linear_call_with_explicit_bias(torch.tanh(_graph_linear_call(xw, w1)), w2, w0)
        a = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xa, a1), a2, a0))
        g = _graph_linear_call(torch.sigmoid(_graph_linear_call(xg, g1)), g2)
    use_fp16_recurrent = _native_graph_fp16_recurrent_enabled(state, fp16_elapsed)
    use_fused_recurrent_output = (
        use_fp16_recurrent or _native_graph_fused_recurrent_output_enabled()
    )
    use_fused_recurrent_raw = use_fp16_recurrent or (
        use_fused_recurrent_output and _native_graph_fused_recurrent_raw_enabled()
    )
    if not use_fused_recurrent_raw:
        kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(A)
        k = k * (1 + (a - 1) * k_a)
    if i == 0:
        v_first.copy_(v)
    elif not v_mixed:
        if v_gate is None:
            v_gate = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xv, v1), v2, v0))
        v = v + (v_first - v) * v_gate
    new_state = None
    if use_fp16_recurrent:
        out = native_fp16_recurrent_output_prepare_raw(
            r.view(1, H, N),
            w.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            a.view(1, H, N),
            state.view(1, H, N, N),
            g.view(1, H, N),
            k_k,
            k_a,
            r_k,
            gn_w,
            gn_b,
            fp16_elapsed,
            advance_elapsed=fp16_advance_elapsed,
            eps=eps,
        ).view(A)
    elif use_fused_recurrent_raw:
        out, new_state = fused_recurrent_output_prepare_raw(
            r.view(1, H, N),
            w.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            a.view(1, H, N),
            state.view(1, H, N, N),
            g.view(1, H, N),
            k_k,
            k_a,
            r_k,
            gn_w,
            gn_b,
            eps=eps,
            block_n=N,
        )
        out = out.view(A)
        new_state = new_state.view(H, N, N)
    elif use_fused_recurrent_output:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
        out, new_state = fused_recurrent_output_prepare(
            r.view(1, H, N),
            w.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            kk.view(1, H, N),
            a.view(1, H, N),
            state.view(1, H, N, N),
            g.view(1, H, N),
            r_k,
            gn_w,
            gn_b,
            eps=eps,
            block_n=N,
        )
        out = out.view(A)
        new_state = new_state.view(H, N, N)
    else:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
        out, new_state = _recurrent_update_unbatched(r, w, k, v, kk, a, state, H, N)
    if use_fused_recurrent_output:
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    elif _native_graph_fused_output_project_enabled() and _graph_linear_is_dense(Ow):
        out = fused_attn_output_project(
            out.view(1, A),
            r.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            g.view(1, A),
            r_k,
            gn_w,
            gn_b,
            Ow,
            None,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
            block_m=_native_graph_fused_output_project_block_m(),
        ).view(D)
    elif _native_graph_fused_output_enabled():
        out = fused_attn_output_prepare(
            out.view(1, A),
            r.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            g.view(1, A),
            r_k,
            gn_w,
            gn_b,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
        ).view(A)
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    else:
        out = F.group_norm(out.view(1, A), H, gn_w, gn_b, eps).view(A)
        sk = (r.view(H, N) * k.view(H, N) * r_k).sum(dim=-1, keepdim=True)
        out = (out + (sk * v.view(H, N)).view(A)) * g
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    if new_state is not None:
        state.copy_(new_state)
    if use_fused_norm_mix:
        if _native_graph_blackwell_norm_mix_enabled(
            residual, out, xpf, layer_index=int(i)
        ):
            residual, fk = blackwell_ffn_add_norm_mix(
                residual, out, xpf, fn_w, fn_b, fx_k, eps=1.0e-5
            )
        else:
            residual, fk = fused_ffn_add_norm_mix_decode(
                residual,
                out,
                xpf,
                fn_w,
                fn_b,
                fx_k,
                num_warps=_native_graph_fused_norm_mix_num_warps(),
            )
    else:
        xpa.copy_(h)
        residual = residual + out
        h2 = F.layer_norm(residual, [D], fn_w, fn_b, 1e-5)
        fxx = xpf - h2
        fk = h2 + fxx * fx_k
        xpf.copy_(h2)
    return _native_graph_ffn_dispatch(fk, fK, fV, residual, sparse_out=sparse_ffn_out)


def _block_ip_batched(
    x,
    state,
    xpa,
    xpf,
    v_first,
    p,
    sparse_ffn_out=None,
    fp16_elapsed=None,
    fp16_advance_elapsed=False,
):
    """In-place batched block step for CUDA-graph capture.

    Shapes:
      x/xpa/xpf: [B,D], v_first: [B,A]
      state: [B, H, N, N]

    This mirrors `block_step_batched` but writes recurrent/cache buffers in
    place so a captured CUDA graph can replay across decode tokens.
    """
    (i, H, N, eps, has_pre,
     pre_w, pre_b, an_w, an_b, fn_w, fn_b,
     x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
     Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
     gn_w, gn_b, fx_k, fK, fV, RKVw) = p
    B = x.shape[0]
    D = int(an_w.numel())
    A = int(H * N)
    equal_width = D == A
    residual = F.layer_norm(x, [D], pre_w, pre_b, 1e-5) if has_pre else x
    use_fused_norm_mix = _native_graph_fused_norm_mix_enabled()
    if use_fused_norm_mix:
        stack_rkv = _native_graph_vkwr_rkv_dispatch(B, D) and RKVw.numel() != 0
        xr, xw, xk, xv, xa, xg = fused_attn_norm_mix6_decode(
            residual,
            xpa,
            an_w,
            an_b,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            num_warps=_native_graph_fused_norm_mix_num_warps(),
            stack_rkv=stack_rkv,
        )
    else:
        h = F.layer_norm(residual, [D], an_w, an_b, 1e-5)
        xx = xpa - h
        xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
        xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    v_gate = None
    v_mixed = False
    lora_dense = _graph_linears_are_dense(w1, w2, a1, a2, v1, v2, g1, g2)
    if _native_graph_fused_projection_enabled() and lora_dense and _graph_linears_are_dense(Rw, Kw, Vw):
        r, k, v, w, a, g, v_gate = fused_rkv_wavg_projection(
            xr,
            xk,
            xv,
            xw,
            xa,
            xg,
            Rw,
            Kw,
            Vw,
            w1,
            a1,
            g1,
            v1,
            w2,
            a2,
            g2,
            v2,
            w0,
            a0,
            None,
            v0,
        )
        a = torch.sigmoid(a)
        v_gate = torch.sigmoid(v_gate)
    elif equal_width and i > 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        B,
        D,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0], _graph_linear_shape(v1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        w, a, g, v = ada_wagv_lora(
            xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2,
            w0, a0, v0, v, v_first, sigmoid_a=True,
        )
        v_mixed = True
    elif equal_width and i == 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        B,
        D,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        w, a, g, _unused_v = ada_wagv_lora(
            xw, xa, xg, xg, w1, a1, g1, g1, w2, a2, g2, g2,
            w0, a0, a0, v, v, sigmoid_a=True, compute_v=False,
        )
    elif equal_width and lora_dense and _native_graph_ada_wag_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        w, a, g = ada_wag_lora(
            xw, xa, xg, w1, a1, g1, w2, a2, g2, w0, a0,
        )
        a = torch.sigmoid(a)
    elif equal_width and i > 0 and lora_dense and _native_graph_sm70_wagv_lora_enabled(B, D):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        w, a, g, v = sm70_wagv_lora(
            xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2, w0, a0, v0, v, v_first,
        )
        a = torch.sigmoid(a)
        v_mixed = True
    elif lora_dense and _native_graph_fused_wavg_lora_enabled(B, D):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        if i == 0:
            w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
            a = a0 + F.linear(F.linear(xa, a1), a2)
            g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
        else:
            block_m, block_r, block_k = _native_graph_fused_wavg_lora_blocks()
            w, a, g, v_gate = fused_wavg_lora(
                xw,
                xa,
                xg,
                xv,
                w1,
                a1,
                g1,
                v1,
                w2,
                a2,
                g2,
                v2,
                w0,
                a0,
                None,
                v0,
                block_m=block_m,
                block_r=block_r,
                block_k=block_k,
                num_warps=_native_graph_fused_wavg_lora_num_warps(),
            )
        a = torch.sigmoid(a)
    elif lora_dense and _native_graph_fused_wag_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        block_m, block_r, block_k = _native_graph_fused_wag_lora_blocks()
        w, a, g = fused_wag_lora(
            xw,
            xa,
            xg,
            w1,
            a1,
            g1,
            w2,
            a2,
            g2,
            w0,
            a0,
            None,
            block_m=block_m,
            block_r=block_r,
            block_k=block_k,
        )
        a = torch.sigmoid(a)
    else:
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        w = _graph_linear_call_with_explicit_bias(torch.tanh(_graph_linear_call(xw, w1)), w2, w0)
        a = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xa, a1), a2, a0))
        g = _graph_linear_call(torch.sigmoid(_graph_linear_call(xg, g1)), g2)
    use_fp16_recurrent = _native_graph_fp16_recurrent_enabled(state, fp16_elapsed)
    use_fused_recurrent_output = (
        use_fp16_recurrent or _native_graph_fused_recurrent_output_enabled()
    )
    use_fused_recurrent_raw = use_fp16_recurrent or (
        use_fused_recurrent_output and _native_graph_fused_recurrent_raw_enabled()
    )
    if not use_fused_recurrent_raw:
        kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, A)
        k = k * (1 + (a - 1) * k_a)
    if i == 0:
        v_first.copy_(v)
    elif not v_mixed:
        if v_gate is None:
            v_gate = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xv, v1), v2, v0))
        v = v + (v_first - v) * v_gate
    new_state = None
    if use_fp16_recurrent:
        out = native_fp16_recurrent_output_prepare_raw(
            r.view(B, H, N),
            w.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            a.view(B, H, N),
            state,
            g.view(B, H, N),
            k_k,
            k_a,
            r_k,
            gn_w,
            gn_b,
            fp16_elapsed,
            advance_elapsed=fp16_advance_elapsed,
            eps=eps,
        ).reshape(B, A)
    elif use_fused_recurrent_raw:
        out, new_state = fused_recurrent_output_prepare_raw(
            r.view(B, H, N),
            w.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            a.view(B, H, N),
            state,
            g.view(B, H, N),
            k_k,
            k_a,
            r_k,
            gn_w,
            gn_b,
            eps=eps,
            block_n=N,
        )
        out = out.reshape(B, A)
    elif use_fused_recurrent_output:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
        out, new_state = fused_recurrent_output_prepare(
            r.view(B, H, N),
            w.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            kk.view(B, H, N),
            a.view(B, H, N),
            state,
            g.view(B, H, N),
            r_k,
            gn_w,
            gn_b,
            eps=eps,
            block_n=N,
        )
        out = out.reshape(B, A)
    else:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
        out, new_state = _recurrent_update_batched(r, w, k, v, kk, a, state, B, H, N)
    if use_fused_recurrent_output:
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    elif _native_graph_fused_output_project_enabled() and _graph_linear_is_dense(Ow):
        out = fused_attn_output_project(
            out,
            r.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            g,
            r_k,
            gn_w,
            gn_b,
            Ow,
            None,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
            block_m=_native_graph_fused_output_project_block_m(),
        )
    elif _native_graph_fused_output_enabled():
        out = fused_attn_output_prepare(
            out,
            r.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            g,
            r_k,
            gn_w,
            gn_b,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
        )
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    else:
        out = F.group_norm(out, H, gn_w, gn_b, eps).view(B, A)
        sk = (r.view(B, H, N) * k.view(B, H, N) * r_k).sum(dim=-1, keepdim=True)
        out = (out + (sk * v.view(B, H, N)).view(B, A)) * g
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    if new_state is not None:
        state.copy_(new_state)
    if use_fused_norm_mix:
        if _native_graph_blackwell_norm_mix_enabled(
            residual, out, xpf, layer_index=int(i)
        ):
            residual, fk = blackwell_ffn_add_norm_mix(
                residual, out, xpf, fn_w, fn_b, fx_k, eps=1.0e-5
            )
        else:
            residual, fk = fused_ffn_add_norm_mix_decode(
                residual,
                out,
                xpf,
                fn_w,
                fn_b,
                fx_k,
                num_warps=_native_graph_fused_norm_mix_num_warps(),
            )
    else:
        xpa.copy_(h)
        residual = residual + out
        h2 = F.layer_norm(residual, [D], fn_w, fn_b, 1e-5)
        fxx = xpf - h2
        fk = h2 + fxx * fx_k
        xpf.copy_(h2)
    return _native_graph_ffn_dispatch(fk, fK, fV, residual, sparse_out=sparse_ffn_out)


def cuda_graph_decode(model, ids, packs, n=128):
    import time
    base = model.model
    device = ids.device
    dtype = base.embeddings.weight.dtype
    nL = len(packs)
    H, N = packs[0][1], packs[0][2]
    hid = base.layers[0].attn.hidden_size
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(nL)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    v_first = torch.zeros(hid, device=device, dtype=dtype)
    tok_id = torch.zeros(1, dtype=torch.long, device=device)
    logits = torch.zeros(base.embeddings.weight.shape[0], device=device, dtype=dtype)
    emb = base.embeddings.weight
    head = model.lm_head
    nw, nb = base.norm.weight, base.norm.bias

    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
    tok_id.copy_(_linear_module(head, F.layer_norm(x, [H * N], nw, nb, 1e-5)).argmax())

    def one_step():
        x = F.embedding(tok_id, emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
        logits.copy_(_linear_module(head, F.layer_norm(x, [H * N], nw, nb, 1e-5)))

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one_step()
            tok_id.copy_(logits.argmax())
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        one_step()

    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        g.replay()
        tok_id.copy_(logits.argmax())
    torch.cuda.synchronize(); dt = time.time() - t0
    return n / dt


def greedy_jit(model, ids, packs, n=40):
    base = model.model
    H, N = packs[0][1], packs[0][2]
    nw, nb = base.norm.weight, base.norm.bias
    state, xpa, xpf, v_first = _init(model, ids.device, base.embeddings.weight.dtype)
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], base.embeddings.weight).reshape(-1)
        x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
    nx = _lm_head(model, F.layer_norm(x, [H * N], nw, nb, 1e-5)).argmax().clone()
    toks = [int(nx)]
    with torch.no_grad():
        for _ in range(n - 1):
            x = F.embedding(nx.reshape(1, 1), base.embeddings.weight).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = _lm_head(model, F.layer_norm(x, [H * N], nw, nb, 1e-5)).argmax()
            toks.append(int(nx))
    return toks


def greedy_graph(model, ids, packs, n=40):
    base = model.model
    device = ids.device
    dtype = base.embeddings.weight.dtype
    nL = len(packs)
    H, N = packs[0][1], packs[0][2]
    hid = base.layers[0].attn.hidden_size
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(nL)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    v_first = torch.zeros(hid, device=device, dtype=dtype)
    tok_id = torch.zeros(1, dtype=torch.long, device=device)
    logits = torch.zeros(base.embeddings.weight.shape[0], device=device, dtype=dtype)
    emb, head = base.embeddings.weight, model.lm_head
    nw, nb = base.norm.weight, base.norm.bias
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
    tok_id.copy_(_linear_module(head, F.layer_norm(x, [H * N], nw, nb, 1e-5)).argmax())
    # snapshot post-prefill state so we can realign after warmup advances it
    st_s = [s.clone() for s in state]
    xpa_s = [s.clone() for s in xpa]
    xpf_s = [s.clone() for s in xpf]
    vf_s = v_first.clone()
    tok_s = tok_id.clone()

    def one_step():
        x = F.embedding(tok_id, emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
        logits.copy_(_linear_module(head, F.layer_norm(x, [H * N], nw, nb, 1e-5)))

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one_step(); tok_id.copy_(logits.argmax())
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        one_step()
    # restore post-prefill state so the captured graph replays from the right point
    for i in range(len(state)):
        state[i].copy_(st_s[i]); xpa[i].copy_(xpa_s[i]); xpf[i].copy_(xpf_s[i])
    v_first.copy_(vf_s)
    tok_id.copy_(tok_s)
    toks = [int(tok_id)]
    for _ in range(n - 1):
        g.replay()
        nt = logits.argmax()
        tok_id.copy_(nt)
        toks.append(int(nt))
    return toks


def fast_generate(model, tokenizer, prompt, max_new_tokens=48, use_graph=True):
    """End-to-end greedy generation via the native (CUDA-graph) decode path.
    Returns the full decoded text (prompt + new tokens). Same result as the
    FLA model's greedy generate(), but ~10x faster on the 5070."""
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    packs, _, _, _ = extract(model)
    fn = greedy_graph if use_graph else greedy_jit
    new_tokens = fn(model, ids, packs, n=max_new_tokens)
    full = ids[0].tolist() + new_tokens
    return tokenizer.decode(full, skip_special_tokens=True)


if __name__ == "__main__":
    import os, sys
    os.environ.setdefault("RWKV_V7_ON", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = sys.argv[1] if len(sys.argv) > 1 else "D:/rwkv7-models/rwkv7-g1d-0.1b-hf"
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    # correctness at fp32 vs fla
    model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, torch_dtype=torch.float32, device_map="cuda").eval()
    packs, H, N, eps = extract(model)
    for prompt in ["The quick brown fox jumps over the lazy dog.",
                   "Once upon a time, in a faraway land,"]:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.no_grad():
            fla = model(ids).logits[0, -1].float().cpu()
            nat = forward(model, ids, packs).float().cpu()
        cos = F.cosine_similarity(fla.unsqueeze(0), nat.unsqueeze(0)).item()
        maxabs = (fla - nat).abs().max().item()
        print(f"[correctness] cos={cos:.6f} maxabs={maxabs:.4f} "
              f"argmax={int(fla.argmax() == nat.argmax())}  {prompt[:36]!r}")
    del model; torch.cuda.empty_cache()
    # speed
    for dt_name, dt in [("fp16", torch.float16), ("fp32", torch.float32)]:
        model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, torch_dtype=dt, device_map="cuda").eval()
        packs, H, N, eps = extract(model)
        ids = tok("The quick brown fox.", return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.no_grad():
            tps_jit = decode_speed(model, ids, packs)
            tps_cg = cuda_graph_decode(model, ids, packs)
            tj = greedy_jit(model, ids, packs)
            tg = greedy_graph(model, ids, packs)
        match = sum(int(a == b) for a, b in zip(tj, tg))
        print(f"[decode {dt_name}] jit-fused {tps_jit:.1f} | cuda-graph {tps_cg:.1f} tok/s | "
              f"graph-correct {match}/{len(tj)} tokens == jit")
        del model; torch.cuda.empty_cache()

    # end-to-end: native greedy token ids vs fla model.generate (must match)
    model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda").eval()
    packs, _, _, _ = extract(model)
    prompt = "User: Hello!\n\nAssistant:"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    with torch.no_grad():
        fla_out = model.generate(ids, max_new_tokens=32, do_sample=False, use_cache=True, pad_token_id=0)
    fla_ids = fla_out[0, ids.shape[1]:].tolist()
    nat_ids = greedy_graph(model, ids, packs, n=32)
    print(f"[e2e] fla   : {tok.decode(fla_ids)!r}")
    print(f"[e2e] native: {tok.decode(nat_ids)!r}")
    print(f"[e2e] token-identical: {fla_ids == nat_ids} ({sum(int(a==b) for a,b in zip(fla_ids,nat_ids))}/{len(fla_ids)})")
