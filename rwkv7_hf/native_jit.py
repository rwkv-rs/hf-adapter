# coding=utf-8
"""TorchScript-native RWKV-7 decode. The ENTIRE per-layer block (LayerNorms +
TMix_one + CMix_one) is fused into one torch.jit.script function, so per token
there is only ~1 C++ call per layer + embedding/head. Math ports the official
RWKV_x070 TMix_one/CMix_one (bit-exact vs FLA, see native.py).

Run: python -m rwkv7_hf.native_jit <hf_dir>
"""
from __future__ import annotations

import os
import threading
from functools import wraps
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .native_jit_dense_step import block_step, block_step_batched
from .native_jit_packing import (
    extract_dense_packs as _extract_dense_packs_impl,
    extract_graph_packs as _extract_graph_packs_impl,
    init_batched_from_packs as _init_batched_from_packs,
    init_state as _init,
)

from .native_jit_linear import (
    graph_linear_is_dense as _graph_linear_is_dense,
    graph_linear_operand as _graph_linear_operand,
    graph_linear_shape as _graph_linear_shape,
    linear_module as _linear_module,
    relayout_ffn_value_weight as _native_graph_relayout_ffn_value_weight,
    try_relayout_ffn_value_weight as _try_relayout_ffn_value_weight,
)
from .native_jit_bnb8 import (
    _bnb8_direct_linear,
    _bnb8_direct_relu_square_linear,
    _bnb8_ffn_mix_quant_enabled,
    _bnb8_prequant_linear,
    _bnb8_rkv_mix_quant_enabled,
    _is_bnb8_linear,
    _native_bnb8_direct_enabled,
    _native_bnb8_policy_block,
    _native_bnb8_policy_flag,
)
from .native_jit_prefill_policy import (
    model_shape_selected as _prefill_model_shape_selected_impl,
    policy_model_shape_selected as _prefill_policy_model_shape_selected_impl,
    self_chunk_h_tiles as _prefill_self_chunk_h_tiles_impl,
    self_chunk_shape_eligible as _prefill_self_chunk_shape_eligible,
    self_chunk_size as _prefill_self_chunk_size_impl,
)


_FP16_ACCUMULATION_LOCK = threading.RLock()


def _cuda_device_guard(device):
    return (
        torch.cuda.device(device)
        if getattr(device, "type", None) == "cuda" and torch.cuda.is_available()
        else nullcontext()
    )

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


def _native_graph_sparse_ffn_low_memory_pack_enabled() -> bool:
    policy = _kernel_policy()
    return env_flag(
        "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_LOW_MEMORY_PACK",
        bool(getattr(policy, "ada_sparse_ffn_low_memory_pack", False)),
    )


def _native_graph_try_relayout_ffn_value_weight(module) -> bool:
    """Apply the fp16 sparse layout only to its exact dense-module contract.

    Exact-card policy can enable low-memory sparse FFN packing by default, but
    Hugging Face may replace an FFN projection with a BnB/Marlin/TorchAO
    module. Those modules must remain callable graph operands; inspecting
    their packed ``weight`` dtype as if it were a dense parameter makes a
    validated 5090 policy reject otherwise supported W8/W4 models.
    """

    return _try_relayout_ffn_value_weight(
        module,
        relayout_fn=_native_graph_relayout_ffn_value_weight,
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
        with _FP16_ACCUMULATION_LOCK:
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

try:  # Keep this separate so older remote-code policy modules still import.
    from .kernel_policy import is_rtx_model_name as _is_rtx_model_name
except Exception:  # pragma: no cover - remote-code/backward-compatible fallback
    try:
        from kernel_policy import is_rtx_model_name as _is_rtx_model_name
    except Exception:
        def _is_rtx_model_name(name: str, model: str) -> bool:
            normalized = "".join(
                character if character.isalnum() else " "
                for character in str(name).lower()
            )
            tokens = tuple(normalized.split())
            model_token = str(model).lower()
            if "rtx" not in tokens or model_token not in tokens:
                return False
            model_index = tokens.index(model_token)
            return bool(
                not {"laptop", "mobile", "maxq", "max", "q", "super", "ti"}.intersection(tokens)
                and all(token == "gpu" for token in tokens[model_index + 1 :])
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
        ada_sparse_ffn_deterministic4_should_use,
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
            ada_sparse_ffn_deterministic4_should_use,
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
        ada_sparse_ffn_deterministic4_should_use = None  # type: ignore[assignment]
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

try:  # pragma: no cover - optional exact official-order SM120 norm/mix
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


from . import native_jit_prefill_runtime_policy as _native_jit_prefill_runtime_policy_impl


def _prefill_runtime_policy_wrapper(name):
    implementation = getattr(_native_jit_prefill_runtime_policy_impl, name)

    @wraps(implementation)
    def compatibility_wrapper(*args, **kwargs):
        _native_jit_prefill_runtime_policy_impl.bind_runtime(globals())
        return implementation(*args, **kwargs)

    return compatibility_wrapper


_native_prefill_fused_scan_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fused_scan_enabled")
_native_prefill_fp16_recurrent_requested = _prefill_runtime_policy_wrapper("_native_prefill_fp16_recurrent_requested")
_native_prefill_fp16_recurrent_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fp16_recurrent_enabled")
_native_prefill_self_chunk_enabled = _prefill_runtime_policy_wrapper("_native_prefill_self_chunk_enabled")
_native_prefill_self_chunk_size = _prefill_runtime_policy_wrapper("_native_prefill_self_chunk_size")
_native_prefill_self_chunk_h_tiles = _prefill_runtime_policy_wrapper("_native_prefill_self_chunk_h_tiles")
_native_prefill_self_chunk_safe_gate = _prefill_runtime_policy_wrapper("_native_prefill_self_chunk_safe_gate")
_native_prefill_dplr_scan_enabled = _prefill_runtime_policy_wrapper("_native_prefill_dplr_scan_enabled")
_native_prefill_fused_residual_gemm_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fused_residual_gemm_enabled")
_native_prefill_dplr_chunk_size = _prefill_runtime_policy_wrapper("_native_prefill_dplr_chunk_size")
_native_prefill_fused_clampw_scan_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fused_clampw_scan_enabled")
_native_prefill_fused_scan_output_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fused_scan_output_enabled")
_native_prefill_default_scan_block_m = _prefill_runtime_policy_wrapper("_native_prefill_default_scan_block_m")
_native_prefill_scan_block_m = _prefill_runtime_policy_wrapper("_native_prefill_scan_block_m")
_native_prefill_scan_num_warps = _prefill_runtime_policy_wrapper("_native_prefill_scan_num_warps")
_native_prefill_model_shape_selected = _prefill_runtime_policy_wrapper("_native_prefill_model_shape_selected")
_native_prefill_policy_model_shape_selected = _prefill_runtime_policy_wrapper("_native_prefill_policy_model_shape_selected")
_native_prefill_fused_shift_mix_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fused_shift_mix_enabled")
_native_prefill_shift_mix_layers = _prefill_runtime_policy_wrapper("_native_prefill_shift_mix_layers")
_native_prefill_shift_mix_launch_profile = _prefill_runtime_policy_wrapper("_native_prefill_shift_mix_launch_profile")
_native_prefill_attn_shift_mix_block_size = _prefill_runtime_policy_wrapper("_native_prefill_attn_shift_mix_block_size")
_native_prefill_shift_mix_num_warps = _prefill_runtime_policy_wrapper("_native_prefill_shift_mix_num_warps")
_native_prefill_ffn_shift_mix_block_size = _prefill_runtime_policy_wrapper("_native_prefill_ffn_shift_mix_block_size")
_native_prefill_fused_state_prep_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fused_state_prep_enabled")
_native_prefill_state_prep_layers = _prefill_runtime_policy_wrapper("_native_prefill_state_prep_layers")
_native_prefill_fused_state_scan_max_batch = _prefill_runtime_policy_wrapper("_native_prefill_fused_state_scan_max_batch")
_native_prefill_fused_state_scan_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fused_state_scan_enabled")
_native_prefill_state_prep_w_dtype = _prefill_runtime_policy_wrapper("_native_prefill_state_prep_w_dtype")
_native_prefill_fused_output_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fused_output_enabled")
_native_prefill_fused_output_project_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fused_output_project_enabled")
_native_prefill_fused_output_project_block_m = _prefill_runtime_policy_wrapper("_native_prefill_fused_output_project_block_m")
_native_prefill_fused_wavg_lora_requested = _prefill_runtime_policy_wrapper("_native_prefill_fused_wavg_lora_requested")
_native_prefill_fused_wavg_lora_max_m = _prefill_runtime_policy_wrapper("_native_prefill_fused_wavg_lora_max_m")
_native_prefill_fused_wavg_lora_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fused_wavg_lora_enabled")
_native_prefill_fused_wavg_lora_blocks = _prefill_runtime_policy_wrapper("_native_prefill_fused_wavg_lora_blocks")
_native_prefill_fused_sequence_ffn_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fused_sequence_ffn_enabled")
_native_prefill_fp16_accum_ffn_key_enabled = _prefill_runtime_policy_wrapper("_native_prefill_fp16_accum_ffn_key_enabled")
_native_prefill_fp16_accum_ffn_key_layers = _prefill_runtime_policy_wrapper("_native_prefill_fp16_accum_ffn_key_layers")
_native_prefill_stacked_rkv_enabled = _prefill_runtime_policy_wrapper("_native_prefill_stacked_rkv_enabled")
_native_prefill_sequence_ffn_blocks = _prefill_runtime_policy_wrapper("_native_prefill_sequence_ffn_blocks")
_native_prefill_sequence_ffn_launch = _prefill_runtime_policy_wrapper("_native_prefill_sequence_ffn_launch")


# Graph policy and projection dispatch are direct aliases. Bind once after
# optional kernels and policy helpers are initialized so token-loop calls gain
# no compatibility wrapper.
from . import native_jit_graph_dispatch as _native_jit_graph_dispatch_impl
_native_jit_graph_dispatch_impl.bind_runtime(globals())
_native_graph_fused_recurrent_output_enabled = _native_jit_graph_dispatch_impl._native_graph_fused_recurrent_output_enabled
_native_graph_fused_recurrent_raw_enabled = _native_jit_graph_dispatch_impl._native_graph_fused_recurrent_raw_enabled
_native_graph_fp16_recurrent_enabled = _native_jit_graph_dispatch_impl._native_graph_fp16_recurrent_enabled
_native_graph_fused_output_enabled = _native_jit_graph_dispatch_impl._native_graph_fused_output_enabled
_native_graph_fused_output_project_enabled = _native_jit_graph_dispatch_impl._native_graph_fused_output_project_enabled
_native_graph_fused_output_project_block_m = _native_jit_graph_dispatch_impl._native_graph_fused_output_project_block_m
_native_graph_fused_projection_enabled = _native_jit_graph_dispatch_impl._native_graph_fused_projection_enabled
_native_graph_fused_wag_lora_enabled = _native_jit_graph_dispatch_impl._native_graph_fused_wag_lora_enabled
_native_graph_sm70_wagv_lora_enabled = _native_jit_graph_dispatch_impl._native_graph_sm70_wagv_lora_enabled
_native_graph_fused_wavg_lora_enabled = _native_jit_graph_dispatch_impl._native_graph_fused_wavg_lora_enabled
_native_graph_fused_norm_mix_enabled = _native_jit_graph_dispatch_impl._native_graph_fused_norm_mix_enabled
_native_graph_fused_norm_mix_num_warps = _native_jit_graph_dispatch_impl._native_graph_fused_norm_mix_num_warps
_native_graph_blackwell_norm_mix_enabled = _native_jit_graph_dispatch_impl._native_graph_blackwell_norm_mix_enabled
_native_graph_sm70_linear_enabled = _native_jit_graph_dispatch_impl._native_graph_sm70_linear_enabled
_native_graph_ada_sparse_ffn_enabled = _native_jit_graph_dispatch_impl._native_graph_ada_sparse_ffn_enabled
_native_graph_ada_linear_enabled = _native_jit_graph_dispatch_impl._native_graph_ada_linear_enabled
_native_graph_ada_linear_should_route = _native_jit_graph_dispatch_impl._native_graph_ada_linear_should_route
_native_graph_ada_wagv_lora_enabled = _native_jit_graph_dispatch_impl._native_graph_ada_wagv_lora_enabled
_native_graph_ada_wag_lora_enabled = _native_jit_graph_dispatch_impl._native_graph_ada_wag_lora_enabled
_native_graph_linear_dispatch = _native_jit_graph_dispatch_impl._native_graph_linear_dispatch
_native_graph_ffn_up_relu2_dispatch = _native_jit_graph_dispatch_impl._native_graph_ffn_up_relu2_dispatch
_native_graph_ffn_down_add_dispatch = _native_jit_graph_dispatch_impl._native_graph_ffn_down_add_dispatch
_native_graph_ffn_dispatch = _native_jit_graph_dispatch_impl._native_graph_ffn_dispatch
prewarm_ada_sparse_ffn = _native_jit_graph_dispatch_impl.prewarm_ada_sparse_ffn
_native_graph_rkv_policy = _native_jit_graph_dispatch_impl._native_graph_rkv_policy
_native_graph_int_env = _native_jit_graph_dispatch_impl._native_graph_int_env
_native_graph_vkwr_rkv_dispatch = _native_jit_graph_dispatch_impl._native_graph_vkwr_rkv_dispatch
_native_graph_rkv_project = _native_jit_graph_dispatch_impl._native_graph_rkv_project
_native_graph_fused_wag_lora_blocks = _native_jit_graph_dispatch_impl._native_graph_fused_wag_lora_blocks
_native_graph_fused_wavg_lora_blocks = _native_jit_graph_dispatch_impl._native_graph_fused_wavg_lora_blocks
_native_graph_fused_wavg_lora_num_warps = _native_jit_graph_dispatch_impl._native_graph_fused_wavg_lora_num_warps


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




def _extract_current_device(model):
    return _extract_dense_packs_impl(
        model,
        rkv_policy=_native_graph_rkv_policy(),
    )


def extract(model):
    """Extract JIT packs under the model weight's CUDA device guard."""

    device = model.model.embeddings.weight.device
    with _cuda_device_guard(device):
        return _extract_current_device(model)


def _extract_graph_current_device(model):
    return _extract_graph_packs_impl(
        model,
        rkv_policy=_native_graph_rkv_policy(),
        sparse_ffn_low_memory_pack_enabled=_native_graph_sparse_ffn_low_memory_pack_enabled,
        try_relayout_ffn_value_weight=_native_graph_try_relayout_ffn_value_weight,
        graph_linear_operand=_graph_linear_operand,
        graph_linear_is_dense=_graph_linear_is_dense,
    )


def extract_graph(model):
    """Extract graph packs under the model weight's CUDA device guard."""

    device = model.model.embeddings.weight.device
    with _cuda_device_guard(device):
        return _extract_graph_current_device(model)


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

    if _native_prefill_fused_scan_enabled(B, T, H * N, num_layers):
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


def _prefill_current_device(
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
    strict_shift_mix_fp16 = bool(
        dtype == torch.float16
        and env_flag("RWKV7_NATIVE_PREFILL_SHIFT_MIX_STRICT_FP16", False)
    )
    strict_attn_default = _native_prefill_policy_model_shape_selected(
        "prefill_attn_shift_mix_strict_fp16_model_shapes",
        B,
        T,
        residual_hidden,
        len(packs),
    )
    strict_attn_shift_mix_fp16 = bool(
        dtype == torch.float16
        and env_flag(
            "RWKV7_NATIVE_PREFILL_ATTN_SHIFT_MIX_STRICT_FP16",
            strict_shift_mix_fp16 or strict_attn_default,
        )
        and _native_prefill_model_shape_selected(
            "RWKV7_NATIVE_PREFILL_ATTN_SHIFT_MIX_STRICT_FP16_MODEL_SHAPES",
            "prefill_attn_shift_mix_strict_fp16_model_shapes",
            B,
            T,
            residual_hidden,
            len(packs),
        )
    )
    strict_ffn_default = _native_prefill_policy_model_shape_selected(
        "prefill_ffn_shift_mix_strict_fp16_model_shapes",
        B,
        T,
        residual_hidden,
        len(packs),
    )
    strict_ffn_shift_mix_fp16 = bool(
        dtype == torch.float16
        and env_flag(
            "RWKV7_NATIVE_PREFILL_FFN_SHIFT_MIX_STRICT_FP16",
            strict_shift_mix_fp16 or strict_ffn_default,
        )
        and _native_prefill_model_shape_selected(
            "RWKV7_NATIVE_PREFILL_FFN_SHIFT_MIX_STRICT_FP16_MODEL_SHAPES",
            "prefill_ffn_shift_mix_strict_fp16_model_shapes",
            B,
            T,
            residual_hidden,
            len(packs),
        )
    )
    attn_shift_mix_block_size = _native_prefill_attn_shift_mix_block_size(
        strict_attn_shift_mix_fp16,
        B,
        T,
        residual_hidden,
        len(packs),
    )
    attn_shift_mix_num_warps = _native_prefill_shift_mix_num_warps(
        "ATTN", B, T, residual_hidden, len(packs)
    )
    ffn_shift_mix_block_size = _native_prefill_ffn_shift_mix_block_size(
        B, T, residual_hidden, len(packs)
    )
    ffn_shift_mix_num_warps = _native_prefill_shift_mix_num_warps(
        "FFN", B, T, residual_hidden, len(packs)
    )
    prefill_shift_mix_layers = (
        _native_prefill_shift_mix_layers(B, T, len(packs))
        if use_prefill_shift_mix
        else set()
    )
    use_prefill_state_prep = _native_prefill_fused_state_prep_enabled(
        B, T, residual_hidden, len(packs)
    )
    use_prefill_output = _native_prefill_fused_output_enabled(
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
        use_layer_shift_mix = bool(
            use_prefill_shift_mix
            and (
                prefill_shift_mix_layers is None
                or layer_idx in prefill_shift_mix_layers
            )
        )
        use_layer_attn_shift_mix = bool(
            use_layer_shift_mix
            and env_flag("RWKV7_NATIVE_PREFILL_FUSED_ATTN_SHIFT_MIX", True)
        )
        use_layer_ffn_shift_mix = bool(
            use_layer_shift_mix
            and env_flag("RWKV7_NATIVE_PREFILL_FUSED_FFN_SHIFT_MIX", True)
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
        use_sequence_attn_mix = (
            use_layer_attn_shift_mix and fused_attn_sequence_shift_mix is not None
        )
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
                block_size=attn_shift_mix_block_size,
                num_warps=attn_shift_mix_num_warps,
                workspace=sequence_attn_mix_workspace,
                strict_fp16_rounding=strict_attn_shift_mix_fp16,
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
        elif use_prefill_output:
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
            if use_layer_ffn_shift_mix and _bnb8_ffn_mix_quant_enabled(fK):
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
            elif use_layer_ffn_shift_mix and fused_ffn_sequence_shift_mix is not None:
                if sequence_ffn_mix_workspace is None:
                    sequence_ffn_mix_workspace = torch.empty_like(h2)
                fk, next_xpf = fused_ffn_sequence_shift_mix(
                    h2,
                    xpf[layer_idx],
                    fx_k,
                    block_size=ffn_shift_mix_block_size,
                    num_warps=ffn_shift_mix_num_warps,
                    workspace=sequence_ffn_mix_workspace,
                    strict_fp16_rounding=strict_ffn_shift_mix_fp16,
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
                use_layer_ffn_shift_mix
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
    """Run prefill with policy detection bound to the input tensor's GPU."""

    with _cuda_device_guard(ids.device):
        return _prefill_current_device(
            model,
            ids,
            packs,
            state=state,
            xpa=xpa,
            xpf=xpf,
            logits_to_keep=logits_to_keep,
            fp16_elapsed=fp16_elapsed,
        )


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
