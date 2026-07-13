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


def _linear_module(module, x: torch.Tensor) -> torch.Tensor:
    """Linear call that also supports native MM8/MM4Linear lm_head modules."""
    if type(module) is torch.nn.Linear:
        return F.linear(x, module.weight, module.bias)
    return module(x)


def _graph_linear_operand(module):
    """Return a dense weight when possible, otherwise retain the quant module.

    Native CUDA-graph decode historically packed bare ``nn.Linear.weight``
    tensors.  MM8/MM4 modules intentionally have no dense weight, so retaining
    the module lets graph capture record their fused dequant GEMV without
    materialising a second fp16 copy of the model.
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


def _graph_linear_call(x: torch.Tensor, operand) -> torch.Tensor:
    if _graph_linear_is_dense(operand):
        return F.linear(x, operand)
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
    from .fused_lora import fused_wag_lora, fused_wag_lora_available, fused_wavg_lora, fused_wavg_lora_available
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_lora import fused_wag_lora, fused_wag_lora_available, fused_wavg_lora, fused_wavg_lora_available
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
            ada_sparse_ffn_should_use,
        )
    except Exception:
        ada_ffn_up = None  # type: ignore[assignment]
        ada_linear = None  # type: ignore[assignment]
        ada_linear_should_use = None  # type: ignore[assignment]
        ada_sparse_ffn_down_add = None  # type: ignore[assignment]
        ada_sparse_ffn_pack_weight = None  # type: ignore[assignment]
        ada_sparse_ffn_should_use = None  # type: ignore[assignment]

try:  # pragma: no cover - optional sm_89 grouped W/A/G/V LoRA
    from .ada_lora import ada_wagv_lora, ada_wagv_lora_should_use
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from ada_lora import ada_wagv_lora, ada_wagv_lora_should_use
    except Exception:
        ada_wagv_lora = None  # type: ignore[assignment]
        ada_wagv_lora_should_use = None  # type: ignore[assignment]


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


def _native_prefill_self_chunk_enabled(tokens: int, head_dim: int) -> bool:
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
    if int(tokens) < min_tokens or int(tokens) % 16 or int(head_dim) != 64:
        return False
    if self_chunk_rwkv7 is None or self_chunk_rwkv7_available is None:
        return False
    try:
        return bool(self_chunk_rwkv7_available())
    except Exception:
        return False


def _native_prefill_self_chunk_size(_batch_size: int) -> int:
    """Return the exact-card sequence chunk size."""

    policy = _kernel_policy()
    default = int(getattr(policy, "prefill_self_chunk_size", 16))
    chunk_size = env_int(
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE",
        default,
        lower=16,
        upper=64,
    )
    if chunk_size not in {16, 32, 64}:
        raise ValueError("RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE must be 16, 32, or 64")
    return chunk_size


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


def _native_prefill_dplr_chunk_size() -> int:
    """Chunk length for the pure-torch DPLR/chunked prefill reference path."""

    return env_int("RWKV7_NATIVE_PREFILL_DPLR_CHUNK_SIZE", 64, lower=1, upper=4096)


def _native_prefill_fused_clampw_scan_enabled() -> bool:
    """Runtime switch for raw-W clampw native prefill recurrent scan."""

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN", False):
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


def _native_prefill_default_scan_block_m(head_dim: int, batch_size: int | None = None) -> int:
    """Architecture-aware default row tile for optional prefill scans."""

    head_dim = int(head_dim)
    policy = _kernel_policy()
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
                return 8 if batch_size is not None and int(batch_size) >= 2 else 4
    return head_dim


def _native_prefill_scan_block_m(head_dim: int, batch_size: int | None = None) -> int:
    """Row tile for optional recurrent scans; explicit env always wins."""

    return env_int(
        "RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M",
        _native_prefill_default_scan_block_m(head_dim, batch_size),
        lower=1,
        upper=int(head_dim),
    )


def _native_prefill_scan_num_warps(head_dim: int, block_m: int | None = None) -> int:
    """Triton warp count for the optional native prefill recurrent scan."""

    if block_m is None:
        block_m = _native_prefill_scan_block_m(head_dim)
    policy = _kernel_policy()
    policy_value = getattr(policy, "prefill_scan_num_warps", None)
    default = int(policy_value) if policy_value is not None else (4 if int(block_m) < int(head_dim) else 8)
    value = env_int("RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS", default, lower=1, upper=8)
    if value not in {1, 2, 4, 8}:
        raise ValueError(f"RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS must be one of 1, 2, 4, or 8; got {value}")
    return value


def _native_prefill_fused_shift_mix_enabled() -> bool:
    """Runtime switch for prefill attention shift-mix fusion telemetry."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX",
        bool(getattr(policy, "fused_prefill_shift_mix", False)),
    ):
        return False
    if fused_attn_shift_mix is None or fused_attn_shift_mix_available is None:
        return False
    try:
        return bool(fused_attn_shift_mix_available())
    except Exception:
        return False


def _native_prefill_fused_state_prep_enabled() -> bool:
    """Runtime switch for the native prefill state-prep fusion probe."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP",
        bool(getattr(policy, "fused_prefill_state_prep", False)),
    ):
        return False
    if fused_prefill_state_prep is None or fused_prefill_state_prep_available is None:
        return False
    try:
        return bool(fused_prefill_state_prep_available())
    except Exception:
        return False


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


def _native_prefill_fused_sequence_ffn_enabled(total_rows: int) -> bool:
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
    if not (min_rows <= int(total_rows) <= max_rows or int(total_rows) in extra_rows):
        return False
    if fused_sequence_ffn is None or fused_sequence_ffn_available is None:
        return False
    try:
        return bool(fused_sequence_ffn_available())
    except Exception:
        return False


def _native_prefill_stacked_rkv_enabled(total_rows: int) -> bool:
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
    return min_rows <= int(total_rows) <= max_rows or int(total_rows) in extra_rows


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
    if not _graph_linear_is_dense(weight):
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
        and sm70_orig_rkv is not None
        and int(rows) in {2, 4}
        and int(hidden_size) >= 2048
    ):
        return sm70_orig_rkv(xr, xk, xv, Rw, Kw, Vw)
    if (
        dense_rkv
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
    # --- block wiring (fuse_norm=False) ---
    if has_pre == 1:
        residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5)
    else:
        residual = x
    h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)

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
    kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(H * N)
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
    out = out.view(H * N)
    out = F.group_norm(out.view(1, H * N), H, gn_w, gn_b, eps).view(H * N)
    sk = (r.view(H, N) * k.view(H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(H, N)).view(H * N)
    out = F.linear(out * g, Ow)
    x = residual + out

    # --- CMix_one ---
    residual = x
    h2 = F.layer_norm(x, [H * N], fn_w, fn_b, 1e-5)
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
    # x/xpa/xpf/v_first:[B,H*N], state:[B,H,N,N].
    B = x.shape[0]
    if has_pre == 1:
        residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5)
    else:
        residual = x
    h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)

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
    kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, H * N)
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
    out = out.view(B, H * N)
    out = F.group_norm(out, H, gn_w, gn_b, eps).view(B, H * N)
    sk = (r.view(B, H, N) * k.view(B, H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(B, H, N)).view(B, H * N)
    out = F.linear(out * g, Ow)
    x = residual + out

    residual = x
    h2 = F.layer_norm(x, [H * N], fn_w, fn_b, 1e-5)
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
    dense_ref = model.model.embeddings.weight
    stack_rkv = _native_graph_rkv_policy() == "vkwr_auto"
    for i, layer in enumerate(layers):
        a = layer.attn
        ref = a.w_lora.lora[0].weight
        vl = getattr(a, "v_lora", None)
        v1 = vl.lora[0].weight if vl is not None else torch.zeros(1, ref.shape[1], device=ref.device, dtype=ref.dtype)
        v2 = vl.lora[2].weight if vl is not None else torch.zeros(hidden, 1, device=ref.device, dtype=ref.dtype)
        v0 = vl.lora[2].bias if vl is not None else torch.zeros(hidden, device=ref.device, dtype=ref.dtype)
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
    stack_rkv = _native_graph_rkv_policy() == "vkwr_auto"
    embed_ref = model.model.embeddings.weight
    for i, layer in enumerate(layers):
        a = layer.attn
        vl = getattr(a, "v_lora", None)
        if vl is not None:
            v1 = _graph_linear_operand(vl.lora[0])
            v2 = _graph_linear_operand(vl.lora[2])
            v0 = vl.lora[2].bias
        else:
            v1 = torch.zeros(1, hidden, device=embed_ref.device, dtype=embed_ref.dtype)
            v2 = torch.zeros(hidden, 1, device=embed_ref.device, dtype=embed_ref.dtype)
            v0 = torch.zeros(hidden, device=embed_ref.device, dtype=embed_ref.dtype)
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
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(n)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    v_first = torch.zeros(hid, device=device, dtype=dtype)
    return state, xpa, xpf, v_first


def _init_batched_from_packs(packs, batch_size: int, device, dtype):
    n = len(packs)
    H = int(packs[0][1])
    N = int(packs[0][2])
    hid = H * N
    state = [torch.zeros(batch_size, H, N, N, device=device, dtype=torch.float32) for _ in range(n)]
    xpa = [torch.zeros(batch_size, hid, device=device, dtype=dtype) for _ in range(n)]
    xpf = [torch.zeros(batch_size, hid, device=device, dtype=dtype) for _ in range(n)]
    return state, xpa, xpf


def step(model, x, state, xpa, xpf, v_first, packs):
    for p in packs:
        x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]] = block_step(x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]], *p)
    return x, state, xpa, xpf, v_first


def step_batched(model, x, state, xpa, xpf, v_first, packs):
    """Batched TorchScript block-step decode for native_model caches.

    Shapes mirror ``rwkv7_hf.native._step_token_batched``: x/xpa/xpf/v_first
    are ``[B, hidden]`` and recurrent state is ``[B, H, N, N]`` per layer.
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
):
    """Run the recurrent prefill scan, using Triton only when explicitly enabled."""

    if w_is_raw and _native_prefill_fused_clampw_scan_enabled():
        scan_block_m = _native_prefill_scan_block_m(N, B)
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

    if _native_prefill_self_chunk_enabled(T, N):
        chunk_size = _native_prefill_self_chunk_size(B)
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
        )
        return out.reshape(B, T, H * N), new_state

    if w_is_log:
        w = torch.exp(w.float())

    if _native_prefill_fused_scan_enabled():
        scan_block_m = _native_prefill_scan_block_m(N, B)
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
    hidden = H * N
    dtype = base.embeddings.weight.dtype
    if state is None or xpa is None or xpf is None:
        state, xpa, xpf = _init_batched_from_packs(packs, B, ids.device, dtype)
    else:
        state = [s.to(device=ids.device, dtype=torch.float32).contiguous() for s in state]
        xpa = [s.to(device=ids.device, dtype=dtype).contiguous() for s in xpa]
        xpf = [s.to(device=ids.device, dtype=dtype).contiguous() for s in xpf]

    x = F.embedding(ids, base.embeddings.weight).reshape(B, T, hidden)
    v_first_seq = torch.zeros(B, T, hidden, device=ids.device, dtype=dtype)
    use_prefill_sequence_ffn = _native_prefill_fused_sequence_ffn_enabled(B * T)
    sequence_ffn_blocks = _native_prefill_sequence_ffn_blocks(B * T) if use_prefill_sequence_ffn else None
    sequence_ffn_launch = _native_prefill_sequence_ffn_launch() if use_prefill_sequence_ffn else None
    sequence_ffn_workspace = None
    sequence_attn_mix_workspace = None
    sequence_ffn_mix_workspace = None
    stacked_rkv_weights = (
        _native_prefill_stacked_rkv_weights(model, packs)
        if _native_prefill_stacked_rkv_enabled(B * T)
        else None
    )
    stacked_rkv_used = False

    for p in packs:
        (i, H, N, eps, has_pre,
         pre_w, pre_b, an_w, an_b, fn_w, fn_b,
         x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
         Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
         gn_w, gn_b, fx_k, fK, fV, _RKVw) = p
        layer_idx = int(i)
        H = int(H)
        N = int(N)
        hidden = H * N

        residual = F.layer_norm(x, [hidden], pre_w, pre_b, 1e-5) if int(has_pre) == 1 else x
        h = F.layer_norm(residual, [hidden], an_w, an_b, 1e-5)
        use_prefill_shift_mix = _native_prefill_fused_shift_mix_enabled()
        use_sequence_attn_mix = use_prefill_shift_mix and fused_attn_sequence_shift_mix is not None
        if use_sequence_attn_mix:
            if sequence_attn_mix_workspace is None:
                sequence_attn_mix_workspace = torch.empty(
                    (6, B, T, hidden), device=h.device, dtype=h.dtype
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
            prev_h = torch.cat([xpa[layer_idx].view(B, 1, hidden), h[:, :-1, :]], dim=1)
            xx = prev_h - h
            xr = h + xx * x_r.view(1, 1, hidden)
            xw = h + xx * x_w.view(1, 1, hidden)
            xk = h + xx * x_k.view(1, 1, hidden)
            xv = h + xx * x_v.view(1, 1, hidden)
            xa = h + xx * x_a.view(1, 1, hidden)
            xg = h + xx * x_g.view(1, 1, hidden)

        use_stacked_rkv = False
        if stacked_rkv_weights:
            row_values = B * T * hidden
            use_stacked_rkv = bool(
                xr.is_contiguous()
                and xk.is_contiguous()
                and xv.is_contiguous()
                and xr.untyped_storage().data_ptr() == xk.untyped_storage().data_ptr()
                and xr.untyped_storage().data_ptr() == xv.untyped_storage().data_ptr()
                and int(xk.storage_offset()) == int(xr.storage_offset()) + row_values
                and int(xv.storage_offset()) == int(xr.storage_offset()) + 2 * row_values
            )
        if use_stacked_rkv:
            stacked_rkv_used = True
            rkv_inputs = xr.as_strided((3, B * T, hidden), (B * T * hidden, hidden, 1))
            rkv = torch.bmm(rkv_inputs, stacked_rkv_weights[layer_idx])
            r = rkv[0].view(B, T, hidden)
            k = rkv[1].view(B, T, hidden)
            v = rkv[2].view(B, T, hidden)
        else:
            r = F.linear(xr, Rw)
            k = F.linear(xk, Kw)
            v = F.linear(xv, Vw)
        v_gate = None
        state_sigmoid_is_raw = False
        defer_state_sigmoid = bool(
            _native_prefill_fused_state_prep_enabled()
            and not _native_prefill_fused_state_scan_enabled(B)
            and not _native_prefill_fused_clampw_scan_enabled()
        )
        use_prefill_wavg_lora = layer_idx > 0 and _native_prefill_fused_wavg_lora_enabled(B * T)
        if use_prefill_wavg_lora:
            block_m, block_r, block_k = _native_prefill_fused_wavg_lora_blocks()
            w, a, g, v_gate = fused_wavg_lora(
                xw.reshape(B * T, hidden),
                xa.reshape(B * T, hidden),
                xg.reshape(B * T, hidden),
                xv.reshape(B * T, hidden),
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
            w = w.view(B, T, hidden)
            a = torch.sigmoid(a.view(B, T, hidden))
            g = g.view(B, T, hidden)
            v_gate = v_gate.view(B, T, hidden)
        else:
            w_mid = F.linear(xw, w1)
            w_mid.tanh_()
            w = F.linear(w_mid, w2, w0)
            a_mid = F.linear(xa, a1)
            a = F.linear(a_mid, a2, a0)
            if not defer_state_sigmoid:
                a.sigmoid_()
            else:
                state_sigmoid_is_raw = True
            g_mid = F.linear(xg, g1)
            g_mid.sigmoid_()
            g = F.linear(g_mid, g2)
            if layer_idx != 0:
                v_mid = F.linear(xv, v1)
                v_gate = F.linear(v_mid, v2, v0)
                if not defer_state_sigmoid:
                    v_gate.sigmoid_()
        use_fused_scan_output = _native_prefill_fused_scan_output_enabled()
        use_self_chunk = _native_prefill_self_chunk_enabled(T, N) and not use_fused_scan_output
        self_chunk_w_is_log = False
        use_clampw_scan = _native_prefill_fused_clampw_scan_enabled() and not use_fused_scan_output
        use_fused_state_scan = _native_prefill_fused_state_scan_enabled(B) and not use_fused_scan_output
        if use_clampw_scan and _native_prefill_fused_state_prep_enabled() and fused_prefill_kv_kk_prep is None:
            use_clampw_scan = False
        state_scan_done = False
        if use_fused_state_scan:
            scan_block_m = _native_prefill_scan_block_m(N, B)
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
                v_first_seq = v.reshape(B, T, hidden)
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
            out = out.reshape(B, T, hidden)
            k = k.reshape(B, T, hidden)
            v = v.reshape(B, T, hidden)
            state_scan_done = True
        elif _native_prefill_fused_state_prep_enabled():
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
            kk = F.normalize((k * k_k.view(1, 1, hidden)).view(B, T, H, N), dim=-1, p=2.0).view(B, T, hidden)
            k = k * (1 + (a - 1) * k_a.view(1, 1, hidden))
            if layer_idx == 0:
                v_first_seq = v
            else:
                v = v + (v_first_seq - v) * v_gate
            if not use_clampw_scan:
                w = torch.exp(-0.606531 * torch.sigmoid(w.float()))

        if use_fused_scan_output:
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
            out = out.reshape(B, T, hidden)
        elif not state_scan_done:
            out, new_state = _native_prefill_scan(
                r, w, k, v, kk, a, state[layer_idx], B, T, H, N,
                w_is_raw=use_clampw_scan,
                w_is_log=self_chunk_w_is_log,
            )
        out_projected = False
        if use_fused_scan_output:
            pass
        elif _native_prefill_fused_output_project_enabled():
            out = fused_attn_output_project(
                out.reshape(B * T, hidden),
                r.reshape(B * T, H, N),
                k.reshape(B * T, H, N),
                v.reshape(B * T, H, N),
                g.reshape(B * T, hidden),
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
            ).view(B, T, hidden)
            out_projected = True
        elif _native_prefill_fused_output_enabled():
            out = fused_attn_output_prepare(
                out.reshape(B * T, hidden),
                r.reshape(B * T, H, N),
                k.reshape(B * T, H, N),
                v.reshape(B * T, H, N),
                g.reshape(B * T, hidden),
                r_k,
                gn_w,
                gn_b,
                num_heads=H,
                head_dim=N,
                head_v_dim=N,
                eps=eps,
            ).view(B, T, hidden)
        else:
            out = F.group_norm(out.reshape(B * T, hidden), H, gn_w, gn_b, eps).view(B, T, hidden)
            sk = (r.view(B, T, H, N) * k.view(B, T, H, N) * r_k.view(1, 1, H, N)).sum(dim=-1, keepdim=True)
            out = (out + (sk * v.view(B, T, H, N)).view(B, T, hidden)) * g
        if not out_projected:
            if _native_prefill_fused_residual_gemm_enabled():
                x = _native_prefill_linear_add_residual(out, Ow, residual)
            else:
                out = F.linear(out, Ow)
                x = residual + out
        else:
            x = residual + out
        xpa[layer_idx] = next_xpa if use_sequence_attn_mix else h[:, -1, :].contiguous()
        state[layer_idx] = new_state.contiguous()

        residual = x
        h2 = F.layer_norm(x, [hidden], fn_w, fn_b, 1e-5)
        if use_prefill_sequence_ffn:
            assert sequence_ffn_blocks is not None
            assert sequence_ffn_launch is not None
            if sequence_ffn_workspace is None:
                sequence_ffn_workspace = (
                    torch.empty((B * T, hidden), device=h2.device, dtype=h2.dtype),
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
            if use_prefill_shift_mix and fused_ffn_sequence_shift_mix is not None:
                if sequence_ffn_mix_workspace is None:
                    sequence_ffn_mix_workspace = torch.empty_like(h2)
                fk, next_xpf = fused_ffn_sequence_shift_mix(
                    h2,
                    xpf[layer_idx],
                    fx_k,
                    workspace=sequence_ffn_mix_workspace,
                )
            else:
                prev_h2 = torch.cat([xpf[layer_idx].view(B, 1, hidden), h2[:, :-1, :]], dim=1)
                fxx = prev_h2 - h2
                fk = h2 + fxx * fx_k.view(1, 1, hidden)
                next_xpf = h2[:, -1, :].contiguous()
            fk = F.linear(fk, fK)
            if (
                use_prefill_shift_mix
                and fused_relu_square is not None
                and fused_relu_square_available is not None
                and fused_relu_square_available()
            ):
                fk = fused_relu_square(fk)
            else:
                fk = torch.relu(fk) ** 2
            if _native_prefill_fused_residual_gemm_enabled():
                x = _native_prefill_linear_add_residual(fk, fV, residual)
            else:
                x = residual + F.linear(fk, fV)
        xpf[layer_idx] = next_xpf

    x = F.layer_norm(x, [hidden], base.norm.weight, base.norm.bias, 1e-5)
    keep = T if logits_to_keep is None or int(logits_to_keep) <= 0 else min(int(logits_to_keep), T)
    logits = _lm_head(model, x[:, -keep:, :])
    setattr(model, "_rwkv7_native_prefill_stacked_rkv_effective", bool(stacked_rkv_used))
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


def _block_ip(x, state, xpa, xpf, v_first, p, sparse_ffn_out=None):
    """In-place (eager) block step for CUDA-graph capture: state/xpa/xpf/v_first
    are fixed buffers updated in place. Same math as block_step."""
    (i, H, N, eps, has_pre,
     pre_w, pre_b, an_w, an_b, fn_w, fn_b,
     x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
     Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
     gn_w, gn_b, fx_k, fK, fV, RKVw) = p
    residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5) if has_pre else x
    use_fused_norm_mix = _native_graph_fused_norm_mix_enabled()
    if use_fused_norm_mix:
        stack_rkv = _native_graph_vkwr_rkv_dispatch(1, H * N) and RKVw.numel() != 0
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
        h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)
        xx = xpa - h
        xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
        xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    v_gate = None
    v_mixed = False
    lora_dense = _graph_linears_are_dense(w1, w2, a1, a2, v1, v2, g1, g2)
    if _native_graph_fused_projection_enabled() and lora_dense and _graph_linears_are_dense(Rw, Kw, Vw):
        r, k, v, w, a, g, v_gate = fused_rkv_wavg_projection(
            xr.view(1, H * N),
            xk.view(1, H * N),
            xv.view(1, H * N),
            xw.view(1, H * N),
            xa.view(1, H * N),
            xg.view(1, H * N),
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
        r = r.view(H * N)
        k = k.view(H * N)
        v = v.view(H * N)
        w = w.view(H * N)
        a = torch.sigmoid(a.view(H * N))
        g = g.view(H * N)
        v_gate = torch.sigmoid(v_gate.view(H * N))
    elif i > 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        1,
        H * N,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0], _graph_linear_shape(v1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, H * N)
        w, a, g, v = ada_wagv_lora(
            xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2,
            w0, a0, v0, v, v_first, sigmoid_a=True,
        )
        v_mixed = True
    elif i == 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        1,
        H * N,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, H * N)
        w, a, g, _unused_v = ada_wagv_lora(
            xw, xa, xg, xg, w1, a1, g1, g1, w2, a2, g2, g2,
            w0, a0, a0, v, v, sigmoid_a=True, compute_v=False,
        )
    elif i > 0 and lora_dense and _native_graph_sm70_wagv_lora_enabled(1, H * N):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, H * N)
        w, a, g, v = sm70_wagv_lora(
            xw.view(1, H * N), xa.view(1, H * N), xg.view(1, H * N), xv.view(1, H * N),
            w1, a1, g1, v1, w2, a2, g2, v2, w0, a0, v0,
            v.view(1, H * N), v_first.view(1, H * N),
        )
        w = w.view(H * N); a = torch.sigmoid(a.view(H * N)); g = g.view(H * N); v = v.view(H * N)
        v_mixed = True
    elif lora_dense and _native_graph_fused_wavg_lora_enabled(1, H * N):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, H * N)
        if i == 0:
            w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
            a = a0 + F.linear(F.linear(xa, a1), a2)
            g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
        else:
            block_m, block_r, block_k = _native_graph_fused_wavg_lora_blocks()
            w, a, g, v_gate = fused_wavg_lora(
                xw.view(1, H * N),
                xa.view(1, H * N),
                xg.view(1, H * N),
                xv.view(1, H * N),
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
            w = w.view(H * N)
            a = a.view(H * N)
            g = g.view(H * N)
            v_gate = v_gate.view(H * N)
        a = torch.sigmoid(a)
    elif lora_dense and _native_graph_fused_wag_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, H * N)
        block_m, block_r, block_k = _native_graph_fused_wag_lora_blocks()
        w, a, g = fused_wag_lora(
            xw.view(1, H * N),
            xa.view(1, H * N),
            xg.view(1, H * N),
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
        w = w.view(H * N)
        a = torch.sigmoid(a.view(H * N))
        g = g.view(H * N)
    else:
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, H * N)
        w = _graph_linear_call_with_explicit_bias(torch.tanh(_graph_linear_call(xw, w1)), w2, w0)
        a = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xa, a1), a2, a0))
        g = _graph_linear_call(torch.sigmoid(_graph_linear_call(xg, g1)), g2)
    use_fused_recurrent_output = _native_graph_fused_recurrent_output_enabled()
    use_fused_recurrent_raw = use_fused_recurrent_output and _native_graph_fused_recurrent_raw_enabled()
    if not use_fused_recurrent_raw:
        kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(H * N)
        k = k * (1 + (a - 1) * k_a)
    if i == 0:
        v_first.copy_(v)
    elif not v_mixed:
        if v_gate is None:
            v_gate = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xv, v1), v2, v0))
        v = v + (v_first - v) * v_gate
    if use_fused_recurrent_raw:
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
        out = out.view(H * N)
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
        out = out.view(H * N)
        new_state = new_state.view(H, N, N)
    else:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
        out, new_state = _recurrent_update_unbatched(r, w, k, v, kk, a, state, H, N)
    if use_fused_recurrent_output:
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    elif _native_graph_fused_output_project_enabled() and _graph_linear_is_dense(Ow):
        out = fused_attn_output_project(
            out.view(1, H * N),
            r.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            g.view(1, H * N),
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
        ).view(H * N)
    elif _native_graph_fused_output_enabled():
        out = fused_attn_output_prepare(
            out.view(1, H * N),
            r.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            g.view(1, H * N),
            r_k,
            gn_w,
            gn_b,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
        ).view(H * N)
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    else:
        out = F.group_norm(out.view(1, H * N), H, gn_w, gn_b, eps).view(H * N)
        sk = (r.view(H, N) * k.view(H, N) * r_k).sum(dim=-1, keepdim=True)
        out = (out + (sk * v.view(H, N)).view(H * N)) * g
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    state.copy_(new_state)
    if use_fused_norm_mix:
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
        h2 = F.layer_norm(residual, [H * N], fn_w, fn_b, 1e-5)
        fxx = xpf - h2
        fk = h2 + fxx * fx_k
        xpf.copy_(h2)
    return _native_graph_ffn_dispatch(fk, fK, fV, residual, sparse_out=sparse_ffn_out)


def _block_ip_batched(x, state, xpa, xpf, v_first, p, sparse_ffn_out=None):
    """In-place batched block step for CUDA-graph capture.

    Shapes:
      x/xpa/xpf/v_first: [B, H*N]
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
    residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5) if has_pre else x
    use_fused_norm_mix = _native_graph_fused_norm_mix_enabled()
    if use_fused_norm_mix:
        stack_rkv = _native_graph_vkwr_rkv_dispatch(B, H * N) and RKVw.numel() != 0
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
        h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)
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
    elif i > 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        B,
        H * N,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0], _graph_linear_shape(v1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, H * N)
        w, a, g, v = ada_wagv_lora(
            xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2,
            w0, a0, v0, v, v_first, sigmoid_a=True,
        )
        v_mixed = True
    elif i == 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        B,
        H * N,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, H * N)
        w, a, g, _unused_v = ada_wagv_lora(
            xw, xa, xg, xg, w1, a1, g1, g1, w2, a2, g2, g2,
            w0, a0, a0, v, v, sigmoid_a=True, compute_v=False,
        )
    elif i > 0 and lora_dense and _native_graph_sm70_wagv_lora_enabled(B, H * N):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, H * N)
        w, a, g, v = sm70_wagv_lora(
            xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2, w0, a0, v0, v, v_first,
        )
        a = torch.sigmoid(a)
        v_mixed = True
    elif lora_dense and _native_graph_fused_wavg_lora_enabled(B, H * N):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, H * N)
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
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, H * N)
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
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, H * N)
        w = _graph_linear_call_with_explicit_bias(torch.tanh(_graph_linear_call(xw, w1)), w2, w0)
        a = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xa, a1), a2, a0))
        g = _graph_linear_call(torch.sigmoid(_graph_linear_call(xg, g1)), g2)
    use_fused_recurrent_output = _native_graph_fused_recurrent_output_enabled()
    use_fused_recurrent_raw = use_fused_recurrent_output and _native_graph_fused_recurrent_raw_enabled()
    if not use_fused_recurrent_raw:
        kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, H * N)
        k = k * (1 + (a - 1) * k_a)
    if i == 0:
        v_first.copy_(v)
    elif not v_mixed:
        if v_gate is None:
            v_gate = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xv, v1), v2, v0))
        v = v + (v_first - v) * v_gate
    if use_fused_recurrent_raw:
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
        out = out.reshape(B, H * N)
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
        out = out.reshape(B, H * N)
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
        out = F.group_norm(out, H, gn_w, gn_b, eps).view(B, H * N)
        sk = (r.view(B, H, N) * k.view(B, H, N) * r_k).sum(dim=-1, keepdim=True)
        out = (out + (sk * v.view(B, H, N)).view(B, H * N)) * g
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    state.copy_(new_state)
    if use_fused_norm_mix:
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
        h2 = F.layer_norm(residual, [H * N], fn_w, fn_b, 1e-5)
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
