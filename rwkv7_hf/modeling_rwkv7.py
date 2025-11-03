# coding=utf-8
"""Remote-code wrapper around FLA RWKV7 HF modules.

Requires flash-linear-attention (`fla`) on PYTHONPATH / installed in the env.
"""
from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

_FLA_IMPORT_ERROR: Exception | None = None

try:
    from .triton_compat import apply_runtime_compat as _rwkv7_apply_runtime_compat
except ImportError:  # pragma: no cover - direct remote-file execution fallback
    try:
        from triton_compat import apply_runtime_compat as _rwkv7_apply_runtime_compat
    except Exception:  # pragma: no cover - compatibility helper is optional
        _rwkv7_apply_runtime_compat = None
if _rwkv7_apply_runtime_compat is not None:
    _rwkv7_apply_runtime_compat()

# Keep native_quant_policy as a first-level remote-code dependency. Some
# Transformers trust_remote_code versions copy only direct relative imports into
# the module cache before scanning helper files such as native_quant_mm8/mm4.
try:  # pragma: no cover - packaging guard
    from .native_quant_policy import NATIVE_MM_POLICIES as _RWKV7_NATIVE_MM_POLICIES
except ImportError:  # pragma: no cover - direct remote-file execution fallback
    try:
        from native_quant_policy import NATIVE_MM_POLICIES as _RWKV7_NATIVE_MM_POLICIES
    except Exception:
        _RWKV7_NATIVE_MM_POLICIES = ("memory", "speed")

try:
    from fla.models.rwkv7.modeling_rwkv7 import RWKV7Model as _RWKV7Model
    from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM as _RWKV7ForCausalLM
    from fla.models.utils import Cache as _FLACache
    from fla.ops.rwkv7.fused_recurrent import fused_mul_recurrent_rwkv7
except Exception as exc:  # pragma: no cover - exercised by fla-free native backend tests
    _FLA_IMPORT_ERROR = exc
    from transformers.cache_utils import Cache as _FLACache
    from transformers.modeling_utils import PreTrainedModel

    class _MissingFLABase(PreTrainedModel):
        """Fallback base that keeps remote-code import alive without FLA."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "flash-linear-attention (`fla`) is required for the optimized "
                "RWKV7ForCausalLM wrapper. Set RWKV7_NATIVE_MODEL=1 to load the "
                "fla-free NativeRWKV7ForCausalLM backend instead."
            ) from _FLA_IMPORT_ERROR

    class _RWKV7Model(_MissingFLABase):
        pass

    class _RWKV7ForCausalLM(_MissingFLABase):
        pass

    def fused_mul_recurrent_rwkv7(*args, **kwargs):
        raise ImportError(
            "flash-linear-attention (`fla`) is required for fused RWKV-7 recurrent ops."
        ) from _FLA_IMPORT_ERROR

try:
    from .configuration_rwkv7 import RWKV7Config
except ImportError:  # pragma: no cover - direct remote-file execution fallback
    from configuration_rwkv7 import RWKV7Config

try:
    from .kernel_policy import current_kernel_policy, env_blocks, env_flag, env_int
except ImportError:  # pragma: no cover - direct remote-file execution fallback
    try:
        from kernel_policy import current_kernel_policy, env_blocks, env_flag, env_int
    except Exception:  # pragma: no cover - older converted model dirs
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

try:
    from .native_jit import block_step as _native_jit_block_step
    from .native_jit import block_step_batched as _native_jit_block_step_batched
    from .native_jit import extract as _native_jit_extract
    from .native_jit import extract_graph as _native_graph_extract
    from .native_jit import _block_ip as _native_graph_block_ip
    from .native_jit import _block_ip_batched as _native_graph_block_ip_batched
    from .native_jit import _native_graph_linear_dispatch
    from .native_jit import prefill as _native_jit_prefill
    from .native_jit import prewarm_ada_sparse_ffn as _native_graph_prewarm_sparse_ffn
except Exception:  # pragma: no cover - optional remote-code fast path
    try:
        from native_jit import block_step as _native_jit_block_step
        from native_jit import block_step_batched as _native_jit_block_step_batched
        from native_jit import extract as _native_jit_extract
        from native_jit import extract_graph as _native_graph_extract
        from native_jit import _block_ip as _native_graph_block_ip
        from native_jit import _block_ip_batched as _native_graph_block_ip_batched
        from native_jit import _native_graph_linear_dispatch
        from native_jit import prefill as _native_jit_prefill
        from native_jit import prewarm_ada_sparse_ffn as _native_graph_prewarm_sparse_ffn
    except Exception:
        _native_jit_block_step = None
        _native_jit_block_step_batched = None
        _native_jit_extract = None
        _native_graph_extract = None
        _native_graph_block_ip = None
        _native_graph_block_ip_batched = None
        _native_graph_linear_dispatch = None
        _native_jit_prefill = None
        _native_graph_prewarm_sparse_ffn = None

# HF dynamic-module discovery copies files referenced by direct relative
# imports in this top-level remote-code file.  The native backend reaches
# ``native.py`` through ``native_model.py``; keep an explicit non-executed edge
# here so fresh caches contain the whole native dependency set.
if False:  # pragma: no cover
    from .dplr_prefill import dplr_chunk_scan as _rwkv7_dplr_dependency_sentinel
    from .dplr_prefill_triton import dplr_chunk_scan_triton as _rwkv7_dplr_triton_dependency_sentinel
    from .fused_attention_projection import fused_rkv_wag_projection as _rwkv7_fused_attn_projection_dependency_sentinel
    from .fused_decode_norm_mix import fused_attn_norm_mix6_decode as _rwkv7_fused_decode_norm_mix_dependency_sentinel
    from .sm70_linear import sm70_linear as _rwkv7_sm70_linear_dependency_sentinel
    from .fused_lora import fused_wag_lora as _rwkv7_fused_lora_dependency_sentinel
    from .fused_output import fused_attn_output_prepare as _rwkv7_fused_output_dependency_sentinel
    from .fused_prefill import fused_prefill_state_prep as _rwkv7_fused_prefill_dependency_sentinel
    from .fused_recurrent_update import fused_recurrent_update as _rwkv7_fused_recurrent_dependency_sentinel
    from .fused_time_mix import fused_attn_shift_mix as _rwkv7_fused_time_mix_dependency_sentinel
    from .native import _init_state_batched as _rwkv7_native_dependency_sentinel


_FALSE_VALUES = {"0", "false", "False", "no", "off"}


def _rwkv7_kernel_policy():
    if current_kernel_policy is None:
        return None
    try:
        return current_kernel_policy(torch_module=torch)
    except Exception:
        return None


def _native_model_backend_requested() -> bool:
    raw = os.environ.get("RWKV7_NATIVE_MODEL")
    if raw is not None:
        return raw not in _FALSE_VALUES
    policy = _rwkv7_kernel_policy()
    profile = getattr(policy, "profile", None)
    family = getattr(profile, "family", None)
    # Some older CUDA families cannot reliably run the optimized FLA/Triton
    # RWKV-7 kernels. Route them to the pure PyTorch backend unless the user
    # overrides it.
    return family in {"pascal", "legacy_cuda"}


def _fast_cache_enabled() -> bool:
    """Runtime switch used by benchmarks to compare cache implementations."""
    policy = _rwkv7_kernel_policy()
    default = bool(getattr(policy, "fast_cache", True))
    return env_flag("RWKV7_FAST_CACHE", default)


def _fast_token_layout() -> str:
    """Select the experimental fast-token tensor layout for A/B benchmarks."""
    layout = os.environ.get("RWKV7_FAST_TOKEN_LAYOUT", "3d").strip().lower()
    return "2d" if layout in {"2d", "flat"} else "3d"


def _normalize_fast_token_backend(backend: str | None) -> str:
    backend = (backend or "auto").strip().lower()
    if backend in {"", "auto", "best"}:
        return "auto"
    if backend in {"native_graph", "cuda_graph", "graph"}:
        return "native_graph"
    return "native_jit" if backend in {"native", "native_jit", "jit"} else "fla"


def _fast_token_backend() -> str:
    """Select the fast-token implementation backend."""
    return _normalize_fast_token_backend(os.environ.get("RWKV7_FAST_TOKEN_BACKEND", "auto"))


def _fast_forward_enabled() -> bool:
    """Allow normal HF forward/generate to use the one-token fast path."""
    return os.environ.get("RWKV7_FAST_FORWARD", "1") not in _FALSE_VALUES


def _fast_forward_quant_enabled() -> bool:
    """Allow quantized HF modules to use the FLA fast-token fallback."""
    return os.environ.get("RWKV7_FAST_FORWARD_QUANT", "1") not in _FALSE_VALUES


def _fast_prefill_enabled() -> bool:
    """Allow normal HF forward/generate to use the native prefill path."""

    policy = _rwkv7_kernel_policy()
    return env_flag("RWKV7_FAST_PREFILL", bool(getattr(policy, "fast_prefill", False)))


def _bnb_skip_policy(policy: str | None = None) -> str:
    if policy is None:
        env_policy = os.environ.get("RWKV7_BNB_SKIP_POLICY")
        if env_policy is None:
            kernel_policy = _rwkv7_kernel_policy()
            env_policy = str(getattr(kernel_policy, "bnb_skip_policy", "memory"))
        policy = env_policy
    policy = str(policy).strip().lower()
    if policy in {"", "default", "small_lora", "memory", "minimal"}:
        return "memory"
    if policy in {"decode", "decode_hot", "hot", "hybrid"}:
        return "decode_hot"
    if policy in {"dense", "all_dense", "no_quant"}:
        return "dense"
    return "memory"


def _cuda_available() -> bool:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(callable(is_available) and is_available())


def _native_graph_cache_size() -> int:
    """Maximum per-model native graph runners to keep for dynamic serving."""
    raw = os.environ.get("RWKV7_NATIVE_GRAPH_CACHE_SIZE", "8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def _native_prefill_graph_enabled() -> bool:
    """Capture fixed-shape inference prefill as one CUDA graph."""

    policy = _rwkv7_kernel_policy()
    return env_flag("RWKV7_NATIVE_PREFILL_GRAPH", bool(getattr(policy, "prefill_graph", False)))


def _native_prefill_graph_cache_size() -> int:
    """Maximum fixed-shape prefill graphs retained by one model."""

    return env_int("RWKV7_NATIVE_PREFILL_GRAPH_CACHE_SIZE", 2, lower=1, upper=16)


def _native_prefill_graph_signature() -> tuple[tuple[str, str | None], ...]:
    """Capture-affecting environment used in the prefill graph cache key."""

    names = (
        "RWKV7_NATIVE_PREFILL_DPLR_SCAN",
        "RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN",
        "RWKV7_NATIVE_PREFILL_FUSED_OUTPUT",
        "RWKV7_NATIVE_PREFILL_FUSED_OUTPUT_PROJECT",
        "RWKV7_NATIVE_PREFILL_FUSED_SCAN",
        "RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT",
        "RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX",
        "RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP",
        "RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN",
        "RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA",
        "RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M",
        "RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS",
        "RWKV7_NATIVE_PREFILL_STATE_PREP_W_DTYPE",
    )
    return tuple((name, os.environ.get(name)) for name in names)


def _native_graph_fused_recurrent_requested() -> bool:
    """Whether native-graph runners should capture the experimental recurrent kernel."""

    policy = _rwkv7_kernel_policy()
    default = bool(getattr(policy, "fused_recurrent", False))
    return env_flag("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT", default)


def _native_graph_fused_recurrent_output_requested() -> bool:
    """Whether native-graph runners should capture recurrent+output-prep fusion."""

    policy = _rwkv7_kernel_policy()
    default = bool(getattr(policy, "fused_recurrent_output", True))
    return env_flag("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT", default)


def _native_graph_fused_recurrent_raw_requested() -> bool:
    """Whether W decay and K/KK preparation are folded into recurrence."""

    policy = _rwkv7_kernel_policy()
    default = bool(getattr(policy, "fused_recurrent_raw", False))
    return env_flag("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW", default)


def _native_graph_fused_output_requested() -> bool:
    """Whether native-graph runners should capture the experimental output-prep kernel."""

    policy = _rwkv7_kernel_policy()
    default = bool(getattr(policy, "fused_output", True))
    return env_flag("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT", default)


def _native_graph_fused_output_project_requested() -> bool:
    """Whether native-graph runners should capture fused output-prep plus ``o_proj``."""

    policy = _rwkv7_kernel_policy()
    default = bool(getattr(policy, "fused_output_project", False))
    return env_flag("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT", default)


def _native_graph_fused_output_project_block_m() -> int:
    policy = _rwkv7_kernel_policy()
    default = int(getattr(policy, "output_project_block_m", 16))
    return env_int("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT_BLOCK_M", default, lower=1, upper=128)


def _native_graph_fused_projection_requested() -> bool:
    """Whether native-graph runners should capture the experimental projection kernel."""

    policy = _rwkv7_kernel_policy()
    default = bool(getattr(policy, "fused_projection", False))
    return env_flag("RWKV7_NATIVE_GRAPH_FUSED_PROJECTION", default)


def _native_graph_fused_wag_lora_requested() -> bool:
    """Whether native-graph runners should capture the W/A/G LoRA fusion probe."""

    policy = _rwkv7_kernel_policy()
    default = bool(getattr(policy, "fused_wag_lora", False))
    return env_flag("RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA", default)


def _native_graph_fused_wavg_lora_requested() -> bool:
    """Whether native-graph runners should capture the W/A/G/V-gate LoRA probe."""

    policy = _rwkv7_kernel_policy()
    default = bool(getattr(policy, "fused_wavg_lora", False))
    return env_flag("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA", default)


def _native_graph_fused_wavg_lora_bsz1_max_hidden() -> int:
    policy = _rwkv7_kernel_policy()
    default = getattr(policy, "wavg_lora_bsz1_max_hidden", None)
    return env_int(
        "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BSZ1_MAX_HIDDEN",
        0 if default is None else int(default),
        lower=0,
    )


def _native_graph_fused_norm_mix_requested() -> bool:
    policy = _rwkv7_kernel_policy()
    default = bool(getattr(policy, "fused_norm_mix", False))
    return env_flag("RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX", default)


def _native_graph_fused_norm_mix_num_warps() -> int:
    policy = _rwkv7_kernel_policy()
    default = int(getattr(policy, "norm_mix_num_warps", 4))
    value = env_int("RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS", default, lower=1, upper=8)
    if value not in {1, 2, 4, 8}:
        raise ValueError(f"RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS must be one of 1, 2, 4, or 8; got {value}")
    return value


def _native_graph_sm70_linear_requested() -> bool:
    policy = _rwkv7_kernel_policy()
    return env_flag("RWKV7_NATIVE_GRAPH_SM70_LINEAR", bool(getattr(policy, "sm70_linear", False)))


def _native_graph_ada_linear_requested() -> bool:
    policy = _rwkv7_kernel_policy()
    return env_flag("RWKV7_NATIVE_GRAPH_ADA_LINEAR", bool(getattr(policy, "ada_linear", False)))


def _native_graph_ada_linear_signature() -> tuple[str, str]:
    policy = _rwkv7_kernel_policy()
    return (
        os.environ.get("RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROWS", str(getattr(policy, "ada_linear_rows", "2 4"))),
        os.environ.get("RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROLES", "auto"),
    )


def _native_graph_ada_wagv_lora_requested() -> bool:
    policy = _rwkv7_kernel_policy()
    return env_flag(
        "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA", bool(getattr(policy, "ada_wagv_lora", False))
    )


def _native_graph_ada_sparse_ffn_requested() -> bool:
    policy = _rwkv7_kernel_policy()
    return env_flag(
        "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN", bool(getattr(policy, "ada_sparse_ffn", False))
    )


def _native_graph_ada_sparse_ffn_signature() -> tuple[int, bool]:
    policy = _rwkv7_kernel_policy()
    raw_rows = os.environ.get(
        "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_MAX_ROWS",
        str(getattr(policy, "ada_sparse_ffn_max_rows", 19)),
    )
    try:
        max_rows = min(19, max(1, int(raw_rows)))
    except ValueError:
        max_rows = int(getattr(policy, "ada_sparse_ffn_max_rows", 19))
    inplace = env_flag(
        "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_INPLACE",
        bool(getattr(policy, "ada_sparse_ffn_inplace", False)),
    )
    return max_rows, inplace


def _native_graph_rkv_policy() -> str:
    """Cache-key visible policy for VKWR-inspired stacked R/K/V projection."""

    policy = _rwkv7_kernel_policy()
    default = str(getattr(policy, "rkv_policy", "manual"))
    raw = os.environ.get("RWKV7_NATIVE_GRAPH_RKV_POLICY", default).strip().lower()
    if raw in {"", "manual", "explicit", "env"}:
        return "manual"
    if raw in {"0", "false", "no", "off", "disabled"}:
        return "off"
    if raw in {"vkwr", "vkwr_auto", "auto", "stacked", "bmm"}:
        return "vkwr_auto"
    return "manual"


def _native_graph_vkwr_rkv_thresholds() -> tuple[int, int]:
    """Return ``(min_hidden, max_rows)`` used by the opt-in RKV policy."""

    vals = []
    for name, default, lower, upper in (
        ("RWKV7_NATIVE_GRAPH_RKV_MIN_HIDDEN", 1, 1, None),
        ("RWKV7_NATIVE_GRAPH_RKV_MAX_ROWS", 64, 1, 4096),
    ):
        raw = os.environ.get(name, str(default)).strip()
        try:
            val = int(raw)
        except ValueError:
            val = default
        val = max(lower, val)
        if upper is not None:
            val = min(upper, val)
        vals.append(val)
    return vals[0], vals[1]


def _native_graph_fused_wag_lora_blocks() -> tuple[int, int, int]:
    policy = _rwkv7_kernel_policy()
    defaults = tuple(getattr(policy, "wag_lora_blocks", (64, 64, 64)))
    return env_blocks(
        ("RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_M", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_R", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_K"),
        defaults,  # type: ignore[arg-type]
        (128, 128, 256),
    )


def _native_graph_fused_wavg_lora_blocks() -> tuple[int, int, int]:
    policy = _rwkv7_kernel_policy()
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


def _native_graph_stats_template() -> dict[str, int]:
    return {"requests": 0, "hits": 0, "misses": 0, "evictions": 0}


def _linear_direct(module, x: torch.Tensor) -> torch.Tensor:
    """Call a Linear module through F.linear to skip small-module dispatch."""
    if type(module) is not torch.nn.Linear:
        return module(x)
    return F.linear(x, module.weight, module.bias)


def _native_graph_head_linear(module, x: torch.Tensor) -> torch.Tensor:
    """Native-graph lm_head with an optional measured sm_70 bsz=1 route."""

    if type(module) is not torch.nn.Linear or module.bias is not None:
        return module(x)
    if _native_graph_linear_dispatch is None:
        return F.linear(x, module.weight)
    return _native_graph_linear_dispatch(x, module.weight, role="head")


def _native_graph_head_linear_into(module, x: torch.Tensor, out: torch.Tensor) -> bool:
    """Let optional packed heads write directly into the graph output buffer."""

    forward_into = getattr(module, "rwkv7_forward_into", None)
    if not callable(forward_into):
        return False
    forward_into(x, out)
    return True


def _lora_direct(module, x: torch.Tensor) -> torch.Tensor:
    """Fast-path FLA LoRA forward used only by inference decode helpers."""
    h = _linear_direct(module.lora[0], x)
    h = module.lora[1](h)
    return _linear_direct(module.lora[2], h)


def _squeeze_token_dim(x: torch.Tensor) -> torch.Tensor:
    """Return `[batch, hidden]` for single-token `[batch, 1, hidden]` tensors."""
    if x.dim() == 3:
        if x.shape[1] != 1:
            raise ValueError("fast-token 2d layout only supports a single sequence position")
        return x[:, 0]
    return x


def _move_first_dim(value: Any, indices: torch.LongTensor) -> Any:
    """Reorder nested tensor state along batch dimension for HF beam helpers."""
    if isinstance(value, torch.Tensor):
        return value.index_select(0, indices.to(value.device))
    if isinstance(value, tuple):
        return tuple(_move_first_dim(v, indices) for v in value)
    if isinstance(value, list):
        return [_move_first_dim(v, indices) for v in value]
    if isinstance(value, dict):
        return {k: _move_first_dim(v, indices) for k, v in value.items()}
    return value


def _clone_cache_value(value: Any) -> Any:
    """Clone nested cache containers without assuming a fixed FLA layout."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_cache_value(v) for v in value)
    if isinstance(value, list):
        return [_clone_cache_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_cache_value(v) for k, v in value.items()}
    return value


def _same_tensor_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return True when two tensor views already represent the same storage slice."""
    try:
        return (
            a.data_ptr() == b.data_ptr()
            and a.storage_offset() == b.storage_offset()
            and tuple(a.shape) == tuple(b.shape)
            and tuple(a.stride()) == tuple(b.stride())
            and a.dtype == b.dtype
            and a.device == b.device
        )
    except Exception:
        return False


def _detach_cache_value(value: Any) -> Any:
    """Detach nested cache tensors while preserving the container layout."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_cache_value(v) for v in value)
    if isinstance(value, list):
        return [_detach_cache_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _detach_cache_value(v) for k, v in value.items()}
    return value


def _to_cache_value(
    value: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    non_blocking: bool = False,
    copy: bool = False,
) -> Any:
    """Move/cast nested cache tensors for CPU offload or device restore."""
    if isinstance(value, torch.Tensor):
        target_dtype = dtype if dtype is not None and value.is_floating_point() else None
        return value.to(device=device, dtype=target_dtype, non_blocking=non_blocking, copy=copy)
    if isinstance(value, tuple):
        return tuple(_to_cache_value(v, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy) for v in value)
    if isinstance(value, list):
        return [_to_cache_value(v, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy) for v in value]
    if isinstance(value, dict):
        return {k: _to_cache_value(v, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy) for k, v in value.items()}
    return value


def _first_tensor_batch_size(value: Any) -> int | None:
    """Return the leading dimension of the first tensor in a nested cache."""
    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.dim() > 0 else None
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    return None


class _RWKV7NativeGraphTokenRunner:
    """CUDA-graph replay helper for bsz=1 native fast-token decode.

    The public cache remains in FLA layout. Each replay copies the current cache
    into fixed native-layout graph buffers, replays one token, then rebinds the
    cache tensors to graph buffer views so callers can keep using the same
    `RWKV7StateCache` object or fall back to the normal HF path.
    """

    def __init__(self, owner: "RWKV7ForCausalLM", packs) -> None:
        if _native_graph_block_ip is None:
            raise RuntimeError("native_graph fast-token backend is unavailable; copy native_jit.py into the model repo")
        if not torch.cuda.is_available():
            raise RuntimeError("native_graph fast-token backend requires CUDA")
        base = owner.model
        self.packs = packs
        self.device = base.embeddings.weight.device
        if self.device.type != "cuda":
            raise RuntimeError("native_graph fast-token backend requires CUDA model weights")
        self.dtype = base.embeddings.weight.dtype
        self.hidden = int(packs[0][1] * packs[0][2])
        self.num_layers = len(packs)
        self.state = [
            torch.zeros(int(p[1]), int(p[2]), int(p[2]), device=self.device, dtype=torch.float32)
            for p in packs
        ]
        self.xpa = [torch.zeros(self.hidden, device=self.device, dtype=self.dtype) for _ in packs]
        self.xpf = [torch.zeros(self.hidden, device=self.device, dtype=self.dtype) for _ in packs]
        # Sparse FFN uses atomic accumulation. Keep one stable destination per
        # layer/runner so independently captured batch graphs never share an
        # allocator-owned intermediate buffer.
        self.sparse_ffn_out = [
            torch.empty(self.hidden, device=self.device, dtype=self.dtype) for _ in packs
        ]
        self.v_first = torch.zeros(self.hidden, device=self.device, dtype=self.dtype)
        self.tok_id = torch.zeros(1, dtype=torch.long, device=self.device)
        self.emb = base.embeddings.weight
        self.head_module = owner.lm_head
        self.vocab_size = int(getattr(self.head_module, "out_features", base.embeddings.weight.shape[0]))
        self.logits = torch.zeros(self.vocab_size, device=self.device, dtype=self.dtype)
        self.norm_w = base.norm.weight
        self.norm_b = base.norm.bias
        self._bound_cache_ref: weakref.ReferenceType[RWKV7StateCache] | None = None
        self.copy_from_cache_calls = 0
        self.copy_from_cache_fast_skips = 0
        self.bind_cache_calls = 0
        self.bind_cache_fast_skips = 0
        self.graph = None
        self._capture()

    def _one_step(self) -> None:
        x = F.embedding(self.tok_id, self.emb).reshape(self.hidden)
        for li, p in enumerate(self.packs):
            x = _native_graph_block_ip(
                x, self.state[li], self.xpa[li], self.xpf[li], self.v_first, p,
                self.sparse_ffn_out[li],
            )
        out = F.layer_norm(x, [self.hidden], self.norm_w, self.norm_b, 1e-5)
        if not _native_graph_head_linear_into(self.head_module, out, self.logits):
            self.logits.copy_(_native_graph_head_linear(self.head_module, out).reshape(-1))

    def _capture(self) -> None:
        warm = torch.cuda.Stream(device=self.device)
        warm.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warm):
            with torch.no_grad():
                for _ in range(3):
                    self._one_step()
        torch.cuda.current_stream(self.device).wait_stream(warm)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._one_step()

    @staticmethod
    def _copy_cache_tensor(dst: torch.Tensor, value: torch.Tensor | None, *, transpose_last: bool = False) -> None:
        if value is None:
            dst.zero_()
            return
        src = value
        if src.dim() >= 1 and src.shape[0] == 1:
            src = src.squeeze(0)
        if transpose_last:
            src = src.transpose(-1, -2)
        if _same_tensor_view(dst, src):
            return
        src = src.to(device=dst.device, dtype=dst.dtype)
        if _same_tensor_view(dst, src):
            return
        dst.copy_(src.contiguous())

    def copy_from_cache(self, past_key_values: "RWKV7StateCache") -> None:
        self.copy_from_cache_calls += 1
        if past_key_values._native_graph_bound_to(self):
            self.copy_from_cache_fast_skips += 1
            return
        self._detach_bound_cache_if_different(past_key_values)
        for li, p in enumerate(self.packs):
            layer_idx = int(p[0])
            state = past_key_values._ensure_layer(layer_idx)
            self._copy_cache_tensor(self.state[li], state.get("recurrent_state"), transpose_last=True)
            self._copy_cache_tensor(self.xpa[li], state.get("conv_state"))
            self._copy_cache_tensor(self.xpf[li], state.get("ffn_state"))

    def _detach_bound_cache_if_different(self, past_key_values: "RWKV7StateCache") -> None:
        ref = self._bound_cache_ref
        previous = ref() if ref is not None else None
        if previous is None or previous is past_key_values:
            return
        for li, p in enumerate(self.packs):
            layer_idx = int(p[0])
            if layer_idx >= len(previous.states):
                continue
            state = previous.states[layer_idx]
            if not isinstance(state, dict):
                continue
            recurrent_view = self.state[li].transpose(-1, -2).unsqueeze(0)
            conv_view = self.xpa[li].unsqueeze(0)
            ffn_view = self.xpf[li].unsqueeze(0)
            if isinstance(state.get("recurrent_state"), torch.Tensor) and _same_tensor_view(state["recurrent_state"], recurrent_view):
                state["recurrent_state"] = state["recurrent_state"].contiguous()
            if isinstance(state.get("conv_state"), torch.Tensor) and _same_tensor_view(state["conv_state"], conv_view):
                state["conv_state"] = state["conv_state"].clone()
            if isinstance(state.get("ffn_state"), torch.Tensor) and _same_tensor_view(state["ffn_state"], ffn_view):
                state["ffn_state"] = state["ffn_state"].clone()
        if hasattr(previous, "_invalidate_native_graph_binding"):
            previous._invalidate_native_graph_binding()
        self._bound_cache_ref = None

    def bind_cache(self, past_key_values: "RWKV7StateCache") -> None:
        self.bind_cache_calls += 1
        if past_key_values._native_graph_bound_to(self):
            self.bind_cache_fast_skips += 1
            return
        for li, p in enumerate(self.packs):
            layer_idx = int(p[0])
            state = past_key_values._ensure_layer(layer_idx)
            # FLA cache layout is transposed relative to the native matmul layout.
            state["recurrent_state"] = self.state[li].transpose(-1, -2).unsqueeze(0)
            state["conv_state"] = self.xpa[li].unsqueeze(0)
            state["ffn_state"] = self.xpf[li].unsqueeze(0)
            state["attn_state"] = None
        self._bound_cache_ref = weakref.ref(past_key_values)
        past_key_values._bind_native_graph_runner(self)

    def copy_stats(self) -> dict[str, int]:
        return {
            "copy_from_cache_calls": int(self.copy_from_cache_calls),
            "copy_from_cache_fast_skips": int(self.copy_from_cache_fast_skips),
            "bind_cache_calls": int(self.bind_cache_calls),
            "bind_cache_fast_skips": int(self.bind_cache_fast_skips),
        }

    def reorder_batch_inplace(self, indices: torch.LongTensor) -> bool:
        if int(indices.numel()) != 1:
            return False
        return True

    def replay(self, token: torch.LongTensor, past_key_values: "RWKV7StateCache") -> torch.Tensor:
        self.copy_from_cache(past_key_values)
        self.tok_id.copy_(token.reshape(1))
        self.graph.replay()
        self.bind_cache(past_key_values)
        return self.logits.view(1, 1, -1).clone()


class _RWKV7NativeGraphBatchedTokenRunner:
    """CUDA-graph replay helper for fixed-batch native fast-token decode."""

    def __init__(self, owner: "RWKV7ForCausalLM", packs, batch_size: int) -> None:
        if _native_graph_block_ip_batched is None:
            raise RuntimeError("native_graph batched fast-token backend is unavailable; copy native_jit.py into the model repo")
        if not torch.cuda.is_available():
            raise RuntimeError("native_graph fast-token backend requires CUDA")
        base = owner.model
        self.packs = packs
        self.batch_size = int(batch_size)
        self.device = base.embeddings.weight.device
        if self.device.type != "cuda":
            raise RuntimeError("native_graph fast-token backend requires CUDA model weights")
        self.dtype = base.embeddings.weight.dtype
        self.hidden = int(packs[0][1] * packs[0][2])
        self.state = [
            torch.zeros(self.batch_size, int(p[1]), int(p[2]), int(p[2]), device=self.device, dtype=torch.float32)
            for p in packs
        ]
        self.xpa = [torch.zeros(self.batch_size, self.hidden, device=self.device, dtype=self.dtype) for _ in packs]
        self.xpf = [torch.zeros(self.batch_size, self.hidden, device=self.device, dtype=self.dtype) for _ in packs]
        self.sparse_ffn_out = [
            torch.empty(self.batch_size, self.hidden, device=self.device, dtype=self.dtype)
            for _ in packs
        ]
        self.v_first = torch.zeros(self.batch_size, self.hidden, device=self.device, dtype=self.dtype)
        self.tok_id = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.emb = base.embeddings.weight
        self.head_module = owner.lm_head
        self.vocab_size = int(getattr(self.head_module, "out_features", base.embeddings.weight.shape[0]))
        self.logits = torch.zeros(self.batch_size, self.vocab_size, device=self.device, dtype=self.dtype)
        self.norm_w = base.norm.weight
        self.norm_b = base.norm.bias
        self._bound_cache_ref: weakref.ReferenceType[RWKV7StateCache] | None = None
        self.copy_from_cache_calls = 0
        self.copy_from_cache_fast_skips = 0
        self.bind_cache_calls = 0
        self.bind_cache_fast_skips = 0
        self.graph = None
        self._capture()

    def _one_step(self) -> None:
        x = F.embedding(self.tok_id, self.emb).reshape(self.batch_size, self.hidden)
        for li, p in enumerate(self.packs):
            x = _native_graph_block_ip_batched(
                x, self.state[li], self.xpa[li], self.xpf[li], self.v_first, p,
                self.sparse_ffn_out[li],
            )
        out = F.layer_norm(x, [self.hidden], self.norm_w, self.norm_b, 1e-5)
        if not _native_graph_head_linear_into(self.head_module, out, self.logits):
            self.logits.copy_(_native_graph_head_linear(self.head_module, out).reshape(self.batch_size, self.vocab_size))

    def _capture(self) -> None:
        warm = torch.cuda.Stream(device=self.device)
        warm.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warm):
            with torch.no_grad():
                for _ in range(3):
                    self._one_step()
        torch.cuda.current_stream(self.device).wait_stream(warm)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._one_step()

    @staticmethod
    def _copy_cache_tensor(dst: torch.Tensor, value: torch.Tensor | None, *, transpose_last: bool = False) -> None:
        if value is None:
            dst.zero_()
            return
        src = value
        if transpose_last:
            src = src.transpose(-1, -2)
        if _same_tensor_view(dst, src):
            return
        src = src.to(device=dst.device, dtype=dst.dtype)
        if _same_tensor_view(dst, src):
            return
        dst.copy_(src.contiguous())

    def copy_from_cache(self, past_key_values: "RWKV7StateCache") -> None:
        self.copy_from_cache_calls += 1
        if past_key_values._native_graph_bound_to(self):
            self.copy_from_cache_fast_skips += 1
            return
        self._detach_bound_cache_if_different(past_key_values)
        for li, p in enumerate(self.packs):
            layer_idx = int(p[0])
            state = past_key_values._ensure_layer(layer_idx)
            self._copy_cache_tensor(self.state[li], state.get("recurrent_state"), transpose_last=True)
            self._copy_cache_tensor(self.xpa[li], state.get("conv_state"))
            self._copy_cache_tensor(self.xpf[li], state.get("ffn_state"))

    def _detach_bound_cache_if_different(self, past_key_values: "RWKV7StateCache") -> None:
        ref = self._bound_cache_ref
        previous = ref() if ref is not None else None
        if previous is None or previous is past_key_values:
            return
        for li, p in enumerate(self.packs):
            layer_idx = int(p[0])
            if layer_idx >= len(previous.states):
                continue
            state = previous.states[layer_idx]
            if not isinstance(state, dict):
                continue
            recurrent_view = self.state[li].transpose(-1, -2)
            conv_view = self.xpa[li]
            ffn_view = self.xpf[li]
            if isinstance(state.get("recurrent_state"), torch.Tensor) and _same_tensor_view(state["recurrent_state"], recurrent_view):
                state["recurrent_state"] = state["recurrent_state"].contiguous()
            if isinstance(state.get("conv_state"), torch.Tensor) and _same_tensor_view(state["conv_state"], conv_view):
                state["conv_state"] = state["conv_state"].clone()
            if isinstance(state.get("ffn_state"), torch.Tensor) and _same_tensor_view(state["ffn_state"], ffn_view):
                state["ffn_state"] = state["ffn_state"].clone()
        if hasattr(previous, "_invalidate_native_graph_binding"):
            previous._invalidate_native_graph_binding()
        self._bound_cache_ref = None

    def bind_cache(self, past_key_values: "RWKV7StateCache") -> None:
        self.bind_cache_calls += 1
        if past_key_values._native_graph_bound_to(self):
            self.bind_cache_fast_skips += 1
            return
        for li, p in enumerate(self.packs):
            layer_idx = int(p[0])
            state = past_key_values._ensure_layer(layer_idx)
            # FLA cache layout is transposed relative to the native matmul layout.
            state["recurrent_state"] = self.state[li].transpose(-1, -2)
            state["conv_state"] = self.xpa[li]
            state["ffn_state"] = self.xpf[li]
            state["attn_state"] = None
        self._bound_cache_ref = weakref.ref(past_key_values)
        past_key_values._bind_native_graph_runner(self)

    def copy_stats(self) -> dict[str, int]:
        return {
            "copy_from_cache_calls": int(self.copy_from_cache_calls),
            "copy_from_cache_fast_skips": int(self.copy_from_cache_fast_skips),
            "bind_cache_calls": int(self.bind_cache_calls),
            "bind_cache_fast_skips": int(self.bind_cache_fast_skips),
        }

    def reorder_batch_inplace(self, indices: torch.LongTensor) -> bool:
        """Reorder captured state buffers without breaking cache binding."""

        if int(indices.numel()) != self.batch_size:
            return False
        idx = indices.to(device=self.device, dtype=torch.long)
        for li in range(len(self.packs)):
            self.state[li].copy_(self.state[li].index_select(0, idx).contiguous())
            self.xpa[li].copy_(self.xpa[li].index_select(0, idx).contiguous())
            self.xpf[li].copy_(self.xpf[li].index_select(0, idx).contiguous())
        self.v_first.copy_(self.v_first.index_select(0, idx).contiguous())
        return True

    def replay(self, token: torch.LongTensor, past_key_values: "RWKV7StateCache") -> torch.Tensor:
        if int(token.numel()) != self.batch_size:
            raise ValueError(f"native_graph runner batch mismatch: got {int(token.numel())}, expected {self.batch_size}")
        self.copy_from_cache(past_key_values)
        self.tok_id.copy_(token.reshape(self.batch_size))
        self.graph.replay()
        self.bind_cache(past_key_values)
        return self.logits.view(self.batch_size, 1, -1).clone()


class RWKV7StateCache(_FLACache):
    """Lightweight recurrent-state cache for RWKV-7 inference.

    FLA's default cache mirrors the evolving Transformers CacheLayer API and is
    intentionally generic. RWKV-7 decode only needs one dictionary per layer
    (`recurrent_state`, `conv_state`, `ffn_state`, and optional `attn_state`), so
    this cache keeps the legacy list-of-dicts layout while still subclassing the
    FLA `Cache` class. That makes it accepted by FLA layers without a conversion
    step and removes per-token CacheLayer bookkeeping from the hot path.
    """

    is_compileable = True

    def __init__(self, seen_tokens: int = 0, **_: Any) -> None:
        # Do not call _FLACache.__init__(): it allocates HF CacheLayer wrappers
        # that are unnecessary for RWKV recurrent decode and add CPU overhead.
        self.states: list[dict[str, Any]] = []
        self._seen_tokens = int(seen_tokens)
        self._rwkv7_cache_metrics: dict[str, int] = {
            "updates": 0,
            "new_layers": 0,
            "clones": 0,
            "detaches": 0,
            "device_moves": 0,
            "select_batch_calls": 0,
            "native_graph_bound_selects": 0,
            "batch_select_calls": 0,
            "reorder_calls": 0,
            "resets": 0,
        }
        self._rwkv7_cache_version = 0
        self._rwkv7_native_graph_bound_runner_id: int | None = None
        self._rwkv7_native_graph_bound_version: int | None = None
        self._rwkv7_native_graph_bound_runner_ref: weakref.ReferenceType | None = None

    def _invalidate_native_graph_binding(self) -> None:
        """Mark native-graph runner bindings stale after cache mutations."""

        self._rwkv7_cache_version += 1
        self._rwkv7_native_graph_bound_runner_id = None
        self._rwkv7_native_graph_bound_version = None
        self._rwkv7_native_graph_bound_runner_ref = None

    def _bind_native_graph_runner(self, runner: object) -> None:
        self._rwkv7_native_graph_bound_runner_id = id(runner)
        self._rwkv7_native_graph_bound_version = int(self._rwkv7_cache_version)
        try:
            self._rwkv7_native_graph_bound_runner_ref = weakref.ref(runner)
        except TypeError:
            self._rwkv7_native_graph_bound_runner_ref = None

    def _native_graph_bound_to(self, runner: object) -> bool:
        return (
            self._rwkv7_native_graph_bound_runner_id == id(runner)
            and self._rwkv7_native_graph_bound_version == int(self._rwkv7_cache_version)
        )

    def _native_graph_bound_runner(self) -> object | None:
        if self._rwkv7_native_graph_bound_version != int(self._rwkv7_cache_version):
            return None
        ref = self._rwkv7_native_graph_bound_runner_ref
        runner = ref() if ref is not None else None
        if runner is not None and self._rwkv7_native_graph_bound_runner_id == id(runner):
            return runner
        return None

    def _ensure_layer(self, layer_idx: int) -> dict[str, Any]:
        empty = {"recurrent_state": None, "attn_state": None, "conv_state": None, "ffn_state": None}
        while len(self.states) <= layer_idx:
            self.states.append(dict(empty))
        if self.states[layer_idx] is None:
            self.states[layer_idx] = dict(empty)
        return self.states[layer_idx]

    def __getitem__(self, layer_idx: int) -> dict[str, Any]:
        if layer_idx < len(self.states):
            return self.states[layer_idx]
        raise KeyError(f"Cache only has {len(self.states)} layers, attempted to access layer {layer_idx}")

    def __iter__(self):
        yield from self.states

    def __len__(self) -> int:
        return len(self.states)

    def update(
        self,
        recurrent_state: Any | None = None,
        attn_state: Any | None = None,
        conv_state: Any | None = None,
        ffn_state: Any | None = None,
        layer_idx: int = 0,
        offset: int | None = 1,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._invalidate_native_graph_binding()
        if cache_kwargs is None:
            cache_kwargs = {}
        offset = 1 if offset is None else int(offset)
        input_size = attn_state[0].shape[1] if attn_state is not None else 0
        window_size = cache_kwargs.get("window_size")

        if len(self.states) <= layer_idx:
            while len(self.states) < layer_idx:
                self.states.append({"recurrent_state": None, "attn_state": None, "conv_state": None, "ffn_state": None})
            if layer_idx == 0:
                self._seen_tokens += offset
            if attn_state is not None and window_size is not None and input_size > window_size:
                attn_state = [state[:, -window_size:].contiguous() for state in attn_state]
            state = {
                "recurrent_state": recurrent_state,
                "attn_state": attn_state,
                "conv_state": conv_state,
                "ffn_state": ffn_state,
            }
            self.states.append(state)
            self._rwkv7_cache_metrics["updates"] += 1
            self._rwkv7_cache_metrics["new_layers"] += 1
            return state

        state = self.states[layer_idx]
        if layer_idx == len(self.states) - 1:
            self._seen_tokens += offset
        if recurrent_state is not None:
            state["recurrent_state"] = recurrent_state
        if attn_state is not None:
            if state.get("attn_state") is None:
                state["attn_state"] = [
                    new_state[:, -window_size:].contiguous()
                    if window_size is not None and new_state.shape[1] > window_size
                    else new_state
                    for new_state in attn_state
                ]
            elif window_size is not None and input_size == 0:
                pass
            elif window_size is not None and state["attn_state"][0].shape[1] >= window_size:
                updated_attn_state = []
                for old_state, new_state in zip(state["attn_state"], attn_state, strict=False):
                    tail = new_state[:, -window_size:]
                    if tail.shape[1] >= window_size:
                        updated_attn_state.append(tail.contiguous())
                    else:
                        old_state = old_state[:, -window_size:].contiguous() if old_state.shape[1] > window_size else old_state
                        old_state = old_state.roll(-input_size, 1)
                        old_state[:, -tail.shape[1]:] = tail
                        updated_attn_state.append(old_state)
                state["attn_state"] = updated_attn_state
            else:
                updated_attn_state = []
                for old_state, new_state in zip(state["attn_state"], attn_state, strict=False):
                    updated = torch.cat([old_state, new_state], 1)
                    if window_size is not None and updated.shape[1] > window_size:
                        updated = updated[:, -window_size:].contiguous()
                    updated_attn_state.append(updated)
                state["attn_state"] = updated_attn_state
        if conv_state is not None:
            state["conv_state"] = conv_state
        if ffn_state is not None:
            state["ffn_state"] = ffn_state
        self._rwkv7_cache_metrics["updates"] += 1
        return state

    def get_seq_length(self, layer_idx: int | None = 0, cache_position=None) -> int:
        if len(self.states) <= (layer_idx or 0):
            return 0
        return self._seen_tokens

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor | None, layer_idx: int = 0) -> tuple[int, int]:
        query_len = int(cache_position.shape[0]) if cache_position is not None else 0
        return int(self.get_seq_length(layer_idx)) + query_len, 0

    def reset(self) -> None:
        self._invalidate_native_graph_binding()
        self.states.clear()
        self._seen_tokens = 0
        self._rwkv7_cache_metrics["resets"] += 1

    def to_legacy_cache(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.states)

    def clone(self) -> "RWKV7StateCache":
        out = type(self)(seen_tokens=self._seen_tokens)
        out.states = [_clone_cache_value(state) for state in self.states]
        out._rwkv7_cache_metrics = dict(self._rwkv7_cache_metrics)
        out._rwkv7_cache_metrics["clones"] += 1
        out._rwkv7_cache_version = int(self._rwkv7_cache_version)
        return out

    def detach(self, *, inplace: bool = True) -> "RWKV7StateCache":
        """Detach cache tensors from autograd graphs for inference serving."""
        target = self if inplace else self.clone()
        target._invalidate_native_graph_binding()
        target.states = [_detach_cache_value(state) for state in target.states]
        target._rwkv7_cache_metrics["detaches"] += 1
        return target

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
        copy: bool = False,
        inplace: bool = True,
    ) -> "RWKV7StateCache":
        """Move/cache tensors between devices, optionally casting float tensors.

        This is primarily for serving systems that compact active rows, offload
        inactive states to CPU, and restore them before decode. Integer tensors
        keep their dtype; floating tensors are cast only when `dtype` is set.
        """
        target = self if inplace else self.clone()
        target._invalidate_native_graph_binding()
        target.states = [
            _to_cache_value(state, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
            for state in target.states
        ]
        target._rwkv7_cache_metrics["device_moves"] += 1
        return target

    def get_batch_size(self) -> int | None:
        return _first_tensor_batch_size(self.states)

    def select_batch(self, indices: torch.LongTensor, *, inplace: bool = True) -> "RWKV7StateCache":
        """Select/reorder active batch rows for dynamic serving.

        `indices` may reorder rows, drop completed rows, or both. The method is
        intentionally cache-only: sequence length is preserved because all
        active requests are assumed to have advanced together.
        """
        target = self if inplace else self.clone()
        runner = target._native_graph_bound_runner() if inplace else None
        if runner is not None and hasattr(runner, "reorder_batch_inplace"):
            try:
                same_size = target.get_batch_size() == int(indices.numel())
            except Exception:
                same_size = False
            if same_size and runner.reorder_batch_inplace(indices):
                target._rwkv7_cache_metrics["select_batch_calls"] += 1
                target._rwkv7_cache_metrics["native_graph_bound_selects"] = (
                    int(target._rwkv7_cache_metrics.get("native_graph_bound_selects", 0)) + 1
                )
                return target
        target._invalidate_native_graph_binding()
        target.states = [_move_first_dim(state, indices) for state in target.states]
        target._rwkv7_cache_metrics["select_batch_calls"] += 1
        return target

    def batch_select(self, indices: torch.LongTensor, *, inplace: bool = True) -> "RWKV7StateCache":
        target = self.select_batch(indices, inplace=inplace)
        target._rwkv7_cache_metrics["batch_select_calls"] += 1
        return target

    def reorder_cache(self, beam_idx: torch.LongTensor):
        target = self.select_batch(beam_idx, inplace=True)
        target._rwkv7_cache_metrics["reorder_calls"] += 1
        return target

    def rwkv7_cache_metrics(self) -> dict[str, Any]:
        """Return lightweight state-cache reuse/reorder counters.

        Serving integrations can sample this dictionary after prefill/decode to
        record whether a request reused one recurrent state cache or kept
        rebuilding/selecting/offloading it. The counters are intentionally
        framework-neutral so HF, benchmark harnesses, and future serving adapters
        can share the same signal.
        """
        metrics: dict[str, Any] = dict(self._rwkv7_cache_metrics)
        metrics.update(
            {
                "layers": len(self.states),
                "seen_tokens": int(self._seen_tokens),
                "batch_size": self.get_batch_size(),
            }
        )
        return metrics

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Any | None = None,
        seen_tokens: int = 0,
        **kwargs: Any,
    ) -> "RWKV7StateCache":
        if isinstance(past_key_values, cls):
            return past_key_values
        cache = cls(seen_tokens=seen_tokens, **kwargs)
        if isinstance(past_key_values, _FLACache) and hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        if isinstance(past_key_values, (list, tuple)):
            empty = {"recurrent_state": None, "attn_state": None, "conv_state": None, "ffn_state": None}
            cache.states = [dict(state) if state is not None else dict(empty) for state in past_key_values]
        return cache


class _RWKV7NativeGraphPrefillRunner:
    """Fixed-shape CUDA-graph replay for inference prefill.

    The recurrent input buffers are stable and read-only during replay. Output
    state is bound to the public cache without a per-layer copy; when this
    runner is reused by another cache, the previous cache is detached first.
    """

    def __init__(
        self,
        owner: "RWKV7ForCausalLM",
        packs,
        batch_size: int,
        prompt_tokens: int,
        logits_to_keep: int,
    ) -> None:
        if _native_jit_prefill is None or not torch.cuda.is_available():
            raise RuntimeError("native prefill graph requires CUDA and native_jit.prefill")
        self.owner = owner
        self.packs = packs
        self.batch_size = int(batch_size)
        self.prompt_tokens = int(prompt_tokens)
        self.logits_to_keep = int(logits_to_keep)
        weight = owner.model.embeddings.weight
        self.device = weight.device
        self.dtype = weight.dtype
        if self.device.type != "cuda":
            raise RuntimeError("native prefill graph requires CUDA model weights")
        self.input_ids = torch.zeros(
            self.batch_size,
            self.prompt_tokens,
            device=self.device,
            dtype=torch.long,
        )
        self.state_inputs: list[torch.Tensor] = []
        self.xpa_inputs: list[torch.Tensor] = []
        self.xpf_inputs: list[torch.Tensor] = []
        for p in packs:
            heads, head_dim = int(p[1]), int(p[2])
            hidden = heads * head_dim
            self.state_inputs.append(
                torch.zeros(self.batch_size, heads, head_dim, head_dim, device=self.device, dtype=torch.float32)
            )
            self.xpa_inputs.append(torch.zeros(self.batch_size, hidden, device=self.device, dtype=self.dtype))
            self.xpf_inputs.append(torch.zeros(self.batch_size, hidden, device=self.device, dtype=self.dtype))
        self.inputs_are_zero = True
        self.logits: torch.Tensor | None = None
        self.state_outputs: list[torch.Tensor] = []
        self.xpa_outputs: list[torch.Tensor] = []
        self.xpf_outputs: list[torch.Tensor] = []
        self.graph: torch.cuda.CUDAGraph | None = None
        self._bound_cache_ref: weakref.ReferenceType[RWKV7StateCache] | None = None
        self._capture()

    def matches(self, batch_size: int, prompt_tokens: int, logits_to_keep: int) -> bool:
        return (
            self.batch_size == int(batch_size)
            and self.prompt_tokens == int(prompt_tokens)
            and self.logits_to_keep == int(logits_to_keep)
        )

    def _run_once(self):
        # native_jit.prefill replaces entries in the Python lists. Pass shallow
        # copies so the stable graph input buffers are never replaced.
        return _native_jit_prefill(
            self.owner,
            self.input_ids,
            self.packs,
            state=list(self.state_inputs),
            xpa=list(self.xpa_inputs),
            xpf=list(self.xpf_inputs),
            logits_to_keep=self.logits_to_keep,
        )

    def _capture(self) -> None:
        warm = torch.cuda.Stream(device=self.device)
        warm.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warm):
            with torch.no_grad():
                for _ in range(3):
                    self._run_once()
        torch.cuda.current_stream(self.device).wait_stream(warm)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            outputs = self._run_once()
        self.logits, self.state_outputs, self.xpa_outputs, self.xpf_outputs = outputs

    @staticmethod
    def _copy_or_zero(dst: torch.Tensor, src: torch.Tensor | None, *, transpose_last: bool = False) -> None:
        if src is None:
            dst.zero_()
            return
        if src.dim() == dst.dim() - 1:
            src = src.unsqueeze(0)
        if transpose_last:
            src = src.transpose(-1, -2)
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    def _load_cache_inputs(self, past: RWKV7StateCache, initial_seen: int) -> None:
        if int(initial_seen) <= 0:
            if not self.inputs_are_zero:
                for value in (*self.state_inputs, *self.xpa_inputs, *self.xpf_inputs):
                    value.zero_()
                self.inputs_are_zero = True
            return
        for li, p in enumerate(self.packs):
            state = past._ensure_layer(int(p[0]))
            self._copy_or_zero(self.state_inputs[li], state.get("recurrent_state"), transpose_last=True)
            self._copy_or_zero(self.xpa_inputs[li], state.get("conv_state"))
            self._copy_or_zero(self.xpf_inputs[li], state.get("ffn_state"))
        self.inputs_are_zero = False

    def _detach_bound_cache_if_different(self, past: RWKV7StateCache) -> None:
        previous = self._bound_cache_ref() if self._bound_cache_ref is not None else None
        if previous is None or previous is past:
            return
        for li, p in enumerate(self.packs):
            layer_idx = int(p[0])
            if layer_idx >= len(previous.states):
                continue
            state = previous.states[layer_idx]
            recurrent_view = self.state_outputs[li].transpose(-1, -2)
            if isinstance(state.get("recurrent_state"), torch.Tensor) and _same_tensor_view(
                state["recurrent_state"], recurrent_view
            ):
                state["recurrent_state"] = recurrent_view.contiguous()
            if isinstance(state.get("conv_state"), torch.Tensor) and _same_tensor_view(
                state["conv_state"], self.xpa_outputs[li]
            ):
                state["conv_state"] = self.xpa_outputs[li].clone()
            if isinstance(state.get("ffn_state"), torch.Tensor) and _same_tensor_view(
                state["ffn_state"], self.xpf_outputs[li]
            ):
                state["ffn_state"] = self.xpf_outputs[li].clone()
        self._bound_cache_ref = None

    def _bind_cache(self, past: RWKV7StateCache, seen_tokens: int) -> None:
        past._invalidate_native_graph_binding()
        for li, p in enumerate(self.packs):
            state = past._ensure_layer(int(p[0]))
            state["recurrent_state"] = self.state_outputs[li].transpose(-1, -2)
            state["conv_state"] = self.xpa_outputs[li]
            state["ffn_state"] = self.xpf_outputs[li]
            state["attn_state"] = None
        past._seen_tokens = int(seen_tokens)
        self._bound_cache_ref = weakref.ref(past)

    def replay(
        self,
        input_ids: torch.LongTensor,
        past: RWKV7StateCache,
        initial_seen: int,
    ) -> tuple[torch.Tensor, RWKV7StateCache]:
        if tuple(input_ids.shape) != (self.batch_size, self.prompt_tokens):
            raise ValueError("native prefill graph input shape changed after capture")
        self._detach_bound_cache_if_different(past)
        self._load_cache_inputs(past, int(initial_seen))
        self.input_ids.copy_(input_ids.to(device=self.device))
        assert self.graph is not None and self.logits is not None
        self.graph.replay()
        self._bind_cache(past, int(initial_seen) + self.prompt_tokens)
        return self.logits.clone(), past


class RWKV7Model(_RWKV7Model):
    config_class = RWKV7Config


class RWKV7ForCausalLM(_RWKV7ForCausalLM):
    config_class = RWKV7Config
    # Transformers >=5 expects dict-like _tied_weights_keys in save_pretrained.
    _tied_weights_keys = {}
    # Generic bitsandbytes quantization is very slow on tiny RWKV-7 LoRA rank
    # projections on some accelerators. Keep those small matrices dense and
    # quantize the large projections/FFN weights instead; this preserves the
    # memory-saving direction while avoiding known low-throughput quantized
    # micro-kernels.
    _rwkv7_bnb_skip_modules = ["lm_head", r".*_lora\.lora\.[02]"]
    # Optional speed/memory trade-off policies for bitsandbytes inference:
    # - memory: quantize all large projection/FFN matrices (smallest footprint).
    # - decode_hot: keep attention r/k/v/o projections dense; validation smoke
    #   showed this can improve W4 cached decode while still keeping a lower
    #   footprint than fp16. FFN key/value remain quantized.
    # - dense: keep all large Linear modules dense (diagnostic upper bound).
    _rwkv7_bnb_policy_extra_skips = {
        "memory": [],
        "decode_hot": [r".*attn\.(r_proj|k_proj|v_proj|o_proj)"],
        "dense": [r".*attn\.(r_proj|k_proj|v_proj|o_proj)", r".*ffn\.(key|value)"],
    }

    @staticmethod
    def _rwkv7_bnb_concrete_skip_modules(policy: str, config: Any | None = None) -> list[str]:
        num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
        if num_layers <= 0:
            return []
        skips: list[str] = []
        for layer_idx in range(num_layers):
            for lora_name in ("w_lora", "a_lora", "g_lora", "v_lora"):
                for linear_idx in (0, 2):
                    skips.append(f"model.layers.{layer_idx}.attn.{lora_name}.lora.{linear_idx}")
            if policy in {"decode_hot", "dense"}:
                for proj_name in ("r_proj", "k_proj", "v_proj", "o_proj"):
                    skips.append(f"model.layers.{layer_idx}.attn.{proj_name}")
            if policy == "dense":
                for ffn_name in ("key", "value"):
                    skips.append(f"model.layers.{layer_idx}.ffn.{ffn_name}")
        return skips

    @classmethod
    def rwkv7_bnb_skip_modules(cls, policy: str | None = None, config: Any | None = None) -> list[str]:
        policy = _bnb_skip_policy(policy)
        return list(
            dict.fromkeys(
                [
                    *cls._rwkv7_bnb_skip_modules,
                    *cls._rwkv7_bnb_policy_extra_skips[policy],
                    *cls._rwkv7_bnb_concrete_skip_modules(policy, config),
                ]
            )
        )

    @classmethod
    def _rwkv7_prepare_bnb_kwargs(cls, pretrained_model_name_or_path, kwargs: dict[str, Any]):
        rwkv7_bnb_skip_policy = _bnb_skip_policy(kwargs.pop("rwkv7_bnb_skip_policy", None))
        quantization_config = kwargs.get("quantization_config")
        if quantization_config is None and (kwargs.get("load_in_8bit") or kwargs.get("load_in_4bit")):
            from transformers import BitsAndBytesConfig

            bnb_kwargs = {}
            for key in list(kwargs.keys()):
                if key.startswith("bnb_4bit_") or key.startswith("llm_int8_") or key in {"load_in_8bit", "load_in_4bit"}:
                    bnb_kwargs[key] = kwargs.pop(key)
            quantization_config = BitsAndBytesConfig(**bnb_kwargs)
            kwargs["quantization_config"] = quantization_config
        if quantization_config is not None and hasattr(quantization_config, "llm_int8_skip_modules"):
            config_for_skip = kwargs.get("config")
            if config_for_skip is None:
                try:
                    config_for_skip = cls.config_class.from_pretrained(pretrained_model_name_or_path)
                except Exception:
                    config_for_skip = None
            existing = list(getattr(quantization_config, "llm_int8_skip_modules", None) or [])
            merged = list(dict.fromkeys([*existing, *cls.rwkv7_bnb_skip_modules(rwkv7_bnb_skip_policy, config_for_skip)]))
            quantization_config.llm_int8_skip_modules = merged
        return rwkv7_bnb_skip_policy, quantization_config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        rwkv7_bnb_skip_policy, quantization_config = cls._rwkv7_prepare_bnb_kwargs(
            pretrained_model_name_or_path,
            kwargs,
        )
        if _native_model_backend_requested():
            # FLA-free backend: explicit opt-in via RWKV7_NATIVE_MODEL, or a
            # conservative policy fallback for CUDA generations whose FLA/Triton
            # kernels are known to be unavailable. Same checkpoint and HF API.
            from .native_model import NativeRWKV7ForCausalLM

            model = NativeRWKV7ForCausalLM.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
            if quantization_config is not None:
                setattr(model, "_rwkv7_bnb_skip_policy", rwkv7_bnb_skip_policy)
                if getattr(model, "config", None) is not None:
                    setattr(model.config, "rwkv7_bnb_skip_policy", rwkv7_bnb_skip_policy)
            return model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if quantization_config is not None:
            setattr(model, "_rwkv7_bnb_skip_policy", rwkv7_bnb_skip_policy)
            if getattr(model, "config", None) is not None:
                setattr(model.config, "rwkv7_bnb_skip_policy", rwkv7_bnb_skip_policy)
        use_native_mm8 = bool(getattr(model.config, "use_native_mm8", False))
        use_native_mm4 = bool(getattr(model.config, "use_native_mm4", False))
        if quantization_config is None:
            if use_native_mm8 and use_native_mm4:
                raise ValueError("use_native_mm8 and use_native_mm4 are mutually exclusive")
            # Persisted native W8/W4: re-quantize eligible linears from fp
            # weights. Deterministic, so it round-trips the saved state.
            # This path is bitsandbytes-free and is also used by Apple/CPU
            # native fallback smokes.
            if use_native_mm8:
                from .native_quant_mm8 import quantize_model_mm8

                replaced = quantize_model_mm8(
                    model,
                    min_params=int(getattr(model.config, "native_mm8_min_params", 8_000_000)),
                    policy=str(getattr(model.config, "native_mm8_policy", "memory")),
                )
                setattr(model, "_rwkv7_native_mm_quantization", "mm8")
                setattr(model, "_rwkv7_native_mm_replaced_modules", int(replaced))
            elif use_native_mm4:
                from .native_quant_mm4 import quantize_model_mm4

                replaced = quantize_model_mm4(
                    model,
                    min_params=int(getattr(model.config, "native_mm4_min_params", 8_000_000)),
                    policy=str(getattr(model.config, "native_mm4_policy", "memory")),
                )
                setattr(model, "_rwkv7_native_mm_quantization", "mm4")
                setattr(model, "_rwkv7_native_mm_replaced_modules", int(replaced))
        return model

    def resize_token_embeddings(self, new_num_tokens: int | None = None, *args, **kwargs):
        """Keep the official RWKV trie vocabulary fixed.

        RWKV-7 checkpoints are tied to the fixed 65k RWKV trie vocabulary and
        the remote tokenizer does not have a safe way to initialize new rows.
        A no-op resize is allowed because some HF/PEFT helpers call it while
        checking model capabilities; changing the vocabulary size is rejected
        early instead of silently producing an invalid embedding/head pair.
        """
        if new_num_tokens is None or int(new_num_tokens) == int(self.config.vocab_size):
            return self.get_input_embeddings()
        raise NotImplementedError(
            "RWKV-7 uses the fixed official trie vocabulary; changing vocab size "
            "with resize_token_embeddings is not supported by this adapter."
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx: torch.LongTensor):
        """GenerationMixin beam-search hook for recurrent RWKV state caches."""
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "reorder_cache"):
            return past_key_values.reorder_cache(beam_idx)
        if isinstance(past_key_values, (tuple, list)):
            return RWKV7StateCache.from_legacy_cache(past_key_values).reorder_cache(beam_idx).to_legacy_cache()
        raise TypeError(f"Unsupported RWKV-7 cache type for beam reorder: {type(past_key_values)!r}")

    @torch.no_grad()
    def rwkv7_forward_one(
        self,
        input_ids: torch.LongTensor,
        past_key_values: RWKV7StateCache | _FLACache | tuple | list | None = None,
        return_dict: bool | None = True,
    ):
        """Inference-only bsz=1 one-token decode path.

        This keeps the standard HF `forward` path untouched for `generate`, PEFT,
        Trainer, and TRL. Serving stacks can call this method after a normal HF
        prefill to avoid the generic 3D module/cache path for recurrent decode.
        It uses the same FLA fused recurrent kernel and the same state layout as
        `RWKV7StateCache`, but performs token shift, FFN shift, and gate output
        correction directly on `[1, 1, hidden]` tensors.
        """
        if self.training:
            raise RuntimeError("rwkv7_forward_one is inference-only; call model.eval() first")
        token = input_ids.reshape(-1)
        if token.numel() != 1:
            raise ValueError("rwkv7_forward_one only supports exactly one token with batch size 1")
        return self.rwkv7_forward_token(input_ids, past_key_values=past_key_values, return_dict=return_dict)

    def rwkv7_last_fast_token_backend(self) -> str | None:
        """Return the effective backend used by the previous fast-token call."""
        return getattr(self, "_rwkv7_last_fast_token_backend", None)

    def rwkv7_last_fast_prefill_backend(self) -> str | None:
        """Return the effective backend used by the previous fast-prefill call."""
        return getattr(self, "_rwkv7_last_fast_prefill_backend", None)

    def rwkv7_native_graph_cache_batch_sizes(self) -> list[int]:
        """Return active batch sizes currently retained in the graph-runner LRU."""
        cache = getattr(self, "_rwkv7_native_graph_runner_cache", None)
        if isinstance(cache, tuple) and len(cache) == 2:
            key = cache[0]
            return [int(key[-1])] if isinstance(key, tuple) and key else []
        if isinstance(cache, OrderedDict):
            return sorted({int(key[-1]) for key in cache.keys() if isinstance(key, tuple) and key})
        return []

    def rwkv7_native_graph_cache_stats(self) -> dict[str, Any]:
        """Return native-graph runner LRU reuse counters for serving telemetry."""
        stats = dict(getattr(self, "_rwkv7_native_graph_cache_stats", _native_graph_stats_template()))
        requests = int(stats.get("requests", 0))
        hits = int(stats.get("hits", 0))
        stats.update(
            {
                "size": len(self.rwkv7_native_graph_cache_batch_sizes()),
                "limit": _native_graph_cache_size(),
                "batch_sizes": self.rwkv7_native_graph_cache_batch_sizes(),
                "hit_rate": (float(hits) / float(requests)) if requests else None,
            }
        )
        return stats

    def rwkv7_native_graph_runner_copy_stats(self) -> dict[str, Any]:
        """Return aggregate native-graph runner cache-copy/binding counters.

        The LRU-level counters above answer whether a captured runner was
        reused for the active batch size.  These runner-level counters answer
        the more serving-specific question: once the runner is reused, did the
        RWKV7StateCache stay bound to the captured buffers or did dynamic
        batching/reorder force a state copy back into the graph inputs?
        """

        cache = getattr(self, "_rwkv7_native_graph_runner_cache", None)
        runners: list[tuple[Any, Any]] = []
        if isinstance(cache, tuple) and len(cache) == 2:
            runners = [cache]
        elif isinstance(cache, OrderedDict):
            runners = list(cache.items())

        totals = {
            "copy_from_cache_calls": 0,
            "copy_from_cache_fast_skips": 0,
            "bind_cache_calls": 0,
            "bind_cache_fast_skips": 0,
        }
        rows: list[dict[str, Any]] = []
        for key, runner in runners:
            stats = runner.copy_stats() if hasattr(runner, "copy_stats") else {}
            batch_size = int(key[-1]) if isinstance(key, tuple) and key else getattr(runner, "batch_size", None)
            row = {"batch_size": int(batch_size) if batch_size is not None else None}
            for name in totals:
                value = int(stats.get(name, 0))
                row[name] = value
                totals[name] += value
            rows.append(row)

        copy_calls = int(totals["copy_from_cache_calls"])
        bind_calls = int(totals["bind_cache_calls"])
        totals["copy_from_cache_fast_skip_rate"] = (
            float(totals["copy_from_cache_fast_skips"]) / float(copy_calls) if copy_calls else None
        )
        totals["bind_cache_fast_skip_rate"] = (
            float(totals["bind_cache_fast_skips"]) / float(bind_calls) if bind_calls else None
        )
        return {
            "totals": totals,
            "runners": sorted(rows, key=lambda row: (-1 if row["batch_size"] is None else int(row["batch_size"]))),
        }

    def rwkv7_reset_native_graph_cache_stats(self) -> dict[str, Any]:
        """Reset and return native-graph cache reuse counters."""
        self._rwkv7_native_graph_cache_stats = _native_graph_stats_template()
        return self.rwkv7_native_graph_cache_stats()

    @torch.inference_mode()
    def rwkv7_warmup_fast_token(
        self,
        batch_sizes: int | list[int] | tuple[int, ...] = (1,),
        backend: str | None = None,
    ) -> dict[int, str]:
        """Pre-initialize fast-token native resources for serving.

        For the native-graph backend this captures and caches graph runners for
        each requested active batch size, removing the first-request graph
        capture from the serving hot path. For native-JIT it extracts/caches the
        packed weights. `backend=None` follows `RWKV7_FAST_TOKEN_BACKEND`, while
        `backend="auto"` uses the same graph -> JIT -> FLA resolution as
        `rwkv7_forward_token`.
        """
        if isinstance(batch_sizes, int):
            sizes = [int(batch_sizes)]
        else:
            sizes = [int(v) for v in batch_sizes]
        if not sizes:
            raise ValueError("rwkv7_warmup_fast_token requires at least one batch size")

        requested = _normalize_fast_token_backend(backend) if backend is not None else _fast_token_backend()
        warmed: dict[int, str] = {}
        for batch_size in sizes:
            if batch_size <= 0:
                raise ValueError("rwkv7_warmup_fast_token batch sizes must be positive")
            chosen = self._rwkv7_resolve_fast_token_backend(batch_size) if requested == "auto" else requested
            if chosen in {"native_graph", "native_jit"} and self._rwkv7_uses_external_quantization():
                chosen = "fla"
            if chosen == "native_graph":
                if not self._rwkv7_can_use_native_backend("native_graph", batch_size):
                    if requested != "auto":
                        raise RuntimeError(f"native_graph fast-token backend is unavailable for batch_size={batch_size}")
                    chosen = "native_jit" if self._rwkv7_can_use_native_backend("native_jit", batch_size) else "fla"
                else:
                    packs = self._rwkv7_native_jit_packs()
                    self._rwkv7_native_graph_runner(packs, batch_size)
            if chosen == "native_jit":
                if not self._rwkv7_can_use_native_backend("native_jit", batch_size):
                    if requested != "auto":
                        raise RuntimeError(f"native_jit fast-token backend is unavailable for batch_size={batch_size}")
                    chosen = "fla"
                else:
                    self._rwkv7_native_jit_packs()
            warmed[batch_size] = chosen
        return warmed

    def _rwkv7_has_multi_cuda_device_map(self) -> bool:
        """Return True when Accelerate placed model blocks on multiple CUDA devices.

        The native/FLA fast-token helpers directly walk submodules and cache
        tensors, so they assume one active CUDA device. Standard HF forward
        with Accelerate hooks can still move tensors between a split device_map;
        in that case the top-level forward/generate path should skip the
        fast-forward shortcut and use the normal HF implementation.
        """
        devices: set[tuple[str, int | None]] = set()
        device_map = getattr(self, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            for value in device_map.values():
                if isinstance(value, int):
                    devices.add(("cuda", int(value)))
                    continue
                dev = torch.device(value) if isinstance(value, str) and value not in {"disk"} else None
                if dev is not None and dev.type == "cuda":
                    devices.add(("cuda", dev.index))
            if len(devices) > 1:
                return True

        parameters = getattr(self, "parameters", None)
        if not callable(parameters):
            return False
        param_devices = {
            (p.device.type, p.device.index)
            for p in parameters()
            if getattr(p, "device", None) is not None and p.device.type == "cuda"
        }
        return len(param_devices) > 1

    def _rwkv7_uses_external_quantization(self) -> bool:
        """Detect generic HF/bitsandbytes quantization wrappers.

        The native fast-token paths expect dense floating-point projection
        weights extracted from the FLA modules. Generic 8-bit/4-bit wrappers are
        still supported through the normal FLA fast-token path until a dedicated
        quantized native path exists.
        """
        if bool(getattr(self, "is_loaded_in_8bit", False)) or bool(getattr(self, "is_loaded_in_4bit", False)):
            return True
        if getattr(self, "hf_quantizer", None) is not None:
            return True
        config = getattr(self, "config", None)
        return getattr(config, "quantization_config", None) is not None

    def _rwkv7_can_use_native_backend(self, backend: str, batch_size: int) -> bool:
        if self._rwkv7_has_multi_cuda_device_map():
            return False
        if self._rwkv7_uses_external_quantization():
            return False
        if backend == "native_jit":
            if _native_jit_block_step is None or _native_jit_extract is None:
                return False
            if int(batch_size) != 1 and _native_jit_block_step_batched is None:
                return False
            try:
                self._rwkv7_native_jit_packs()
            except Exception:
                return False
            return True
        if backend == "native_graph":
            if int(batch_size) == 1:
                if _native_graph_block_ip is None:
                    return False
            elif _native_graph_block_ip_batched is None:
                return False
            weight = self.model.embeddings.weight
            if not _cuda_available() or getattr(weight.device, "type", None) != "cuda":
                return False
            try:
                self._rwkv7_native_jit_packs()
            except Exception:
                return False
            return True
        return backend == "fla"

    def _rwkv7_resolve_fast_token_backend(self, batch_size: int) -> str:
        requested = _fast_token_backend()
        if requested != "auto":
            if self._rwkv7_uses_external_quantization() and requested in {"native_graph", "native_jit"}:
                # Bitsandbytes/HF quantization wraps Linear weights in packed
                # int8/int4 modules. The native paths extract and replay dense
                # floating-point projection weights, so a global serving env
                # such as RWKV7_FAST_TOKEN_BACKEND=native_graph must not make
                # quantized generate crash. Keep quantized fast-forward on the
                # FLA tensor path until a real native quant kernel exists.
                return "fla"
            return requested
        if self._rwkv7_can_use_native_backend("native_graph", batch_size):
            return "native_graph"
        if self._rwkv7_can_use_native_backend("native_jit", batch_size):
            return "native_jit"
        return "fla"

    @torch.no_grad()
    def rwkv7_forward_token(
        self,
        input_ids: torch.LongTensor,
        past_key_values: RWKV7StateCache | _FLACache | tuple | list | None = None,
        return_dict: bool | None = True,
    ):
        """Inference-only one-token decode path for any batch size.

        `input_ids` may be shaped `[batch]` or `[batch, 1]`. This is the batched
        version of `rwkv7_forward_one`: it keeps the standard HF `forward` path
        unchanged, but lets serving benchmarks bypass generic sequence/cache
        handling for one-token recurrent decode after a normal HF prefill. The
        default `RWKV7_FAST_TOKEN_BACKEND=auto` resolves to native graph replay,
        native JIT, or the FLA tensor path depending on runtime availability.
        """
        if self.training:
            raise RuntimeError("rwkv7_forward_token is inference-only; call model.eval() first")
        if input_ids.dim() == 1:
            token = input_ids
        elif input_ids.dim() == 2 and input_ids.shape[1] == 1:
            token = input_ids[:, 0]
        else:
            raise ValueError("rwkv7_forward_token expects input_ids shaped [batch] or [batch, 1]")
        if token.numel() == 0:
            raise ValueError("rwkv7_forward_token requires a non-empty batch")
        if not isinstance(past_key_values, RWKV7StateCache):
            past_key_values = RWKV7StateCache.from_legacy_cache(past_key_values)

        requested_backend = _fast_token_backend()
        backend = self._rwkv7_resolve_fast_token_backend(int(token.numel()))
        self._rwkv7_last_fast_token_backend = backend
        if backend == "native_graph":
            try:
                return self._rwkv7_forward_token_native_graph(token, past_key_values, return_dict)
            except Exception:
                if requested_backend != "auto":
                    raise
                backend = "native_jit" if self._rwkv7_can_use_native_backend("native_jit", int(token.numel())) else "fla"
                self._rwkv7_last_fast_token_backend = backend
        if backend == "native_jit":
            return self._rwkv7_forward_token_native_jit(token, past_key_values, return_dict)

        if _fast_token_layout() == "2d":
            return self._rwkv7_forward_token_2d(token, past_key_values, return_dict)

        x = self.model.embeddings(token.view(-1, 1))
        v_first = None
        for layer_idx, layer in enumerate(self.model.layers):
            state = past_key_values._ensure_layer(layer_idx)
            residual = layer.pre_norm(x) if hasattr(layer, "pre_norm") else x
            attn_input = layer.attn_norm(residual)
            attn_out, recurrent_state, conv_state, v_first = self._rwkv7_attn_one(
                layer.attn,
                attn_input,
                state,
                v_first,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            ffn_input = layer.ffn_norm(hidden_states)
            ffn_out, ffn_state = self._rwkv7_ffn_one(layer.ffn, ffn_input, state)
            x = residual + ffn_out
            state["recurrent_state"] = recurrent_state
            state["conv_state"] = conv_state
            state["ffn_state"] = ffn_state
            state["attn_state"] = None

        past_key_values._seen_tokens += 1
        hidden_states = self.model.norm(x)
        logits = _linear_direct(self.lm_head, hidden_states)
        if not return_dict:
            return logits, past_key_values
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _rwkv7_native_jit_packs(self, *, for_graph: bool = False):
        if _native_jit_block_step is None or _native_jit_block_step_batched is None or _native_jit_extract is None:
            raise RuntimeError("native_jit fast-token backend is unavailable; copy native_jit.py into the model repo")
        if for_graph and _native_graph_extract is None:
            raise RuntimeError("native_graph operand extraction is unavailable; copy native_jit.py into the model repo")
        cache_name = "_rwkv7_native_graph_pack_cache" if for_graph else "_rwkv7_native_jit_pack_cache"
        cache = getattr(self, cache_name, None)
        weight = self.model.embeddings.weight
        key = (
            weight.device.type,
            weight.device.index,
            weight.dtype,
            _native_graph_rkv_policy(),
            _native_graph_vkwr_rkv_thresholds(),
            bool(for_graph),
            str(getattr(self, "_rwkv7_native_mm_quantization", "none")),
            int(getattr(self, "_rwkv7_native_mm_replaced_modules", 0)),
        )
        if cache is None or cache[0] != key:
            extractor = _native_graph_extract if for_graph else _native_jit_extract
            packs, _, _, _ = extractor(self)
            setattr(self, cache_name, (key, packs))
            return packs
        return cache[1]

    def _rwkv7_native_graph_runner(self, packs, batch_size: int):
        weight = self.model.embeddings.weight
        key = (
            weight.device.type,
            weight.device.index,
            weight.dtype,
            len(packs),
            int(packs[0][1]),
            int(packs[0][2]),
            _native_graph_fused_recurrent_requested(),
            _native_graph_fused_recurrent_output_requested(),
            _native_graph_fused_recurrent_raw_requested(),
            _native_graph_fused_output_requested(),
            _native_graph_fused_output_project_requested(),
            _native_graph_fused_output_project_block_m(),
            _native_graph_fused_projection_requested(),
            _native_graph_fused_wag_lora_requested(),
            _native_graph_fused_wag_lora_blocks(),
            _native_graph_fused_wavg_lora_requested(),
            _native_graph_fused_wavg_lora_bsz1_max_hidden(),
            _native_graph_fused_wavg_lora_blocks(),
            _native_graph_fused_norm_mix_requested(),
            _native_graph_fused_norm_mix_num_warps(),
            _native_graph_sm70_linear_requested(),
            _native_graph_ada_linear_requested(),
            _native_graph_ada_linear_signature(),
            _native_graph_ada_wagv_lora_requested(),
            _native_graph_ada_sparse_ffn_requested(),
            _native_graph_ada_sparse_ffn_signature(),
            _native_graph_rkv_policy(),
            _native_graph_vkwr_rkv_thresholds(),
            int(batch_size),
        )
        cache = getattr(self, "_rwkv7_native_graph_runner_cache", None)
        if isinstance(cache, tuple) and len(cache) == 2:
            cache = OrderedDict([cache])
        elif not isinstance(cache, OrderedDict):
            cache = OrderedDict()
        self._rwkv7_native_graph_runner_cache = cache
        stats = getattr(self, "_rwkv7_native_graph_cache_stats", None)
        if not isinstance(stats, dict):
            stats = _native_graph_stats_template()
            self._rwkv7_native_graph_cache_stats = stats
        stats["requests"] = int(stats.get("requests", 0)) + 1

        runner = cache.get(key)
        if runner is not None:
            stats["hits"] = int(stats.get("hits", 0)) + 1
            cache.move_to_end(key)
            return runner

        stats["misses"] = int(stats.get("misses", 0)) + 1
        if _native_graph_ada_sparse_ffn_requested() and _native_graph_prewarm_sparse_ffn is not None:
            _native_graph_prewarm_sparse_ffn(packs, int(batch_size))
        if int(batch_size) == 1:
            runner = _RWKV7NativeGraphTokenRunner(self, packs)
        else:
            runner = _RWKV7NativeGraphBatchedTokenRunner(self, packs, int(batch_size))
        cache[key] = runner
        cache.move_to_end(key)
        while len(cache) > _native_graph_cache_size():
            cache.popitem(last=False)
            stats["evictions"] = int(stats.get("evictions", 0)) + 1
        return runner

    def rwkv7_clear_native_graph_cache(self) -> int:
        """Drop captured native-graph runners and return how many were kept.

        Serving stacks can call this when changing traffic profiles or before
        a memory-sensitive phase. The cache otherwise behaves as a small LRU
        keyed by device, dtype, model shape, and active batch size.
        """
        cache = getattr(self, "_rwkv7_native_graph_runner_cache", None)
        if isinstance(cache, OrderedDict):
            size = len(cache)
            cache.clear()
            return size
        if isinstance(cache, tuple):
            self._rwkv7_native_graph_runner_cache = OrderedDict()
            return 1
        self._rwkv7_native_graph_runner_cache = OrderedDict()
        return 0

    def _rwkv7_native_prefill_graph_runner(
        self,
        packs,
        batch_size: int,
        prompt_tokens: int,
        logits_to_keep: int,
    ) -> _RWKV7NativeGraphPrefillRunner:
        weight = self.model.embeddings.weight
        key = (
            weight.device.type,
            weight.device.index,
            weight.dtype,
            int(batch_size),
            int(prompt_tokens),
            int(logits_to_keep),
            _native_prefill_graph_signature(),
            str(getattr(self, "_rwkv7_native_mm_quantization", "none")),
        )
        cache = getattr(self, "_rwkv7_native_prefill_graph_runner_cache", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            self._rwkv7_native_prefill_graph_runner_cache = cache
        runner = cache.get(key)
        if runner is not None:
            cache.move_to_end(key)
            self._rwkv7_native_prefill_graph_hot_runner = runner
            return runner
        runner = _RWKV7NativeGraphPrefillRunner(
            self,
            packs,
            int(batch_size),
            int(prompt_tokens),
            int(logits_to_keep),
        )
        cache[key] = runner
        cache.move_to_end(key)
        while len(cache) > _native_prefill_graph_cache_size():
            cache.popitem(last=False)
        self._rwkv7_native_prefill_graph_hot_runner = runner
        return runner

    def rwkv7_clear_native_prefill_graph_cache(self) -> int:
        """Drop fixed-shape native prefill CUDA graphs."""

        cache = getattr(self, "_rwkv7_native_prefill_graph_runner_cache", None)
        if not isinstance(cache, OrderedDict):
            self._rwkv7_native_prefill_graph_runner_cache = OrderedDict()
            self._rwkv7_native_prefill_graph_hot_runner = None
            return 0
        size = len(cache)
        cache.clear()
        self._rwkv7_native_prefill_graph_hot_runner = None
        return size

    def rwkv7_warmup_fast_prefill(
        self,
        shapes: tuple[int, int] | list[tuple[int, int]] | tuple[tuple[int, int], ...] = ((1, 512),),
        *,
        logits_to_keep: int = 1,
    ) -> dict[str, str]:
        """Pre-capture fixed ``(batch, prompt_tokens)`` serving shapes."""

        if self.training:
            raise RuntimeError("rwkv7_warmup_fast_prefill is inference-only; call model.eval() first")
        if isinstance(shapes, tuple) and len(shapes) == 2 and all(isinstance(value, int) for value in shapes):
            normalized = [shapes]
        else:
            normalized = list(shapes)
        if not normalized:
            raise ValueError("rwkv7_warmup_fast_prefill requires at least one shape")
        packs = self._rwkv7_native_jit_packs()
        warmed: dict[str, str] = {}
        for batch_size, prompt_tokens in normalized:
            if int(batch_size) <= 0 or int(prompt_tokens) <= 0:
                raise ValueError("prefill warmup batch and prompt sizes must be positive")
            self._rwkv7_native_prefill_graph_runner(
                packs,
                int(batch_size),
                int(prompt_tokens),
                int(logits_to_keep),
            )
            warmed[f"{int(batch_size)}x{int(prompt_tokens)}"] = "native_prefill_graph"
        return warmed

    def _rwkv7_native_prefill_initial_state(
        self,
        past_key_values: RWKV7StateCache,
        packs,
        batch_size: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        weight = self.model.embeddings.weight
        device = weight.device
        dtype = weight.dtype
        state_native: list[torch.Tensor] = []
        xpa: list[torch.Tensor] = []
        xpf: list[torch.Tensor] = []
        for p in packs:
            layer_idx = int(p[0])
            H = int(p[1])
            N = int(p[2])
            hidden = H * N
            layer_state = past_key_values._ensure_layer(layer_idx)

            recurrent = layer_state.get("recurrent_state")
            if isinstance(recurrent, torch.Tensor):
                if recurrent.dim() == 3:
                    recurrent = recurrent.unsqueeze(0)
                recurrent = recurrent.to(device=device, dtype=torch.float32).transpose(-1, -2).contiguous()
                if int(recurrent.shape[0]) != int(batch_size):
                    raise ValueError(f"native prefill recurrent_state batch mismatch on layer {layer_idx}: {tuple(recurrent.shape)}")
            else:
                recurrent = torch.zeros(batch_size, H, N, N, device=device, dtype=torch.float32)
            state_native.append(recurrent)

            conv = layer_state.get("conv_state")
            if isinstance(conv, torch.Tensor):
                if conv.dim() == 1:
                    conv = conv.unsqueeze(0)
                conv = conv.to(device=device, dtype=dtype).contiguous()
                if int(conv.shape[0]) != int(batch_size):
                    raise ValueError(f"native prefill conv_state batch mismatch on layer {layer_idx}: {tuple(conv.shape)}")
            else:
                conv = torch.zeros(batch_size, hidden, device=device, dtype=dtype)
            xpa.append(conv.reshape(batch_size, hidden))

            ffn = layer_state.get("ffn_state")
            if isinstance(ffn, torch.Tensor):
                if ffn.dim() == 1:
                    ffn = ffn.unsqueeze(0)
                ffn = ffn.to(device=device, dtype=dtype).contiguous()
                if int(ffn.shape[0]) != int(batch_size):
                    raise ValueError(f"native prefill ffn_state batch mismatch on layer {layer_idx}: {tuple(ffn.shape)}")
            else:
                ffn = torch.zeros(batch_size, hidden, device=device, dtype=dtype)
            xpf.append(ffn.reshape(batch_size, hidden))
        return state_native, xpa, xpf

    @torch.no_grad()
    def rwkv7_prefill_native(
        self,
        input_ids: torch.LongTensor,
        past_key_values: RWKV7StateCache | _FLACache | tuple | list | None = None,
        logits_to_keep: int = 1,
        return_dict: bool | None = True,
    ):
        """Inference-only native prefill path with optional fused recurrent scan.

        This is the serving bridge for the native fused prefill work: it runs
        prompt prefill layer-wise through `native_jit.prefill`, writes the final
        recurrent/shift state back into `RWKV7StateCache`, and leaves the cache
        layout compatible with the existing native-graph one-token decode path.
        Set `RWKV7_NATIVE_PREFILL_FUSED_SCAN=1` to force the Triton scan
        prototype; otherwise the method uses the same math with the safe torch
        recurrent loop.
        """

        if self.training:
            raise RuntimeError("rwkv7_prefill_native is inference-only; call model.eval() first")
        if _native_jit_prefill is None:
            raise RuntimeError("native prefill backend is unavailable; copy native_jit.py into the model repo")
        self._rwkv7_last_fast_prefill_backend = "native_prefill"
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError("rwkv7_prefill_native expects input_ids shaped [batch, seq]")
        if int(input_ids.shape[1]) <= 0:
            raise ValueError("rwkv7_prefill_native requires at least one token")
        batch_size = int(input_ids.shape[0])
        if self._rwkv7_uses_external_quantization():
            raise RuntimeError("native prefill currently requires dense floating-point weights")
        source_seen = None
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            try:
                source_seen = int(past_key_values.get_seq_length())
            except Exception:
                source_seen = None
        past = RWKV7StateCache.from_legacy_cache(past_key_values)
        initial_seen = source_seen if source_seen is not None else (int(past.get_seq_length()) if hasattr(past, "get_seq_length") else 0)
        if _native_prefill_graph_enabled():
            keep_value = 0 if logits_to_keep is None else int(logits_to_keep)
            prompt_tokens = int(input_ids.shape[1])
            runner = getattr(self, "_rwkv7_native_prefill_graph_hot_runner", None)
            if not isinstance(runner, _RWKV7NativeGraphPrefillRunner) or not runner.matches(
                batch_size, prompt_tokens, keep_value
            ):
                packs = self._rwkv7_native_jit_packs()
                runner = self._rwkv7_native_prefill_graph_runner(
                    packs,
                    batch_size,
                    prompt_tokens,
                    keep_value,
                )
            logits, past = runner.replay(input_ids, past, initial_seen)
            self._rwkv7_last_fast_prefill_backend = "native_prefill_graph"
            if not return_dict:
                return logits, past
            return CausalLMOutputWithPast(logits=logits, past_key_values=past)
        packs = self._rwkv7_native_jit_packs()
        state_native, xpa, xpf = self._rwkv7_native_prefill_initial_state(past, packs, batch_size)
        logits, state_native, xpa, xpf = _native_jit_prefill(
            self,
            input_ids.to(self.model.embeddings.weight.device),
            packs,
            state=state_native,
            xpa=xpa,
            xpf=xpf,
            logits_to_keep=logits_to_keep,
        )

        past._invalidate_native_graph_binding()
        for li, p in enumerate(packs):
            layer_idx = int(p[0])
            layer_state = past._ensure_layer(layer_idx)
            layer_state["recurrent_state"] = state_native[li].transpose(-1, -2).contiguous()
            layer_state["conv_state"] = xpa[li].contiguous()
            layer_state["ffn_state"] = xpf[li].contiguous()
            layer_state["attn_state"] = None
        past._seen_tokens = initial_seen + int(input_ids.shape[1])
        if not return_dict:
            return logits, past
        return CausalLMOutputWithPast(logits=logits, past_key_values=past)

    @torch.no_grad()
    def rwkv7_prefill_chunks(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        chunk_size: int = 2048,
        past_key_values: RWKV7StateCache | _FLACache | tuple | list | None = None,
        logits_to_keep: int = 1,
        return_dict: bool | None = True,
        **kwargs,
    ):
        """Inference-only chunked prefill helper for serving stacks.

        This keeps the normal HF `forward` implementation as the source of
        truth, but splits a long prompt into smaller chunks while carrying the
        recurrent `RWKV7StateCache` between chunks. Intermediate chunks request
        only the final logit to avoid large temporary logits tensors; the final
        chunk honors `logits_to_keep`.
        """
        if self.training:
            raise RuntimeError("rwkv7_prefill_chunks is inference-only; call model.eval() first")
        if input_ids.dim() != 2:
            raise ValueError("rwkv7_prefill_chunks expects input_ids shaped [batch, seq]")
        if int(input_ids.shape[1]) <= 0:
            raise ValueError("rwkv7_prefill_chunks requires at least one token")
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if attention_mask is not None and tuple(attention_mask.shape[:2]) != tuple(input_ids.shape[:2]):
            raise ValueError("attention_mask must have the same [batch, seq] shape as input_ids")

        total = int(input_ids.shape[1])
        past = RWKV7StateCache.from_legacy_cache(past_key_values)
        initial_seen = int(past.get_seq_length()) if hasattr(past, "get_seq_length") else 0
        out = None
        kwargs.pop("use_cache", None)
        kwargs.pop("past_key_values", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("logits_to_keep", None)
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            chunk_kwargs = dict(kwargs)
            if attention_mask is not None:
                chunk_kwargs["attention_mask"] = attention_mask[:, start:end]
            chunk_mask = chunk_kwargs.pop("attention_mask", None)
            if _native_prefill_graph_enabled() and chunk_mask is None:
                out = self.rwkv7_prefill_native(
                    input_ids[:, start:end],
                    past_key_values=past,
                    logits_to_keep=logits_to_keep if end == total else 1,
                    return_dict=True,
                )
            else:
                out = self(
                    input_ids[:, start:end],
                    attention_mask=chunk_mask,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=logits_to_keep if end == total else 1,
                    return_dict=True,
                    **chunk_kwargs,
                )
            past = out.past_key_values
        if out is None:
            raise RuntimeError("unreachable: chunked prefill produced no output")
        if hasattr(out.past_key_values, "_seen_tokens"):
            out.past_key_values._seen_tokens = initial_seen + total
        if not return_dict:
            return out.logits, out.past_key_values
        return out

    @torch.no_grad()
    def rwkv7_speculative_generate(
        self,
        input_ids: torch.LongTensor,
        draft_model: torch.nn.Module,
        max_new_tokens: int = 32,
        draft_tokens: int = 4,
        eos_token_id: int | list[int] | tuple[int, ...] | None = None,
        return_stats: bool = False,
        logits_to_keep: int = 1,
        **forward_kwargs,
    ):
        """Greedy HF-compatible speculative decoding for RWKV draft models.

        This is an initial inference-only helper for the HF track: a smaller
        RWKV/HF model proposes up to `draft_tokens` tokens, while this target
        model verifies them with the normal cached HF `forward()` path. The
        helper intentionally stays model-API based instead of depending on a
        serving runtime, so it works with remote-code HF models and can later be
        wired into faster native-token backends.

        Scope: batch size 1 and greedy decoding. If a draft token mismatches the
        target greedy token, the target token is emitted and the draft cache is
        rebuilt from the accepted prefix.
        """
        if self.training:
            raise RuntimeError("rwkv7_speculative_generate is inference-only; call model.eval() first")
        if draft_model is None:
            raise ValueError("rwkv7_speculative_generate requires a draft_model")
        if getattr(draft_model, "training", False):
            raise RuntimeError("draft_model must be in eval mode for speculative decoding")
        if input_ids.dim() != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("rwkv7_speculative_generate currently supports input_ids shaped [1, seq]")
        if int(input_ids.shape[1]) <= 0:
            raise ValueError("rwkv7_speculative_generate requires at least one prompt token")
        max_new_tokens = int(max_new_tokens)
        draft_tokens = int(draft_tokens)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if draft_tokens <= 0:
            raise ValueError("draft_tokens must be positive")
        if max_new_tokens == 0:
            stats = {
                "generated_tokens": 0,
                "proposed_tokens": 0,
                "accepted_tokens": 0,
                "corrected_tokens": 0,
                "resyncs": 0,
                "resync_tokens": 0,
                "full_resync_tokens": 0,
                "resync_saved_tokens": 0,
                "target_forward_calls": 0,
                "draft_forward_calls": 0,
                "acceptance_rate": None,
            }
            return {"sequences": input_ids, "stats": stats} if return_stats else input_ids

        eos_ids = {int(eos_token_id)} if isinstance(eos_token_id, int) else (
            {int(v) for v in eos_token_id} if eos_token_id is not None else set()
        )
        prefill_kwargs = dict(forward_kwargs)
        step_kwargs = {
            k: v for k, v in forward_kwargs.items()
            if k not in {"attention_mask", "position_ids", "cache_position", "past_key_values", "use_cache", "return_dict", "logits_to_keep"}
        }

        stats = {
            "generated_tokens": 0,
            "proposed_tokens": 0,
            "accepted_tokens": 0,
            "corrected_tokens": 0,
            "resyncs": 0,
            "resync_tokens": 0,
            "full_resync_tokens": 0,
            "resync_saved_tokens": 0,
            "target_forward_calls": 0,
            "draft_forward_calls": 0,
            "acceptance_rate": None,
        }

        def _forward(model, tokens, past=None, *, prefill: bool = False, keep: int | None = None):
            kwargs = dict(prefill_kwargs if prefill else step_kwargs)
            kwargs.pop("past_key_values", None)
            kwargs.pop("use_cache", None)
            kwargs.pop("return_dict", None)
            kwargs.pop("logits_to_keep", None)
            return model(
                tokens,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                logits_to_keep=logits_to_keep if keep is None else keep,
                **kwargs,
            )

        def _argmax_token(logits: torch.Tensor) -> torch.LongTensor:
            return torch.argmax(logits[:, -1, :], dim=-1).to(device=input_ids.device)

        def _append_token(sequence: torch.LongTensor, token: torch.LongTensor) -> torch.LongTensor:
            return torch.cat([sequence, token.reshape(1, 1).to(sequence.device)], dim=1)

        def _append_tokens(sequence: torch.LongTensor, tokens: list[torch.LongTensor]) -> torch.LongTensor:
            if not tokens:
                return sequence
            return torch.cat([sequence] + [tok.reshape(1, 1).to(sequence.device) for tok in tokens], dim=1)

        def _is_eos(token: torch.LongTensor) -> bool:
            return bool(eos_ids and int(token.reshape(-1)[0].detach().cpu()) in eos_ids)

        def _clone_past(past):
            if hasattr(past, "clone"):
                return past.clone()
            return RWKV7StateCache.from_legacy_cache(past).clone()

        generated = input_ids
        target_out = _forward(self, generated, prefill=True)
        stats["target_forward_calls"] += 1
        target_past = target_out.past_key_values
        target_next = _argmax_token(target_out.logits)

        draft_out = _forward(draft_model, generated, prefill=True)
        stats["draft_forward_calls"] += 1
        draft_past = draft_out.past_key_values
        draft_next = _argmax_token(draft_out.logits)

        while stats["generated_tokens"] < max_new_tokens:
            proposals: list[torch.LongTensor] = []
            # Draft proposal generation advances its cache. Keep the cache at
            # the start of this proposal block so a later mismatch can resync
            # from the accepted prefix plus target correction instead of
            # replaying the full prompt/generated sequence.
            draft_past_before_block = _clone_past(draft_past)
            for _ in range(min(draft_tokens, max_new_tokens - stats["generated_tokens"])):
                proposal = draft_next.reshape(1).to(input_ids.device)
                proposals.append(proposal)
                stats["proposed_tokens"] += 1
                draft_out = _forward(draft_model, proposal.reshape(1, 1), past=draft_past)
                stats["draft_forward_calls"] += 1
                draft_past = draft_out.past_key_values
                draft_next = _argmax_token(draft_out.logits)

            if not proposals:
                break

            proposal_ids = torch.cat([p.reshape(1, 1).to(input_ids.device) for p in proposals], dim=1)
            verify_out = _forward(self, proposal_ids, past=_clone_past(target_past), keep=len(proposals))
            stats["target_forward_calls"] += 1
            verify_logits = verify_out.logits
            target_predictions = [target_next.reshape(1)]
            for pos in range(max(0, len(proposals) - 1)):
                target_predictions.append(torch.argmax(verify_logits[:, pos, :], dim=-1).to(device=input_ids.device))

            accepted_prefix: list[torch.LongTensor] = []
            mismatch = False
            stop_after_append = False
            for idx, proposal in enumerate(proposals):
                expected = target_predictions[idx].reshape(1)
                if int(proposal.reshape(-1)[0]) == int(expected.reshape(-1)[0]):
                    accepted_prefix.append(proposal)
                    stats["accepted_tokens"] += 1
                    stats["generated_tokens"] += 1
                    if _is_eos(proposal) or stats["generated_tokens"] >= max_new_tokens:
                        stop_after_append = True
                        break
                    continue

                generated = _append_tokens(generated, accepted_prefix)
                correction = expected
                generated = _append_token(generated, correction)
                stats["corrected_tokens"] += 1
                stats["generated_tokens"] += 1
                mismatch = True
                if not _is_eos(correction) and stats["generated_tokens"] < max_new_tokens:
                    repair_tokens = torch.cat(
                        [tok.reshape(1, 1).to(input_ids.device) for tok in [*accepted_prefix, correction]],
                        dim=1,
                    )
                    target_out = _forward(self, repair_tokens, past=_clone_past(target_past), keep=1)
                    stats["target_forward_calls"] += 1
                    target_past = target_out.past_key_values
                    target_next = _argmax_token(target_out.logits)
                    draft_out = _forward(draft_model, repair_tokens, past=draft_past_before_block, keep=1)
                    stats["draft_forward_calls"] += 1
                    draft_past = draft_out.past_key_values
                    draft_next = _argmax_token(draft_out.logits)
                    stats["resyncs"] += 1
                    stats["resync_tokens"] += int(repair_tokens.shape[1])
                    stats["full_resync_tokens"] += int(generated.shape[1])
                    stats["resync_saved_tokens"] = max(0, int(stats["full_resync_tokens"]) - int(stats["resync_tokens"]))
                stop_after_append = True
                break

            if not mismatch:
                generated = _append_tokens(generated, accepted_prefix)
                if len(accepted_prefix) == len(proposals):
                    target_past = verify_out.past_key_values
                    target_next = _argmax_token(verify_logits)
                elif not stop_after_append:
                    target_out = _forward(self, generated, prefill=True)
                    stats["target_forward_calls"] += 1
                    target_past = target_out.past_key_values
                    target_next = _argmax_token(target_out.logits)

            if _is_eos(generated[:, -1]) or stats["generated_tokens"] >= max_new_tokens:
                break

        if stats["proposed_tokens"]:
            stats["acceptance_rate"] = float(stats["accepted_tokens"]) / float(stats["proposed_tokens"])
        return {"sequences": generated, "stats": stats} if return_stats else generated

    @staticmethod
    def _native_state_tensor(
        value: torch.Tensor | None,
        shape: tuple[int, ...],
        *,
        device,
        dtype,
        transpose_last: bool = False,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(shape, device=device, dtype=dtype)
        if value.dim() >= 1 and value.shape[0] == 1:
            value = value.squeeze(0)
        if transpose_last:
            value = value.transpose(-1, -2)
        return value.contiguous()

    def _rwkv7_forward_token_native_jit(
        self,
        token: torch.LongTensor,
        past_key_values: RWKV7StateCache,
        return_dict: bool | None = True,
    ):
        """TorchScript block-step fast path for recurrent decode.

        This bridges the standard HF/RWKV7StateCache prefill state into the
        native JIT layout, executes one token, then writes the updated state back
        to the same cache object. It is opt-in via
        `RWKV7_FAST_TOKEN_BACKEND=native_jit`.
        """
        packs = self._rwkv7_native_jit_packs()
        base = self.model
        dtype = base.embeddings.weight.dtype
        device = token.device
        hidden = int(packs[0][1] * packs[0][2])
        batch_size = int(token.numel())
        x = F.embedding(token.reshape(batch_size), base.embeddings.weight).reshape(batch_size, hidden)
        v_first = torch.zeros(batch_size, hidden, device=device, dtype=dtype)

        for p in packs:
            layer_idx, num_heads, head_dim = int(p[0]), int(p[1]), int(p[2])
            state = past_key_values._ensure_layer(layer_idx)
            recurrent_state = self._native_state_tensor(
                state.get("recurrent_state"),
                (batch_size, num_heads, head_dim, head_dim),
                device=device,
                dtype=torch.float32,
                transpose_last=True,
            )
            xpa = self._native_state_tensor(
                state.get("conv_state"),
                (batch_size, hidden),
                device=device,
                dtype=dtype,
            )
            xpf = self._native_state_tensor(
                state.get("ffn_state"),
                (batch_size, hidden),
                device=device,
                dtype=dtype,
            )
            if batch_size == 1:
                x1, xpa1, xpf1, vf1, rs1 = _native_jit_block_step(
                    x.reshape(hidden),
                    xpa.reshape(hidden),
                    xpf.reshape(hidden),
                    v_first.reshape(hidden),
                    recurrent_state.reshape(num_heads, head_dim, head_dim),
                    *p,
                )
                x = x1.reshape(1, hidden)
                xpa = xpa1.reshape(1, hidden)
                xpf = xpf1.reshape(1, hidden)
                v_first = vf1.reshape(1, hidden)
                recurrent_state = rs1.reshape(1, num_heads, head_dim, head_dim)
            else:
                x, xpa, xpf, v_first, recurrent_state = _native_jit_block_step_batched(
                    x,
                    xpa,
                    xpf,
                    v_first,
                    recurrent_state,
                    *p,
                )
            # FLA's cache stores the recurrent matrix transposed relative to
            # the official/native matmul layout. Keep the public cache in FLA
            # layout so callers can still fall back to the normal HF path.
            state["recurrent_state"] = recurrent_state.transpose(-1, -2).contiguous()
            state["conv_state"] = xpa.contiguous()
            state["ffn_state"] = xpf.contiguous()
            state["attn_state"] = None

        past_key_values._seen_tokens += 1
        hidden_states = F.layer_norm(x, [hidden], base.norm.weight, base.norm.bias, 1e-5)
        logits = _linear_direct(self.lm_head, hidden_states).view(batch_size, 1, -1)
        if not return_dict:
            return logits, past_key_values
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _rwkv7_forward_token_native_graph(
        self,
        token: torch.LongTensor,
        past_key_values: RWKV7StateCache,
        return_dict: bool | None = True,
    ):
        """CUDA-graph replay backend for fixed-batch recurrent decode.

        This is an opt-in serving fast path via
        `RWKV7_FAST_TOKEN_BACKEND=native_graph`. Graph runners are cached in a
        small LRU per model instance, keyed by active batch size. Set
        `RWKV7_NATIVE_GRAPH_CACHE_SIZE` to tune the retained runner count.
        """
        packs = self._rwkv7_native_jit_packs(for_graph=True)
        runner = self._rwkv7_native_graph_runner(packs, int(token.numel()))
        logits = runner.replay(token, past_key_values)
        past_key_values._seen_tokens += 1
        if not return_dict:
            return logits, past_key_values
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _rwkv7_forward_token_2d(
        self,
        token: torch.LongTensor,
        past_key_values: RWKV7StateCache,
        return_dict: bool | None = True,
    ):
        """Experimental 2D fast-token path used by layout A/B benchmarks."""
        x = _squeeze_token_dim(self.model.embeddings(token))
        v_first = None
        for layer_idx, layer in enumerate(self.model.layers):
            state = past_key_values._ensure_layer(layer_idx)
            residual = _squeeze_token_dim(layer.pre_norm(x)) if hasattr(layer, "pre_norm") else x
            attn_input = _squeeze_token_dim(layer.attn_norm(residual))
            attn_out, recurrent_state, conv_state, v_first = self._rwkv7_attn_one_2d(
                layer.attn,
                attn_input,
                state,
                v_first,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            ffn_input = _squeeze_token_dim(layer.ffn_norm(hidden_states))
            ffn_out, ffn_state = self._rwkv7_ffn_one_2d(layer.ffn, ffn_input, state)
            x = residual + ffn_out
            state["recurrent_state"] = recurrent_state
            state["conv_state"] = conv_state
            state["ffn_state"] = ffn_state
            state["attn_state"] = None

        past_key_values._seen_tokens += 1
        hidden_states = _squeeze_token_dim(self.model.norm(x))
        logits = _linear_direct(self.lm_head, hidden_states).unsqueeze(1)
        if not return_dict:
            return logits, past_key_values
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _rwkv7_attn_one(self, attn, hidden_states: torch.Tensor, state: dict[str, Any], v_first: torch.Tensor | None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        if seq_len != 1:
            raise ValueError("_rwkv7_attn_one expects [batch, 1, hidden] input")
        num_heads, head_dim = attn.num_heads, attn.head_dim
        conv_cache = state.get("conv_state")
        if conv_cache is None:
            prev = torch.zeros_like(hidden_states)
        else:
            prev = conv_cache.unsqueeze(1) if conv_cache.dim() == 2 else conv_cache
        delta = prev - hidden_states
        xr = torch.addcmul(hidden_states, delta, attn.x_r)
        xw = torch.addcmul(hidden_states, delta, attn.x_w)
        xk = torch.addcmul(hidden_states, delta, attn.x_k)
        xv = torch.addcmul(hidden_states, delta, attn.x_v)
        xa = torch.addcmul(hidden_states, delta, attn.x_a)
        xg = torch.addcmul(hidden_states, delta, attn.x_g)

        r = _linear_direct(attn.r_proj, xr)
        w = -0.6065306597126334 * _lora_direct(attn.w_lora, xw).sigmoid()
        k = _linear_direct(attn.k_proj, xk)
        v = _linear_direct(attn.v_proj, xv)
        if attn.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, _lora_direct(attn.v_lora, xv).sigmoid())
        a = _lora_direct(attn.a_lora, xa).sigmoid()
        g = _lora_direct(attn.g_lora, xg)

        kk = F.normalize(
            (k * attn.k_k).view(batch_size, seq_len, num_heads, head_dim),
            dim=-1,
            p=2.0,
        )
        k = k.addcmul(k * (a - 1), attn.k_a)
        r, w, k, a = (t.view(batch_size, seq_len, num_heads, head_dim) for t in (r, w, k, a))
        v = v.view(batch_size, seq_len, num_heads, attn.head_v_dim)

        o, recurrent_state = fused_mul_recurrent_rwkv7(
            r=r,
            w=w,
            k=k,
            v=v,
            kk=kk,
            a=a,
            scale=1.0,
            initial_state=state.get("recurrent_state"),
            output_final_state=True,
        )
        o = attn.g_norm(o.reshape(batch_size * seq_len, attn.value_dim)).view(batch_size, seq_len, attn.value_dim)
        correction = ((r * k * attn.r_k.view(1, 1, num_heads, head_dim)).sum(-1, keepdim=True) * v).reshape(o.shape)
        o = _linear_direct(attn.o_proj, (o + correction) * g)
        return o, recurrent_state, hidden_states[:, -1], v_first

    def _rwkv7_attn_one_2d(self, attn, hidden_states: torch.Tensor, state: dict[str, Any], v_first: torch.Tensor | None):
        batch_size, hidden_size = hidden_states.shape
        num_heads, head_dim = attn.num_heads, attn.head_dim
        conv_cache = state.get("conv_state")
        if conv_cache is None:
            prev = torch.zeros_like(hidden_states)
        else:
            prev = conv_cache[:, -1] if conv_cache.dim() == 3 else conv_cache
        delta = prev - hidden_states
        xr = torch.addcmul(hidden_states, delta, attn.x_r.view(1, -1))
        xw = torch.addcmul(hidden_states, delta, attn.x_w.view(1, -1))
        xk = torch.addcmul(hidden_states, delta, attn.x_k.view(1, -1))
        xv = torch.addcmul(hidden_states, delta, attn.x_v.view(1, -1))
        xa = torch.addcmul(hidden_states, delta, attn.x_a.view(1, -1))
        xg = torch.addcmul(hidden_states, delta, attn.x_g.view(1, -1))

        r = _linear_direct(attn.r_proj, xr)
        w = -0.6065306597126334 * _lora_direct(attn.w_lora, xw).sigmoid()
        k = _linear_direct(attn.k_proj, xk)
        v = _linear_direct(attn.v_proj, xv)
        if attn.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, _lora_direct(attn.v_lora, xv).sigmoid())
        a = _lora_direct(attn.a_lora, xa).sigmoid()
        g = _lora_direct(attn.g_lora, xg)

        kk = F.normalize(
            (k * attn.k_k.view(1, -1)).view(batch_size, num_heads, head_dim),
            dim=-1,
            p=2.0,
        )
        k = k.addcmul(k * (a - 1), attn.k_a.view(1, -1))
        r, w, k, a = (t.view(batch_size, 1, num_heads, head_dim) for t in (r, w, k, a))
        v = v.view(batch_size, 1, num_heads, attn.head_v_dim)

        o, recurrent_state = fused_mul_recurrent_rwkv7(
            r=r,
            w=w,
            k=k,
            v=v,
            kk=kk.unsqueeze(1),
            a=a,
            scale=1.0,
            initial_state=state.get("recurrent_state"),
            output_final_state=True,
        )
        o = attn.g_norm(o.reshape(batch_size, attn.value_dim))
        correction = ((r * k * attn.r_k.view(1, 1, num_heads, head_dim)).sum(-1, keepdim=True) * v).reshape(
            batch_size, attn.value_dim
        )
        o = _linear_direct(attn.o_proj, (o + correction) * g)
        return o, recurrent_state, hidden_states, v_first

    @staticmethod
    def _rwkv7_ffn_one(ffn, hidden_states: torch.Tensor, state: dict[str, Any]):
        ffn_cache = state.get("ffn_state")
        if ffn_cache is None:
            prev = torch.zeros_like(hidden_states)
        else:
            prev = ffn_cache.unsqueeze(1) if ffn_cache.dim() == 2 else ffn_cache
        delta = prev - hidden_states
        k = torch.addcmul(hidden_states, delta, ffn.x_k.view(1, 1, -1))
        out = _linear_direct(ffn.value, torch.relu(_linear_direct(ffn.key, k)) ** 2)
        return out, hidden_states[:, -1]

    @staticmethod
    def _rwkv7_ffn_one_2d(ffn, hidden_states: torch.Tensor, state: dict[str, Any]):
        ffn_cache = state.get("ffn_state")
        if ffn_cache is None:
            prev = torch.zeros_like(hidden_states)
        else:
            prev = ffn_cache[:, -1] if ffn_cache.dim() == 3 else ffn_cache
        delta = prev - hidden_states
        k = torch.addcmul(hidden_states, delta, ffn.x_k.view(1, -1))
        out = _linear_direct(ffn.value, torch.relu(_linear_direct(ffn.key, k)) ** 2)
        return out, hidden_states

    def _rwkv7_forward_fast_candidate(self, args: tuple[Any, ...], kwargs: dict[str, Any], effective_use_cache: bool):
        if not effective_use_cache or not _fast_forward_enabled():
            return None
        if self.training or torch.is_grad_enabled():
            return None
        if self._rwkv7_has_multi_cuda_device_map():
            return None
        if self._rwkv7_uses_external_quantization() and not _fast_forward_quant_enabled():
            return None
        if kwargs.get("past_key_values") is None:
            return None
        if kwargs.get("inputs_embeds") is not None or kwargs.get("labels") is not None:
            return None
        if kwargs.get("output_attentions") is True or kwargs.get("output_hidden_states") is True:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor):
            return None
        if input_ids.dim() == 1:
            return input_ids if int(input_ids.numel()) > 0 else None
        if input_ids.dim() == 2 and int(input_ids.shape[1]) == 1 and int(input_ids.shape[0]) > 0:
            return input_ids
        return None

    def _rwkv7_forward_prefill_candidate(self, args: tuple[Any, ...], kwargs: dict[str, Any], effective_use_cache: bool):
        if not effective_use_cache or not _fast_prefill_enabled():
            return None
        if self.training or torch.is_grad_enabled():
            return None
        if _native_jit_prefill is None:
            return None
        if self._rwkv7_has_multi_cuda_device_map() or self._rwkv7_uses_external_quantization():
            return None
        if kwargs.get("inputs_embeds") is not None or kwargs.get("labels") is not None:
            return None
        if kwargs.get("output_attentions") is True or kwargs.get("output_hidden_states") is True:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor):
            return None
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2 or int(input_ids.shape[0]) <= 0 or int(input_ids.shape[1]) <= 1:
            return None

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape[:2]) != tuple(input_ids.shape[:2]):
                return None
            try:
                if not bool(torch.all(attention_mask != 0).detach().cpu().item()):
                    return None
            except Exception:
                return None

        past_key_values = kwargs.get("past_key_values")
        if past_key_values is None:
            return input_ids
        try:
            if hasattr(past_key_values, "get_seq_length") and int(past_key_values.get_seq_length()) == 0:
                return input_ids
            if isinstance(past_key_values, RWKV7StateCache) and len(past_key_values) == 0:
                return input_ids
            if isinstance(past_key_values, (tuple, list)) and len(past_key_values) == 0:
                return input_ids
        except Exception:
            return None
        return None

    def forward(self, *args, **kwargs):
        # Normalize `[batch]` single-token input into `[batch, 1]` before the
        # generic FLA path. The native fast-token shortcut accepts either shape,
        # but quantized/bitsandbytes models intentionally fall back to FLA; with
        # 1-D ids the embedding output is `[batch, hidden]`, which makes FLA
        # attention crash because it expects `[batch, seq, hidden]`.
        if args and isinstance(args[0], torch.Tensor) and args[0].dim() == 1:
            args = (args[0].unsqueeze(1),) + tuple(args[1:])
        elif isinstance(kwargs.get("input_ids"), torch.Tensor) and kwargs["input_ids"].dim() == 1:
            kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(1)
        use_cache = kwargs.get("use_cache")
        effective_use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        if effective_use_cache and _fast_cache_enabled():
            past_key_values = kwargs.get("past_key_values")
            if not isinstance(past_key_values, RWKV7StateCache):
                kwargs["past_key_values"] = RWKV7StateCache.from_legacy_cache(past_key_values)
        prefill_input_ids = self._rwkv7_forward_prefill_candidate(args, kwargs, effective_use_cache)
        if prefill_input_ids is not None:
            return_dict = kwargs.get("return_dict")
            if return_dict is None:
                return_dict = getattr(self.config, "use_return_dict", True)
            return self.rwkv7_prefill_native(
                prefill_input_ids,
                past_key_values=kwargs.get("past_key_values"),
                logits_to_keep=kwargs.get("logits_to_keep", 1),
                return_dict=return_dict,
            )
        fast_input_ids = self._rwkv7_forward_fast_candidate(args, kwargs, effective_use_cache)
        if fast_input_ids is not None:
            return_dict = kwargs.get("return_dict")
            if return_dict is None:
                return_dict = getattr(self.config, "use_return_dict", True)
            return self.rwkv7_forward_token(
                fast_input_ids,
                past_key_values=kwargs.get("past_key_values"),
                return_dict=return_dict,
            )
        return super().forward(*args, **kwargs)
