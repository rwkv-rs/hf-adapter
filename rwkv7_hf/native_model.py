# coding=utf-8
"""Canonical FLA-free RWKV-7 model for Hugging Face Transformers.

Inference dispatches to compiled full-sequence prefill and fixed-batch CUDA
graphs when the runtime is eligible, with native JIT and eager PyTorch fallbacks.
Training keeps the ordinary differentiable PyTorch path unless an explicitly
selected native training backend owns the full forward/backward contract.
"""
from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .native import (
    _init_state_batched,
    _native_device_map_transfer,
    _step_token_batched,
    attn_step_batched,
    ffn_step_batched,
)
from .kernel_policy import current_kernel_policy

# Some Transformers releases only copy files directly referenced by the
# remote-code entrypoint. Keep static discovery edges to the dependencies
# reached through native.py/native_jit.py/native_quant_mm*.py without importing
# optional Triton kernels at runtime.
if False:  # pragma: no cover
    from .ada_lora import ada_wagv_lora as _native_ada_lora_dependency_sentinel
    from .ada_sparse_ffn import ada_linear as _native_ada_sparse_ffn_dependency_sentinel
    from .blackwell_norm_mix import blackwell_ffn_add_norm_mix as _native_sm120_norm_mix_dependency_sentinel
    from .dplr_prefill import dplr_chunk_scan as _native_dplr_dependency_sentinel
    from .dplr_prefill_triton import dplr_chunk_scan_triton as _native_dplr_triton_dependency_sentinel
    from .fused_attention_projection import fused_rkv_wag_projection as _native_fused_attn_projection_dependency_sentinel
    from .fused_decode_norm_mix import fused_attn_norm_mix6_decode as _native_fused_decode_norm_mix_dependency_sentinel
    from .fused_elementwise import fused_relu_square as _native_fused_elementwise_dependency_sentinel
    from .fused_ffn import fused_sequence_ffn as _native_fused_ffn_dependency_sentinel
    from .fused_lora import fused_wag_lora as _native_fused_lora_dependency_sentinel
    from .fused_output import fused_attn_output_prepare as _native_fused_output_dependency_sentinel
    from .fused_prefill import fused_prefill_state_prep as _native_fused_prefill_dependency_sentinel
    from .fused_recurrent_update import fused_recurrent_update as _native_fused_recurrent_dependency_sentinel
    from .fused_time_mix import fused_attn_shift_mix as _native_fused_time_mix_dependency_sentinel
    from .kernel_policy import current_kernel_policy as _native_kernel_policy_dependency_sentinel
    from .native_quant_bnb8 import fused_bnb8_relu_square_quant as _native_bnb8_dependency_sentinel
    from .native_quant_policy import normalize_native_mm_policy as _native_quant_policy_dependency_sentinel
    from .native_wkv_fp16 import native_fp16_sequence as _native_wkv_fp16_dependency_sentinel  # noqa: F401
    from .self_chunk_A_fwd import chunk_dplr_fwd_intra as _native_self_chunk_a_dependency_sentinel
    from .self_chunk_cumsum import chunk_rwkv6_fwd_cumsum as _native_self_chunk_cumsum_dependency_sentinel
    from .self_chunk_h_fwd import chunk_dplr_fwd_h as _native_self_chunk_h_dependency_sentinel
    from .self_chunk_o_fwd import chunk_dplr_fwd_o as _native_self_chunk_o_dependency_sentinel
    from .self_chunk_rwkv7 import self_chunk_rwkv7 as _native_self_chunk_dependency_sentinel
    from .self_chunk_utils import check_shared_mem as _native_self_chunk_utils_dependency_sentinel
    from .self_chunk_wy_fwd import prepare_wy_repr_fwd as _native_self_chunk_wy_dependency_sentinel
    from .sm70_linear import sm70_linear as _native_sm70_linear_dependency_sentinel
    from .sm70_quant import w4_linear as _native_sm70_quant_dependency_sentinel
    from .sm70_wagv import sm70_wagv_lora as _native_sm70_wagv_dependency_sentinel

_FALSE_VALUES = {"0", "false", "False", "no", "off"}


def _bnb_skip_policy(policy: str | None = None) -> str:
    if policy is None:
        env_policy = os.environ.get("RWKV7_BNB_SKIP_POLICY")
        if env_policy is None:
            env_policy = str(getattr(current_kernel_policy(), "bnb_skip_policy", "memory"))
        policy = env_policy
    policy = str(policy).strip().lower()
    if policy in {"", "default", "small_lora", "memory", "minimal"}:
        return "memory"
    if policy in {"decode", "decode_hot", "hot", "hybrid"}:
        return "decode_hot"
    if policy in {"output", "output_hot", "o_proj", "o_proj_hot"}:
        return "output_hot"
    if policy in {"prefill", "prefill_hot", "throughput"}:
        return "prefill_hot"
    if policy in {"decode_rk", "rk_dense"}:
        return "decode_rk"
    if policy in {"dense", "all_dense", "no_quant"}:
        return "dense"
    return "memory"


def _bnb_prefill_value_stride() -> int:
    raw = os.environ.get("RWKV7_BNB_PREFILL_VALUE_STRIDE", "8").strip()
    try:
        return min(max(1, int(raw)), 4096)
    except ValueError:
        return 8


def _bnb_int8_threshold_override() -> float | None:
    raw = os.environ.get("RWKV7_BNB_INT8_THRESHOLD")
    if raw is None:
        raw = getattr(current_kernel_policy(), "bnb_int8_threshold", None)
    if raw is None or str(raw).strip().lower() in {"", "default", "library", "none"}:
        return None
    value = float(raw)
    if value < 0.0:
        raise ValueError("RWKV7_BNB_INT8_THRESHOLD must be non-negative")
    return value

try:
    from .native_jit import extract as _native_jit_extract
    from .native_jit import extract_graph as _native_graph_extract
    from .native_jit import prefill as _native_jit_prefill
    from .native_jit import step_batched as _native_jit_step_batched
except Exception:  # pragma: no cover - optional native acceleration
    _native_jit_extract = None
    _native_graph_extract = None
    _native_jit_prefill = None
    _native_jit_step_batched = None

try:
    from .native_graph_runtime import (
        NativeGraphRunner as _NativeGraphRunner,
        native_graph_available as _native_graph_available,
        native_graph_cache_size as _native_graph_cache_size,
        native_graph_runtime_signature as _native_graph_runtime_signature,
        native_graph_stats_template as _native_graph_stats_template,
    )
except Exception:  # pragma: no cover - optional CUDA graph acceleration
    _NativeGraphRunner = None
    _native_graph_available = lambda: False
    _native_graph_cache_size = lambda: 8
    _native_graph_runtime_signature = lambda: ()
    _native_graph_stats_template = lambda: {"requests": 0, "hits": 0, "misses": 0, "evictions": 0}

try:  # pragma: no cover - Transformers version compatibility
    from transformers.cache_utils import Cache as _HFCache
except Exception:  # pragma: no cover
    class _HFCache:
        pass


class _NativeRWKV7LegacyCache(tuple):
    """Tuple-compatible legacy cache carrying RWKV recurrent sequence length."""

    def __new__(cls, state, xpa, xpf, v_first, seen_tokens: int = 0):
        obj = super().__new__(cls, (state, xpa, xpf, v_first))
        obj._seen_tokens = int(seen_tokens)
        return obj

    def get_seq_length(self, layer_idx: int | None = 0, cache_position=None) -> int:
        if layer_idx is not None:
            layer_idx = int(layer_idx)
            state = self[0]
            if layer_idx < 0:
                return 0
            if state is not None and layer_idx >= len(state):
                return 0
            if state is None and layer_idx != 0:
                return 0
        return self._seen_tokens

    @property
    def seen_tokens(self) -> int:
        return int(self._seen_tokens)

    @seen_tokens.setter
    def seen_tokens(self, value: int) -> None:
        self._seen_tokens = int(value)

    def to_legacy_cache(self):
        return self


class NativeRWKV7Cache(_HFCache):
    """HF Cache-contract wrapper for ``NativeRWKV7ForCausalLM`` recurrent state.

    Native decode threads ``(state, xpa, xpf, v_first)`` as its recurrent
    cache (state=list per layer, xpa/xpf=list per layer, v_first is cross-layer).
    That raw tuple does not satisfy the HF ``Cache`` contract that
    ``GenerationMixin``/``Trainer`` want (``get_seq_length`` etc.). This wrapper
    stores the tuple but subclasses the HF ``Cache`` base so it is accepted,
    and stays **iterable** so existing tuple-unpacking in ``forward`` and
    ``_reorder_cache`` keeps working unchanged.
    """

    is_compileable = True

    def __init__(self, state=None, xpa=None, xpf=None, v_first=None, seen_tokens: int = 0):
        # Skip _HFCache.__init__: it allocates CacheLayer wrappers that RWKV
        # recurrent decode does not need (mirrors RWKV7StateCache).
        self._state = state
        self._xpa = xpa
        self._xpf = xpf
        self._v_first = v_first
        self._seen_tokens = int(seen_tokens)
        self.layers = []
        self._rwkv7_cache_metrics = {
            "clones": 0,
            "detaches": 0,
            "device_moves": 0,
            "select_batch_calls": 0,
            "batch_select_calls": 0,
            "batch_select_indices_calls": 0,
            "batch_repeat_interleave_calls": 0,
            "reorder_calls": 0,
            "crops": 0,
            "resets": 0,
            "native_graph_bound_selects": 0,
        }
        self._rwkv7_cache_version = 0
        self._rwkv7_native_graph_bound_runner_id: int | None = None
        self._rwkv7_native_graph_bound_version: int | None = None
        self._rwkv7_native_graph_bound_runner_ref: weakref.ReferenceType | None = None

    def _invalidate_native_graph_binding(self) -> None:
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
        return runner if runner is not None and self._rwkv7_native_graph_bound_runner_id == id(runner) else None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(seen_tokens={self._seen_tokens}, "
            f"batch_size={self.get_batch_size()}, layers={len(self._state) if self._state is not None else 0})"
        )

    def __iter__(self):
        yield self._state
        yield self._xpa
        yield self._xpf
        yield self._v_first

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx):
        return self.to_legacy_cache()[idx]

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized()

    @property
    def is_sliding(self) -> bool:
        return False

    @property
    def max_batch_size(self) -> int | None:
        return self.get_batch_size()

    @property
    def max_cache_len(self) -> int:
        return -1

    @property
    def seen_tokens(self) -> int:
        return int(self._seen_tokens)

    @seen_tokens.setter
    def seen_tokens(self, value: int) -> None:
        self._seen_tokens = int(value)

    @property
    def states(self) -> list[dict[str, torch.Tensor | None]]:
        """RWKV7StateCache-style per-layer view for serving helpers.

        The native backend stores state in tuple-compatible parallel lists, but
        existing dynamic-batch/offload utilities often inspect ``cache.states``
        from the production HF wrapper.  Return a fresh read-only view so those
        helpers can find tensors without mutating the native layout.
        """

        if self._state is None:
            return []
        layer_count = len(self._state)
        xpa = self._xpa if self._xpa is not None else [None] * layer_count
        xpf = self._xpf if self._xpf is not None else [None] * layer_count
        return [
            {
                "recurrent_state": self._state[idx],
                "attn_state": xpa[idx] if idx < len(xpa) else None,
                "conv_state": None,
                "ffn_state": xpf[idx] if idx < len(xpf) else None,
            }
            for idx in range(layer_count)
        ]

    def get_seq_length(self, layer_idx: int | None = 0, cache_position=None) -> int:
        if layer_idx is not None:
            layer_idx = int(layer_idx)
            if layer_idx < 0:
                return 0
            if self._state is not None and layer_idx >= len(self._state):
                return 0
            if self._state is None and layer_idx != 0:
                return 0
        return self._seen_tokens

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor | int | None, layer_idx: int = 0) -> tuple[int, int]:
        if cache_position is None:
            query_len = 0
        elif isinstance(cache_position, torch.Tensor):
            query_len = int(cache_position.numel())
        else:
            query_len = int(cache_position)
        return int(self.get_seq_length(layer_idx)) + query_len, 0

    def to_legacy_cache(self):
        return _NativeRWKV7LegacyCache(
            self._state,
            self._xpa,
            self._xpf,
            self._v_first,
            seen_tokens=self._seen_tokens,
        )

    def clone(self) -> "NativeRWKV7Cache":
        def clone_list(values):
            if values is None:
                return None
            return [v.clone() for v in values]

        out = type(self)(
            clone_list(self._state),
            clone_list(self._xpa),
            clone_list(self._xpf),
            self._v_first.clone() if self._v_first is not None else None,
            seen_tokens=self._seen_tokens,
        )
        out._rwkv7_cache_metrics = dict(self._rwkv7_cache_metrics)
        out._rwkv7_cache_metrics["clones"] += 1
        return out

    def reset(self) -> None:
        self._invalidate_native_graph_binding()
        self._state = None
        self._xpa = None
        self._xpf = None
        self._v_first = None
        self._seen_tokens = 0
        self._rwkv7_cache_metrics["resets"] += 1

    def detach(self, *, inplace: bool = True) -> "NativeRWKV7Cache":
        target = self if inplace else self.clone()
        target._invalidate_native_graph_binding()

        def detach_list(values):
            if values is None:
                return None
            return [v.detach() for v in values]

        target._state = detach_list(target._state)
        target._xpa = detach_list(target._xpa)
        target._xpf = detach_list(target._xpf)
        if target._v_first is not None:
            target._v_first = target._v_first.detach()
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
    ) -> "NativeRWKV7Cache":
        target = self if inplace else self.clone()
        target._invalidate_native_graph_binding()

        def move_tensor(value: torch.Tensor) -> torch.Tensor:
            kwargs = {"non_blocking": non_blocking, "copy": copy}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None and value.is_floating_point():
                kwargs["dtype"] = dtype
            if len(kwargs) == 2:
                return value.clone() if copy else value
            target_device = kwargs.get("device")
            if (
                target_device is not None
                and value.device.type == "cuda"
                and torch.device(target_device).type == "cuda"
                and value.device != torch.device(target_device)
            ):
                moved = _native_device_map_transfer(value, target_device)
                if dtype is not None and moved.dtype != dtype:
                    moved = moved.to(dtype=dtype)
                return moved.clone() if copy else moved
            return value.to(**kwargs)

        def move_list(values):
            if values is None:
                return None
            return [move_tensor(v) for v in values]

        target._state = move_list(target._state)
        target._xpa = move_list(target._xpa)
        target._xpf = move_list(target._xpf)
        if target._v_first is not None:
            target._v_first = move_tensor(target._v_first)
        target._rwkv7_cache_metrics["device_moves"] += 1
        return target

    def get_batch_size(self) -> int | None:
        for values in (self._state, self._xpa, self._xpf):
            if values:
                return int(values[0].shape[0])
        if self._v_first is not None:
            return int(self._v_first.shape[0])
        return None

    def select_batch(self, indices: torch.LongTensor, *, inplace: bool = True) -> "NativeRWKV7Cache":
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices, dtype=torch.long)
        else:
            indices = indices.to(dtype=torch.long)
        target = self if inplace else type(self)(
            self._state,
            self._xpa,
            self._xpf,
            self._v_first,
            seen_tokens=self._seen_tokens,
        )
        target._rwkv7_cache_metrics = dict(self._rwkv7_cache_metrics)
        runner = target._native_graph_bound_runner() if inplace else None
        if runner is not None and target.get_batch_size() == int(indices.numel()):
            if hasattr(runner, "reorder_batch_inplace") and runner.reorder_batch_inplace(indices):
                target._rwkv7_cache_metrics["select_batch_calls"] += 1
                target._rwkv7_cache_metrics["native_graph_bound_selects"] += 1
                return target
        target._invalidate_native_graph_binding()

        def select_list(values):
            if values is None:
                return None
            return [v.index_select(0, _native_device_map_transfer(indices, v.device)) for v in values]

        target._state = select_list(target._state)
        target._xpa = select_list(target._xpa)
        target._xpf = select_list(target._xpf)
        if target._v_first is not None:
            target._v_first = target._v_first.index_select(
                0, _native_device_map_transfer(indices, target._v_first.device)
            )
        target._rwkv7_cache_metrics["select_batch_calls"] += 1
        return target

    def batch_select(self, indices: torch.LongTensor, *, inplace: bool = True) -> "NativeRWKV7Cache":
        target = self.select_batch(indices, inplace=inplace)
        target._rwkv7_cache_metrics["batch_select_calls"] += 1
        return target

    def compact(self, indices: torch.LongTensor, *, inplace: bool = True) -> "NativeRWKV7Cache":
        return self.batch_select(indices, inplace=inplace)

    def batch_select_indices(self, indices: torch.Tensor):
        target = self.select_batch(indices, inplace=True)
        target._rwkv7_cache_metrics["batch_select_indices_calls"] += 1
        return target

    def batch_repeat_interleave(self, repeats: int):
        repeats = int(repeats)
        if repeats <= 0:
            raise ValueError("NativeRWKV7Cache.batch_repeat_interleave requires repeats > 0")

        self._invalidate_native_graph_binding()

        def repeat_list(values):
            if values is None:
                return None
            return [v.repeat_interleave(repeats, dim=0) for v in values]

        self._state = repeat_list(self._state)
        self._xpa = repeat_list(self._xpa)
        self._xpf = repeat_list(self._xpf)
        if self._v_first is not None:
            self._v_first = self._v_first.repeat_interleave(repeats, dim=0)
        self._rwkv7_cache_metrics["batch_repeat_interleave_calls"] += 1
        return self

    def crop(self, max_length: int):
        max_length = int(max_length)
        target_length = self._seen_tokens + max_length if max_length < 0 else max_length
        if target_length >= self._seen_tokens:
            return self
        if target_length <= 0:
            self._rwkv7_cache_metrics["crops"] += 1
            self.reset()
            return self
        raise NotImplementedError(
            "NativeRWKV7Cache cannot crop recurrent state to a shorter positive prefix; "
            "run a fresh prefill for that prefix instead."
        )

    def _is_initialized(self, layer_idx: int | None = None) -> bool:
        if self._state is None or self._xpa is None or self._xpf is None or self._v_first is None:
            return False
        if layer_idx is not None and (int(layer_idx) < 0 or int(layer_idx) >= len(self._state)):
            return False
        return True

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        return self._is_initialized(layer_idx) and self._seen_tokens > 0

    def update(self, *args, **kwargs):
        raise NotImplementedError(
            "NativeRWKV7Cache is not a Transformer KV cache; update it through "
            "NativeRWKV7ForCausalLM.forward(..., past_key_values=...)."
        )

    def update_recurrent_state(self, *args, **kwargs):
        raise NotImplementedError(
            "NativeRWKV7Cache stores RWKV-7 state as (state, xpa, xpf, v_first); "
            "update it through NativeRWKV7ForCausalLM.forward(..., past_key_values=...)."
        )

    def update_conv_state(self, *args, **kwargs):
        raise NotImplementedError("NativeRWKV7Cache does not have convolution state.")

    def update_indexer(self, *args, **kwargs):
        raise NotImplementedError("NativeRWKV7Cache does not have an indexer key cache.")

    def early_initialization(self, *args, **kwargs):
        raise NotImplementedError(
            "NativeRWKV7Cache cannot be early-initialized as a Transformer KV cache; "
            "native recurrent state is initialized by NativeRWKV7ForCausalLM.forward."
        )

    def offload(self, *args, **kwargs):
        raise NotImplementedError("Use NativeRWKV7Cache.to(device='cpu') to offload native recurrent state.")

    def prefetch(self, *args, **kwargs):
        raise NotImplementedError("Use NativeRWKV7Cache.to(device=...) to restore native recurrent state.")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        target = self.select_batch(beam_idx, inplace=True)
        target._rwkv7_cache_metrics["reorder_calls"] += 1
        return target

    def rwkv7_cache_metrics(self) -> dict:
        metrics = dict(self._rwkv7_cache_metrics)
        metrics.update(
            {
                "seen_tokens": int(self._seen_tokens),
                "batch_size": self.get_batch_size(),
                "layers": len(self._state) if self._state is not None else 0,
            }
        )
        return metrics

    @classmethod
    def from_legacy_cache(cls, legacy, seen_tokens: int = 0):
        if legacy is None:
            return cls(seen_tokens=seen_tokens)
        if isinstance(legacy, NativeRWKV7Cache):
            return legacy
        seen = int(seen_tokens)
        if hasattr(legacy, "get_seq_length"):
            try:
                legacy_seen = int(legacy.get_seq_length())
                if legacy_seen == 0:
                    return cls(seen_tokens=seen_tokens)
                seen = legacy_seen
            except Exception:
                pass
        if hasattr(legacy, "to_legacy_cache"):
            legacy = legacy.to_legacy_cache()
        if legacy is None:
            return cls(seen_tokens=seen_tokens)
        if isinstance(legacy, (list, tuple)) and len(legacy) == 0:
            return cls(seen_tokens=seen_tokens)
        if not isinstance(legacy, (list, tuple)) or len(legacy) != 4:
            raise TypeError(
                "NativeRWKV7Cache.from_legacy_cache expects None, an empty cache, "
                "or a 4-tuple recurrent cache"
            )
        state, xpa, xpf, v_first = legacy
        return cls(state, xpa, xpf, v_first, seen_tokens=seen)


def _cache_seen(past_key_values) -> int:
    """Best-effort seen-token count from a native cache (wrapper or raw tuple)."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            return 0
    return 0


def _native_cache_tuple_or_none(past_key_values):
    """Return the native recurrent tuple, or ``None`` for an empty HF cache.

    Some Transformers generation paths pre-create a default ``DynamicCache``.
    RWKV recurrent state cannot consume Transformer KV cache layers, but an
    empty cache is equivalent to no cache and should run a full prompt prefill.
    """

    if past_key_values is None:
        return None
    try:
        values = tuple(past_key_values)
    except Exception as exc:
        if _cache_seen(past_key_values) == 0:
            return None
        raise TypeError(f"Unsupported NativeRWKV7 cache type: {type(past_key_values)!r}") from exc
    if len(values) == 4 and all(value is not None for value in values):
        return values
    if _cache_seen(past_key_values) == 0:
        return None
    raise TypeError(
        "NativeRWKV7 expects a NativeRWKV7Cache or 4-tuple recurrent cache; "
        f"got {type(past_key_values)!r} with length {len(values)}"
    )


def _native_cache_batch_size(native_cache) -> int | None:
    if native_cache is None:
        return None
    state, xpa, xpf, v_first = native_cache
    for values in (state, xpa, xpf):
        if values:
            return int(values[0].shape[0])
    if v_first is not None:
        return int(v_first.shape[0])
    return None


def _validate_native_cache_batch_size(native_cache, batch_size: int) -> None:
    cache_batch_size = _native_cache_batch_size(native_cache)
    if cache_batch_size is not None and int(cache_batch_size) != int(batch_size):
        raise ValueError(
            "NativeRWKV7 cache batch size must match inputs "
            f"(cache batch={cache_batch_size}, input batch={batch_size})"
        )


def _copy_native_cache_tuple(native_cache):
    state, xpa, xpf, v_first = native_cache
    return list(state), list(xpa), list(xpf), v_first


def _maybe_legacy_native_cache(cache, return_legacy_cache: bool | None):
    if cache is not None and return_legacy_cache is True:
        return cache.to_legacy_cache()
    return cache


def _native_last_token_slice(value):
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value.reshape(1)
        return value[:, -1:] if value.dim() > 1 else value[-1:]
    return value


def _native_model_jit_enabled() -> bool:
    return os.environ.get("RWKV7_NATIVE_MODEL_JIT", "1") not in _FALSE_VALUES


def _native_model_backend_requested() -> str:
    raw = os.environ.get("RWKV7_NATIVE_MODEL_BACKEND")
    if raw is None:
        return "auto" if _native_model_jit_enabled() else "eager"
    backend = raw.strip().lower()
    aliases = {
        "": "auto",
        "graph": "native_graph",
        "cuda_graph": "native_graph",
        "jit": "native_jit",
        "torch": "eager",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"auto", "eager", "native_jit", "native_graph"}:
        raise ValueError(
            "RWKV7_NATIVE_MODEL_BACKEND must be auto, eager, native_jit, or native_graph; "
            f"got {raw!r}"
        )
    return backend


def _native_prefill_graph_enabled(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    raw = os.environ.get("RWKV7_NATIVE_PREFILL_GRAPH")
    if raw is not None:
        selected = raw not in _FALSE_VALUES
    else:
        policy = current_kernel_policy(torch_module=torch)
        selected = bool(getattr(policy, "prefill_graph", False))
        max_layers = getattr(policy, "prefill_graph_max_layers", None)
        if selected and max_layers is not None:
            selected = (
                num_layers is not None
                and int(num_layers) <= int(max_layers)
            )
        shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(policy, "prefill_graph_model_shapes", ())
            if len(shape) == 4
        }
        if selected and shapes:
            if None in (batch_size, prompt_tokens, hidden_size, num_layers):
                selected = False
            else:
                selected = (
                    int(hidden_size),
                    int(num_layers),
                    int(batch_size),
                    int(prompt_tokens),
                ) in shapes
    return bool(
        selected
        and torch.cuda.is_available()
        and _native_jit_prefill is not None
    )


def _native_prefill_graph_cache_size() -> int:
    policy = current_kernel_policy(torch_module=torch)
    default = int(getattr(policy, "prefill_graph_cache_size", 2))
    try:
        value = int(
            os.environ.get(
                "RWKV7_NATIVE_PREFILL_GRAPH_CACHE_SIZE",
                str(default),
            )
        )
    except ValueError:
        value = default
    return max(1, min(value, 16))


def _native_prefill_external_quant_graph_enabled() -> bool:
    """Whether external-quant sequence prefill may enter CUDA capture."""

    raw = os.environ.get("RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT_GRAPH")
    if raw is not None:
        return raw not in _FALSE_VALUES
    policy = current_kernel_policy(torch_module=torch)
    return bool(getattr(policy, "native_external_quant_prefill_graph", False))


def _native_prefill_graph_signature() -> tuple[tuple[str, str], ...]:
    """Return every explicit prefill setting that changes a captured graph."""

    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("RWKV7_NATIVE_PREFILL_")
        )
    )


def _validate_native_attention_mask(
    attention_mask,
    batch_size: int,
    seq_len: int,
    device=None,
    *,
    allow_trailing: bool = False,
):
    """Validate and normalize the native/upstream attention-mask contract.

    RWKV recurrent state is order-sensitive and does not have Transformer-style
    random-access KV masking.  All-ones masks are equivalent to no mask.  Masked
    tokens are handled by skipping recurrent-state updates for those batch rows.
    """

    if attention_mask is None:
        return None
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("NativeRWKV7 attention_mask must be a torch.Tensor when provided")
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.view(1, -1)
    if attention_mask.dim() != 2:
        raise ValueError("NativeRWKV7 attention_mask must be shaped [batch, seq]")
    if int(attention_mask.shape[0]) != int(batch_size):
        raise ValueError("NativeRWKV7 attention_mask batch size must match inputs")
    if int(attention_mask.shape[1]) != int(seq_len):
        if not allow_trailing or int(attention_mask.shape[1]) < int(seq_len):
            raise ValueError("NativeRWKV7 attention_mask must have the same [batch, seq] shape as inputs")
        attention_mask = attention_mask[:, -seq_len:]
    mask = attention_mask.to(device=device) if device is not None else attention_mask
    mask = mask[:, :seq_len] != 0
    if mask.numel() and bool(torch.all(mask).detach().cpu().item()):
        return None
    return mask


def _blend_native_recurrent_state(mask: torch.Tensor, old_state, state, old_xpa, xpa, old_xpf, xpf, old_v_first, v_first):
    """Keep old recurrent rows where ``mask`` is false."""

    if bool(torch.all(mask).detach().cpu().item()):
        return state, xpa, xpf, v_first
    state_mask = mask.view(-1, 1, 1, 1)
    hidden_mask = mask.view(-1, 1)
    state = [
        torch.where(_native_device_map_transfer(state_mask, new.device), new, old)
        for old, new in zip(old_state, state, strict=False)
    ]
    xpa = [
        torch.where(_native_device_map_transfer(hidden_mask, new.device), new, old)
        for old, new in zip(old_xpa, xpa, strict=False)
    ]
    xpf = [
        torch.where(_native_device_map_transfer(hidden_mask, new.device), new, old)
        for old, new in zip(old_xpf, xpf, strict=False)
    ]
    v_first = torch.where(
        _native_device_map_transfer(hidden_mask, v_first.device), v_first, old_v_first
    )
    return state, xpa, xpf, v_first


def _validate_native_output_attentions(output_attentions, config) -> None:
    requested = bool(getattr(config, "output_attentions", False) if output_attentions is None else output_attentions)
    if requested:
        raise NotImplementedError("NativeRWKV7 does not expose Transformer-style attention maps")


def _resolve_native_logits_to_keep(logits_to_keep=None, num_logits_to_keep=None):
    if logits_to_keep is None:
        return num_logits_to_keep
    if num_logits_to_keep is None:
        return logits_to_keep
    if isinstance(logits_to_keep, torch.Tensor) or isinstance(num_logits_to_keep, torch.Tensor):
        try:
            left = torch.as_tensor(logits_to_keep).detach().cpu()
            right = torch.as_tensor(num_logits_to_keep).detach().cpu()
            same = torch.equal(left, right)
        except Exception:
            same = False
    else:
        same = int(logits_to_keep) == int(num_logits_to_keep)
    if not same:
        raise ValueError("logits_to_keep and num_logits_to_keep must match when both are provided")
    return logits_to_keep


def _slice_native_logits(logits: torch.Tensor, logits_to_keep):
    if logits_to_keep is None:
        return logits
    if isinstance(logits_to_keep, torch.Tensor):
        if logits_to_keep.dim() == 0:
            logits_to_keep = int(logits_to_keep.detach().cpu().item())
        else:
            positions = logits_to_keep.to(device=logits.device, dtype=torch.long)
            return logits.index_select(1, positions)
    keep = int(logits_to_keep)
    if keep <= 0:
        return logits
    return logits[:, -min(keep, int(logits.shape[1])) :, :]


def _step_token_batched_with_hidden(model, x, state, xpa, xpf, v_first):
    """Native eager token step that also returns per-layer hidden outputs."""

    layer_hiddens = []
    for i, layer in enumerate(model.model.layers):
        layer_device = layer.attn_norm.weight.device
        if x.device != layer_device:
            x = _native_device_map_transfer(x, layer_device)
        if state[i].device != layer_device:
            state[i] = _native_device_map_transfer(state[i], layer_device)
        if xpa[i].device != layer_device:
            xpa[i] = _native_device_map_transfer(xpa[i], layer_device)
        if xpf[i].device != layer_device:
            xpf[i] = _native_device_map_transfer(xpf[i], layer_device)
        if v_first.device != layer_device:
            v_first = _native_device_map_transfer(v_first, layer_device)
        attn = layer.attn
        residual = layer.pre_norm(x) if hasattr(layer, "pre_norm") else x
        h = layer.attn_norm(residual)
        a, xpa[i], state[i], v_first = attn(h, xpa[i], v_first, state[i])
        x = residual + a
        residual = x
        h2 = layer.ffn_norm(x)
        f, xpf[i] = layer.ffn(h2, xpf[i])
        x = residual + f
        layer_hiddens.append(x)
    return x, state, xpa, xpf, v_first, layer_hiddens


class NativeRWKV7Config(PretrainedConfig):
    """Standalone RWKV-7 config carrying converted checkpoint fields."""

    model_type = "rwkv7_native"

    def __init__(self, **kwargs):
        # RWKV checkpoints have an independent output head. PretrainedConfig
        # otherwise defaults this to True, which makes from_pretrained replace
        # lm_head with the embedding matrix before native MM packing.
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.vocab_size = kwargs.get("vocab_size", 65536)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
        self.num_heads = kwargs.get("num_heads", None) or kwargs.get("num_attention_heads", None)
        requested_attention_width = int(
            kwargs.get("attention_hidden_size", self.hidden_size)
        )
        requested_head_dim = kwargs.get("head_dim", None)
        if self.num_heads is None and requested_head_dim is None:
            requested_head_dim = (
                64 if requested_attention_width % 64 == 0 else requested_attention_width
            )
        if requested_head_dim is None:
            if requested_attention_width % int(self.num_heads):
                raise ValueError("attention_hidden_size must be divisible by num_heads")
            requested_head_dim = requested_attention_width // int(self.num_heads)
        self.head_dim = int(requested_head_dim)
        if self.num_heads is None:
            if requested_attention_width % self.head_dim:
                raise ValueError("attention_hidden_size must be divisible by head_dim")
            self.num_heads = requested_attention_width // self.head_dim
        self.attention_hidden_size = int(
            kwargs.get("attention_hidden_size", self.num_heads * self.head_dim)
        )
        if self.attention_hidden_size != int(self.num_heads) * int(self.head_dim):
            raise ValueError("attention_hidden_size must equal num_heads * head_dim")
        self.num_attention_heads = self.num_heads
        self.intermediate_size = kwargs.get("intermediate_size", self.hidden_size * 4)
        self.decay_low_rank_dim = kwargs.get("decay_low_rank_dim", 64)
        self.gate_low_rank_dim = kwargs.get("gate_low_rank_dim", 128)
        self.a_low_rank_dim = kwargs.get("a_low_rank_dim", 64)
        self.v_low_rank_dim = kwargs.get("v_low_rank_dim", 32)
        self.layer_types = kwargs.get("layer_types", None)
        self.use_cache = kwargs.get("use_cache", True)
        self.use_native_mm8 = kwargs.get("use_native_mm8", False)
        self.native_mm8_min_params = kwargs.get("native_mm8_min_params", 8_000_000)
        self.native_mm8_policy = kwargs.get("native_mm8_policy", "memory")
        self.use_native_mm4 = kwargs.get("use_native_mm4", False)
        self.native_mm4_min_params = kwargs.get("native_mm4_min_params", 8_000_000)
        self.native_mm4_policy = kwargs.get("native_mm4_policy", "memory")
        self.native_mm4_group_size = kwargs.get("native_mm4_group_size", 0)
        self.native_mm4_group_policy = kwargs.get("native_mm4_group_policy", "all")
        if getattr(self, "auto_map", None) is None:
            self.auto_map = {
                "AutoConfig": "native_model.NativeRWKV7Config",
                "AutoModel": "native_model.NativeRWKV7Model",
                "AutoModelForCausalLM": "native_model.NativeRWKV7ForCausalLM",
            }


class _LoRA(nn.Module):
    """Matches converted keys: ``*_lora.lora.{0,2}.weight`` / ``lora.2.bias``."""

    def __init__(
        self,
        input_size: int,
        low_rank: int,
        bias: bool,
        *,
        output_size: int | None = None,
    ):
        super().__init__()
        output_size = input_size if output_size is None else int(output_size)
        self.lora = nn.Sequential(
            nn.Linear(input_size, low_rank, bias=False),
            nn.Identity(),
            nn.Linear(low_rank, output_size, bias=bias),
        )

    def forward(self, x):
        return self.lora(x)


class NativeRWKV7Attention(nn.Module):
    """TMix module with attributes consumed by ``rwkv7_hf.native.attn_step``."""

    def __init__(self, config: NativeRWKV7Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attention_hidden_size = getattr(
            config,
            "attention_hidden_size",
            config.num_heads * config.head_dim,
        )
        # Wrapper/FLA-compatible value-width aliases used by generic kernel
        # probes. The native implementation currently shares the recurrent
        # key/value projection width, so these are exact, metadata-only names.
        self.value_dim = int(self.attention_hidden_size)
        self.head_v_dim = self.value_dim // self.num_heads
        hidden = config.hidden_size
        attention_hidden = self.attention_hidden_size
        for p in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, p, nn.Parameter(torch.zeros(1, 1, hidden)))
        self.k_k = nn.Parameter(torch.zeros(attention_hidden))
        self.k_a = nn.Parameter(torch.zeros(attention_hidden))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.r_proj = nn.Linear(hidden, attention_hidden, bias=False)
        self.k_proj = nn.Linear(hidden, attention_hidden, bias=False)
        self.v_proj = nn.Linear(hidden, attention_hidden, bias=False)
        self.o_proj = nn.Linear(attention_hidden, hidden, bias=False)
        self.w_lora = _LoRA(
            hidden, config.decay_low_rank_dim, bias=True, output_size=attention_hidden
        )
        self.a_lora = _LoRA(
            hidden, config.a_low_rank_dim, bias=True, output_size=attention_hidden
        )
        self.g_lora = _LoRA(
            hidden, config.gate_low_rank_dim, bias=False, output_size=attention_hidden
        )
        if layer_idx != 0:
            self.v_lora = _LoRA(
                hidden, config.v_low_rank_dim, bias=True, output_size=attention_hidden
            )
        self.g_norm = nn.GroupNorm(
            self.num_heads, attention_hidden, eps=self.head_dim * 1e-5
        )

    def forward(
        self,
        x: torch.Tensor,
        x_prev: torch.Tensor | None = None,
        v_first: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ):
        """Run one native attention step through ``Module.__call__``.

        DeepSpeed ZeRO-3 gathers partitioned parameters from module pre-forward
        hooks.  The original native loop passed ``self`` into the functional
        helper directly, which bypassed this module call for raw TMix
        parameters such as ``x_r`` / ``r_k`` / ``g_norm.weight`` and left them
        sharded under ZeRO-3.  Keeping this thin forward wrapper makes the same
        math usable for normal eager execution and ZeRO-3 resume training.
        """
        train_temp_forward = getattr(self, "_rwkv7_train_temp_forward", None)
        if callable(train_temp_forward):
            return train_temp_forward(x, x_prev)
        if x_prev is None or v_first is None or state is None:
            raise ValueError("native token attention requires x_prev, v_first, and recurrent state")
        return attn_step_batched(self, self.layer_idx, x, x_prev, v_first, state)


class NativeRWKV7FFN(nn.Module):
    """CMix module with attributes consumed by ``rwkv7_hf.native.ffn_step``."""

    def __init__(self, config: NativeRWKV7Config):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, x_prev: torch.Tensor | None = None):
        """Run one native FFN step through ``Module.__call__`` for ZeRO-3 hooks."""
        train_temp_forward = getattr(self, "_rwkv7_train_temp_forward", None)
        if callable(train_temp_forward):
            return train_temp_forward(x)
        if x_prev is None:
            raise ValueError("native token FFN requires x_prev recurrent state")
        return ffn_step_batched(self, x, x_prev)


class NativeRWKV7Layer(nn.Module):
    def __init__(self, config: NativeRWKV7Config, layer_idx: int):
        super().__init__()
        self.attn = NativeRWKV7Attention(config, layer_idx)
        self.ffn = NativeRWKV7FFN(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        if layer_idx == 0:
            self.pre_norm = nn.LayerNorm(config.hidden_size)


class NativeRWKV7Model(PreTrainedModel):
    config_class = NativeRWKV7Config
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["NativeRWKV7Layer"]
    supports_gradient_checkpointing = True
    _tied_weights_keys = {}

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config: NativeRWKV7Config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NativeRWKV7Layer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def resize_token_embeddings(self, new_num_tokens: int | None = None, *args, **kwargs):
        """RWKV checkpoints use the fixed official trie vocabulary."""

        if new_num_tokens is None or int(new_num_tokens) == int(self.config.vocab_size):
            return self.get_input_embeddings()
        raise NotImplementedError(
            "RWKV-7 uses the fixed official trie vocabulary; changing vocab size "
            "with resize_token_embeddings is not supported by this adapter."
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask=None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        position_ids=None,
        cache_position=None,
        token_type_ids=None,
        head_mask=None,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("NativeRWKV7Model accepts either input_ids or inputs_embeds, not both")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("NativeRWKV7Model requires input_ids or inputs_embeds")
        if input_ids is not None:
            if input_ids.dim() == 1:
                input_ids = input_ids.view(1, -1)
            if input_ids.dim() != 2:
                raise ValueError("NativeRWKV7Model expects input_ids shaped [batch, seq]")
            batch_size, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
            device, dtype = input_ids.device, self.embeddings.weight.dtype
        else:
            if inputs_embeds.dim() != 3:
                raise ValueError("NativeRWKV7Model expects inputs_embeds shaped [batch, seq, hidden]")
            if int(inputs_embeds.shape[-1]) != int(self.config.hidden_size):
                raise ValueError("NativeRWKV7Model inputs_embeds last dimension must match hidden_size")
            batch_size, seq_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
            device, dtype = inputs_embeds.device, inputs_embeds.dtype
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("NativeRWKV7Model requires a non-empty batch and sequence")
        native_cache = _native_cache_tuple_or_none(past_key_values)
        _validate_native_cache_batch_size(native_cache, batch_size)
        native_attention_mask = _validate_native_attention_mask(
            attention_mask,
            batch_size,
            seq_len,
            device=device,
            allow_trailing=native_cache is not None,
        )
        _validate_native_output_attentions(output_attentions, self.config)
        if return_dict is None:
            return_dict = bool(getattr(self.config, "return_dict", True))
        output_hidden_states = bool(
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        use_cache = bool(self.config.use_cache if use_cache is None else use_cache)

        class _Runner:
            pass

        runner = _Runner()
        runner.model = self
        if native_cache is None:
            state, xpa, xpf, v_first = _init_state_batched(runner, batch_size, device, dtype)
            seen = seq_len
        else:
            state, xpa, xpf, v_first = _copy_native_cache_tuple(native_cache)
            seen = _cache_seen(past_key_values) + seq_len

        final_hidden = []
        hidden_buckets = [[] for _ in range(self.config.num_hidden_layers + 1)] if output_hidden_states else None
        hidden_size = int(self.config.hidden_size)
        last_normed = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        last_layer_hiddens = (
            [torch.zeros(batch_size, hidden_size, device=device, dtype=dtype) for _ in range(self.config.num_hidden_layers + 1)]
            if hidden_buckets is not None
            else None
        )
        for t in range(seq_len):
            x = inputs_embeds[:, t] if inputs_embeds is not None else self.embeddings(input_ids[:, t])
            token_mask = native_attention_mask[:, t] if native_attention_mask is not None else None
            if token_mask is not None:
                old_state, old_xpa, old_xpf, old_v_first = list(state), list(xpa), list(xpf), v_first
            if hidden_buckets is not None:
                emb_hidden = x
                if token_mask is not None:
                    emb_hidden = torch.where(token_mask.view(batch_size, 1).to(x.device), emb_hidden, last_layer_hiddens[0])
                hidden_buckets[0].append(emb_hidden)
                x, state, xpa, xpf, v_first, layer_hiddens = _step_token_batched_with_hidden(
                    runner, x, state, xpa, xpf, v_first
                )
                normed = self.norm(x)
                if token_mask is not None:
                    state, xpa, xpf, v_first = _blend_native_recurrent_state(
                        token_mask, old_state, state, old_xpa, xpa, old_xpf, xpf, old_v_first, v_first
                    )
                    mask_h = token_mask.view(batch_size, 1).to(normed.device)
                    normed = torch.where(mask_h, normed, last_normed)
                    layer_hiddens = [
                        torch.where(mask_h.to(layer_hidden.device), layer_hidden, last_layer_hiddens[layer_idx + 1])
                        for layer_idx, layer_hidden in enumerate(layer_hiddens)
                    ]
                for layer_idx, layer_hidden in enumerate(layer_hiddens, start=1):
                    hidden_buckets[layer_idx].append(normed if layer_idx == self.config.num_hidden_layers else layer_hidden)
                last_layer_hiddens = [emb_hidden] + [
                    normed if layer_idx == self.config.num_hidden_layers else layer_hidden
                    for layer_idx, layer_hidden in enumerate(layer_hiddens, start=1)
                ]
            else:
                x, state, xpa, xpf, v_first = _step_token_batched(runner, x, state, xpa, xpf, v_first)
                normed = self.norm(x)
                if token_mask is not None:
                    state, xpa, xpf, v_first = _blend_native_recurrent_state(
                        token_mask, old_state, state, old_xpa, xpa, old_xpf, xpf, old_v_first, v_first
                    )
                    normed = torch.where(token_mask.view(batch_size, 1).to(normed.device), normed, last_normed)
            final_hidden.append(normed)
            last_normed = normed

        last_hidden_state = torch.stack(final_hidden, dim=1)
        new_cache = NativeRWKV7Cache(state, xpa, xpf, v_first, seen_tokens=seen) if use_cache else None
        hidden_states = None
        if hidden_buckets is not None:
            hidden_states = tuple(torch.stack(bucket, dim=1) for bucket in hidden_buckets)
        if not return_dict:
            values = (last_hidden_state, new_cache, hidden_states)
            return tuple(v for v in values if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=new_cache,
            hidden_states=hidden_states,
        )


class _NativePrefillGraphRunner:
    """Fixed-shape CUDA graph for the canonical Native HF prefill path."""

    def __init__(
        self,
        owner: "NativeRWKV7ForCausalLM",
        packs,
        batch_size: int,
        prompt_tokens: int,
        logits_to_keep: int | None,
    ) -> None:
        if not _native_prefill_graph_enabled(
            batch_size,
            prompt_tokens,
            int(owner.config.hidden_size),
            int(owner.config.num_hidden_layers),
        ):
            raise RuntimeError("native prefill graph is not enabled or available")
        self.owner = owner
        self.packs = packs
        self.batch_size = int(batch_size)
        self.prompt_tokens = int(prompt_tokens)
        self.logits_to_keep = None if logits_to_keep is None else int(logits_to_keep)
        self.runtime_signature = _native_prefill_graph_signature()
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
        self.logits: torch.Tensor | None = None
        self.state_outputs: list[torch.Tensor] = []
        self.xpa_outputs: list[torch.Tensor] = []
        self.xpf_outputs: list[torch.Tensor] = []
        attention_hidden = int(
            getattr(
                owner.config,
                "attention_hidden_size",
                owner.config.num_heads * owner.config.head_dim,
            )
        )
        self.v_first = torch.zeros(
            self.batch_size,
            attention_hidden,
            device=self.device,
            dtype=self.dtype,
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self._bound_cache_ref: weakref.ReferenceType[NativeRWKV7Cache] | None = None
        self._capture()

    def _run_once(self):
        return _native_jit_prefill(
            self.owner,
            self.input_ids,
            self.packs,
            logits_to_keep=self.logits_to_keep,
        )

    def _capture(self) -> None:
        warm = torch.cuda.Stream(device=self.device)
        warm.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warm), torch.inference_mode():
            for _ in range(3):
                self._run_once()
        torch.cuda.current_stream(self.device).wait_stream(warm)
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            outputs = self._run_once()
        self.logits, self.state_outputs, self.xpa_outputs, self.xpf_outputs = outputs

    def matches(
        self,
        batch_size: int,
        prompt_tokens: int,
        logits_to_keep: int | None,
    ) -> bool:
        normalized_keep = None if logits_to_keep is None else int(logits_to_keep)
        return bool(
            self.batch_size == int(batch_size)
            and self.prompt_tokens == int(prompt_tokens)
            and self.logits_to_keep == normalized_keep
            and self.runtime_signature == _native_prefill_graph_signature()
        )

    def _detach_bound_cache(self) -> None:
        previous = self._bound_cache_ref() if self._bound_cache_ref is not None else None
        if previous is None:
            return
        # The first decode replay replaces/binds the cache to its own stable
        # buffers. In that common generate flow this prefill graph no longer
        # owns the cache and can immediately reuse its outputs.
        if not previous._native_graph_bound_to(self):
            self._bound_cache_ref = None
            return
        previous._state = [value.clone() for value in previous._state]
        previous._xpa = [value.clone() for value in previous._xpa]
        previous._xpf = [value.clone() for value in previous._xpf]
        previous._v_first = previous._v_first.clone()
        previous._invalidate_native_graph_binding()
        self._bound_cache_ref = None

    def replay(
        self,
        input_ids: torch.Tensor,
        *,
        seen_tokens: int,
    ) -> tuple[torch.Tensor, NativeRWKV7Cache]:
        if tuple(input_ids.shape) != (self.batch_size, self.prompt_tokens):
            raise ValueError("native prefill graph input shape changed after capture")
        if input_ids.device != self.device or input_ids.dtype != torch.long:
            raise ValueError("native prefill graph input must be CUDA int64 on the model device")
        if self.graph is None or self.logits is None:
            raise RuntimeError("native prefill graph was not captured")
        self._detach_bound_cache()
        self.input_ids.copy_(input_ids)
        self.graph.replay()
        cache = NativeRWKV7Cache(
            self.state_outputs,
            self.xpa_outputs,
            self.xpf_outputs,
            self.v_first,
            seen_tokens=int(seen_tokens),
        )
        cache._bind_native_graph_runner(self)
        self._bound_cache_ref = weakref.ref(cache)
        # Public HF forward owns its returned logits. A later replay may reuse
        # the graph buffers before the caller has finished consuming them.
        return self.logits.clone(), cache

    def detach_bound_cache(self) -> None:
        self._detach_bound_cache()


class NativeRWKV7ForCausalLM(PreTrainedModel, GenerationMixin):
    """Experimental batched native PyTorch CausalLM for converted RWKV-7 weights."""

    config_class = NativeRWKV7Config
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["NativeRWKV7Layer"]
    supports_gradient_checkpointing = True
    # Transformers >=5 expects dict-like _tied_weights_keys; RWKV-7 ties nothing.
    _tied_weights_keys = {}
    _rwkv7_bnb_skip_modules = ["lm_head", r".*_lora\.lora\.[02]"]
    _rwkv7_bnb_policy_extra_skips = {
        "memory": [],
        "output_hot": [r".*attn\.o_proj"],
        "decode_rk": [r".*attn\.(r_proj|k_proj)"],
        "decode_hot": [r".*attn\.(r_proj|k_proj|v_proj|o_proj)"],
        "prefill_hot": [r".*attn\.(r_proj|k_proj|v_proj|o_proj)", r".*ffn\.key"],
        "dense": [r".*attn\.(r_proj|k_proj|v_proj|o_proj)", r".*ffn\.(key|value)"],
    }

    @property
    def all_tied_weights_keys(self):
        return {}

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # RWKV recurrent state is not a Transformer KV cache.  Returning False
        # keeps GenerationMixin from pre-allocating DynamicCache for this model
        # family, while forward still treats an empty DynamicCache as no cache
        # for compatibility with older/newer Transformers variants.
        return False

    def __init__(self, config: NativeRWKV7Config):
        super().__init__(config)
        self.model = NativeRWKV7Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False

    @staticmethod
    def _rwkv7_bnb_concrete_skip_modules(
        policy: str,
        config: Any | None = None,
    ) -> list[str]:
        num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
        if num_layers <= 0:
            return []
        prefill_value_stride = _bnb_prefill_value_stride()
        quantized_prefill_values = {
            layer_idx
            for layer_idx in range(num_layers)
            if (layer_idx + 1) % prefill_value_stride == 0
        }
        if policy == "prefill_hot" and not quantized_prefill_values:
            quantized_prefill_values.add(num_layers - 1)
        skips: list[str] = []
        for layer_idx in range(num_layers):
            for lora_name in ("w_lora", "a_lora", "g_lora", "v_lora"):
                for linear_idx in (0, 2):
                    skips.append(
                        f"model.layers.{layer_idx}.attn.{lora_name}.lora.{linear_idx}"
                    )
            if policy == "output_hot":
                skips.append(f"model.layers.{layer_idx}.attn.o_proj")
            if policy in {"decode_rk", "decode_hot", "prefill_hot", "dense"}:
                proj_names = (
                    ("r_proj", "k_proj")
                    if policy == "decode_rk"
                    else ("r_proj", "k_proj", "v_proj", "o_proj")
                )
                for proj_name in proj_names:
                    skips.append(f"model.layers.{layer_idx}.attn.{proj_name}")
            if policy == "prefill_hot":
                skips.append(f"model.layers.{layer_idx}.ffn.key")
                if layer_idx not in quantized_prefill_values:
                    skips.append(f"model.layers.{layer_idx}.ffn.value")
            if policy == "dense":
                for ffn_name in ("key", "value"):
                    skips.append(f"model.layers.{layer_idx}.ffn.{ffn_name}")
        return skips

    @classmethod
    def rwkv7_bnb_skip_modules(
        cls,
        policy: str | None = None,
        config: Any | None = None,
    ) -> list[str]:
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
    def _rwkv7_prepare_bnb_kwargs(
        cls,
        pretrained_model_name_or_path,
        kwargs: dict[str, Any],
    ):
        policy = _bnb_skip_policy(kwargs.pop("rwkv7_bnb_skip_policy", None))
        quantization_config = kwargs.get("quantization_config")
        if quantization_config is None and (
            kwargs.get("load_in_8bit") or kwargs.get("load_in_4bit")
        ):
            from transformers import BitsAndBytesConfig

            bnb_kwargs = {}
            for key in list(kwargs):
                if (
                    key.startswith("bnb_4bit_")
                    or key.startswith("llm_int8_")
                    or key in {"load_in_8bit", "load_in_4bit"}
                ):
                    bnb_kwargs[key] = kwargs.pop(key)
            quantization_config = BitsAndBytesConfig(**bnb_kwargs)
            kwargs["quantization_config"] = quantization_config
        if quantization_config is not None and bool(
            getattr(quantization_config, "load_in_8bit", False)
        ):
            threshold = _bnb_int8_threshold_override()
            if threshold is not None:
                quantization_config.llm_int8_threshold = float(threshold)
        if quantization_config is not None and hasattr(
            quantization_config,
            "llm_int8_skip_modules",
        ):
            config_for_skip = kwargs.get("config")
            if config_for_skip is None:
                try:
                    config_for_skip = cls.config_class.from_pretrained(
                        pretrained_model_name_or_path
                    )
                except Exception:
                    config_for_skip = None
            existing = list(
                getattr(quantization_config, "llm_int8_skip_modules", None) or []
            )
            quantization_config.llm_int8_skip_modules = list(
                dict.fromkeys(
                    [*existing, *cls.rwkv7_bnb_skip_modules(policy, config_for_skip)]
                )
            )
        return policy, quantization_config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load dense weights, then apply optional native W8/W4 quantization.

        The native backend is the Apple/CPU/AMD fallback path, so its quantized
        route must not depend on bitsandbytes.  Persisted ``use_native_mm8`` or
        ``use_native_mm4`` config flags re-pack eligible ``nn.Linear`` modules
        after the fp weights are loaded.  The packed buffers are deterministic
        from the dense weights and therefore do not need to be stored in the
        checkpoint.
        """

        bnb_skip_policy, quantization_config = cls._rwkv7_prepare_bnb_kwargs(
            pretrained_model_name_or_path,
            kwargs,
        )
        loaded = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        # Transformers returns ``(model, loading_info)`` when requested. Keep
        # that standard API shape while applying config-driven packing to the
        # actual model instance.
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        if quantization_config is not None:
            setattr(model, "_rwkv7_bnb_skip_policy", bnb_skip_policy)
            setattr(model.config, "rwkv7_bnb_skip_policy", bnb_skip_policy)
        model.apply_native_mm_quantization_from_config()
        if isinstance(loaded, tuple):
            return (model, *loaded[1:])
        return model

    def apply_native_mm_quantization_from_config(self) -> int:
        """Apply config-driven native MM8/MM4 module replacement.

        Returns the number of replaced modules.  This helper is intentionally
        public-ish for tests and local Apple harnesses that construct a tiny
        native model directly instead of going through ``from_pretrained``.
        """

        use_mm8 = bool(getattr(self.config, "use_native_mm8", False))
        use_mm4 = bool(getattr(self.config, "use_native_mm4", False))
        if not (use_mm8 or use_mm4):
            setattr(self, "_rwkv7_native_mm_quantization", None)
            setattr(self, "_rwkv7_native_mm_replaced_modules", 0)
            return 0
        if use_mm8 and use_mm4:
            raise ValueError("use_native_mm8 and use_native_mm4 are mutually exclusive")
        if use_mm8:
            from .native_quant_mm8 import quantize_model_mm8

            replaced = int(
                quantize_model_mm8(
                    self,
                    min_params=int(getattr(self.config, "native_mm8_min_params", 8_000_000)),
                    policy=str(getattr(self.config, "native_mm8_policy", "memory")),
                )
            )
            quantization = "mm8"
        else:
            from .native_quant_mm4 import quantize_model_mm4

            replaced = int(
                quantize_model_mm4(
                    self,
                    min_params=int(getattr(self.config, "native_mm4_min_params", 8_000_000)),
                    policy=str(getattr(self.config, "native_mm4_policy", "memory")),
                    group_size=int(getattr(self.config, "native_mm4_group_size", 0)),
                    group_policy=str(
                        getattr(self.config, "native_mm4_group_policy", "all")
                    ),
                )
            )
            quantization = "mm4"
        setattr(self, "_rwkv7_native_mm_quantization", quantization)
        setattr(self, "_rwkv7_native_mm_replaced_modules", replaced)
        # Existing JIT packs are dense-weight dependent; invalidate them after
        # swapping modules to avoid stale dense packs across manual calls.
        self._clear_native_jit_pack_cache()
        return replaced

    def _clear_native_jit_pack_cache(self) -> None:
        if hasattr(self, "_rwkv7_native_model_jit_pack_cache"):
            delattr(self, "_rwkv7_native_model_jit_pack_cache")
        if hasattr(self, "_rwkv7_native_graph_pack_cache"):
            delattr(self, "_rwkv7_native_graph_pack_cache")
        if hasattr(self, "_rwkv7_native_adapter_layers_present"):
            delattr(self, "_rwkv7_native_adapter_layers_present")
        if hasattr(self, "_rwkv7_native_external_quantized"):
            delattr(self, "_rwkv7_native_external_quantized")
        if hasattr(self, "_rwkv7_native_model_quantized_cached"):
            delattr(self, "_rwkv7_native_model_quantized_cached")
        self.rwkv7_clear_native_graph_cache()
        self.rwkv7_clear_native_prefill_graph_cache()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        self._clear_native_jit_pack_cache()

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder
        self._clear_native_jit_pack_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self._clear_native_jit_pack_cache()

    def resize_token_embeddings(self, new_num_tokens: int | None = None, *args, **kwargs):
        """RWKV checkpoints use the fixed official trie vocabulary."""

        if new_num_tokens is None or int(new_num_tokens) == int(self.config.vocab_size):
            return self.get_input_embeddings()
        raise NotImplementedError(
            "RWKV-7 uses the fixed official trie vocabulary; changing vocab size "
            "with resize_token_embeddings is not supported by this adapter."
        )

    def rwkv7_native_model_last_decode_backend(self) -> str | None:
        """Return the backend used by the previous native-model decode call."""
        return getattr(self, "_rwkv7_native_model_last_decode_backend", None)

    def rwkv7_native_model_last_prefill_backend(self) -> str | None:
        """Return the backend used by the previous native-model prefill call."""
        return getattr(self, "_rwkv7_native_model_last_prefill_backend", None)

    @torch.inference_mode()
    def rwkv7_prefill_native(
        self,
        input_ids: torch.LongTensor,
        past_key_values: NativeRWKV7Cache | tuple | list | None = None,
        logits_to_keep: int = 1,
        return_dict: bool | None = True,
    ):
        """Inference-only prefill through the native model backend.

        CUDA prompts use the compiled prefill/graph route when eligible, and
        eligible cache continuations reuse compiled prefill with the existing
        recurrent state. CPU, quantized, adapter, and masked calls retain the
        same public contract through the native eager implementation.
        """

        if self.training:
            raise RuntimeError("rwkv7_prefill_native is inference-only; call model.eval() first")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError("rwkv7_prefill_native expects input_ids shaped [batch, seq]")
        if int(input_ids.shape[0]) <= 0 or int(input_ids.shape[1]) <= 0:
            raise ValueError("rwkv7_prefill_native requires a non-empty batch and sequence")

        self._rwkv7_native_model_last_prefill_backend = "native_eager"
        out = self(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=logits_to_keep,
            return_dict=True,
        )
        self._rwkv7_last_fast_prefill_backend = self.rwkv7_native_model_last_prefill_backend()
        if not return_dict:
            return out.logits, out.past_key_values
        return out

    @torch.inference_mode()
    def rwkv7_prefill_chunks(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        chunk_size: int = 2048,
        past_key_values: NativeRWKV7Cache | tuple | list | None = None,
        logits_to_keep: int = 1,
        return_dict: bool | None = True,
        **kwargs,
    ):
        """Prefill a long prompt in recurrent-cache-preserving chunks."""

        if self.training:
            raise RuntimeError("rwkv7_prefill_chunks is inference-only; call model.eval() first")
        if input_ids.dim() != 2:
            raise ValueError("rwkv7_prefill_chunks expects input_ids shaped [batch, seq]")
        if int(input_ids.shape[0]) <= 0 or int(input_ids.shape[1]) <= 0:
            raise ValueError("rwkv7_prefill_chunks requires a non-empty batch and sequence")
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if attention_mask is not None and tuple(attention_mask.shape[:2]) != tuple(input_ids.shape[:2]):
            raise ValueError("attention_mask must have the same [batch, seq] shape as input_ids")

        total = int(input_ids.shape[1])
        initial_seen = _cache_seen(past_key_values)
        past = past_key_values
        out = None
        kwargs.pop("use_cache", None)
        kwargs.pop("past_key_values", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("logits_to_keep", None)
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None
            out = self(
                input_ids=input_ids[:, start:end],
                attention_mask=chunk_mask,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=logits_to_keep if end == total else 1,
                return_dict=True,
                **kwargs,
            )
            past = out.past_key_values
        if out is None:
            raise RuntimeError("unreachable: chunked prefill produced no output")
        if hasattr(out.past_key_values, "seen_tokens"):
            out.past_key_values.seen_tokens = initial_seen + total
        if not return_dict:
            return out.logits, out.past_key_values
        return out

    @torch.inference_mode()
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
        """Greedy batch-one speculative decoding through standard HF calls."""

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
        if max_new_tokens == 0:
            return {"sequences": input_ids, "stats": stats} if return_stats else input_ids

        eos_ids = (
            {int(eos_token_id)}
            if isinstance(eos_token_id, int)
            else ({int(value) for value in eos_token_id} if eos_token_id is not None else set())
        )
        prefill_kwargs = dict(forward_kwargs)
        step_kwargs = {
            key: value
            for key, value in forward_kwargs.items()
            if key
            not in {
                "attention_mask",
                "position_ids",
                "cache_position",
                "past_key_values",
                "use_cache",
                "return_dict",
                "logits_to_keep",
            }
        }

        def _forward(model, tokens, past=None, *, prefill: bool = False, keep: int | None = None):
            call_kwargs = dict(prefill_kwargs if prefill else step_kwargs)
            for key in ("past_key_values", "use_cache", "return_dict", "logits_to_keep"):
                call_kwargs.pop(key, None)
            return model(
                tokens,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                logits_to_keep=logits_to_keep if keep is None else keep,
                **call_kwargs,
            )

        def _argmax_token(logits: torch.Tensor) -> torch.LongTensor:
            return torch.argmax(logits[:, -1, :], dim=-1).to(device=input_ids.device)

        def _append_token(sequence: torch.LongTensor, token: torch.LongTensor) -> torch.LongTensor:
            return torch.cat([sequence, token.reshape(1, 1).to(sequence.device)], dim=1)

        def _append_tokens(sequence: torch.LongTensor, tokens: list[torch.LongTensor]) -> torch.LongTensor:
            if not tokens:
                return sequence
            return torch.cat(
                [sequence] + [token.reshape(1, 1).to(sequence.device) for token in tokens],
                dim=1,
            )

        def _is_eos(token: torch.LongTensor) -> bool:
            return bool(eos_ids and int(token.reshape(-1)[0].detach().cpu()) in eos_ids)

        def _clone_past(past):
            if hasattr(past, "clone"):
                return past.clone()
            return NativeRWKV7Cache.from_legacy_cache(past).clone()

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

            proposal_ids = torch.cat(
                [proposal.reshape(1, 1).to(input_ids.device) for proposal in proposals],
                dim=1,
            )
            verify_out = _forward(
                self,
                proposal_ids,
                past=_clone_past(target_past),
                keep=len(proposals),
            )
            stats["target_forward_calls"] += 1
            verify_logits = verify_out.logits
            target_predictions = [target_next.reshape(1)]
            for position in range(max(0, len(proposals) - 1)):
                target_predictions.append(
                    torch.argmax(verify_logits[:, position, :], dim=-1).to(input_ids.device)
                )

            accepted_prefix: list[torch.LongTensor] = []
            mismatch = False
            stop_after_append = False
            for index, proposal in enumerate(proposals):
                expected = target_predictions[index].reshape(1)
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
                        [
                            token.reshape(1, 1).to(input_ids.device)
                            for token in [*accepted_prefix, correction]
                        ],
                        dim=1,
                    )
                    target_out = _forward(
                        self,
                        repair_tokens,
                        past=_clone_past(target_past),
                        keep=1,
                    )
                    stats["target_forward_calls"] += 1
                    target_past = target_out.past_key_values
                    target_next = _argmax_token(target_out.logits)
                    draft_out = _forward(
                        draft_model,
                        repair_tokens,
                        past=draft_past_before_block,
                        keep=1,
                    )
                    stats["draft_forward_calls"] += 1
                    draft_past = draft_out.past_key_values
                    draft_next = _argmax_token(draft_out.logits)
                    stats["resyncs"] += 1
                    stats["resync_tokens"] += int(repair_tokens.shape[1])
                    stats["full_resync_tokens"] += int(generated.shape[1])
                    stats["resync_saved_tokens"] = max(
                        0,
                        int(stats["full_resync_tokens"]) - int(stats["resync_tokens"]),
                    )
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
            stats["acceptance_rate"] = float(stats["accepted_tokens"]) / float(
                stats["proposed_tokens"]
            )
        return {"sequences": generated, "stats": stats} if return_stats else generated

    def rwkv7_last_fast_token_backend(self) -> str | None:
        """Return the backend selected by the previous fast-token call."""

        return self.rwkv7_native_model_last_decode_backend()

    def rwkv7_last_fast_prefill_backend(self) -> str | None:
        """Return the backend selected by the previous fast-prefill call."""

        return self.rwkv7_native_model_last_prefill_backend()

    def _rwkv7_has_multi_cuda_device_map(self) -> bool:
        """Return whether Accelerate split this native model across CUDA GPUs."""

        devices: set[tuple[str, int | None]] = set()
        device_map = getattr(self, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            for value in device_map.values():
                if isinstance(value, int):
                    devices.add(("cuda", int(value)))
                    continue
                dev = (
                    torch.device(value)
                    if isinstance(value, str) and value != "disk"
                    else None
                )
                if dev is not None and dev.type == "cuda":
                    devices.add(("cuda", dev.index))
            # ``hf_device_map`` is Accelerate's authoritative final placement.
            # Returning here also keeps this predicate O(number of map entries)
            # rather than walking every parameter on each decode token.
            return len(devices) > 1

        cached = getattr(self, "_rwkv7_multi_cuda_device_map_cached", None)
        if isinstance(cached, bool):
            return cached
        param_devices = {
            (param.device.type, param.device.index)
            for param in self.parameters()
            if param.device.type == "cuda"
        }
        result = len(param_devices) > 1
        self._rwkv7_multi_cuda_device_map_cached = result
        return result

    def _rwkv7_prepare_accelerate_pp_output(self) -> None:
        """Keep mixed-device recurrent caches out of Accelerate's P2P hook.

        The native forward already returns logits/hidden outputs to the input
        device with the validated transfer helper.  Accelerate's default
        top-level ``io_same_device`` post-hook would additionally recurse into
        ``NativeRWKV7Cache`` and move every layer state with raw CUDA P2P.  Apart
        from destroying useful cache colocation, that can corrupt state on
        virtual hosts with broken peer copies.  Disable only that redundant
        output relocation; per-module placement hooks remain active.
        """

        if not self._rwkv7_has_multi_cuda_device_map():
            return
        hook = getattr(self, "_hf_hook", None)
        if hook is not None and hasattr(hook, "io_same_device"):
            hook.io_same_device = False

    def _native_prefill_can_run(
        self,
        input_ids: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool,
        use_cache: bool,
        logits_to_keep,
    ) -> bool:
        if _native_model_backend_requested() == "eager":
            return False
        if self._rwkv7_has_multi_cuda_device_map():
            return False
        if self.training or torch.is_grad_enabled() or _native_jit_prefill is None:
            return False
        if not use_cache or input_ids is None or input_ids.dim() != 2 or int(input_ids.shape[1]) <= 1:
            return False
        if input_ids.device.type != "cuda" or self.model.embeddings.weight.device.type != "cuda":
            return False
        if input_ids.device != self.model.embeddings.weight.device:
            return False
        if attention_mask is not None or output_hidden_states or self._native_model_has_adapter_layers():
            return False
        if isinstance(logits_to_keep, torch.Tensor) and logits_to_keep.dim() > 0:
            return False
        return True

    def _native_prefill_graph_can_run(self, batch_size: int, prompt_tokens: int) -> bool:
        # Generic bitsandbytes/external quantized operators may synchronize or
        # allocate while executing and therefore fail CUDA stream capture.
        # Keep their native sequence prefill eager unless an exact-card policy
        # or explicit environment override has validated graph safety.
        if (
            self._native_model_external_quantized()
            and not _native_prefill_external_quant_graph_enabled()
        ):
            return False
        return _native_prefill_graph_enabled(
            int(batch_size),
            int(prompt_tokens),
            int(self.config.hidden_size),
            int(self.config.num_hidden_layers),
        )

    def _native_prefill(
        self,
        input_ids: torch.LongTensor,
        *,
        logits_to_keep,
        seen_tokens: int,
        initial_cache=None,
    ):
        batch_size = int(input_ids.shape[0])
        prompt_tokens = int(input_ids.shape[1])
        if initial_cache is None and self._native_prefill_graph_can_run(
            batch_size,
            prompt_tokens,
        ):
            runner = getattr(self, "_rwkv7_native_prefill_graph_hot_runner", None)
            if not isinstance(runner, _NativePrefillGraphRunner) or not runner.matches(
                batch_size,
                prompt_tokens,
                logits_to_keep,
            ):
                runner = self._native_prefill_graph_runner(
                    batch_size,
                    prompt_tokens,
                    logits_to_keep,
                )
            else:
                stats = getattr(self, "_rwkv7_native_prefill_graph_cache_stats", None)
                if not isinstance(stats, dict):
                    stats = _native_graph_stats_template()
                    self._rwkv7_native_prefill_graph_cache_stats = stats
                stats["requests"] = int(stats.get("requests", 0)) + 1
                stats["hits"] = int(stats.get("hits", 0)) + 1
            logits, cache = runner.replay(input_ids, seen_tokens=int(seen_tokens))
            self._rwkv7_native_model_last_prefill_backend = "native_prefill_graph"
            return logits, cache
        packs = self._native_graph_packs()
        state = xpa = xpf = None
        if initial_cache is not None:
            state, xpa, xpf, _ = _copy_native_cache_tuple(initial_cache)
        logits, state, xpa, xpf = _native_jit_prefill(
            self,
            input_ids,
            packs,
            state=state,
            xpa=xpa,
            xpf=xpf,
            logits_to_keep=logits_to_keep,
        )
        v_first = torch.zeros(
            int(input_ids.shape[0]),
            int(
                getattr(
                    self.config,
                    "attention_hidden_size",
                    self.config.num_heads * self.config.head_dim,
                )
            ),
            device=input_ids.device,
            dtype=self.model.embeddings.weight.dtype,
        )
        cache = NativeRWKV7Cache(state, xpa, xpf, v_first, seen_tokens=int(seen_tokens))
        self._rwkv7_native_model_last_prefill_backend = (
            "native_prefill_continuation" if initial_cache is not None else "native_prefill"
        )
        return logits, cache

    def _native_prefill_graph_runner(
        self,
        batch_size: int,
        prompt_tokens: int,
        logits_to_keep,
    ) -> _NativePrefillGraphRunner:
        packs = self._native_graph_packs()
        weight = self.model.embeddings.weight
        normalized_keep = None if logits_to_keep is None else int(logits_to_keep)
        key = (
            weight.device.type,
            weight.device.index,
            weight.dtype,
            len(packs),
            int(packs[0][1]),
            int(packs[0][2]),
            int(batch_size),
            int(prompt_tokens),
            normalized_keep,
            _native_prefill_graph_signature(),
            str(getattr(self, "_rwkv7_native_mm_quantization", "none")),
        )
        cache = getattr(self, "_rwkv7_native_prefill_graph_runner_cache", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            self._rwkv7_native_prefill_graph_runner_cache = cache
        stats = getattr(self, "_rwkv7_native_prefill_graph_cache_stats", None)
        if not isinstance(stats, dict):
            stats = _native_graph_stats_template()
            self._rwkv7_native_prefill_graph_cache_stats = stats
        stats["requests"] = int(stats.get("requests", 0)) + 1
        runner = cache.get(key)
        if runner is not None:
            stats["hits"] = int(stats.get("hits", 0)) + 1
            cache.move_to_end(key)
            self._rwkv7_native_prefill_graph_hot_runner = runner
            return runner
        stats["misses"] = int(stats.get("misses", 0)) + 1
        while len(cache) >= _native_prefill_graph_cache_size():
            _, evicted = cache.popitem(last=False)
            if getattr(self, "_rwkv7_native_prefill_graph_hot_runner", None) is evicted:
                self._rwkv7_native_prefill_graph_hot_runner = None
            evicted.detach_bound_cache()
            stats["evictions"] = int(stats.get("evictions", 0)) + 1
        runner = _NativePrefillGraphRunner(
            self,
            packs,
            int(batch_size),
            int(prompt_tokens),
            normalized_keep,
        )
        cache[key] = runner
        self._rwkv7_native_prefill_graph_hot_runner = runner
        return runner

    def _native_graph_can_run(
        self,
        token_ids: torch.Tensor | None,
        cache: NativeRWKV7Cache,
        *,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool,
    ) -> bool:
        if self._rwkv7_has_multi_cuda_device_map():
            return False
        requested = _native_model_backend_requested()
        if requested not in {"auto", "native_graph"}:
            return False
        if requested == "auto":
            policy = current_kernel_policy(torch_module=torch)
            max_layers = getattr(policy, "native_graph_max_layers", None)
            num_layers = int(getattr(self.config, "num_hidden_layers", 0) or 0)
            if max_layers is not None and num_layers > int(max_layers):
                return False
        if self.training or torch.is_grad_enabled() or not _native_graph_available():
            return False
        if self._native_model_has_adapter_layers():
            return False
        if self._native_model_quantized() and not self._native_model_native_quant_graph_safe():
            return False
        if token_ids is None or token_ids.dim() != 2 or int(token_ids.shape[1]) != 1:
            return False
        if attention_mask is not None or output_hidden_states or not isinstance(cache, NativeRWKV7Cache):
            return False
        if token_ids.device.type != "cuda" or self.model.embeddings.weight.device.type != "cuda":
            return False
        if token_ids.device != self.model.embeddings.weight.device:
            return False
        if not cache.is_initialized or cache.get_batch_size() != int(token_ids.shape[0]):
            return False
        return True

    def _native_graph_packs(self):
        if _native_graph_extract is None:
            raise RuntimeError("native_graph operand extraction is unavailable")
        weight = self.model.embeddings.weight
        key = (
            weight.device.type,
            weight.device.index,
            weight.dtype,
            str(getattr(self, "_rwkv7_native_mm_quantization", "none")),
            int(getattr(self, "_rwkv7_native_mm_replaced_modules", 0)),
            _native_graph_runtime_signature(),
        )
        cache = getattr(self, "_rwkv7_native_graph_pack_cache", None)
        if cache is None or cache[0] != key:
            packs, _, _, _ = _native_graph_extract(self)
            self._rwkv7_native_graph_pack_cache = (key, packs)
            return packs
        return cache[1]

    def _native_graph_runner(self, batch_size: int):
        if _NativeGraphRunner is None:
            raise RuntimeError("native_graph runtime is unavailable")
        packs = self._native_graph_packs()
        weight = self.model.embeddings.weight
        key = (
            weight.device.type,
            weight.device.index,
            weight.dtype,
            len(packs),
            int(packs[0][1]),
            int(packs[0][2]),
            str(getattr(self, "_rwkv7_native_mm_quantization", "none")),
            int(getattr(self, "_rwkv7_native_mm_replaced_modules", 0)),
            _native_graph_runtime_signature(),
            int(batch_size),
        )
        cache = getattr(self, "_rwkv7_native_graph_runner_cache", None)
        if not isinstance(cache, OrderedDict):
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
        while len(cache) >= _native_graph_cache_size():
            _, evicted = cache.popitem(last=False)
            if hasattr(evicted, "detach_bound_cache"):
                evicted.detach_bound_cache()
            stats["evictions"] = int(stats.get("evictions", 0)) + 1
        runner = _NativeGraphRunner(self, packs, int(batch_size))
        cache[key] = runner
        return runner

    def rwkv7_native_graph_cache_batch_sizes(self) -> list[int]:
        cache = getattr(self, "_rwkv7_native_graph_runner_cache", None)
        if not isinstance(cache, dict):
            return []
        return sorted({int(key[-1]) for key in cache if isinstance(key, tuple) and key})

    def rwkv7_native_graph_cache_stats(self) -> dict[str, Any]:
        stats = dict(getattr(self, "_rwkv7_native_graph_cache_stats", _native_graph_stats_template()))
        requests = int(stats.get("requests", 0))
        hits = int(stats.get("hits", 0))
        stats.update(
            {
                "size": len(self.rwkv7_native_graph_cache_batch_sizes()),
                "limit": _native_graph_cache_size(),
                "batch_sizes": self.rwkv7_native_graph_cache_batch_sizes(),
                "hit_rate": float(hits) / float(requests) if requests else None,
            }
        )
        return stats

    def rwkv7_native_prefill_graph_cache_shapes(self) -> list[tuple[int, int]]:
        cache = getattr(self, "_rwkv7_native_prefill_graph_runner_cache", None)
        if not isinstance(cache, dict):
            return []
        return sorted(
            {
                (int(runner.batch_size), int(runner.prompt_tokens))
                for runner in cache.values()
            }
        )

    def rwkv7_native_prefill_graph_cache_stats(self) -> dict[str, Any]:
        stats = dict(
            getattr(
                self,
                "_rwkv7_native_prefill_graph_cache_stats",
                _native_graph_stats_template(),
            )
        )
        requests = int(stats.get("requests", 0))
        hits = int(stats.get("hits", 0))
        shapes = self.rwkv7_native_prefill_graph_cache_shapes()
        stats.update(
            {
                "size": len(shapes),
                "limit": _native_prefill_graph_cache_size(),
                "shapes": shapes,
                "hit_rate": float(hits) / float(requests) if requests else None,
            }
        )
        return stats

    def rwkv7_native_graph_runner_copy_stats(self) -> dict[str, Any]:
        cache = getattr(self, "_rwkv7_native_graph_runner_cache", None)
        runners = list(cache.items()) if isinstance(cache, dict) else []
        totals = {
            "copy_from_cache_calls": 0,
            "copy_from_cache_fast_skips": 0,
            "bind_cache_calls": 0,
            "bind_cache_fast_skips": 0,
        }
        rows = []
        for key, runner in runners:
            row = {"batch_size": int(key[-1]) if isinstance(key, tuple) and key else None}
            runner_stats = runner.copy_stats() if hasattr(runner, "copy_stats") else {}
            for name in totals:
                value = int(runner_stats.get(name, 0))
                row[name] = value
                totals[name] += value
            rows.append(row)
        copy_calls = totals["copy_from_cache_calls"]
        bind_calls = totals["bind_cache_calls"]
        totals["copy_from_cache_fast_skip_rate"] = (
            float(totals["copy_from_cache_fast_skips"]) / float(copy_calls) if copy_calls else None
        )
        totals["bind_cache_fast_skip_rate"] = (
            float(totals["bind_cache_fast_skips"]) / float(bind_calls) if bind_calls else None
        )
        return {"totals": totals, "runners": rows}

    def rwkv7_clear_native_graph_cache(self) -> int:
        cache = getattr(self, "_rwkv7_native_graph_runner_cache", None)
        if not isinstance(cache, dict):
            self._rwkv7_native_graph_runner_cache = OrderedDict()
            return 0
        runners = list(cache.values())
        for runner in runners:
            if hasattr(runner, "detach_bound_cache"):
                runner.detach_bound_cache()
        cache.clear()
        if not isinstance(cache, OrderedDict):
            self._rwkv7_native_graph_runner_cache = OrderedDict()
        return len(runners)

    def rwkv7_clear_native_prefill_graph_cache(self) -> int:
        cache = getattr(self, "_rwkv7_native_prefill_graph_runner_cache", None)
        if not isinstance(cache, dict):
            self._rwkv7_native_prefill_graph_runner_cache = OrderedDict()
            self._rwkv7_native_prefill_graph_hot_runner = None
            return 0
        runners = list(cache.values())
        for runner in runners:
            runner.detach_bound_cache()
        cache.clear()
        if not isinstance(cache, OrderedDict):
            self._rwkv7_native_prefill_graph_runner_cache = OrderedDict()
        self._rwkv7_native_prefill_graph_hot_runner = None
        return len(runners)

    def rwkv7_reset_native_graph_cache_stats(self) -> dict[str, Any]:
        self._rwkv7_native_graph_cache_stats = _native_graph_stats_template()
        return self.rwkv7_native_graph_cache_stats()

    def rwkv7_reset_native_prefill_graph_cache_stats(self) -> dict[str, Any]:
        self._rwkv7_native_prefill_graph_cache_stats = _native_graph_stats_template()
        return self.rwkv7_native_prefill_graph_cache_stats()

    @torch.inference_mode()
    def rwkv7_warmup_fast_token(
        self,
        batch_sizes: int | list[int] | tuple[int, ...] = (1,),
        backend: str | None = None,
    ) -> dict[int, str]:
        sizes = [int(batch_sizes)] if isinstance(batch_sizes, int) else [int(value) for value in batch_sizes]
        if not sizes or any(value <= 0 for value in sizes):
            raise ValueError("rwkv7_warmup_fast_token requires positive batch sizes")
        requested = _native_model_backend_requested() if backend is None else str(backend).strip().lower()
        warmed = {}
        for batch_size in sizes:
            chosen = requested
            if chosen in {"auto", "native_graph"} and _native_graph_available():
                self._native_graph_runner(batch_size)
                chosen = "native_graph"
            elif chosen in {"auto", "native_jit"} and self._native_jit_packs() is not None:
                chosen = "native_jit"
            else:
                chosen = "eager"
            warmed[batch_size] = chosen
        return warmed

    @torch.inference_mode()
    def rwkv7_forward_token(
        self,
        input_ids: torch.LongTensor,
        past_key_values: NativeRWKV7Cache | tuple | list | None = None,
        return_dict: bool | None = True,
        *,
        copy_logits: bool = True,
    ):
        """Decode one token per sequence through the canonical native backend.

        ``copy_logits=False`` exposes the CUDA-graph output buffer directly for
        serving loops that consume logits before the next replay. The default
        returns an owning tensor and preserves ordinary HF output semantics.
        """

        if self.training:
            raise RuntimeError("rwkv7_forward_token is inference-only; call model.eval() first")
        if input_ids.dim() == 1:
            token_ids = input_ids.reshape(-1, 1)
        elif input_ids.dim() == 2 and int(input_ids.shape[1]) == 1:
            token_ids = input_ids
        else:
            raise ValueError("rwkv7_forward_token expects input_ids shaped [batch] or [batch, 1]")
        if int(token_ids.shape[0]) == 0:
            raise ValueError("rwkv7_forward_token requires a non-empty batch")

        cache = past_key_values
        if cache is not None and not isinstance(cache, NativeRWKV7Cache):
            cache = NativeRWKV7Cache.from_legacy_cache(cache)
        if (
            isinstance(cache, NativeRWKV7Cache)
            and self._native_graph_can_run(
                token_ids,
                cache,
                attention_mask=None,
                output_hidden_states=False,
            )
        ):
            runner = self._native_graph_runner(int(token_ids.shape[0]))
            logits = runner.replay(token_ids, cache, copy_logits=bool(copy_logits))
            cache.seen_tokens = _cache_seen(cache) + 1
            self._rwkv7_native_model_last_decode_backend = "native_graph"
            if not return_dict:
                return logits, cache
            return CausalLMOutputWithPast(logits=logits, past_key_values=cache)

        result = self(
            token_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=return_dict,
        )
        return result

    @torch.inference_mode()
    def rwkv7_forward_one(
        self,
        input_ids: torch.LongTensor,
        past_key_values: NativeRWKV7Cache | tuple | list | None = None,
        return_dict: bool | None = True,
        *,
        copy_logits: bool = True,
    ):
        """Backward-compatible batch-one alias for ``rwkv7_forward_token``."""

        batch_size = 1 if input_ids.dim() == 1 and input_ids.numel() == 1 else int(input_ids.shape[0])
        if batch_size != 1:
            raise ValueError("rwkv7_forward_one expects batch size 1")
        return self.rwkv7_forward_token(
            input_ids,
            past_key_values=past_key_values,
            return_dict=return_dict,
            copy_logits=copy_logits,
        )

    def _native_model_quantized(self) -> bool:
        """True if layer projections were replaced by quantized modules.

        The JIT decode path extracts raw layer ``.weight`` tensors into packs,
        which cannot represent bnb or native MM8/MM4 layer replacements.  When
        layers are quantized, decode must use the eager per-token path whose
        module calls invoke the quantized linears.  ``lm_head``-only quantization
        is safe for JIT because ``native_jit._lm_head`` calls the module.
        Detected by class name to avoid importing optional quantization deps.
        """
        cached = getattr(self, "_rwkv7_native_model_quantized_cached", None)
        if isinstance(cached, bool):
            return cached
        quantized_names = {"Linear4bit", "Linear8bit", "Linear8bitLt", "MM8Linear", "MM4Linear"}
        try:
            detected = any(
                type(module).__name__ in quantized_names
                for module in self.model.layers.modules()
            )
        except Exception:
            return False
        self._rwkv7_native_model_quantized_cached = bool(detected)
        return bool(detected)

    def _native_model_external_quantized(self) -> bool:
        """True when layer projections use non-native quantized wrappers."""

        cached = getattr(self, "_rwkv7_native_external_quantized", None)
        if isinstance(cached, bool):
            return cached
        external_names = {"Linear4bit", "Linear8bit", "Linear8bitLt"}
        try:
            detected = any(
                type(module).__name__ in external_names
                for module in self.model.layers.modules()
            )
        except Exception:
            return False
        self._rwkv7_native_external_quantized = bool(detected)
        return bool(detected)

    def _native_model_native_quant_graph_safe(self) -> bool:
        """Whether all quantized layer operands are graph-safe native modules.

        ``native_jit.extract_graph`` retains MM8/MM4 modules as callables and
        the graph runtime uses their preallocated-output hooks. Generic BnB or
        other external wrappers remain fail-closed.
        """

        native_names = {"MM8Linear", "MM4Linear"}
        external_names = {"Linear4bit", "Linear8bit", "Linear8bitLt"}
        seen_native = False
        try:
            modules = self.model.layers.modules()
        except Exception:
            return False
        for module in modules:
            name = type(module).__name__
            if name in external_names:
                return False
            if name in native_names:
                seen_native = True
                if not callable(getattr(module, "rwkv7_forward_into", None)):
                    return False
        return seen_native

    def _native_model_has_adapter_layers(self) -> bool:
        """True when PEFT-style adapter wrappers sit inside native layers."""

        adapter_metadata_present = bool(
            getattr(self, "peft_config", None)
            or getattr(self, "_hf_peft_config_loaded", False)
        )
        cached = getattr(self, "_rwkv7_native_adapter_layers_present", None)
        if cached is True:
            return True
        if cached is False and not adapter_metadata_present:
            return False
        try:
            modules = self.model.layers.modules()
        except Exception:
            return False
        for module in modules:
            cls = type(module)
            cls_module = getattr(cls, "__module__", "")
            if (
                cls_module.startswith("peft.")
                and (hasattr(module, "base_layer") or hasattr(module, "lora_A") or hasattr(module, "lora_B"))
            ):
                self._rwkv7_native_adapter_layers_present = True
                return True
            if hasattr(module, "base_layer") and (hasattr(module, "lora_A") or hasattr(module, "lora_B")):
                self._rwkv7_native_adapter_layers_present = True
                return True
        self._rwkv7_native_adapter_layers_present = False
        return False

    def _native_model_requires_eager_decode(self) -> bool:
        """Native JIT packs raw dense weights, so wrappers must use eager decode."""

        return self._native_model_quantized() or self._native_model_has_adapter_layers()

    def _native_jit_packs(self):
        if _native_model_backend_requested() == "eager":
            return None
        if not _native_model_jit_enabled() or _native_jit_extract is None or _native_jit_step_batched is None:
            return None
        if self._native_model_requires_eager_decode():
            return None
        weight = self.model.embeddings.weight
        key = (weight.device.type, weight.device.index, weight.dtype)
        cache = getattr(self, "_rwkv7_native_model_jit_pack_cache", None)
        if cache is None or cache[0] != key:
            extracted = _native_jit_extract(self)
            packs = extracted[0] if isinstance(extracted, tuple) and len(extracted) == 4 else extracted
            self._rwkv7_native_model_jit_pack_cache = (key, packs)
            return packs
        return cache[1]

    def _run(
        self,
        token_ids: torch.Tensor | None,
        state,
        xpa,
        xpf,
        v_first,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_jit: bool = False,
        collect_all: bool = False,
        output_hidden_states: bool = False,
    ):
        """Sequentially advance over token ids or embeddings.

        The eager fallback is sequential over time but vectorized over batch.
        Optimized inference prefill and decode are selected before this helper.

        When ``collect_all`` is enabled, returns per-token logits shaped
        ``[batch, seq, vocab]``. This keeps the FLA-free native path compatible
        with standard CausalLM training losses without changing the optimized
        decode path, which only materializes the final token logits.
        """
        if token_ids is None and inputs_embeds is None:
            raise ValueError("NativeRWKV7ForCausalLM._run requires token_ids or inputs_embeds")
        if token_ids is not None and token_ids.dim() != 2:
            raise ValueError("NativeRWKV7ForCausalLM._run expects token ids shaped [batch, seq]")
        if inputs_embeds is not None and inputs_embeds.dim() != 3:
            raise ValueError("NativeRWKV7ForCausalLM._run expects inputs_embeds shaped [batch, seq, hidden]")
        seq_len = int(inputs_embeds.shape[1] if inputs_embeds is not None else token_ids.shape[1])
        batch_size = int(inputs_embeds.shape[0] if inputs_embeds is not None else token_ids.shape[0])
        base = self.model
        x = None
        packs = (
            self._native_jit_packs()
            if use_jit
            and not self._rwkv7_has_multi_cuda_device_map()
            and not output_hidden_states
            and attention_mask is None
            else None
        )
        backend = "native_jit" if packs is not None else "eager"
        all_logits = [] if collect_all else None
        all_hidden = [] if collect_all or output_hidden_states else None
        hidden_buckets = [[] for _ in range(self.config.num_hidden_layers + 1)] if output_hidden_states else None
        hidden_size = int(self.config.hidden_size)
        dtype = inputs_embeds.dtype if inputs_embeds is not None else base.embeddings.weight.dtype
        device = inputs_embeds.device if inputs_embeds is not None else token_ids.device
        last_normed = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        last_layer_hiddens = (
            [torch.zeros(batch_size, hidden_size, device=device, dtype=dtype) for _ in range(self.config.num_hidden_layers + 1)]
            if hidden_buckets is not None
            else None
        )
        for t in range(seq_len):
            x = inputs_embeds[:, t] if inputs_embeds is not None else base.embeddings(token_ids[:, t])
            token_mask = attention_mask[:, t] if attention_mask is not None else None
            if token_mask is not None:
                old_state, old_xpa, old_xpf, old_v_first = list(state), list(xpa), list(xpf), v_first
            if hidden_buckets is not None:
                emb_hidden = x
                if token_mask is not None:
                    emb_hidden = torch.where(token_mask.view(batch_size, 1).to(x.device), emb_hidden, last_layer_hiddens[0])
                hidden_buckets[0].append(emb_hidden)
            if packs is not None:
                x, state, xpa, xpf, v_first = _native_jit_step_batched(self, x, state, xpa, xpf, v_first, packs)
            elif hidden_buckets is not None:
                x, state, xpa, xpf, v_first, layer_hiddens = _step_token_batched_with_hidden(
                    self, x, state, xpa, xpf, v_first
                )
            else:
                x, state, xpa, xpf, v_first = _step_token_batched(self, x, state, xpa, xpf, v_first)
            normed = base.norm(x)
            if token_mask is not None:
                state, xpa, xpf, v_first = _blend_native_recurrent_state(
                    token_mask, old_state, state, old_xpa, xpa, old_xpf, xpf, old_v_first, v_first
                )
                mask_h = token_mask.view(batch_size, 1).to(normed.device)
                normed = torch.where(mask_h, normed, last_normed)
                if hidden_buckets is not None:
                    layer_hiddens = [
                        torch.where(mask_h.to(layer_hidden.device), layer_hidden, last_layer_hiddens[layer_idx + 1])
                        for layer_idx, layer_hidden in enumerate(layer_hiddens)
                    ]
            if hidden_buckets is not None:
                for layer_idx, layer_hidden in enumerate(layer_hiddens, start=1):
                    hidden_buckets[layer_idx].append(
                        normed if layer_idx == self.config.num_hidden_layers else layer_hidden
                    )
                last_layer_hiddens = [emb_hidden] + [
                    normed if layer_idx == self.config.num_hidden_layers else layer_hidden
                    for layer_idx, layer_hidden in enumerate(layer_hiddens, start=1)
                ]
            if all_hidden is not None:
                all_hidden.append(normed)
            if all_logits is not None:
                all_logits.append(self.lm_head(normed))
            last_normed = normed
        if x is None:
            raise ValueError("NativeRWKV7ForCausalLM requires at least one token")
        if use_jit:
            self._rwkv7_native_model_last_decode_backend = (
                None if self._rwkv7_has_multi_cuda_device_map() else backend
            )
        if all_logits is not None:
            logits = torch.stack(all_logits, dim=1)
        else:
            logits = self.lm_head(normed).view(batch_size, 1, -1)
        last_hidden_state = torch.stack(all_hidden, dim=1) if all_hidden is not None else normed.view(batch_size, 1, -1)
        hidden_states = None
        if hidden_buckets is not None:
            hidden_states = tuple(torch.stack(bucket, dim=1) for bucket in hidden_buckets)
        # Accelerate's top-level IO hook normally returns PP outputs to the
        # input device.  Perform that transfer ourselves so hosts with falsely
        # advertised/broken CUDA P2P cannot corrupt logits before generation.
        logits = _native_device_map_transfer(logits, device)
        last_hidden_state = _native_device_map_transfer(last_hidden_state, device)
        if hidden_states is not None:
            hidden_states = tuple(
                _native_device_map_transfer(layer_hidden, device)
                for layer_hidden in hidden_states
            )
        return logits, state, xpa, xpf, v_first, last_hidden_state, hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask=None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep=None,
        num_logits_to_keep=None,
        position_ids=None,
        cache_position=None,
        token_type_ids=None,
        head_mask=None,
        return_legacy_cache: bool | None = None,
        **kwargs,
    ):
        self._rwkv7_prepare_accelerate_pp_output()
        train_temp_forward = getattr(self, "_rwkv7_train_temp_forward", None)
        if callable(train_temp_forward):
            return train_temp_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
                labels=labels,
                logits_to_keep=logits_to_keep,
                num_logits_to_keep=num_logits_to_keep,
                position_ids=position_ids,
                cache_position=cache_position,
                token_type_ids=token_type_ids,
                head_mask=head_mask,
                return_legacy_cache=return_legacy_cache,
                **kwargs,
            )
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("NativeRWKV7ForCausalLM accepts either input_ids or inputs_embeds, not both")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("NativeRWKV7ForCausalLM requires input_ids or inputs_embeds")
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.view(1, -1)
        if input_ids is not None and input_ids.dim() != 2:
            raise ValueError("Experimental NativeRWKV7ForCausalLM expects input_ids shaped [batch, seq]")
        if inputs_embeds is not None:
            if inputs_embeds.dim() != 3:
                raise ValueError("NativeRWKV7ForCausalLM expects inputs_embeds shaped [batch, seq, hidden]")
            if int(inputs_embeds.shape[-1]) != int(self.config.hidden_size):
                raise ValueError("NativeRWKV7ForCausalLM inputs_embeds last dimension must match hidden_size")
        batch_size = int(input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0])
        seq_len = int(input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1])
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("NativeRWKV7ForCausalLM requires a non-empty batch and sequence")
        native_cache = _native_cache_tuple_or_none(past_key_values)
        _validate_native_cache_batch_size(native_cache, batch_size)
        _validate_native_output_attentions(output_attentions, self.config)
        if return_dict is None:
            return_dict = bool(getattr(self.config, "return_dict", True))
        base = self.model
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        dtype = inputs_embeds.dtype if inputs_embeds is not None else base.embeddings.weight.dtype
        native_attention_mask = _validate_native_attention_mask(
            attention_mask,
            batch_size,
            seq_len,
            device=device,
            allow_trailing=native_cache is not None,
        )
        output_hidden_states = bool(
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        use_cache = bool(self.config.use_cache if use_cache is None else use_cache)
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.view(1, -1)
            if tuple(labels.shape[:2]) != (batch_size, seq_len):
                raise ValueError("NativeRWKV7ForCausalLM labels must have the same shape as inputs")
            if native_cache is not None:
                raise ValueError("NativeRWKV7ForCausalLM does not support labels with past_key_values")
            state, xpa, xpf, v_first = _init_state_batched(self, batch_size, device, dtype)
            logits, state, xpa, xpf, v_first, last_hidden_state, hidden_states = self._run(
                input_ids,
                state,
                xpa,
                xpf,
                v_first,
                inputs_embeds=inputs_embeds if input_ids is None else None,
                attention_mask=native_attention_mask,
                use_jit=False,
                collect_all=True,
                output_hidden_states=output_hidden_states,
            )
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if shift_logits.numel() == 0 or not bool((shift_labels != -100).any().detach().cpu().item()):
                loss = logits.float().sum() * 0.0
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]).float(),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            new_cache = NativeRWKV7Cache(state, xpa, xpf, v_first, seen_tokens=seq_len) if use_cache else None
            new_cache = _maybe_legacy_native_cache(new_cache, return_legacy_cache)
            if not return_dict:
                values = (loss, logits, new_cache, hidden_states)
                return tuple(v for v in values if v is not None)
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=new_cache,
                hidden_states=hidden_states,
            )

        logits_to_keep = _resolve_native_logits_to_keep(logits_to_keep, num_logits_to_keep)
        if native_cache is None and self._native_prefill_can_run(
            input_ids,
            attention_mask=native_attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
        ):
            logits, new_cache = self._native_prefill(
                input_ids,
                logits_to_keep=logits_to_keep,
                seen_tokens=seq_len,
            )
            logits = _slice_native_logits(logits, logits_to_keep)
            new_cache = _maybe_legacy_native_cache(new_cache, return_legacy_cache)
            if not return_dict:
                return logits, new_cache
            return CausalLMOutputWithPast(logits=logits, past_key_values=new_cache)
        if native_cache is not None and self._native_prefill_can_run(
            input_ids,
            attention_mask=native_attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
        ):
            logits, new_cache = self._native_prefill(
                input_ids,
                logits_to_keep=logits_to_keep,
                seen_tokens=_cache_seen(past_key_values) + seq_len,
                initial_cache=native_cache,
            )
            logits = _slice_native_logits(logits, logits_to_keep)
            new_cache = _maybe_legacy_native_cache(new_cache, return_legacy_cache)
            if not return_dict:
                return logits, new_cache
            return CausalLMOutputWithPast(logits=logits, past_key_values=new_cache)
        if (
            native_cache is not None
            and use_cache
            and isinstance(past_key_values, NativeRWKV7Cache)
            and self._native_graph_can_run(
                input_ids,
                past_key_values,
                attention_mask=native_attention_mask,
                output_hidden_states=output_hidden_states,
            )
        ):
            runner = self._native_graph_runner(batch_size)
            logits = runner.replay(input_ids, past_key_values)
            past_key_values.seen_tokens = _cache_seen(past_key_values) + 1
            self._rwkv7_native_model_last_decode_backend = "native_graph"
            logits = _slice_native_logits(logits, logits_to_keep)
            new_cache = _maybe_legacy_native_cache(past_key_values, return_legacy_cache)
            if not return_dict:
                return logits, new_cache
            return CausalLMOutputWithPast(logits=logits, past_key_values=new_cache)
        if native_cache is None:
            state, xpa, xpf, v_first = _init_state_batched(self, batch_size, device, dtype)
            toks = input_ids
            use_jit = False
            seen = seq_len
            collect_all = True  # full forward -> all-token logits [B, seq, vocab] (HF CausalLM semantics; DPO/eval need per-token logprobs)
        else:
            state, xpa, xpf, v_first = _copy_native_cache_tuple(native_cache)
            toks = input_ids
            use_jit = seq_len == 1
            seen = _cache_seen(past_key_values) + seq_len
            collect_all = seq_len > 1
        logits, state, xpa, xpf, v_first, last_hidden_state, hidden_states = self._run(
            toks,
            state,
            xpa,
            xpf,
            v_first,
            inputs_embeds=inputs_embeds if toks is None else None,
            attention_mask=native_attention_mask,
            use_jit=use_jit,
            collect_all=collect_all,
            output_hidden_states=output_hidden_states,
        )
        logits = _slice_native_logits(logits, logits_to_keep)
        new_cache = NativeRWKV7Cache(state, xpa, xpf, v_first, seen_tokens=seen) if use_cache else None
        new_cache = _maybe_legacy_native_cache(new_cache, return_legacy_cache)
        if not return_dict:
            values = (logits, new_cache, hidden_states)
            return tuple(v for v in values if v is not None)
        return CausalLMOutputWithPast(logits=logits, past_key_values=new_cache, hidden_states=hidden_states)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx: torch.LongTensor):
        """Beam/select helper for batched native recurrent caches."""
        native_cache = _native_cache_tuple_or_none(past_key_values)
        if native_cache is None:
            return None
        if hasattr(past_key_values, "reorder_cache"):
            return past_key_values.reorder_cache(beam_idx)
        state, xpa, xpf, v_first = native_cache
        index = _native_device_map_transfer(beam_idx, v_first.device)
        seen = _cache_seen(past_key_values)
        reordered = NativeRWKV7Cache(
            [s.index_select(0, _native_device_map_transfer(index, s.device)) for s in state],
            [x.index_select(0, _native_device_map_transfer(index, x.device)) for x in xpa],
            [x.index_select(0, _native_device_map_transfer(index, x.device)) for x in xpf],
            v_first.index_select(0, index),
            seen_tokens=seen,
        )
        return reordered.to_legacy_cache() if isinstance(past_key_values, tuple) else reordered

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids=None,
        head_mask=None,
        return_legacy_cache: bool | None = None,
        **kwargs,
    ):
        # Ensure GenerationMixin gets a cache on the first step. Earlier H1 code
        # only enabled cache after a cache already existed, causing full-prefix
        # recomputation on every greedy token.
        native_cache = _native_cache_tuple_or_none(past_key_values)
        model_inputs = {}
        if native_cache is not None:
            if input_ids is not None:
                model_inputs["input_ids"] = _native_last_token_slice(input_ids)
            elif inputs_embeds is not None:
                model_inputs["inputs_embeds"] = _native_last_token_slice(inputs_embeds)
            else:
                model_inputs["input_ids"] = input_ids
        elif inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids
        use_cache = kwargs.get("use_cache", True)
        if use_cache is None:
            use_cache = True
        model_inputs["past_key_values"] = past_key_values
        model_inputs["use_cache"] = use_cache
        if return_legacy_cache is not None:
            model_inputs["return_legacy_cache"] = return_legacy_cache
        if head_mask is not None:
            model_inputs["head_mask"] = head_mask
        if token_type_ids is not None:
            if native_cache is not None:
                token_type_ids = _native_last_token_slice(token_type_ids)
            model_inputs["token_type_ids"] = token_type_ids
        if kwargs.get("attention_mask") is not None:
            attention_mask = kwargs["attention_mask"]
            model_inputs["attention_mask"] = _native_last_token_slice(attention_mask) if native_cache is not None else attention_mask
        if "logits_to_keep" in kwargs:
            model_inputs["logits_to_keep"] = kwargs["logits_to_keep"]
        if "num_logits_to_keep" in kwargs:
            model_inputs["num_logits_to_keep"] = kwargs["num_logits_to_keep"]
        if "output_hidden_states" in kwargs:
            model_inputs["output_hidden_states"] = kwargs["output_hidden_states"]
        if "output_attentions" in kwargs:
            model_inputs["output_attentions"] = kwargs["output_attentions"]
        if "return_dict" in kwargs:
            model_inputs["return_dict"] = kwargs["return_dict"]
        if "position_ids" in kwargs:
            position_ids = kwargs["position_ids"]
            if native_cache is not None:
                position_ids = _native_last_token_slice(position_ids)
            model_inputs["position_ids"] = position_ids
        if "cache_position" in kwargs:
            cache_position = kwargs["cache_position"]
            if native_cache is not None:
                cache_position = _native_last_token_slice(cache_position)
            model_inputs["cache_position"] = cache_position
        return model_inputs


try:  # pragma: no cover - exercised through save_pretrained/AutoModel smoke.
    NativeRWKV7Config.register_for_auto_class()
    NativeRWKV7Model.register_for_auto_class("AutoModel")
    NativeRWKV7ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
except Exception:
    pass
