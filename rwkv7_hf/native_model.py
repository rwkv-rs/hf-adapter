# coding=utf-8
"""Experimental native RWKV-7 PyTorch model (no mandatory FLA runtime).

This is the H1 native-Transformers track: a correctness-first RWKV-7 model
implemented with plain ``torch.nn`` modules and the math in ``rwkv7_hf.native``.
It loads the same converted HF checkpoints as the production FLA-backed wrapper,
so it can serve as the long-term upstream / AMD / CPU fallback base.

Important: this module is intentionally experimental and sequential. It is not a
replacement for the optimized wrapper path yet. Current scope is batched forward,
incremental greedy generation, and regression tests against the wrapper.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .native import _init_state_batched, _step_token_batched, attn_step_batched, ffn_step_batched

# Some Transformers releases only copy files directly referenced by the
# remote-code entrypoint. Keep static discovery edges to the dependencies
# reached through native.py/native_jit.py/native_quant_mm*.py without importing
# optional Triton kernels at runtime.
if False:  # pragma: no cover
    from .ada_lora import ada_wagv_lora as _native_ada_lora_dependency_sentinel
    from .ada_sparse_ffn import ada_linear as _native_ada_sparse_ffn_dependency_sentinel
    from .dplr_prefill import dplr_chunk_scan as _native_dplr_dependency_sentinel
    from .dplr_prefill_triton import dplr_chunk_scan_triton as _native_dplr_triton_dependency_sentinel
    from .fused_attention_projection import fused_rkv_wag_projection as _native_fused_attn_projection_dependency_sentinel
    from .fused_decode_norm_mix import fused_attn_norm_mix6_decode as _native_fused_decode_norm_mix_dependency_sentinel
    from .fused_elementwise import fused_relu_square as _native_fused_elementwise_dependency_sentinel
    from .fused_lora import fused_wag_lora as _native_fused_lora_dependency_sentinel
    from .fused_output import fused_attn_output_prepare as _native_fused_output_dependency_sentinel
    from .fused_prefill import fused_prefill_state_prep as _native_fused_prefill_dependency_sentinel
    from .fused_recurrent_update import fused_recurrent_update as _native_fused_recurrent_dependency_sentinel
    from .fused_time_mix import fused_attn_shift_mix as _native_fused_time_mix_dependency_sentinel
    from .kernel_policy import current_kernel_policy as _native_kernel_policy_dependency_sentinel
    from .native_quant_policy import normalize_native_mm_policy as _native_quant_policy_dependency_sentinel
    from .sm70_linear import sm70_linear as _native_sm70_linear_dependency_sentinel
    from .sm70_quant import w4_linear as _native_sm70_quant_dependency_sentinel
    from .sm70_wagv import sm70_wagv_lora as _native_sm70_wagv_dependency_sentinel

_FALSE_VALUES = {"0", "false", "False", "no", "off"}

try:
    from .native_jit import extract as _native_jit_extract
    from .native_jit import step_batched as _native_jit_step_batched
except Exception:  # pragma: no cover - optional native acceleration
    _native_jit_extract = None
    _native_jit_step_batched = None

try:  # pragma: no cover - optional Cache base for HF GenerationMixin/Trainer compat
    from fla.models.utils import Cache as _FLACache
except Exception:  # pragma: no cover
    try:
        from transformers.cache_utils import Cache as _FLACache
    except Exception:
        class _FLACache:  # minimal fallback so native_model imports without fla or transformers
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


class NativeRWKV7Cache(_FLACache):
    """HF Cache-contract wrapper for ``NativeRWKV7ForCausalLM`` recurrent state.

    Native decode threads ``(state, xpa, xpf, v_first)`` as its recurrent
    cache (state=list per layer, xpa/xpf=list per layer, v_first is cross-layer).
    That raw tuple does not satisfy the HF ``Cache`` contract that
    ``GenerationMixin``/``Trainer`` want (``get_seq_length`` etc.). This wrapper
    stores the tuple but subclasses the FLA/HF ``Cache`` base so it is accepted,
    and stays **iterable** so existing tuple-unpacking in ``forward`` and
    ``_reorder_cache`` keeps working unchanged.
    """

    is_compileable = True

    def __init__(self, state=None, xpa=None, xpf=None, v_first=None, seen_tokens: int = 0):
        # Skip _FLACache.__init__: it allocates CacheLayer wrappers that RWKV
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
        }

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
        return self.has_previous_state()

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
        self._state = None
        self._xpa = None
        self._xpf = None
        self._v_first = None
        self._seen_tokens = 0
        self._rwkv7_cache_metrics["resets"] += 1

    def detach(self, *, inplace: bool = True) -> "NativeRWKV7Cache":
        target = self if inplace else self.clone()

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

        def move_tensor(value: torch.Tensor) -> torch.Tensor:
            kwargs = {"non_blocking": non_blocking, "copy": copy}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None and value.is_floating_point():
                kwargs["dtype"] = dtype
            if len(kwargs) == 2:
                return value.clone() if copy else value
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

        def select_list(values):
            if values is None:
                return None
            return [v.index_select(0, indices.to(v.device)) for v in values]

        target._state = select_list(target._state)
        target._xpa = select_list(target._xpa)
        target._xpf = select_list(target._xpf)
        if target._v_first is not None:
            target._v_first = target._v_first.index_select(0, indices.to(target._v_first.device))
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

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        if self._state is None or self._xpa is None or self._xpf is None or self._v_first is None:
            return False
        if layer_idx is not None and (int(layer_idx) < 0 or int(layer_idx) >= len(self._state)):
            return False
        return self._seen_tokens > 0

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
    state = [torch.where(state_mask.to(new.device), new, old) for old, new in zip(old_state, state, strict=False)]
    xpa = [torch.where(hidden_mask.to(new.device), new, old) for old, new in zip(old_xpa, xpa, strict=False)]
    xpf = [torch.where(hidden_mask.to(new.device), new, old) for old, new in zip(old_xpf, xpf, strict=False)]
    v_first = torch.where(hidden_mask.to(v_first.device), v_first, old_v_first)
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
        self.head_dim = kwargs.get("head_dim", 64)
        self.num_heads = kwargs.get("num_heads", None) or kwargs.get("num_attention_heads", None)
        if self.num_heads is None:
            requested_attention_width = kwargs.get("attention_hidden_size", None)
            width = self.hidden_size if requested_attention_width is None else int(requested_attention_width)
            if width % self.head_dim:
                raise ValueError(
                    "attention_hidden_size must be divisible by head_dim"
                )
            self.num_heads = width // self.head_dim
        self.attention_hidden_size = int(
            kwargs.get("attention_hidden_size", self.num_heads * self.head_dim)
        )
        if self.attention_hidden_size != self.num_heads * self.head_dim:
            raise ValueError(
                "attention_hidden_size must equal num_heads * head_dim"
            )
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
        self.attention_hidden_size = config.attention_hidden_size
        hidden = config.hidden_size
        attention_hidden = config.attention_hidden_size
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
            hidden,
            config.decay_low_rank_dim,
            bias=True,
            output_size=attention_hidden,
        )
        self.a_lora = _LoRA(
            hidden,
            config.a_low_rank_dim,
            bias=True,
            output_size=attention_hidden,
        )
        self.g_lora = _LoRA(
            hidden,
            config.gate_low_rank_dim,
            bias=False,
            output_size=attention_hidden,
        )
        if layer_idx != 0:
            self.v_lora = _LoRA(
                hidden,
                config.v_low_rank_dim,
                bias=True,
                output_size=attention_hidden,
            )
        self.g_norm = nn.GroupNorm(
            self.num_heads, attention_hidden, eps=self.head_dim * 1e-5
        )

    def forward(self, x: torch.Tensor, x_prev: torch.Tensor, v_first: torch.Tensor, state: torch.Tensor):
        """Run one native attention step through ``Module.__call__``.

        DeepSpeed ZeRO-3 gathers partitioned parameters from module pre-forward
        hooks.  The original native loop passed ``self`` into the functional
        helper directly, which bypassed this module call for raw TMix
        parameters such as ``x_r`` / ``r_k`` / ``g_norm.weight`` and left them
        sharded under ZeRO-3.  Keeping this thin forward wrapper makes the same
        math usable for normal eager execution and ZeRO-3 resume training.
        """
        return attn_step_batched(self, self.layer_idx, x, x_prev, v_first, state)


class NativeRWKV7FFN(nn.Module):
    """CMix module with attributes consumed by ``rwkv7_hf.native.ffn_step``."""

    def __init__(self, config: NativeRWKV7Config):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, x_prev: torch.Tensor):
        """Run one native FFN step through ``Module.__call__`` for ZeRO-3 hooks."""
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


class NativeRWKV7ForCausalLM(PreTrainedModel, GenerationMixin):
    """Experimental batched native PyTorch CausalLM for converted RWKV-7 weights."""

    config_class = NativeRWKV7Config
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["NativeRWKV7Layer"]
    supports_gradient_checkpointing = True
    # Transformers >=5 expects dict-like _tied_weights_keys; RWKV-7 ties nothing.
    _tied_weights_keys = {}

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

    @classmethod
    def from_pretrained(cls, *model_args, **kwargs):
        """Load dense weights, then apply optional native W8/W4 quantization.

        The native backend is the Apple/CPU/AMD fallback path, so its quantized
        route must not depend on bitsandbytes.  Persisted ``use_native_mm8`` or
        ``use_native_mm4`` config flags re-pack eligible ``nn.Linear`` modules
        after the fp weights are loaded.  The packed buffers are deterministic
        from the dense weights and therefore do not need to be stored in the
        checkpoint.
        """

        loaded = super().from_pretrained(*model_args, **kwargs)
        # Transformers returns ``(model, loading_info)`` when requested. Keep
        # that standard API shape while applying config-driven packing to the
        # actual model instance.
        model = loaded[0] if isinstance(loaded, tuple) else loaded
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

    def _native_model_quantized(self) -> bool:
        """True if layer projections were replaced by quantized modules.

        The JIT decode path extracts raw layer ``.weight`` tensors into packs,
        which cannot represent bnb or native MM8/MM4 layer replacements.  When
        layers are quantized, decode must use the eager per-token path whose
        module calls invoke the quantized linears.  ``lm_head``-only quantization
        is safe for JIT because ``native_jit._lm_head`` calls the module.
        Detected by class name to avoid importing optional quantization deps.
        """
        quantized_names = {"Linear4bit", "Linear8bit", "Linear8bitLt", "MM8Linear", "MM4Linear"}
        try:
            return any(type(module).__name__ in quantized_names for module in self.model.layers.modules())
        except Exception:
            return False

    def _native_model_has_adapter_layers(self) -> bool:
        """True when PEFT-style adapter wrappers sit inside native layers."""

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
                return True
            if hasattr(module, "base_layer") and (hasattr(module, "lora_A") or hasattr(module, "lora_B")):
                return True
        return False

    def _native_model_requires_eager_decode(self) -> bool:
        """Native JIT packs raw dense weights, so wrappers must use eager decode."""

        return self._native_model_quantized() or self._native_model_has_adapter_layers()

    def _native_jit_packs(self):
        if not _native_model_jit_enabled() or _native_jit_extract is None or _native_jit_step_batched is None:
            return None
        if self.config.attention_hidden_size != self.config.hidden_size:
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

        The experimental native model is still correctness-first and sequential
        over time, but the per-token math is vectorized over batch rows. This
        keeps the native/upstream fallback path aligned with wrapper bsz tests
        without claiming it replaces the optimized wrapper backend.

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
        packs = self._native_jit_packs() if use_jit and not output_hidden_states and attention_mask is None else None
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
            self._rwkv7_native_model_last_decode_backend = backend
        if all_logits is not None:
            logits = torch.stack(all_logits, dim=1)
        else:
            logits = self.lm_head(normed).view(batch_size, 1, -1)
        last_hidden_state = torch.stack(all_hidden, dim=1) if all_hidden is not None else normed.view(batch_size, 1, -1)
        hidden_states = None
        if hidden_buckets is not None:
            hidden_states = tuple(torch.stack(bucket, dim=1) for bucket in hidden_buckets)
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
        """Minimal beam/select helper for experimental batched native caches."""
        native_cache = _native_cache_tuple_or_none(past_key_values)
        if native_cache is None:
            return None
        if hasattr(past_key_values, "reorder_cache"):
            return past_key_values.reorder_cache(beam_idx)
        state, xpa, xpf, v_first = native_cache
        index = beam_idx.to(v_first.device)
        seen = _cache_seen(past_key_values)
        reordered = NativeRWKV7Cache(
            [s.index_select(0, index.to(s.device)) for s in state],
            [x.index_select(0, index.to(x.device)) for x in xpa],
            [x.index_select(0, index.to(x.device)) for x in xpf],
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
