# coding=utf-8
"""Remote-code wrapper around FLA RWKV7 HF modules.

Requires flash-linear-attention (`fla`) on PYTHONPATH / installed in the env.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F
from fla.models.rwkv7.modeling_rwkv7 import RWKV7Model as _RWKV7Model
from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM as _RWKV7ForCausalLM
from fla.models.utils import Cache as _FLACache
from fla.ops.rwkv7.fused_recurrent import fused_mul_recurrent_rwkv7
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_rwkv7 import RWKV7Config
except ImportError:  # pragma: no cover - direct remote-file execution fallback
    from configuration_rwkv7 import RWKV7Config

try:
    from .native_jit import block_step as _native_jit_block_step
    from .native_jit import block_step_batched as _native_jit_block_step_batched
    from .native_jit import extract as _native_jit_extract
    from .native_jit import _block_ip as _native_graph_block_ip
    from .native_jit import _block_ip_batched as _native_graph_block_ip_batched
except Exception:  # pragma: no cover - optional remote-code fast path
    try:
        from native_jit import block_step as _native_jit_block_step
        from native_jit import block_step_batched as _native_jit_block_step_batched
        from native_jit import extract as _native_jit_extract
        from native_jit import _block_ip as _native_graph_block_ip
        from native_jit import _block_ip_batched as _native_graph_block_ip_batched
    except Exception:
        _native_jit_block_step = None
        _native_jit_block_step_batched = None
        _native_jit_extract = None
        _native_graph_block_ip = None
        _native_graph_block_ip_batched = None


_FALSE_VALUES = {"0", "false", "False", "no", "off"}


def _fast_cache_enabled() -> bool:
    """Runtime switch used by benchmarks to compare cache implementations."""
    return os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES


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


def _linear_direct(module, x: torch.Tensor) -> torch.Tensor:
    """Call a Linear module through F.linear to skip small-module dispatch."""
    return F.linear(x, module.weight, module.bias)


def _lora_direct(module, x: torch.Tensor) -> torch.Tensor:
    """Fast-path FLA LoRA forward used only by inference decode helpers."""
    h = F.linear(x, module.lora[0].weight, module.lora[0].bias)
    h = module.lora[1](h)
    return F.linear(h, module.lora[2].weight, module.lora[2].bias)


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
        self.v_first = torch.zeros(self.hidden, device=self.device, dtype=self.dtype)
        self.tok_id = torch.zeros(1, dtype=torch.long, device=self.device)
        self.logits = torch.zeros(base.embeddings.weight.shape[0], device=self.device, dtype=self.dtype)
        self.emb = base.embeddings.weight
        self.head = owner.lm_head.weight
        self.head_bias = owner.lm_head.bias
        self.norm_w = base.norm.weight
        self.norm_b = base.norm.bias
        self.graph = None
        self._capture()

    def _one_step(self) -> None:
        x = F.embedding(self.tok_id, self.emb).reshape(self.hidden)
        for li, p in enumerate(self.packs):
            x = _native_graph_block_ip(x, self.state[li], self.xpa[li], self.xpf[li], self.v_first, p)
        out = F.layer_norm(x, [self.hidden], self.norm_w, self.norm_b, 1e-5)
        self.logits.copy_(F.linear(out, self.head, self.head_bias).reshape(-1))

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
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype).contiguous())

    def copy_from_cache(self, past_key_values: "RWKV7StateCache") -> None:
        for li, p in enumerate(self.packs):
            layer_idx = int(p[0])
            state = past_key_values._ensure_layer(layer_idx)
            self._copy_cache_tensor(self.state[li], state.get("recurrent_state"), transpose_last=True)
            self._copy_cache_tensor(self.xpa[li], state.get("conv_state"))
            self._copy_cache_tensor(self.xpf[li], state.get("ffn_state"))

    def bind_cache(self, past_key_values: "RWKV7StateCache") -> None:
        for li, p in enumerate(self.packs):
            layer_idx = int(p[0])
            state = past_key_values._ensure_layer(layer_idx)
            # FLA cache layout is transposed relative to the native matmul layout.
            state["recurrent_state"] = self.state[li].transpose(-1, -2).unsqueeze(0)
            state["conv_state"] = self.xpa[li].unsqueeze(0)
            state["ffn_state"] = self.xpf[li].unsqueeze(0)
            state["attn_state"] = None

    def replay(self, token: torch.LongTensor, past_key_values: "RWKV7StateCache") -> torch.Tensor:
        self.copy_from_cache(past_key_values)
        self.tok_id.copy_(token.reshape(1))
        self.graph.replay()
        self.bind_cache(past_key_values)
        return self.logits.view(1, 1, -1)


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
        self.v_first = torch.zeros(self.batch_size, self.hidden, device=self.device, dtype=self.dtype)
        self.tok_id = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.logits = torch.zeros(self.batch_size, base.embeddings.weight.shape[0], device=self.device, dtype=self.dtype)
        self.emb = base.embeddings.weight
        self.head = owner.lm_head.weight
        self.head_bias = owner.lm_head.bias
        self.norm_w = base.norm.weight
        self.norm_b = base.norm.bias
        self.graph = None
        self._capture()

    def _one_step(self) -> None:
        x = F.embedding(self.tok_id, self.emb).reshape(self.batch_size, self.hidden)
        for li, p in enumerate(self.packs):
            x = _native_graph_block_ip_batched(x, self.state[li], self.xpa[li], self.xpf[li], self.v_first, p)
        out = F.layer_norm(x, [self.hidden], self.norm_w, self.norm_b, 1e-5)
        self.logits.copy_(F.linear(out, self.head, self.head_bias))

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
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype).contiguous())

    def copy_from_cache(self, past_key_values: "RWKV7StateCache") -> None:
        for li, p in enumerate(self.packs):
            layer_idx = int(p[0])
            state = past_key_values._ensure_layer(layer_idx)
            self._copy_cache_tensor(self.state[li], state.get("recurrent_state"), transpose_last=True)
            self._copy_cache_tensor(self.xpa[li], state.get("conv_state"))
            self._copy_cache_tensor(self.xpf[li], state.get("ffn_state"))

    def bind_cache(self, past_key_values: "RWKV7StateCache") -> None:
        for li, p in enumerate(self.packs):
            layer_idx = int(p[0])
            state = past_key_values._ensure_layer(layer_idx)
            # FLA cache layout is transposed relative to the native matmul layout.
            state["recurrent_state"] = self.state[li].transpose(-1, -2)
            state["conv_state"] = self.xpa[li]
            state["ffn_state"] = self.xpf[li]
            state["attn_state"] = None

    def replay(self, token: torch.LongTensor, past_key_values: "RWKV7StateCache") -> torch.Tensor:
        if int(token.numel()) != self.batch_size:
            raise ValueError(f"native_graph runner batch mismatch: got {int(token.numel())}, expected {self.batch_size}")
        self.copy_from_cache(past_key_values)
        self.tok_id.copy_(token.reshape(self.batch_size))
        self.graph.replay()
        self.bind_cache(past_key_values)
        return self.logits.view(self.batch_size, 1, -1)


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
        self.states.clear()
        self._seen_tokens = 0

    def to_legacy_cache(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.states)

    def clone(self) -> "RWKV7StateCache":
        out = type(self)(seen_tokens=self._seen_tokens)
        out.states = [_clone_cache_value(state) for state in self.states]
        return out

    def detach(self, *, inplace: bool = True) -> "RWKV7StateCache":
        """Detach cache tensors from autograd graphs for inference serving."""
        target = self if inplace else self.clone()
        target.states = [_detach_cache_value(state) for state in target.states]
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
        target.states = [
            _to_cache_value(state, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
            for state in target.states
        ]
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
        target.states = [_move_first_dim(state, indices) for state in target.states]
        return target

    def batch_select(self, indices: torch.LongTensor, *, inplace: bool = True) -> "RWKV7StateCache":
        return self.select_batch(indices, inplace=inplace)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        return self.select_batch(beam_idx, inplace=True)

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


class RWKV7Model(_RWKV7Model):
    config_class = RWKV7Config


class RWKV7ForCausalLM(_RWKV7ForCausalLM):
    config_class = RWKV7Config
    # Transformers >=5 expects dict-like _tied_weights_keys in save_pretrained.
    _tied_weights_keys = {}

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

    def rwkv7_native_graph_cache_batch_sizes(self) -> list[int]:
        """Return active batch sizes currently retained in the graph-runner LRU."""
        cache = getattr(self, "_rwkv7_native_graph_runner_cache", None)
        if isinstance(cache, tuple) and len(cache) == 2:
            key = cache[0]
            return [int(key[-1])] if isinstance(key, tuple) and key else []
        if isinstance(cache, OrderedDict):
            return sorted({int(key[-1]) for key in cache.keys() if isinstance(key, tuple) and key})
        return []

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

    def _rwkv7_native_jit_packs(self):
        if _native_jit_block_step is None or _native_jit_block_step_batched is None or _native_jit_extract is None:
            raise RuntimeError("native_jit fast-token backend is unavailable; copy native_jit.py into the model repo")
        cache = getattr(self, "_rwkv7_native_jit_pack_cache", None)
        weight = self.model.embeddings.weight
        key = (weight.device.type, weight.device.index, weight.dtype)
        if cache is None or cache[0] != key:
            packs, _, _, _ = _native_jit_extract(self)
            self._rwkv7_native_jit_pack_cache = (key, packs)
            return packs
        return cache[1]

    def _rwkv7_native_graph_runner(self, packs, batch_size: int):
        weight = self.model.embeddings.weight
        key = (weight.device.type, weight.device.index, weight.dtype, len(packs), int(packs[0][1]), int(packs[0][2]), int(batch_size))
        cache = getattr(self, "_rwkv7_native_graph_runner_cache", None)
        if isinstance(cache, tuple) and len(cache) == 2:
            cache = OrderedDict([cache])
        elif not isinstance(cache, OrderedDict):
            cache = OrderedDict()
        self._rwkv7_native_graph_runner_cache = cache

        runner = cache.get(key)
        if runner is not None:
            cache.move_to_end(key)
            return runner

        if int(batch_size) == 1:
            runner = _RWKV7NativeGraphTokenRunner(self, packs)
        else:
            runner = _RWKV7NativeGraphBatchedTokenRunner(self, packs, int(batch_size))
        cache[key] = runner
        cache.move_to_end(key)
        while len(cache) > _native_graph_cache_size():
            cache.popitem(last=False)
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
            out = self(
                input_ids[:, start:end],
                attention_mask=chunk_kwargs.pop("attention_mask", None),
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
        logits = F.linear(hidden_states, self.lm_head.weight, self.lm_head.bias).view(batch_size, 1, -1)
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
        packs = self._rwkv7_native_jit_packs()
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
        if self._rwkv7_uses_external_quantization():
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

    def forward(self, *args, **kwargs):
        use_cache = kwargs.get("use_cache")
        effective_use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        if effective_use_cache and _fast_cache_enabled():
            past_key_values = kwargs.get("past_key_values")
            if not isinstance(past_key_values, RWKV7StateCache):
                kwargs["past_key_values"] = RWKV7StateCache.from_legacy_cache(past_key_values)
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
