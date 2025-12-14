# coding=utf-8
"""FLA-free CUDA graph runtime for native RWKV-7 token decode."""
from __future__ import annotations

import os
import weakref
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .kernel_policy import current_kernel_policy, env_flag

if TYPE_CHECKING:
    from .native_model import NativeRWKV7Cache, NativeRWKV7ForCausalLM

try:
    from .native_jit import (
        _block_ip,
        _block_ip_batched,
        _native_graph_linear_dispatch,
        prewarm_ada_sparse_ffn,
    )
except Exception:  # pragma: no cover - optional CUDA/Triton acceleration
    _block_ip = None
    _block_ip_batched = None
    _native_graph_linear_dispatch = None
    prewarm_ada_sparse_ffn = None

try:
    from .native_wkv_fp16 import (
        native_fp16_recurrent_available,
        native_fp16_recurrent_build_error,
    )
except Exception:  # pragma: no cover - optional CUDA extension
    native_fp16_recurrent_available = None
    native_fp16_recurrent_build_error = None

def native_graph_available() -> bool:
    """Return whether this process can capture the native decode graph."""

    return bool(torch.cuda.is_available() and _block_ip is not None and _block_ip_batched is not None)


def native_graph_cache_size() -> int:
    raw = os.environ.get("RWKV7_NATIVE_GRAPH_CACHE_SIZE", "8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def native_graph_stats_template() -> dict[str, int]:
    return {"requests": 0, "hits": 0, "misses": 0, "evictions": 0}


def native_graph_state_dtype(model_dtype: torch.dtype) -> torch.dtype:
    """Resolve graph-state dtype with explicit overrides taking precedence."""

    policy = current_kernel_policy(torch_module=torch)
    raw = os.environ.get(
        "RWKV7_NATIVE_GRAPH_STATE_DTYPE",
        str(getattr(policy, "native_graph_state_dtype", "fp32")),
    ).strip().lower()
    if raw in {"fp32", "float32"}:
        return torch.float32
    if raw in {"fp16", "float16", "half"}:
        return torch.float16 if model_dtype == torch.float16 else torch.float32
    raise ValueError(
        "RWKV7_NATIVE_GRAPH_STATE_DTYPE must be fp32 or fp16; "
        f"got {raw!r}"
    )


def native_graph_precompute_embedding_enabled() -> bool:
    policy = current_kernel_policy(torch_module=torch)
    return env_flag(
        "RWKV7_NATIVE_GRAPH_PRECOMPUTE_EMB_LN0",
        bool(getattr(policy, "native_graph_precompute_embedding", False)),
    )


def native_graph_fp16_recurrent_enabled() -> bool:
    policy = current_kernel_policy(torch_module=torch)
    return env_flag(
        "RWKV7_NATIVE_GRAPH_FP16_RECURRENT",
        bool(getattr(policy, "native_graph_fp16_recurrent", False)),
    )


def native_graph_runtime_signature() -> tuple[tuple[str, str], ...]:
    """Capture all graph-affecting overrides in the runner-cache key."""

    prefixes = (
        "RWKV7_NATIVE_GRAPH_",
        "RWKV7_FUSED_",
        "RWKV7_NATIVE_MM",
    )
    return tuple(sorted((key, value) for key, value in os.environ.items() if key.startswith(prefixes)))


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    try:
        return (
            left.data_ptr() == right.data_ptr()
            and left.storage_offset() == right.storage_offset()
            and tuple(left.shape) == tuple(right.shape)
            and tuple(left.stride()) == tuple(right.stride())
            and left.dtype == right.dtype
            and left.device == right.device
        )
    except Exception:
        return False


def _head_linear_into(module, value: torch.Tensor, output: torch.Tensor) -> None:
    forward_into = getattr(module, "rwkv7_forward_into", None)
    if callable(forward_into):
        forward_into(value, output)
        return
    if type(module) is torch.nn.Linear and module.bias is None:
        if _native_graph_linear_dispatch is None:
            result = F.linear(value, module.weight)
        else:
            result = _native_graph_linear_dispatch(value, module.weight, role="head")
    else:
        result = module(value)
    output.copy_(result.reshape_as(output))


class NativeGraphRunner:
    """One fixed-batch CUDA graph whose buffers use native cache layout."""

    def __init__(self, owner: "NativeRWKV7ForCausalLM", packs, batch_size: int) -> None:
        if not native_graph_available():
            raise RuntimeError("native_graph requires CUDA and rwkv7_hf.native_jit graph blocks")
        self.owner_ref = weakref.ref(owner)
        self.packs = list(packs)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("native_graph batch size must be positive")
        base = owner.model
        self.device = base.embeddings.weight.device
        if self.device.type != "cuda":
            raise RuntimeError("native_graph requires model weights on CUDA")
        self.dtype = base.embeddings.weight.dtype
        self.state_dtype = native_graph_state_dtype(self.dtype)
        self.fp16_recurrent = self.state_dtype == torch.float16
        if self.fp16_recurrent:
            if not native_graph_fp16_recurrent_enabled():
                raise RuntimeError(
                    "FP16 native graph state requires "
                    "RWKV7_NATIVE_GRAPH_FP16_RECURRENT=1"
                )
            if (
                native_fp16_recurrent_available is None
                or not native_fp16_recurrent_available(build=True)
            ):
                error = (
                    native_fp16_recurrent_build_error()
                    if native_fp16_recurrent_build_error is not None
                    else "extension import failed"
                )
                raise RuntimeError(
                    f"FP16 native recurrent extension is unavailable: {error}"
                )
        self.hidden = int(owner.config.hidden_size)
        self.attention_hidden = int(
            getattr(owner.config, "attention_hidden_size", packs[0][1] * packs[0][2])
        )
        self.num_layers = len(packs)
        self.embeddings = base.embeddings.weight
        self.precomputed_embedding_ln0 = False
        if native_graph_precompute_embedding_enabled():
            first = list(self.packs[0])
            if not bool(first[4]):
                raise RuntimeError(
                    "RWKV7_NATIVE_GRAPH_PRECOMPUTE_EMB_LN0 requires a first-layer pre-norm"
                )
            pre_weight = first[5]
            pre_bias = first[6]
            key = (
                int(self.embeddings.data_ptr()),
                int(pre_weight.data_ptr()),
                int(pre_bias.data_ptr()),
                self.embeddings.device,
                self.embeddings.dtype,
                tuple(self.embeddings.shape),
            )
            cached = getattr(owner, "_rwkv7_native_graph_precomputed_embedding_ln0", None)
            if not isinstance(cached, tuple) or len(cached) != 2 or cached[0] != key:
                normalized = F.layer_norm(
                    self.embeddings,
                    [self.hidden],
                    pre_weight,
                    pre_bias,
                    1e-5,
                ).contiguous()
                cached = (key, normalized)
                owner._rwkv7_native_graph_precomputed_embedding_ln0 = cached
            self.embeddings = cached[1]
            first[4] = False
            self.packs[0] = tuple(first)
            self.precomputed_embedding_ln0 = True
        self.single = self.batch_size == 1
        if self.single:
            self.state = [
                torch.zeros(
                    int(pack[1]),
                    int(pack[2]),
                    int(pack[2]),
                    device=self.device,
                    dtype=self.state_dtype,
                )
                for pack in packs
            ]
            self.elapsed = (
                torch.zeros(1, device=self.device, dtype=torch.int32)
                if self.fp16_recurrent
                else None
            )
            self.xpa = [torch.zeros(self.hidden, device=self.device, dtype=self.dtype) for _ in packs]
            self.xpf = [torch.zeros(self.hidden, device=self.device, dtype=self.dtype) for _ in packs]
            self.sparse_ffn_out = [torch.empty(self.hidden, device=self.device, dtype=self.dtype) for _ in packs]
            self.v_first = torch.zeros(self.attention_hidden, device=self.device, dtype=self.dtype)
        else:
            self.state = [
                torch.zeros(
                    self.batch_size,
                    int(pack[1]),
                    int(pack[2]),
                    int(pack[2]),
                    device=self.device,
                    dtype=self.state_dtype,
                )
                for pack in packs
            ]
            self.elapsed = (
                torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
                if self.fp16_recurrent
                else None
            )
            self.xpa = [
                torch.zeros(self.batch_size, self.hidden, device=self.device, dtype=self.dtype) for _ in packs
            ]
            self.xpf = [
                torch.zeros(self.batch_size, self.hidden, device=self.device, dtype=self.dtype) for _ in packs
            ]
            self.sparse_ffn_out = [
                torch.empty(self.batch_size, self.hidden, device=self.device, dtype=self.dtype) for _ in packs
            ]
            self.v_first = torch.zeros(
                self.batch_size,
                self.attention_hidden,
                device=self.device,
                dtype=self.dtype,
            )
        self.token_ids = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.head = owner.lm_head
        self.norm_weight = base.norm.weight
        self.norm_bias = base.norm.bias
        vocab_size = int(getattr(self.head, "out_features", self.embeddings.shape[0]))
        self.logits = torch.empty(self.batch_size, vocab_size, device=self.device, dtype=self.dtype)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._bound_cache_ref: weakref.ReferenceType | None = None
        self.copy_from_cache_calls = 0
        self.copy_from_cache_fast_skips = 0
        self.bind_cache_calls = 0
        self.bind_cache_fast_skips = 0
        self._capture()

    def _one_step(self) -> None:
        if self.single:
            hidden = F.embedding(self.token_ids, self.embeddings).reshape(self.hidden)
            for layer_index, pack in enumerate(self.packs):
                hidden = _block_ip(
                    hidden,
                    self.state[layer_index],
                    self.xpa[layer_index],
                    self.xpf[layer_index],
                    self.v_first,
                    pack,
                    self.sparse_ffn_out[layer_index],
                    self.elapsed,
                    layer_index + 1 == self.num_layers,
                )
            hidden = F.layer_norm(hidden, [self.hidden], self.norm_weight, self.norm_bias, 1e-5)
            _head_linear_into(self.head, hidden, self.logits.reshape(-1))
            return

        hidden = F.embedding(self.token_ids, self.embeddings).reshape(self.batch_size, self.hidden)
        for layer_index, pack in enumerate(self.packs):
            hidden = _block_ip_batched(
                hidden,
                self.state[layer_index],
                self.xpa[layer_index],
                self.xpf[layer_index],
                self.v_first,
                pack,
                self.sparse_ffn_out[layer_index],
                self.elapsed,
                layer_index + 1 == self.num_layers,
            )
        hidden = F.layer_norm(hidden, [self.hidden], self.norm_weight, self.norm_bias, 1e-5)
        _head_linear_into(self.head, hidden, self.logits)

    def _capture(self) -> None:
        if prewarm_ada_sparse_ffn is not None:
            prewarm_ada_sparse_ffn(self.packs, self.batch_size)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            for _ in range(3):
                self._one_step()
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._one_step()

    def _native_view(self, value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0) if self.single else value

    def _copy_cache_tensor(self, target: torch.Tensor, source: torch.Tensor | None) -> None:
        if source is None:
            target.zero_()
            return
        if self.single and source.dim() == target.dim() + 1 and int(source.shape[0]) == 1:
            source = source.squeeze(0)
        if _same_tensor_view(target, source):
            return
        converted = source.to(device=target.device, dtype=target.dtype)
        if not _same_tensor_view(target, converted):
            target.copy_(converted.contiguous())

    def _detach_bound_cache_if_different(self, cache: "NativeRWKV7Cache") -> None:
        previous = self._bound_cache_ref() if self._bound_cache_ref is not None else None
        if previous is None or previous is cache:
            return
        if previous._native_graph_bound_to(self):
            previous._state = [self._native_view(value).clone() for value in self.state]
            previous._xpa = [self._native_view(value).clone() for value in self.xpa]
            previous._xpf = [self._native_view(value).clone() for value in self.xpf]
            previous._v_first = self._native_view(self.v_first).clone()
            previous._invalidate_native_graph_binding()
        self._bound_cache_ref = None

    def copy_from_cache(self, cache: "NativeRWKV7Cache") -> None:
        self.copy_from_cache_calls += 1
        if cache._native_graph_bound_to(self):
            self.copy_from_cache_fast_skips += 1
            return
        self._detach_bound_cache_if_different(cache)
        if cache._state is None or cache._xpa is None or cache._xpf is None:
            raise ValueError("native_graph requires an initialized NativeRWKV7Cache")
        if len(cache._state) != self.num_layers:
            raise ValueError("native_graph cache layer count does not match the model")
        for layer_index in range(self.num_layers):
            self._copy_cache_tensor(self.state[layer_index], cache._state[layer_index])
            self._copy_cache_tensor(self.xpa[layer_index], cache._xpa[layer_index])
            self._copy_cache_tensor(self.xpf[layer_index], cache._xpf[layer_index])
        if self.elapsed is not None:
            self.elapsed.fill_(int(cache.get_seq_length()))

    def bind_cache(self, cache: "NativeRWKV7Cache") -> None:
        self.bind_cache_calls += 1
        if cache._native_graph_bound_to(self):
            self.bind_cache_fast_skips += 1
            return
        cache._state = [self._native_view(value) for value in self.state]
        cache._xpa = [self._native_view(value) for value in self.xpa]
        cache._xpf = [self._native_view(value) for value in self.xpf]
        cache._v_first = self._native_view(self.v_first)
        self._bound_cache_ref = weakref.ref(cache)
        cache._bind_native_graph_runner(self)

    def detach_bound_cache(self) -> None:
        cache = self._bound_cache_ref() if self._bound_cache_ref is not None else None
        if cache is not None and cache._native_graph_bound_to(self):
            cache._state = [self._native_view(value).clone() for value in self.state]
            cache._xpa = [self._native_view(value).clone() for value in self.xpa]
            cache._xpf = [self._native_view(value).clone() for value in self.xpf]
            cache._v_first = self._native_view(self.v_first).clone()
            cache._invalidate_native_graph_binding()
        self._bound_cache_ref = None

    def reorder_batch_inplace(self, indices: torch.LongTensor) -> bool:
        if int(indices.numel()) != self.batch_size:
            return False
        if self.single:
            return True
        index = indices.to(device=self.device, dtype=torch.long)
        for layer_index in range(self.num_layers):
            self.state[layer_index].copy_(self.state[layer_index].index_select(0, index).contiguous())
            self.xpa[layer_index].copy_(self.xpa[layer_index].index_select(0, index).contiguous())
            self.xpf[layer_index].copy_(self.xpf[layer_index].index_select(0, index).contiguous())
        if self.elapsed is not None:
            self.elapsed.copy_(self.elapsed.index_select(0, index).contiguous())
        self.v_first.copy_(self.v_first.index_select(0, index).contiguous())
        return True

    def replay(
        self,
        token_ids: torch.LongTensor,
        cache: "NativeRWKV7Cache",
        *,
        copy_logits: bool = True,
    ) -> torch.Tensor:
        if int(token_ids.numel()) != self.batch_size:
            raise ValueError(
                f"native_graph runner batch mismatch: got {int(token_ids.numel())}, expected {self.batch_size}"
            )
        self.copy_from_cache(cache)
        self.token_ids.copy_(token_ids.reshape(self.batch_size))
        if self.graph is None:
            raise RuntimeError("native_graph runner was not captured")
        self.graph.replay()
        self.bind_cache(cache)
        logits = self.logits.view(self.batch_size, 1, -1)
        return logits.clone() if copy_logits else logits

    def copy_stats(self) -> dict[str, int]:
        return {
            "copy_from_cache_calls": int(self.copy_from_cache_calls),
            "copy_from_cache_fast_skips": int(self.copy_from_cache_fast_skips),
            "bind_cache_calls": int(self.bind_cache_calls),
            "bind_cache_fast_skips": int(self.bind_cache_fast_skips),
        }


__all__ = [
    "NativeGraphRunner",
    "native_graph_available",
    "native_graph_cache_size",
    "native_graph_runtime_signature",
    "native_graph_precompute_embedding_enabled",
    "native_graph_fp16_recurrent_enabled",
    "native_graph_state_dtype",
    "native_graph_stats_template",
]
