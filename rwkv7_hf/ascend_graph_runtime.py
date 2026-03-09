# coding=utf-8
# Ported from rwkv-rs/rwkv7-ascend-npu@b6391271f into the canonical HF adapter.
"""Fixed-batch NPUGraph decode for the native RWKV-7 Hugging Face model.

The CUDA native graph runner is built around CUDA-only packed JIT blocks.  On
Ascend the ordinary native PyTorch modules are already the correctness path,
including weight-only FFN modules.  Capturing that whole token step removes the
per-layer Python/CANN dispatch cost while keeping the recurrent cache at fixed
addresses owned by the graph runner.

This module deliberately imports no ``torch_npu`` at package import time.
``enable_ascend`` registers ``torch.npu`` before a runner is constructed.
"""

from __future__ import annotations

import os
import weakref
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .native import _init_state_batched, _step_token_batched

if TYPE_CHECKING:
    from .native_model import NativeRWKV7Cache, NativeRWKV7ForCausalLM


def ascend_graph_available() -> bool:
    """Return whether the current process exposes the torch-npu graph API."""

    npu = getattr(torch, "npu", None)
    return bool(
        npu is not None
        and callable(getattr(npu, "is_available", None))
        and npu.is_available()
        and hasattr(npu, "NPUGraph")
        and callable(getattr(npu, "graph", None))
    )


def ascend_graph_cache_size() -> int:
    raw = os.environ.get("RWKV7_ASCEND_GRAPH_CACHE_SIZE", "3").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def ascend_graph_runtime_signature() -> tuple[tuple[str, str], ...]:
    prefixes = ("RWKV7_ASCEND_GRAPH_", "RWKV7_ASCEND_QUANT_")
    return tuple(
        sorted(
            (key, value)
            for key, value in os.environ.items()
            if key.startswith(prefixes)
        )
    )


def ascend_graph_module_signature(
    owner: "NativeRWKV7ForCausalLM",
) -> tuple[tuple[object, ...], ...]:
    """Identify packed modules whose buffers are baked into a captured graph.

    Quantization can replace dense projections after an FP16 graph has already
    been cached.  Buffer pointers make that mutation part of the cache key, so
    the old graph can never be replayed against newly installed W8/W4 modules.
    Dense parameter updates still require the public explicit cache-clear API,
    as they do for the CUDA graph backend.
    """

    signature: list[tuple[object, ...]] = []
    for name, module in owner.model.layers.named_modules():
        qweight = getattr(module, "qweight", None)
        if not isinstance(qweight, torch.Tensor):
            qweight = getattr(module, "q_weight", None)
        scales = getattr(module, "scales", None)
        if not isinstance(scales, torch.Tensor):
            scales = getattr(module, "scale", None)
        if not isinstance(qweight, torch.Tensor) or not qweight.numel():
            continue
        if not isinstance(scales, torch.Tensor) or not scales.numel():
            continue
        signature.append(
            (
                name,
                type(module).__module__,
                type(module).__name__,
                int(getattr(module, "bit", 0)),
                int(getattr(module, "group_size", 0)),
                tuple(qweight.shape),
                qweight.dtype,
                int(qweight.data_ptr()),
                tuple(scales.shape),
                scales.dtype,
                int(scales.data_ptr()),
            )
        )
    return tuple(signature)


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        left.data_ptr() == right.data_ptr()
        and left.storage_offset() == right.storage_offset()
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
        and left.dtype == right.dtype
        and left.device == right.device
    )


class AscendGraphRunner:
    """One fixed-batch NPUGraph with graph-resident recurrent state."""

    def __init__(self, owner: "NativeRWKV7ForCausalLM", batch_size: int) -> None:
        if not ascend_graph_available():
            raise RuntimeError("Ascend NPUGraph is unavailable")
        if owner.model.embeddings.weight.device.type != "npu":
            raise RuntimeError("AscendGraphRunner requires model weights on NPU")
        self.owner_ref = weakref.ref(owner)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("Ascend graph batch size must be positive")
        self.device = owner.model.embeddings.weight.device
        self.dtype = owner.model.embeddings.weight.dtype
        torch.npu.set_device(self.device)
        self.state, self.xpa, self.xpf, self.v_first = _init_state_batched(
            owner,
            self.batch_size,
            self.device,
            self.dtype,
        )
        self.token_ids = torch.zeros(
            self.batch_size,
            dtype=torch.long,
            device=self.device,
        )
        self.logits = torch.empty(
            self.batch_size,
            int(owner.config.vocab_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.graph = None
        self._bound_cache_ref: weakref.ReferenceType | None = None
        self.copy_from_cache_calls = 0
        self.copy_from_cache_fast_skips = 0
        self.bind_cache_calls = 0
        self.bind_cache_fast_skips = 0
        self._capture()

    def _one_step(self) -> None:
        owner = self.owner_ref()
        if owner is None:
            raise RuntimeError("Ascend graph owner was released")
        hidden = F.embedding(self.token_ids, owner.model.embeddings.weight)
        # _step_token_batched replaces list entries. Pass shallow copies so the
        # graph's fixed input buffers remain stable and copy results back below.
        hidden, next_state, next_xpa, next_xpf, next_v_first = _step_token_batched(
            owner,
            hidden,
            list(self.state),
            list(self.xpa),
            list(self.xpf),
            self.v_first,
        )
        hidden = owner.model.norm(hidden)
        next_logits = owner.lm_head(hidden)
        for target, source in zip(self.state, next_state):
            target.copy_(source)
        for target, source in zip(self.xpa, next_xpa):
            target.copy_(source)
        for target, source in zip(self.xpf, next_xpf):
            target.copy_(source)
        self.v_first.copy_(next_v_first)
        self.logits.copy_(next_logits.reshape_as(self.logits))

    def _capture(self) -> None:
        with torch.inference_mode():
            for _ in range(2):
                self._one_step()
        torch.npu.synchronize()
        warmup_stream = torch.npu.Stream()
        warmup_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(warmup_stream), torch.inference_mode():
            for _ in range(2):
                self._one_step()
        torch.npu.current_stream().wait_stream(warmup_stream)
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph), torch.inference_mode():
            self._one_step()
        torch.npu.synchronize()

    def _copy_tensor(self, target: torch.Tensor, source: torch.Tensor | None) -> None:
        if source is None:
            target.zero_()
            return
        if _same_tensor_view(target, source):
            return
        converted = source.to(device=target.device, dtype=target.dtype)
        if not _same_tensor_view(target, converted):
            target.copy_(converted.contiguous())

    def _detach_bound_cache_if_different(self, cache: "NativeRWKV7Cache") -> None:
        previous = (
            self._bound_cache_ref() if self._bound_cache_ref is not None else None
        )
        if previous is None or previous is cache:
            return
        if previous._native_graph_bound_to(self):
            previous._state = [value.clone() for value in self.state]
            previous._xpa = [value.clone() for value in self.xpa]
            previous._xpf = [value.clone() for value in self.xpf]
            previous._v_first = self.v_first.clone()
            previous._invalidate_native_graph_binding()
        self._bound_cache_ref = None

    def copy_from_cache(self, cache: "NativeRWKV7Cache") -> None:
        self.copy_from_cache_calls += 1
        if cache._native_graph_bound_to(self):
            self.copy_from_cache_fast_skips += 1
            return
        self._detach_bound_cache_if_different(cache)
        if cache._state is None or cache._xpa is None or cache._xpf is None:
            raise ValueError("Ascend graph requires an initialized cache")
        if len(cache._state) != len(self.state):
            raise ValueError("Ascend graph cache layer count mismatch")
        for target, source in zip(self.state, cache._state):
            self._copy_tensor(target, source)
        for target, source in zip(self.xpa, cache._xpa):
            self._copy_tensor(target, source)
        for target, source in zip(self.xpf, cache._xpf):
            self._copy_tensor(target, source)
        self._copy_tensor(self.v_first, cache._v_first)

    def bind_cache(self, cache: "NativeRWKV7Cache") -> None:
        self.bind_cache_calls += 1
        if cache._native_graph_bound_to(self):
            self.bind_cache_fast_skips += 1
            return
        cache._state = list(self.state)
        cache._xpa = list(self.xpa)
        cache._xpf = list(self.xpf)
        cache._v_first = self.v_first
        self._bound_cache_ref = weakref.ref(cache)
        cache._bind_native_graph_runner(self)

    def detach_bound_cache(self) -> None:
        cache = self._bound_cache_ref() if self._bound_cache_ref is not None else None
        if cache is not None and cache._native_graph_bound_to(self):
            cache._state = [value.clone() for value in self.state]
            cache._xpa = [value.clone() for value in self.xpa]
            cache._xpf = [value.clone() for value in self.xpf]
            cache._v_first = self.v_first.clone()
            cache._invalidate_native_graph_binding()
        self._bound_cache_ref = None

    def reorder_batch_inplace(self, indices: torch.LongTensor) -> bool:
        if int(indices.numel()) != self.batch_size:
            return False
        index = indices.to(device=self.device, dtype=torch.long)
        for values in (self.state, self.xpa, self.xpf):
            for value in values:
                value.copy_(value.index_select(0, index).contiguous())
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
                "Ascend graph runner batch mismatch: "
                f"got {int(token_ids.numel())}, expected {self.batch_size}"
            )
        self.copy_from_cache(cache)
        self.token_ids.copy_(token_ids.reshape(self.batch_size))
        if self.graph is None:
            raise RuntimeError("Ascend graph runner was not captured")
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
