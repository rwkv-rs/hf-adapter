# coding=utf-8
"""Fixed-shape CUDA graph runner for native RWKV-7 prefill."""
from __future__ import annotations

import weakref

import torch

from .model_cache import NativeRWKV7Cache
from .native_jit import (
    _native_prefill_fp16_recurrent_requested,
)


def _native_model_entrypoint():
    # Resolve lazily to avoid a module cycle while preserving the historical
    # native_model monkeypatch/debug surface used by graph-policy tests.
    from . import native_model

    return native_model


def _native_jit_prefill(*args, **kwargs):
    return _native_model_entrypoint()._native_jit_prefill(*args, **kwargs)


def _native_prefill_graph_enabled(*args, **kwargs):
    return _native_model_entrypoint()._native_prefill_graph_enabled(*args, **kwargs)


def _native_prefill_graph_signature():
    return _native_model_entrypoint()._native_prefill_graph_signature()


class _NativePrefillGraphRunner:
    """Fixed-shape CUDA graph for the canonical Native HF prefill path."""

    def __init__(
        self,
        owner: "NativeRWKV7ForCausalLM",
        packs,
        batch_size: int,
        prompt_tokens: int,
        logits_to_keep: int | None,
        carry_state: bool = False,
    ) -> None:
        if not _native_prefill_graph_enabled(
            batch_size,
            prompt_tokens,
            int(owner.config.hidden_size),
            int(owner.config.num_hidden_layers),
            owner.model.embeddings.weight.device,
        ):
            raise RuntimeError("native prefill graph is not enabled or available")
        self.owner = owner
        self.packs = packs
        self.batch_size = int(batch_size)
        self.prompt_tokens = int(prompt_tokens)
        self.logits_to_keep = None if logits_to_keep is None else int(logits_to_keep)
        self.carry_state = bool(carry_state)
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
        self.state_inputs: list[torch.Tensor] = []
        self.xpa_inputs: list[torch.Tensor] = []
        self.xpf_inputs: list[torch.Tensor] = []
        self.fp16_elapsed: torch.Tensor | None = None
        if self.carry_state:
            heads = int(packs[0][1])
            head_dim = int(packs[0][2])
            hidden = int(packs[0][7].numel())
            recurrent_dtype = (
                torch.float16
                if (
                    self.dtype == torch.float16
                    and head_dim == 64
                    and _native_prefill_fp16_recurrent_requested()
                )
                else torch.float32
            )
            self.state_inputs = [
                torch.zeros(
                    self.batch_size,
                    heads,
                    head_dim,
                    head_dim,
                    device=self.device,
                    dtype=recurrent_dtype,
                )
                for _ in packs
            ]
            self.xpa_inputs = [
                torch.zeros(
                    self.batch_size,
                    hidden,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in packs
            ]
            self.xpf_inputs = [
                torch.zeros(
                    self.batch_size,
                    hidden,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in packs
            ]
            self.fp16_elapsed = torch.zeros(
                self.batch_size,
                device=self.device,
                dtype=torch.int32,
            )
        self.inputs_are_zero = True
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
        if not self.carry_state:
            return _native_jit_prefill(
                self.owner,
                self.input_ids,
                self.packs,
                logits_to_keep=self.logits_to_keep,
            )
        return _native_jit_prefill(
            self.owner,
            self.input_ids,
            self.packs,
            state=list(self.state_inputs),
            xpa=list(self.xpa_inputs),
            xpf=list(self.xpf_inputs),
            logits_to_keep=self.logits_to_keep,
            fp16_elapsed=self.fp16_elapsed,
        )

    @staticmethod
    def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
        return bool(
            left.device == right.device
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and tuple(left.stride()) == tuple(right.stride())
            and left.storage_offset() == right.storage_offset()
            and left.untyped_storage().data_ptr()
            == right.untyped_storage().data_ptr()
        )

    @classmethod
    def _copy_if_different(cls, dst: torch.Tensor, src: torch.Tensor) -> None:
        if cls._same_tensor_view(dst, src):
            return
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    def _reset_inputs(self) -> None:
        for value in (*self.state_inputs, *self.xpa_inputs, *self.xpf_inputs):
            value.zero_()
        self.inputs_are_zero = True

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
            logits, state_outputs, xpa_outputs, xpf_outputs = outputs
            # Make graph outputs the stable inputs for the next replay. This
            # keeps recurrent-cache continuation inside one CUDA graph launch
            # instead of falling back to eager prefill for every later chunk.
            if self.carry_state:
                for dst, src in zip(self.state_inputs, state_outputs):
                    self._copy_if_different(dst, src)
                for dst, src in zip(self.xpa_inputs, xpa_outputs):
                    self._copy_if_different(dst, src)
                for dst, src in zip(self.xpf_inputs, xpf_outputs):
                    self._copy_if_different(dst, src)
        self.logits = logits
        self.state_outputs = (
            list(self.state_inputs) if self.carry_state else list(state_outputs)
        )
        self.xpa_outputs = (
            list(self.xpa_inputs) if self.carry_state else list(xpa_outputs)
        )
        self.xpf_outputs = (
            list(self.xpf_inputs) if self.carry_state else list(xpf_outputs)
        )
        # Warmup/capture executes the graph body and therefore advances the
        # stable state buffers. The first real replay must still start empty.
        if self.carry_state:
            self._reset_inputs()

    def matches(
        self,
        batch_size: int,
        prompt_tokens: int,
        logits_to_keep: int | None,
        carry_state: bool = False,
    ) -> bool:
        normalized_keep = None if logits_to_keep is None else int(logits_to_keep)
        return bool(
            self.batch_size == int(batch_size)
            and self.prompt_tokens == int(prompt_tokens)
            and self.logits_to_keep == normalized_keep
            and self.carry_state == bool(carry_state)
            and self.runtime_signature == _native_prefill_graph_signature()
        )

    def _cache_uses_inputs(self, initial_cache) -> bool:
        if initial_cache is None or not self.carry_state:
            return False
        try:
            state, xpa, xpf, _ = initial_cache
        except Exception:
            return False
        return bool(
            len(state) == len(self.state_inputs)
            and len(xpa) == len(self.xpa_inputs)
            and len(xpf) == len(self.xpf_inputs)
            and all(
                self._same_tensor_view(src, dst)
                for src, dst in zip(state, self.state_inputs)
            )
            and all(
                self._same_tensor_view(src, dst)
                for src, dst in zip(xpa, self.xpa_inputs)
            )
            and all(
                self._same_tensor_view(src, dst)
                for src, dst in zip(xpf, self.xpf_inputs)
            )
        )

    def _load_cache_inputs(self, initial_cache) -> None:
        if not self.carry_state:
            return
        if initial_cache is None:
            if not self.inputs_are_zero:
                self._reset_inputs()
            return
        if self._cache_uses_inputs(initial_cache):
            self.inputs_are_zero = False
            return
        state, xpa, xpf, _ = initial_cache
        if not (
            len(state) == len(self.state_inputs)
            and len(xpa) == len(self.xpa_inputs)
            and len(xpf) == len(self.xpf_inputs)
        ):
            raise ValueError("native prefill graph cache layer count mismatch")
        for dst, src in zip(self.state_inputs, state):
            self._copy_if_different(dst, src)
        for dst, src in zip(self.xpa_inputs, xpa):
            self._copy_if_different(dst, src)
        for dst, src in zip(self.xpf_inputs, xpf):
            self._copy_if_different(dst, src)
        self.inputs_are_zero = False

    def _detach_bound_cache(self, initial_cache=None) -> None:
        previous = self._bound_cache_ref() if self._bound_cache_ref is not None else None
        if previous is None:
            return
        if initial_cache is previous and self._cache_uses_inputs(initial_cache):
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
        initial_cache=None,
    ) -> tuple[torch.Tensor, NativeRWKV7Cache]:
        if tuple(input_ids.shape) != (self.batch_size, self.prompt_tokens):
            raise ValueError("native prefill graph input shape changed after capture")
        if input_ids.device != self.device or input_ids.dtype != torch.long:
            raise ValueError("native prefill graph input must be CUDA int64 on the model device")
        if self.graph is None or self.logits is None:
            raise RuntimeError("native prefill graph was not captured")
        if initial_cache is not None and not self.carry_state:
            raise ValueError("native prefill graph runner was not captured for cache continuation")
        self._detach_bound_cache(initial_cache)
        self._load_cache_inputs(initial_cache)
        if self.fp16_elapsed is not None:
            self.fp16_elapsed.fill_(int(seen_tokens) - self.prompt_tokens)
        self.input_ids.copy_(input_ids)
        self.graph.replay()
        previous = self._bound_cache_ref() if self._bound_cache_ref is not None else None
        if (
            previous is not None
            and initial_cache is previous
            and previous._native_graph_bound_to(self)
            and self._cache_uses_inputs(initial_cache)
        ):
            cache = previous
            cache.seen_tokens = int(seen_tokens)
        else:
            cache = NativeRWKV7Cache(
                self.state_inputs if self.carry_state else self.state_outputs,
                self.xpa_inputs if self.carry_state else self.xpa_outputs,
                self.xpf_inputs if self.carry_state else self.xpf_outputs,
                self.v_first,
                seen_tokens=int(seen_tokens),
            )
        self.inputs_are_zero = not self.carry_state
        cache._bind_native_graph_runner(self)
        self._bound_cache_ref = weakref.ref(cache)
        # Public HF forward owns its returned logits. A later replay may reuse
        # the graph buffers before the caller has finished consuming them.
        return self.logits.clone(), cache

    def detach_bound_cache(self) -> None:
        self._detach_bound_cache()
