# coding=utf-8
"""Fixed-shape CUDA graph runner for native RWKV-7 prefill."""
from __future__ import annotations

import weakref

import torch

from .model_cache import NativeRWKV7Cache


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
