# coding=utf-8
"""Public native prefill and fast-token serving APIs for RWKV-7."""
from __future__ import annotations

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from .kernel_package import kernel_runtime_report
from .model_cache import NativeRWKV7Cache, _cache_seen


def _native_model_entrypoint():
    # Preserve the historical native_model monkeypatch/debug surface lazily.
    from . import native_model

    return native_model


def _native_graph_available() -> bool:
    return bool(_native_model_entrypoint()._native_graph_available())


def _ascend_graph_available() -> bool:
    return bool(_native_model_entrypoint()._ascend_graph_available())


def _native_model_backend_requested() -> str:
    return _native_model_entrypoint()._native_model_backend_requested()


class _NativeFastAPIMixin:
    def rwkv7_native_model_last_decode_backend(self) -> str | None:
        """Return the backend used by the previous native-model decode call."""
        return getattr(self, "_rwkv7_native_model_last_decode_backend", None)

    def rwkv7_native_model_last_prefill_backend(self) -> str | None:
        """Return the backend used by the previous native-model prefill call."""
        return getattr(self, "_rwkv7_native_model_last_prefill_backend", None)

    def rwkv7_runtime_report(self) -> dict:
        """Report the effective model routes and native-kernel package state."""

        try:
            device = next(self.parameters()).device
            rendered_device = str(device)
        except (AttributeError, StopIteration):
            device = None
            rendered_device = None
        return {
            "schema_version": 1,
            "device": rendered_device,
            "requested_model_backend": _native_model_backend_requested(),
            "last_prefill_backend": self.rwkv7_native_model_last_prefill_backend(),
            "last_decode_backend": self.rwkv7_native_model_last_decode_backend(),
            "kernels": kernel_runtime_report(torch_module=torch, device=device),
        }

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
                _rwkv7_prefill_graph_continuation=True,
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

    def rwkv7_last_fast_token_backend(self) -> str | None:
        """Return the backend selected by the previous fast-token call."""

        return self.rwkv7_native_model_last_decode_backend()

    def rwkv7_last_fast_prefill_backend(self) -> str | None:
        """Return the backend selected by the previous fast-prefill call."""

        return self.rwkv7_native_model_last_prefill_backend()

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
            weight_device = self.model.embeddings.weight.device.type
            graph_available = (
                _ascend_graph_available()
                if weight_device == "npu"
                else _native_graph_available()
            )
            if chosen in {"auto", "native_graph"} and graph_available:
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
