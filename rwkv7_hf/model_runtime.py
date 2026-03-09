# coding=utf-8
"""Backend-neutral runtime dispatch for the native RWKV-7 HF model.

The public and monkeypatch-compatible runtime symbols intentionally remain in
``native_model``.  This mixin resolves those symbols lazily at call time so
tests, downstream instrumentation, and optional CUDA/ROCm backends keep the
historical entrypoint surface while model ownership stays modular.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from .model_cache import (
    NativeRWKV7Cache,
    _copy_native_cache_tuple,
)
from .model_prefill_graph import _NativePrefillGraphRunner


def _native_model_entrypoint():
    """Resolve the canonical entrypoint without creating an import cycle."""

    from . import native_model

    return native_model


def _entrypoint_call(name: str, *args, **kwargs):
    return getattr(_native_model_entrypoint(), name)(*args, **kwargs)


def _cuda_device_guard(device):
    return _entrypoint_call("_cuda_device_guard", device)


def _native_model_backend_requested() -> str:
    return _entrypoint_call("_native_model_backend_requested")


def _native_model_jit_enabled() -> bool:
    return _entrypoint_call("_native_model_jit_enabled")


def _native_tensor_parallel_active(model) -> bool:
    return _entrypoint_call("_native_tensor_parallel_active", model)


def _native_prefill_graph_enabled(*args, **kwargs) -> bool:
    return _entrypoint_call("_native_prefill_graph_enabled", *args, **kwargs)


def _native_prefill_external_quant_graph_enabled(device=None) -> bool:
    return _entrypoint_call("_native_prefill_external_quant_graph_enabled", device)


def _native_prefill_graph_cache_size(device=None) -> int:
    return _entrypoint_call("_native_prefill_graph_cache_size", device)


def _native_prefill_graph_signature():
    return _entrypoint_call("_native_prefill_graph_signature")


def _native_graph_available() -> bool:
    return _entrypoint_call("_native_graph_available")


def _native_graph_cache_size() -> int:
    return _entrypoint_call("_native_graph_cache_size")


def _native_graph_runtime_signature():
    return _entrypoint_call("_native_graph_runtime_signature")


def _native_graph_stats_template():
    return _entrypoint_call("_native_graph_stats_template")


def _ascend_graph_available() -> bool:
    return _entrypoint_call("_ascend_graph_available")


def _ascend_graph_cache_size() -> int:
    return _entrypoint_call("_ascend_graph_cache_size")


def _ascend_graph_module_signature(owner):
    return _entrypoint_call("_ascend_graph_module_signature", owner)


def _ascend_graph_runtime_signature():
    return _entrypoint_call("_ascend_graph_runtime_signature")


class _NativeRuntimeMixin:
    """Prefill/decode backend selection, graph caches, and route safety."""

    def _native_prefill_can_run(
        self,
        input_ids: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool,
        use_cache: bool,
        logits_to_keep,
    ) -> bool:
        native_jit_prefill = getattr(_native_model_entrypoint(), "_native_jit_prefill", None)
        if _native_model_backend_requested() == "eager":
            return False
        if _native_tensor_parallel_active(self):
            return False
        if self._rwkv7_has_multi_cuda_device_map():
            return False
        if self.training or torch.is_grad_enabled() or native_jit_prefill is None:
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

    def _native_prefill(
        self,
        input_ids: torch.LongTensor,
        *,
        logits_to_keep,
        seen_tokens: int,
        initial_cache=None,
    ):
        entrypoint = _native_model_entrypoint()
        prefill_graph_runner_cls = entrypoint._NativePrefillGraphRunner
        native_jit_prefill = entrypoint._native_jit_prefill
        if native_jit_prefill is None:
            raise RuntimeError("native prefill runtime is unavailable")
        batch_size = int(input_ids.shape[0])
        prompt_tokens = int(input_ids.shape[1])
        graph_quant_safe = initial_cache is None and self._native_prefill_graph_quant_safe(
            input_ids.device
        )
        if (
            graph_quant_safe
            and _native_prefill_graph_enabled(
                batch_size,
                prompt_tokens,
                int(self.config.hidden_size),
                int(self.config.num_hidden_layers),
                input_ids.device,
            )
        ):
            runner = getattr(self, "_rwkv7_native_prefill_graph_hot_runner", None)
            if not isinstance(runner, prefill_graph_runner_cls) or not runner.matches(
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
        logits, state, xpa, xpf = native_jit_prefill(
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
        weight = self.model.embeddings.weight
        guard = _cuda_device_guard(weight.device)
        with guard:
            return _NativeRuntimeMixin._native_prefill_graph_runner_current_device(
                self,
                batch_size,
                prompt_tokens,
                logits_to_keep,
            )

    def _native_prefill_graph_runner_current_device(
        self,
        batch_size: int,
        prompt_tokens: int,
        logits_to_keep,
    ) -> _NativePrefillGraphRunner:
        prefill_graph_runner_cls = _native_model_entrypoint()._NativePrefillGraphRunner
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
        while len(cache) >= _native_prefill_graph_cache_size(weight.device):
            _, evicted = cache.popitem(last=False)
            if getattr(self, "_rwkv7_native_prefill_graph_hot_runner", None) is evicted:
                self._rwkv7_native_prefill_graph_hot_runner = None
            evicted.detach_bound_cache()
            stats["evictions"] = int(stats.get("evictions", 0)) + 1
        runner = prefill_graph_runner_cls(
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
        requested = _native_model_backend_requested()
        if requested not in {"auto", "native_graph"}:
            return False
        if _native_tensor_parallel_active(self):
            return False
        if self._rwkv7_has_multi_cuda_device_map():
            return False
        weight_device = self.model.embeddings.weight.device
        graph_available = (
            _ascend_graph_available()
            if weight_device.type == "npu"
            else _native_graph_available()
        )
        if self.training or torch.is_grad_enabled() or not graph_available:
            return False
        if self._native_model_has_adapter_layers():
            return False
        if (
            self._native_model_quantized()
            and not self._native_model_quant_graph_safe(weight_device.type)
        ):
            return False
        if token_ids is None or token_ids.dim() != 2 or int(token_ids.shape[1]) != 1:
            return False
        if attention_mask is not None or output_hidden_states or not isinstance(cache, NativeRWKV7Cache):
            return False
        if weight_device.type not in {"cuda", "npu"}:
            return False
        if token_ids.device.type != weight_device.type:
            return False
        if token_ids.device != weight_device:
            return False
        if not cache.is_initialized or cache.get_batch_size() != int(token_ids.shape[0]):
            return False
        return True

    def _native_graph_packs(self):
        native_graph_extract = getattr(_native_model_entrypoint(), "_native_graph_extract", None)
        if native_graph_extract is None:
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
            packs, _, _, _ = native_graph_extract(self)
            self._rwkv7_native_graph_pack_cache = (key, packs)
            return packs
        return cache[1]

    def _native_graph_runner(self, batch_size: int):
        weight = self.model.embeddings.weight
        if weight.device.type == "npu":
            return _NativeRuntimeMixin._native_graph_runner_current_device(
                self,
                batch_size,
            )
        guard = _cuda_device_guard(weight.device)
        with guard:
            return _NativeRuntimeMixin._native_graph_runner_current_device(
                self,
                batch_size,
            )

    def _native_graph_runner_current_device(self, batch_size: int):
        entrypoint = _native_model_entrypoint()
        weight = self.model.embeddings.weight
        if weight.device.type == "npu":
            runner_cls = getattr(entrypoint, "_AscendGraphRunner", None)
            if runner_cls is None:
                raise RuntimeError("Ascend native_graph runtime is unavailable")
            key = (
                weight.device.type,
                weight.device.index,
                weight.dtype,
                len(self.model.layers),
                str(getattr(self, "_rwkv7_native_mm_quantization", "none")),
                int(getattr(self, "_rwkv7_native_mm_replaced_modules", 0)),
                _ascend_graph_module_signature(self),
                _ascend_graph_runtime_signature(),
                int(batch_size),
            )
            cache_limit = _ascend_graph_cache_size()
            runner_args = (self, int(batch_size))
        else:
            runner_cls = getattr(entrypoint, "_NativeGraphRunner", None)
            if runner_cls is None:
                raise RuntimeError("native_graph runtime is unavailable")
            packs = self._native_graph_packs()
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
            cache_limit = _native_graph_cache_size()
            runner_args = (self, packs, int(batch_size))
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
        while len(cache) >= cache_limit:
            _, evicted = cache.popitem(last=False)
            if hasattr(evicted, "detach_bound_cache"):
                evicted.detach_bound_cache()
            stats["evictions"] = int(stats.get("evictions", 0)) + 1
        runner = runner_cls(*runner_args)
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
        graph_cache_limit = (
            _ascend_graph_cache_size()
            if self.model.embeddings.weight.device.type == "npu"
            else _native_graph_cache_size()
        )
        stats.update(
            {
                "size": len(self.rwkv7_native_graph_cache_batch_sizes()),
                "limit": graph_cache_limit,
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
                "limit": _native_prefill_graph_cache_size(
                    self.model.embeddings.weight.device
                ),
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

    def _native_model_quantized(self) -> bool:
        """True if layer projections were replaced by quantized modules.

        The JIT decode path extracts raw layer ``.weight`` tensors into packs,
        which cannot represent bnb or native MM8/MM4 layer replacements.  When
        layers are quantized, decode must use the eager per-token path whose
        module calls invoke the quantized linears.  ``lm_head``-only quantization
        is safe for JIT because ``native_jit._lm_head`` calls the module.
        Detected by class name to avoid importing optional quantization deps.
        """
        quantized_names = {
            "Linear4bit",
            "Linear8bit",
            "Linear8bitLt",
            "MM8Linear",
            "MM4Linear",
            "AscendW8A16Linear",
            "AscendWeightOnlyLinear",
            "AscendW4A16Linear",
        }
        try:
            return any(type(module).__name__ in quantized_names for module in self.model.layers.modules())
        except Exception:
            return False

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

    def _native_model_ascend_quant_graph_safe(self) -> bool:
        """Allow only adapter-owned Ascend packed modules in NPUGraph decode."""

        ascend_names = {
            "AscendW8A16Linear",
            "AscendWeightOnlyLinear",
            "AscendW4A16Linear",
        }
        external_names = {
            "Linear4bit",
            "Linear8bit",
            "Linear8bitLt",
            "MM8Linear",
            "MM4Linear",
        }
        seen_ascend = False
        try:
            modules = self.model.layers.modules()
        except Exception:
            return False
        for module in modules:
            name = type(module).__name__
            if name in external_names:
                return False
            if name in ascend_names:
                seen_ascend = True
        return seen_ascend

    def _native_model_quant_graph_safe(self, device_type: str) -> bool:
        if str(device_type) == "npu":
            return self._native_model_ascend_quant_graph_safe()
        return self._native_model_native_quant_graph_safe()

    def _native_prefill_graph_quant_safe(
        self,
        device: int | str | torch.device | None = None,
    ) -> bool:
        """Fail closed for external quant modules on unvalidated graph lanes.

        Native MM8/MM4 modules expose graph-safe preallocated-output hooks.
        Bitsandbytes and other external wrappers may synchronize or inspect
        tensor values during forward, so they require an explicit per-card
        policy (or environment override) before CUDA graph capture.
        """

        if not self._native_model_quantized():
            return True
        if self._native_model_native_quant_graph_safe():
            return True
        return _native_prefill_external_quant_graph_enabled(device)

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
        entrypoint = _native_model_entrypoint()
        native_jit_extract = getattr(entrypoint, "_native_jit_extract", None)
        native_jit_step_batched = getattr(entrypoint, "_native_jit_step_batched", None)
        if _native_model_backend_requested() == "eager":
            return None
        if _native_tensor_parallel_active(self):
            return None
        if self._rwkv7_has_multi_cuda_device_map():
            return None
        if not _native_model_jit_enabled() or native_jit_extract is None or native_jit_step_batched is None:
            return None
        if self._native_model_requires_eager_decode():
            return None
        weight = self.model.embeddings.weight
        if weight.device.type == "supa":
            return None
        key = (weight.device.type, weight.device.index, weight.dtype)
        cache = getattr(self, "_rwkv7_native_model_jit_pack_cache", None)
        if cache is None or cache[0] != key:
            extracted = native_jit_extract(self)
            packs = extracted[0] if isinstance(extracted, tuple) and len(extracted) == 4 else extracted
            self._rwkv7_native_model_jit_pack_cache = (key, packs)
            return packs
        return cache[1]
