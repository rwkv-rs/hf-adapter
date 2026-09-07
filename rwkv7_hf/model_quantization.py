# coding=utf-8
"""Quantized loading and native W8/W4 replacement for RWKV-7 CausalLM.

This module owns the Hugging Face bitsandbytes load policy and config-driven
native MM8/MM4 module replacement. Runtime policy helpers remain reachable
through ``native_model`` so existing monkeypatch and per-GPU policy entrypoints
keep their historical behavior.
"""
from __future__ import annotations

from typing import Any


def _native_model_entrypoint():
    # Import lazily to avoid a cycle while native_model assembles the mixin MRO.
    from . import native_model

    return native_model


def _bnb_prefill_value_stride() -> int:
    return _native_model_entrypoint()._bnb_prefill_value_stride()


def _bnb_skip_policy(*args, **kwargs):
    return _native_model_entrypoint()._bnb_skip_policy(*args, **kwargs)


def _bnb_int8_threshold_override(*args, **kwargs):
    return _native_model_entrypoint()._bnb_int8_threshold_override(*args, **kwargs)


def single_cuda_device_from_device_map(*args, **kwargs):
    return _native_model_entrypoint().single_cuda_device_from_device_map(*args, **kwargs)


class _NativeQuantizationMixin:
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
        hardware_policy, policy_device = single_cuda_device_from_device_map(
            kwargs.get("device_map")
        )
        policy = _bnb_skip_policy(
            kwargs.pop("rwkv7_bnb_skip_policy", None),
            policy_device=policy_device,
            hardware_policy=hardware_policy,
        )
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
            threshold = _bnb_int8_threshold_override(
                policy_device=policy_device,
                hardware_policy=hardware_policy,
            )
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
        use_fp8 = bool(getattr(self.config, "use_native_fp8", False))
        if not (use_mm8 or use_mm4 or use_fp8):
            setattr(self, "_rwkv7_native_mm_quantization", None)
            setattr(self, "_rwkv7_native_mm_replaced_modules", 0)
            return 0
        active = sum([use_mm8, use_mm4, use_fp8])
        if active > 1:
            raise ValueError(
                "use_native_mm8, use_native_mm4, and use_native_fp8 are mutually exclusive"
            )
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
        if use_fp8:
            from .native_quant_fp8 import quantize_model_fp8

            replaced = int(
                quantize_model_fp8(
                    self,
                    min_params=int(getattr(self.config, "native_fp8_min_params", 8_000_000)),
                    policy=str(getattr(self.config, "native_fp8_policy", "memory")),
                )
            )
            quantization = "fp8"
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
        self.rwkv7_clear_native_graph_cache()
        self.rwkv7_clear_native_prefill_graph_cache()
