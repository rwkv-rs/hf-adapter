# coding=utf-8
"""Default-off Q4_K_M-inspired native mixed W4/W8 profile.

This is not a GGUF bit-for-bit implementation. It keeps the more sensitive
FFN value/down projection and output head at W8 while using W4 for the other
size-gated matrices, matching the repository's existing MLX quality policy.
"""
from __future__ import annotations

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .native_quant_mm4 import MM4Linear
from .native_quant_mm8 import MM8Linear
from .native_quant_policy import normalize_native_mm_policy, should_quantize_linear


def native_q4km_bits_for_module(name: str) -> int:
    sensitive = (
        name == "lm_head"
        or name.endswith(".lm_head")
        or name.endswith("ffn.value")
        or name.endswith("attn.r_proj")
        or name.endswith("attn.v_proj")
    )
    return 8 if sensitive else 4


def quantize_model_q4km(
    model,
    *,
    min_params: int = 8_000_000,
    policy: str = "memory",
    fused: bool = True,
) -> int:
    """Replace eligible linears with the native mixed W4/W8 profile."""

    if torch is None:
        raise RuntimeError("quantize_model_q4km requires torch")
    policy = normalize_native_mm_policy(policy)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and should_quantize_linear(
            name,
            int(module.weight.numel()),
            min_params=min_params,
            policy=policy,
        )
    ]
    histogram = {"4": 0, "8": 0}
    for full_name in targets:
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        linear = getattr(parent, attr)
        bits = native_q4km_bits_for_module(full_name)
        replacement = (
            MM8Linear(linear, fused=fused)
            if bits == 8
            else MM4Linear(linear, fused=fused)
        )
        setattr(parent, attr, replacement)
        histogram[str(bits)] += 1
    setattr(model, "_rwkv7_native_mm_quantization", "mm4_q4km")
    setattr(model, "_rwkv7_native_mm_replaced_modules", len(targets))
    setattr(model, "_rwkv7_native_mm_bits_histogram", histogram)
    for cache_attr in (
        "_rwkv7_native_jit_pack_cache",
        "_rwkv7_native_graph_pack_cache",
        "_rwkv7_native_graph_runner_cache",
        "_rwkv7_native_prefill_graph_runner_cache",
        "_rwkv7_native_prefill_graph_hot_runner",
    ):
        if hasattr(model, cache_attr):
            delattr(model, cache_attr)
    return len(targets)
