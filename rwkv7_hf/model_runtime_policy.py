# coding=utf-8
"""Environment and hardware policy for the native Transformers model.

This module owns policy decisions only.  The stable ``native_model`` facade
keeps small wrappers around these functions so existing tests, integrations,
and hardware-specific monkeypatches continue to target the historical module.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Callable

import torch

from .kernel_policy import current_kernel_policy


FALSE_VALUES = {"0", "false", "False", "no", "off"}


def cuda_device_guard(device, *, torch_module=torch):
    return (
        torch_module.cuda.device(device)
        if getattr(device, "type", None) == "cuda"
        and torch_module.cuda.is_available()
        else nullcontext()
    )


def bnb_skip_policy(
    policy: str | None = None,
    *,
    policy_device: int | str | None = None,
    hardware_policy: bool = True,
    kernel_policy_fn: Callable[..., Any] = current_kernel_policy,
) -> str:
    if policy is None:
        env_policy = os.environ.get("RWKV7_BNB_SKIP_POLICY")
        if env_policy is None and hardware_policy:
            env_policy = str(
                getattr(
                    kernel_policy_fn(device=policy_device),
                    "bnb_skip_policy",
                    "memory",
                )
            )
        if env_policy is None:
            env_policy = "memory"
        policy = env_policy
    policy = str(policy).strip().lower()
    if policy in {"", "default", "small_lora", "memory", "minimal"}:
        return "memory"
    if policy in {"decode", "decode_hot", "hot", "hybrid"}:
        return "decode_hot"
    if policy in {"output", "output_hot", "o_proj", "o_proj_hot"}:
        return "output_hot"
    if policy in {"prefill", "prefill_hot", "throughput"}:
        return "prefill_hot"
    if policy in {"decode_rk", "rk_dense"}:
        return "decode_rk"
    if policy in {"dense", "all_dense", "no_quant"}:
        return "dense"
    return "memory"


def bnb_prefill_value_stride() -> int:
    raw = os.environ.get("RWKV7_BNB_PREFILL_VALUE_STRIDE", "8").strip()
    try:
        return min(max(1, int(raw)), 4096)
    except ValueError:
        return 8


def bnb_int8_threshold_override(
    *,
    policy_device: int | str | None = None,
    hardware_policy: bool = True,
    kernel_policy_fn: Callable[..., Any] = current_kernel_policy,
) -> float | None:
    raw = os.environ.get("RWKV7_BNB_INT8_THRESHOLD")
    if raw is None and hardware_policy:
        raw = getattr(
            kernel_policy_fn(device=policy_device),
            "bnb_int8_threshold",
            None,
        )
    if raw is None or str(raw).strip().lower() in {"", "default", "library", "none"}:
        return None
    value = float(raw)
    if value < 0.0:
        raise ValueError("RWKV7_BNB_INT8_THRESHOLD must be non-negative")
    return value


def native_model_jit_enabled(
    *,
    kernel_policy_fn: Callable[..., Any] = current_kernel_policy,
) -> bool:
    raw = os.environ.get("RWKV7_NATIVE_MODEL_JIT")
    if raw is not None:
        return raw not in FALSE_VALUES
    try:
        # Compatibility/private-use runtimes with no validated JIT path must
        # not inherit another vendor's packed execution assumptions.
        return getattr(kernel_policy_fn().profile, "family", None) not in {
            "metax",
            "biren",
        }
    except Exception:
        return True


def native_model_backend_requested(
    *,
    jit_enabled_fn: Callable[[], bool] = native_model_jit_enabled,
) -> str:
    raw = os.environ.get("RWKV7_NATIVE_MODEL_BACKEND")
    if raw is None:
        # ``RWKV7_FAST_TOKEN_BACKEND`` is the public switch used by the
        # production HF wrapper and by the cross-model benchmark protocol.
        # Honor its native-compatible values in the FLA-free model too so a
        # requested ``native_jit`` fair line cannot silently become an
        # ``auto`` / CUDA-graph run after repo-code model dispatch.
        legacy = os.environ.get("RWKV7_FAST_TOKEN_BACKEND")
        if legacy is not None:
            normalized_legacy = legacy.strip().lower()
            if normalized_legacy in {
                "",
                "auto",
                "eager",
                "torch",
                "native",
                "native_jit",
                "jit",
                "native_graph",
                "cuda_graph",
                "graph",
            }:
                raw = legacy
    if raw is None:
        return "auto" if jit_enabled_fn() else "eager"
    backend = raw.strip().lower()
    aliases = {
        "": "auto",
        "graph": "native_graph",
        "cuda_graph": "native_graph",
        "jit": "native_jit",
        "torch": "eager",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"auto", "eager", "native_jit", "native_graph"}:
        raise ValueError(
            "RWKV7_NATIVE_MODEL_BACKEND must be auto, eager, native_jit, or native_graph; "
            f"got {raw!r}"
        )
    return backend


def native_prefill_graph_enabled(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    device: int | str | torch.device | None = None,
    *,
    native_jit_prefill_available: bool,
    torch_module=torch,
    kernel_policy_fn: Callable[..., Any] = current_kernel_policy,
) -> bool:
    raw = os.environ.get("RWKV7_NATIVE_PREFILL_GRAPH")
    if raw is not None:
        selected = raw not in FALSE_VALUES
    else:
        policy = kernel_policy_fn(device=device, torch_module=torch_module)
        selected = bool(getattr(policy, "prefill_graph", False))
        shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(policy, "prefill_graph_model_shapes", ())
            if len(shape) == 4
        }
        if selected and shapes:
            if None in (batch_size, prompt_tokens, hidden_size, num_layers):
                selected = False
            else:
                selected = (
                    int(hidden_size),
                    int(num_layers),
                    int(batch_size),
                    int(prompt_tokens),
                ) in shapes
    return bool(
        selected
        and torch_module.cuda.is_available()
        and native_jit_prefill_available
    )


def native_prefill_external_quant_graph_enabled(
    device: int | str | torch.device | None = None,
    *,
    torch_module=torch,
    kernel_policy_fn: Callable[..., Any] = current_kernel_policy,
) -> bool:
    """Allow external quantized modules in prefill graphs only on proven lanes."""

    raw = os.environ.get("RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT_GRAPH")
    if raw is not None:
        return raw not in FALSE_VALUES
    policy = kernel_policy_fn(device=device, torch_module=torch_module)
    return bool(getattr(policy, "native_external_quant_prefill_graph", False))


def native_prefill_graph_cache_size(
    device: int | str | torch.device | None = None,
    *,
    torch_module=torch,
    kernel_policy_fn: Callable[..., Any] = current_kernel_policy,
) -> int:
    policy = kernel_policy_fn(device=device, torch_module=torch_module)
    default = int(getattr(policy, "prefill_graph_cache_size", 2))
    try:
        value = int(
            os.environ.get(
                "RWKV7_NATIVE_PREFILL_GRAPH_CACHE_SIZE",
                str(default),
            )
        )
    except ValueError:
        value = default
    return max(1, min(value, 16))


def native_prefill_graph_signature() -> tuple[tuple[str, str], ...]:
    """Return every explicit prefill setting that changes a captured graph."""

    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("RWKV7_NATIVE_PREFILL_")
        )
    )


__all__ = [
    "FALSE_VALUES",
    "bnb_int8_threshold_override",
    "bnb_prefill_value_stride",
    "bnb_skip_policy",
    "cuda_device_guard",
    "native_model_backend_requested",
    "native_model_jit_enabled",
    "native_prefill_external_quant_graph_enabled",
    "native_prefill_graph_cache_size",
    "native_prefill_graph_enabled",
    "native_prefill_graph_signature",
]
