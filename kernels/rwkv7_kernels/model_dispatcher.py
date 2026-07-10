"""Whole-model capability dispatch for the optional kernel package.

This module is the only public bridge from a clean Hugging Face model object to
performance implementations.  Backend implementations live below
``rwkv7_kernels.model`` and may inspect the documented RWKV-7 module structure,
but may not import or replace ``rwkv7_hf.modeling_rwkv7``.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from .protocol import (
    support_result,
    validate_model_request,
)
_NOT_MIGRATED = (
    "whole-model backend-v2 is not available for this shape; "
    "the adapter will use its readable reference layer loop"
)
_MODEL_IMPL_ENV = "RWKV7_MODEL_KERNEL_IMPL"
_MODEL_IMPLS = ("auto", "dense")
_DENSE_IMPLEMENTATION = "native-torchscript-dense-sequential-v2"


def _phase(request: dict[str, Any]) -> str:
    if bool(request["training"]) or bool(request.get("grad_enabled", False)):
        return "training"
    hidden = request.get("hidden_states")
    if isinstance(hidden, torch.Tensor) and hidden.ndim >= 2:
        return "decode" if int(hidden.shape[1]) == 1 else "prefill"
    return "prefill"


def _requested_implementation() -> str:
    value = os.environ.get(_MODEL_IMPL_ENV, "auto").strip().lower()
    if value not in _MODEL_IMPLS:
        choices = ", ".join(_MODEL_IMPLS)
        raise ValueError(f"{_MODEL_IMPL_ENV} must be one of {choices}; got {value!r}")
    return value


def _probe_dense(owner: Any, request: dict[str, Any]):
    if request["model_kind"] != "base":
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="dense-v2 currently accepts the base model boundary only",
            phase=_phase(request),
        )
    hidden = request.get("hidden_states")
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="dense-v2 requires [B,T,D] hidden_states",
            phase=_phase(request),
        )
    if hidden.device.type != "cuda":
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="dense-v2 is an NVIDIA CUDA implementation",
            phase=_phase(request),
        )
    if hidden.dtype not in (torch.float16, torch.bfloat16):
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="dense-v2 requires FP16 or BF16 model tensors",
            phase=_phase(request),
        )
    if bool(request["training"]) or bool(request.get("grad_enabled", False)):
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="dense-v2 is inference-only while training kernels migrate",
            phase=_phase(request),
        )
    if not hasattr(owner, "layers") or not hasattr(owner, "norm"):
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="owner does not expose the clean RWKV7 base-model structure",
            phase=_phase(request),
        )
    return support_result(
        supported=True,
        implementation=_DENSE_IMPLEMENTATION,
        reason="explicit dense-v2 diagnostic implementation selected",
        phase=_phase(request),
    )


def probe_model_forward_v1(owner: Any, request: dict[str, Any]):
    """Return whether a migrated whole-model implementation accepts a call."""

    validate_model_request(request)
    if _requested_implementation() == "dense":
        return _probe_dense(owner, request)
    # Keep production auto disabled until every phase in the frozen one-shot
    # inventory has passed. Explicit dense diagnostics exercise the final ABI
    # without advertising a half-migrated production route.
    return support_result(
        supported=False,
        implementation="rwkv7-model-backend-v2",
        reason=_NOT_MIGRATED,
        phase=_phase(request),
    )


def model_forward_v1(owner: Any, request: dict[str, Any]):
    """Execute a supported whole-model request.

    Calling this after a negative probe is a protocol error.  Concrete phase
    dispatch is added here only after decode, prefill, cache and training
    implementations all satisfy the frozen backend-v2 acceptance matrix.
    """

    validate_model_request(request)
    implementation = _requested_implementation()
    if implementation != "dense":
        raise RuntimeError(_NOT_MIGRATED)
    support = _probe_dense(owner, request)
    if not support["supported"]:
        raise RuntimeError(support["reason"])
    from .model.dense import run_base_model

    return run_base_model(owner, request)


__all__ = ["model_forward_v1", "probe_model_forward_v1"]
