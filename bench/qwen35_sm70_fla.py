"""Volta-compatible Qwen3.5 FLA prefill binding.

FLA's chunk Gated Delta Rule prefill reaches a KKT ``tl.dot`` lowering that is
not supported by the production Triton stack on sm_70.  FLA's fused recurrent
operator implements the same Gated Delta Rule recurrence, accepts a complete
``[B, T, ...]`` sequence, and is also the operator used by Qwen3.5 cached
decode.  Bind that fused FLA operator for multi-token prefill on Volta while
leaving newer architectures on the faster chunk implementation.
"""
from __future__ import annotations

from typing import Any


def qwen35_sm70_recurrent_gated_delta_rule(*args, **kwargs):
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    # HF's chunk call passes only arguments supported by the recurrent API in
    # normal and packed-sequence inference. Keep this guard fail-closed if a
    # future Transformers release starts passing chunk-only controls.
    unsupported = {"cu_seqlens_cpu", "cp_context"}.intersection(kwargs)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(f"sm70 recurrent prefill does not support: {names}")
    return fused_recurrent_gated_delta_rule(*args, **kwargs)


def bind_qwen35_sm70_fla(model: Any) -> int:
    """Bind FLA's fused recurrent prefill to every Qwen3.5 Gated DeltaNet layer."""

    replaced = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        if not hasattr(module, "chunk_gated_delta_rule"):
            continue
        module.chunk_gated_delta_rule = qwen35_sm70_recurrent_gated_delta_rule
        replaced += 1
    model._qwen35_sm70_fla_layers = int(replaced)
    model._qwen35_sm70_fla_prefill = "fused_recurrent"
    return replaced


__all__ = ["bind_qwen35_sm70_fla", "qwen35_sm70_recurrent_gated_delta_rule"]
