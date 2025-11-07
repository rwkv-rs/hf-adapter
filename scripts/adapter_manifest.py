#!/usr/bin/env python3
# coding=utf-8
"""Single source of truth for files shipped with converted HF checkpoints.

Keep this module dependency-free: converter and sync tools import it before
optional ML/Apple dependencies are available. Runtime import closure is checked
by ``tests/test_sync_hf_adapter_code.py``.
"""
from __future__ import annotations


ADAPTER_FILES = [
    "ada_lora.py",
    "ada_sparse_ffn.py",
    "configuration_rwkv7.py",
    "dplr_prefill.py",
    "dplr_prefill_triton.py",
    "fused_attention_projection.py",
    "fused_ffn.py",
    "fused_lora.py",
    "fused_decode_norm_mix.py",
    "fused_norm_mix.py",
    "fused_output.py",
    "fused_prefill.py",
    "fused_projection.py",
    "fused_recurrent_update.py",
    "fused_elementwise.py",
    "fused_time_mix.py",
    "kernel_policy.py",
    "mlx_bridge.py",
    "mlx_dplr_prefill.py",
    "mlx_model.py",
    "mlx_mix.py",
    "mlx_policy.py",
    "mlx_quant.py",
    "mlx_scan.py",
    "mlx_session.py",
    "mlx_state.py",
    "mlx_wkv.py",
    "modeling_rwkv7.py",
    "native.py",
    "native_jit.py",
    "native_model.py",
    "native_quant.py",
    "native_quant_a8w8.py",
    "native_quant_mm4.py",
    "native_quant_mm8.py",
    "native_quant_torchao.py",
    "native_quant_policy.py",
    "sm70_linear.py",
    "sm70_quant.py",
    "sm70_wagv.py",
    "triton_compat.py",
    "tokenization_rwkv7.py",
]


__all__ = ["ADAPTER_FILES"]
