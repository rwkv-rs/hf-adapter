# coding=utf-8
"""Linear operands and low-memory packing shared by native JIT paths.

The functions in this module are deliberately independent from kernel policy
and optional CUDA/Triton implementations.  ``native_jit`` remains the stable
compatibility entrypoint and re-exports the historical underscore names.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


def dense_linear_module(module) -> bool:
    """Return whether ``module`` exposes the ordinary dense Linear contract.

    The clean HF reference uses a small ``nn.Linear`` subclass solely to keep
    its fixed-row reference GEMM readable.  The optional backend must treat
    that class as an ordinary dense operand without accidentally classifying
    quantized Linear subclasses whose weight is a tensor subclass.
    """

    module_type = type(module)
    exact_torch_linear = module_type is torch.nn.Linear
    class_forward = module_type.__dict__.get("forward")
    explicit_rwkv7_contract = (
        class_forward is not None
        and getattr(class_forward, "_rwkv7_dense_linear_contract", False) is True
    )
    return bool(
        (exact_torch_linear or explicit_rwkv7_contract)
        and type(getattr(module, "weight", None)) is torch.nn.Parameter
    )


def linear_module(module, x: torch.Tensor) -> torch.Tensor:
    """Call a dense or native quantized linear module."""

    if dense_linear_module(module):
        # The readable RWKV7 Linear owns a batch/length-invariant 128-row
        # reference GEMM for rank-3 model activations. Whole-model prefill
        # must retain that public numerical contract at the vocabulary head;
        # bypassing ``forward`` here lets cuBLAS choose a B*T-dependent
        # reduction and can move FP16 logits outside the fixed acceptance
        # envelope. Decode remains rank-1/2 and therefore keeps the direct
        # zero-overhead F.linear path below.
        class_forward = type(module).__dict__.get("forward")
        if (
            x.ndim == 3
            and class_forward is not None
            and getattr(class_forward, "_rwkv7_dense_linear_contract", False) is True
        ):
            return module(x)
        return F.linear(x, module.weight, module.bias)
    return module(x)


def graph_linear_operand(module):
    """Return a dense weight or retain a callable packed quantized module."""

    if dense_linear_module(module):
        return module.weight
    return module


def graph_linear_is_dense(operand) -> bool:
    return isinstance(operand, torch.Tensor)


def graph_linear_shape(operand) -> tuple[int, int]:
    if isinstance(operand, torch.Tensor):
        return int(operand.shape[0]), int(operand.shape[1])
    return int(operand.out_features), int(operand.in_features)


def relayout_ffn_value_weight(module):
    """Store an FFN down weight in the sparse kernel's transposed layout.

    The exposed parameter keeps its original ``[hidden, ffn]`` shape, so
    ``F.linear`` and state-dict names remain compatible. Its backing storage is
    contiguous as ``[ffn, hidden]``, allowing the sparse decode kernel to reuse
    the same bytes instead of allocating a second full-size model copy.
    """

    if not dense_linear_module(module) or module.bias is not None:
        raise TypeError("low-memory sparse FFN packing requires a bias-free nn.Linear")
    if getattr(module, "_rwkv7_sparse_low_memory_layout", False):
        return module.weight
    weight = module.weight
    if weight.dim() != 2:
        raise ValueError("low-memory sparse FFN packing requires a rank-2 weight")
    packed = weight.detach().transpose(0, 1).contiguous()
    module.weight = torch.nn.Parameter(
        packed.transpose(0, 1), requires_grad=bool(weight.requires_grad)
    )
    module._rwkv7_sparse_low_memory_layout = True
    return module.weight


def try_relayout_ffn_value_weight(
    module,
    *,
    relayout_fn: Callable = relayout_ffn_value_weight,
) -> bool:
    """Apply sparse layout only to its exact dense fp16 CUDA contract."""

    if not dense_linear_module(module) or module.bias is not None:
        return False
    weight = module.weight
    if type(weight) is not torch.nn.Parameter:
        return False
    if weight.device.type != "cuda" or weight.dtype != torch.float16:
        return False
    relayout_fn(module)
    return True


__all__ = [
    "dense_linear_module",
    "graph_linear_is_dense",
    "graph_linear_operand",
    "graph_linear_shape",
    "linear_module",
    "relayout_ffn_value_weight",
    "try_relayout_ffn_value_weight",
]
