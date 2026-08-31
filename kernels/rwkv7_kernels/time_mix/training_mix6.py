"""Explicit-shift native Mix6 leaf for the readable HF training loop.

The Hugging Face model computes the masked previous-token tensor.  This leaf
owns only six independent ``x + (shifted - x) * mix`` expressions and their
autograd edge; it never sees a model, cache, mask, module, or optimizer.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch

from ..nvidia.extension_build import (
    CUDA_EXTENSION_OPTIMIZATION_FLAGS,
    cuda_build_verbose,
    cuda_extension_build_environment,
    cuda_include_paths,
    resolve_cuda_home,
    validate_bf16_cuda_runtime,
)


IMPLEMENTATION = "native-nvidia-rwkv7-mix6-training-v1"
OPERATOR_NAMESPACE = "rwkv7_tmix_mix6_shifted_bf16_v1"
NATIVE_BACKWARD_MIN_ROWS = 32

_LOAD_LOCK = threading.Lock()
_LOADED = False
_LOAD_ERROR: BaseException | None = None
_CAPABILITY_LOCK = threading.Lock()
_CAPABILITY_DEVICES: set[tuple[str, int | None]] = set()


def _device_key(device: torch.device) -> tuple[str, int | None]:
    device = torch.device(device)
    return device.type, device.index


def _bf16_capability_available(device: torch.device) -> bool:
    key = _device_key(device)
    with _CAPABILITY_LOCK:
        if key in _CAPABILITY_DEVICES:
            return True
    if torch.cuda.get_device_capability(device) < (8, 0):
        return False
    with _CAPABILITY_LOCK:
        _CAPABILITY_DEVICES.add(key)
    return True


def _unsupported(reason: str) -> dict[str, Any]:
    return {
        "supported": False,
        "implementation": IMPLEMENTATION,
        "reason": reason,
    }


def _operator_registered() -> bool:
    try:
        namespace = getattr(torch.ops, OPERATOR_NAMESPACE)
        getattr(namespace, "forward")
        getattr(namespace, "backward")
    except (AttributeError, RuntimeError):
        return False
    return True


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1] / "nvidia" / "csrc" / "training"


def load_mix6_training_cuda_extension(
    *,
    device: torch.device | int | None = None,
    verbose: bool | None = None,
) -> None:
    """JIT-load only the explicit-shift Mix6 operator for ``device``."""

    global _LOADED, _LOAD_ERROR
    if _LOADED or _operator_registered():
        _LOADED = True
        return
    if _LOAD_ERROR is not None:
        raise RuntimeError(
            "Mix6 training CUDA extension previously failed to load"
        ) from _LOAD_ERROR
    with _LOAD_LOCK:
        if _LOADED or _operator_registered():
            _LOADED = True
            return
        try:
            capability = validate_bf16_cuda_runtime(device)
            arch_list = f"{capability[0]}.{capability[1]}"
            with cuda_extension_build_environment(arch_list=arch_list):
                from torch.utils import cpp_extension

                previous_cuda_home = cpp_extension.CUDA_HOME
                try:
                    cuda_home = resolve_cuda_home(cpp_extension)
                    if cuda_home is None:
                        raise RuntimeError(
                            "Mix6 training CUDA JIT requires a local CUDA toolkit; "
                            "set CUDA_HOME to the toolkit matching the PyTorch "
                            "CUDA build"
                        )
                    root = _source_root()
                    cpp_extension.load(
                        name="rwkv7_kernels_tmix_mix6_shifted_bf16_v1",
                        sources=[
                            str(root / "rwkv7_tmix_mix6_shifted_bf16_v1.cpp"),
                            str(root / "rwkv7_tmix_mix6_shifted_bf16_v1.cu"),
                        ],
                        extra_cflags=["-O3"],
                        extra_cuda_cflags=list(CUDA_EXTENSION_OPTIMIZATION_FLAGS),
                        # Pip-split CUDA installations may expose the public
                        # headers separately from ``crt/host_config.h``.  The
                        # toolkit target include is therefore part of this
                        # leaf's reproducible build contract.
                        extra_include_paths=cuda_include_paths(
                            cuda_home,
                            include_target=True,
                        ),
                        is_python_module=False,
                        verbose=cuda_build_verbose(verbose),
                    )
                finally:
                    cpp_extension.CUDA_HOME = previous_cuda_home
            if not _operator_registered():
                raise RuntimeError(f"extension did not register {OPERATOR_NAMESPACE}")
            _LOADED = True
        except BaseException as exc:
            _LOAD_ERROR = exc
            raise RuntimeError(
                f"Mix6 training CUDA extension failed to load: {exc}"
            ) from exc


def probe_mix6_training_v1(
    value: torch.Tensor,
    shifted: torch.Tensor,
    *mixes: torch.Tensor,
) -> dict[str, Any]:
    """Report whether one tensor-only Mix6 request can use native CUDA."""

    if not isinstance(value, torch.Tensor) or not isinstance(shifted, torch.Tensor):
        return _unsupported("Mix6 value and shifted inputs must be tensors")
    if value.ndim != 3 or tuple(shifted.shape) != tuple(value.shape):
        return _unsupported("Mix6 value and shifted inputs must share [B,T,C]")
    batch, tokens, channels = value.shape
    if batch <= 0 or tokens <= 0 or channels <= 0:
        return _unsupported(
            "Mix6 requires non-empty batch, token, and channel dimensions"
        )
    if channels % 2:
        return _unsupported("native Mix6 requires an even channel dimension")
    if len(mixes) != 6 or any(
        not isinstance(mix, torch.Tensor) or mix.numel() != channels for mix in mixes
    ):
        return _unsupported("Mix6 requires six channel-sized parameter tensors")
    tensors = (value, shifted, *mixes)
    if not all(tensor.is_cuda for tensor in tensors):
        return _unsupported("native Mix6 requires CUDA tensors")
    if any(tensor.device != value.device for tensor in tensors):
        return _unsupported("all Mix6 tensors must share one CUDA device")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        return _unsupported("native Mix6 requires BF16 tensors")
    if not torch.cuda.is_available():
        return _unsupported("native Mix6 requires an available CUDA runtime")
    if not _bf16_capability_available(value.device):
        return _unsupported("native BF16 Mix6 requires sm80 or newer")
    return {
        "supported": True,
        "implementation": IMPLEMENTATION,
        "reason": ("explicit-shift BF16 CUDA Mix6 forward/autograd leaf is supported"),
    }


def _canonical_mix6(
    value: torch.Tensor,
    shifted: torch.Tensor,
    mixes: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    delta = shifted - value
    return tuple(value + delta * mix.reshape(1, 1, -1) for mix in mixes)


def _canonical_mix6_higher_order_backward(
    value: torch.Tensor,
    shifted: torch.Tensor,
    mixes: tuple[torch.Tensor, ...],
    output_grads: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Express the local VJP without traversing either input's parent graph.

    ``value`` and ``shifted`` are independent inputs at the custom-autograd
    boundary even when both were derived from the same hidden state. Calling
    ``autograd.grad`` on those non-leaf tensors from inside ``backward`` can
    otherwise consume their shared parent graph before the outer engine gets
    to it.  This explicit local VJP preserves higher-order differentiation and
    the same fixed six-term accumulation order as the CUDA kernel.
    """

    direct = output_grads[0]
    grad_delta = output_grads[0] * mixes[0].reshape(1, 1, -1)
    for gradient, mix in zip(output_grads[1:], mixes[1:], strict=True):
        direct = direct + gradient
        grad_delta = grad_delta + gradient * mix.reshape(1, 1, -1)
    difference = shifted - value
    mix_gradients = tuple(
        (gradient * difference).sum(dim=(0, 1)).reshape_as(mix)
        for gradient, mix in zip(output_grads, mixes, strict=True)
    )
    return direct - grad_delta, grad_delta, *mix_gradients


def _bf16x2_contiguous(value: torch.Tensor) -> torch.Tensor:
    """Return storage safe for the CUDA leaf's packed two-BF16 accesses."""

    packed = value.contiguous()
    # ``contiguous()`` may return a view whose storage_offset is odd.  Such a
    # tensor is logically contiguous but its data pointer is only two-byte
    # aligned, which is insufficient for the kernel's BF16x2 loads.
    if packed.data_ptr() % 4:
        packed = packed.clone(memory_format=torch.contiguous_format)
    return packed


class _Mix6Shifted(torch.autograd.Function):
    """Native first-order leaf with exact PyTorch higher-order replay."""

    @staticmethod
    def forward(ctx, value, shifted, x_r, x_w, x_k, x_v, x_a, x_g):
        saved = (value, shifted, x_r, x_w, x_k, x_v, x_a, x_g)
        inputs = tuple(_bf16x2_contiguous(item) for item in saved)
        ctx.save_for_backward(*saved)
        namespace = getattr(torch.ops, OPERATOR_NAMESPACE)
        return tuple(namespace.forward(*inputs))

    @staticmethod
    def backward(ctx, grad_r, grad_w, grad_k, grad_v, grad_a, grad_g):
        value, shifted, *mixes = ctx.saved_tensors
        output_grads = tuple(
            _bf16x2_contiguous(item if item is not None else torch.zeros_like(value))
            for item in (grad_r, grad_w, grad_k, grad_v, grad_a, grad_g)
        )
        create_graph = torch.is_grad_enabled()
        rows = int(value.shape[0]) * int(value.shape[1])
        if create_graph:
            return _canonical_mix6_higher_order_backward(
                value,
                shifted,
                tuple(mixes),
                output_grads,
            )
        if rows < NATIVE_BACKWARD_MIN_ROWS:
            with torch.enable_grad():
                references = tuple(
                    item.detach().requires_grad_(True)
                    for item in (value, shifted, *mixes)
                )
                value_ref, shifted_ref, *mix_refs = references
                outputs = _canonical_mix6(
                    value_ref,
                    shifted_ref,
                    tuple(mix_refs),
                )
                return tuple(
                    torch.autograd.grad(
                        outputs,
                        references,
                        output_grads,
                    )
                )

        inputs = tuple(_bf16x2_contiguous(item) for item in (value, shifted, *mixes))
        namespace = getattr(torch.ops, OPERATOR_NAMESPACE)
        return tuple(namespace.backward(*output_grads, *inputs))


def _run_mix6_training(
    value: torch.Tensor,
    shifted: torch.Tensor,
    *mixes: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    load_mix6_training_cuda_extension(device=value.device)
    flattened = tuple(mix.reshape(-1) for mix in mixes)
    return tuple(_Mix6Shifted.apply(value, shifted, *flattened))


def mix6_training_v1(
    value: torch.Tensor,
    shifted: torch.Tensor,
    *mixes: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Execute six validated shifted-input mixes through native CUDA autograd."""

    support = probe_mix6_training_v1(value, shifted, *mixes)
    if not support["supported"]:
        raise RuntimeError(str(support["reason"]))
    return _run_mix6_training(value, shifted, *mixes)


__all__ = [
    "IMPLEMENTATION",
    "NATIVE_BACKWARD_MIN_ROWS",
    "OPERATOR_NAMESPACE",
    "load_mix6_training_cuda_extension",
    "mix6_training_v1",
    "probe_mix6_training_v1",
]
