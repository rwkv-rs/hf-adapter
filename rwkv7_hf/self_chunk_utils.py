"""Small dependency-free utility surface for vendored inference kernels."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

IS_AMD = bool(getattr(torch.version, "hip", None))
IS_GATHER_SUPPORTED = False
USE_CUDA_GRAPH = False
autotune_cache_kwargs: dict = {}


def check_shared_mem(architecture: str = "", device_index: int | None = None) -> bool:
    """Report the wider-tile capability used by the vendored forward kernel.

    The original FLA helper distinguishes devices by their available shared
    memory. Keep that decision local and dependency-free: CUDA sm80+ devices
    use the 32-wide capability tier, while sm90+ can use the 64-wide tier.
    Older CUDA and ROCm devices retain the conservative
    16-wide fallback.
    """

    if IS_AMD or not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(device_index)
    except Exception:
        return False
    name = str(architecture).strip().lower()
    if name == "hopper":
        return int(major) >= 9
    if name == "ampere":
        return int(major) >= 8
    return False


def prepare_chunk_indices(cu_seqlens, chunk_size, *args, **kwargs):
    if cu_seqlens is None:
        return None
    raise NotImplementedError("self chunk prefill currently supports equal-length batches")


def prepare_chunk_offsets(cu_seqlens, chunk_size, *args, **kwargs):
    if cu_seqlens is None:
        return None
    raise NotImplementedError("self chunk prefill currently supports equal-length batches")


@triton.jit
def exp2(x):
    return tl.exp2(x)


def gather(*_args, **_kwargs):  # dead branch when IS_GATHER_SUPPORTED=False
    raise NotImplementedError
