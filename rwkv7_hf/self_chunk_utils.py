"""Small dependency-free utility surface for vendored inference kernels."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

IS_AMD = bool(getattr(torch.version, "hip", None))
IS_GATHER_SUPPORTED = False
USE_CUDA_GRAPH = False
autotune_cache_kwargs: dict = {}


def check_shared_mem(*_args, **_kwargs) -> bool:
    # The conservative 16/32-wide tiles fit every supported CUDA generation.
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
