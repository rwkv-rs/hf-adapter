# coding=utf-8
"""Native RWKV-7 recurrent update selection and eager fallback math."""
from __future__ import annotations

import torch


_OWNED_NAMES = {'_native_graph_fused_recurrent_enabled', '_recurrent_update_batched', '_recurrent_update_unbatched'} | {"bind_runtime"}
_RUNTIME_NAMES = ("_kernel_policy", "env_flag", "fused_recurrent_update", "fused_recurrent_update_available")


def bind_runtime(runtime: dict[str, object]) -> None:
    for name in _RUNTIME_NAMES:
        if name in runtime and name not in _OWNED_NAMES:
            globals()[name] = runtime[name]


def _native_graph_fused_recurrent_enabled() -> bool:
    """Runtime switch for the experimental native-graph recurrent Triton path."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT", bool(getattr(policy, "fused_recurrent", False))):
        return False
    if fused_recurrent_update is None or fused_recurrent_update_available is None:
        return False
    try:
        return bool(fused_recurrent_update_available())
    except Exception:
        return False

def _recurrent_update_unbatched(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    H: int,
    N: int,
):
    if _native_graph_fused_recurrent_enabled():
        out, new_state = fused_recurrent_update(
            r.view(1, H, N),
            w.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            kk.view(1, H, N),
            a.view(1, H, N),
            state.view(1, H, N, N),
            block_n=N,
        )
        return out.reshape(H * N), new_state.reshape(H, N, N)
    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    new_state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = new_state.to(r.dtype) @ r.view(H, N, 1)
    return out.view(H * N), new_state

def _recurrent_update_batched(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    B: int,
    H: int,
    N: int,
):
    if _native_graph_fused_recurrent_enabled():
        out, new_state = fused_recurrent_update(
            r.view(B, H, N),
            w.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            kk.view(B, H, N),
            a.view(B, H, N),
            state,
            block_n=N,
        )
        return out.reshape(B, H * N), new_state
    vk = v.view(B, H, N, 1) @ k.view(B, H, 1, N)
    ab = (-kk).view(B, H, N, 1) @ (kk * a).view(B, H, 1, N)
    new_state = state * w.view(B, H, 1, N) + state @ ab.float() + vk.float()
    out = new_state.to(r.dtype) @ r.view(B, H, N, 1)
    return out.view(B, H * N), new_state

__all__ = ['_native_graph_fused_recurrent_enabled', '_recurrent_update_unbatched', '_recurrent_update_batched']
