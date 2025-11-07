# coding=utf-8
"""Recurrent MLX state-cache primitives shared by model and serving layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .mlx_bridge import require_mlx


def _mx():
    return require_mlx()


def _as_list(values: Iterable[int] | Any) -> list[int]:
    if isinstance(values, list):
        return [int(v) for v in values]
    if isinstance(values, tuple):
        return [int(v) for v in values]
    try:
        return [int(v) for v in values.tolist()]
    except Exception:
        return [int(v) for v in values]


@dataclass
class MLXRWKV7State:
    """MLX recurrent state cache for RWKV-7 decode.

    Layout mirrors the native PyTorch cache:

    - ``recurrent_state[layer]``: ``[B, H, N, N]`` fp32 WKV state
    - ``attn_x_prev[layer]``: ``[B, hidden]`` previous attention input
    - ``ffn_x_prev[layer]``: ``[B, hidden]`` previous FFN input
    - ``v_first``: ``[B, hidden]`` first-layer value stream

    ``select_batch`` / ``reorder_cache`` give the MLX path the same dynamic
    batching seam used by HF serving caches.
    """

    recurrent_state: list[Any]
    attn_x_prev: list[Any]
    ffn_x_prev: list[Any]
    v_first: Any
    seen_tokens: int = 0

    @property
    def batch_size(self) -> int:
        return int(self.v_first.shape[0])

    @property
    def num_layers(self) -> int:
        return len(self.recurrent_state)

    def clone(self) -> "MLXRWKV7State":
        mx = _mx()
        cloned = MLXRWKV7State(
            [mx.array(x) for x in self.recurrent_state],
            [mx.array(x) for x in self.attn_x_prev],
            [mx.array(x) for x in self.ffn_x_prev],
            mx.array(self.v_first),
            seen_tokens=int(self.seen_tokens),
        )
        mx.eval(cloned.v_first, *cloned.recurrent_state, *cloned.attn_x_prev, *cloned.ffn_x_prev)
        return cloned

    def select_batch(self, indices: Iterable[int] | Any) -> "MLXRWKV7State":
        mx = _mx()
        idx = mx.array(_as_list(indices), dtype=mx.int32)
        selected = MLXRWKV7State(
            [mx.take(x, idx, axis=0) for x in self.recurrent_state],
            [mx.take(x, idx, axis=0) for x in self.attn_x_prev],
            [mx.take(x, idx, axis=0) for x in self.ffn_x_prev],
            mx.take(self.v_first, idx, axis=0),
            seen_tokens=int(self.seen_tokens),
        )
        mx.eval(selected.v_first, *selected.recurrent_state, *selected.attn_x_prev, *selected.ffn_x_prev)
        return selected

    def reorder_cache(self, indices: Iterable[int] | Any) -> "MLXRWKV7State":
        return self.select_batch(indices)

    def compact(self, indices: Iterable[int] | Any) -> "MLXRWKV7State":
        return self.select_batch(indices)

    def detach(self) -> "MLXRWKV7State":
        # MLX arrays are eager/lazy value arrays, not torch autograd tensors; a
        # clone gives callers an explicit cache boundary.
        return self.clone()

__all__ = ["MLXRWKV7State"]
