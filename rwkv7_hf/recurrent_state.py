# coding=utf-8
"""Physical recurrent-state layout and decode-route contracts.

RWKV-7's recurrent matrix is square for the production head size, so the
legacy ``[V, K]`` and experimental ``[K, V]`` layouts have identical tensor
shapes.  Callers therefore must carry an explicit layout tag instead of
inferring semantics from shape or stride.

This module intentionally owns contracts only.  Kernel availability and
exact-card promotion remain in the runtime/kernel policy layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RecurrentStateLayout(str, Enum):
    """Physical ordering of the last two recurrent-state dimensions."""

    VK_V1 = "vk_v1"
    KV_V2 = "kv_v2"


_LAYOUT_ALIASES = {
    "vk": RecurrentStateLayout.VK_V1,
    "vk_v1": RecurrentStateLayout.VK_V1,
    "v1": RecurrentStateLayout.VK_V1,
    "kv": RecurrentStateLayout.KV_V2,
    "kv_v2": RecurrentStateLayout.KV_V2,
    "v2": RecurrentStateLayout.KV_V2,
}


def normalize_recurrent_state_layout(
    value: RecurrentStateLayout | str | None,
    *,
    default: RecurrentStateLayout = RecurrentStateLayout.VK_V1,
) -> RecurrentStateLayout:
    """Normalize a layout name while failing closed on unknown ABI values."""

    if value is None:
        return default
    if isinstance(value, RecurrentStateLayout):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"", "auto", "default"}:
        return default
    try:
        return _LAYOUT_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(layout.value for layout in RecurrentStateLayout)
        raise ValueError(
            f"unknown RWKV-7 recurrent-state layout {value!r}; expected {choices}"
        ) from exc


def recurrent_state_layout_of(
    value: Any,
    *,
    default: RecurrentStateLayout = RecurrentStateLayout.VK_V1,
) -> RecurrentStateLayout:
    """Read a cache/layout tag without depending on a concrete cache class."""

    tagged = getattr(value, "_rwkv7_state_layout", None)
    if tagged is None:
        tagged = getattr(value, "state_layout", None)
    return normalize_recurrent_state_layout(tagged, default=default)


def convert_recurrent_state_tensor(
    state: Any,
    source: RecurrentStateLayout | str,
    target: RecurrentStateLayout | str,
):
    """Convert one state tensor between ``vk_v1`` and ``kv_v2``.

    Both supported layouts are transposes of one another.  A contiguous output
    is required because the persistent CUDA kernels consume a physical layout,
    not merely a transposed view.
    """

    source_layout = normalize_recurrent_state_layout(source)
    target_layout = normalize_recurrent_state_layout(target)
    if source_layout == target_layout:
        return state
    if not hasattr(state, "dim") or int(state.dim()) < 2:
        raise ValueError("RWKV-7 recurrent state must have at least two dimensions")
    return state.transpose(-1, -2).contiguous()


def convert_recurrent_state_list(
    states: list[Any] | tuple[Any, ...] | None,
    source: RecurrentStateLayout | str,
    target: RecurrentStateLayout | str,
):
    """Convert a per-layer state collection while retaining list semantics."""

    if states is None:
        return None
    source_layout = normalize_recurrent_state_layout(source)
    target_layout = normalize_recurrent_state_layout(target)
    if source_layout == target_layout:
        return list(states)
    return [
        convert_recurrent_state_tensor(state, source_layout, target_layout)
        for state in states
    ]


@dataclass(frozen=True)
class RecurrentDecodeRoute:
    """Resolved persistent-state ABI for one decode runner/cache."""

    state_layout: RecurrentStateLayout
    state_dtype: str
    implementation: str
    source: str

    def signature(self) -> tuple[str, str, str, str]:
        return (
            self.state_layout.value,
            self.state_dtype,
            self.implementation,
            self.source,
        )


def resolve_recurrent_decode_route(
    *,
    state_dtype: Any,
    requested_layout: RecurrentStateLayout | str | None = None,
    kv_v2_kernel_available: bool = False,
    kv_v2_policy_selected: bool = False,
) -> RecurrentDecodeRoute:
    """Resolve layout once when a fixed-batch decode runner is created.

    The initial contract deliberately fails closed to ``vk_v1``.  ``kv_v2``
    can only be selected by an explicit request or an evidence-gated policy,
    and only after its kernel is present.
    """

    explicit = requested_layout is not None and str(
        requested_layout
    ).strip().lower() not in {
        "",
        "auto",
        "default",
    }
    if explicit:
        layout = normalize_recurrent_state_layout(requested_layout)
        source = "explicit"
    elif kv_v2_policy_selected:
        layout = RecurrentStateLayout.KV_V2
        source = "policy"
    else:
        layout = RecurrentStateLayout.VK_V1
        source = "conservative_fallback"

    if layout == RecurrentStateLayout.KV_V2 and not kv_v2_kernel_available:
        raise RuntimeError(
            "kv_v2 recurrent-state layout was selected, but no kv_v2 decode "
            "kernel is available"
        )
    implementation = (
        "kv_v2" if layout == RecurrentStateLayout.KV_V2 else "existing_vk_v1"
    )
    return RecurrentDecodeRoute(
        state_layout=layout,
        state_dtype=str(state_dtype),
        implementation=implementation,
        source=source,
    )


__all__ = [
    "RecurrentDecodeRoute",
    "RecurrentStateLayout",
    "convert_recurrent_state_list",
    "convert_recurrent_state_tensor",
    "normalize_recurrent_state_layout",
    "recurrent_state_layout_of",
    "resolve_recurrent_decode_route",
]
