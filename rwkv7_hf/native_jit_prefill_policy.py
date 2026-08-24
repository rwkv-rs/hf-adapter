# coding=utf-8
"""Pure shape and tiling policy helpers for native JIT prefill routes."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _parse_model_shapes(raw: str, *, env_name: str) -> set[tuple[int, int, int, int]]:
    shapes: set[tuple[int, int, int, int]] = set()
    try:
        for item in raw.replace(",", " ").split():
            values = tuple(int(value) for value in item.lower().split("x"))
            if len(values) != 4 or any(value <= 0 for value in values):
                raise ValueError
            shapes.add(values)
    except ValueError as exc:
        raise ValueError(f"{env_name} must contain HxLxBxT tuples") from exc
    return shapes


def self_chunk_shape_eligible(
    *,
    policy: Any,
    tokens: int,
    head_dim: int,
    batch_size: int | None,
    hidden_size: int | None,
    num_layers: int | None,
    min_tokens: int,
    raw_model_shapes: str | None,
) -> bool:
    env_name = "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MODEL_SHAPES"
    if raw_model_shapes is None:
        model_shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(policy, "prefill_self_chunk_model_shapes", ())
            if len(shape) == 4
        }
    else:
        model_shapes = _parse_model_shapes(raw_model_shapes, env_name=env_name)
    exact_model_shape = (
        (int(hidden_size), int(num_layers), int(batch_size), int(tokens))
        if None not in (hidden_size, num_layers, batch_size)
        else None
    )
    if bool(getattr(policy, "prefill_self_chunk_model_shapes_only", False)):
        if exact_model_shape not in model_shapes:
            return False
    return not (
        (int(tokens) < int(min_tokens) and exact_model_shape not in model_shapes)
        or int(tokens) % 16
        or int(head_dim) != 64
    )


def self_chunk_size(
    *,
    policy: Any,
    batch_size: int,
    tokens: int | None,
    env_int_fn: Callable[..., int],
) -> int:
    default = int(getattr(policy, "prefill_self_chunk_size", 16))
    if tokens is not None:
        for policy_batch, policy_tokens, policy_size in getattr(
            policy,
            "prefill_self_chunk_shape_sizes",
            (),
        ):
            if (int(batch_size), int(tokens)) == (int(policy_batch), int(policy_tokens)):
                default = int(policy_size)
                break
    chunk_size = env_int_fn(
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE",
        default,
        lower=16,
        upper=64,
    )
    if chunk_size not in {16, 32, 64}:
        raise ValueError("RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE must be 16, 32, or 64")
    return chunk_size


def self_chunk_h_tiles(
    *,
    policy: Any,
    batch_size: int,
    tokens: int,
) -> tuple[int, int] | None:
    for policy_batch, policy_tokens, policy_bv, policy_bc in getattr(
        policy,
        "prefill_self_chunk_h_tile_shapes",
        (),
    ):
        if (int(batch_size), int(tokens)) == (int(policy_batch), int(policy_tokens)):
            return int(policy_bv), int(policy_bc)
    return None


def model_shape_selected(
    *,
    policy: Any,
    env_name: str,
    policy_name: str,
    raw: str | None,
    batch_size: int | None,
    prompt_tokens: int | None,
    hidden_size: int | None,
    num_layers: int | None,
) -> bool:
    """Select an exact shape or a bounded dynamic model profile.

    Exact shape environment overrides keep their historical semantics: when
    ``raw`` is provided, only those HxLxBxT tuples are considered.  Otherwise
    a policy may complement ``*_model_shapes`` with ``*_model_profiles``.
    Profiles are HxLxBmaxxTmaxxRowsMax and intentionally cover only shape-safe
    fusions; exact-card graph and reduced-precision routes keep separate exact
    allowlists.
    """

    if raw is None:
        shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(policy, policy_name, ())
            if len(shape) == 4
        }
        profile_name = (
            policy_name[: -len("_shapes")] + "_profiles"
            if policy_name.endswith("_shapes")
            else policy_name + "_profiles"
        )
        profiles = {
            tuple(int(value) for value in profile)
            for profile in getattr(policy, profile_name, ())
            if len(profile) == 5
        }
    else:
        shapes = _parse_model_shapes(raw, env_name=env_name)
        profiles = set()
    if not shapes and not profiles:
        return True
    if None in (batch_size, prompt_tokens, hidden_size, num_layers):
        return False
    target = (
        int(hidden_size),
        int(num_layers),
        int(batch_size),
        int(prompt_tokens),
    )
    if target in shapes:
        return True
    hidden, layers, batch, tokens = target
    return any(
        hidden == profile_hidden
        and layers == profile_layers
        and 1 <= batch <= max_batch
        and 2 <= tokens <= max_tokens
        and batch * tokens <= max_total_tokens
        for (
            profile_hidden,
            profile_layers,
            max_batch,
            max_tokens,
            max_total_tokens,
        ) in profiles
    )


def policy_model_shape_selected(
    *,
    policy: Any,
    policy_name: str,
    batch_size: int | None,
    prompt_tokens: int | None,
    hidden_size: int | None,
    num_layers: int | None,
) -> bool:
    if None in (batch_size, prompt_tokens, hidden_size, num_layers):
        return False
    target = (
        int(hidden_size),
        int(num_layers),
        int(batch_size),
        int(prompt_tokens),
    )
    return target in {
        tuple(int(value) for value in shape)
        for shape in getattr(policy, policy_name, ())
        if len(shape) == 4
    }


__all__ = [
    "model_shape_selected",
    "policy_model_shape_selected",
    "self_chunk_h_tiles",
    "self_chunk_shape_eligible",
    "self_chunk_size",
]
