#!/usr/bin/env python3
from __future__ import annotations

import torch

from rwkv7_hf import native_jit
from rwkv7_hf.native import _eager_recurrent_state
from rwkv7_hf.native_wkv_fp16 import (
    native_fp16_recurrent_should_use,
    native_fp16_sequence,
)


def test_fp16_recurrent_policy_is_narrow() -> None:
    assert native_fp16_recurrent_should_use(
        state_dtype=torch.float16,
        input_dtype=torch.float16,
        head_dim=64,
    )
    assert not native_fp16_recurrent_should_use(
        state_dtype=torch.float32,
        input_dtype=torch.float16,
        head_dim=64,
    )
    assert not native_fp16_recurrent_should_use(
        state_dtype=torch.float16,
        input_dtype=torch.bfloat16,
        head_dim=64,
    )
    assert not native_fp16_recurrent_should_use(
        state_dtype=torch.float16,
        input_dtype=torch.float16,
        head_dim=32,
    )


def test_eager_fallback_promotes_exact_card_fp16_cache_to_fp32() -> None:
    state = torch.randn(2, 3, 4, 4, dtype=torch.float16)
    promoted = _eager_recurrent_state(state)

    assert promoted.dtype == torch.float32
    torch.testing.assert_close(promoted, state.float(), rtol=0.0, atol=0.0)
    fp32 = state.float()
    assert _eager_recurrent_state(fp32) is fp32


def test_prefill_fp16_recurrent_is_explicit_and_shape_gated(monkeypatch) -> None:
    # Do not inherit an exact-card default (RTX 5090 enables this path) while
    # testing the generic explicit opt-in and shape gates.
    monkeypatch.setattr(native_jit, "_kernel_policy", lambda: None)
    state = torch.zeros(1, 2, 64, 64, dtype=torch.float16)
    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_FP16_RECURRENT", raising=False)
    assert not native_jit._native_prefill_fp16_recurrent_enabled(state)

    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_FP16_RECURRENT", "1")
    monkeypatch.setattr(native_jit, "native_fp16_sequence", object())
    assert native_jit._native_prefill_fp16_recurrent_enabled(state)
    assert not native_jit._native_prefill_fp16_recurrent_enabled(state.float())
    assert not native_jit._native_prefill_fp16_recurrent_enabled(
        torch.zeros(1, 2, 32, 32, dtype=torch.float16)
    )


def test_prefill_state_allocation_preserves_default_fp32(monkeypatch) -> None:
    packs = [(0, 2, 64, None, None, None, None, torch.ones(128))]
    state, _, _ = native_jit._init_batched_from_packs(
        packs,
        3,
        torch.device("cpu"),
        torch.float16,
    )
    assert state[0].shape == (3, 2, 64, 64)
    assert state[0].dtype == torch.float32

    state, _, _ = native_jit._init_batched_from_packs(
        packs,
        3,
        torch.device("cpu"),
        torch.float16,
        state_dtype=torch.float16,
    )
    assert state[0].dtype == torch.float16


def test_fp16_sequence_rejects_non_rank_four_inputs_before_build() -> None:
    tensor = torch.zeros(1, 64, dtype=torch.float16)
    try:
        native_fp16_sequence(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            torch.zeros(1, dtype=torch.int32),
        )
    except ValueError as exc:
        assert "rank four" in str(exc)
    else:
        raise AssertionError("rank validation must run before extension loading")
