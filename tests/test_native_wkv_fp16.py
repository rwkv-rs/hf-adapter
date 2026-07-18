#!/usr/bin/env python3
from __future__ import annotations

import torch

from rwkv7_hf.native_wkv_fp16 import native_fp16_recurrent_should_use


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
