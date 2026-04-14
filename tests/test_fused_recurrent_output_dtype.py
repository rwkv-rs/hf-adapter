from __future__ import annotations

import torch

from rwkv7_hf.fused_recurrent_update import (
    fused_recurrent_output_state_dtype_supported,
)


def test_fused_recurrent_output_accepts_fp32_and_fp16_state() -> None:
    assert fused_recurrent_output_state_dtype_supported(torch.float32)
    assert fused_recurrent_output_state_dtype_supported(torch.float16)
    assert not fused_recurrent_output_state_dtype_supported(torch.bfloat16)
