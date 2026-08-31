from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def source_kernel_package(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    for name in tuple(sys.modules):
        if name == "rwkv7_kernels" or name.startswith("rwkv7_kernels."):
            sys.modules.pop(name)


def test_fused_prefill_decay_uses_the_clean_model_constant_exactly():
    fused_prefill = importlib.import_module("rwkv7_kernels.nvidia.fused_prefill")
    torch.manual_seed(0)
    w_raw = torch.randn(2, 3, 8, dtype=torch.float16)
    k_raw = torch.randn_like(w_raw)
    v_raw = torch.randn_like(w_raw)
    a = torch.sigmoid(torch.randn_like(w_raw))
    k_k = torch.randn(8, dtype=torch.float16)
    k_a = torch.randn(8, dtype=torch.float16)

    decay, *_ = fused_prefill.fused_prefill_state_prep(
        w_raw,
        k_raw,
        v_raw,
        a,
        k_k,
        k_a,
        num_heads=2,
        head_dim=4,
        w_transform="decay",
        force_fallback=True,
    )
    log_decay, *_ = fused_prefill.fused_prefill_state_prep(
        w_raw,
        k_raw,
        v_raw,
        a,
        k_k,
        k_a,
        num_heads=2,
        head_dim=4,
        w_transform="log_decay",
        force_fallback=True,
    )

    decay_base = math.exp(-0.5)
    expected_log = -decay_base * torch.sigmoid(w_raw.float())
    assert fused_prefill._RWKV7_DECAY_BASE == decay_base
    assert torch.equal(log_decay, expected_log)
    assert torch.equal(decay, torch.exp(expected_log))
