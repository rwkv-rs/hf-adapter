from __future__ import annotations

import torch

from rwkv7_hf.training_precision import (
    merged_logits_pass,
    peft_trainer_precision_kwargs,
)


def test_precast_fp16_peft_uses_unscaled_trainer_mode() -> None:
    assert peft_trainer_precision_kwargs("fp16") == {
        "fp16": False,
        "bf16": False,
    }
    assert peft_trainer_precision_kwargs(torch.float16) == {
        "fp16": False,
        "bf16": False,
    }


def test_bf16_peft_keeps_trainer_bf16_autocast() -> None:
    assert peft_trainer_precision_kwargs("bf16") == {
        "fp16": False,
        "bf16": True,
    }
    assert peft_trainer_precision_kwargs(torch.float32) == {
        "fp16": False,
        "bf16": False,
    }


def test_fp16_merge_gate_is_bounded_and_requires_exact_top1() -> None:
    metrics = {
        "max_abs": 0.25,
        "mean_abs": 0.005,
        "cosine": 0.99999,
        "top1_match_rate": 1.0,
    }
    kwargs = {
        "dtype": "fp16",
        "strict_max_abs": 1e-4,
        "fp16_max_abs": 0.5,
        "fp16_max_mean_abs": 0.05,
        "fp16_min_cosine": 0.9999,
    }
    assert merged_logits_pass(metrics, **kwargs)
    assert not merged_logits_pass({**metrics, "max_abs": 0.75}, **kwargs)
    assert not merged_logits_pass({**metrics, "top1_match_rate": 0.99}, **kwargs)
    assert not merged_logits_pass(metrics, **{**kwargs, "dtype": "fp32"})
