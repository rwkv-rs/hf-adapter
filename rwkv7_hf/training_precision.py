"""Precision helpers for PEFT training through HF/TRL trainers.

Native RWKV checkpoints are normally loaded directly in their compute dtype,
while PEFT keeps LoRA parameters in FP32.  Enabling Trainer's FP16 AMP mode on
top of an already-FP16 base adds a GradScaler with an initial scale of 65536.
That scale can overflow the recurrent FP16 activation-gradient path before the
dynamic scaler has backed off, causing short Trainer/TRL jobs to skip every
optimizer step.  For this common PEFT setup, leave Trainer AMP disabled: base
GEMMs remain FP16 because the base weights and activations are FP16, gradients
are unscaled, and the trainable LoRA parameters remain FP32.

BF16 has no GradScaler and can continue to use Trainer's normal BF16 mode.
Full-parameter FP16 training should use the train_temp/DeepSpeed recipe with
master weights rather than this PEFT-specific helper.
"""
from __future__ import annotations

from typing import Any

import torch


def peft_trainer_precision_kwargs(dtype: Any) -> dict[str, bool]:
    """Return HF TrainingArguments precision flags for a pre-cast PEFT base."""

    if isinstance(dtype, str):
        normalized = dtype.strip().lower()
        is_fp16 = normalized in {"fp16", "float16", "half"}
        is_bf16 = normalized in {"bf16", "bfloat16"}
    else:
        is_fp16 = dtype is torch.float16
        is_bf16 = dtype is torch.bfloat16
    # ``is_fp16`` is intentionally evaluated for clarity/documentation even
    # though both FP16 and FP32 use Trainer's unscaled mode here.
    return {"fp16": False, "bf16": bool(is_bf16 and not is_fp16)}


def merged_logits_pass(
    metrics: dict[str, float],
    *,
    dtype: str,
    strict_max_abs: float,
    fp16_max_abs: float,
    fp16_max_mean_abs: float,
    fp16_min_cosine: float,
) -> bool:
    """Validate PEFT merge output without pretending FP16 merge is lossless."""

    if dtype != "fp16":
        return metrics["max_abs"] <= strict_max_abs
    return bool(
        metrics["max_abs"] <= fp16_max_abs
        and metrics["mean_abs"] <= fp16_max_mean_abs
        and metrics["cosine"] >= fp16_min_cosine
        and metrics["top1_match_rate"] == 1.0
    )
