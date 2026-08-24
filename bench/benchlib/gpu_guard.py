"""Fail-closed GPU product matching shared by acceptance runners."""

from __future__ import annotations

from rwkv7_hf.kernel_policy import is_rtx_model_name


def matches_gpu_product(
    detected_name: str,
    *,
    rtx_model: str | None = None,
    exact_name: str | None = None,
) -> bool:
    if (rtx_model is None) == (exact_name is None):
        raise ValueError("provide exactly one of rtx_model or exact_name")
    if exact_name is not None:
        return detected_name.strip() == exact_name.strip()
    return is_rtx_model_name(detected_name, str(rtx_model))
