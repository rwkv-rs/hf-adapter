from __future__ import annotations

from rwkv7_hf.native_quant_mm4 import (
    _mm4_batched_dot_device_supported,
    _mm4_batched_dot_profile_supported,
)


def test_mm4_batched_dot_exact_device_policy() -> None:
    assert _mm4_batched_dot_device_supported(8, 6, "NVIDIA GeForce RTX 3090")
    assert _mm4_batched_dot_device_supported(8, 9, "NVIDIA GeForce RTX 4090")
    assert _mm4_batched_dot_device_supported(12, 0, "NVIDIA GeForce RTX 5090")
    assert not _mm4_batched_dot_device_supported(8, 9, "NVIDIA GeForce RTX 4070")
    assert not _mm4_batched_dot_device_supported(8, 9, "NVIDIA GeForce RTX 40900")
    assert not _mm4_batched_dot_device_supported(8, 9, "NVIDIA GeForce RTX 4090 Laptop GPU")
    assert not _mm4_batched_dot_device_supported(8, 0, "NVIDIA A100")


def test_mm4_batched_dot_hip_policy_is_exact_architecture_only() -> None:
    assert _mm4_batched_dot_profile_supported(
        11,
        0,
        "AMD Radeon Graphics",
        is_hip=True,
        architecture="gfx1100:sramecc-:xnack-",
    )
    assert not _mm4_batched_dot_profile_supported(
        11,
        0,
        "AMD Radeon Graphics",
        is_hip=True,
        architecture="gfx1101",
    )
    # ROCm capability numbers must never inherit NVIDIA sm_120 policy.
    assert not _mm4_batched_dot_profile_supported(
        12,
        0,
        "AMD Radeon Graphics",
        is_hip=True,
        architecture="gfx1200",
    )
