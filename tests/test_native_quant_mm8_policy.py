from __future__ import annotations

from rwkv7_hf.native_quant_mm8 import _mm8_batched_dot_profile_supported


def test_mm8_batched_dot_exact_architecture_policy() -> None:
    assert _mm8_batched_dot_profile_supported(
        11, is_hip=True, architecture="gfx1100"
    )
    assert _mm8_batched_dot_profile_supported(
        11, is_hip=True, architecture="gfx1100:sramecc-:xnack-"
    )
    assert not _mm8_batched_dot_profile_supported(
        11, is_hip=True, architecture="gfx1101"
    )
    assert not _mm8_batched_dot_profile_supported(
        9, is_hip=True, architecture="gfx942"
    )
    assert _mm8_batched_dot_profile_supported(
        12, is_hip=False, architecture=None
    )
    assert not _mm8_batched_dot_profile_supported(
        11, is_hip=False, architecture=None
    )
