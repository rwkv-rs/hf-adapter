#!/usr/bin/env python3
from __future__ import annotations

import os

from rwkv7_hf.kernel_policy import (
    ADAPTATION_RULES,
    adaptation_rule_for_profile,
    classify_gpu,
    env_flag,
    env_int,
    policy_for_profile,
)


def test_gpu_family_classification() -> None:
    cases = [
        ("Tesla P100-PCIE-16GB", (6, 0), "pascal"),
        ("Tesla V100-PCIE-32GB", (7, 0), "volta"),
        ("NVIDIA GeForce RTX 4090", (8, 9), "ada"),
        ("NVIDIA H100 SXM", (9, 0), "hopper"),
        ("NVIDIA GeForce RTX 5070 Laptop GPU", (12, 0), "blackwell"),
        ("AMD Instinct MI300X", None, "amd_hip"),
    ]
    for name, capability, family in cases:
        profile = classify_gpu(name, capability, is_hip=name.startswith("AMD"))
        assert profile.family == family, (name, profile)


def test_policy_defaults_are_conservative() -> None:
    pascal = policy_for_profile(classify_gpu("Tesla P100", (6, 0)))
    assert not pascal.fused_output
    assert not pascal.fused_recurrent_output

    v100 = policy_for_profile(classify_gpu("Tesla V100-PCIE-32GB", (7, 0)))
    assert v100.fused_output
    assert v100.fused_recurrent_output
    assert not v100.fused_projection
    assert not v100.fused_output_project

    ada = policy_for_profile(classify_gpu("NVIDIA GeForce RTX 4090", (8, 9)))
    assert ada.fused_output
    assert ada.fused_recurrent_output
    assert not ada.fused_projection

    blackwell = policy_for_profile(classify_gpu("NVIDIA GeForce RTX 5070 Laptop GPU", (12, 0)))
    assert blackwell.fused_output
    assert blackwell.fused_recurrent_output
    assert not blackwell.fused_projection


def test_every_policy_family_has_an_adaptation_rule() -> None:
    cases = [
        classify_gpu(None, None),
        classify_gpu("old cuda", (5, 2)),
        classify_gpu("Tesla P100", (6, 0)),
        classify_gpu("Tesla V100-PCIE-32GB", (7, 0)),
        classify_gpu("NVIDIA T4", (7, 5)),
        classify_gpu("NVIDIA A100-SXM4-80GB", (8, 0)),
        classify_gpu("NVIDIA GeForce RTX 4090", (8, 9)),
        classify_gpu("NVIDIA H100 SXM", (9, 0)),
        classify_gpu("NVIDIA GeForce RTX 5070 Laptop GPU", (12, 0)),
        classify_gpu("AMD Instinct MI300X", None, is_hip=True),
    ]
    for profile in cases:
        rule = adaptation_rule_for_profile(profile)
        assert rule.family == profile.family, (profile, rule)
        assert rule.required_functional
        assert rule.required_benchmarks
        assert rule.promotion_rule

    # The registry is intentionally broader than the live test cases because it
    # also documents unvalidated fallback families.
    for family in ("unknown_cuda", "legacy_cuda", "pascal", "volta", "ada", "blackwell", "amd_hip"):
        assert family in ADAPTATION_RULES


def test_env_helpers_override_defaults() -> None:
    old = os.environ.get("RWKV7_TEST_FLAG")
    old_int = os.environ.get("RWKV7_TEST_INT")
    try:
        os.environ.pop("RWKV7_TEST_FLAG", None)
        assert env_flag("RWKV7_TEST_FLAG", True)
        assert not env_flag("RWKV7_TEST_FLAG", False)
        os.environ["RWKV7_TEST_FLAG"] = "0"
        assert not env_flag("RWKV7_TEST_FLAG", True)
        os.environ["RWKV7_TEST_FLAG"] = "1"
        assert env_flag("RWKV7_TEST_FLAG", False)
        os.environ["RWKV7_TEST_FLAG"] = "TRUE"
        assert env_flag("RWKV7_TEST_FLAG", False)

        os.environ["RWKV7_TEST_INT"] = "999"
        assert env_int("RWKV7_TEST_INT", 16, lower=1, upper=128) == 128
        os.environ["RWKV7_TEST_INT"] = "bad"
        assert env_int("RWKV7_TEST_INT", 16, lower=1, upper=128) == 16
    finally:
        if old is None:
            os.environ.pop("RWKV7_TEST_FLAG", None)
        else:
            os.environ["RWKV7_TEST_FLAG"] = old
        if old_int is None:
            os.environ.pop("RWKV7_TEST_INT", None)
        else:
            os.environ["RWKV7_TEST_INT"] = old_int


def main() -> int:
    test_gpu_family_classification()
    test_policy_defaults_are_conservative()
    test_every_policy_family_has_an_adaptation_rule()
    test_env_helpers_override_defaults()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
