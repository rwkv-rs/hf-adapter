from __future__ import annotations

import json
from pathlib import Path

from rwkv7_hf.deepspeed_config import (
    deepspeed_config_basename,
    select_deepspeed_config,
)
from rwkv7_hf.kernel_policy import classify_gpu


_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "deepspeed"


def test_generic_deepspeed_profiles_preserve_historical_scale() -> None:
    for stage in (2, 3):
        generic = json.loads((_CONFIG_ROOT / f"zero{stage}.json").read_text())
        v100 = json.loads((_CONFIG_ROOT / f"zero{stage}_v100.json").read_text())
        assert generic["fp16"]["initial_scale_power"] == 16
        assert v100["fp16"]["initial_scale_power"] == 8
        assert generic["zero_optimization"]["stage"] == stage
        assert v100["zero_optimization"]["stage"] == stage


def test_deepspeed_profile_selection_is_exact_v100_only() -> None:
    v100 = classify_gpu("Tesla V100-PCIE-32GB", (7, 0))
    profiles = (
        classify_gpu("NVIDIA TITAN V", (7, 0)),
        classify_gpu("Tesla T4", (7, 5)),
        classify_gpu("NVIDIA A100-SXM4-80GB", (8, 0)),
        classify_gpu("NVIDIA GeForce RTX 4090", (8, 9)),
        classify_gpu("NVIDIA GeForce RTX 5090", (12, 0)),
    )
    assert deepspeed_config_basename(2, v100) == "zero2_v100.json"
    assert deepspeed_config_basename(3, v100) == "zero3_v100.json"
    for profile in profiles:
        assert deepspeed_config_basename(2, profile) == "zero2.json"
        assert deepspeed_config_basename(3, profile) == "zero3.json"


def test_deepspeed_explicit_override_wins_over_card_policy() -> None:
    rtx4090 = classify_gpu("NVIDIA GeForce RTX 4090", (8, 9))
    selected = select_deepspeed_config(
        _CONFIG_ROOT,
        3,
        override="zero3_v100_offload.json",
        profile=rtx4090,
    )
    assert selected.name == "zero3_v100_offload.json"
