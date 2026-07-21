#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwkv7_hf.deepspeed_config import (
    deepspeed_config_basename,
    select_deepspeed_config,
)
from rwkv7_hf.kernel_policy import classify_gpu


REQUIRED_TOP_LEVEL = {
    "train_micro_batch_size_per_gpu",
    "gradient_accumulation_steps",
    "fp16",
    "bf16",
    "zero_optimization",
}


def validate(path: Path, expected_stage: int, *, initial_scale_power: int) -> None:
    cfg = json.loads(path.read_text())
    missing = REQUIRED_TOP_LEVEL - set(cfg)
    assert not missing, f"{path} missing keys: {sorted(missing)}"
    zero = cfg["zero_optimization"]
    assert int(zero["stage"]) == expected_stage, (path, zero)
    assert cfg["train_micro_batch_size_per_gpu"] == "auto", cfg
    assert cfg["gradient_accumulation_steps"] == "auto", cfg
    assert cfg["fp16"]["enabled"] == "auto", cfg
    assert int(cfg["fp16"]["initial_scale_power"]) == initial_scale_power, cfg
    assert cfg["bf16"]["enabled"] == "auto", cfg
    assert zero.get("contiguous_gradients") is True, zero
    if expected_stage == 2:
        assert zero.get("reduce_scatter") is True, zero
        assert zero.get("allgather_partitions") is True, zero
    if expected_stage == 3:
        assert zero.get("stage3_gather_16bit_weights_on_model_save") is True, zero
        for key in ("reduce_bucket_size", "stage3_prefetch_bucket_size", "stage3_param_persistence_threshold"):
            assert zero.get(key) == "auto", (key, zero)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="configs/deepspeed")
    args = ap.parse_args()
    root = Path(args.config_dir)
    # Generic presets preserve the historical cross-card default. V100's
    # recurrent FP16 path uses the separately validated lower initial scale.
    validate(root / "zero2.json", 2, initial_scale_power=16)
    validate(root / "zero3.json", 3, initial_scale_power=16)
    validate(root / "zero2_v100.json", 2, initial_scale_power=8)
    validate(root / "zero3_v100.json", 3, initial_scale_power=8)

    v100 = classify_gpu("Tesla V100-PCIE-32GB", (7, 0))
    titan_v = classify_gpu("NVIDIA TITAN V", (7, 0))
    t4 = classify_gpu("Tesla T4", (7, 5))
    rtx4090 = classify_gpu("NVIDIA GeForce RTX 4090", (8, 9))
    assert deepspeed_config_basename(2, v100) == "zero2_v100.json"
    assert deepspeed_config_basename(3, v100) == "zero3_v100.json"
    for profile in (titan_v, t4, rtx4090):
        assert deepspeed_config_basename(2, profile) == "zero2.json"
        assert deepspeed_config_basename(3, profile) == "zero3.json"
    assert select_deepspeed_config(root, 2, profile=v100).name == "zero2_v100.json"
    assert select_deepspeed_config(root, 3, profile=rtx4090).name == "zero3.json"
    assert select_deepspeed_config(
        root,
        3,
        override="zero3_v100_offload.json",
        profile=rtx4090,
    ).name == "zero3_v100_offload.json"
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
