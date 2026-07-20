#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_TOP_LEVEL = {
    "train_micro_batch_size_per_gpu",
    "gradient_accumulation_steps",
    "fp16",
    "bf16",
    "zero_optimization",
}


def validate(path: Path, expected_stage: int) -> None:
    cfg = json.loads(path.read_text())
    missing = REQUIRED_TOP_LEVEL - set(cfg)
    assert not missing, f"{path} missing keys: {sorted(missing)}"
    zero = cfg["zero_optimization"]
    assert int(zero["stage"]) == expected_stage, (path, zero)
    assert cfg["train_micro_batch_size_per_gpu"] == "auto", cfg
    assert cfg["gradient_accumulation_steps"] == "auto", cfg
    assert cfg["fp16"]["enabled"] == "auto", cfg
    # Native RWKV's recurrent FP16 activation-gradient path can overflow the
    # generic 2**16 initial loss scale before a short ZeRO job gets a useful
    # optimizer step. Start dynamic scaling at the validated 2**8 value; it
    # still backs off automatically if a larger checkpoint needs it.
    assert int(cfg["fp16"]["initial_scale_power"]) <= 8, cfg
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
    validate(root / "zero2.json", 2)
    validate(root / "zero3.json", 3)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
