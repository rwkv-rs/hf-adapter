from __future__ import annotations

import torch
from types import SimpleNamespace

from scripts.run_train_temp_native_recipe import (
    build_deepspeed_config,
    inspect_deepspeed_gradients,
    validate_batch,
)
from scripts.run_train_temp_official_recipe import load_recipe


def test_native_recipe_matches_official_b16_zero2_contract() -> None:
    recipe = load_recipe()
    config = build_deepspeed_config(recipe)

    assert config["train_micro_batch_size_per_gpu"] == 16
    assert config["train_batch_size"] == 16
    assert config["gradient_accumulation_steps"] == 1
    assert config["gradient_clipping"] == 1.0
    assert config["bf16"]["enabled"] is True
    assert config["zero_optimization"]["stage"] == 2
    assert config["zero_optimization"]["allgather_bucket_size"] == 200_000_000
    assert config["zero_optimization"]["reduce_bucket_size"] == 200_000_000


def test_native_recipe_rejects_non_official_batch_shape() -> None:
    recipe = load_recipe()
    tokens = torch.arange(17, dtype=torch.long).repeat(2, 1)
    batch = {"input_ids": tokens[:, :-1], "targets": tokens[:, 1:]}

    try:
        validate_batch(batch, recipe)
    except ValueError as exc:
        assert "batch shape must be" in str(exc)
    else:
        raise AssertionError("expected non-B16/T512 batch to be rejected")


def test_inspect_deepspeed_gradients_reads_zero2_partitions() -> None:
    class Optimizer:
        averaged_gradients = {0: [torch.ones(3), torch.full((2,), float("nan"))]}

    class Module:
        def parameters(self):
            return []

    engine = SimpleNamespace(optimizer=Optimizer(), module=Module())
    assert inspect_deepspeed_gradients(engine) == (
        "deepspeed_zero_partition",
        2,
        1,
    )
