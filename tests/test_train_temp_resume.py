from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from rwkv7_hf.train_temp_resume import (
    capture_rng_state,
    restore_training_checkpoint,
    save_training_checkpoint,
    state_sha256,
)


def test_training_checkpoint_restores_model_optimizer_rng_and_progress(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    np.random.seed(12)
    random.seed(13)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    provenance = {"backend": "unit", "sequence_sha256": "abc", "seed": 13}
    path = tmp_path / "resume.pt"
    metadata = save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        provenance=provenance,
        next_step=7,
        train_curve=[{"step": 7, "loss": 1.0}],
        validation_curve=[{"step": 0, "loss": 2.0}],
        runtime_s_accumulated=3.5,
    )
    saved_rng = capture_rng_state()
    saved_rng_digest = state_sha256(saved_rng)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100)
    optimizer.param_groups[0]["lr"] = 9.0
    torch.manual_seed(99)
    np.random.seed(98)
    random.seed(97)

    progress, report = restore_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_provenance=provenance,
    )
    assert metadata["next_step"] == 7
    assert progress["next_step"] == 7
    assert progress["train_curve"] == [{"step": 7, "loss": 1.0}]
    assert progress["runtime_s_accumulated"] == 3.5
    assert report["model_state_restored"] is True
    assert report["optimizer_state_restored"] is True
    assert report["rng_state_restored"] is True
    assert state_sha256(capture_rng_state()) == saved_rng_digest


def test_training_checkpoint_rejects_provenance_mismatch(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "resume.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        provenance={"seed": 1},
        next_step=0,
        train_curve=[],
        validation_curve=[],
        runtime_s_accumulated=0.0,
    )

    try:
        restore_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_provenance={"seed": 2},
        )
    except RuntimeError as exc:
        assert "provenance mismatch" in str(exc)
    else:
        raise AssertionError("provenance mismatch was accepted")


def test_training_checkpoint_rejects_corrupt_payload_before_model_load(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "resume.pt"
    provenance = {"seed": 1}
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        provenance=provenance,
        next_step=0,
        train_curve=[],
        validation_curve=[],
        runtime_s_accumulated=0.0,
    )
    original_model_digest = state_sha256(model.state_dict())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    first_key = next(iter(payload["model_state"]))
    payload["model_state"][first_key].add_(1)
    torch.save(payload, path)

    try:
        restore_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_provenance=provenance,
        )
    except RuntimeError as exc:
        assert "payload digest mismatch" in str(exc)
    else:
        raise AssertionError("corrupt checkpoint payload was accepted")
    assert state_sha256(model.state_dict()) == original_model_digest
