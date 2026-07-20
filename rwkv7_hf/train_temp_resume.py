"""Fail-closed checkpoint helpers for train_temp alignment runs."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1


def _hash_update(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_update(digest, key)
            _hash_update(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
        for item in value:
            _hash_update(digest, item)
        return
    if value is None:
        digest.update(b"none\0")
        return
    digest.update(type(value).__name__.encode("ascii", errors="replace"))
    digest.update(b"\0")
    digest.update(repr(value).encode("utf-8"))
    digest.update(b"\0")


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            # NumPy's MT19937 keys are uint32.  PyTorch 2.5 can construct a
            # torch.uint32 tensor but its legacy checkpoint serializer cannot
            # pickle the corresponding TypedStorage.  Store the same values
            # losslessly as int64 and cast back to uint32 during restore.  This
            # keeps resume checkpoints portable across legacy torch 2.5
            # environments and newer torch releases.
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA RNG device count mismatch: "
                f"checkpoint={len(cuda_states)} runtime={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    provenance: dict[str, Any],
    next_step: int,
    train_curve: list[dict[str, Any]],
    validation_curve: list[dict[str, Any]],
    runtime_s_accumulated: float,
) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_state = _cpu_copy(model.state_dict())
    optimizer_state = _cpu_copy(optimizer.state_dict())
    rng_state = capture_rng_state()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "axis": "train_temp_alignment_checkpoint",
        "provenance": provenance,
        "next_step": int(next_step),
        "train_curve": train_curve,
        "validation_curve": validation_curve,
        "runtime_s_accumulated": float(runtime_s_accumulated),
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "rng_state": rng_state,
        "model_state_sha256": state_sha256(model_state),
        "optimizer_state_sha256": state_sha256(optimizer_state),
        "rng_state_sha256": state_sha256(rng_state),
    }
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return {
        "path": str(output.resolve()),
        "sha256": _file_sha256(output),
        "bytes": output.stat().st_size,
        "next_step": int(next_step),
        "model_state_sha256": payload["model_state_sha256"],
        "optimizer_state_sha256": payload["optimizer_state_sha256"],
        "rng_state_sha256": payload["rng_state_sha256"],
    }


def restore_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported train_temp checkpoint schema")
    if payload.get("axis") != "train_temp_alignment_checkpoint":
        raise RuntimeError("not a train_temp alignment checkpoint")
    actual_provenance = payload.get("provenance")
    if actual_provenance != expected_provenance:
        raise RuntimeError(
            "train_temp checkpoint provenance mismatch: "
            f"{actual_provenance!r} != {expected_provenance!r}"
        )

    payload_digests = {
        "model_state_sha256": state_sha256(payload["model_state"]),
        "optimizer_state_sha256": state_sha256(payload["optimizer_state"]),
        "rng_state_sha256": state_sha256(payload["rng_state"]),
    }
    digest_mismatches = {
        key: {"expected": str(payload[key]), "actual": actual}
        for key, actual in payload_digests.items()
        if actual != str(payload[key])
    }
    if digest_mismatches:
        raise RuntimeError(
            f"train_temp checkpoint payload digest mismatch: {digest_mismatches}"
        )

    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng_state(payload["rng_state"])
    model_digest = state_sha256(model.state_dict())
    optimizer_digest = state_sha256(optimizer.state_dict())
    rng_digest = state_sha256(capture_rng_state())
    expected_model_digest = payload_digests["model_state_sha256"]
    expected_optimizer_digest = payload_digests["optimizer_state_sha256"]
    expected_rng_digest = payload_digests["rng_state_sha256"]
    report = {
        "path": str(checkpoint_path.resolve()),
        "sha256": _file_sha256(checkpoint_path),
        "next_step": int(payload["next_step"]),
        "model_state_sha256": model_digest,
        "optimizer_state_sha256": optimizer_digest,
        "rng_state_sha256": rng_digest,
        "model_state_restored": model_digest == expected_model_digest,
        "optimizer_state_restored": optimizer_digest == expected_optimizer_digest,
        "rng_state_restored": rng_digest == expected_rng_digest,
    }
    if not all(
        report[key]
        for key in (
            "model_state_restored",
            "optimizer_state_restored",
            "rng_state_restored",
        )
    ):
        raise RuntimeError(f"train_temp checkpoint restore digest mismatch: {report}")
    metadata = {
        "next_step": int(payload["next_step"]),
        "train_curve": list(payload["train_curve"]),
        "validation_curve": list(payload["validation_curve"]),
        "runtime_s_accumulated": float(payload.get("runtime_s_accumulated", 0.0)),
    }
    del payload
    return metadata, report
