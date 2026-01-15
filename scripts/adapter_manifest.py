#!/usr/bin/env python3
# coding=utf-8
"""Single source of truth for files shipped with converted HF checkpoints.

Keep this module dependency-free: converter and sync tools import it before
optional ML/Apple dependencies are available. Runtime import closure is checked
by ``tests/test_sync_hf_adapter_code.py``.
"""
from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from typing import Iterable


ADAPTER_FILES = [
    "ada_lora.py",
    "ada_sparse_ffn.py",
    "blackwell_norm_mix.py",
    "dplr_prefill.py",
    "dplr_prefill_triton.py",
    "extension_build.py",
    "fused_attention_projection.py",
    "fused_ffn.py",
    "fused_lora.py",
    "fused_decode_norm_mix.py",
    "fused_norm_mix.py",
    "fused_output.py",
    "fused_prefill.py",
    "fused_projection.py",
    "fused_recurrent_update.py",
    "fused_elementwise.py",
    "fused_time_mix.py",
    "kernel_policy.py",
    "mlx_bridge.py",
    "mlx_cache.py",
    "mlx_dplr_prefill.py",
    "mlx_model.py",
    "mlx_mix.py",
    "mlx_norm.py",
    "mlx_policy.py",
    "mlx_quant.py",
    "mlx_scan.py",
    "mlx_scheduler.py",
    "mlx_session.py",
    "mlx_state.py",
    "mlx_wkv.py",
    "model_cache.py",
    "model_config.py",
    "native.py",
    "native_jit.py",
    "native_graph_runtime.py",
    "native_model.py",
    "native_quant.py",
    "native_quant_a8w8.py",
    "native_quant_bnb8.py",
    "native_quant_mm4.py",
    "native_quant_mm8.py",
    "native_quant_bn_tn.py",
    "marlin_autotune.py",
    "native_quant_marlin.py",
    "native_quant_marlin_sources.py",
    "native_quant_torchao.py",
    "native_quant_policy.py",
    "native_wkv_fp16.py",
    "remote_code/__init__.py",
    "self_chunk_A_fwd.py",
    "self_chunk_cumsum.py",
    "self_chunk_h_fwd.py",
    "self_chunk_o_fwd.py",
    "self_chunk_rwkv7.py",
    "self_chunk_utils.py",
    "self_chunk_wy_fwd.py",
    "sm70_linear.py",
    "sm70_quant.py",
    "sm70_wagv.py",
    "triton_compat.py",
    "tokenization_rwkv7.py",
]

# These files were shipped by the historical FLA-backed remote-code adapter.
# Native checkpoints remove them so stale files cannot suggest or restore the
# retired default route after an in-place code sync.
LEGACY_REMOTE_CODE_FILES = [
    "configuration_rwkv7.py",
    "modeling_rwkv7.py",
]


def normalize_manifest_path(name: str) -> PurePosixPath:
    """Validate and normalize one repository-relative manifest path.

    Manifest paths always use POSIX separators, including on Windows. Rejecting
    absolute, parent-relative, ambiguous, and backslash paths keeps conversion
    and in-place sync from writing outside their intended roots.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("adapter manifest paths must be non-empty strings")
    if "\\" in name:
        raise ValueError(f"adapter manifest paths must use '/' separators: {name!r}")
    path = PurePosixPath(name)
    normalized = path.as_posix()
    if (
        path.is_absolute()
        or not path.parts
        or normalized == "."
        or normalized != name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe adapter manifest path: {name!r}")
    return path


def validate_manifest_paths(names: Iterable[str]) -> tuple[PurePosixPath, ...]:
    """Return validated paths and reject duplicate destinations."""

    paths = tuple(normalize_manifest_path(name) for name in names)
    counts: dict[PurePosixPath, int] = {}
    for path in paths:
        counts[path] = counts.get(path, 0) + 1
    duplicates = sorted(path.as_posix() for path, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate adapter manifest paths: {duplicates}")
    return paths


def _contained_path(root: Path, relative: PurePosixPath) -> Path:
    """Resolve a manifest path while refusing symlink/path traversal escapes."""

    root = root.resolve()
    candidate = root.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"adapter manifest path escapes root {root}: {relative.as_posix()!r}"
        ) from exc
    return candidate


def copy_manifest_files(
    source_root: Path,
    destination_root: Path,
    names: Iterable[str],
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Copy manifest files, creating nested destination directories as needed."""

    source_root = Path(source_root)
    destination_root = Path(destination_root)
    copied: list[Path] = []
    for relative in validate_manifest_paths(names):
        source = _contained_path(source_root, relative)
        destination = _contained_path(destination_root, relative)
        if not source.is_file():
            raise FileNotFoundError(f"adapter source missing: {source}")
        copied.append(destination)
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    return copied


def remove_manifest_files(
    destination_root: Path,
    names: Iterable[str],
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Remove existing manifest files without traversing outside the model dir."""

    destination_root = Path(destination_root)
    removed: list[Path] = []
    for relative in validate_manifest_paths(names):
        destination = _contained_path(destination_root, relative)
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                raise IsADirectoryError(
                    f"adapter manifest file is a directory: {destination}"
                )
            removed.append(destination)
            if not dry_run:
                destination.unlink()
    return removed


__all__ = [
    "ADAPTER_FILES",
    "LEGACY_REMOTE_CODE_FILES",
    "copy_manifest_files",
    "normalize_manifest_path",
    "remove_manifest_files",
    "validate_manifest_paths",
]
