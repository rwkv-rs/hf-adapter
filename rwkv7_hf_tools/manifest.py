#!/usr/bin/env python3
# coding=utf-8
"""Traversal-safe file helpers used by the self-contained model converter."""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from typing import Iterable


def normalize_manifest_path(name: str) -> PurePosixPath:
    """Validate and normalize one repository-relative manifest path.

    Manifest paths always use POSIX separators, including on Windows. Rejecting
    absolute, parent-relative, ambiguous, and backslash paths keeps conversion
    and in-place sync from writing outside their intended roots.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("model manifest paths must be non-empty strings")
    if "\\" in name:
        raise ValueError(f"model manifest paths must use '/' separators: {name!r}")
    path = PurePosixPath(name)
    normalized = path.as_posix()
    if (
        path.is_absolute()
        or not path.parts
        or normalized == "."
        or normalized != name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe model manifest path: {name!r}")
    return path


def validate_manifest_paths(names: Iterable[str]) -> tuple[PurePosixPath, ...]:
    """Return validated paths and reject duplicate destinations."""

    paths = tuple(normalize_manifest_path(name) for name in names)
    counts: dict[PurePosixPath, int] = {}
    for path in paths:
        counts[path] = counts.get(path, 0) + 1
    duplicates = sorted(path.as_posix() for path, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate model manifest paths: {duplicates}")
    return paths


def _contained_path(
    root: Path,
    relative: PurePosixPath,
    *,
    allow_leaf_symlink: bool = False,
) -> Path:
    """Resolve a manifest path while refusing symlink/path traversal escapes."""

    root = root.resolve()
    candidate = root.joinpath(*relative.parts)
    # A leaf symlink inside a Hugging Face cache snapshot may point to the
    # shared blobs directory. Removing that leaf is safe and must not be
    # confused with following it for a write. Parent-directory symlinks remain
    # forbidden.
    resolved = (
        candidate.parent.resolve(strict=False) / candidate.name
        if allow_leaf_symlink
        else candidate.resolve(strict=False)
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"model manifest path escapes root {root}: {relative.as_posix()!r}"
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
            raise FileNotFoundError(f"model source missing: {source}")
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
        destination = _contained_path(
            destination_root, relative, allow_leaf_symlink=True
        )
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                raise IsADirectoryError(
                    f"model manifest file is a directory: {destination}"
                )
            removed.append(destination)
            if not dry_run:
                destination.unlink()
    return removed


__all__ = [
    "copy_manifest_files",
    "normalize_manifest_path",
    "remove_manifest_files",
    "validate_manifest_paths",
]
