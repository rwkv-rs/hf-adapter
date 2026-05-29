"""Lazy loader for an environment-specific RWKV-7 binary kernel wheel."""

from __future__ import annotations

import importlib
import json
from copy import deepcopy
from importlib.resources import files
from typing import Any


def manifest() -> dict[str, Any]:
    path = files(__package__).joinpath("_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def load_extension(name: str) -> Any:
    data = manifest()
    extensions = data.get("extensions", {})
    if name not in extensions:
        raise KeyError(f"kernel extension {name!r} is not included in this wheel")
    return importlib.import_module(str(extensions[name]))


MANIFEST = deepcopy(manifest())

__all__ = ["MANIFEST", "load_extension", "manifest"]
