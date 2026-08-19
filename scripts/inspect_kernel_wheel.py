#!/usr/bin/env python3
"""Inspect and validate an RWKV-7 binary kernel wheel without installing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any


PROTOCOL = "rwkv7-kernel-package-v1"
MANIFEST_SUFFIX = "rwkv7_kernels/_manifest.json"
REQUIRED_FIELDS = (
    "distribution",
    "version",
    "adapter_specifier",
    "python_series",
    "python_abi",
    "platform_system",
    "platform_machine",
    "torch_series",
    "torch_cuda_version",
    "torch_cxx11_abi",
    "architectures",
    "extensions",
    "source_commit",
    "source_tree_sha256",
)


def inspect_wheel(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        manifests = [
            name for name in archive.namelist() if name.endswith(MANIFEST_SUFFIX)
        ]
        if len(manifests) != 1:
            raise RuntimeError(
                f"{path.name} must contain exactly one {MANIFEST_SUFFIX}; found {manifests}"
            )
        manifest = json.loads(archive.read(manifests[0]))
        binary_modules = sorted(
            name
            for name in archive.namelist()
            if name.startswith("rwkv7_kernels/_C/")
            and name.rsplit("/", 1)[-1].endswith((".so", ".pyd"))
        )
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError(f"unexpected kernel protocol: {manifest.get('protocol')!r}")
    if manifest.get("schema_version") != 1:
        raise RuntimeError(
            f"unexpected kernel schema: {manifest.get('schema_version')!r}"
        )
    missing_fields = [name for name in REQUIRED_FIELDS if manifest.get(name) is None]
    if missing_fields:
        raise RuntimeError(f"kernel manifest missing required fields: {missing_fields}")
    if manifest["distribution"] != "rwkv7-kernels":
        raise RuntimeError(
            f"unexpected kernel distribution: {manifest['distribution']!r}"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest["source_tree_sha256"])):
        raise RuntimeError("source_tree_sha256 must be a lowercase SHA256 digest")
    extensions = manifest["extensions"]
    if not isinstance(extensions, dict) or not extensions:
        raise RuntimeError("kernel manifest extensions must be a non-empty mapping")
    invalid_modules = {
        name: module
        for name, module in extensions.items()
        if module != f"rwkv7_kernels._C.{name}"
    }
    if invalid_modules:
        raise RuntimeError(f"unexpected extension module paths: {invalid_modules}")
    expected = set(extensions)
    actual = {Path(name).name.split(".", 1)[0] for name in binary_modules}
    if expected != actual:
        raise RuntimeError(
            f"binary module mismatch: manifest={sorted(expected)}, wheel={sorted(actual)}"
        )
    return {
        "filename": path.name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "manifest": manifest,
        "binary_modules": binary_modules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "status": "passed",
        "wheels": [inspect_wheel(path.resolve()) for path in args.wheels],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"RESULT: FAIL: {exc}", file=sys.stderr)
        raise
