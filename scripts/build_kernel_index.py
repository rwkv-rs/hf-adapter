#!/usr/bin/env python3
"""Generate the hash-pinned public index consumed by rwkv7-hf-kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

try:
    from .inspect_kernel_wheel import inspect_wheel
except ImportError:  # pragma: no cover - direct script execution
    from inspect_kernel_wheel import inspect_wheel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    entries = []
    for wheel in args.wheels:
        item = inspect_wheel(wheel.resolve())
        item["url"] = f"{base_url}/{quote(item['filename'])}"
        entries.append(item)
    entries.sort(key=lambda item: item["filename"])
    index = {
        "schema_version": 1,
        "protocol": "rwkv7-kernel-index-v1",
        "release_tag": args.release_tag,
        "wheels": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, indent=2, sort_keys=True))
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
