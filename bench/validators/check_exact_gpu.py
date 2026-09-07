#!/usr/bin/env python3
"""Fail-closed exact CUDA product check for acceptance entrypoints."""

from __future__ import annotations

# Support direct ``python bench/<category>/<script>.py`` execution.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.benchlib.gpu_guard import matches_gpu_product  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    product = parser.add_mutually_exclusive_group(required=True)
    product.add_argument("--model", help="desktop RTX model number")
    product.add_argument("--exact-name", help="exact CUDA product name")
    parser.add_argument("--name", required=True, help="detected CUDA product name")
    args = parser.parse_args()
    matches = matches_gpu_product(
        args.name,
        rtx_model=args.model,
        exact_name=args.exact_name,
    )
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
