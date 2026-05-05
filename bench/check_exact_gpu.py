#!/usr/bin/env python3
"""Fail-closed exact CUDA product check for acceptance entrypoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv7_hf.kernel_policy import is_rtx_model_name  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    product = parser.add_mutually_exclusive_group(required=True)
    product.add_argument("--model", help="desktop RTX model number")
    product.add_argument("--exact-name", help="exact CUDA product name")
    parser.add_argument("--name", required=True, help="detected CUDA product name")
    args = parser.parse_args()
    matches = (
        args.name.strip() == args.exact_name.strip()
        if args.exact_name is not None
        else is_rtx_model_name(args.name, args.model)
    )
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
