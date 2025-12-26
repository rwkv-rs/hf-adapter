#!/usr/bin/env python3
"""Fail-closed exact desktop RTX product check for acceptance entrypoints."""
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
    parser.add_argument("--model", required=True, help="desktop RTX model number")
    parser.add_argument("--name", required=True, help="detected CUDA product name")
    args = parser.parse_args()
    return 0 if is_rtx_model_name(args.name, args.model) else 1


if __name__ == "__main__":
    raise SystemExit(main())
