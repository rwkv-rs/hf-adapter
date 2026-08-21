#!/usr/bin/env python3
# coding=utf-8
"""Backward-compatible source-checkout wrapper for ``rwkv7-hf convert``."""

from __future__ import annotations

import sys
from pathlib import Path


# Running an absolute ``scripts/...`` path sets sys.path[0] to ``scripts``.
# Add the checkout root so this wrapper always reaches the adjacent package,
# even when the current working directory is elsewhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv7_hf.converter import *  # noqa: F401,F403,E402
from rwkv7_hf.converter import main as _main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(_main())
