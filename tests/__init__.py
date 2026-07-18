"""Repository test package.

The explicit package marker prevents unrelated third-party ``tests`` packages
from shadowing helpers imported by the MLX regression modules.
"""

from __future__ import annotations

import sys

from . import apple_silicon_utils as _apple_silicon_utils

# Apple smoke files are also executable as standalone scripts and therefore
# use the historical top-level import.  Preserve that entry point when pytest
# imports the same files through the explicit ``tests`` package.
sys.modules.setdefault("apple_silicon_utils", _apple_silicon_utils)
