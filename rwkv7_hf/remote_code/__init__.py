"""Remote-code packaging namespace reserved for staged adapter refactors.

The current Hugging Face entrypoints remain flat. This package marker verifies
that conversion and sync preserve nested manifest paths before runtime modules
are reorganized.
"""

LAYOUT_VERSION = 1

__all__ = ["LAYOUT_VERSION"]
