"""Versioned optional operator protocol consumed by :mod:`rwkv7_hf`."""

from importlib.metadata import PackageNotFoundError, version

from .backend import execute_optional_v4
from .protocol import RWKV7_KERNEL_API_VERSION

try:
    __version__ = version("rwkv7-kernels")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = [
    "__version__",
    "RWKV7_KERNEL_API_VERSION",
    "execute_optional_v4",
]
