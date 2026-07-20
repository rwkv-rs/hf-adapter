"""Versioned optional operator protocol consumed by :mod:`rwkv7_hf`."""

from importlib.metadata import PackageNotFoundError, version

from .dispatcher import probe_recurrent_v1, recurrent_v1
from .model_dispatcher import model_forward_v1, probe_model_forward_v1
from .protocol import RWKV7_KERNEL_API_VERSION
from .training_dispatcher import (
    linear_training_v1,
    probe_linear_training_v1,
    probe_recurrent_training_v1,
    recurrent_training_v1,
)

try:
    __version__ = version("rwkv7-kernels")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = [
    "__version__",
    "RWKV7_KERNEL_API_VERSION",
    "linear_training_v1",
    "model_forward_v1",
    "probe_linear_training_v1",
    "probe_model_forward_v1",
    "probe_recurrent_v1",
    "probe_recurrent_training_v1",
    "recurrent_v1",
    "recurrent_training_v1",
]
