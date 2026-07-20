"""Optional stateless linear operators used by RWKV-7 training."""

from .training_flattened import linear_training_v1, probe_linear_training_v1

__all__ = ["linear_training_v1", "probe_linear_training_v1"]
