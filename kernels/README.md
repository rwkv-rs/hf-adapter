# rwkv7-kernels

Optional, versioned operator backends for the readable `rwkv7-hf` model.  The
base model never depends on this distribution.  Unsupported devices, dtypes,
training requests, or shapes remain on the PyTorch reference implementation.

The promoted v1 `auto` policy uses the native Triton scan for the
latency-critical one-token FP16 decode shape and captures the readable
multi-token recurrence in reusable CUDA graphs. The Graph route keeps the
reference operation order and is bit-exact; the decode route stays within the
published FP16 gate with identical greedy tokens. V1 is inference-only;
autograd requests automatically retain the readable path.

`RWKV7_KERNEL_IMPL=auto` is the default. Explicit `graph` and `triton` modes
remain available for isolated validation and fair operator benchmarks. The
multi-token Triton scan is still experimental and is not silently selected by
`auto`; it cannot be promoted until its own model-level numerical and speed
gates pass. Set `RWKV7_KERNEL_TRACE_PATH=/path/route.json` when a subprocess
must persist counts for the implementations that actually executed.
