# rwkv7-kernels

Optional, versioned operator backends for the readable `rwkv7-hf` model.  The
base model never depends on this distribution.  Unsupported devices, dtypes,
training requests, or shapes remain on the PyTorch reference implementation.

The promoted v1 backend captures the readable recurrence in reusable CUDA
graphs.  It keeps the reference operation order and is bit-exact while
removing most per-token launch overhead.  More aggressive Triton experiments
remain on the performance branch until every model-level numerical gate
passes.  V1 is inference-only; autograd requests automatically retain the
readable path.

The first clean-boundary port of the prior fused recurrent scan is available
for validation with `RWKV7_KERNEL_IMPL=triton`. The default remains `graph`;
`auto` prefers Triton only for supported FP16 K=V=64 inference requests. The
Triton lane cannot become the package default until operator, cache, greedy,
full lm_eval, and three-way FLA evidence pass on the release devices.
