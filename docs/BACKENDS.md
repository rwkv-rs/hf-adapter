# Backend boundaries and hardware validation rules

This adapter targets many devices, but the codebase should stay backend-driven
rather than card-driven.

> Hardware cards are validation rows. Backends and capabilities are code
> boundaries.

## Layering contract

```text
HF public API
  AutoModelForCausalLM / generate / Trainer / PEFT / TRL / save_pretrained
  Must not contain per-card branches.

Native PyTorch backend
  CPU / CUDA / MPS compatibility implementation.
  May branch on framework capabilities such as device.type, dtype support,
  cache support, and optional package availability.

CUDA performance backend
  Triton, CUDA graph, fused fp16, and fused quant kernels.
  May branch on normalized GPU family through rwkv7_hf.kernel_policy only.

Apple backend
  MPS compatibility path, MLX correctness reference, and future MLX/Metal
  fused kernels.
  May branch on backend availability (MPS / MLX / Metal), not on a specific
  Apple chip model.

Tests / scripts / bench / docs
  Own the hardware matrix: exact card names, machine names, benchmark rows,
  CI commands, and validation evidence live here.
```

## Runtime selection

Converted checkpoints use `NativeRWKV7Config`, `NativeRWKV7Model`, and
`NativeRWKV7ForCausalLM` in their Auto* metadata. Base installation and
`.[cuda]` therefore have no mandatory FLA dependency; CUDA adds Triton/native
fusion capability without changing the public model class.

The historical FLA wrapper is a separately selected reference implementation.
Install `.[fla-reference]` only for a benchmark that explicitly verifies the
reference class and effective operators. Qwen full-FLA comparisons are also
reference workloads and do not change the RWKV user runtime.

`RWKV7_NATIVE_MODEL` is retained only for old converted directories and
historical scripts. New conversions and refreshed model directories must work
without setting it. Use `scripts/sync_hf_adapter_code.py MODEL` to migrate old
Auto* metadata before reporting a native-default result.

## Allowed hardware-specific locations

Exact card or chip names are allowed in:

- `docs/**`
- `tests/**`
- `scripts/**`
- `bench/**`
- top-level status / roadmap docs such as `README.md`, `HF_STATUS.md`,
  `HF_TODO.md`, `BENCHMARK.md`, and `CONTRIBUTIONS.md`
- `rwkv7_hf/kernel_policy.py`, the single centralized runtime default-policy
  file for normalized accelerator families

They should not be scattered across model implementation files such as
`rwkv7_hf/modeling_rwkv7.py`, `rwkv7_hf/native_model.py`, or fused kernel
wrappers.  Those files should ask about capabilities:

- `device.type == "cuda"` / `"mps"` / `"cpu"`
- optional backend availability (`triton`, `mlx`, Metal extension)
- dtype support
- graph-capture support
- fused-kernel availability
- normalized policy family returned by `rwkv7_hf.kernel_policy`

## What not to add

Avoid code like:

```python
if "V100" in device_name:
    ...
elif "A100" in device_name:
    ...
elif "Apple M5" in device_name:
    ...
```

Prefer:

```python
policy = current_kernel_policy(torch_module=torch)
if tensor.is_cuda and policy.fused_output:
    ...
elif tensor.device.type == "mps":
    ...
else:
    ...
```

For Apple-specific work, prefer:

```python
if mlx_available() and backend == "mlx":
    ...
elif metal_available() and backend == "metal":
    ...
else:
    native_torch_fallback()
```

## Promotion rule

New optimized defaults require evidence, not assumptions:

1. correctness parity against the reference path;
2. cache / dynamic-batch / chunked-prefill behavior where serving is claimed;
3. memory telemetry;
4. speed rows for the target backend and model size;
5. docs or benchmark JSONL rows that identify the exact tested hardware.

The evidence may name cards.  The implementation should remain backend /
capability based.
