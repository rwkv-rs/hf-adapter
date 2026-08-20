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
  CPU / CUDA / MPS / MUSA compatibility implementation.
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

MUSA backend
  Moore Threads `torch_musa` compatibility plus optional exact-device kernels.
  MTT S70 is a legacy first-generation validation card with a frozen SDK 4.2.0
  stack, no Tensor Core, and impractically slow fp16 compute; its retained lane
  uses fp16 storage/IO with fp32 recurrent state and compute. Those limits are
  not backend-wide defaults: later S4000/S5000-class devices have more complete
  capabilities but remain unvalidated here. CUDA/Triton/FLA and quantized paths
  are not inherited. Every capability must come from MUSA documentation or
  retained exact-device evidence.

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

Optional binary CUDA extensions are distributed independently as exact-runtime
`rwkv7-kernels` wheels. They are discovered through a versioned manifest and
loaded before the historical JIT path; an incompatible wheel is never imported.
The base package and every conservative fallback remain usable without this
binary companion. See [prebuilt kernel wheels](KERNEL_WHEELS.md).

For users, detection is automatic but installation is explicit:

```bash
rwkv7-hf-kernels recommend
# Install only when the command lists one exact build.
rwkv7-hf-kernels install
```

No card name or backend flag is added to model-loading code after installation.
The default `auto` policy performs the manifest check and safe route selection.

`RWKV7_NATIVE_MODEL` is retained only for old converted directories and
historical scripts. New conversions and refreshed model directories must work
without setting it. Use `scripts/sync_hf_adapter_code.py MODEL` to migrate old
Auto* metadata before reporting a native-default result.

## Versioned lifecycle and deprecation policy

The public HF surface (`AutoConfig`, `AutoModelForCausalLM`, tokenizer,
`generate`, recurrent cache, PEFT/Trainer/TRL interfaces, and serialized
checkpoint keys) follows the package release lifecycle. Experimental runtime
flags and internal module paths do not become stable merely because a benchmark
used them.

Lifecycle states are:

| State | Contract |
|---|---|
| Stable | Backward compatible within the current major release. A replacement and migration command must ship before removal. |
| Compatibility | Retained for old converted directories or scripts while users migrate. It may warn, but must preserve the documented behavior. |
| Experimental | Opt-in and allowed to change, but removal still requires a dated table entry and release-note notice. |
| Reference only | Kept for correctness/performance A/B evidence; never selected silently by the production runtime. |

Normal removals use this minimum window:

1. mark the surface deprecated in release `X.Y` and document its replacement;
2. keep it working through at least the next minor release `X.(Y+1)`;
3. remove it no earlier than `X.(Y+2)` and only after old converted-model
   sync/load and current save/reload gates pass without it.

Security or correctness emergencies may shorten the window, but the release
notes must identify the affected surface, reason, replacement, and exact last
supported release. Silent removal is forbidden.

### Current lifecycle table

| Surface | State in 0.6 | Replacement / migration | Warning release | Earliest removal |
|---|---|---|---|---|
| `native_model.NativeRWKV7*` Auto* entrypoints and public re-exports | Stable | None; this is the canonical HF surface | n/a | Next major only |
| Flat converted-model remote-code dependency namespace | Stable compatibility boundary | Prove nested offline imports across the supported Transformers range first | Not scheduled | Not scheduled |
| Old module paths kept as import shims after source splits | Compatibility | Import the documented new owner; converted `auto_map` remains stable | First release after replacement | Two minor releases after replacement |
| `RWKV7_NATIVE_MODEL` selector | Compatibility; deprecated in 0.6 | Refresh the model with `scripts/sync_hf_adapter_code.py MODEL`; native is already the default | 0.6 | 0.8 |
| `RWKV7_NATIVE_MODEL_BACKEND` and `RWKV7_NATIVE_MODEL_JIT` | Experimental | Use default auto routing unless collecting an explicit A/B artifact | Not scheduled | Not scheduled |
| `RWKV7_KERNELS_MODE` and the `rwkv7-kernel-package-v1` manifest | Experimental in 0.8 | Default `auto`; use `prebuilt` only for strict acceptance | Not scheduled | Not scheduled |
| Historical FLA-backed RWKV wrapper | Reference only | Canonical Native/no-FLA Auto* model | Not scheduled | Not scheduled |

Adding a new experimental flag does not reserve it forever. Before deleting or
renaming one, add it to this table with a replacement, warning release, and
earliest removal release. Compatibility shims must remain thin and must not own
independent runtime state or kernel registration.

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

- `device.type == "cuda"` / `"mps"` / `"musa"` / `"cpu"`
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
