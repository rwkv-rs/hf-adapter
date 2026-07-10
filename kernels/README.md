# rwkv7-kernels

Optional, versioned NVIDIA backends for the readable `rwkv7-hf` model. The
base model never depends on this distribution. Installing or removing this
wheel does not replace the model/config/cache classes, checkpoint keys, or HF
forward/generation contract. Unsupported devices, dtypes, training requests,
adapters, or shapes remain on the PyTorch reference implementation in `auto`.

The promoted v1 `auto` policy uses the native Triton scan for the
latency-critical one-token FP16 decode shape and captures the readable
multi-token recurrence in reusable CUDA graphs. The Graph route keeps the
reference operation order and is bit-exact; the decode route stays within the
published FP16 gate with identical greedy tokens. V1 is inference-only;
autograd requests automatically retain the readable path.

Backend API v2 adds one whole-model capability boundary around the same public
model. Its internal implementations cover fused sequence prefill, fused cached
decode, package-owned CUDA Graph/state pools, SM70/Ada/Blackwell policies,
native and external W8/W4/A8W8 adapters, and BF16 train-temp autograd. Public
cache state remains canonical `[B,H,K,V]`; internal layouts never escape the
wheel. Adapter-wrapped FFN training fails closed to reference autograd rather
than bypassing PEFT parameters.

During pre-release validation the whole-model implementation is selected
explicitly:

```bash
RWKV7_BACKEND=optimized \
RWKV7_MODEL_KERNEL_IMPL=native \
RWKV7_KERNEL_IMPL=auto \
python your_hf_program.py
```

`RWKV7_BACKEND=auto` is the production-safe mode: an unavailable or
unsupported optional implementation falls back. Production whole-model
`RWKV7_MODEL_KERNEL_IMPL=auto` is not promoted until the immutable wheel has
passed the documented three-device correctness, HF/training, FLA, speed,
quantization and 144-unit lm_eval gates.

`RWKV7_KERNEL_IMPL=auto` is the default. Explicit `graph` and `triton` modes
remain available for isolated validation and fair operator benchmarks. The
multi-token Triton scan is still experimental and is not silently selected by
`auto`; it cannot be promoted until its own model-level numerical and speed
gates pass. Set `RWKV7_KERNEL_TRACE_PATH=/path/route.json` when a subprocess
must persist counts for the implementations that actually executed.

Quantization is opt-in through `rwkv7_kernels.quantization`; it never writes
hardware policy into `RWKV7Config` or `RWKV7Cache`. See
[`docs/KERNEL_BACKEND_V2_DESIGN.md`](../docs/KERNEL_BACKEND_V2_DESIGN.md) and
[`docs/NVIDIA_MIGRATION_AUDIT.md`](../docs/NVIDIA_MIGRATION_AUDIT.md) for the
frozen boundary and byte-audited migration inventory.
