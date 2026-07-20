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
decode, package-owned CUDA Graph/state pools, SM70/Ada/Blackwell policies, and
native and external W8/W4/A8W8 adapters. Public cache state remains canonical
`[B,H,K,V]`; internal layouts never escape the wheel.

## Optional training leaves

Training keeps the readable Hugging Face layer loop and replaces only two
mathematical leaves:

- `recurrent_training_v1`: the canonical RWKV-7 recurrence, with public
  `[B,H,K,V]` state and explicit output/state gradients;
- `linear_training_v1`: one stateless `[B,T,C]` projection flattened to a
  cuBLAS-backed PyTorch linear call.

Their probes are named `probe_recurrent_training_v1` and
`probe_linear_training_v1`.  The exact recurrent candidate records
`torch-cuda-rwkv7-batched-matrix-recurrent-training-v1`: it retains the public
`state @ (a @ b)` mixed-precision order while batching independent samples and
heads into PyTorch CUDA matrix operations.  PyTorch owns its ordinary autograd
graph.  The historical factorized CUDA diagnostic records
`native-nvidia-rwkv7-factorized-recurrent-training-v1`; its optional flattened projection
records `torch-cuda-rwkv7-flattened-linear-training-v1`.  The package never owns an
HF model class, parameter, adapter, optimizer, cache, loss, or checkpoint.
Consequently Trainer, Accelerate, PEFT and TRL still observe ordinary PyTorch
modules and autograd.

The exact matrix leaf requires CUDA but no compiler or JIT extension.  It
supports FP16/BF16 vectors, FP32 canonical state, arbitrary head/value widths,
nonzero initial state, arbitrary token length, and two-dimensional left/right
padding masks.  Batch regrouping and full-gradient parity are explicit release
gates.

The factorized leaf lazily compiles vendored C++/CUDA sources. The kernel wheel
therefore installs Ninja and requires a local `nvcc` toolkit matching the CUDA
major/minor used by PyTorch. The release harness records `CUDA_HOME`,
`TORCH_EXTENSIONS_DIR`, the compiler version, and the toolchain provenance
SHA256. The current leaf supports BF16, head size 64, zero initial state and
sm80+; arbitrary token lengths are no-op-padded to 16-token chunks. Masked
requests use the exact matrix route instead. Unsupported requests fail closed
to reference autograd in `auto`. SM70 uses an explicitly labelled reference
fallback and does not claim native CUDA training.

The linear leaf is selected only when `batch * tokens >= 128`. Below that
boundary the fixed 128-row reference projection is both the numerical contract
and the correct small-matrix route; the recurrent CUDA leaf may still run.
This gate is recorded in the validation JSON and prevents a requested CUDA
policy from being mistaken for an executed linear implementation.

Adaptive validation is opt-in and fail-closed:

```bash
RWKV7_BACKEND=auto \
RWKV7_TRAINING_KERNEL_IMPL=adaptive \
python your_hf_training_program.py
```

`adaptive` selects the factorized recurrent and flattened linear leaves only
for a fully active batch whose token length is divisible by 16. A masked,
unaligned, stateful, unsupported-dtype, or unsupported-device request selects
the exact matrix recurrence and reference linears instead. The actual leaf
name, never the `adaptive` selector, is reported in evidence.
`RWKV7_TRAINING_KERNEL_IMPL=matrix` and
`RWKV7_TRAINING_KERNEL_IMPL=factorized` isolate either program without
compatibility aliases.

Production `RWKV7_TRAINING_KERNEL_IMPL=auto` remains on reference autograd
until the full-model output, loss, gradient, padding, checkpointing, HF
ecosystem and finetune release gates pass. Adapter-wrapped training must never
be counted as optimized merely because CUDA was requested; evidence must show
both actual leaf routes above and the readable
`torch-reference-model-v1` layer loop.

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
passed the documented release-device correctness, HF/training, FLA, speed,
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

The built wheel embeds both `MIGRATION_MANIFEST.json` and
`CAPABILITY_INVENTORY.json`. The first verifies all 102 historical NVIDIA
transfers: 100 are byte-identical, while graph-cache binding and the historical
training-source dispatch are declared clean-boundary adaptations. The second
maps all 102 payloads to the actual recurrent, prefill, decode, graph/state,
SM70/Ada/Blackwell, quantization, and training runtime routes. Release auditing
rejects a wheel that merely ships an unreachable source archive.

`SOURCE_SCOPE.json` closes the denominator: it classifies every file in the
153-file historical performance-package tree and cryptographically rebuilds
the frozen Git tree ID. This distinguishes byte-identical NVIDIA code, adapted
protocol/model glue, canonical HF ownership, relocated tools, intentionally
separate non-NVIDIA backends, and the one retired non-kernel helper.
