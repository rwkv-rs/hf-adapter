# rwkv7-kernels

Optional, versioned NVIDIA implementations for the readable `rwkv7-hf`
model. The base model never depends on this distribution. Installing or
removing the wheel does not replace model/config/cache classes, checkpoint
keys, or the Hugging Face forward and generation contract.

`RWKV7_BACKEND=auto` uses an optional route only when the installed wheel
accepts the exact device, dtype, shape, state, mask, and autograd request.
Unsupported work returns to the PyTorch reference implementation.
`RWKV7_BACKEND=reference` disables the plugin; `optimized` is a strict
diagnostic mode and raises instead of hiding an unsupported or failed route.

## One public API-v4 facade

The kernel distribution exposes one execution entry point:

```python
RWKV7_KERNEL_API_VERSION = 4
execute_optional_v4(kind, *args, **kwargs) -> envelope
```

The package top level exports exactly three symbols: `__version__`,
`RWKV7_KERNEL_API_VERSION`, and `execute_optional_v4`. Historical v1
dispatch/probe helpers remain private implementation adapters and are not part
of the supported package API.

`kind` is exactly one of:

- `training_program` — reserve the atomic optional-training preflight boundary;
- `model_forward` — fused prefill/decode or another accepted model request;
- `linear_training` — flattened stateless projection;
- `mix6_training` — six-way shifted-input construction;
- `recurrent` — inference or training recurrence using canonical tensors.

Every call returns the same outer envelope:

```text
api_version, kind, supported, implementation, reason, result, phase
```

The wheel ships the machine-readable `KERNEL_PLUGIN_API.json`; it is the
authoritative API-v4 operation, envelope, cache-layout, and failure-policy
contract. The complete 1.0.0 distribution inputs are SHA-256 locked by the
repository's `RELEASE_SOURCE_FREEZE.json`. New implementations plug into this
contract instead of adding imports to the HF model package.

An unsupported negative capability decision is side-effect-free and always has `result=None`. Backend selection,
capability checks, environment parsing, probes, execution, implementation
errors, and trace accounting are owned by this wheel. Its internal dispatchers
may adapt older implementation functions, but those functions are not the HF
ABI. `rwkv7_hf.ops_rwkv7` performs one lazy API-version check, validates this
common envelope, and either returns its result or executes the readable
fallback.

## Inference implementations

The wheel contains native Triton/CUDA recurrence, fused sequence prefill,
fused cached decode, package-owned CUDA Graph/state pools, and SM70, Ada, and
Blackwell policies. Explicit `graph` and `triton` implementation selectors are
available for isolated validation. A requested selector is never accepted as
proof of execution; release evidence records the returned `implementation` and
`phase`.

Public recurrent state remains canonical `[B,H,K,V]`. Internal packed,
`[V,K]`, graph-static, or pooled layouts are converted at the facade boundary
and never escape the wheel. The wheel does not own a Hugging Face model class,
parameter, loss, cache, checkpoint, optimizer, or adapter.

## Optional training program

Training always keeps the readable `modeling_rwkv7.py` layer loop. It can
replace three mathematical leaves without hiding the model from Trainer,
Accelerate, PEFT, or TRL:

- recurrent state update — factorized CUDA where certified, exact batched
  matrix recurrence otherwise;
- flattened `[B*T,C]` linear projection;
- explicit-shift Mix6 construction and gradients.

The model resolves one immutable `RWKV7ExecutionContext` before the layer
loop. Modeling passes it explicitly across blocks, TMix/CMix, recurrence,
Mix6, the LM-head boundary, and gradient-checkpoint replay. Two narrow routing bridges carry the resolved value without changing public
interfaces. One transfers decoder context to the LM head across the standard
Transformers output boundary. Standard `nn.Linear`, PEFT, and quantization
wrappers must retain `forward(x)`, so the other is the lexically scoped
`linear_execution_context` `ContextVar`, which bridges the already resolved
context to owned `RWKV7Linear` leaves. Checkpoint replay republishes
that same lexical scope. Neither bridge makes a second decision or carries
hardware policy or tensor state. Two other `ContextVar` objects retain only
last-route and last-context evidence; neither participates in selection.

`RWKV7_TRAINING_KERNEL_IMPL=adaptive` enables the API-v4 atomic preflight. The
current certificate is deliberately narrow: dense B4/T128, fully active mask,
zero initial state, aligned tokens, gradient-bearing inputs, and head size 64.
It preloads the native dependencies before the layer loop and binds one
program identity to the immutable execution context. The factorized
recurrent, flattened-linear, and Mix6 leaves revalidate their concrete tensors
against that identity; an unexpected decline is a fail-closed error.

Outside the certified domain, an explicitly adaptive request uses separately
gated exact-matrix/reference leaves. A model boundary that cannot prove
autograd eligibility selects one complete reference program. Strict
`optimized` requires the atomic certificate and fails outside its domain;
explicit `matrix` and `factorized` selectors remain operator diagnostics. The
factorized CUDA and Mix6 leaves compile lazily, so Ninja and a local `nvcc`
toolkit matching `torch.version.cuda` are required. Ordinary large projections
use one flattened GEMM; 4x FFN projections use a bounded 320-row grouping that
passed the complete-gradient gate without giving up the large-batch launch
reduction.

For whole-model inference, `model_forward` receives the caller's canonical
cache directly so native decode can bind it zero-copy to persistent CUDA Graph
buffers. A negative capability result must be side-effect-free. After positive
execution begins, any exception or malformed payload fails closed; the HF
facade does not recompute reference math over a cache that may already have
been bound or updated.

Production promotion requires route-proven output/state/logits/loss/all-
gradient parity, checkpoint consistency, HF ecosystem tests, SFT/DPO/GRPO,
speed comparison, and lm_eval from one immutable wheel pair. Historical
device evidence is not relabelled for changed bytes.

## Quantization and trace evidence

Quantization is opt-in through `rwkv7_kernels.quantization`. Native
W8/W4/A8W8, BN/TN, BitsAndBytes, Marlin, and TorchAO adapters do not add
hardware fields to `RWKV7Config` or private layout fields to `RWKV7Cache`.

Set `RWKV7_KERNEL_TRACE_PATH=/path/route.json` when a subprocess must persist
actual implementation counts. Policy names such as `auto`, `optimized`,
`adaptive`, `graph`, or `triton` are requests, not execution evidence.

## Migration audit

The built wheel embeds `MIGRATION_MANIFEST.json` and
`CAPABILITY_INVENTORY.json`. The manifest verifies all 102 historical NVIDIA
destinations: **86 byte-identical transfers and 16 declared clean-boundary
adaptations**. The capability inventory maps all 102 payloads exactly once to
16 runtime families.

`SOURCE_SCOPE.json` closes the denominator over the complete 153-file
historical performance tree: 86 byte-migrated NVIDIA files, 26 adapted
protocol/glue files, 7 canonical reference files, 6 relocated/retired tools,
27 separate-hardware files, and 1 retired non-kernel helper. Release auditing
recomputes those hashes from the built wheel rather than trusting the checkout.

See [`docs/KERNEL_BACKEND_V2_DESIGN.md`](../docs/KERNEL_BACKEND_V2_DESIGN.md)
and [`docs/NVIDIA_MIGRATION_AUDIT.md`](../docs/NVIDIA_MIGRATION_AUDIT.md).
