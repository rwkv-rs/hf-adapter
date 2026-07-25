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

The current API-v4 request does not contain the concrete projection
weights/biases and Mix6 parameter tensors needed to prove all three leaves
before execution. It therefore issues no optimized-training certificate:
`auto` runs the complete readable reference training program and strict
`optimized` fails at the model boundary. The three private leaf dispatchers
remain available for isolated numerical and performance diagnostics, but a
standalone leaf result is not a production HF training-program claim. This is
intentional fail-closed behavior; the boundary may be enabled only after a
future request binds and preloads the complete recurrent/linear/Mix6 plan.

Isolated adaptive diagnostics can exercise the factorized recurrent and
flattened-linear/Mix6 leaves for shapes covered by their numerical matrices.
They never receive a model-level certificate. The factorized CUDA leaf is
compiled lazily, so Ninja and a local `nvcc` toolkit matching
`torch.version.cuda` are required for that diagnostic route.

For whole-model inference, `model_forward` receives the caller's canonical
cache directly so native decode can bind it zero-copy to persistent CUDA Graph
buffers. A negative capability result must be side-effect-free. After positive
execution begins, any exception or malformed payload fails closed; the HF
facade does not recompute reference math over a cache that may already have
been bound or updated.

Production promotion requires route-proven output/state/logits/loss/all-
gradient parity, checkpoint consistency, HF ecosystem tests, SFT/DPO/GRPO,
speed comparison, and lm_eval from one immutable wheel pair. Historical
device evidence is not relabelled for changed bytes. The complete local suite must be rerun after the current source settles; the
new immutable-wheel RTX 4080 gate is still pending.

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
destinations: **89 byte-identical transfers and 13 declared clean-boundary
adaptations**. The capability inventory maps all 102 payloads exactly once to
16 runtime families.

`SOURCE_SCOPE.json` closes the denominator over the complete 153-file
historical performance tree: 89 byte-migrated NVIDIA files, 23 adapted
protocol/glue files, 7 canonical reference files, 6 relocated/retired tools,
27 separate-hardware files, and 1 retired non-kernel helper. Release auditing
recomputes those hashes from the built wheel rather than trusting the checkout.

See [`docs/KERNEL_BACKEND_V2_DESIGN.md`](../docs/KERNEL_BACKEND_V2_DESIGN.md)
and [`docs/NVIDIA_MIGRATION_AUDIT.md`](../docs/NVIDIA_MIGRATION_AUDIT.md).
