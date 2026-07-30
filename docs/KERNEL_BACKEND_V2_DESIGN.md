# RWKV7 NVIDIA backend-v2 design

This document freezes the clean boundary between the Hugging Face reference
model and the optional NVIDIA implementation. Kernel implementations may evolve
behind it; model names, checkpoint keys, canonical cache semantics, and the
single API-v4 facade do not.

## Ownership

`rwkv7_hf/` owns the readable model, configuration, tokenizer, loss,
generation contract, and canonical `RWKV7Cache`. A converted Hub model remains
usable with only Torch and Transformers.

`kernels/rwkv7_kernels/` owns hardware decisions and optimized execution:
Triton/CUDA operators, operand packing, CUDA Graph runners, internal state
pools, shape/device routing, quantized linear adapters, toolchain discovery,
trace accounting, and training autograd extensions. No device name, tile,
threshold, environment variable, graph runner, quantizer, or compiled-source
loader belongs in model configuration, modeling, or cache.

## Public protocol: API v4

The optional package has one public execution function:

```python
RWKV7_KERNEL_API_VERSION = 4

execute_optional_v4(
    kind: Literal[
        "training_program",
        "model_forward",
        "linear_training",
        "mix6_training",
        "recurrent",
    ],
    *args,
    **kwargs,
) -> OptionalKernelEnvelope
```

The package top level exports exactly `__version__`,
`RWKV7_KERNEL_API_VERSION`, and `execute_optional_v4`. Historical v1 helpers
are private adapters behind this facade, not supported public entry points.


Every operation returns the same plain mapping:

```text
api_version: int
kind: one of the five operation kinds
supported: bool
implementation: str
reason: str
result: object | None
phase: training | prefill | decode | implementation-defined diagnostic phase
```

An unsupported negative capability decision **must** return `result=None` and
must not mutate public cache or model state; `auto` may then execute the
readable fallback. Once positive `model_forward` execution begins, an exception
or malformed payload fails closed even in `auto`; strict `optimized` also
raises. The HF package validates the common envelope and each
operation payload; it constructs Transformers output classes itself.

Capability checks and legacy dispatcher adapters are private implementation
details of the kernel wheel. They are not imported by `rwkv7_hf`, are not a
second public protocol, and may be refactored without changing API v4.

## Operation contracts

### `training_program`

Reserves one optional atomic-program decision before the readable training
layer loop. The current request contains model facts but not the concrete
projection weights/biases and Mix6 tensors required to preflight all leaves and
their lazy dependencies. Consequently it always returns unsupported with
`result=None`: `auto` selects the complete reference program and strict
`optimized` fails at the model boundary. A future supported result may contain
a bound `program_id` only after the complete leaf plan can be validated before
the first leaf.

### `model_forward`

Receives an `RWKV7Model` or `RWKV7ForCausalLM` owner plus a normalized request.
The wheel uses structural attributes only; it never imports
`rwkv7_hf.modeling_rwkv7`. The request mirrors the stable HF fields needed by
the implementation: model kind, hidden/input tensors, attention mask, public
cache, labels, cache/hidden-state flags, cache position, logits selection,
training, checkpointing, and grad state.

A supported result is a plain mapping containing the appropriate tensors and
public cache (`last_hidden_state` for the base model, or `logits` and optional
`loss` for causal LM). It cannot contain Transformers output objects. The caller's canonical cache is
passed directly rather than cloned, allowing native decode to bind it zero-copy
to persistent CUDA Graph buffers. The negative capability path must be
side-effect-free. After positive execution starts, failure never triggers
reference recomputation over a cache that may already be bound or updated.

### `linear_training`

Consumes one projection input, weight, and optional bias plus the explicit
training certificate/facts. It returns a tensor matching the reference fixed-
row linear contract.

### `mix6_training`

Consumes hidden/shifted inputs and the six mix parameters plus the explicit
certificate/facts. It returns the six mixed tensors with ordinary autograd
semantics.

### `recurrent`

Inference consumes canonical R/W/K/V/A/B tensors, canonical initial state,
and the 2-D mask. Training additionally consumes the explicit program
certificate/facts. A supported result is `(output, final_state)` with public
state canonicalized to `[B,H,K,V]`.

## Explicit execution context

`RWKV7ExecutionContext` is a frozen model-owned value resolved once per model
call. It records training status, full-mask status, initial-state provenance,
autograd eligibility, reference-program requirement, optional program ID,
token alignment, implementation, and reason. It is passed explicitly through
blocks, TMix/CMix, recurrence, Mix6, the LM-head boundary, and checkpoint
replay.

The context does not store hardware policy, model/cache tensors, parameters,
optimizer state, kernels, or graph runners. Two narrow routing `ContextVar`
bridges carry the resolved value: one transfers decoder context to the LM head
across the standard output boundary, and lexical `linear_execution_context`
republishes it
for standard `nn.Linear`/PEFT/quantization `forward(x)` calls, including inside
checkpoint replay. They bridge routing but do not resolve policy. Two other
`ContextVar` values are evidence-only snapshots for last-route reporting and
cannot alter a subsequent call.

## Cache and layout contract

- Public recurrent state is `[B,H,K,V]`, FP32 unless the reference contract is
  deliberately changed.
- Per-layer attention and FFN shifts are `[B,C]`.
- `seen_tokens` is updated once by the owning HF forward.
- Masked tokens do not update recurrent or shift state.
- Batch reorder/select/repeat stays in `RWKV7Cache`.
- Packed, `[V,K]`, pooled, and graph-static buffers live only in the optional
  wheel and are converted/bound at API entry and exit.

## Implementation inventory

The optional wheel may implement:

- dense/fused sequence prefill and one-token decode;
- recurrent FP16/FP32 update and fused recurrent output;
- DPLR/self-chunk/shape-selected prefill;
- projection, norm/shift/mix, W/A/G/V LoRA, FFN, and output fusion;
- fixed-batch CUDA Graph capture/replay and package-owned state pools;
- SM70, Ada, and Blackwell routing;
- native W8/W4/A8W8/BN/TN and BitsAndBytes/Marlin/TorchAO adapters;
- recurrent, flattened-linear, and Mix6 training leaves.

Formal HF training retains `modeling_rwkv7.py` as its only program. The
historical opaque whole-model training runtime is preserved as a private
source diagnostic, not a formal training route.

Apple/MLX/CoreML, Ascend, MUSA, Biren, and MetaX remain separate optional
hardware distributions. Their exclusion from the NVIDIA wheel is an ownership
boundary, not a deletion of history.

## Integration invariants

- No optimized subclass, monkeypatch, duplicate model layer, or native cache.
- Installing/uninstalling `rwkv7-kernels` changes only implementation choice.
- `modeling_rwkv7.py` always retains a complete readable path.
- The adaptive API-v4 preflight may certify recurrent, linear, and Mix6 as one
  immutable training program; leaves still validate their concrete tensors.
- The current fast domain is dense B4/T128 with zero initial state, an active
  mask, aligned tokens, autograd inputs, and head size 64. Other explicitly
  adaptive requests use individually gated exact-matrix/reference leaves, and
  an unprovable autograd boundary uses one coherent reference program. Strict
  optimized requires the certificate and fails before the layer loop outside
  that domain.
- Package-free Hub loading remains valid when the wheel is absent.

## One-shot acceptance

A release candidate is one immutable HF/kernel wheel pair. Changed bytes must
pass, without relabelling historical evidence:

- operator output/state/all-gradient parity and determinism;
- 0.1B/0.4B/1.5B logits, cache, padding, greedy, and beam behavior;
- AutoModel/AutoModelForCausalLM, save/reload, and package-free fallback;
- Trainer/Accelerate/PEFT plus SFT/DPO/GRPO and resume;
- reference/optimized/FLA lm_eval equivalence;
- whole-model prefill/decode/training speed against reference and pinned FLA;
- the agreed RTX 4080 gate, followed by RTX 4090 and larger-model loading
  smoke. V100 evidence remains historical and is not a v1.0 release gate.

The local suite is necessary but does not substitute for immutable-wheel GPU
validation and route evidence.
