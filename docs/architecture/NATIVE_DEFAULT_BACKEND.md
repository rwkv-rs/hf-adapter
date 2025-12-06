# Native-Default Backend Architecture

Status: **accepted for implementation; not yet promoted as the repository default**.

## Decision

`NativeRWKV7Config`, `NativeRWKV7Model`, and `NativeRWKV7ForCausalLM` will be
the canonical Hugging Face classes. A normal converted checkpoint must load
them through Auto* metadata without requiring `RWKV7_NATIVE_MODEL` or an
installed `flash-linear-attention` package.

Production RWKV modules must not import FLA. The previous FLA-backed wrapper
will be reduced to an explicit **FLA reference backend** used for migration A/B
tests and historical evidence. Qwen full-FLA remains the optimized comparison
baseline and is outside the RWKV runtime-removal rule.

## Why the Change Is Staged

The repository currently has two meanings of "native":

1. `native_graph` and fused kernels inside the FLA-backed wrapper provide the
   fastest measured decode path.
2. `NativeRWKV7ForCausalLM` is genuinely FLA-free, but its full-sequence path
   is token-sequential and does not yet own the graph runner or official
   `train_temp` training kernels.

An RTX 5070 Laptop pre-migration probe on the same 0.4B checkpoint,
fp16/B1/prompt32/decode16 recorded pure-native eager at `31.35 tok/s`,
pure-native JIT at `41.68 tok/s`, and wrapper-hosted native graph at
`226.3 tok/s`. Changing only an environment-variable default would create a
5.43x decode regression and is rejected.

## Runtime Boundaries

The target layering is:

```text
Transformers Auto* / Generation / Trainer
  -> canonical NativeRWKV7 model and recurrent cache
  -> FLA-free graph, fused prefill/decode and native W8/W4 runtime
  -> optional official train_temp CUDA full-sequence training backend

Explicit development-only references
  -> RWKV FLA reference backend for A/B migration checks
  -> Qwen full-FLA optimized competitor baseline
```

FLA is not forbidden as a separately selected benchmark dependency. It is
forbidden as an implicit import, superclass, cache base, or default execution
dependency of the canonical RWKV model.

## Promotion Gates

- FLA-blocked clean import and Auto* load without `RWKV7_NATIVE_MODEL`.
- HF load/generate/cache/dynamic-batch/save-reload contract parity.
- Trainer, PEFT, TRL and checkpoint-resume parity.
- Native W8/W4 functionality and card-local performance claims remain valid.
- Official `train_temp` B16/T512 backward, optimizer and convergence gates.
- RTX 5070 native-default B1/B2/B4/B8 decode reaches at least 0.95x the
  previous wrapper-hosted native-graph exact-shape rows with logits and greedy
  parity.

## Alternatives Rejected

- **Flip `RWKV7_NATIVE_MODEL=1` by default:** simple, but retains hidden FLA
  architecture and causes the measured decode regression.
- **Delete the FLA wrapper immediately:** removes the only current owner of the
  proven graph runtime before that runtime is reusable.
- **Keep FLA indefinitely as the model superclass:** preserves short-term
  behavior but does not satisfy the native/upstream/AMD/clean-install target.

