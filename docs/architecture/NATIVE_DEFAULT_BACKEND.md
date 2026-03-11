# Native-Default Backend Architecture

Status: **canonical Native inference and exact measured training migration implemented**.

## Decision

`NativeRWKV7Config`, `NativeRWKV7Model`, and `NativeRWKV7ForCausalLM` are the
canonical Hugging Face classes. A normal converted checkpoint loads
them through Auto* metadata without requiring `RWKV7_NATIVE_MODEL` or an
installed `flash-linear-attention` package.

Production RWKV modules must not import FLA. The previous FLA-backed wrapper is
an explicit **FLA reference backend** used for migration A/B
tests and historical evidence. Qwen full-FLA remains the optimized comparison
baseline and is outside the RWKV runtime-removal rule.

## Why the Change Was Staged

Before the migration, the repository had two meanings of "native":

1. `native_graph` and fused kernels inside the FLA-backed wrapper provided the
   fastest measured decode path at that review point.
2. `NativeRWKV7ForCausalLM` was genuinely FLA-free, but its full-sequence path
   was token-sequential and did not yet own the graph runner or exact
   `train_temp` training path.

An RTX 5070 Laptop pre-migration probe on the same 0.4B checkpoint,
fp16/B1/prompt32/decode16 recorded pure-native eager at `31.35 tok/s`,
pure-native JIT at `41.68 tok/s`, and wrapper-hosted native graph at
`226.3 tok/s`. Changing only an environment-variable default would create a
5.43x decode regression and is rejected.

The FLA-free migration now owns both compiled prefill and CUDA-graph decode.
On the same RTX 5070 Laptop / 0.4B / fp16 checkpoint:

- B1 native graph decode is `223.47 tok/s`, or `0.9875x` the retained
  `226.3 tok/s` wrapper-hosted row; logits cosine is `0.99999988` and greedy
  matches 32/32.
- B1/B2/B4/B8 prompt-32 prefill and decode probes all pass greedy alignment.
  Minimum prefill/decode logits cosine is `0.99999940`/`0.99999875`.
- Compiled native prefill is `3.44x-6.57x` the eager native fallback for the
  measured B1/B2/B4/B8 rows.

These rows closed the inference migration checkpoint. Later RTX 5090 evidence
also closed the exact 12x768 BF16 B16/T512 `train_temp` migration lane: 399/399
gradients and parameter deltas match, the paired three-seed, continuous
5,000-step, and 2,500+2,500 resume gates pass, and measured throughput is
`1.00049x-1.00255x` the official path. See
[`../../BENCHMARK.md`](../../BENCHMARK.md) and
[`../TRAINING.md`](../TRAINING.md). Broader model sizes, cards, distributed
topologies, and training recipes remain separate validation scopes.

## Runtime Boundaries

The current layering is:

```text
Transformers Auto* / Generation / Trainer
  -> canonical NativeRWKV7 model and recurrent cache
  -> FLA-free graph, fused prefill/decode, native W8/W4 and train_temp runtime

Explicit development-only references
  -> RWKV FLA reference backend for A/B migration checks
  -> official train_temp CUDA reference for exact training checks
  -> Qwen full-FLA optimized competitor baseline
```

FLA is not forbidden as a separately selected benchmark dependency. It is
forbidden as an implicit import, superclass, cache base, or default execution
dependency of the canonical RWKV model.

## Accepted Promotion Gates

- FLA-blocked clean import and Auto* load without `RWKV7_NATIVE_MODEL`.
- HF load/generate/cache/dynamic-batch/save-reload contract parity.
- Trainer, PEFT, TRL and checkpoint-resume parity.
- Native W8/W4 functionality and card-local performance claims remain valid.
- Exact RTX 5090 12x768 BF16 `train_temp` B16/T512 backward, optimizer,
  multi-seed, 5,000-step, resume, memory, and throughput gates.
- RTX 5070 native-default B1/B2/B4/B8 decode reaches at least 0.95x the
  previous wrapper-hosted native-graph exact-shape rows with logits and greedy
  parity.

These gates promote only their declared shapes and cards. New GPU families,
model sizes, dtypes, distributed layouts, or training recipes still require
their own correctness, memory, and performance evidence.

## Alternatives Rejected

- **Flip `RWKV7_NATIVE_MODEL=1` before migration:** would have retained hidden
  FLA architecture and caused the measured decode regression.
- **Delete the FLA wrapper before runtime extraction:** would have removed the
  then-current owner of the proven graph runtime before it was reusable.
- **Keep FLA indefinitely as the model superclass:** would have preserved
  short-term behavior but not satisfied the native/upstream/AMD/clean-install
  target.
