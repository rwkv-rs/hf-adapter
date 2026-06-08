<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  original_work_scope:
    - Transformers adaptation contract
    - recurrent-state serving contract
    - quantization integration contract
  metadata: ../reference/provenance.yaml
  license: MIT
-->

# RWKV-7 vLLM porting guide

## Scope

This is an implementation guide for a **separate** vLLM model plugin or
upstream integration. It deliberately avoids importing vLLM here. The goal is
to identify which behavior is portable and which Hugging Face machinery must
be replaced.

RWKV-7 is recurrent. Its decode cache has constant size with respect to
sequence length, but it is not a Transformer KV cache. A correct integration
therefore needs a request-state manager rather than forcing RWKV state into
paged KV blocks.

## Source-of-truth map

| Contract | Canonical source | Serving-engine use |
|---|---|---|
| Config and module shapes | `rwkv7_hf/native_model.py::NativeRWKV7Config` | Parse config and construct layers |
| Model/module names | `NativeRWKV7Attention`, `NativeRWKV7FFN`, `NativeRWKV7Layer` | Weight loader targets |
| Token-step mathematics | `rwkv7_hf/native.py` | Correctness oracle |
| Batched state initialization | `rwkv7_hf/native.py::_init_state_batched` | State-pool shape/dtype contract |
| HF cache behavior | `rwkv7_hf/native_model.py::NativeRWKV7Cache` | Semantic reference, not production allocator |
| Official checkpoint conversion | `rwkv7_hf/converter.py` | Loader name/transpose rules |
| Native W8/W4 formats | `rwkv7_hf/native_quant.py`, `rwkv7_hf/native_quant_mm8.py`, `rwkv7_hf/native_quant_mm4.py`, `rwkv7_hf/native_quant_a8w8.py`, `rwkv7_hf/native_quant_marlin.py` | Quant method and packed-weight loaders |
| Volta/Turing W8/W4 | `rwkv7_hf/sm70_quant.py` | Optional SM7x kernels |
| Marlin BF16/W4 | `rwkv7_hf/native_quant_marlin.py` | Optional Ada/Blackwell kernel backend |
| Hardware policy | `rwkv7_hf/kernel_policy.py` | Evidence source; translate into capability gates |
| Chunk recurrence reference | `rwkv7_hf/self_chunk_rwkv7.py`, `rwkv7_hf/mlx_dplr_prefill.py` | Prefill implementation/reference |

When sources disagree, use the official RWKV-LM mathematics first, then
`rwkv7_hf/native.py` for the adapter's tensor layout, then accepted tests and
benchmark artifacts. Optimized kernels are not a separate mathematical
specification.

## What can be reused

The following contracts are runtime-independent:

- configuration fields and validation;
- converted safetensors parameter names;
- embedding, block, final norm, and output-head structure;
- RWKV-7 token recurrence;
- recurrent state shapes and dtypes;
- W8/W4 packing and dequantization definitions;
- reference logits and state-transition tests;
- exact-card benchmark evidence.

The following code can initially be called as an external custom op if its
license and build integration are preserved:

- fused projection/norm/mix kernels;
- native graph/JIT decode operators;
- SM7x DP4A W8/W4 operators;
- vendored Marlin BF16/W4 operators;
- self-chunk DPLR prefill kernels.

Direct reuse is optional. The ABI and numerical behavior are mandatory.

## What must be replaced

Do not transplant these Hugging Face abstractions into the production engine:

- `PreTrainedModel` and `GenerationMixin`;
- `NativeRWKV7Cache` as the allocator;
- `prepare_inputs_for_generation`;
- HF beam-cache compatibility helpers;
- Python token loops;
- environment-variable dispatch as the primary policy mechanism;
- model-wide `device_map` movement.

Implement engine-native equivalents:

1. model registration and configuration parsing;
2. safetensors weight loader;
3. state-slot allocator;
4. packed/ragged prefill runner;
5. continuous-batching decode runner;
6. prefix-state cache;
7. quantization methods and packed-weight loader;
8. CUDA Graph buckets;
9. tensor/pipeline parallel plans;
10. metrics, eviction, and failure recovery.

## Minimal model interface

A runtime-neutral implementation can be organized around:

```python
class RWKV7RuntimeModel:
    def allocate_state(self, slots: int, *, device, activation_dtype):
        ...

    def prefill(
        self,
        token_ids,          # packed [total_tokens]
        cu_seqlens,        # [num_requests + 1]
        slot_ids,          # [num_requests]
        state_pool,
    ):
        """Return request-final logits and update each request slot."""

    def decode(
        self,
        token_ids,          # [num_scheduled]
        slot_ids,           # [num_scheduled]
        state_pool,
    ):
        """Run exactly one token per scheduled request."""
```

The state pool is specified in
[`RWKV7_STATE_CACHE_ABI.md`](RWKV7_STATE_CACHE_ABI.md). It must support
non-contiguous request slots without copying the whole pool each step.

## Prefill and decode contracts

### Decode

For each scheduled request:

1. load embedding for one token;
2. gather the request's layer state;
3. execute layers in order;
4. update recurrent, attention-mix, and FFN-mix state in place;
5. apply final norm and output head;
6. increment that request's `seen_tokens` by one.

Rows in a dynamic batch are mathematically independent. No tensor may mix
request rows except GEMM batching.

### Prefill

The correctness fallback processes each request in token order using the same
token-step recurrence. The production route may use chunked DPLR/WY kernels,
provided that:

- the final state equals sequential recurrence within the accepted tolerance;
- the last valid token's logits match;
- chunk boundaries do not leak across requests;
- partial final chunks use a masked kernel or a sequential fallback;
- an attention mask never advances state for a padded token.

The engine may pack requests into `[total_tokens]` with `cu_seqlens`. Dense
`[B,T]` padding is not required.

## Dynamic batching design

Use stable request slots:

```text
scheduler request id -> state slot id -> tensors for every layer
```

At a decode step the scheduler emits `slot_ids`. The runner either:

- gathers those slots into a contiguous batch, runs kernels, and scatters
  updated state back; or
- passes slot indices directly to kernels that read/write the pool.

The second approach avoids state copies and is the production target. Fixed
CUDA Graph buckets may pad the scheduled row count, but inactive rows must be
masked and must not modify persistent state.

## Chunked prefill implementation sequence

1. **Sequential oracle:** run the token recurrence and retain state after every
   token.
2. **Chunk-local projections:** produce `w`, `k`, `v`, `kk`, `a`, and `r`.
3. **Chunk summary:** encode each chunk's affine state transition.
4. **Prefix combine:** compute each chunk's start state from the request's
   incoming state.
5. **Chunk apply/output:** independently generate per-token recurrence outputs
   and the final state.
6. **Ragged packing:** add `cu_seqlens` and prevent cross-request combine.
7. **Scheduler integration:** allow prefill chunks to interleave with decode
   without changing request state order.

`rwkv7_hf/mlx_dplr_prefill.py` exposes explicit compact
summary/combine/apply reference functions. The CUDA/Triton self-chunk path is a
performance implementation, not the easiest oracle.

## Prefix-state cache

A prefix entry must include:

```text
token sequence or collision-safe token digest
checkpoint/config fingerprint
tokenizer/vocabulary fingerprint
adapter/LoRA fingerprint
quantization format and group size
state tensors
seen_tokens
state dtype/layout version
```

The entry represents state **after** consuming the cached tokens. On a hit,
clone or copy-on-write the state into a request slot and prefill only the
suffix. Never share a mutable state slot between live requests.

## Parallelism

### Pipeline parallel

Partition whole layers. A stage owns the recurrent state for its layers.
Per-token stage boundaries must transfer:

- residual hidden state `[rows,D]`;
- `v_first` `[rows,A]`, once produced by layer 0 for the current token;
- active-row/request metadata.

The next token cannot enter a later stage with the previous token's
`v_first`. Pipeline scheduling must preserve token order per request.

### Tensor parallel

A first implementation should keep the recurrent head dimension local:

- shard attention heads across ranks when `H % TP == 0`;
- shard `r/k/v` and compatible low-rank outputs by attention width;
- keep each local recurrent state `[H_local,N,N]`;
- row-parallel reduce `o_proj`;
- use standard column/row parallel FFN partitioning;
- shard or vocabulary-parallelize `lm_head`.

Low-rank W/A/G/V projections and `v_first` coupling require explicit
collective placement. Do not assume a Llama TP plan is correct without
per-layer state/logit tests.

## Hardware dispatch

Dispatch on capabilities and validated profiles, not a global GPU-name branch:

```text
backend, compute capability, dtype, shape, rows, quant format,
group size, graph support, and measured profile version
```

Exact-card schedules in `kernel_policy.py` are evidence-backed defaults. An
unrecognized device must select a conservative correct fallback. Never apply
5090 Marlin BN/TN values to Ada, Volta, Turing, ROCm, or another Blackwell card
without measurement.

## Recommended implementation phases

| Phase | Deliverable | Exit gate |
|---|---|---|
| 0 | Config and weight loader | Every tensor loaded with exact shape |
| 1 | Eager FP32/FP16 token step | State/logit parity with native HF |
| 2 | Stateful decode runner | Mixed-length continuous batching parity |
| 3 | Sequential packed prefill | Ragged batch parity |
| 4 | Chunked prefill | Chunk-size and state parity |
| 5 | Prefix-state cache | Hit/miss and copy-on-write tests |
| 6 | W8/W4 loaders and kernels | Footprint, quality, and speed gates |
| 7 | CUDA Graph buckets | No inactive-slot mutation |
| 8 | TP/PP | Single-rank parity and distributed stress |
| 9 | Production close | Full acceptance matrix |

Do not optimize before Phase 1 state parity passes. RWKV state errors may leave
the next token unchanged but diverge hundreds of tokens later.

## Non-goals and common mistakes

- Do not allocate paged KV blocks per historical token.
- Do not use one global `seen_tokens` for a mixed-length batch.
- Do not crop a recurrent state to an earlier positive prefix; recompute it.
- Do not reuse mutable prefix state across requests.
- Do not update padded/inactive rows.
- Do not call a generic quantized `forward` and then apply a fused epilogue a
  second time.
- Do not describe Marlin storage as the GPTQ/OBQ calibration algorithm.
- Do not claim universal GPU support from one exact-card benchmark.
