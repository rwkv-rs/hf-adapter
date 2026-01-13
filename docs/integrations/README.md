<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  metadata: ../reference/provenance.yaml
  license: MIT
-->

# Serving-engine implementation references

This directory documents the contracts that a separate vLLM, SGLang, or other
serving-engine integration needs. It does **not** add a serving-engine runtime
dependency to this Hugging Face adapter.

Read in this order:

1. [`VLLM_PORTING_GUIDE.md`](VLLM_PORTING_GUIDE.md) — integration boundaries
   and an implementation sequence.
2. [`../architecture/RWKV7_OPERATOR_SPEC.md`](../architecture/RWKV7_OPERATOR_SPEC.md)
   — runtime-independent RWKV-7 block and recurrence contract.
3. [`RWKV7_STATE_CACHE_ABI.md`](RWKV7_STATE_CACHE_ABI.md) — request state pool,
   dynamic batching, prefix reuse, and chunked prefill.
4. [`../quantization/VLLM_QUANTIZATION_PORTING.md`](../quantization/VLLM_QUANTIZATION_PORTING.md)
   — W8/W4 formats, kernel interfaces, and hardware dispatch.
5. [`VLLM_CHECKPOINT_MAPPING.md`](VLLM_CHECKPOINT_MAPPING.md) — configuration
   and parameter-name mapping.
6. [`../validation/VLLM_ACCEPTANCE.md`](../validation/VLLM_ACCEPTANCE.md) —
   correctness, scheduling, memory, performance, and distributed gates.

The canonical implementation remains the native Transformers path. Serving
engines should reuse the mathematical and serialization contracts, while
replacing the Hugging Face generation/cache shell with their own scheduler,
request-state allocator, model runner, and distributed execution.
