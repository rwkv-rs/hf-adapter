# Qwen3.5 Full-FLA bsz8 Comparison Design

> **Historical design, implemented and promoted.** Final evidence is
> [`../../bench/5070_qwen35_full_fla_bsz8_20260714/README.md`](../../bench/5070_qwen35_full_fla_bsz8_20260714/README.md).
> This file preserves the design rationale; current status lives in
> `HF_STATUS.md` and `BENCHMARK.md`.

## Goal

Compare RWKV-7 1.5B with official Qwen3.5 2B on the RTX 5070 Laptop at
`bsz=8`. The Qwen reference must use FLA chunk prefill, FLA fused-recurrent
decode, FLA fused gated normalization, and an accelerated causal convolution.
Any Transformers Torch fallback invalidates the reference row.

## Backend

The Windows environment has flash-linear-attention 0.5.1 but no compatible
`causal-conv1d` wheel. FLA already ships Triton causal-convolution prefill and
cached-update kernels. A benchmark-local adapter converts the Transformers
Qwen layout `[B,D,T]` to FLA `[B,T,D]`, preserves the in-place convolution
cache contract, and binds those callables to every live Qwen3.5 GatedDeltaNet
layer. Packed `seq_idx` input is rejected until it has an explicit oracle.

The strict operator contract accepts either the official `causal-conv1d`
extension or this FLA Triton route. It records the exact live origins and fails
closed when any linear-attention layer retains a Torch convolution.

## Fairness Metrics

Each row records unique logical parameters, active logical parameters, and
physical model footprint. Dense RWKV and Qwen activate all logical parameters;
the implementation also supports a top-k expert adjustment for future MoE
references. Quantized wrappers use their original logical shape rather than
packed storage size.

The comparator reports raw token throughput, model efficiency, and hardware
logical work rate separately. Model efficiency is `tok/s / active-B`: it
normalizes throughput for the number of parameters activated per token and is
an acceptance gate. Hardware work rate is `tok/s * active_parameters`; it is
useful utilization telemetry but is not a model-efficiency gate because it
inherently rewards a larger dense model for doing more work per token.

Peak runtime working set is peak allocated VRAM minus physical model footprint.
It discloses activation, state, graph-pool, and temporary-allocation pressure;
total peak VRAM remains the capacity gate. Parameter applications for prefill
and decode are recorded explicitly from active parameters and exact token
counts.

## Matrix And Gates

The matrix is restricted to `bsz=8`, prompt 128/512/2048, decode 128/512, and
fp16/BNB8/BNB4. Required gates are complete coverage, full optimized Qwen
bindings, raw prefill/decode speed, active-parameter-normalized model
efficiency, model footprint and peak VRAM, finite logits, and separate
full-FLA versus baseline correctness probes. Performance claims remain
exact-card and do not imply model-quality superiority.
