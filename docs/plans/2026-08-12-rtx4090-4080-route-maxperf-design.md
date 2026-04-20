# RTX 4090 reuse of exact RTX 4080 Ada routes

> Lifecycle: active exact-card design for the 2026-08-12 RTX 4090 validation.

## Context and decision

RTX 4080 and RTX 4090 are both Ada `sm_89`, but the repository deliberately
isolates measured defaults by exact product name. The current RTX 4090 policy
therefore does not inherit three recent RTX 4080 wins: scoped FP16 GEMM
accumulation during native Prefill, the batch-8 grouped W/A/V projection BMM,
and the 7.2B batch-8 Triton FP16 recurrent-state route. This is correct safety
behavior; the present work will reuse the implementations while independently
proving their dispatch settings on the RTX 4090.

The selected order is:

1. screen full-Prefill and block-only FP16 GEMM accumulation on the 0.4B,
   1.5B, and 2.9B B1/B8 P128/P512/P2048 shapes;
2. screen the existing grouped W/A/V tensor-core BMM on B8 decode for the
   0.4B, 1.5B, and 2.9B checkpoints;
3. screen FP16 recurrent state on the 7.2B/B8 decode shape if it fits with the
   required correctness oracle;
4. tune the existing scan tile, self-chunk, and graph settings only where the
   exact 4090 end-to-end profile identifies them as a remaining hotspot.

The first route is preferred because dense projections and FFN dominate the
measured Prefill budget, it changes no public HF or cache contract, and both
the latest RTX 4080 and RTX 3090 work demonstrate that the existing PyTorch
scope can accelerate these GEMMs without replacing cuBLAS. The RTX 4090 uses
its driver-compatible PyTorch 2.7.1/CUDA 12.6 runtime rather than copying the
RTX 4080 CUDA 13 runtime.

## Alternatives and boundaries

Blindly enabling all exact-4080 policy tuples for every `sm_89` card is
rejected: identical compute capability does not imply identical occupancy,
clock, memory-bandwidth, or launch winners. A new wrapper/cache optimization is
also rejected because the repository profile attributes the important Prefill
gap to kernels and GEMMs. Compact WY/DPLR and deeper state-scan/output fusion
remain the next research options only if the three existing Ada routes fail to
produce a stable exact-card win.

This work is inference-performance validation. Parameter-adjusted Qwen3.5
ratios compare active inference work and do not claim that RWKV model quality
exceeds Qwen3.5.

## Acceptance and fallback

Candidate screening uses same-process alternating A/B order. A route is
eligible for an exact RTX 4090 default only when the claimed shape has a stable
end-to-end improvement, prompt and cache-handoff cosine at least `0.9999`, exact
greedy-token parity, finite logits, and non-negative peak-VRAM behavior. A
noise-sized or order-sensitive result remains explicit telemetry.

The final evidence matrix covers dense FP16 Prefill and decode for B1/B8 at
P128/P512/P2048 and D128/D512 for the 0.4B/0.8B, 1.5B/2B, and 2.9B/4B RWKV /
Qwen3.5 pairs. Any 7.2B promotion is separately limited to the measured B8
shape. Card name, model shape, dtype, PyTorch feature availability, and
environment-disable flags must all fail closed; adjacent Ada products and
unlisted shapes retain their current policy.

Repository tests must cover policy isolation, explicit overrides, scoped
restoration of process-global FP16 accumulation, and benchmark/analyzer gates.
Exact-card artifacts record the commit, model hashes, runtime versions, raw
rows, summaries, and reproduction commands.
