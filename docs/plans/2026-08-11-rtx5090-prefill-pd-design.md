# RTX 5090 B1/B8 Prefill PD Design

> Historical design record. Implementation and final evidence are linked in
> [Final outcome](#final-outcome); this file is not an active roadmap.

## Goal and acceptance contract

Optimize the Hugging Face RWKV-7 prefill path on one exact RTX 5090 while
preserving all HF and recurrent-cache semantics. The comparison uses the
official Qwen3.5 checkpoints and the fused FLA/Triton reference route from the
repository benchmark. Each configured `(model pair, batch size, prompt length)`
cell must pass independently; a median cannot hide a regression.

The primary metric follows the parameter-adjusted calculation merged in PR
#107:

`prefill PD = RWKV prefill tok/s / Qwen prefill tok/s * RWKV active parameters / Qwen active parameters`

The initial matrix is dense fp16, B1/B8, prompt 128/512/2048, decode 128, and
prefill chunk size 512. Every Qwen row must prove its live FLA chunk/recurrent,
Triton convolution, and fused normalization bindings. Every promoted RWKV route
must retain finite logits, cache handoff, greedy-token alignment, and the
repository cosine threshold.

## Optimization sequence

Use three increasingly invasive lanes. First, run exact-shape A/B tests for
already implemented state-prep, stacked-RKV, shift-mix, sequence-FFN, fused
output, and BLAS routing. This is the lowest-risk lane because it only changes
Blackwell dispatch after end-to-end evidence. Second, test limited fp16
accumulation only on identified projection layers; this requires stronger
logit/greedy validation and is rejected if the numerical gate moves. Third,
only if the remaining gap cannot be closed by existing compiled boundaries,
profile the slow cell and implement one deeper fused boundary targeted at the
dominant per-layer bucket. Wrapper/cache micro-optimization is outside the
performance plan.

For each lane, compare the candidate against a repeated default-policy row on
the same resident GPU. Promote a setting only when the median improves and the
gain survives a confirmation run. Record exact software, model, flags, timing,
memory, and correctness metadata. Keep all new 5090 routes exact-shape gated so
V100, Ampere, Ada, and other Blackwell cards retain their current defaults.

## Delivery

The final change should contain only confirmed kernel-policy or kernel work,
tests for routing and fail-closed behavior, benchmark artifacts or compact
reproducible summaries, and updated 5090 documentation. Run local CPU tests and
remote exact-card correctness/performance gates before committing. Push the
isolated `btlqql/rtx5090-prefill-pd` branch with the active `yyqdbngt` GitHub
identity and open a repository-compliant pull request.

## Final outcome

Implemented and validated on 2026-08-11. The winning route combines exact-shape
CUDA graph dispatch with a scoped `allow_fp16_accumulation` prefill region; the
7.2B shapes additionally reuse the already compiled shift-mix, state-prep,
output-prep, and stacked-RKV boundaries. The switch is restored after every
call and fails closed outside FP16, the exact RTX 5090 shapes, and the default
single-GPU worker contract.

The final artifact is
[`../../bench/5090_g1i_qwen35_prefill_pd_20260811/README.md`](../../bench/5090_g1i_qwen35_prefill_pd_20260811/README.md).
It passes 24/24 parameter-adjusted prefill cells with minimum PD `1.028427x`,
24/24 full-FLA Qwen reference contracts, and 21/21 changed-route correctness
rows with minimum cosine `0.99999994` and exact greedy/cache handoff.
