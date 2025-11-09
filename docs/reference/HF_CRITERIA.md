# RWKV-7 HF adapter validation criteria

Low-level acceptance rules. Current pass/partial status is maintained in
[`../ACCEPTANCE.md`](../ACCEPTANCE.md); promoted numbers are in
[`../../BENCHMARK.md`](../../BENCHMARK.md).

## Required gates

| Area | Minimum evidence |
|---|---|
| Correctness | Official/HF logits or top-k/cosine alignment, deterministic greedy window and save/reload roundtrip |
| Transformers API | Auto classes, forward, labels/loss, masks, `generate(use_cache=True)` and cache reorder/select behavior |
| PEFT/TRL | Non-finite checks, non-zero gradients/updates, adapter lifecycle, Trainer/SFT/DPO/GRPO smoke |
| Cache | recurrent state select/reorder/drop/compact, chunked-prefill parity, offload/restore and telemetry |
| Performance | same device/model/checkpoint/dtype/shape; warmup; repeated or paired timing; prefill/decode/batch and peak memory |
| Quantization | lower footprint, finite logits, cosine/greedy gate, explicit policy/replaced modules and controlled fp16 comparison |
| Distributed | exact card count/config, base run, checkpoint resume and parameter/loss continuity evidence |
| Large model | conversion provenance/checksum, load/forward/generate and peak memory |
| Speculative | target-oracle correctness, acceptance/rejection telemetry and end-to-end speed |

## Promotion levels

- **Smoke:** proves the path executes.
- **Validated:** covers a meaningful compatibility or training matrix.
- **Production-close:** includes reproducible correctness, performance, memory
  and fail-closed regression gates for a declared scope.

A narrow pass must state its scope. It must not be generalized to every model,
batch, dtype or card family.

## Comparison rules

1. Never compare incompatible batch/prompt/decode shapes without labeling it.
2. Keep current-session and historical high-water references separately.
3. Prefer same-process paired fp16 timing for quant speed.
4. Report aggregate batch throughput separately from per-sequence latency.
5. MATH500 claims retain seed, sampling settings, task count, rollout and
   accuracy gates; speed alone cannot pass.
6. Card-specific tuning belongs behind policy/dispatch and requires fallback
   tests for older architectures.
7. Negative and partial experiments stay in dated artifacts to prevent repeated
   dead ends, but are not promoted to the current benchmark summary.

## Required files for promoted evidence

- `bench/<topic>_<hardware>_<date>/README.md`
- raw JSONL/JSON and relevant logs
- exact environment and commands
- concise result table and explicit limitations
- tests or fail-closed gate command
- update to `bench/INDEX.md`, then canonical status documents if the accepted
  state changed

## Current largest gaps

The active gaps are maintained only in [`../../HF_TODO.md`](../../HF_TODO.md):
full-memory fused W8/W4, broader Albatross P2/P3, missing hardware, longer
training/ZeRO-3 resume, Apple family/CoreML completion and production PP/TP.
