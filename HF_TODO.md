# RWKV-7 HF adapter TODO

Only **unfinished, actionable HF-adapter work** belongs here. Completed
experiments and historical plans belong in benchmark artifacts or Git history.
Native vLLM/SGLang scheduler work is out of scope for this file.

Last updated: **2026-07-26**. Audited against upstream main commit
`4d1de1733b90e99eaf9c104eb73639eb221e3ad2`.

## Scope and current boundary

The current HF milestone is complete and the repository is suitable for a
public `0.6.0` HF-adapter release. Universal production scope remains
`PARTIAL`: the remaining work is cross-card and cross-shape performance,
full-memory quantized speed, broader task quality, missing hardware, and
production distributed execution.

This audit already includes the July 25 native-model module split, nested
remote-code manifest support, and card-gated external quant-prefill graphs.
Those merged changes are not repeated as TODO items.

Do not convert the unchecked roadmap, section count, or status-row count into
a repository-wide completion percentage. Report completion only for a named
scope. Current status and promoted evidence live in
[`HF_STATUS.md`](HF_STATUS.md), [`BENCHMARK.md`](BENCHMARK.md), and
[`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md).

## P0 — Universal production gaps

### 1. Full-memory W8/W4 performance

Goal: retain the large footprint reduction of broad projection quantization
while remaining fp16-or-faster for every promoted prefill and decode shape.

- [ ] Fuse broad quantized R/K/V/output and FFN projection work, including the
      required activation and residual epilogues, instead of relying on
      output-head-only speed policies.
- [ ] Close the Tesla T4 full-model lane: current DP4A W8/W4 reduces footprint
      and wins B1 decode, but full-model prefill and small-model B4/B8 decode
      remain below fp16.
- [ ] Close V100 full-memory prefill. The head-only group-256 speed profile is
      accepted, while the broad-memory path remains a separate incomplete lane.
- [ ] Validate broad-memory all-phase speed on RTX 3090, RTX 4090, and at least
      one Ampere professional card without inheriting schedules across cards.
- [ ] Preserve paired fp16 timing, physical footprint, peak VRAM, cosine,
      same-next, route-provenance, and fallback gates in every promoted row.

Acceptance: every promoted profile lowers physical model footprint, preserves
the declared quality gates, and is no slower than the matching fp16 row for all
named prefill/decode cells. See
[`docs/QUANTIZATION.md`](docs/QUANTIZATION.md).

### 2. Final RWKV-LM and Albatross matrices

- [ ] Close the Tesla T4 gap across 0.1B–2.9B. Current same-card ratios are
      `0.4888x–0.8649x` for native-graph decode and `0.5385x–0.7671x` for
      B1/T512 fused prefill.
- [ ] Run a fresh same-card, same-session RTX 5090 Albatross matrix using the
      current native-default code and current official checkpoint set.
- [ ] Add current g1h B1/B2/B4 matrices for RTX 3090 and RTX 4090; the promoted
      broad matrices on those cards are currently bsz8-scoped.
- [ ] Recheck the RTX 4090 prompt-512 historical high-water reference under the
      current timing and cache policy.
- [ ] Extend beyond V100 P1 to larger-model P2/P3 prefill/decode rows with
      explicit host RAM and VRAM ceilings.

Acceptance: candidate and reference must share the exact card, checkpoint,
dtype, batch, prompt/decode lengths, cache policy, timing method, and process
state. Correctness and memory gates remain mandatory alongside throughput.

### 3. Broader optimized-Qwen exact-card coverage

- [ ] Extend RTX 5070 full-FLA coverage from bsz8 to bsz1/2/4 and add the
      larger 4B/9B comparison pairs.
- [ ] Extend the V100 optimized-Qwen matrix beyond prompt512/decode64 and the
      current 1.5B/2B pair.
- [ ] Add fail-closed optimized-Qwen matrices on Ampere professional cards and
      H100/Hopper.
- [ ] Keep raw throughput, active-parameter-normalized work, correctness,
      physical footprint, and peak VRAM as separate gates; never promote a
      Torch-fallback Qwen row as a full-FLA reference.

### 4. Missing exact-card hardware

- [ ] H100/Hopper: bf16, large-model, quant, batch, cache, training, and
      same-card performance rows.
- [ ] AMD/ROCm: native/no-FLA load/generate, recurrent cache, training,
      quantization, and performance on a real ROCm device.
- [ ] Other Turing/RTX 20 products: validate independently and do not inherit
      Tesla T4 prefill or DP4A quant routing from `sm_75` alone.
- [ ] Add exact-card evidence for additional RTX 50-series and constrained
      laptop/low-memory devices.
- [ ] Reproduce the promoted Apple results on M1–M4 and Pro/Max/Ultra variants.

Use [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md) and the hardware report
template in [`CONTRIBUTING.md`](CONTRIBUTING.md).

### 5. Model quality and long-context evaluation

- [ ] Add reproducible instruction, reasoning, math, code, multilingual,
      Chinese, and long-context evaluations for comparable RWKV-7 and Qwen3.5
      checkpoints.
- [ ] Separate model-quality results from engine throughput and
      active-parameter-normalized performance claims.
- [ ] Extend quantized quality beyond short cosine/greedy gates with named
      datasets, prompts, decoding parameters, seeds, and retained raw outputs.
- [ ] Define release gates for long-context state stability, chunk handoff, and
      quality drift across fp16, W8, and W4.

## P1 — Training and distributed closure

### 6. Longer training evidence

- [ ] Extend 0.4B/1.5B/2.9B/7.2B SFT, DPO, and GRPO beyond compatibility
      smoke steps with loss, throughput, memory, and checkpoint evidence.
- [ ] Expand ZeRO-3 resume to larger models and more card combinations,
      including distributed optimizer/scheduler and RNG continuity.
- [ ] Add H100 and AMD training matrices.
- [ ] Reproduce the accepted native train_temp convergence and resume contract
      on broader model sizes and distributed configurations.

### 7. PP/TP and multi-device behavior

- [ ] Define the exact HF-scope PP and TP acceptance contract.
- [ ] Promote multi-device generation beyond `device_map` smoke.
- [ ] Add correctness, recurrent-state ownership, failure recovery, memory,
      and throughput gates for real TP/PP paths.
- [ ] Document unsupported combinations and fail explicitly instead of
      silently falling back to a different execution mode.

### 8. Apple MLX and CoreML completion

- [ ] Validate Qwen3.5 2B/4B+ pairs with the common formal quality rubric.
- [ ] Capture true peak-to-peak memory rather than loaded-memory proxies.
- [ ] Close CoreML INT4/LUT4 quality and confirm ANE placement and occupancy.
- [ ] Stabilize full-memory W8/W4 speed at long contexts and batch sizes above
      one.
- [ ] Convert remaining guarded experiments into maintainable policy tables
      with explicit fallback telemetry.

See [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md)
for the promoted M5 boundary.

## P2 — Packaging and maintenance

### 9. Hub, release, and CI experience

- [ ] Publish a clean Hub example with conversion provenance and checkpoint
      checksums.
- [ ] Add end-user SFT/LoRA/DPO examples with tiny reproducible datasets.
- [ ] Test a supported Transformers/PEFT/TRL version range in CI.
- [ ] Add scheduled clean-install CPU plus optional CUDA and Apple jobs.
- [ ] Document migration and deprecation policy for experimental backends.

### 10. Architecture and remote-code maintenance

- [ ] Continue the runtime/kernel ownership split while keeping
      `native_model.py`, converted `auto_map`, parameter names, and old-model
      loading backward compatible.
- [ ] Prove any nested runtime import layout offline across the supported
      Transformers range before moving remote-code dependencies out of the
      current flat namespace.
- [ ] Add and enforce pytest markers for CPU, CUDA, card families, slow tests,
      and model-required suites before reorganizing tests or scripts.
- [ ] Reduce duplicated benchmark/session utilities only after preserving
      artifact readers and historical reproduction commands.
- [ ] Keep card-specific routing isolated and retain cross-card regression
      tests whenever a policy or kernel default changes.

### 11. Speculative decoding

- [ ] Add CUDA target/draft end-to-end speed and acceptance artifacts.
- [ ] Validate multiple draft sizes, acceptance/rejection rates, cache handoff,
      and memory behavior.
- [ ] Preserve exact target-distribution and target-greedy correctness gates.
- [ ] Leave DFlash and serving-engine scheduler integration to their own
      projects.

## PR completion checklist

This is a per-PR template, not a list of outstanding project tasks:

- Exact hardware/runtime/model/dtype recorded.
- Reproduction command included.
- Raw JSONL/log and concise README included.
- Correctness, speed, memory, and route provenance reported together.
- Negative or partial results described honestly.
- Canonical status/benchmark/TODO documents updated only when status changes.
- `python tests/test_markdown_links.py` passes.
- Relevant unit, smoke, and cross-card isolation tests pass.
