# RWKV-7 HF adapter TODO

Only **unfinished, actionable HF-adapter work** belongs here. Completed
experiments and historical plans belong in benchmark artifacts or Git history.
Native vLLM/SGLang scheduler work is out of scope for this file.

Last updated: **2026-07-28**. Audited against upstream main commit
`22237b6b66ec492fefba5669cb3fa6f5bb518007`.

## Scope and current boundary

The current HF milestone is complete and the repository is suitable for a
public `0.6.0` HF-adapter release. Universal production scope remains
`PARTIAL`: the remaining work is cross-card and cross-shape performance,
full-memory quantized speed, broader task quality, missing hardware, and
broader distributed-training evidence.

This audit already includes the July 25 native-model module split, nested
remote-code manifest support, and card-gated external quant-prefill graphs.
Those merged changes are not repeated as TODO items.

PP/TP are closed for the declared dense-inference HF scope. Two-V100 manual
layer-split `device_map` generation matches the single-device reference and
keeps recurrent state on the owning stages. Separately, Transformers-native
`tp_plan="auto"` now shards vocabulary, attention, FFN, and output matrices;
the two-V100 gate proves shard shapes, logits/generation parity, rank agreement,
and `0.52031x/0.611611x` B1/B8 local peak-VRAM ratios. Recurrent WKV state
remains explicitly replicated, and quantized TP plus TP training remain
separate evidence lanes.
See [`docs/integrations/HF_TENSOR_PARALLEL.md`](docs/integrations/HF_TENSOR_PARALLEL.md).
Native serving-engine executors remain separate projects.

The audit also found several other completed lanes that must not be reopened as
generic TODOs:

- the accepted RTX 5090 full-MATH500 and compression-alignment gates;
- the promoted Apple M5 MLX pairs/shapes for raw peak-memory comparisons,
  long-context/chunk handoff, sustained multi-session checks, and the MLX
  policy/module split with fallback telemetry;
- the exact RTX 5090 native `train_temp` tensor, convergence, long-run, resume,
  and steady-memory lane (only broader sizes/distributed reproduction remains);
- the V100 CUDA target/draft speculative artifact with speed, acceptance,
  correction, memory, and target-greedy equality;
- end-user PEFT/LoRA/SFT/DPO/GRPO commands backed by deterministic tiny datasets;
- scheduled clean-install CPU and Apple CI, plus cross-card policy-isolation
  regression tests;
- the versioned experimental-backend deprecation window, centrally enforced
  pytest marker policy, and minimum/current Transformers-PEFT-TRL CI lanes; and
- selected/hybrid B8 W8/W4 speed lanes on RTX 3090 and RTX 4090. These do not
  close the separate full-memory quantized-speed target below.

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
- [ ] Close the *full-memory* (not already accepted selected/hybrid B8)
      all-phase speed lane on RTX 3090, RTX 4090, and at least one Ampere
      professional card without inheriting schedules across cards.

Acceptance: every promoted profile lowers physical model footprint, preserves
the declared cosine/same-next gates, records route/fallback provenance plus
physical footprint and peak VRAM, and is no slower than the matching fp16 row
for all named prefill/decode cells. See
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
- [ ] Extend the current V100 P1 floor and existing small-model/B8 P3 cells to
      larger-model P2/P3 prefill/decode rows with explicit host RAM and VRAM
      ceilings.

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

Acceptance: raw throughput, active-parameter-normalized work, correctness,
physical footprint, and peak VRAM remain separate gates. A Torch-fallback Qwen
row is never a full-FLA reference.

### 4. Missing exact-card hardware

- [ ] H100/Hopper: bf16, large-model, quant, batch, cache, training, and
      same-card performance rows.
- [x] AMD gfx1100/ROCm 7.2.1: fully native HF load/generate, PEFT, recurrent
      cache/chunked prefill, bf16 Trainer and B1/B2/B4/B8 baseline on a real GPU.
- [x] AMD gfx1100: exact-architecture fused decode and output-head MM8/MM4
      B1/B2/B4/B8 speed/quality rows through 13.3B; all 40 quant cells pass.
- [ ] AMD/ROCm: fused prefill, full-model fused W8/W4, MI-series, longer
      training and same-card official/Albatross performance.
- [ ] Other Turing/RTX 20 products: validate independently and do not inherit
      Tesla T4 prefill or DP4A quant routing from `sm_75` alone.
- [ ] Moore Threads MUSA: exact-card MTT S70 standalone kernel parity,
      HF load/forward/cache/generate, 64-token eager/WKV equality, autograd
      fallback, B1/B2 smoke and one paired B1/T128 performance row now pass.
      Close real PEFT/LoRA package integration, broader models/shapes and deeper
      recurrent-state gates; do not inherit bf16, graph, Triton, quantization,
      multi-device, training-kernel, CUDA or ROCm behavior.
- [ ] Add exact-card evidence for additional RTX 50-series and constrained
      laptop/low-memory devices.
- [ ] Reproduce the promoted Apple results on M1–M4 and Pro/Max/Ultra variants.

Use [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md) and the hardware report
template in [`CONTRIBUTING.md`](CONTRIBUTING.md).

### 5. Model quality and long-context evaluation

- [ ] Beyond the accepted MATH500 math gate, add reproducible instruction,
      reasoning, code, multilingual/Chinese, and long-context evaluations for
      comparable RWKV-7 and Qwen3.5 checkpoints.
- [ ] Extend quantized quality beyond short cosine/greedy gates with named
      datasets, prompts, decoding parameters, seeds, and retained raw outputs.
- [ ] Generalize the existing functional long-context/chunk-handoff checks into
      cross-card release gates for state stability and task-quality drift across
      fp16, W8, and W4.

Acceptance: model-quality results remain separate from engine throughput and
active-parameter-normalized performance claims.

## P1 — Training and distributed closure

### 6. Longer training evidence

- [ ] Extend 0.4B/1.5B/2.9B/7.2B SFT, DPO, and GRPO beyond compatibility
      smoke steps with loss, throughput, memory, and checkpoint evidence.
- [ ] Expand ZeRO-3 resume to larger models and more card combinations,
      including distributed optimizer/scheduler and RNG continuity.
- [ ] Add H100 and longer/larger AMD bf16 training matrices.
- [ ] Reproduce the accepted native train_temp convergence and resume contract
      on broader model sizes and distributed configurations.

### 7. Apple MLX and CoreML completion

- [ ] Add retained formal response-quality scoring to the accepted 1.5B-vs-2B
      performance pair, then extend the common rubric to 4B+ pairs.
- [ ] Close CoreML INT4/LUT4 quality and confirm ANE placement and occupancy.
- [ ] Make broad/full-memory W8/W4 fp16-or-faster on the already validated
      long-context and multi-session/batched shapes.

See [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md)
for the promoted M5 boundary.

## P2 — Packaging and maintenance

### 8. Hub, release, and CI experience

- [ ] Publish a clean Hub example with conversion provenance and checkpoint
      checksums.
- [ ] Add a scheduled clean-install CUDA job; the weekly clean-install CPU and
      Apple jobs already exist.

### 9. Architecture and remote-code maintenance

- [ ] Continue splitting the remaining `native_jit.py`, `modeling_rwkv7.py`,
      and MLX runtime/kernel monoliths behind the stable `native_model.py`
      facade while preserving converted `auto_map`, parameter names, and
      old-model loading.
- [ ] Prove any nested runtime import layout offline across the supported
      Transformers range before moving remote-code dependencies out of the
      current flat namespace.
- [ ] Reduce duplicated benchmark/session utilities only after preserving
      artifact readers and historical reproduction commands.

Acceptance: card-specific routing remains isolated in the policy layer, and
every policy/kernel-default change retains the existing cross-card regression
suite.

### 10. Speculative decoding

- [ ] Extend beyond the accepted V100 0.4B-target/0.1B-draft and Apple M5 lanes
      to multiple CUDA/Apple draft sizes, longer shapes, acceptance/rejection
      behavior, cache handoff, memory, and repeatable end-to-end speed gains.

Acceptance: exact target-distribution and target-greedy correctness gates remain
mandatory. DFlash and serving-engine scheduler integration stay in separate
projects and are not HF-adapter TODOs.

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
