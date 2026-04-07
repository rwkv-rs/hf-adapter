# RWKV-7 HF adapter TODO

This file records **non-blocking post-release expansion projects**. Completed
HF deliverables and dated experiments belong in `HF_STATUS.md`,
`BENCHMARK.md`, or the immutable evidence directories under `bench/`.
Native vLLM/SGLang scheduler work remains outside this repository.

Last updated: **2026-08-09**. Audited against `main` commit
`045bac1b769240facd290e1ac8232e8b1ca39778` and the published `v0.6.0`
release.

## Scope and current boundary

The current HF milestone is complete. The public `v0.6.0` adapter release is
also complete for its declared, evidence-backed scope. There are **no remaining
blocking items** for that milestone.

Completion is profile-based rather than unbounded: a capability is accepted
for the models, cards, dtypes, batches, shapes, and software stacks named by
its evidence. Adding another card or benchmark shape extends the matrix; it
does not reopen the completed release.

Dense HF inference PP/TP is closed for the declared scope. Native vLLM/SGLang
executors, quantized TP, TP training, and scheduler-level speculative decoding
remain separate projects rather than missing HF adapter features.

## Accepted HF deliverables

| Public requirement | Completion record |
|---|---|
| RWKV-LM / Albatross correctness, speed and memory | Complete for the promoted exact-card profiles in `BENCHMARK.md`; every row keeps its own model, batch, shape, dtype, memory, correctness, and reference boundary |
| Transformers adapter | Complete: conversion, Auto classes, generation, recurrent cache, masks, labels/loss, save/reload, remote code, and Native/no-FLA default |
| PEFT and RL ecosystem | Complete for the published compatibility and exact-training matrix: LoRA, Trainer, SFT, DPO, GRPO, gradient checkpointing, checkpoint resume, and accepted `train_temp` lanes |
| Serving-like HF primitives | Complete in HF scope: dynamic state selection/reorder, chunked prefill, state offload/restore, telemetry, and cache handoff |
| Hardware support | Complete for the declared support policy and recorded exact-card matrix, including NVIDIA, AMD, Apple, Ascend, Biren, MetaX, MUSA, and CPU fallback boundaries |
| W8/W4 inference | Complete for functional loading/generation, reduced physical footprint, quality gates, and promoted exact-card speed profiles |
| PP/TP and ZeRO | Complete for dense HF inference PP/TP and the published ZeRO-2/3 smoke/resume matrix |
| Initial speculative decoding | Complete as an experimental HF/Apple capability with target/draft correctness gates |

Canonical evidence and the precise limits of each accepted profile live in
[`HF_STATUS.md`](HF_STATUS.md), [`BENCHMARK.md`](BENCHMARK.md),
[`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md), and
[`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md).

## Completed current-main expansions

The following work landed after the `v0.6.0` release and is therefore no
longer an open item:

- **RTX 4080/V100 B8 decode policy:** exact-card grouped decode and V100 WAVG
  launch tuning landed with paired speed and greedy gates in
  [`bench/4080_v100_decode_tuning_20260808/`](bench/4080_v100_decode_tuning_20260808/README.md).
- **RTX 4080 B8 grouped projections:** 0.4B/1.5B/2.9B W/A/V tensor-core BMM
  routes pass repeated A/B timing, exact first-step logits and greedy
  `4,608/4,608`; see
  [`bench/4080_b8_projection_bmm_20260809/`](bench/4080_b8_projection_bmm_20260809/README.md).
- **RTX 4080 7.2B/B8 dense capacity and state traffic:** the fail-closed
  FP16-state route reaches `344.39 tok/s`, `1.0301x` its FP32-state route,
  saves `123.88 MiB`, and matches greedy `12,288/12,288`; see
  [`bench/4080_7p2b_fp16_state_20260809/`](bench/4080_7p2b_fp16_state_20260809/README.md).
- **Domestic accelerator integration:** Ascend 910B3, Biren BR106M, MetaX
  C500 and Moore Threads MUSA have repository-integrated, fail-closed support
  boundaries. Their broader performance matrices remain optional extensions,
  not missing HF adapter APIs.

## Post-release expansion projects

The projects below are useful future work, but none blocks or downgrades the
completed `v0.6.0` HF deliverable.

### Wider performance evidence

- Close only the still-unmeasured **full-model** W8/W4 cells, especially T4
  all-phase quant, broader V100 prefill and RTX 4080 7.2B full-model quant.
  Existing head-only, packed-MM4, Marlin and MLX speed lanes remain accepted.
- Add same-session RWKV-LM/Albatross rows only where an exact card/model/batch
  profile lacks them. V100, RTX 4090 and RTX 5090 already have promoted
  reference lanes; the principal current Ada hole is RTX 4080 7.2B/B8 against
  a same-card reference rather than its now-closed internal FP32-state route.
- Expand optimized-Qwen full-FLA comparisons only to new exact cards or model
  pairs. RTX 3090/4080/4090/5070/5090 already have promoted model-pair evidence
  for their named scopes.

### Additional hardware products

- Add H100/Hopper, MI-series, more RTX 50 products, other Turing cards, and
  Apple M1-M4/Pro/Max/Ultra as independent exact-product evidence. These are
  new-product validations, not unresolved support on already promoted cards.
- Rerun Ascend 910B3, Biren BR106M, MetaX C500, and later MUSA products against
  future main revisions when hardware is available. The currently accepted
  integration scope remains pinned to its documented standalone evidence.

### Broader training and quality

- Extend large-model SFT/DPO/GRPO, ZeRO-3 resume, distributed convergence, and
  multi-day soak coverage beyond the already-passing Trainer/TRL/ZeRO smoke,
  checkpoint-resume and RTX 5090 5,000-step matrices.
- Add more instruction, code, math, multilingual, quantized-quality, and long-
  context datasets. These extend quality evidence and do not change API
  compatibility acceptance.
- Extend Apple CoreML/ANE and MLX response-quality matrices beyond the promoted
  M5 profiles.

### Packaging and maintenance

- Publish additional Hub checkpoints with conversion provenance when model
  redistribution permits it.
- Add optional scheduled GPU clean-install lanes as runner capacity allows;
  the required CPU and HF ecosystem CI lanes are already active.
- Continue splitting large internal modules and deduplicating benchmark tools
  without changing the stable Auto*, checkpoint, cache, or remote-code ABI.
- Expand speculative decoding to more target/draft pairs and serving engines
  as a separate optimization project.

## Completion reporting rule

Do not convert the number of post-release ideas into a completion percentage.
Report the named scope:

- `RWKV-7 HF adapter v0.6.0`: **COMPLETE**;
- an exact card/model/profile: use its status in the hardware matrix and
  benchmark artifact;
- native vLLM/SGLang/DFlash: separate project status.

An accepted HF capability is reopened only if a code change invalidates its
contract or regression gate, not because a new GPU or benchmark shape exists.

## PR completion checklist

Every future enhancement PR must state:

1. exact scope and environment;
2. correctness and fallback behavior;
3. prefill and decode results where performance is claimed;
4. physical footprint and peak memory where quantization is claimed;
5. raw evidence paths and reproduction commands;
6. which existing accepted profiles were rerun for regression isolation.
