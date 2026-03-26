# RWKV-7 HF adapter TODO

This file records **non-blocking post-release expansion projects**. Completed
HF deliverables and dated experiments belong in `HF_STATUS.md`,
`BENCHMARK.md`, or the immutable evidence directories under `bench/`.
Native vLLM/SGLang scheduler work remains outside this repository.

Last updated: **2026-08-07**. Audited against `main` commit
`2fe20a322ffc9ffb363300044dbb74fc55d48c33` and the published `v0.6.0`
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

## Post-release expansion projects

The projects below are useful future work, but none blocks or downgrades the
completed `v0.6.0` HF deliverable.

### Wider performance matrices

- Extend full-memory fused W8/W4 beyond the already promoted card-local
  profiles, especially T4 full-model, V100 broad-prefill, and additional
  Ampere/Ada shapes.
- Add same-session RWKV-LM/Albatross rows for more model, batch, prompt, and
  decode combinations without replacing existing accepted profiles.
- Expand optimized-Qwen full-FLA comparisons to more exact cards and model
  pairs while keeping raw throughput, active-parameter-normalized work,
  quality, physical footprint, and peak VRAM separate.

### Additional hardware products

- Add H100/Hopper, MI-series, more RTX 50 products, other Turing cards, and
  Apple M1-M4/Pro/Max/Ultra as independent exact-product evidence.
- Rerun Ascend 910B3, Biren BR106M, MetaX C500, and later MUSA products against
  future main revisions when hardware is available. The currently accepted
  integration scope remains pinned to its documented standalone evidence.

### Broader training and quality

- Extend large-model SFT/DPO/GRPO, ZeRO-3 resume, distributed convergence, and
  multi-day soak coverage beyond the published compatibility matrix.
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
