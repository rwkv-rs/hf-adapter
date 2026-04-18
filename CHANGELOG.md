# Changelog

This file records user-visible and evidence-backed changes to the RWKV-7
Hugging Face adapter. Benchmark claims remain scoped to the exact hardware,
model, dtype, batch and sequence shape named by their linked artifact.

## Unreleased

### Hardware backends

- Integrated the optional Moore Threads MUSA backend with an exact-card legacy
  support boundary in [PR #87](https://github.com/rwkv-rs/hf-adapter/pull/87),
  contributed by `@KakaruHayate`.
- Integrated Huawei Ascend 910B3 Native HF, NPUGraph and W8 contracts in
  [PR #93](https://github.com/rwkv-rs/hf-adapter/pull/93).
- Integrated the MetaX C500/MXMACA Native HF backend in
  [PR #94](https://github.com/rwkv-rs/hf-adapter/pull/94).
- Integrated the Biren BR106M/SUPA BF16 backend in
  [PR #95](https://github.com/rwkv-rs/hf-adapter/pull/95), contributed by
  `@yyqdbngt`. This contributor is independent from Wang Yue.

### Performance

- Added the exact RTX 3090 latest-checkpoint B1/B8 dense-FP16 prefill lane.
  All 24 parameter-adjusted Qwen3.5 cells and all 15 FP16-accumulation
  prompt/cache-handoff correctness rows pass; unmeasured Ampere products and
  shapes remain on conservative routes.
- Added exact RTX 5070 Laptop Native/no-FLA routes for measured 0.4B/1.5B
  prefill and decode shapes. Raw recurrent, shape-gated norm/mix and B8 FP16
  state are promoted; projection/LoRA and sub-threshold launch probes remain
  disabled.
- Added exact Tesla V100 0.4B/1.5B B8 FP16 recurrent state. Opposite-order
  paired processes measure `1.0216x-1.0288x`, save
  `16.875-58.125 MiB`, and retain exact recorded greedy traces.

- Added exact-card RTX 4080 and V100 B8 decode tuning in
  [PR #100](https://github.com/rwkv-rs/hf-adapter/pull/100). The V100 WAVG
  launch improves paired 0.4B/1.5B/2.9B B8 decode by
  `1.0114x-1.0312x` while retaining greedy parity.
- Added an exact RTX 4080/B8 grouped W/A/V tensor-core projection route in
  [PR #101](https://github.com/rwkv-rs/hf-adapter/pull/101). The promoted
  0.4B/1.5B/2.9B medians are `1.1267x/1.0942x/1.0809x` the previous route,
  with exact first-step logits and greedy `4,608/4,608`.
- Added an exact RTX 4080 7.2B/B8 FP16 recurrent-state route in
  [PR #102](https://github.com/rwkv-rs/hf-adapter/pull/102). It records
  `344.39 tok/s`, `1.0301x` the FP32-state route, `-123.88 MiB` median peak
  allocation and greedy `12,288/12,288`.

### Maintenance and documentation

- Made offline regression independent of accelerator availability in
  [PR #97](https://github.com/rwkv-rs/hf-adapter/pull/97).
- Closed and synchronized the v0.6 HF milestone documentation in
  [PR #99](https://github.com/rwkv-rs/hf-adapter/pull/99).
- Added a current evidence index, project summary, updated contributor
  attribution and explicit separation of completed work from post-release
  expansion projects.

## [v0.6.0](https://github.com/rwkv-rs/hf-adapter/releases/tag/v0.6.0) - 2026-07-24

- Completed the declared HF adapter milestone: official checkpoint conversion,
  Transformers Auto classes, generation and recurrent cache, PEFT/Trainer/TRL,
  DeepSpeed ZeRO smoke/resume, dense HF PP/TP, W8/W4 functionality, speculative
  decoding and profile-bounded production evidence.
- Promoted the Native/no-FLA model as the canonical HF implementation while
  retaining FLA as an explicit reference backend.
- Published exact-card performance, correctness and memory evidence across the
  supported NVIDIA, AMD and Apple profiles available at release time.

Canonical current status is maintained in [`HF_STATUS.md`](HF_STATUS.md);
numeric results are maintained in [`BENCHMARK.md`](BENCHMARK.md).
