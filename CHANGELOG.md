# Changelog

This file records user-visible and evidence-backed changes to the RWKV-7
Hugging Face adapter. Benchmark claims remain scoped to the exact hardware,
model, dtype, batch and sequence shape named by their linked artifact.

## Unreleased

### Documentation

- Unified ordinary-user entrypoints around the published `rwkv7-hf==0.8.0`
  package, public 0.1B model, doctor, exact-kernel recommendation, and
  one-command smoke report. Source checkout and `.pth` conversion are now
  explicitly separated as developer/publisher workflows.
- Synchronized the six Hugging Face model cards and thin-entrypoint install
  hints with the current runtime while preserving their immutable v0.7.0
  conversion manifests and release tags.

### Conversion

- Made package-backed `thin` Hugging Face entrypoints the default converter
  output, matching the published model repositories. The explicit
  `--adapter-layout bundled` mode retains self-contained runtime snapshots for
  offline and archival workflows. Thin wrapper classes remain thin after model
  and tokenizer `save_pretrained()` round trips.

## [v0.8.0](https://github.com/rwkv-rs/hf-adapter/releases/tag/v0.8.0) - 2026-08-20

### Prebuilt CUDA kernel distribution

- Added the versioned `rwkv7-kernel-package-v1` manifest and exact runtime
  compatibility checks for Python ABI, platform, PyTorch, CUDA, C++ ABI, GPU
  architecture, and adapter version.
- Added the stable optional `rwkv7-kernels` distribution, reproducible
  exact-architecture wheel builder, wheel inspector, SHA256 release index, and
  two initial validated lanes: CPython 3.11 / CUDA 12.4 / Torch 2.5 / SM70 and
  CPython 3.11 / CUDA 12.4 / Torch 2.6 / SM89.
- Native CUDA operators now select a compatible prebuilt extension before JIT
  compilation and preserve the portable fallback. `RWKV7_KERNELS_MODE`
  provides explicit `auto`, `prebuilt`, `jit`, and `portable` policies.

### User validation and diagnostics

- Extended `rwkv7-hf-doctor` with per-device prebuilt-package compatibility,
  selected build lanes, and strict prebuilt readiness.
- Added `rwkv7-hf-kernels` for status, exact-build recommendation, release-index
  inspection, and hash-pinned wheel installation.
- Added `rwkv7-hf-smoke` for a one-command public-model load, prefill, greedy
  decode, finite-logit check, timing, memory, and runtime-backend report.
- Added exact-card release validation for V100/SM70 and RTX 4080/SM89 plus a
  self-hosted GitHub Actions workflow for rebuilding and attaching verified
  wheels to a release.

## [v0.7.1](https://github.com/rwkv-rs/hf-adapter/releases/tag/v0.7.1) - 2026-08-19

### Runtime diagnostics

- Added `python -m rwkv7_hf.doctor` and the `rwkv7-hf-doctor` console command
  to report installed runtime versions, visible accelerator profiles, CUDA and
  Triton build prerequisites, cache paths, and exact-card kernel-policy
  candidates without compiling kernels or downloading model weights.
- Added JSON output for reproducible hardware and policy issue reports.

### Distribution and published models

- Published the complete RWKV7-G1 FP16 Transformers family (0.1B, 0.4B,
  1.5B, 2.9B, 7.2B, and 13.3B) as independently loadable Hugging Face model
  repositories using the pinned `rwkv7-hf==0.7.0` thin runtime.
- Added the public `RWKV7-G1 Transformers` Hugging Face Collection, model and
  memory matrix, direct Hub quick start, and a one-command release verifier.
- Standardized the 7.2B and 13.3B repositories on bounded Safetensors shards
  and retained source/output SHA256 provenance in each conversion manifest.

## [v0.7.0](https://github.com/rwkv-rs/hf-adapter/releases/tag/v0.7.0) - 2026-08-12

### Distribution

- Published the first PyPI distribution as `rwkv7-hf`; the Python import
  remains `rwkv7_hf` and the supported runtime range remains Python 3.10-3.12,
  Transformers `>=5.12.1,<6`, PEFT `>=0.19.1,<1`, and TRL `>=1.7,<2`.
- Added a GitHub Trusted Publishing release workflow with isolated build,
  metadata validation, artifact transfer, and OIDC-based PyPI upload.

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

- Added the exact RTX 3090 latest-checkpoint B1/B8 dense-FP16 prefill lane and
  its max-performance follow-up. All 24 parameter-adjusted Qwen3.5 cells pass
  at minimum/median `1.227477x/1.467758x`, and all 25 FP16-accumulation
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
- Closed the RTX 4080 latest-checkpoint dense-FP16 matrix: all `36/36`
  parameter-adjusted Prefill and `36/36` Decode cells exceed Qwen3.5, with
  global minima `1.068520x/1.140700x`.
- Added exact RTX 4090 block-scoped FP16 accumulation, grouped W/A/V BMM and
  1.5B/B1/P2048 self-chunk routing. The final full-FLA/Triton-conv Qwen3.5
  comparison passes all `36/36` adjusted Prefill and `36/36` Decode cells at
  minima `1.108265x/4.158943x`.
- Promoted the latest RTX 5090 exact-card Prefill routes and strict Qwen3.5
  matrix: all `24/24` parameter-adjusted Prefill cells pass, with graph/eager
  continuation correctness and conservative 7.2B memory routing retained.

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
