# RWKV-7 HF adapter TODO

Only **unfinished, actionable HF-adapter work** belongs here. Completed
experiments and historical plans belong in benchmark artifacts or Git history.
Native vLLM/SGLang scheduler work is out of scope for this file.

Last updated: **2026-07-20**.

## Current milestone — COMPLETE

The active V100/full-FLA/documentation milestone is complete:

- V100 0.1B/0.4B/1.5B dense Albatross P1 and native W8/W4 speed lane;
- V100 RWKV-7 1.5B versus full-FLA Qwen3.5-2B target-only B1/B8 raw and
  active-parameter work gates;
- RTX 4090 small-model and 7.2B bsz8 promoted matrices;
- RTX 5070 full-FLA 1.5B/2B bsz8 promoted matrix;
- RTX 5090 full-FLA Qwen B1/B8 8/8 matrix, production BN/TN W4
  1.5B/2.9B/7.2B/13.3B B1/B8 matrix, and latest g1h 13.3B boundary;
- PEFT/Trainer/TRL and current ZeRO-2/3 smoke/resume matrix;
- canonical documentation refresh and full Markdown freshness audit.

The unchecked items below are the **project-wide remaining roadmap** needed for
universal “all cards/all shapes/upstream” claims. They are not unfinished work
from the completed current milestone.

Do not convert the unchecked roadmap, checkbox counts, or status-row counts
into a repository-wide completion percentage. Completion is reported only for
an explicitly named milestone; universal production scope remains `PARTIAL`
until its own acceptance gates close.

## P0 — Remaining universal production gaps

### 1. Full-memory W8/W4 performance

Goal: obtain the large memory reduction of broad projection quantization while
remaining fp16-or-faster across representative batch/prompt/decode shapes.

- [ ] Fuse quantized R/K/V/output and FFN projections instead of relying on
      selected-module speed policy.
- [x] Close RTX 5090 g1h 1.5B/2.9B/7.2B/13.3B B1/B8 W4 prefill and decode
      for the exact measured prompt128/decode128 lane. Exact-model quality
      profiles reach `0.5298x–0.6250x` BF16 footprint and minimum
      `1.0010x/1.1854x` prefill/decode while preserving cosine `>=0.9995` and
      same-next 8/8. The group-128 per-launch BN/TN audit passes 280/280 rows;
      group-32 experimental coverage passes another 48/48. Evidence:
      [`bench/5090_bntn_all_models_20260716/`](bench/5090_bntn_all_models_20260716/README.md).
- [ ] Add all-phase fused quant prefill for the remaining cards/shapes;
      decode-only wins are insufficient.
- [ ] Close the Tesla T4 full-model lane. Exact-card DP4A W8/W4 now reduces
      footprint to `0.5291x–0.6331x` / `0.3004x–0.4542x` and wins every B1
      decode row, but prefill is `0.1272x–0.6984x` and small-model B4/B8 decode
      remains below fp16. The separate head-only speed lane passes 26/26 decode
      cells at `>=1.0207x`, but is not a substitute for broad memory closure.
- [ ] Validate the same large-payload contract on V100, 4090 and at least one
      Ampere professional card; RTX 5090 exact-lane evidence is complete.
- [ ] Preserve cosine, same-next, footprint and paired timing gates.
- [x] Add 0.4B/1.5B/2.9B/7.2B/13.3B boundary rows. The four g1h profiles are
      promoted; g1d 0.4B full-FFN is explicitly rejected and remains on its
      previous head-only/generic fallback.

Acceptance: every promoted row lowers footprint, passes correctness, and meets
the declared same-card fp16 equivalence/speed threshold. See
[`docs/QUANTIZATION.md`](docs/QUANTIZATION.md).

### 2. Final Albatross/RWKV-LM matrix

- [ ] Close the exact Tesla T4 gap measured on 2026-07-20: native-graph decode
      is `0.4888x–0.8649x` and B1/T512 fused prefill is
      `0.5385x–0.7671x` same-card Albatross across 0.1B–2.9B.
- [ ] Fresh same-card/same-session RTX 5090 Albatross full rerun.
- [ ] Extend P2/P3 beyond the V100 canonical P1 matrix.
- [ ] Recheck RTX 4090 prompt-512 historical high-water reference.
- [ ] Add larger-model prefill/decode rows with explicit memory ceilings.
- [ ] Keep shape, dtype, checkpoint and timing method identical.

### 2a. Broaden optimized-Qwen exact-card coverage

The initial optimized-reference milestones are closed: RTX 5070 bsz8 passes
its 18-cell fp16/W8/W4 matrix, and V100 1.5B/2B target-only B1/B8 passes raw
and active-work gates against full-FLA/Triton-conv Qwen. Remaining work:

- [ ] Extend RTX 5070 full-FLA coverage to bsz1/2/4 and larger 4B/9B pairs.
- [ ] Extend V100 beyond prompt512/decode64 and the 1.5B/2B pair.
- [ ] Add optimized-Qwen exact-card matrices on Ampere and Hopper.
- [ ] Keep raw throughput, `tok/s * active parameters`, correctness and memory
      as separate fail-closed gates; never substitute Torch-fallback rows.

Acceptance: every promoted Qwen reference row reports the full FLA core, norm
and accelerated causal-convolution route. Historical
`qwen_fla_gated_delta_rule_torch_conv` and forced-Torch V100 rows remain
diagnostics only.

### 3. Missing hardware

- [ ] H100/Hopper: bf16, large model, quant, batch and training rows.
- [ ] AMD/ROCm: native/no-FLA load/generate, training, cache and performance.
- [x] Tesla T4: exact-card compatibility, fallback policy, 0.1B–2.9B HF/cache,
      prefill/decode, quant and declared training-integration matrix. It is
      `Validated`, not production-close; dense/quant performance gaps remain
      in the sections above.
- [ ] Other Turing / RTX 20 products: exact-card validation. Do not inherit T4
      prefill or DP4A quant routing from compute capability alone.
- [ ] Additional RTX 50-series and laptop/low-memory cards.
- [ ] Apple M1–M4, Pro/Max/Ultra reproduction.

Use [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md) and the hardware report
template in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## P1 — Training and distributed closure

### 4. Longer training evidence

- [x] Record the exact RTX 5090 12x768 B16/T512 real-MiniPile loss curves,
      tokens/s, peak/steady memory and all 399 trainable deltas. Broader model
      sizes and cards remain in the items below.
- [ ] Extend 0.4B/1.5B/2.9B/7.2B SFT/DPO/GRPO beyond smoke steps.
- [x] Add single-card 2,500+2,500 checkpoint resume with model, optimizer,
      scheduler/sample position and Python/NumPy/torch CPU/CUDA RNG continuity
      checks. Distributed resume remains part of the ZeRO item below.
- [ ] Expand ZeRO-3 resume to larger models and more card combinations.
- [ ] Add H100 and AMD training matrices.

### 5. PP/TP and multi-device behavior

- [ ] Define the exact HF-scope PP and TP acceptance contract.
- [ ] Promote multi-device generation beyond `device_map` smoke.
- [ ] Add correctness and throughput gates for real TP/PP paths.
- [ ] Document unsupported combinations rather than silently falling back.

## P1 — Apple production completion

### 6. MLX and CoreML

- [ ] Reproduce promoted M5 gates on additional M-series devices.
- [ ] Validate Qwen3.5 2B/4B+ pairs with a formal response-quality rubric.
- [ ] Capture true peak-to-peak memory instead of loaded-memory proxies.
- [ ] Close CoreML INT4/LUT4 quality and confirm ANE placement/occupancy.
- [ ] Stabilize full-memory W8/W4 speed at long contexts and batch>1.
- [ ] Convert current guarded/experimental choices into maintainable policy
      tables with explicit fallback telemetry.

Detailed historical Apple experiments remain in
[`docs/hardware/APPLE_SILICON.md`](docs/hardware/APPLE_SILICON.md); the promoted
snapshot is [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md).

## P2 — Packaging and maintenance

### 7. Hub/release experience

- [ ] Publish a clean Hub example with conversion provenance and checksums.
- [ ] Add end-user SFT/LoRA/DPO examples with tiny reproducible datasets.
- [ ] Test a supported Transformers/PEFT/TRL version range in CI.
- [ ] Add clean-install CPU plus optional CUDA/Apple scheduled jobs.
- [ ] Document migration/deprecation policy for experimental backends.

### 8. Upstream and architecture

- [ ] Continue native Transformers upstreamability without breaking the wrapper.
- [ ] Keep card names and card-specific tuning out of core model logic; route
      through policy modules and tested dispatch.
- [ ] Reduce duplicated benchmark/session utilities after evidence is preserved.
- [ ] Maintain remote-code direct-import closure tests for fresh HF caches.

### 9. Speculative decoding

- [ ] Add CUDA target/draft end-to-end speed and acceptance artifacts.
- [ ] Validate multiple draft sizes and rejection rates.
- [ ] Preserve exact target-distribution/correctness requirements.
- [ ] Leave DFlash and serving-engine scheduler integration to their own projects.

## PR completion checklist

This is a per-PR template, not a list of outstanding project tasks:

- Exact hardware/runtime/model/dtype recorded.
- Reproduction command included.
- Raw JSONL/log and concise README included.
- Correctness, speed and memory reported together.
- Negative or partial results described honestly.
- Canonical status/benchmark/TODO documents updated only when status changes.
- `python tests/test_markdown_links.py` passes.
- Relevant unit/smoke tests pass.
