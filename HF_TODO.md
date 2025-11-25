# RWKV-7 HF adapter TODO

Only **unfinished, actionable HF-adapter work** belongs here. Completed
experiments and historical plans belong in benchmark artifacts or Git history.
Native vLLM/SGLang scheduler work is out of scope for this file.

Last updated: **2026-07-15**.

## P0 — Final production gaps

### 1. Full-memory W8/W4 performance

Goal: obtain the large memory reduction of broad projection quantization while
remaining fp16-or-faster across representative batch/prompt/decode shapes.

- [ ] Fuse quantized R/K/V/output and FFN projections instead of relying on
      selected-module speed policy.
- [ ] Add fused quant prefill; decode-only wins are insufficient.
- [ ] Validate V100, 4090, 5090 and at least one Ampere professional card.
- [ ] Preserve cosine, same-next, footprint and paired timing gates.
- [ ] Add 0.4B/1.5B/2.9B/7.2B/13.3B boundary rows where memory permits.

Acceptance: every promoted row lowers footprint, passes correctness, and meets
the declared same-card fp16 equivalence/speed threshold. See
[`docs/QUANTIZATION.md`](docs/QUANTIZATION.md).

### 2. Final Albatross/RWKV-LM matrix

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
- [ ] Turing NVIDIA: compatibility and fallback policy.
- [ ] Additional RTX 50-series and laptop/low-memory cards.
- [ ] Apple M1–M4, Pro/Max/Ultra reproduction.

Use [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md) and the hardware report
template in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## P1 — Training and distributed closure

### 4. Longer training evidence

- [ ] Record loss curve, samples/s or tokens/s, peak memory and trainable delta.
- [ ] Extend 0.4B/1.5B/2.9B/7.2B SFT/DPO/GRPO beyond smoke steps.
- [ ] Add checkpoint resume with optimizer/scheduler/RNG continuity checks.
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

- [ ] Exact hardware/runtime/model/dtype recorded.
- [ ] Reproduction command included.
- [ ] Raw JSONL/log and concise README included.
- [ ] Correctness, speed and memory reported together.
- [ ] Negative or partial results described honestly.
- [ ] Canonical status/benchmark/TODO documents updated only when status changes.
- [ ] `python tests/test_markdown_links.py` passes.
- [ ] Relevant unit/smoke tests pass.
