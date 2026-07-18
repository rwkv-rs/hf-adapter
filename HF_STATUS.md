# RWKV-7 Hugging Face adapter status

Canonical current snapshot. Scope: Transformers loading/generation/training,
PEFT/TRL, recurrent cache helpers, quantized HF inference, hardware validation
and reproducible performance evidence. Native vLLM/SGLang work is separate.

Last updated: **2026-07-19**.

## Overall status

| Area | Status | Current conclusion |
|---|---|---|
| Official `.pth` → HF conversion | **PASS** | Published sizes are shape-inferred and converted to safetensors; low-memory 13.3B conversion is validated on a 48GB/no-swap host |
| Transformers API | **PASS** | Auto classes, save/reload, generation/cache, masks and labels/loss |
| Official/HF correctness | **PASS for current gates** | top-k/cosine/greedy, save/reload, cache handoff, MATH500 and compression checks |
| PEFT | **PASS** | LoRA forward/backward and adapter save/load/merge |
| Trainer / TRL | **PASS for compatibility matrix; train_temp exact lane accepted** | Trainer, SFT, DPO and GRPO smoke; RTX 5090 BF16 12x768 Native B16/T512 has exact backward/step tensors, paired real-MiniPile 3-seed x 1,000-step, continuous 5,000-step and checkpoint-resume gates |
| DeepSpeed ZeRO-2/3 | **PASS for current smoke matrix** | base and selected resume evidence across V100/A100/A800/A6000 setups |
| Recurrent state cache | **PASS** | select/reorder/drop/compact, offload/restore, chunked prefill and telemetry |
| Native/no-FLA backend | **PASS for HF compatibility and exact measured 5090 lanes** | load/generate/cache/PEFT/Trainer/TRL pass; exact Native training is `1.00049x` official by paired-seed median and `1.00255x` over 5,000 steps; 7.2B fp16-state decode is `1.0010x/1.0104x`, and 2.9B/13.3B B1/B8 prefill passes 12/12 same-precision cells; broader performance remains card-local |
| W8/W4 functionality and memory | **PASS** | bnb and native/MLX paths load/generate and reduce footprint |
| Universal W8/W4 speed | **PARTIAL** | selected V100/4090/5090 speed lanes pass; V100 MM4 closes 1.5B/2.9B/7.2B cached-decode profiles 7/7 each, and RTX 5090 g1h 1.5B/2.9B/7.2B/13.3B have all-phase exact-model Marlin W4 closes at `0.5298x–0.6250x` footprint; universal prefill/full-projection coverage remains open |
| Production performance | **PARTIAL / strong card-local closes** | V100 Albatross/native-quant lanes plus 1.5B/full-FLA-Qwen B1/B8 active-work gates, RTX 4090 all published 0.4B–7.2B/Qwen3.5 bsz8 pairs, RTX 5090 and Apple M5 have promoted artifacts; RTX 5070 1.5B RWKV vs full-FLA Qwen3.5 2B also passes its bsz8 gates in 18/18 cells |
| Apple M5 1.5B target-only | **PASS for checked B8 profile** | true B8, T133/decode64, no draft and no prefix coalescing; active-normalized prefill/decode=`1.1406x/1.1394x` Qwen3.5 2B, raw peak=`1.790/2.152GB`, fidelity passes |
| Full common-card coverage | **PARTIAL** | H100, AMD/ROCm, Turing and broader Apple/50-series evidence remain open |
| PP/TP | **PARTIAL** | HF multi-device/device-map smoke exists; production TP matrix is not closed |
| Speculative decoding | **EXPERIMENTAL PASS** | HF-compatible harness and Apple target-greedy oracle evidence exist |

## Completion reporting rule

Report completion against an explicitly named scope:

- **Current HF milestone:** `COMPLETE`. This means every item listed under
  `Current milestone — COMPLETE` in [`HF_TODO.md`](HF_TODO.md) is closed.
- **Public HF-adapter release milestone:** suitable for release under the
  boundaries in [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md).
- **Universal all-card/all-shape production requirements:** `PARTIAL`. The
  open boundaries are listed below and in [`HF_TODO.md`](HF_TODO.md).

There is **no official repository-wide completion percentage**. Do not turn
roadmap checkbox counts or the number of `PASS`/`PARTIAL` rows into a percentage;
the scopes have different acceptance gates and are not equally weighted.

## Hardware summary

| Platform | Status | Canonical evidence / boundary |
|---|---|---|
| V100 32GB | **Production-close for measured lanes** | Dense Albatross/Qwen lanes remain; packed MM4 cached decode passes exact 1.5B memory+group128+fused, 2.9B group256 speed and 7.2B memory profiles 7/7 each with decode minima `1.0255x/1.0111x/1.0810x`, lower footprint and complete greedy equality. Full-memory prefill remains open; [`bench/v100_sm70_mm4_bntn_20260716/`](bench/v100_sm70_mm4_bntn_20260716/README.md), [`bench/v100_active_b1b8_20260715/`](bench/v100_active_b1b8_20260715/README.md) |
| RTX 4090 | **Production-close for measured 0.4B–7.2B bsz8 lanes** | Small pairs pass 54/54 with minimum dense prefill/decode `1.041959x`/`4.214362x` across the three pair minima, plus the separate 7.2B/9B 18/18 close; all use fail-closed native/full-FLA, active-work and quant-local speed/memory gates; task quality and other batches remain open; [`bench/4090_small_bsz8_20260715/`](bench/4090_small_bsz8_20260715/README.md), [`bench/4090_g1h_7p2_bsz8_20260715/`](bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 5070 Laptop | **Production-close for measured bsz8 full-FLA lane** | 1.5B RWKV vs 2B Qwen: 36/36 raw rows and 18/18 strict cells pass; minimum prefill/decode speedups are `1.082707x`/`1.795119x`, minimum tok/s per active-B ratios are `1.333940x`/`2.211641x`, and footprint/peak VRAM are no larger in 18/18; all Qwen performance rows use FLA core, norm, and Triton conv with no Torch fallback; model quality is not covered; [`bench/5070_qwen35_full_fla_bsz8_20260714/`](bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| RTX 5090 | **Production-close for measured Qwen/W4/train_temp/Native lanes** | Native B16/T512 train_temp passes exact tensors, paired real-data multi-seed, 5,000-step and resume gates at `1.00049x–1.00255x` official throughput. Exact fp16-state Native decode passes B1/B8 at `1.0010x/1.0104x`; 2.9B/13.3B B1/B8 prompt128/512/2048 prefill passes 12/12 tensor/state/greedy and speed cells. The current official Gradio page also has real-browser B1/B8 generation screenshots and byte-identical 54-token official/Native output; its B8 page row is near parity rather than a lead. [`bench/5090_gradio_native_hf_frontend_ab_20260719/`](bench/5090_gradio_native_hf_frontend_ab_20260719/README.md), [`bench/5090_native_official_fp16_production_20260718/`](bench/5090_native_official_fp16_production_20260718/README.md), [`bench/5090_native_train_temp_real_minipile_20260718/`](bench/5090_native_train_temp_real_minipile_20260718/README.md), [`bench/5090_bntn_all_models_20260716/`](bench/5090_bntn_all_models_20260716/README.md) |
| Apple M5 | **Production-close for measured MLX pairs** | B1 speculative gates plus the separate 1.5B B8 target-only cold gate; the latter uses no draft/cache and passes active-normalized prefill/decode at `1.1406x/1.1394x` Qwen3.5 2B with lower raw peak memory; [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md) |
| A100 40GB / A800 80GB / A6000 48GB | **Validated** | Large-model API/training/quant/ZeRO matrices; production performance remains card-specific |
| GTX 1080 Ti | **Smoke** | compatibility evidence, not full production-close |
| H100 / AMD / Turing | **Open** | real-card matrix required |

Full matrix: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md).

## Current release blockers

These are the remaining technical boundaries, not already-completed history:

1. Extend the RTX 5090 exact-model W4 FFN close to still-dense projections,
   W8, old cards and the remaining new-card matrix while retaining all-phase
   fp16-or-faster performance and the larger footprint reduction.
2. Extend the exact RTX 5090 fp16-state decode/prefill close beyond the measured
   7.2B decode and 2.9B/13.3B prefill profiles to more cards and shapes; retain
   same-card Albatross P2/P3 for larger models.
3. Add H100, AMD/ROCm and Turing evidence.
4. Complete broader Apple-family, CoreML INT4/ANE and formal quality coverage.
5. Extend the accepted single-5090 B1/B16 train_temp lanes beyond the current
   real-MiniPile 5,000-step evidence to larger models, multi-day runs,
   additional cards and distributed training; extend ZeRO-3
   optimizer/scheduler/RNG resume evidence.
6. Close production PP/TP rather than treating `device_map` smoke as TP proof.

## Canonical documents

- Official requirement mapping: [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)
- Current numeric summary: [`BENCHMARK.md`](BENCHMARK.md)
- Hardware: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md)
- Performance: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)
- Quantization: [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md)
- Training: [`docs/TRAINING.md`](docs/TRAINING.md)
- Remaining contributor work: [`HF_TODO.md`](HF_TODO.md)
