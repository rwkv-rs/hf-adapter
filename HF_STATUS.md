# RWKV-7 Hugging Face adapter status

Canonical current snapshot. Scope: Transformers loading/generation/training,
PEFT/TRL, recurrent cache helpers, quantized HF inference, hardware validation
and reproducible performance evidence. Native vLLM/SGLang work is separate.

Last updated: **2026-07-16**.

## Overall status

| Area | Status | Current conclusion |
|---|---|---|
| Official `.pth` → HF conversion | **PASS** | Published sizes are shape-inferred and converted to safetensors; low-memory 13.3B conversion is validated on a 48GB/no-swap host |
| Transformers API | **PASS** | Auto classes, save/reload, generation/cache, masks and labels/loss |
| Official/HF correctness | **PASS for current gates** | top-k/cosine/greedy, save/reload, cache handoff, MATH500 and compression checks |
| PEFT | **PASS** | LoRA forward/backward and adapter save/load/merge |
| Trainer / TRL | **PASS for compatibility matrix** | Trainer, SFT, DPO and GRPO smoke across tested CUDA and Apple paths |
| DeepSpeed ZeRO-2/3 | **PASS for current smoke matrix** | base and selected resume evidence across V100/A100/A800/A6000 setups |
| Recurrent state cache | **PASS** | select/reorder/drop/compact, offload/restore, chunked prefill and telemetry |
| Native/no-FLA backend | **PASS as opt-in compatibility path** | load/generate/cache/PEFT/Trainer/TRL smoke; not the default wrapper |
| W8/W4 functionality and memory | **PASS** | bnb and native/MLX paths load/generate and reduce footprint |
| Universal W8/W4 speed | **PARTIAL** | selected V100/4090/5090 speed lanes pass; RTX 5090 7.2B now has an all-phase Marlin W4 FFN-hybrid close with `0.5298x` footprint, while universal/full-projection coverage remains open |
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
| V100 32GB | **Production-close for measured lanes** | Dense Albatross P1 and native W8/W4 speed pass; the current 1.5B/full-FLA-Qwen3.5-2B target-only B1/B8 artifact passes raw prefill/decode minima `2.815921x/5.270432x` and active-work minima `2.285574x/4.277804x`, with 32-token route probes. Older Torch-fallback Qwen matrices remain historical only; [`bench/v100_production_close_20260711/`](bench/v100_production_close_20260711/README.md), [`bench/v100_active_b1b8_20260715/`](bench/v100_active_b1b8_20260715/README.md) |
| RTX 4090 | **Production-close for measured 0.4B–7.2B bsz8 lanes** | Small pairs pass 54/54 with minimum dense prefill/decode `1.041959x`/`4.214362x` across the three pair minima, plus the separate 7.2B/9B 18/18 close; all use fail-closed native/full-FLA, active-work and quant-local speed/memory gates; task quality and other batches remain open; [`bench/4090_small_bsz8_20260715/`](bench/4090_small_bsz8_20260715/README.md), [`bench/4090_g1h_7p2_bsz8_20260715/`](bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 5070 Laptop | **Production-close for measured bsz8 full-FLA lane** | 1.5B RWKV vs 2B Qwen: 36/36 raw rows and 18/18 strict cells pass; minimum prefill/decode speedups are `1.082707x`/`1.795119x`, minimum tok/s per active-B ratios are `1.333940x`/`2.211641x`, and footprint/peak VRAM are no larger in 18/18; all Qwen performance rows use FLA core, norm, and Triton conv with no Torch fallback; model quality is not covered; [`bench/5070_qwen35_full_fla_bsz8_20260714/`](bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| RTX 5090 | **Production-close for measured B1/B8 Qwen, W4 and g1h 13.3B lanes** | The current-main full-FLA Qwen3.5 matrix passes 8/8 batch-pairs from 0.4B/0.8B through 7.2B/9B. The new paired BF16/W4 lane passes 1.5B and 7.2B at B1/B8 in both prefill and decode; 7.2B uses 64 Marlin W4 FFN matrices plus a TorchAO W4 head, reaches minimum `1.0835x/1.4872x` prefill/decode and `0.5298x` footprint, with final cosine `>=0.99955124` and same-next 4/4. Fresh official g1h 13.3B load/generate also passes; [`bench/5090_marlin_w4_hybrid_20260716/`](bench/5090_marlin_w4_hybrid_20260716/README.md), [`bench/5090_g1h_qwen35_b1_b8_20260715/`](bench/5090_g1h_qwen35_b1_b8_20260715/README.md), [`bench/5090_g1h_13p3_20260715/`](bench/5090_g1h_13p3_20260715/README.md) |
| Apple M5 | **Production-close for measured MLX pairs** | B1 speculative gates plus the separate 1.5B B8 target-only cold gate; the latter uses no draft/cache and passes active-normalized prefill/decode at `1.1406x/1.1394x` Qwen3.5 2B with lower raw peak memory; [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md) |
| A100 40GB / A800 80GB / A6000 48GB | **Validated** | Large-model API/training/quant/ZeRO matrices; production performance remains card-specific |
| GTX 1080 Ti | **Smoke** | compatibility evidence, not full production-close |
| H100 / AMD / Turing | **Open** | real-card matrix required |

Full matrix: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md).

## Current release blockers

These are the remaining technical boundaries, not already-completed history:

1. Extend the RTX 5090 7.2B W4 FFN-hybrid close to all desired projections,
   W8, old cards and the remaining new-card matrix while retaining all-phase
   fp16-or-faster performance and the larger footprint reduction.
2. Extend same-card Albatross P2/P3 to larger models and rerun the final 5090
   reference live on the same card/session.
3. Add H100, AMD/ROCm and Turing evidence.
4. Complete broader Apple-family, CoreML INT4/ANE and formal quality coverage.
5. Extend long training and larger ZeRO-3 checkpoint-resume evidence.
6. Close production PP/TP rather than treating `device_map` smoke as TP proof.

## Canonical documents

- Official requirement mapping: [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)
- Current numeric summary: [`BENCHMARK.md`](BENCHMARK.md)
- Hardware: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md)
- Performance: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)
- Quantization: [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md)
- Training: [`docs/TRAINING.md`](docs/TRAINING.md)
- Remaining contributor work: [`HF_TODO.md`](HF_TODO.md)
