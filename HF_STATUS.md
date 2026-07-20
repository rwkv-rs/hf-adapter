# RWKV-7 Hugging Face adapter status

Canonical current snapshot. Scope: Transformers loading/generation/training,
PEFT/TRL, recurrent cache helpers, quantized HF inference, hardware validation
and reproducible performance evidence. Native vLLM/SGLang work is separate.

Last updated: **2026-07-20**.

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
| Native/no-FLA backend | **PASS for HF compatibility and exact measured 5090 lanes** | load/generate/cache/PEFT/Trainer/TRL pass; exact Native training is `1.00049x` official by paired-seed median and `1.00255x` over 5,000 steps; 7.2B fp16-state decode is `1.0010x/1.0104x`, and 2.9B/13.3B B1/B8 prefill passes 12/12 same-precision cells |
| W8/W4 functionality and memory | **PASS** | bnb and native/MLX paths load/generate and reduce footprint |
| Validated W8/W4 speed lanes | **PASS for measured profiles** | V100 MM4 closes 1.5B/2.9B/7.2B cached-decode profiles 7/7 each; Tesla T4 exact-card head-speed W8/W4 closes 26/26 decode cells at `>=1.0207x` fp16 with greedy parity; RTX 4080 B1/B8 output-head A8W8/W4 pass all 36 exact complete-cell speed/correctness gates per route; RTX 5090 g1h 1.5B/2.9B/7.2B/13.3B pass all-phase exact-model Marlin W4 at `0.5298x–0.6250x` footprint |
| Production performance | **PARTIAL / strong card-local closes** | V100 Albatross/native-quant lanes plus 1.5B/full-FLA-Qwen B1/B8 active-work gates; RTX 4080, RTX 4090, RTX 5070, RTX 5090 and Apple M5 have promoted exact-card artifacts for their named shapes; cross-card and model-quality conclusions remain separate |
| Apple M5 1.5B target-only | **PASS for checked B8 profile** | true B8, T133/decode64, no draft and no prefix coalescing; active-normalized prefill/decode=`1.1406x/1.1394x` Qwen3.5 2B, raw peak=`1.790/2.152GB`, fidelity passes |
| Full common-card coverage | **PARTIAL** | Tesla T4 is validated; H100, AMD/ROCm, other Turing products and broader Apple/50-series evidence remain open |
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
| RTX 4080 | **Production-close for measured Native HF B1/B8 and capacity lanes** | The 0.4B/0.8B, 1.5B/2B and 2.9B/4B full-FLA-Qwen matrices pass 6/6 with dense prefill/decode minima `1.012285x/1.435296x`; output-head A8W8/W4 pass all 36 complete-cell gates per route. 7.2B fp16 fits through B4/P128; 13.3B MM8/MM4 fit as quant-only capacity routes because fp16 exceeds 16GB. Task quality and long-run/distributed training remain separate; [`bench/4080_full_model_ladder_20260719/`](bench/4080_full_model_ladder_20260719/README.md) |
| RTX 5070 Laptop | **Production-close for measured bsz8 full-FLA lane** | 1.5B RWKV vs 2B Qwen: 36/36 raw rows and 18/18 strict cells pass; minimum prefill/decode speedups are `1.082707x`/`1.795119x`, minimum tok/s per active-B ratios are `1.333940x`/`2.211641x`, and footprint/peak VRAM are no larger in 18/18; all Qwen performance rows use FLA core, norm, and Triton conv with no Torch fallback; model quality is not covered; [`bench/5070_qwen35_full_fla_bsz8_20260714/`](bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| RTX 5090 | **Production-close for measured Qwen/W4/train_temp/Native lanes** | Native B16/T512 train_temp passes exact tensors, paired real-data multi-seed, 5,000-step and resume gates at `1.00049x–1.00255x` official throughput. Exact fp16-state Native decode passes B1/B8 at `1.0010x/1.0104x`; 2.9B/13.3B B1/B8 prompt128/512/2048 prefill passes 12/12 tensor/state/greedy and speed cells. The full-FLA Qwen3.5 matrix passes 8/8 B1/B8 pairs and 144/144 cells with dense prefill/decode minima `1.0226x/2.8130x`. [`bench/5090_g1h_qwen35_b1_b8_20260715/`](bench/5090_g1h_qwen35_b1_b8_20260715/README.md), [`bench/5090_native_official_fp16_production_20260718/`](bench/5090_native_official_fp16_production_20260718/README.md), [`bench/5090_native_train_temp_real_minipile_20260718/`](bench/5090_native_train_temp_real_minipile_20260718/README.md), [`bench/5090_bntn_all_models_20260716/`](bench/5090_bntn_all_models_20260716/README.md) |
| Apple M5 | **Production-close for measured MLX pairs** | B1 speculative gates plus the separate 1.5B B8 target-only cold gate; the latter uses no draft/cache and passes active-normalized prefill/decode at `1.1406x/1.1394x` Qwen3.5 2B with lower raw peak memory; [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md) |
| A100 40GB / A800 80GB / A6000 48GB | **Validated** | Large-model API/training/quant/ZeRO matrices; production performance remains card-specific |
| GTX 1080 Ti | **Smoke** | compatibility evidence, not full production-close |
| Tesla T4 | **Validated, not production-close** | 0.1B–2.9B HF/cache/prefill/decode/training integration passes; head-speed W8/W4 decode passes 26/26. Dense decode remains `0.4888x–0.8649x` and B1/T512 fused prefill `0.5385x–0.7671x` Albatross; full-model all-phase quant speed remains open; [`bench/t4_production_close_20260720/`](bench/t4_production_close_20260720/README.md) |
| H100 / AMD / other Turing | **Open** | real-card matrix required; other `sm_75` products do not inherit exact-T4 promotion |

Full matrix: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md).

## Canonical documents

- Official requirement mapping: [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)
- Current numeric summary: [`BENCHMARK.md`](BENCHMARK.md)
- Hardware: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md)
- Performance: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)
- Quantization: [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md)
- Training: [`docs/TRAINING.md`](docs/TRAINING.md)
