# Results and evidence index

This page is a compact navigation layer for reviewers and contributors. It does
not replace [`../BENCHMARK.md`](../BENCHMARK.md), the hardware matrix, or raw
dated artifacts. A row means only that the linked, exact profile has evidence;
it does not imply an unbounded all-card or all-shape claim.

Last updated: **2026-08-16**. The released baseline was audited at `main`
`045bac1b769240facd290e1ac8232e8b1ca39778`.

The canonical current cross-card throughput view is
[`QWEN35_LATEST_P_D_TOKPS.md`](QWEN35_LATEST_P_D_TOKPS.md). It reports the
latest strict artifacts in model-size -> GPU -> B1/B8 order and renders every
Prefill/Decode tok/s value with zero or one decimal place.

## Inference performance and quantization

| Evidence ID | Platform | Scope | Promoted conclusion | Source |
|---|---|---|---|---|
| `v100-dense-ref` | V100 32GB | Dense Albatross and 1.5B/Qwen B1/B8 | Production-close for the recorded reference lanes | [`v100_production_close_20260711`](../bench/v100_production_close_20260711/README.md), [`v100_active_b1b8_20260715`](../bench/v100_active_b1b8_20260715/README.md) |
| `v100-mm4` | V100 32GB | 1.5B/2.9B/7.2B packed-MM4 decode | Three exact profiles pass speed, footprint and complete greedy gates | [`v100_sm70_mm4_bntn_20260716`](../bench/v100_sm70_mm4_bntn_20260716/README.md) |
| `v100-b8-wavg` | V100 32GB | 0.4B/1.5B/2.9B B8 | Exact-card launch tuning adds `1.0114x-1.0312x` with greedy parity | [`4080_v100_decode_tuning_20260808`](../bench/4080_v100_decode_tuning_20260808/README.md) |
| `v100-b8-fp16-state` | V100 32GB | 0.4B/1.5B B8 | Both process orders pass at `1.0216x-1.0288x`, `-16.875` to `-58.125 MiB`, with exact greedy traces | [`v100_exact_card_20260811`](../bench/v100_exact_card_20260811/README.md) |
| `v100-paired-pd-v1` | V100 32GB | Frozen-reference 0.4B/0.8B through 7.2B/9B, B1/B8, P128/512/2048, D128/512 | Raw and parameter-adjusted Prefill/Decode all pass 48/48; adjusted minima `1.808536x/1.120373x`; 8/8 FLA/native probes plus 7.2B/B8 graph closure pass. Speed only, not quality/E2E | [`v100_qwen35_paired_pd_v1_20260814`](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| `t4-production` | Tesla T4 | 0.1B-2.9B dense/cache/quant/training | Compatibility and head-quant lanes pass; dense/full-model performance limits remain explicit | [`t4_production_close_20260720`](../bench/t4_production_close_20260720/README.md) |
| `3090-g1h-b8` | RTX 3090 | 7.2B vs Qwen3.5-9B, dense/W8/W4 B8 | Strict current B8 matrix passes 18/18 | [`3090_g1h_7p2_bsz8_20260714`](../bench/3090_g1h_7p2_bsz8_20260714/README.md) |
| `3090-paired-pd-v2` | RTX 3090 | Current optimized Qwen3.5 0.8B/2B/4B/9B versus RWKV 0.4B/1.5B/2.9B/7.2B, B1/B8, P128/512/2048, D128/512 | Raw and parameter-adjusted Prefill/Decode pass 48/48 at minima `1.574925x/1.208324x/1.253926x/1.017763x`; 8/8 512-token FLA/native checks pass. Speed only, not quality/E2E | [`3090_qwen35_paired_pd_v2_20260816`](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| `3090-latest-prefill-pd` | RTX 3090 | latest g1d/g1i 0.4B-7.2B versus full-FLA Qwen3.5, B1/B8 | Strict parameter-adjusted prefill passes 24/24 at minimum/median `1.227477x/1.467758x`; correctness passes 25/25 | [`3090_g1i_qwen35_maxperf_20260812`](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| `4080-qwen-pairs` | RTX 4080 | 0.4B/1.5B/2.9B versus Qwen3.5 B1/B8 | Six dense pair matrices and exact output-head quant lanes pass | [`4080_full_model_ladder_20260719`](../bench/4080_full_model_ladder_20260719/README.md) |
| `4080-adjusted-pd` | RTX 4080 | 0.4B/1.5B/2.9B versus Qwen3.5 B1/B8 | Adjusted Prefill and Decode each pass 36/36 cells; minima `1.068520x/1.140700x` | [`4080_adjusted_pd_20260811`](../bench/4080_adjusted_pd_20260811/README.md) |
| `4080-paired-pd-v1` | RTX 4080 | Same-runtime 0.4B/0.8B through 2.9B/4B, B1/B8, P128/512/2048, D128/512 | Raw and parameter-adjusted Prefill/Decode all pass 36/36; adjusted minima `1.051333x/1.022115x`; 6/6 512-token native-graph/FLA probes pass. Speed only, not quality/E2E | [`4080_qwen35_paired_pd_v1_20260814`](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| `4080-b8-projection` | RTX 4080 | 0.4B/1.5B/2.9B B8 | Grouped W/A/V projection gains `1.1267x/1.0942x/1.0809x`, greedy `4,608/4,608` | [`4080_b8_projection_bmm_20260809`](../bench/4080_b8_projection_bmm_20260809/README.md) |
| `4080-7p2-state` | RTX 4080 | 7.2B/B8 FP16 state | `344.39 tok/s`, `1.0301x`, `-123.88 MiB`, greedy `12,288/12,288` | [`4080_7p2b_fp16_state_20260809`](../bench/4080_7p2b_fp16_state_20260809/README.md) |
| `4090-b8-matrix` | RTX 4090 | 0.4B-7.2B pair ladder, dense/W8/W4 B8 | Small-model 54/54 and 7.2B 18/18 matrices pass | [`4090_small_bsz8_20260715`](../bench/4090_small_bsz8_20260715/README.md), [`4090_g1h_7p2_bsz8_20260715`](../bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| `4090-4080-routes` | RTX 4090 | Latest 0.4B/1.5B/2.9B B1/B8 Prefill and B8 decode | 108/108 paired accumulation, 18/18 default-policy Prefill and 9/9 grouped-BMM rows pass; BMM gains `1.1259x-1.2002x` | [`4090_4080_routes_20260812`](../bench/4090_4080_routes_20260812/README.md) |
| `4090-adjusted-pd` | RTX 4090 | Latest 0.4B/1.5B/2.9B versus full-FLA/Triton-conv Qwen3.5, B1/B8 | Adjusted Prefill and Decode pass 36/36 each at minima `1.108265x/4.158943x`; exact 1.5B/B1/P2048 self-chunk route gains `1.2539x` | [`4090_adjusted_pd_20260812`](../bench/4090_adjusted_pd_20260812/README.md) |
| `4090-hf-best-optimized-v1` | RTX 4090 | Latest 0.4B/1.5B/2.9B/7.2B best-optimized HF versus official FLA/causal-conv Qwen3.5, B1/B8 | Unified 96/96-row contract passes; adjusted Prefill and Decode pass 48/48 each at minima `1.060506x/1.829468x`; 7.2B B1/B8 is complete | [`4090_hf_best_optimized_v1_20260812`](../bench/4090_hf_best_optimized_v1_20260812/README.md) |
| `4090-qwen-paired-pd-v2` | RTX 4090 | Current optimized Qwen3.5 0.8B/2B/4B/9B versus RWKV 0.4B/1.5B/2.9B/7.2B, B1/B8, P128/512/2048, D128/512 | Raw and parameter-adjusted Prefill/Decode pass 48/48 at minima `1.415206x/1.148668x/1.276285x/1.026173x`; 8/8 512-token FLA/native checks pass | [`4090_qwen35_paired_pd_v2_20260815`](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| `5070-b8-qwen` | RTX 5070 Laptop | 1.5B versus Qwen3.5-2B B8 | Full-FLA dense and fp16/W8/W4 gates pass for the measured lane | [`5070_qwen35_full_fla_bsz8_20260714`](../bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| `5070-native-exact` | RTX 5070 Laptop | Native 0.4B/1.5B B1/B2/B4/B8, P128/P512 | Graph+scan prefill, raw recurrent, shape-gated norm/mix and B8 FP16 state pass; negative fusion candidates fail closed | [`5070_max_perf_20260811`](../bench/5070_max_perf_20260811/README.md) |
| `5090-paired-decode-v1` | RTX 5090 | Frozen-reference 0.4B/0.8B through 7.2B/9B, B1/B8, P128/512/2048, D128/512 | Strict parameter-adjusted Decode passes 48/48 at minimum `1.029966x`; 8/8 native-graph/FLA 512-token correctness probes pass. Decode-only, not quality/Prefill/E2E | [`5090_qwen35_paired_decode_v1_20260813`](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |
| `5090-qwen-best-hf-v2` | RTX 5090 | Qwen3.5 0.8B/2B/4B/9B, B1/B8, P128/512/2048, D128/512 | Source artifact remains reference-only: 48/48 official FLA/causal-conv rows pass with fixed per-model StaticCache Graph routes; paired Decode is reported separately above | [`5090_qwen35_best_optimized_hf_v1_20260813`](../bench/5090_qwen35_best_optimized_hf_v1_20260813/README.md) |
| `5090-qwen-matrix` | RTX 5090 | 0.4B-7.2B versus Qwen3.5, B1/B8 | Historical under the superseding protocol; its original 8/8 model/batch pairs and 144/144 full-FLA cells pass remain valid only for that artifact | [`5090_g1h_qwen35_b1_b8_20260715`](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md) |
| `5090-w4` | RTX 5090 | g1h 1.5B/2.9B/7.2B/13.3B, B1/B8 | All-phase W4 speed, footprint and correctness pass 8/8 | [`5090_bntn_all_models_20260716`](../bench/5090_bntn_all_models_20260716/README.md) |
| `5090-native-official` | RTX 5090 | Native versus official v3a decode/prefill | 7.2B decode and 2.9B/13.3B 12-cell prefill scopes pass | [`5090_native_official_fp16_production_20260718`](../bench/5090_native_official_fp16_production_20260718/README.md) |
| `apple-m5` | Apple M5 | Selected MLX/Qwen pairs and W4 | Production-close for the named M5 profiles | [`APPLE_PRODUCTION_CLOSE.md`](hardware/APPLE_PRODUCTION_CLOSE.md) |
| `amd-gfx1100` | AMD gfx1100 | Native HF, fused decode and output-head W8/W4 | Native compatibility passes; fused decode and 40/40 head-quant decode rows are promoted | [`AMD_ROCM_HF_VALIDATION.md`](validation/AMD_ROCM_HF_VALIDATION.md) |

## Training, ecosystem and parallelism

| Evidence ID | Scope | Current conclusion | Source |
|---|---|---|---|
| `hf-ecosystem` | Auto classes, PEFT, Trainer, SFT, DPO, GRPO | Published compatibility matrix passes | [`TRAINING.md`](TRAINING.md), [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md) |
| `zero-resume` | ZeRO-2/3 base and selected resume paths | Current bounded multi-card smoke matrix passes | [`TRAINING.md`](TRAINING.md), [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md) |
| `5090-train-temp` | Native B16/T512 BF16, real MiniPile | Exact step, three paired seeds, 5,000 steps and 2,500+2,500 resume pass | [`5090_native_train_temp_real_minipile_20260718`](../bench/5090_native_train_temp_real_minipile_20260718/README.md) |
| `hf-pp-tp` | Dense HF inference PP/TP | Layer-split `device_map` and Transformers-native `tp_plan="auto"` gates pass for their declared scopes | [`ACCEPTANCE.md`](ACCEPTANCE.md), [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md) |
| `hf-state-cache` | Recurrent state and serving-like helpers | Select/reorder/drop/compact, offload/restore, dynamic-batch behavior and chunked-prefill parity pass | [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md), [`ACCEPTANCE.md`](ACCEPTANCE.md) |

## Integrated accelerator compatibility

| Platform | Accepted repository scope | Source | Contribution provenance |
|---|---|---|---|
| Huawei Ascend 910B3 | Native eager/JIT, cache, chunked prefill, fixed-batch NPUGraph and exact 7.2B W8 route | [`HUAWEI_ASCEND.md`](hardware/HUAWEI_ASCEND.md) | [PR #93](https://github.com/rwkv-rs/hf-adapter/pull/93) |
| Biren BR106M | BF16 Native eager, FP32 recurrent state and bounded PEFT/Trainer compatibility | [`BIREN_BR106M.md`](hardware/BIREN_BR106M.md) | [PR #95](https://github.com/rwkv-rs/hf-adapter/pull/95), `@yyqdbngt` |
| MetaX C500 | Native eager FP32/FP16/BF16 and real 0.4B Trainer/PEFT scope | [`METAX_C500.md`](hardware/METAX_C500.md) | [PR #94](https://github.com/rwkv-rs/hf-adapter/pull/94) |
| Moore Threads MUSA | Exact-card legacy S70 Native compatibility and paired kernel evidence | [`MUSA.md`](hardware/MUSA.md) | [PR #87](https://github.com/rwkv-rs/hf-adapter/pull/87), `@KakaruHayate` |

## Interpretation

- Use [`../HF_STATUS.md`](../HF_STATUS.md) for completion reporting.
- Use [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) for product boundaries.
- Use [`../BENCHMARK.md`](../BENCHMARK.md) for exact promoted numbers.
- Use [`../bench/INDEX.md`](../bench/INDEX.md) to find raw artifacts.
- Use [`../CONTRIBUTORS.md`](../CONTRIBUTORS.md) and
  [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md) for contribution provenance.
