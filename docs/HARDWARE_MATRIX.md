# Hardware validation matrix

Canonical current hardware status for the HF adapter. Detailed experiment logs
remain in `bench/` and platform-specific documents.

Last updated: **2026-08-12**. The released baseline was audited at `main`
`045bac1b769240facd290e1ac8232e8b1ca39778`.

The `v0.6.0` HF release is complete for the declared support policy and the
evidence-backed profiles below. `Validated`, `Smoke`, or unmeasured additional
products do not represent missing HF APIs: they describe the strength and
boundary of exact-hardware evidence. New products extend this matrix without
reopening the released adapter milestone.

## Status definitions

- **Production-close:** promoted correctness, performance, memory and regression evidence.
- **Validated:** meaningful API/training/quantization matrix exists, but the production performance gate is incomplete.
- **Smoke:** load/forward/generate or a narrow compatibility path is proven.
- **Open:** no current repository evidence sufficient for a support claim.

## Matrix

| Platform | Status | Models / scope | Strongest current evidence | Open work |
|---|---|---|---|---|
| Tesla V100 32GB, sm70 | **Production-close** | dense/Qwen lanes; packed-MM4 cached decode for 1.5B/2.9B/7.2B; exact-B8 WAVG; exact 0.4B/1.5B B8 FP16 state; larger inference/training smoke | Albatross P1 and MM4 gates remain; FP16 state passes both A/B orders at `1.0216x-1.0288x`, saves `16.875-58.125 MiB`, and matches `4,096/4,096` greedy tokens per model; other batches and Volta cards remain FP32 | Larger-model FP16-state rows, P2/P3, full-memory prefill and broader optimized-Qwen shapes |
| RTX 3090 24GB, sm86 | **Production-close for latest-checkpoint B1/B8 prefill and retained quant lanes** | g1d/g1i 0.4B/1.5B/2.9B/7.2B vs Qwen3.5 0.8B/2B/4B/9B; earlier B8 W8/W4 | Strict per-cell parameter-adjusted prefill passes 24/24 at minimum/median `1.227477x/1.467758x`; 24/24 full-FLA references and 25/25 prompt/cache-handoff correctness rows pass | B2/B4 latest checkpoints, task-quality evaluation, multi-GPU and broader Ampere portability |
| RTX 4090 24GB, sm89 | **Production-close for measured bsz8 lanes** | RWKV 0.4B/1.5B/2.9B/7.2B vs Qwen3.5 0.8B/2B/4B/9B, dense/W8/W4 | Small-model matrix passes 54/54 and 7.2B passes 18/18; dense prefill/decode, active-work, full Qwen FLA, quant speed and quant-local memory gates pass | bsz1/2/4 latest matrix, task quality, full-memory W4, other Ada cards |
| RTX 4080 16GB, sm89 | **Production-close for measured Native HF B1/B8 and capacity lanes** | Native HF 0.4B/1.5B/2.9B vs full-FLA Qwen3.5 0.8B/2B/4B; optimized 7.2B/B8 decode; 13.3B capacity | parameter-adjusted Prefill and Decode each pass 36/36 cells, minima `1.068520x/1.140700x`; output-head quant complete-cell minima `1.003101x/1.015996x`; 7.2B/B8 FP16-state decode is median `344.39 tok/s`, `1.0301x` FP32 state, `-123.88 MiB`, and greedy `12288/12288`; 13.3B MM8/MM4 fit | 7.2B same-card Qwen/Albatross close, task quality, long-run/distributed training and full-model quant speed |
| RTX 5090, sm120 | **Production-close for measured Qwen/W4/train_temp/Native lanes** | Existing Qwen/W4/MATH lanes; Native B16/T512 training; Native fp16-state 7.2B decode and 2.9B/13.3B prefill | Full-FLA Qwen 8/8; BN/TN W4 B1/B8; exact-step plus real-MiniPile 3-seed/5,000-step/resume training; Native decode `1.0010x/1.0104x` and prefill 12/12 at `1.0029x–1.5690x` versus pinned same-precision v3a | Broader models/cards/quality, cross-harness memory parity and distributed train_temp |
| Apple M5 16GB | **Production-close for MLX measured pairs** | 0.4B vs Qwen3.5 0.8B; 1.5B vs Qwen3.5 2B; MPS training smoke | Tiled DPLR, guarded compiled/speculative decode, W4 memory and same-device gates | M1–M4/Pro/Max/Ultra, CoreML INT4/ANE, larger quality matrix |
| A100 40GB | **Validated** | 0.1B–7.2B inference/training | fp16/bf16, Trainer/SFT/DPO, resume, ZeRO-2/3 base | 80GB lane, performance close, larger ZeRO-3 resume |
| A800 80GB | **Validated** | 0.1B–13.3B mixed matrix | 13.3B quant smoke, native MM8/MM4, single/dual-card ZeRO | Native quant speed remains below fp16 on larger models |
| RTX A6000 48GB | **Validated** | 0.1B–7.2B; dual-card training to 2.9B | API/training/resume/ZeRO and quant memory evidence | Quant speed and production performance gate |
| GTX 1080 Ti, sm61 | **Smoke / compatibility** | 0.1B and 0.4B fp16 | Native/no-FLA fallback, bnb and native-mm smoke, batch sweep | Training, larger models and quant speed |
| Tesla T4 15GB, sm75 | **Validated** | 0.1B/0.4B/1.5B/2.9B HF, cache, fused prefill, native-graph decode, W8/W4 and training integration | 123 dense/cache rows; exact-T4 DP4A quant; head-speed W8/W4 decode `>=1.0207x` fp16; Trainer/PEFT/TRL and single-GPU ZeRO/resume matrix | Dense decode `0.4888x–0.8649x` and B1/T512 prefill `0.5385x–0.7671x` Albatross; full-model all-phase quant speed |
| RTX 5070 Laptop, sm120 | **Production-close for measured full-FLA and Native lanes** | 1.5B RWKV vs full-FLA Qwen3.5 2B at B8; Native 0.4B/1.5B B1/B2/B4/B8, P128/P512 | full-FLA matrix passes 18/18; Native graph+scan handoff passes; raw recurrent gains `1.0272x-1.1265x`; promoted norm/mix gains `1.0373x-1.1630x`; B8 FP16 state saves `16.875-58.125 MiB` | Other model pairs, broader full-FLA batches, 2.9B Native matrix, and model-quality evaluation |
| H100 / Hopper | **Additional product coverage** | not part of the released exact-card matrix | conservative CUDA fallback policy | Add bf16, large-model, quant, training and performance evidence before product-specific promotion |
| AMD gfx1100 / ROCm 7.2.1 | **Validated with exact-card decode lanes** | 0.1B compatibility/training; 0.1B-13.3B fused decode; 0.4B-13.3B output-head W8/W4 | Fully native HF/PEFT/cache/chunked-prefill/Trainer; dense fused decode remains positive through 13.3B; 40/40 output-head quant B1/B2/B4/B8 decode rows beat fp16 with greedy parity | Fused prefill, full-model quant speed/2.9B W4 quality, MI-series and same-card Albatross |
| Moore Threads MTT S70 / MUSA 4.2.0 | **Smoke; exact-card legacy scope** | First-generation 0.1B HF lane; no Tensor Core and slow fp16 compute, so retained kernels use fp16 storage/IO with fp32 compute/state | Standalone parity, 64-token eager/WKV equality, autograd eager fallback, B1/B2 smoke and paired evidence; WKV prefill `1.214072x`, decode `1.000000x`; opt-in shift-mix prefill median `1.050809x`, decode neutral, peak memory equal | SDK 4.2.0 is frozen; S4000/S5000 capabilities require independent exact-card validation and must not inherit S70 limits; no broad bf16/quant/graph/multi-device/training-kernel claim |
| Huawei Ascend 910B3 / CANN 8.5.0 | **Accepted integrated compatibility scope** | Native eager/JIT, recurrent cache, chunked prefill, fixed-batch NPUGraph decode, exact 7.2B W8 B1/B4/B8 | Import-safe torch-npu runtime; standalone real-7.2B alignment and graph/W8 evidence at pinned commit `b6391271f` | Future-main reruns, broader PEFT/TRL, multi-NPU, graph-prefill, W4, and other Ascend products extend the matrix |
| Biren106M 32GB / BIRENSUPA 1.11 | **Accepted integrated compatibility scope** | BF16 auto-load/forward/cached-generate for 0.1B-13.3B; 0.1B cache/chunked-prefill/dynamic-batch/save-reload/PEFT/Trainer | Exact SUPA routing, FP16 fail-closed, FP32 recurrent state, eager GroupNorm decomposition and low-memory BF16 conversion at `47322bf` | Future-main reruns, B1-B8 RWKV-LM/Albatross, W8/W4, TRL/ZeRO, optimized training and multi-card extend the profile |
| MetaX C500 64GB / MXMACA 3.5.3.20 | **Accepted integrated compatibility scope** | Native eager FP32/FP16/BF16 tiny; real 0.4B FP16 inference and BF16 Trainer/LoRA | Exact product routing prevents CUDA capability 8.0 from inheriting NVIDIA Ampere kernels; CPU-oracle, cache/chunked-prefill/generation/backward/save-reload and FP32 PEFT merge evidence at `f2653e2` | Future-main reruns, all-model B1-B8, same-card RWKV-LM/Albatross, W8/W4, TRL/ZeRO and multi-card extend the profile |
| Other Turing / RTX 20 | **Additional product coverage** | exact-card validation required before promotion | conservative family routing only | Must not inherit Tesla T4 prefill or DP4A quant promotion by `sm_75` alone |
| CPU | **Experimental fallback** | Tiny/native tests | Import-safe native model and CPU tests | Production performance is not a target yet |

## Promoted artifacts

- V100: [`../bench/v100_production_close_20260711/README.md`](../bench/v100_production_close_20260711/README.md)
- V100 packed MM4 BN/TN: [`../bench/v100_sm70_mm4_bntn_20260716/README.md`](../bench/v100_sm70_mm4_bntn_20260716/README.md)
- V100 full-FLA Qwen B1/B8: [`../bench/v100_active_b1b8_20260715/README.md`](../bench/v100_active_b1b8_20260715/README.md)
- V100 / RTX 4080 exact-B8 decode tuning: [`../bench/4080_v100_decode_tuning_20260808/README.md`](../bench/4080_v100_decode_tuning_20260808/README.md)
- V100 exact 0.4B/1.5B B8 FP16 state: [`../bench/v100_exact_card_20260811/README.md`](../bench/v100_exact_card_20260811/README.md)
- RTX 3090 g1h 7.2B: [`../bench/3090_g1h_7p2_bsz8_20260714/README.md`](../bench/3090_g1h_7p2_bsz8_20260714/README.md)
- RTX 3090 max-performance parameter-adjusted prefill: [`../bench/3090_g1i_qwen35_maxperf_20260812/README.md`](../bench/3090_g1i_qwen35_maxperf_20260812/README.md)
- RTX 4090 g1h 7.2B: [`../bench/4090_g1h_7p2_bsz8_20260715/README.md`](../bench/4090_g1h_7p2_bsz8_20260715/README.md)
- RTX 4090 small models: [`../bench/4090_small_bsz8_20260715/README.md`](../bench/4090_small_bsz8_20260715/README.md)
- RTX 4080 Native HF, full-FLA Qwen and capacity ladder: [`../bench/4080_full_model_ladder_20260719/README.md`](../bench/4080_full_model_ladder_20260719/README.md)
- RTX 4080 all-six parameter-adjusted P/D close: [`../bench/4080_adjusted_pd_20260811/README.md`](../bench/4080_adjusted_pd_20260811/README.md)
- RTX 4080 grouped B8 W/A/V projection: [`../bench/4080_b8_projection_bmm_20260809/README.md`](../bench/4080_b8_projection_bmm_20260809/README.md)
- RTX 4080 7.2B/B8 Triton FP16 state: [`../bench/4080_7p2b_fp16_state_20260809/README.md`](../bench/4080_7p2b_fp16_state_20260809/README.md)
- RTX 5090 full-FLA Qwen B1/B8: [`../bench/5090_g1h_qwen35_b1_b8_20260715/README.md`](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md)
- RTX 5090 latest-checkpoint parameter-adjusted prefill: [`../bench/5090_g1i_qwen35_prefill_pd_20260811/README.md`](../bench/5090_g1i_qwen35_prefill_pd_20260811/README.md)
- RTX 5090 single-card prefill ceiling: [`../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md`](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md)
- RTX 5090 latest g1h 13.3B: [`../bench/5090_g1h_13p3_20260715/README.md`](../bench/5090_g1h_13p3_20260715/README.md)
- RTX 5090 g1h BN/TN Tensor Core W4 matrix: [`../bench/5090_bntn_all_models_20260716/README.md`](../bench/5090_bntn_all_models_20260716/README.md)
- RTX 5090 MATH500 and quant pressure: [`../bench/5090_blackwell_production_close_20260712/README.md`](../bench/5090_blackwell_production_close_20260712/README.md)
- RTX 5090 official train_temp alignment: [`../bench/5090_train_temp_alignment_20260717/README.md`](../bench/5090_train_temp_alignment_20260717/README.md)
- RTX 5090 Native B16 train_temp alignment: [`../bench/5090_native_train_temp_b16_20260718/README.md`](../bench/5090_native_train_temp_b16_20260718/README.md)
- RTX 5090 Native fused decode: [`../bench/5090_native_decode_fused_20260718/README.md`](../bench/5090_native_decode_fused_20260718/README.md)
- RTX 5090 Native HF Gradio and unchanged official shell: [`../bench/5090_native_hf_gradio_train_temp_20260718/README.md`](../bench/5090_native_hf_gradio_train_temp_20260718/README.md)
- RTX 5090 Native fp16-state official inference: [`../bench/5090_native_official_fp16_production_20260718/README.md`](../bench/5090_native_official_fp16_production_20260718/README.md)
- RTX 5090 Native real-MiniPile train_temp: [`../bench/5090_native_train_temp_real_minipile_20260718/README.md`](../bench/5090_native_train_temp_real_minipile_20260718/README.md)
- RTX 5070 Laptop: [`../bench/5070_qwen35_full_fla_bsz8_20260714/README.md`](../bench/5070_qwen35_full_fla_bsz8_20260714/README.md)
- RTX 5070 Laptop exact Native backend: [`../bench/5070_max_perf_20260811/README.md`](../bench/5070_max_perf_20260811/README.md)
- Apple M5: [`hardware/APPLE_PRODUCTION_CLOSE.md`](hardware/APPLE_PRODUCTION_CLOSE.md)
- A100: [`validation/A100_HF_VALIDATION.md`](validation/A100_HF_VALIDATION.md)
- A800: [`validation/A800_HF_VALIDATION.md`](validation/A800_HF_VALIDATION.md)
- V100 training/compatibility: [`validation/V100_HF_VALIDATION.md`](validation/V100_HF_VALIDATION.md)
- AMD ROCm: [`validation/AMD_ROCM_HF_VALIDATION.md`](validation/AMD_ROCM_HF_VALIDATION.md)
- Moore Threads MUSA: [`hardware/MUSA.md`](hardware/MUSA.md), [`../bench/musa_s70_validation_20260728/`](../bench/musa_s70_validation_20260728/README.md), [`../bench/musa_s70_shift_mix_20260728/`](../bench/musa_s70_shift_mix_20260728/README.md)
- Huawei Ascend NPU: [`hardware/HUAWEI_ASCEND.md`](hardware/HUAWEI_ASCEND.md), [standalone source and raw evidence](https://github.com/rwkv-rs/rwkv7-ascend-npu/tree/b6391271f0ddb606dad5e97a65fa4742e82fcd50)
- Biren BR106M: [`hardware/BIREN_BR106M.md`](hardware/BIREN_BR106M.md), [standalone source and raw evidence](https://github.com/yyqdbngt/rwkv7-biren-br106m/tree/47322bfaffc2e662fa989863c3fda4d74f02fc32)
- MetaX C500: [`hardware/METAX_C500.md`](hardware/METAX_C500.md), [standalone source and raw evidence](https://github.com/123123213weqw/rwkv7-metax-c500/tree/f2653e20250821ec48534e5e08b07d59effb985c)
- Tesla T4: [`hardware/TURING_T4.md`](hardware/TURING_T4.md), [`../bench/t4_production_close_20260720/`](../bench/t4_production_close_20260720/README.md)
- Blackwell history: [`hardware/BLACKWELL_50SERIES.md`](hardware/BLACKWELL_50SERIES.md)

## Adding a card

A hardware PR must record exact device, driver/runtime versions, model and
dtype, commands, raw JSONL/logs, correctness checks, footprint/peak memory and
throughput. Promotion to production-close additionally requires a fail-closed
comparison gate and repeated/paired measurements where clock or process state
can bias results.
