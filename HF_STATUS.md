# RWKV-7 Hugging Face adapter status

Canonical current snapshot. Scope: Transformers loading/generation/training,
PEFT/TRL, recurrent cache helpers, quantized HF inference, hardware validation
and reproducible performance evidence. Native vLLM/SGLang work is separate.

Last updated: **2026-08-11**. The released baseline was audited at `main`
`045bac1b769240facd290e1ac8232e8b1ca39778` after the `v0.6.0` release and
the merged RTX 4080/V100 B8 optimization series (#100-#102).

## Overall status

| Area | Status | Current conclusion |
|---|---|---|
| HF v0.6 adapter deliverable | **COMPLETE** | The released, evidence-backed HF scope is accepted; later cards, shapes, datasets, and serving engines extend the matrix without reopening this milestone |
| Official `.pth` → HF conversion | **PASS** | Published sizes are shape-inferred and converted to safetensors; low-memory 13.3B conversion is validated on a 48GB/no-swap host |
| Transformers API | **PASS** | Auto classes, save/reload, generation/cache, masks and labels/loss |
| Official/HF correctness | **PASS for current gates** | top-k/cosine/greedy, save/reload, cache handoff, MATH500 and compression checks |
| PEFT | **PASS** | LoRA forward/backward and adapter save/load/merge |
| Trainer / TRL | **PASS for compatibility matrix; train_temp exact lane accepted** | Trainer, SFT, DPO and GRPO smoke; RTX 5090 BF16 12x768 Native B16/T512 has exact backward/step tensors, paired real-MiniPile 3-seed x 1,000-step, continuous 5,000-step and checkpoint-resume gates |
| DeepSpeed ZeRO-2/3 | **PASS for current smoke matrix** | base and selected resume evidence across V100/A100/A800/A6000 setups |
| Recurrent state cache | **PASS** | select/reorder/drop/compact, offload/restore, chunked prefill and telemetry |
| Native/no-FLA backend | **PASS for HF compatibility and exact measured Native lanes** | load/generate/cache/PEFT/Trainer/TRL pass; exact RTX 5090 training and FP16-state lanes remain accepted; RTX 4080 7.2B/B8 FP16 state passes; RTX 5070 Laptop now has exact 0.4B/1.5B prefill/decode shape routing; V100 0.4B/1.5B B8 FP16 state passes paired speed/memory/greedy gates |
| W8/W4 functionality and memory | **PASS** | bnb and native/MLX paths load/generate and reduce footprint |
| Validated W8/W4 speed lanes | **PASS for measured profiles** | V100 MM4 closes 1.5B/2.9B/7.2B cached-decode profiles 7/7 each and the separate 1.5B group256 P128/D128 B1/B2/B4/B8 all-phase speed matrix 4/4; Tesla T4 exact-card head-speed W8/W4 closes 26/26 decode cells at `>=1.0207x` fp16 with greedy parity; RTX 4080 B1/B8 output-head A8W8/W4 pass all 36 exact complete-cell speed/correctness gates per route; RTX 5090 g1h 1.5B/2.9B/7.2B/13.3B pass all-phase exact-model Marlin W4 at `0.5298x–0.6250x` footprint |
| Production performance | **PASS for declared exact-card profiles** | V100 retains Albatross/native-quant lanes and adds exact 0.4B/1.5B B8 FP16-state decode; RTX 5070 Laptop adds exact Native prefill/decode schedules while retaining its full-FLA Qwen lane; RTX 4080/4090/5090 and Apple M5 retain their named promoted profiles; additional shapes remain post-release expansion |
| Apple M5 1.5B target-only | **PASS for checked B8 profile** | true B8, T133/decode64, no draft and no prefix coalescing; active-normalized prefill/decode=`1.1406x/1.1394x` Qwen3.5 2B, raw peak=`1.790/2.152GB`, fidelity passes |
| Huawei Ascend 910B3 | **PASS for integrated compatibility scope** | Import-safe torch-npu runtime, eager/JIT, fixed-batch NPUGraph and exact 7.2B W8 B1/B4/B8 route are integrated from pinned standalone commit `b6391271f`; future-main reruns and broader training/graph/quant profiles are independent extensions |
| Biren BR106M | **PASS for integrated compatibility scope** | BF16 native eager, FP32 recurrent state, decomposed GroupNorm and low-memory conversion support are integrated from `47322bf`; all released 0.1B-13.3B checkpoints have standalone functional evidence; performance, quant and multi-card profiles can be extended independently |
| MetaX C500 | **PASS for integrated compatibility scope** | Exact-card routing prevents MXMACA's CUDA-compatible capability 8.0 from inheriting NVIDIA Ampere kernels; native eager FP32/FP16/BF16 tiny and real 0.4B HF/Trainer/PEFT evidence is pinned at `f2653e2`; broader model/performance/quant profiles are post-release work |
| Hardware support policy | **PASS for declared release scope** | NVIDIA, AMD, Apple, Ascend, Biren, MetaX, MUSA and CPU fallback boundaries are represented with fail-closed routing; unmeasured products do not inherit another card's promotion |
| PP/TP boundary | **PASS for dense HF inference scope** | Two-V100 `device_map` PP-style generation matches single GPU; actual Transformers `tp_plan="auto"` shards embedding/attention/FFN/head weights at B1/B8 with exact greedy parity, minimum logits cosine `0.99999821`, and `0.52031x/0.611611x` local peak VRAM. Recurrent state is explicitly replicated; quantized TP/training and serving-engine executors are separate lanes |
| Speculative decoding | **EXPERIMENTAL PASS** | HF-compatible harness and Apple target-greedy oracle evidence exist |

## Completion reporting rule

Report completion against an explicitly named scope:

- **Current HF milestone:** `COMPLETE`.
- **Published HF-adapter `v0.6.0` release:** `COMPLETE` under the profile-based
  boundaries in [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md).
- **Official HF adapter deliverable:** `COMPLETE` for the accepted conversion,
  Transformers, training, cache, hardware-policy, quantization, PP/TP, and
  exact-card performance scope.
- **New cards, shapes, quality suites, and serving engines:** post-release
  expansion projects listed in [`HF_TODO.md`](HF_TODO.md); they do not
  downgrade or reopen the completed release.

There is **no official repository-wide completion percentage**. Do not turn
roadmap item counts or the number of status rows into a percentage;
the scopes have different acceptance gates and are not equally weighted.

## Hardware summary

| Platform | Status | Canonical evidence / boundary |
|---|---|---|
| V100 32GB | **Production-close for measured lanes** | Dense Albatross/Qwen and packed-MM4 lanes are retained. Exact-V100 B8 WAVG remains positive. The new exact 0.4B/1.5B B8 FP16-state route passes both A/B orders at `1.0216x-1.0288x`, saves `16.875-58.125 MiB`, and matches `4,096/4,096` greedy tokens per model across the two processes; B1/B2/B4 and non-V100 Volta products stay FP32. Wider full-memory shapes remain post-release extensions; [`bench/v100_exact_card_20260811/`](bench/v100_exact_card_20260811/README.md), [`bench/4080_v100_decode_tuning_20260808/`](bench/4080_v100_decode_tuning_20260808/README.md), [`bench/v100_sm70_mm4_bntn_20260716/`](bench/v100_sm70_mm4_bntn_20260716/README.md), [`bench/v100_active_b1b8_20260715/`](bench/v100_active_b1b8_20260715/README.md) |
| RTX 4090 | **Production-close for measured 0.4B–7.2B bsz8 lanes** | Small pairs pass 54/54 with minimum dense prefill/decode `1.041959x`/`4.214362x` across the three pair minima, plus the separate 7.2B/9B 18/18 close; all use fail-closed native/full-FLA, active-work and quant-local speed/memory gates; task quality and other batches remain open; [`bench/4090_small_bsz8_20260715/`](bench/4090_small_bsz8_20260715/README.md), [`bench/4090_g1h_7p2_bsz8_20260715/`](bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 4080 | **Production-close for measured Native HF B1/B8 and capacity lanes** | The 0.4B/0.8B, 1.5B/2B and 2.9B/4B full-FLA-Qwen matrices pass 6/6 with dense prefill/decode minima `1.012285x/1.435296x`; output-head A8W8/W4 pass all 36 complete-cell gates per route. Exact-B8 grouped W/A/V projections improve 0.4B/1.5B/2.9B decode by `1.1267x/1.0942x/1.0809x` with greedy `4,608/4,608`. The 7.2B/B8 FP16-state route reaches `344.39 tok/s`, `1.0301x` the FP32-state route, saves `123.88 MiB`, and matches greedy `12,288/12,288`; this supersedes the earlier B4-only dense-capacity statement. 13.3B MM8/MM4 remain quant-only capacity routes because fp16 exceeds 16GB. Task quality and long-run/distributed training remain separate; [`bench/4080_b8_projection_bmm_20260809/`](bench/4080_b8_projection_bmm_20260809/README.md), [`bench/4080_7p2b_fp16_state_20260809/`](bench/4080_7p2b_fp16_state_20260809/README.md), [`bench/4080_full_model_ladder_20260719/`](bench/4080_full_model_ladder_20260719/README.md) |
| RTX 5070 Laptop | **Production-close for measured full-FLA and Native lanes** | The prior 1.5B/full-FLA-Qwen B8 matrix remains 18/18. The new exact-card Native pass promotes 0.4B/1.5B P128/P512 prefill graph+scan, raw recurrent decode, shape-gated norm/mix, and B8 FP16 state. Final P128 decode is `301.9/2,059.2 tok/s` for 0.4B B1/B8 and `113.4/799.1 tok/s` for 1.5B B1/B8; negative projection/LoRA candidates remain disabled; [`bench/5070_max_perf_20260811/`](bench/5070_max_perf_20260811/README.md), [`bench/5070_qwen35_full_fla_bsz8_20260714/`](bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| RTX 5090 | **Production-close for measured Qwen/W4/train_temp/Native lanes** | Native B16/T512 train_temp passes exact tensors, paired real-data multi-seed, 5,000-step and resume gates at `1.00049x–1.00255x` official throughput. Exact fp16-state Native decode passes B1/B8 at `1.0010x/1.0104x`; 2.9B/13.3B B1/B8 prompt128/512/2048 prefill passes 12/12 tensor/state/greedy and speed cells. The full-FLA Qwen3.5 matrix passes 8/8 B1/B8 pairs and 144/144 cells with dense prefill/decode minima `1.0226x/2.8130x`; the latest-checkpoint dense-FP16 close passes 24/24 parameter-adjusted prefill cells at minimum/median `1.072987x/1.317515x`, with 8/8 graph-continuation correctness rows. [`bench/5090_g1i_qwen35_prefill_pd_sota_20260811/`](bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md), [`bench/5090_g1h_qwen35_b1_b8_20260715/`](bench/5090_g1h_qwen35_b1_b8_20260715/README.md), [`bench/5090_native_official_fp16_production_20260718/`](bench/5090_native_official_fp16_production_20260718/README.md), [`bench/5090_native_train_temp_real_minipile_20260718/`](bench/5090_native_train_temp_real_minipile_20260718/README.md), [`bench/5090_bntn_all_models_20260716/`](bench/5090_bntn_all_models_20260716/README.md) |
| Apple M5 | **Production-close for measured MLX pairs** | B1 speculative gates plus the separate 1.5B B8 target-only cold gate; the latter uses no draft/cache and passes active-normalized prefill/decode at `1.1406x/1.1394x` Qwen3.5 2B with lower raw peak memory; [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md) |
| A100 40GB / A800 80GB / A6000 48GB | **Validated** | Large-model API/training/quant/ZeRO matrices; production performance remains card-specific |
| GTX 1080 Ti | **Smoke** | compatibility evidence, not full production-close |
| Tesla T4 | **Validated compatibility and quant profile** | 0.1B–2.9B HF/cache/prefill/decode/training integration passes; head-speed W8/W4 decode passes 26/26. Dense decode is `0.4888x–0.8649x` and B1/T512 fused prefill is `0.5385x–0.7671x` Albatross; production-close dense/full-model expansion is a separate post-release profile; [`bench/t4_production_close_20260720/`](bench/t4_production_close_20260720/README.md) |
| AMD gfx1100 / ROCm 7.2.1 | **Validated with exact-card decode lanes** | Fully native HF/PEFT/cache/chunked-prefill/bf16 Trainer pass; fused decode and output-head W8/W4 are validated through 13.3B, with quant decode beating paired fp16 in 40/40 B1/B2/B4/B8 rows. Fused prefill, full-model quant, MI-series and same-card official/Albatross remain open; [`docs/validation/AMD_ROCM_HF_VALIDATION.md`](docs/validation/AMD_ROCM_HF_VALIDATION.md) |
| Huawei Ascend 910B3 / CANN 8.5.0 | **Accepted integrated compatibility scope** | Native HF eager/JIT/cache/chunked-prefill, fixed-batch NPUGraph and exact 7.2B FP16 W8 B1/B4/B8 are ported with fail-closed policy; broader PEFT/TRL, multi-NPU, graph-prefill and W4 profiles are optional extensions; [`docs/hardware/HUAWEI_ASCEND.md`](docs/hardware/HUAWEI_ASCEND.md) |
| Biren BR106M / BIRENSUPA 1.11 | **Accepted integrated compatibility scope** | BF16 auto-load/generate through 13.3B plus 0.1B cache/chunked-prefill/dynamic-batch/save-reload/PEFT/Trainer evidence are ported; FP16 and compile fail closed; paired performance, quant, TRL/ZeRO and multi-card are optional extensions; [`docs/hardware/BIREN_BR106M.md`](docs/hardware/BIREN_BR106M.md) |
| MetaX C500 / MXMACA 3.5.3.20 | **Accepted integrated compatibility scope** | Native eager/no-FLA, CPU-oracle FP16/BF16, cache/chunked-prefill/generation/backward and real 0.4B Trainer/PEFT evidence are ported with conservative routing; broader models, performance, quant and multi-card are optional extensions; [`docs/hardware/METAX_C500.md`](docs/hardware/METAX_C500.md) |
| H100 / other Turing | **Additional product coverage** | not part of the released exact-card matrix; future evidence must not inherit exact-T4 or other-card promotion |

Full matrix: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md).

## Canonical documents

- Official requirement mapping: [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)
- Current numeric summary: [`BENCHMARK.md`](BENCHMARK.md)
- Hardware: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md)
- Performance: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)
- Quantization: [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md)
- Training: [`docs/TRAINING.md`](docs/TRAINING.md)
