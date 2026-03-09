# RWKV-7 Hugging Face adapter status

Canonical current snapshot. Scope: Transformers loading/generation/training,
PEFT/TRL, recurrent cache helpers, quantized HF inference, hardware validation
and reproducible performance evidence. Native vLLM/SGLang work is separate.

Last updated: **2026-08-04**.

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
| Validated W8/W4 speed lanes | **PASS for measured profiles** | V100 MM4 closes 1.5B/2.9B/7.2B cached-decode profiles 7/7 each and the separate 1.5B group256 P128/D128 B1/B2/B4/B8 all-phase speed matrix 4/4; Tesla T4 exact-card head-speed W8/W4 closes 26/26 decode cells at `>=1.0207x` fp16 with greedy parity; RTX 4080 B1/B8 output-head A8W8/W4 pass all 36 exact complete-cell speed/correctness gates per route; RTX 5090 g1h 1.5B/2.9B/7.2B/13.3B pass all-phase exact-model Marlin W4 at `0.5298x–0.6250x` footprint |
| Production performance | **PARTIAL / strong card-local closes** | V100 Albatross/native-quant lanes plus 1.5B/full-FLA-Qwen B1/B8 active-work gates; RTX 4080, RTX 4090, RTX 5070, RTX 5090 and Apple M5 have promoted exact-card artifacts for their named shapes; cross-card and model-quality conclusions remain separate |
| Apple M5 1.5B target-only | **PASS for checked B8 profile** | true B8, T133/decode64, no draft and no prefix coalescing; active-normalized prefill/decode=`1.1406x/1.1394x` Qwen3.5 2B, raw peak=`1.790/2.152GB`, fidelity passes |
| Huawei Ascend 910B3 | **HF integration ported; exact standalone evidence** | Import-safe torch-npu runtime, eager/JIT, fixed-batch NPUGraph and exact 7.2B W8 B1/B4/B8 route are integrated from pinned standalone commit `b6391271f`; a current-main real-device rerun remains open |
| Biren BR106M | **HF compatibility ported; exact standalone evidence** | BF16 native eager, FP32 recurrent state, decomposed GroupNorm and low-memory conversion support are integrated from `47322bf`; all released 0.1B-13.3B checkpoints have standalone functional evidence, with current-main, performance, quant and multi-card gates open |
| MetaX C500 | **HF compatibility ported; exact standalone evidence** | Exact-card routing prevents MXMACA's CUDA-compatible capability 8.0 from inheriting NVIDIA Ampere kernels; native eager FP32/FP16/BF16 tiny and real 0.4B HF/Trainer/PEFT evidence is pinned at `f2653e2`, with current-main rerun and performance/quant gates open |
| Full common-card coverage | **PARTIAL** | Tesla T4 and AMD gfx1100 compatibility are validated; gfx1100 exact decode and output-head W8/W4 lanes now pass through 13.3B, while H100, AMD prefill/full-model quant, MI-series, other Turing products and broader Apple/50-series evidence remain open |
| PP/TP boundary | **PASS for dense HF inference scope** | Two-V100 `device_map` PP-style generation matches single GPU; actual Transformers `tp_plan="auto"` shards embedding/attention/FFN/head weights at B1/B8 with exact greedy parity, minimum logits cosine `0.99999821`, and `0.52031x/0.611611x` local peak VRAM. Recurrent state is explicitly replicated; quantized TP/training and serving-engine executors are separate lanes |
| Speculative decoding | **EXPERIMENTAL PASS** | HF-compatible harness and Apple target-greedy oracle evidence exist |

## Completion reporting rule

Report completion against an explicitly named scope:

- **Current HF milestone:** `COMPLETE`. The remaining universal work is listed
  under `Scope and current boundary` in [`HF_TODO.md`](HF_TODO.md).
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
| V100 32GB | **Production-close for measured lanes** | Dense Albatross/Qwen lanes remain; packed MM4 cached decode passes exact 1.5B memory+group128+fused, 2.9B group256 speed and 7.2B memory profiles 7/7 each. The separate 1.5B speed/group256 P128/D128 matrix passes B1/B2/B4/B8 prefill/decode at minima `1.0032x/1.0011x` with `0.9344x` footprint. Long-row dequant+BLAS raises 1.5B full-memory B1/B8 prefill to `0.7816x/0.9076x`, but full-memory all-shape speed remains open; [`bench/v100_sm70_prefill_dequant_20260723/`](bench/v100_sm70_prefill_dequant_20260723/README.md), [`bench/v100_sm70_mm4_bntn_20260716/`](bench/v100_sm70_mm4_bntn_20260716/README.md), [`bench/v100_active_b1b8_20260715/`](bench/v100_active_b1b8_20260715/README.md) |
| RTX 4090 | **Production-close for measured 0.4B–7.2B bsz8 lanes** | Small pairs pass 54/54 with minimum dense prefill/decode `1.041959x`/`4.214362x` across the three pair minima, plus the separate 7.2B/9B 18/18 close; all use fail-closed native/full-FLA, active-work and quant-local speed/memory gates; task quality and other batches remain open; [`bench/4090_small_bsz8_20260715/`](bench/4090_small_bsz8_20260715/README.md), [`bench/4090_g1h_7p2_bsz8_20260715/`](bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 4080 | **Production-close for measured Native HF B1/B8 and capacity lanes** | The 0.4B/0.8B, 1.5B/2B and 2.9B/4B full-FLA-Qwen matrices pass 6/6 with dense prefill/decode minima `1.012285x/1.435296x`; output-head A8W8/W4 pass all 36 complete-cell gates per route. 7.2B fp16 fits through B4/P128; 13.3B MM8/MM4 fit as quant-only capacity routes because fp16 exceeds 16GB. Task quality and long-run/distributed training remain separate; [`bench/4080_full_model_ladder_20260719/`](bench/4080_full_model_ladder_20260719/README.md) |
| RTX 5070 Laptop | **Production-close for measured bsz8 full-FLA lane** | 1.5B RWKV vs 2B Qwen: 36/36 raw rows and 18/18 strict cells pass; minimum prefill/decode speedups are `1.082707x`/`1.795119x`, minimum tok/s per active-B ratios are `1.333940x`/`2.211641x`, and footprint/peak VRAM are no larger in 18/18; all Qwen performance rows use FLA core, norm, and Triton conv with no Torch fallback; model quality is not covered; [`bench/5070_qwen35_full_fla_bsz8_20260714/`](bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| RTX 5090 | **Production-close for measured Qwen/W4/train_temp/Native lanes** | Native B16/T512 train_temp passes exact tensors, paired real-data multi-seed, 5,000-step and resume gates at `1.00049x–1.00255x` official throughput. Exact fp16-state Native decode passes B1/B8 at `1.0010x/1.0104x`; 2.9B/13.3B B1/B8 prompt128/512/2048 prefill passes 12/12 tensor/state/greedy and speed cells. The full-FLA Qwen3.5 matrix passes 8/8 B1/B8 pairs and 144/144 cells with dense prefill/decode minima `1.0226x/2.8130x`. [`bench/5090_g1h_qwen35_b1_b8_20260715/`](bench/5090_g1h_qwen35_b1_b8_20260715/README.md), [`bench/5090_native_official_fp16_production_20260718/`](bench/5090_native_official_fp16_production_20260718/README.md), [`bench/5090_native_train_temp_real_minipile_20260718/`](bench/5090_native_train_temp_real_minipile_20260718/README.md), [`bench/5090_bntn_all_models_20260716/`](bench/5090_bntn_all_models_20260716/README.md) |
| Apple M5 | **Production-close for measured MLX pairs** | B1 speculative gates plus the separate 1.5B B8 target-only cold gate; the latter uses no draft/cache and passes active-normalized prefill/decode at `1.1406x/1.1394x` Qwen3.5 2B with lower raw peak memory; [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md) |
| A100 40GB / A800 80GB / A6000 48GB | **Validated** | Large-model API/training/quant/ZeRO matrices; production performance remains card-specific |
| GTX 1080 Ti | **Smoke** | compatibility evidence, not full production-close |
| Tesla T4 | **Validated, not production-close** | 0.1B–2.9B HF/cache/prefill/decode/training integration passes; head-speed W8/W4 decode passes 26/26. Dense decode remains `0.4888x–0.8649x` and B1/T512 fused prefill `0.5385x–0.7671x` Albatross; full-model all-phase quant speed remains open; [`bench/t4_production_close_20260720/`](bench/t4_production_close_20260720/README.md) |
| AMD gfx1100 / ROCm 7.2.1 | **Validated with exact-card decode lanes** | Fully native HF/PEFT/cache/chunked-prefill/bf16 Trainer pass; fused decode and output-head W8/W4 are validated through 13.3B, with quant decode beating paired fp16 in 40/40 B1/B2/B4/B8 rows. Fused prefill, full-model quant, MI-series and same-card official/Albatross remain open; [`docs/validation/AMD_ROCM_HF_VALIDATION.md`](docs/validation/AMD_ROCM_HF_VALIDATION.md) |
| Huawei Ascend 910B3 / CANN 8.5.0 | **Integrated from exact-stack standalone validation** | Native HF eager/JIT/cache/chunked-prefill, fixed-batch NPUGraph and exact 7.2B FP16 W8 B1/B4/B8 are ported with fail-closed policy; current-main 7.2B rerun, PEFT/TRL, multi-NPU, graph prefill and W4 production remain open; [`docs/hardware/HUAWEI_ASCEND.md`](docs/hardware/HUAWEI_ASCEND.md) |
| Biren BR106M / BIRENSUPA 1.11 | **Integrated from exact-card standalone validation** | BF16 auto-load/generate through 13.3B plus 0.1B cache/chunked-prefill/dynamic-batch/save-reload/PEFT/Trainer evidence are ported; FP16 and compile fail closed; current-main rerun, paired performance, quant, TRL/ZeRO and multi-card remain open; [`docs/hardware/BIREN_BR106M.md`](docs/hardware/BIREN_BR106M.md) |
| MetaX C500 / MXMACA 3.5.3.20 | **Integrated from exact-card standalone validation** | Native eager/no-FLA, CPU-oracle FP16/BF16, cache/chunked-prefill/generation/backward and real 0.4B Trainer/PEFT evidence are ported with conservative routing; current-main rerun, full model matrix, Albatross/RWKV-LM performance, quant and multi-card remain open; [`docs/hardware/METAX_C500.md`](docs/hardware/METAX_C500.md) |
| H100 / other Turing | **Open** | real-card matrix required; other `sm_75` products do not inherit exact-T4 promotion |

Full matrix: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md).

## Canonical documents

- Official requirement mapping: [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)
- Current numeric summary: [`BENCHMARK.md`](BENCHMARK.md)
- Hardware: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md)
- Performance: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)
- Quantization: [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md)
- Training: [`docs/TRAINING.md`](docs/TRAINING.md)
