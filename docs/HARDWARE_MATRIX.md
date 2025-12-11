# Hardware validation matrix

Canonical current hardware status for the HF adapter. Detailed experiment logs
remain in `bench/` and platform-specific documents.

Last updated: **2026-07-18**.

## Status definitions

- **Production-close:** promoted correctness, performance, memory and regression evidence.
- **Validated:** meaningful API/training/quantization matrix exists, but the production performance gate is incomplete.
- **Smoke:** load/forward/generate or a narrow compatibility path is proven.
- **Open:** no current repository evidence sufficient for a support claim.

## Matrix

| Platform | Status | Models / scope | Strongest current evidence | Open work |
|---|---|---|---|---|
| Tesla V100 32GB, sm70 | **Production-close** | dense/Qwen lanes; packed-MM4 cached decode for 1.5B/2.9B/7.2B; larger inference/training smoke | Albatross P1; three exact MM4 profiles pass 7/7 each with lower footprint and complete greedy equality; 1.5B alone opts into fused epilogues | Larger-model P2/P3, full-memory prefill and broader optimized-Qwen shapes |
| RTX 3090 24GB, sm86 | **Production-close for measured bsz8 lanes** | g1h 7.2B vs Qwen3.5-9B plus 1.5B/2B and 2.9B/4B pairs | Latest 7.2B dense/W8/W4 matrix passes 18/18; dense decode active-work, Qwen FLA, quant speed and physical-memory gates pass | bsz1/2/4 latest-g1h matrix, task-quality evaluation, multi-GPU |
| RTX 4090 24GB, sm89 | **Production-close for measured bsz8 lanes** | RWKV 0.4B/1.5B/2.9B/7.2B vs Qwen3.5 0.8B/2B/4B/9B, dense/W8/W4 | Small-model matrix passes 54/54 and 7.2B passes 18/18; dense prefill/decode, active-work, full Qwen FLA, quant speed and quant-local memory gates pass | bsz1/2/4 latest matrix, task quality, full-memory W4, other Ada cards |
| RTX 5090, sm120 | **Production-close for measured Qwen/W4/train_temp/Native-decode lanes** | Existing Qwen/W4/MATH lanes, single-GPU train_temp, Native 7.2B FP32-state B1/B8 decode | Full-FLA Qwen 8/8; BN/TN W4 B1/B8; Native train exact/cohort/resume; Native decode `1.0041x/1.0045x` precision-matched v3a with active-extension/logits/512-token gates | Native train parity, fp16-state/prefill/memory, broader models/cards/quality and distributed train_temp |
| Apple M5 16GB | **Production-close for MLX measured pairs** | 0.4B vs Qwen3.5 0.8B; 1.5B vs Qwen3.5 2B; MPS training smoke | Tiled DPLR, guarded compiled/speculative decode, W4 memory and same-device gates | M1–M4/Pro/Max/Ultra, CoreML INT4/ANE, larger quality matrix |
| A100 40GB | **Validated** | 0.1B–7.2B inference/training | fp16/bf16, Trainer/SFT/DPO, resume, ZeRO-2/3 base | 80GB lane, performance close, larger ZeRO-3 resume |
| A800 80GB | **Validated** | 0.1B–13.3B mixed matrix | 13.3B quant smoke, native MM8/MM4, single/dual-card ZeRO | Native quant speed remains below fp16 on larger models |
| RTX A6000 48GB | **Validated** | 0.1B–7.2B; dual-card training to 2.9B | API/training/resume/ZeRO and quant memory evidence | Quant speed and production performance gate |
| GTX 1080 Ti, sm61 | **Smoke / compatibility** | 0.1B and 0.4B fp16 | Native/no-FLA fallback, bnb and native-mm smoke, batch sweep | Training, larger models and quant speed |
| RTX 5070 Laptop, sm120 | **Production-close for measured bsz8 lane** | 1.5B RWKV vs full-FLA Qwen3.5 2B, fp16/W8/W4 | 18/18 speed, active-parameter efficiency, footprint, peak-VRAM, full-FLA binding, and greedy/cosine gates pass | Other model pairs, bsz1/2/4 full-FLA, and model-quality evaluation |
| H100 / Hopper | **Open** | — | — | bf16, large-model, quant, training and performance matrix |
| AMD / ROCm | **Open** | Native PyTorch direction | Import-safe/no-FLA architecture exists | Real ROCm card validation and kernels |
| Turing NVIDIA | **Open** | — | — | Compatibility and performance matrix |
| CPU | **Experimental fallback** | Tiny/native tests | Import-safe native model and CPU tests | Production performance is not a target yet |

## Promoted artifacts

- V100: [`../bench/v100_production_close_20260711/README.md`](../bench/v100_production_close_20260711/README.md)
- V100 packed MM4 BN/TN: [`../bench/v100_sm70_mm4_bntn_20260716/README.md`](../bench/v100_sm70_mm4_bntn_20260716/README.md)
- V100 full-FLA Qwen B1/B8: [`../bench/v100_active_b1b8_20260715/README.md`](../bench/v100_active_b1b8_20260715/README.md)
- RTX 3090 g1h 7.2B: [`../bench/3090_g1h_7p2_bsz8_20260714/README.md`](../bench/3090_g1h_7p2_bsz8_20260714/README.md)
- RTX 4090 g1h 7.2B: [`../bench/4090_g1h_7p2_bsz8_20260715/README.md`](../bench/4090_g1h_7p2_bsz8_20260715/README.md)
- RTX 4090 small models: [`../bench/4090_small_bsz8_20260715/README.md`](../bench/4090_small_bsz8_20260715/README.md)
- RTX 5090 full-FLA Qwen B1/B8: [`../bench/5090_g1h_qwen35_b1_b8_20260715/README.md`](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md)
- RTX 5090 latest g1h 13.3B: [`../bench/5090_g1h_13p3_20260715/README.md`](../bench/5090_g1h_13p3_20260715/README.md)
- RTX 5090 g1h BN/TN Tensor Core W4 matrix: [`../bench/5090_bntn_all_models_20260716/README.md`](../bench/5090_bntn_all_models_20260716/README.md)
- RTX 5090 MATH500 and quant pressure: [`../bench/5090_blackwell_production_close_20260712/README.md`](../bench/5090_blackwell_production_close_20260712/README.md)
- RTX 5090 official train_temp alignment: [`../bench/5090_train_temp_alignment_20260717/README.md`](../bench/5090_train_temp_alignment_20260717/README.md)
- RTX 5090 Native B16 train_temp alignment: [`../bench/5090_native_train_temp_b16_20260718/README.md`](../bench/5090_native_train_temp_b16_20260718/README.md)
- RTX 5090 Native fused decode: [`../bench/5090_native_decode_fused_20260718/README.md`](../bench/5090_native_decode_fused_20260718/README.md)
- RTX 5090 Native HF Gradio and unchanged official shell: [`../bench/5090_native_hf_gradio_train_temp_20260718/README.md`](../bench/5090_native_hf_gradio_train_temp_20260718/README.md)
- RTX 5070 Laptop: [`../bench/5070_qwen35_full_fla_bsz8_20260714/README.md`](../bench/5070_qwen35_full_fla_bsz8_20260714/README.md)
- Apple M5: [`hardware/APPLE_PRODUCTION_CLOSE.md`](hardware/APPLE_PRODUCTION_CLOSE.md)
- A100: [`validation/A100_HF_VALIDATION.md`](validation/A100_HF_VALIDATION.md)
- A800: [`validation/A800_HF_VALIDATION.md`](validation/A800_HF_VALIDATION.md)
- V100 training/compatibility: [`validation/V100_HF_VALIDATION.md`](validation/V100_HF_VALIDATION.md)
- Blackwell history: [`hardware/BLACKWELL_50SERIES.md`](hardware/BLACKWELL_50SERIES.md)

## Adding a card

A hardware PR must record exact device, driver/runtime versions, model and
dtype, commands, raw JSONL/logs, correctness checks, footprint/peak memory and
throughput. Promotion to production-close additionally requires a fail-closed
comparison gate and repeated/paired measurements where clock or process state
can bias results.
