# RWKV-7 Hugging Face adapter status

Canonical current snapshot. Scope: Transformers loading/generation/training,
PEFT/TRL, recurrent cache helpers, quantized HF inference, hardware validation
and reproducible performance evidence. Native vLLM/SGLang work is separate.

Last updated: **2026-07-14**.

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
| Universal W8/W4 speed | **PARTIAL** | selected V100/4090/5090 speed lanes pass; full-memory quant remains open |
| Production performance | **PARTIAL / strong card-local closes** | V100, measured 4090 lanes, RTX 5090 and Apple M5 have promoted artifacts; RTX 5070 1.5B RWKV vs verified-Qwen-FLA 2B now passes the exact-card speed and memory gates in all 72/72 cells |
| Full common-card coverage | **PARTIAL** | H100, AMD/ROCm, Turing and broader Apple/50-series evidence remain open |
| PP/TP | **PARTIAL** | HF multi-device/device-map smoke exists; production TP matrix is not closed |
| Speculative decoding | **EXPERIMENTAL PASS** | HF-compatible harness and Apple target-greedy oracle evidence exist |

## Hardware summary

| Platform | Status | Canonical evidence / boundary |
|---|---|---|
| V100 32GB | **Production-close for RWKV/Albatross lanes** | Dense Albatross P1 plus native W8/W4 speed lane; Qwen3.5 Torch-fallback matrices are historical diagnostics rather than optimized-Qwen acceptance; [`bench/v100_production_close_20260711/`](bench/v100_production_close_20260711/README.md), [`bench/qwen35_v100_hf_matrix_20260712/`](bench/qwen35_v100_hf_matrix_20260712/README.md), [`bench/v100_qwen35_full_matrix_20260713/`](bench/v100_qwen35_full_matrix_20260713/README.md) |
| RTX 4090 | **Production-close for measured lanes** | All measured 0.4B decode batches pass; current-session bsz4 prefill passes; historical high-water remains |
| RTX 5070 Laptop | **Production-close for measured Qwen FLA performance lane** | 1.5B RWKV vs 2B Qwen: 144/144 raw rows and 72/72 strict cells pass; minimum prefill/decode speedups are `1.109682x`/`1.466175x`, and footprint/peak VRAM are no larger in 72/72; Qwen convolution remains Torch fallback and model quality is not covered; [`bench/5070_qwen35_fla_native_prefill_20260714/`](bench/5070_qwen35_fla_native_prefill_20260714/README.md) |
| RTX 5090 | **Production-close** | Quant pressure, 13.3B low-memory conversion and full MATH500; [`bench/5090_blackwell_production_close_20260712/`](bench/5090_blackwell_production_close_20260712/README.md) |
| Apple M5 | **Production-close for measured MLX pairs** | Selected Qwen3.5 comparison gates and CoreML state correctness; [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md) |
| A100 40GB / A800 80GB / A6000 48GB | **Validated** | Large-model API/training/quant/ZeRO matrices; production performance remains card-specific |
| GTX 1080 Ti | **Smoke** | compatibility evidence, not full production-close |
| H100 / AMD / Turing | **Open** | real-card matrix required |

Full matrix: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md).

## Current release blockers

These are the remaining technical boundaries, not already-completed history:

1. Full-memory fused W8/W4 projection/prefill must become fp16-or-faster across
   old and new cards while retaining the larger footprint reduction.
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
