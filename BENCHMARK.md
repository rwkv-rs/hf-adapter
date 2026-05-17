# Benchmarks

Last updated: **2026-08-17**.

This page contains only current promoted benchmark lines. The full Prefill /
Decode table is available in [Chinese](docs/QWEN35_LATEST_P_D_TOKPS.md) and
[English](docs/QWEN35_LATEST_P_D_TOKPS_EN.md), ordered by model size, GPU, and
B1/B8 with zero or one decimal place.

The exact retained artifact set is machine-readable in
[`bench/CURRENT_ARTIFACTS.json`](bench/CURRENT_ARTIFACTS.json).

## Qwen3.5 paired Prefill / Decode

| GPU | Model pairs | Cells | Raw Prefill | Raw Decode | Parameter-adjusted Prefill | Parameter-adjusted Decode | Evidence |
|---|---:|---:|---|---|---|---|---|
| Tesla V100 32GB | 4 | 48 | 48/48 PASS | 48/48 PASS | 48/48 PASS | 48/48 PASS | [artifact](bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| RTX 3090 | 4 | 48 | 48/48 PASS | 48/48 PASS | 48/48 PASS | 48/48 PASS | [artifact](bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| RTX 4080 | 3 | 36 | 36/36 PASS | 36/36 PASS | 36/36 PASS | 36/36 PASS | [artifact](bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| RTX 4090 | 4 | 48 | 48/48 PASS | 48/48 PASS | 48/48 PASS | 48/48 PASS | [artifact](bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 5090 | 4 | 48 | 48/48 PASS | 48/48 PASS | 48/48 PASS* | 48/48 PASS | [paired candidate](bench/5090_qwen35_paired_decode_v1_20260813/README.md), [Qwen reference](bench/5090_qwen35_best_optimized_hf_v1_20260813/README.md) |

`*` The RTX 5090 formal promotion gate is paired Decode. Prefill is separately
reported from the same retained RWKV candidate and frozen Qwen reference, and
all 48 recomputed raw and parameter-adjusted Prefill cells exceed 1.0x.

The comparison uses exact unrounded throughput and active-parameter adjustment:

```text
adjusted ratio = (RWKV tok/s / Qwen tok/s)
               * (RWKV active parameters / Qwen active parameters)
```

The smaller RWKV active-parameter count therefore multiplies the raw ratio by
a value below 1.0. The adjustment is not inverted.

## Current exact-card lines

| Platform | Current result | Evidence |
|---|---|---|
| Tesla V100 | Dense Native HF is production-close to the recorded Albatross reference; packed MM4 Decode and W4 Prefill have separate accepted lines | [dense](bench/v100_production_close_20260711/README.md), [MM4](bench/v100_sm70_mm4_bntn_20260716/README.md), [W4 Prefill](bench/v100_sm70_prefill_dequant_20260723/README.md) |
| RTX 4080 | 7.2B/B8 FP16-state Decode reaches **344.39 tok/s**, `1.0301x` FP32-state, with greedy **12,288/12,288** | [artifact](bench/4080_7p2b_fp16_state_20260809/README.md) |
| RTX 4080 | B8 grouped W/A/V projection improves 0.4B/1.5B/2.9B Decode with exact greedy alignment | [artifact](bench/4080_b8_projection_bmm_20260809/README.md) |
| RTX 4090 | Current exact Native routes and small/large B8 quantization matrices are retained | [routes](bench/4090_4080_routes_20260812/README.md), [small](bench/4090_small_bsz8_20260715/README.md), [7.2B](bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 5070 Laptop | Native exact-card prefill/decode and Native quant loading are retained; superseded RWKV FLA speed matrices were removed | [Native](bench/5070_max_perf_20260811/README.md), [quant loading](bench/5070_native_memory_loading_20260716/README.md) |
| RTX 5090 | Native-vs-official inference, W4 BN/TN, and real MiniPile train_temp lines pass | [inference](bench/5090_native_official_fp16_production_20260718/README.md), [W4](bench/5090_bntn_all_models_20260716/README.md), [training](bench/5090_native_train_temp_real_minipile_20260718/README.md) |
| Tesla T4 | Native HF/cache/quant/training exact-card scope passes | [artifact](bench/t4_production_close_20260720/README.md) |
| AMD gfx1100 | Native exact-card close and output-head quantization pass their declared scopes | [Native](bench/amd_gfx1100_full_close_20260730/README.md), [quantization](bench/amd_gfx1100_quant_20260728/README.md) |
| Apple M5 | Current B1/B8 paired rows and the compact production-close bundle are retained | [B1](bench/apple_bsz1_active_m5_20260715/README.md), [B8](bench/apple_bsz8_active_m5_20260714/README.md) |
| Moore Threads S70 | Native compatibility and the opt-in shift-mix kernel line pass | [compatibility](bench/musa_s70_validation_20260728/README.md), [kernel](bench/musa_s70_shift_mix_20260728/README.md) |

## Runtime and correctness rules

- Timing is CUDA-event or platform-equivalent timing after warmup; current
  paired matrices use seven measured samples per cell.
- B8 throughput is aggregate tok/s.
- Current paired evidence uses B1/B8, P128/P512/P2048 and D128/D512.
- Performance promotion requires finite outputs, route telemetry, complete
  cell coverage, and the correctness contract recorded by each artifact.
- FLA remains an explicit RWKV compatibility/reference backend and current
  correctness oracle. Native is the retained RWKV performance route.
- Qwen may use its own official optimized operators; this does not make FLA the
  RWKV production backend.

## Reproduce

Start from the artifact README for the exact model hashes, environment lock,
route flags, and validator command. Current shared entry points are listed in
[`bench/INDEX.md`](bench/INDEX.md).

For the consolidated throughput table and arithmetic checks:

```bash
python -m pytest -q tests/test_qwen35_comparison_layout.py
python -m pytest -q tests/test_current_benchmark_artifacts.py
python tests/test_markdown_links.py
```
