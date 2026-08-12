# RWKV-7 vs Qwen3.5: Complete Parameter and Speed Comparison

Updated: **2026-08-12**. All numbers come from promoted same-device evidence
on the current main branch. See [`BENCHMARK.md`](../BENCHMARK.md) for the full
history, quantization lanes, and cell-level telemetry. [中文版](QWEN35_SPEED_COMPARISON_ZH.md)

## Results at a glance

> The promoted NVIDIA dense-FP16 comparison contains **32 measured
> GPU/model/batch combinations**. The main table also carries an explicit empty
> row for the **unmeasured RTX 4090 7.2B/9B B1 gap**, plus **3 Apple M5
> target-only W4 combinations**. Every measured row
> has RWKV-7 ahead of Qwen3.5 in median raw Prefill and Decode throughput.
> Raw Prefill/Decode reaches **7.90x / 20.51x**. After discounting the natural
> speed advantage of the smaller model, parameter-size-adjusted
> Prefill/Decode reaches **4.73x / 12.29x**.

**RTX 4080 is now complete at cell level: all 36/36 parameter-size-adjusted
Prefill cells and 36/36 Decode cells exceed `1.00x`; the minima are
`1.068520x / 1.140700x`.**

**RTX 3090 now also closes the latest g1d/g1i checkpoint matrix: every one of
the 24 B1/B8, P128/P512/P2048 cells has parameter-adjusted Prefill `>=1.00x`
against fail-closed full-FLA Qwen3.5; the minimum/median is now
`1.227477x/1.467758x`.**

**RTX 4090 now closes its latest 0.4B/1.5B/2.9B matrix at cell level: all
36/36 parameter-size-adjusted Prefill cells and 36/36 Decode cells exceed
`1.00x`; the minima are `1.108265x / 4.158943x`.**

- `1.02x` means RWKV throughput is 1.02 times Qwen throughput, or about 2%
  faster.
- Prefill processes the input prompt. Decode generates tokens one at a time
  and is the closer match for sustained chat generation.
- The NVIDIA table uses **raw dense-FP16 tok/s** and also reports
  parameter-size-adjusted speed. Except for the explicitly labeled V100 and
  latest RTX 3090/5090 shapes, `6 cells` means the median across
  `P128/512/2048 × D128/512`; the latest RTX 3090/5090 `3 cells` use
  `P128/512/2048 × D128`.
- These are inference throughput comparisons. They do not claim that one
  model has better instruction following, reasoning, coding, multilingual, or
  other task quality; those require separate evaluation rows.

## Parameter accounting

The model names are release tiers. Active parameter counts recorded by
benchmark telemetry are shown in billions, rounded to three decimal places:

| Model pair (RWKV / Qwen3.5) | RWKV active params | Qwen active params |
|---|---:|---:|
| 0.4B / 0.8B | `0.451B` | `0.752B` |
| 1.5B / 2B | `1.527B` | `1.882B` |
| 2.9B / 4B | `2.948B` | `4.206B` |
| 7.2B / 9B | `7.199B` | `8.954B` |

- **Raw speed ratio** = RWKV tok/s ÷ Qwen tok/s. This is the throughput seen
  directly by the user.
- **Parameter-size-adjusted speed ratio** linearly scales Qwen using the exact
  active-parameter counts retained in the artifacts. The table omits a separate
  parameter-ratio column and shows active parameters directly.
- Example: the latest RTX 4090 0.4B/0.8B B8 row has median raw Prefill `2.22x`
  and about `1.33x` after active-parameter adjustment.

## NVIDIA: complete promoted same-device matrix

The table lists every GPU, model pair, and batch combination in the promoted
optimized-Qwen evidence. `RWKV P / D tok/s` and `Qwen P / D tok/s` are the
separately computed median throughputs over the declared cells, rounded to three
decimal places. `Raw P / D` and `Adjusted P / D` are medians of the matched
cell-level ratios, so they need not equal a division of the two displayed
throughput medians.
The RTX 4090 7.2B/9B B1 row is retained with dashes to make clear that the
same-device measurement does not yet exist rather than being omitted from the
documentation.

For RTX 4080, the stricter cell-level gate now passes **36/36 adjusted
Prefill cells and 36/36 adjusted Decode cells**; the full-matrix minima are
`1.068520x / 1.140700x`.

For RTX 4090, the latest strict gate also passes **36/36 adjusted Prefill
cells and 36/36 adjusted Decode cells**; its minima are
`1.108265x / 4.158943x`.

| GPU | Model pair | Batch | Scope | RWKV active params | Qwen active params | RWKV P / D tok/s | Qwen P / D tok/s | Raw P / D | Adjusted P / D | Evidence |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V100 32GB | 1.5B / 2B | B1 | P512/D64 | 1.527B | 1.882B | **10,425.596 / 151.357** | **3,702.375 / 25.596** | **2.82x / 5.91x** | **2.29x / 4.80x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| V100 32GB | 1.5B / 2B | B8 | P512/D64 | 1.527B | 1.882B | **20,729.017 / 816.606** | **3,833.197 / 154.941** | **5.41x / 5.27x** | **4.39x / 4.28x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| RTX 3090 | 0.4B / 0.8B | B1 | 3 cells | 0.451B | 0.752B | **29,368.244 / 293.131** | **7,155.265 / 26.529** | **4.10x / 11.05x** | **2.46x / 6.62x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 0.4B / 0.8B | B8 | 3 cells | 0.451B | 0.752B | **78,949.489 / 1,691.636** | **32,678.327 / 213.479** | **2.47x / 7.93x** | **1.48x / 4.75x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 1.5B / 2B | B1 | 3 cells | 1.527B | 1.882B | **17,641.354 / 164.035** | **8,528.864 / 28.516** | **2.12x / 5.75x** | **1.72x / 4.67x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 1.5B / 2B | B8 | 3 cells | 1.527B | 1.882B | **29,162.697 / 984.864** | **16,416.432 / 220.473** | **1.66x / 4.47x** | **1.34x / 3.63x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 2.9B / 4B | B1 | 3 cells | 2.948B | 4.206B | **11,774.089 / 88.681** | **5,657.408 / 19.247** | **2.08x / 4.61x** | **1.46x / 3.23x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 2.9B / 4B | B8 | 3 cells | 2.948B | 4.206B | **15,776.063 / 596.485** | **7,093.916 / 150.580** | **2.14x / 3.96x** | **1.50x / 2.78x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 7.2B / 9B | B1 | 3 cells | 7.199B | 8.954B | **5,763.950 / 46.434** | **3,616.109 / 19.718** | **1.63x / 2.35x** | **1.31x / 1.89x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 7.2B / 9B | B8 | 3 cells | 7.199B | 8.954B | **6,632.697 / 341.752** | **4,155.688 / 164.172** | **1.60x / 2.08x** | **1.28x / 1.67x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 4080 | 0.4B / 0.8B | B1 | 6 cells, all pass | 0.451B | 0.752B | **45,537.844 / 492.031** | **24,889.204 / 100.247** | **1.83x / 4.91x** | **1.10x / 2.94x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 0.4B / 0.8B | B8 | 6 cells, all pass | 0.451B | 0.752B | **103,570.967 / 3,205.784** | **50,003.817 / 768.019** | **1.98x / 4.17x** | **1.19x / 2.50x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 1.5B / 2B | B1 | 6 cells, all pass | 1.527B | 1.882B | **30,857.745 / 193.892** | **19,871.050 / 101.785** | **1.55x / 1.90x** | **1.26x / 1.55x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 1.5B / 2B | B8 | 6 cells, all pass | 1.527B | 1.882B | **38,144.151 / 1,356.277** | **21,602.088 / 765.144** | **1.76x / 1.77x** | **1.43x / 1.44x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 2.9B / 4B | B1 | 6 cells, all pass | 2.948B | 4.206B | **14,276.348 / 102.670** | **8,818.521 / 62.804** | **1.75x / 1.63x** | **1.22x / 1.15x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 2.9B / 4B | B8 | 6 cells, all pass | 2.948B | 4.206B | **19,517.145 / 729.021** | **9,824.341 / 415.948** | **1.99x / 1.75x** | **1.40x / 1.23x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4090 | 0.4B / 0.8B | B1 | 6 cells, all pass | 0.451B | 0.752B | **63,022.409 / 584.850** | **8,634.647 / 28.521** | **7.90x / 20.51x** | **4.73x / 12.29x** | [4090 latest P/D](../bench/4090_adjusted_pd_20260812/README.md) |
| RTX 4090 | 0.4B / 0.8B | B8 | 6 cells, all pass | 0.451B | 0.752B | **144,237.564 / 3,842.216** | **65,764.741 / 215.563** | **2.22x / 17.85x** | **1.33x / 10.69x** | [4090 latest P/D](../bench/4090_adjusted_pd_20260812/README.md) |
| RTX 4090 | 1.5B / 2B | B1 | 6 cells, all pass | 1.527B | 1.882B | **36,206.083 / 251.560** | **8,787.079 / 28.968** | **4.12x / 8.68x** | **3.34x / 7.05x** | [4090 latest P/D](../bench/4090_adjusted_pd_20260812/README.md) |
| RTX 4090 | 1.5B / 2B | B8 | 6 cells, all pass | 1.527B | 1.882B | **57,115.628 / 1,717.617** | **37,024.909 / 219.174** | **1.54x / 7.84x** | **1.25x / 6.36x** | [4090 latest P/D](../bench/4090_adjusted_pd_20260812/README.md) |
| RTX 4090 | 2.9B / 4B | B1 | 6 cells, all pass | 2.948B | 4.206B | **19,152.772 / 136.274** | **6,237.235 / 20.552** | **3.64x / 6.63x** | **2.55x / 4.64x** | [4090 latest P/D](../bench/4090_adjusted_pd_20260812/README.md) |
| RTX 4090 | 2.9B / 4B | B8 | 6 cells, all pass | 2.948B | 4.206B | **28,454.425 / 954.106** | **14,954.801 / 160.094** | **1.90x / 5.96x** | **1.33x / 4.18x** | [4090 latest P/D](../bench/4090_adjusted_pd_20260812/README.md) |
| RTX 4090 | 7.2B / 9B | B1 | **not measured** | 7.199B | 8.954B | — | — | — | — | [current evidence is B8 only](../bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 4090 | 7.2B / 9B | B8 | 6 cells | 7.199B | 8.954B | **9,453.237 / 448.603** | **8,441.540 / 201.751** | **1.12x / 2.22x** | **0.90x / 1.79x** | [4090 7.2B](../bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 5070 Laptop | 1.5B / 2B | B8 | 6 cells | 1.527B | 1.882B | **10,769.749 / 690.089** | **8,239.454 / 268.649** | **1.33x / 2.62x** | **1.08x / 2.13x** | [5070](../bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| RTX 5090 | 0.4B / 0.8B | B1 | 3 cells | 0.451B | 0.752B | **58,104.948 / 1,121.486** | **15,886.187 / 56.664** | **3.86x / 19.79x** | **2.31x / 11.85x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 1.5B / 2B | B1 | 3 cells | 1.527B | 1.882B | **33,697.614 / 547.344** | **15,795.251 / 56.667** | **2.16x / 9.63x** | **1.75x / 7.82x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 2.9B / 4B | B1 | 3 cells | 2.948B | 4.206B | **21,787.270 / 309.185** | **11,794.854 / 41.328** | **1.87x / 7.49x** | **1.31x / 5.25x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 7.2B / 9B | B1 | 3 cells | 7.199B | 8.954B | **14,875.687 / 145.995** | **10,651.870 / 41.721** | **1.42x / 3.50x** | **1.14x / 2.81x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 0.4B / 0.8B | B8 | 3 cells | 0.451B | 0.752B | **206,364.189 / 3,431.711** | **93,885.606 / 429.382** | **2.24x / 7.99x** | **1.34x / 4.79x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 1.5B / 2B | B8 | 3 cells | 1.527B | 1.882B | **82,339.449 / 2,060.857** | **50,353.472 / 434.033** | **1.43x / 4.77x** | **1.16x / 3.87x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 2.9B / 4B | B8 | 3 cells | 2.948B | 4.206B | **37,325.812 / 1,247.143** | **22,253.241 / 317.418** | **1.69x / 3.92x** | **1.19x / 2.75x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 7.2B / 9B | B8 | 3 cells | 7.199B | 8.954B | **19,624.283 / 867.325** | **12,261.806 / 318.630** | **1.54x / 2.72x** | **1.24x / 2.19x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |

This complete table preserves every promoted GPU/model/batch result and makes
the raw and parameter-size-adjusted ratios directly comparable.

### RTX 3090 latest-checkpoint strict gate

The latest RTX 3090 artifact uses RWKV-7 g1d 0.4B and 2026-08-05 g1i
1.5B/2.9B/7.2B against official Qwen3.5 0.8B/2B/4B/9B. It checks every
B1/B8 and P128/P512/P2048 cell independently at D128. All `24/24` Qwen rows
verify FLA, Triton causal convolution, live fused bindings and the full-fused
contract.

The strict prefill gate passes `24/24`: raw Prefill minimum/median is
`1.531589x/2.076170x`, while parameter-adjusted Prefill minimum/median is
`1.227477x/1.467758x`. Raw Decode minimum/median is
`2.069838x/4.524636x`, and adjusted Decode minimum/median is
`1.664218x/3.433680x`. The narrowest adjusted cell is 0.4B/0.8B B8/P512,
where RWKV delivers `78,949.489 tok/s` versus Qwen `38,534.012 tok/s`, or
`1.227477x` after parameter adjustment.

The exact-shape FP16-accumulation oracle passes `25/25` direct and
chunk-carried prompt/cache-handoff rows at cosine `>=0.9999` with exact greedy
tokens. The promoted route is restricted to exact RTX 3090 model, batch and
token-block shapes. See the
[immutable evidence](../bench/3090_g1i_qwen35_maxperf_20260812/README.md).

### RTX 4090 latest-checkpoint strict gate

The latest RTX 4090 artifact compares RWKV-7 g1d 0.4B and g1i 1.5B/2.9B
with official Qwen3.5 0.8B/2B/4B across B1/B8,
P128/P512/P2048 and D128/D512. All `36/36` Qwen rows verify FLA chunk Gated
DeltaNet, fused-recurrent Decode, fused gated normalization, and the
repository Triton causal-convolution kernels.

Every cell passes both strict gates: adjusted Prefill is `36/36` with global
minimum/median `1.108265x/2.306890x`, and adjusted Decode is `36/36` with
global minimum/median `4.158943x/6.693394x`. The former red
1.5B/B1/P2048 cells now use an exact-card tile-16 self-chunk plus stacked-R/K/V
route, which reaches `1.2539x` its local control. Its forward/reverse A/B
passes Prompt/Decode cosine `>=0.9999`, greedy-token equality, and cache
handoff. See the
[immutable evidence](../bench/4090_adjusted_pd_20260812/README.md).

### RTX 5090 latest-checkpoint strict gate

The latest RTX 5090 rows use RWKV-7 g1d 0.4B plus the 2026-08-05 g1i
1.5B/2.9B/7.2B checkpoints against official Qwen3.5 0.8B/2B/4B/9B. All 24
Qwen reference cells verify FLA, Triton causal convolution, live fused
bindings, and the full-fused contract.

Unlike the row medians above, the strict gate checks every B1/B8 and
P128/P512/P2048 cell independently. All `24/24` cells pass: raw Prefill has
minimum/median `1.347871x/1.819072x`, and parameter-adjusted Prefill has
minimum/median `1.072987x/1.317515x`. Raw Decode has minimum/median
`2.710952x/6.104568x`, while parameter-adjusted Decode has minimum/median
`2.179692x/4.330813x`.

The 0.4B/B1/P2048 candidate reaches `61,343.8 tok/s`, `2.2495x` its prior
candidate row. The graph-versus-eager P2048 oracle passes `8/8` model/batch
rows with prompt/post-cache-handoff cosine minima
`0.99999988/0.99999994` and exact greedy tokens. Removing the negative 7.2B
stacked-RKV route lowers its candidate peak from `17.4-18.6 GiB` to
`14.3-15.5 GiB`. See the
[immutable evidence](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md).

### Apple M5: complete target-only W4 comparison

Apple MLX W4 is shown separately so that backend and precision remain
consistent within each table. Concrete throughput columns are aggregate tok/s
medians, rounded to three decimal places.

| Model pair | Batch / shape | RWKV active params | Qwen active params | RWKV P / D tok/s | Qwen P / D tok/s | Raw P / D | Adjusted P / D | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.4B / 0.8B | B8, cold, P512 chars/D64 | 0.451B | 0.752B | **11,650.464 / 992.304** | **5,702.266 / 487.152** | **2.04x / 2.04x** | **1.22x / 1.22x** | [M5 B8](../bench/apple_bsz8_active_m5_20260714/README.md) |
| 1.5B / 2B | B1, P512 chars/D64 | 1.527B | 1.882B | **2,126.058 / 129.152** | **1,272.860 / 89.941** | **1.67x / 1.44x** | **1.36x / 1.17x** | [M5 B1](../bench/apple_bsz1_active_m5_20260715/README.md) |
| 1.5B / 2B | B8, cold, P512 chars/D64 | 1.527B | 1.882B | **2,249.150 / 185.593** | **1,600.504 / 132.205** | **1.41x / 1.40x** | **1.14x / 1.14x** | [M5 B8](../bench/apple_bsz8_active_m5_20260714/README.md) |

## AMD and other hardware

Beyond the NVIDIA and Apple comparisons above, the repository covers AMD
ROCm, Turing/Ampere/Hopper, and dedicated accelerator backends.

| Platform | Implemented and validated capabilities | Entry point |
|---|---|---|
| AMD Navi 31 / `gfx1100` | Native HF, cache, chunked Prefill, PEFT, BF16 Trainer, fused Decode from 0.1B through 13.3B, and 40/40 output-head W8/W4 Decode rows | [AMD ROCm validation](validation/AMD_ROCM_HF_VALIDATION.md) |
| AMD MI series, `gfx1101/gfx1102` | Portable ROCm/HIP path, architecture detection, and portable dispatch | [Hardware matrix](HARDWARE_MATRIX.md) |
| Apple M5 | Target-only W4 comparisons for 0.4B/0.8B and 1.5B/2B, plus MLX, MPS, and CoreML workflows | [Apple guide](APPLE_USAGE.md) |
| NVIDIA T4 | Native HF, quantization, training, and production-close validation | [T4 evidence](../bench/t4_production_close_20260720/README.md) |
| NVIDIA A100/A800 | Ampere CUDA, training, parallel, and HF workflows | [Hardware matrix](HARDWARE_MATRIX.md) |
| NVIDIA H100 | Hopper CUDA, Transformers/HF, and benchmark execution path | [Performance guide](PERFORMANCE.md) |
| Ascend, Biren, MetaX, MUSA | Dedicated backends, runtime integration, and compatibility validation | [Hardware matrix](HARDWARE_MATRIX.md) |

### AMD `gfx1100` measured throughput

FP16, P128, cached decode. `Speedup` compares the fused RWKV route with the
generic RWKV route.

| RWKV-7 | B1 Decode | B8 aggregate Decode | Fused / generic speedup (B1 / B8) |
|---|---:|---:|---:|
| 0.1B | 347.1 tok/s | 2,666.5 tok/s | `1.88x / 2.04x` |
| 0.4B | 141.8 tok/s | 1,073.2 tok/s | `1.75x / 1.74x` |
| 1.5B | 71.3 tok/s | 514.2 tok/s | `1.40x / 1.47x` |
| 2.9B | 47.7 tok/s | 353.0 tok/s | `1.37x / 1.41x` |
| 7.2B | 29.7 tok/s | 213.9 tok/s | `1.23x / 1.29x` |
| 13.3B | 15.5 tok/s | 113.2 tok/s | `1.21x / 1.29x` |

The gfx1100 output-head W8/W4 route is faster than matching RWKV FP16 in all
40/40 Decode rows across 0.4B–13.3B and B1/B2/B4/B8. See the
[AMD validation guide](validation/AMD_ROCM_HF_VALIDATION.md),
[fused Decode evidence](../bench/amd_gfx1100_fused_decode_20260728/README.md),
and [0.4B–13.3B regression evidence](../bench/amd_gfx1100_rebase_validation_20260728/README.md).

## Comparison contract

- NVIDIA rows use the same GPU, batch size, prompt/decode lengths, and FP16
  precision. Apple rows use each model's promoted MLX W4 route.
- Every NVIDIA Qwen3.5 row records and verifies the optimized
  **FLA + Triton causal-conv** path and fused operator bindings.
- NVIDIA RWKV uses repository Native prefill and native-graph cached decode.
  Apple uses target-only MLX W4 for both sides.
- RTX 4080 uses each model's validated optimized runtime: PyTorch 2.11 with
  exact-shape FP16 accumulation for RWKV, and PyTorch 2.6 with full FLA for
  Qwen. Versions and backend telemetry are recorded; GPU, shapes, batch, and
  FP16 precision are identical.
- Release tiers are paired directly, such as 7.2B versus 9B. The tables expose
  raw tok/s, exact active parameter counts, and parameter-size-adjusted speed.
- The NVIDIA table is consistently dense FP16. The Apple table is consistently
  MLX W4.

## Live GPU reproduction

The commands below reload RWKV-7 and Qwen3.5 on the GPU, run warmup and timed
measurements, and regenerate Prefill, Decode, parameter-size-adjusted speed,
and backend-binding results.

### 1. Prepare the environment and models

Match PyTorch, CUDA, Triton, Transformers, FLA, and bitsandbytes to the
`environment.json` or `environment.txt` in the corresponding evidence
directory. Use the validated environment for each GPU.

```bash
git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
git checkout 28f724259f8438cfcc71de40cf33889c6cf2396e

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[cuda,fla-reference,quant,torchao]"
```

Prepare two local directories:

1. an HF model converted from an official RWKV-7 `.pth` checkpoint with
   [`scripts/convert_rwkv7_to_hf.py`](../scripts/convert_rwkv7_to_hf.py);
2. the official Qwen3.5 HF model directory.

See the [user guide](USER_GUIDE.md) for conversion instructions, then check
the RWKV directory:

```bash
python examples/check_environment.py --model /path/to/rwkv7-model-hf
```

The output should contain `RESULT: READY` and `[PASS] Model directory`.

### 2. Complete RTX 4090 example: 1.5B versus 2B, B8

```bash
OUT=/tmp/rwkv-qwen35-4090-b8

PYTHON_BIN=python \
BATCH_SIZES=8 \
PREFILL_CHUNK_SIZE=512 \
  bench/run_4090_qwen35_pair_acceptance.sh \
  rwkv-1.5b__qwen3.5-2b \
  /path/to/rwkv7-g1h-1.5b-hf \
  /path/to/Qwen3.5-2B \
  "$OUT"

test "$(cat "$OUT/pipeline_exit_code.txt")" = 0
python - "$OUT/summary_active_work.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
speed = summary["speed"]
adjusted = summary["active_parameter_work"]
print(
    "raw prefill/decode median:",
    speed["median_prefill_speedup"],
    speed["median_decode_speedup"],
)
print(
    "parameter-adjusted prefill/decode:",
    adjusted["median_prefill_throughput_ratio"],
    adjusted["median_decode_throughput_ratio"],
)
print("red cells:", len(summary["red_cells"]))
PY
```

A complete run reports exit code 0, `pipeline_exit_code.txt=0`,
`red cells: 0`, and the Qwen full-FLA path.

### 3. Other NVIDIA GPU entry points

| GPU | Live measurement entry point | Scope |
|---|---|---|
| V100 | [Commands in the V100 evidence](../bench/v100_active_b1b8_20260715/README.md#reproduce) | 1.5B/2B, B1/B8 |
| RTX 3090 latest checkpoints | [`bench/run_3090_adjusted_prefill_pd.sh`](../bench/run_3090_adjusted_prefill_pd.sh) | Four model pairs, B1/B8, P128/512/2048, D128; strict per-cell adjusted-Prefill gate plus 25 correctness rows |
| RTX 4080 | [`bench/run_4080_adjusted_pd.sh`](../bench/run_4080_adjusted_pd.sh) | Runs all three pairs at B1/B8 and requires adjusted P/D `>1.00x` in every one of the 36 cells |
| RTX 4090 latest checkpoints | [`bench/run_4090_adjusted_pd.sh`](../bench/run_4090_adjusted_pd.sh) | Three model pairs, B1/B8, P128/512/2048, D128/512; requires adjusted P/D `>1.00x` in all 36 cells |
| RTX 5070 Laptop | [`bench/run_5070_qwen35_full_fla_bsz8.ps1`](../bench/run_5070_qwen35_full_fla_bsz8.ps1) | PowerShell with `-RwkvModel`, `-QwenModel`, and `-OutDir` |
| RTX 5090 | [`bench/run_5090_qwen35_full_matrix.sh`](../bench/run_5090_qwen35_full_matrix.sh) | Four model pairs, B1/B8 full matrix |
| RTX 5090 latest checkpoints | [Commands in the strict-gate evidence](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md#reproduce-the-gate) | Four model pairs, B1/B8, P128/512/2048, D128 |

Each runner verifies the exact GPU, backend bindings, matrix coverage, and
acceptance gates, and writes `pipeline_exit_code.txt`,
`matrix_failures.txt`, `summary*.json`, and full logs.

### 4. Apple M5 live GPU measurement

Use the MLX environment and local W4 model directory:

```bash
PYTHON_BIN=/path/to/python \
MODEL_ROOT=/path/to/models \
COOLDOWN_SECONDS=30 \
INITIAL_COOLDOWN_SECONDS=60 \
  scripts/run_apple_bsz8_target_only_acceptance.sh
```

The script runs RWKV-7 0.4B/1.5B and Qwen3.5 0.8B/2B on the Apple GPU and
reports raw Prefill/Decode, parameter-size-adjusted ratios, peak memory, and
token consistency.

### 5. AMD `gfx1100` live GPU measurement

Use a PyTorch environment for ROCm 7.2.1 and make `/dev/kfd` and
`/dev/dri/render*` available:

```bash
OUT=/tmp/rwkv7-amd-gfx1100
mkdir -p "$OUT"
set -o pipefail

bash bench/run_amd_rocm_hf_validation.sh \
  HF_DIR=/path/to/rwkv7-g1d-0.1b-hf \
  OUT_DIR="$OUT" |& tee "$OUT/console.log"

grep -F "AMD ROCm HF VALIDATION PASS" "$OUT/console.log"
```

The runner verifies HIP visibility and `gcnArchName`; exact `gfx1100` selects
the promoted fused route and architecture tuning. To reproduce only the fused
Decode A/B, use the [focused evidence command](../bench/amd_gfx1100_fused_decode_20260728/README.md#reproduce).
