# RWKV-7 vs Qwen3.5: Complete Parameter and Speed Comparison

Updated: **2026-08-12**. All numbers come from promoted same-device evidence
on the current main branch. See [`BENCHMARK.md`](../BENCHMARK.md) for the full
history, quantization lanes, and cell-level telemetry. [中文版](QWEN35_SPEED_COMPARISON_ZH.md)

## Results at a glance

> The promoted NVIDIA dense-FP16 comparison contains **29 GPU/model/batch
> combinations**, plus **3 Apple M5 target-only W4 combinations**. Every row
> has RWKV-7 ahead of Qwen3.5 in median raw Prefill and Decode throughput.
> Raw Prefill/Decode reaches **5.41x / 19.79x**. After discounting the natural
> speed advantage of the smaller model, parameter-size-adjusted
> Prefill/Decode reaches **4.39x / 11.85x**. In the more direct Prefill + Decode
> end-to-end result, all **116/116 measured NVIDIA cells** and all **29/29
> combination medians** beat Qwen3.5 both raw and after parameter-size
> adjustment; all **3/3 Apple M5 combinations** also remain ahead after
> adjustment.

**RTX 4080 is now complete at cell level: all 36/36 parameter-size-adjusted
Prefill cells and 36/36 Decode cells exceed `1.00x`; the minima are
`1.068520x / 1.140700x`.**

**RTX 3090 now also closes the latest g1d/g1i checkpoint matrix: every one of
the 24 B1/B8, P128/P512/P2048 cells has parameter-adjusted Prefill `>=1.00x`
against fail-closed full-FLA Qwen3.5; the minimum/median is now
`1.227477x/1.467758x`.**

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

The model names are release tiers. The exact active parameter counts recorded
by benchmark telemetry are:

| Model pair (RWKV / Qwen3.5) | Exact RWKV active params | Exact Qwen active params | RWKV/Qwen param ratio |
|---|---:|---:|---:|
| 0.4B / 0.8B | `450,767,872` | `752,393,024` | `0.599112` |
| 1.5B / 2B | `1,527,404,544` | `1,881,825,088` | `0.811661` |
| 2.9B / 4B | `2,947,735,040` | `4,205,751,296` | `0.700882` |
| 7.2B / 9B | `7,199,141,888` | `8,953,803,264` | `0.804032` |

- **Raw speed ratio** = RWKV tok/s ÷ Qwen tok/s. This is the throughput seen
  directly by the user.
- **Parameter-size-adjusted speed ratio** = raw speed ratio × RWKV/Qwen active
  parameter ratio. This linearly scales Qwen to the RWKV active parameter size
  and discounts the natural speed advantage of the smaller model.
- **End-to-end speed ratio** = (Qwen Prefill time + Qwen Decode time) ÷ (RWKV
  Prefill time + RWKV Decode time). **Parameter-size-adjusted E2E ratio** = raw
  E2E ratio × RWKV/Qwen active parameter ratio. E2E time in this guide covers
  the timed inference phase and excludes model loading.
- Example: RTX 4090, 0.4B/0.8B, B8 has raw Prefill `1.75x` and parameter ratio
  `0.599112`, giving `1.75 × 0.599112 ≈ 1.05x` after adjustment.

## NVIDIA: complete promoted same-device matrix

The table lists every GPU, model pair, and batch combination in the promoted
optimized-Qwen evidence. `Raw P / D` and `Adjusted P / D` are median
Prefill/Decode ratios. `E2E raw / adjusted` is the median ratio after combining
Prefill and Decode time within every measured cell.

For RTX 4080, the stricter cell-level gate now passes **36/36 adjusted
Prefill cells and 36/36 adjusted Decode cells**; the full-matrix minima are
`1.068520x / 1.140700x`.

| GPU | Model pair | Batch | Scope | RWKV active params | Qwen active params | Param ratio | Raw P / D | Adjusted P / D | E2E raw / adjusted | Evidence |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| V100 32GB | 1.5B / 2B | B1 | P512/D64 | 1.527405B | 1.881825B | `0.811661` | **2.82x / 5.91x** | **2.29x / 4.80x** | **5.59x / 4.54x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| V100 32GB | 1.5B / 2B | B8 | P512/D64 | 1.527405B | 1.881825B | `0.811661` | **5.41x / 5.27x** | **4.39x / 4.28x** | **5.30x / 4.30x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| RTX 3090 | 0.4B / 0.8B | B1 | 3 cells | 0.450768B | 0.752393B | `0.599112` | **4.10x / 11.05x** | **2.46x / 6.62x** | **10.91x / 6.54x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 0.4B / 0.8B | B8 | 3 cells | 0.450768B | 0.752393B | `0.599112` | **2.47x / 7.93x** | **1.48x / 4.75x** | **7.46x / 4.47x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 1.5B / 2B | B1 | 3 cells | 1.527405B | 1.881825B | `0.811661` | **2.12x / 5.75x** | **1.72x / 4.67x** | **5.62x / 4.56x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 1.5B / 2B | B8 | 3 cells | 1.527405B | 1.881825B | `0.811661` | **1.66x / 4.47x** | **1.34x / 3.63x** | **4.14x / 3.36x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 2.9B / 4B | B1 | 3 cells | 2.947735B | 4.205751B | `0.700882` | **2.08x / 4.61x** | **1.46x / 3.23x** | **4.50x / 3.15x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 2.9B / 4B | B8 | 3 cells | 2.947735B | 4.205751B | `0.700882` | **2.14x / 3.96x** | **1.50x / 2.78x** | **3.72x / 2.61x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 7.2B / 9B | B1 | 3 cells | 7.199142B | 8.953803B | `0.804032` | **1.63x / 2.35x** | **1.31x / 1.89x** | **2.33x / 1.87x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 7.2B / 9B | B8 | 3 cells | 7.199142B | 8.953803B | `0.804032` | **1.60x / 2.08x** | **1.28x / 1.67x** | **1.99x / 1.60x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 4080 | 0.4B / 0.8B | B1 | 6 cells, all pass | 0.450768B | 0.752393B | `0.599112` | **1.83x / 4.91x** | **1.10x / 2.94x** | **4.82x / 2.88x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 0.4B / 0.8B | B8 | 6 cells, all pass | 0.450768B | 0.752393B | `0.599112` | **1.98x / 4.17x** | **1.19x / 2.50x** | **4.04x / 2.42x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 1.5B / 2B | B1 | 6 cells, all pass | 1.527405B | 1.881825B | `0.811661` | **1.55x / 1.90x** | **1.26x / 1.55x** | **1.90x / 1.54x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 1.5B / 2B | B8 | 6 cells, all pass | 1.527405B | 1.881825B | `0.811661` | **1.76x / 1.77x** | **1.43x / 1.44x** | **1.77x / 1.44x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 2.9B / 4B | B1 | 6 cells, all pass | 2.947735B | 4.205751B | `0.700882` | **1.75x / 1.63x** | **1.22x / 1.15x** | **1.63x / 1.15x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 2.9B / 4B | B8 | 6 cells, all pass | 2.947735B | 4.205751B | `0.700882` | **1.99x / 1.75x** | **1.40x / 1.23x** | **1.77x / 1.24x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4090 | 0.4B / 0.8B | B8 | 6 cells | 0.450768B | 0.752393B | `0.599112` | **1.75x / 12.15x** | **1.05x / 7.28x** | **11.46x / 6.86x** | [4090 small](../bench/4090_small_bsz8_20260715/README.md) |
| RTX 4090 | 1.5B / 2B | B8 | 6 cells | 1.527405B | 1.881825B | `0.811661` | **1.11x / 5.66x** | **0.90x / 4.59x** | **5.30x / 4.30x** | [4090 small](../bench/4090_small_bsz8_20260715/README.md) |
| RTX 4090 | 2.9B / 4B | B8 | 6 cells | 2.947735B | 4.205751B | `0.700882` | **1.42x / 4.24x** | **1.00x / 2.97x** | **3.99x / 2.80x** | [4090 small](../bench/4090_small_bsz8_20260715/README.md) |
| RTX 4090 | 7.2B / 9B | B8 | 6 cells | 7.199142B | 8.953803B | `0.804032` | **1.12x / 2.22x** | **0.90x / 1.79x** | **2.11x / 1.69x** | [4090 7.2B](../bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 5070 Laptop | 1.5B / 2B | B8 | 6 cells | 1.527405B | 1.881825B | `0.811661` | **1.33x / 2.62x** | **1.08x / 2.13x** | **2.48x / 2.02x** | [5070](../bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| RTX 5090 | 0.4B / 0.8B | B1 | 3 cells | 0.450768B | 0.752393B | `0.599112` | **3.86x / 19.79x** | **2.31x / 11.85x** | **18.63x / 11.16x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 1.5B / 2B | B1 | 3 cells | 1.527405B | 1.881825B | `0.811661` | **2.16x / 9.63x** | **1.75x / 7.82x** | **9.34x / 7.58x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 2.9B / 4B | B1 | 3 cells | 2.947735B | 4.205751B | `0.700882` | **1.87x / 7.49x** | **1.31x / 5.25x** | **7.13x / 5.00x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 7.2B / 9B | B1 | 3 cells | 7.199142B | 8.953803B | `0.804032` | **1.42x / 3.50x** | **1.14x / 2.81x** | **3.43x / 2.76x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 0.4B / 0.8B | B8 | 3 cells | 0.450768B | 0.752393B | `0.599112` | **2.24x / 7.99x** | **1.34x / 4.79x** | **7.61x / 4.56x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 1.5B / 2B | B8 | 3 cells | 1.527405B | 1.881825B | `0.811661` | **1.43x / 4.77x** | **1.16x / 3.87x** | **4.48x / 3.63x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 2.9B / 4B | B8 | 3 cells | 2.947735B | 4.205751B | `0.700882` | **1.69x / 3.92x** | **1.19x / 2.75x** | **3.68x / 2.58x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 7.2B / 9B | B8 | 3 cells | 7.199142B | 8.953803B | `0.804032` | **1.54x / 2.72x** | **1.24x / 2.19x** | **2.54x / 2.05x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |

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
`2.179692x/4.330813x`. Raw and adjusted E2E are also above `1.00x` in all
`24/24` cells.

The 0.4B/B1/P2048 candidate reaches `61,343.8 tok/s`, `2.2495x` its prior
candidate row. The graph-versus-eager P2048 oracle passes `8/8` model/batch
rows with prompt/post-cache-handoff cosine minima
`0.99999988/0.99999994` and exact greedy tokens. Removing the negative 7.2B
stacked-RKV route lowers its candidate peak from `17.4-18.6 GiB` to
`14.3-15.5 GiB`. See the
[immutable evidence](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md).

### Apple M5: complete target-only W4 comparison

Apple MLX W4 is shown separately so that backend and precision remain
consistent within each table.

| Model pair | Batch / shape | RWKV active params | Qwen active params | Raw P / D | Adjusted P / D | E2E raw / adjusted | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| 0.4B / 0.8B | B8, cold, P512 chars/D64 | 0.450768B | 0.752393B | **2.04x / 2.04x** | **1.22x / 1.22x** | **2.02x / 1.21x** | [M5 B8](../bench/apple_bsz8_active_m5_20260714/README.md) |
| 1.5B / 2B | B1, P512 chars/D64 | 1.527405B | 1.881825B | **1.67x / 1.44x** | **1.36x / 1.17x** | **1.45x / 1.17x** | [M5 B1](../bench/apple_bsz1_active_m5_20260715/README.md) |
| 1.5B / 2B | B8, cold, P512 chars/D64 | 1.527405B | 1.881825B | **1.41x / 1.40x** | **1.14x / 1.14x** | **1.37x / 1.11x** | [M5 B8](../bench/apple_bsz8_active_m5_20260714/README.md) |

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
  raw tok/s, exact active parameter counts, parameter-size-adjusted speed, and
  Prefill + Decode end-to-end speed.
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
from statistics import median

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
raw_e2e = []
adjusted_e2e = []
for cell in summary["cells"]:
    batch = cell["batch_size"]
    candidate_s = (
        batch * cell["prompt_tokens"] / cell["candidate_prefill_tokps_total"]
        + batch * cell["decode_tokens"] / cell["candidate_decode_tokps_total"]
    )
    reference_s = (
        batch * cell["prompt_tokens"] / cell["reference_prefill_tokps_total"]
        + batch * cell["decode_tokens"] / cell["reference_decode_tokps_total"]
    )
    raw_ratio = reference_s / candidate_s
    raw_e2e.append(raw_ratio)
    adjusted_e2e.append(raw_ratio * cell["active_parameter_ratio"])
print("raw/parameter-adjusted E2E median:", median(raw_e2e), median(adjusted_e2e))
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
