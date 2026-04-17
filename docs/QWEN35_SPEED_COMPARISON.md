# RWKV-7 vs Qwen3.5: Complete Parameter and Speed Comparison

Updated: **2026-08-11**. All numbers come from promoted same-device evidence
on the current main branch. See [`BENCHMARK.md`](../BENCHMARK.md) for the full
history, quantization lanes, and cell-level telemetry. [中文版](QWEN35_SPEED_COMPARISON_ZH.md)

## Results at a glance

> The promoted NVIDIA dense-FP16 comparison contains **24 GPU/model/batch
> combinations**, plus **3 Apple M5 target-only W4 combinations**. Every row
> has RWKV-7 ahead of Qwen3.5 in median raw Prefill and Decode throughput.
> Raw Prefill/Decode reaches **5.47x / 12.15x**. After discounting the natural
> speed advantage of the smaller model, parameter-size-adjusted
> Prefill/Decode reaches **4.39x / 7.28x**. In the more direct Prefill + Decode
> end-to-end result, all **134/134 measured NVIDIA cells** and all **24/24
> combination medians** beat Qwen3.5 both raw and after parameter-size
> adjustment; all **3/3 Apple M5 combinations** also remain ahead after
> adjustment.

**RTX 4080 is now complete: parameter-size-adjusted Prefill and Decode medians
exceed `1.00x` for all three model pairs at both B1 and B8, and every adjusted
E2E median also remains ahead.**

- `1.02x` means RWKV throughput is 1.02 times Qwen throughput, or about 2%
  faster.
- Prefill processes the input prompt. Decode generates tokens one at a time
  and is the closer match for sustained chat generation.
- The NVIDIA table uses **raw dense-FP16 tok/s** and also reports
  parameter-size-adjusted speed. Except for the explicitly labeled V100
  shapes, `6 cells` means the median across
  `P128/512/2048 × D128/512`.

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

| GPU | Model pair | Batch | Scope | RWKV active params | Qwen active params | Param ratio | Raw P / D | Adjusted P / D | E2E raw / adjusted | Evidence |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| V100 32GB | 1.5B / 2B | B1 | P512/D64 | 1.527405B | 1.881825B | `0.811661` | **2.82x / 5.91x** | **2.29x / 4.80x** | **5.59x / 4.54x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| V100 32GB | 1.5B / 2B | B8 | P512/D64 | 1.527405B | 1.881825B | `0.811661` | **5.41x / 5.27x** | **4.39x / 4.28x** | **5.30x / 4.30x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| RTX 3090 | 1.5B / 2B | B8 | 6 cells | 1.527405B | 1.881825B | `0.811661` | **1.08x / 3.42x** | **0.88x / 2.77x** | **3.18x / 2.58x** | [3090 small](../bench/3090_small_bsz8_20260714/README.md) |
| RTX 3090 | 2.9B / 4B | B8 | 6 cells | 2.947735B | 4.205751B | `0.700882` | **1.36x / 2.96x** | **0.96x / 2.07x** | **2.77x / 1.94x** | [3090 small](../bench/3090_small_bsz8_20260714/README.md) |
| RTX 3090 | 7.2B / 9B | B8 | 6 cells | 7.199142B | 8.953803B | `0.804032` | **1.06x / 1.81x** | **0.86x / 1.45x** | **1.70x / 1.36x** | [3090 7.2B](../bench/3090_g1h_7p2_bsz8_20260714/README.md) |
| RTX 4080 | 0.4B / 0.8B | B1 | 6 cells | 0.450768B | 0.752393B | `0.599112` | **1.82x / 4.89x** | **1.09x / 2.93x** | **4.81x / 2.88x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 0.4B / 0.8B | B8 | 6 cells | 0.450768B | 0.752393B | `0.599112` | **1.77x / 4.16x** | **1.06x / 2.49x** | **3.93x / 2.36x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 1.5B / 2B | B1 | 6 cells | 1.527405B | 1.881825B | `0.811661` | **1.54x / 1.90x** | **1.25x / 1.55x** | **1.90x / 1.54x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 1.5B / 2B | B8 | 6 cells | 1.527405B | 1.881825B | `0.811661` | **1.76x / 1.77x** | **1.43x / 1.44x** | **1.77x / 1.44x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 2.9B / 4B | B1 | 6 cells | 2.947735B | 4.205751B | `0.700882` | **1.75x / 1.63x** | **1.22x / 1.15x** | **1.63x / 1.14x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 2.9B / 4B | B8 | 6 cells | 2.947735B | 4.205751B | `0.700882` | **1.99x / 1.75x** | **1.40x / 1.23x** | **1.77x / 1.24x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4090 | 0.4B / 0.8B | B8 | 6 cells | 0.450768B | 0.752393B | `0.599112` | **1.75x / 12.15x** | **1.05x / 7.28x** | **11.46x / 6.86x** | [4090 small](../bench/4090_small_bsz8_20260715/README.md) |
| RTX 4090 | 1.5B / 2B | B8 | 6 cells | 1.527405B | 1.881825B | `0.811661` | **1.11x / 5.66x** | **0.90x / 4.59x** | **5.30x / 4.30x** | [4090 small](../bench/4090_small_bsz8_20260715/README.md) |
| RTX 4090 | 2.9B / 4B | B8 | 6 cells | 2.947735B | 4.205751B | `0.700882` | **1.42x / 4.24x** | **1.00x / 2.97x** | **3.99x / 2.80x** | [4090 small](../bench/4090_small_bsz8_20260715/README.md) |
| RTX 4090 | 7.2B / 9B | B8 | 6 cells | 7.199142B | 8.953803B | `0.804032` | **1.12x / 2.22x** | **0.90x / 1.79x** | **2.11x / 1.69x** | [4090 7.2B](../bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 5070 Laptop | 1.5B / 2B | B8 | 6 cells | 1.527405B | 1.881825B | `0.811661` | **1.33x / 2.62x** | **1.08x / 2.13x** | **2.48x / 2.02x** | [5070](../bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| RTX 5090 | 0.4B / 0.8B | B1 | 6 cells | 0.450768B | 0.752393B | `0.599112` | **5.47x / 10.90x** | **3.28x / 6.53x** | **10.80x / 6.47x** | [5090](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md) |
| RTX 5090 | 1.5B / 2B | B1 | 6 cells | 1.527405B | 1.881825B | `0.811661` | **3.26x / 6.74x** | **2.64x / 5.47x** | **6.67x / 5.41x** | [5090](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md) |
| RTX 5090 | 2.9B / 4B | B1 | 6 cells | 2.947735B | 4.205751B | `0.700882` | **2.72x / 5.24x** | **1.90x / 3.67x** | **5.12x / 3.59x** | [5090](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md) |
| RTX 5090 | 7.2B / 9B | B1 | 6 cells | 7.199142B | 8.953803B | `0.804032` | **1.21x / 2.91x** | **0.97x / 2.34x** | **2.88x / 2.31x** | [5090](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md) |
| RTX 5090 | 0.4B / 0.8B | B8 | 6 cells | 0.450768B | 0.752393B | `0.599112` | **1.61x / 7.20x** | **0.97x / 4.31x** | **6.99x / 4.19x** | [5090](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md) |
| RTX 5090 | 1.5B / 2B | B8 | 6 cells | 1.527405B | 1.881825B | `0.811661` | **1.19x / 4.59x** | **0.97x / 3.73x** | **4.40x / 3.57x** | [5090](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md) |
| RTX 5090 | 2.9B / 4B | B8 | 6 cells | 2.947735B | 4.205751B | `0.700882` | **1.48x / 3.81x** | **1.04x / 2.67x** | **3.65x / 2.56x** | [5090](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md) |
| RTX 5090 | 7.2B / 9B | B8 | 6 cells | 7.199142B | 8.953803B | `0.804032` | **1.04x / 2.83x** | **0.84x / 2.28x** | **2.66x / 2.14x** | [5090](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md) |

This complete table preserves every promoted GPU/model/batch result and makes
the raw and parameter-size-adjusted ratios directly comparable.

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
| RTX 3090 | [`bench/run_3090_qwen35_pair_acceptance.sh`](../bench/run_3090_qwen35_pair_acceptance.sh) | Same arguments as the 4090 runner |
| RTX 4080 | [`bench/run_4080_adjusted_pd.sh`](../bench/run_4080_adjusted_pd.sh) | Runs all three pairs at B1/B8 and requires adjusted P/D `>1.00x` for all six groups |
| RTX 5070 Laptop | [`bench/run_5070_qwen35_full_fla_bsz8.ps1`](../bench/run_5070_qwen35_full_fla_bsz8.ps1) | PowerShell with `-RwkvModel`, `-QwenModel`, and `-OutDir` |
| RTX 5090 | [`bench/run_5090_qwen35_full_matrix.sh`](../bench/run_5090_qwen35_full_matrix.sh) | Four model pairs, B1/B8 full matrix |

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
