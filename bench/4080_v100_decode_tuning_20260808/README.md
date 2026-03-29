# RTX 4080 and V100 B8 decode tuning (2026-08-08)

This artifact records two exact-card native-graph decode changes and the
post-change B1/B2/B4/B8 regression sweep. It deliberately does not promote one
launch configuration across GPU families:

- **RTX 4080:** extend the grouped W/A/G/V graph route through B8. B8 retains
  cuBLAS for the low-rank projections and folds the surrounding activations,
  biases, sigmoid and V interpolation into the captured graph. The custom
  sm89 CUDA GEMV extension remains limited to its measured B1-B4 range.
- **Tesla V100:** retain the existing WAVG launch at B1/B2/B4 and select
  `(block_m, block_r, block_k, warps) = (32, 32, 256, 4)` only at B8. Other
  sm70 products do not inherit this exact-V100 launch.

All model loads use `--code-source repo`; the rows therefore measure the
current checkout rather than checkpoint-bundled remote code.

## Environment

| Device | Software | Notes |
|---|---|---|
| NVIDIA GeForce RTX 4080 16GB, sm89 | PyTorch 2.6.0+cu124, Triton 3.2.0, driver 595.71.05 | An unrelated sidecar retained about 8.8GB but was idle during the accepted rows. Peak-memory fields are process-local. |
| Tesla V100-PCIE-32GB, sm70 | PyTorch 2.5.1+cu124, Triton 3.3, driver 580.173.02 | GPU 0 of the two-card host; single-device inference. |

Models:

- `rwkv7-g1d-0.4b-hf`
- `rwkv7-g1g-1.5b-hf`
- `rwkv7-g1g-2.9b-hf` for the V100 B8 launch A/B

## Paired optimization results

### RTX 4080 grouped W/A/G/V graph route

Three independent loads per checkpoint, B8, prompt 64, 32 warmup steps, 512
timing steps, fixed-token timing, and 32 untimed greedy correctness steps:

| Model | Baseline median | Candidate median | Median speedup | Greedy | Min cosine | Graph-pool delta |
|---|---:|---:|---:|---:|---:|---:|
| 0.4B | 2.9089 ms | 2.7875 ms | **1.0434x** | 768/768 | 1.000000 | +9.1MB |
| 1.5B | 6.5509 ms | 6.4253 ms | **1.0195x** | 768/768 | 1.0000001 | +9.1MB |

The speedup ranges are `1.0428x-1.0438x` for 0.4B and
`1.0193x-1.0196x` for 1.5B. Raw rows:

- `4080_0.4b_b8_ada_wagv_ab.jsonl`
- `4080_1.5b_b8_ada_wagv_ab.jsonl`

The 4080 2.9B/B8 promotion was not attempted because the unrelated resident
process left insufficient free device memory for the additional graph pool.
No 2.9B/4080 claim is made by this artifact.

### V100 B8 WAVG launch

The baseline is `(32, 64, 256, 8)` and the candidate is
`(32, 32, 256, 4)`. Correctness uses 64 greedy steps outside timing.

| Model | Timing steps | Observed speedup | Greedy | Min cosine |
|---|---:|---:|---:|---:|
| 0.4B | 512 | **1.0114x-1.0141x** | 64/64 | 0.99999994 |
| 1.5B | 512 | **1.0290x-1.0300x** | 64/64 | 1.00000000 |
| 2.9B | 256 | **1.0308x-1.0312x** | 64/64 | 1.00000000 |

Raw paired rows are in `v100_b8_wavg_launch_ab.jsonl`. B1 and B2 are neutral;
B4 measured only `1.0006x-1.0011x`, so the policy intentionally preserves the
old launch outside B8.

## Current-policy B1/B2/B4/B8 sweep

Prompt 128, decode 128, four warmup calls, three prefill runs. Decode uses
`rwkv7_forward_token` with the effective `native_graph` backend.

### 0.4B

| Device | B | Prefill tok/s | Decode tok/s | Decode ms/step | Peak MB |
|---|---:|---:|---:|---:|---:|
| RTX 4080 | 1 | 23,422.7 | 485.6 | 2.06 | 913.4 |
| RTX 4080 | 2 | 39,524.7 | 740.0 | 2.70 | 961.9 |
| RTX 4080 | 4 | 54,635.5 | 1,452.8 | 2.75 | 1,044.0 |
| RTX 4080 | 8 | 69,168.4 | 2,811.3 | 2.85 | 1,191.9 |
| V100 | 1 | 10,878.8 | 438.0 | 2.28 | 1,104.2 |
| V100 | 2 | 21,634.4 | 747.0 | 2.68 | 1,358.2 |
| V100 | 4 | 31,981.0 | 1,220.7 | 3.28 | 1,637.3 |
| V100 | 8 | 45,494.3 | 1,967.8 | 4.07 | 1,793.2 |

### 1.5B

| Device | B | Prefill tok/s | Decode tok/s | Decode ms/step | Peak MB |
|---|---:|---:|---:|---:|---:|
| RTX 4080 | 1 | 13,852.9 | 195.1 | 5.13 | 2,987.7 |
| RTX 4080 | 2 | 19,062.1 | 325.8 | 6.14 | 3,067.9 |
| RTX 4080 | 4 | 22,817.1 | 649.2 | 6.16 | 3,214.5 |
| RTX 4080 | 8 | 24,044.8 | 1,241.2 | 6.45 | 3,491.4 |
| V100 | 1 | 6,785.8 | 229.4 | 4.36 | 3,752.2 |
| V100 | 2 | 11,643.9 | 376.4 | 5.31 | 4,618.9 |
| V100 | 4 | 15,619.5 | 607.4 | 6.59 | 5,536.2 |
| V100 | 8 | 18,708.3 | 927.8 | 8.62 | 5,830.5 |

Raw sweep rows are the four `*_batch_sweep_current.jsonl` files.

## Reproduction

Current-code batch sweep:

```bash
PYTHONPATH=. python bench/bench_batch_sweep.py \
  --hf-dir /path/to/model \
  --code-source repo \
  --dtype fp16 --device cuda \
  --fast-token-backend native_graph \
  --batch-sizes 1 2 4 8 \
  --prompt-tokens 128 --decode-tokens 128 \
  --warmup 4 --runs 3
```

RTX 4080 B8 grouped-route A/B:

```bash
PYTHONPATH=. python bench/bench_native_graph_ada_wagv_lora.py \
  --hf-dir /path/to/model \
  --code-source repo --dtype fp16 --device cuda \
  --batch-size 8 --prompt-tokens 64 \
  --correctness-steps 32 --warmup 32 --steps 512 --fixed-token
```

V100 launch A/B:

```bash
PYTHONPATH=. python bench/bench_native_graph_fused_wavg_lora.py \
  --hf-dir /path/to/model \
  --code-source repo --dtype fp16 --device cuda \
  --batch-size 8 --prompt-tokens 64 \
  --correctness-steps 64 --warmup 32 --steps 512 --fixed-token \
  --compare-launch \
  --baseline-block-m 32 --baseline-block-r 64 --baseline-block-k 256 \
  --baseline-num-warps 8 \
  --block-m 32 --block-r 32 --block-k 256 --num-warps 4
```

## Regression result

- RTX 4080 focused policy/kernel/hot-path suite: **60 passed**.
- V100 focused policy/kernel/hot-path suite: **58 passed, 2 skipped**; the two
  skips are the expected sm89/sm120-only Ada extension tests.
- Additional benchmark-contract suite after harness changes: **31 passed** on
  RTX 4080.
- All batch-sweep rows selected `native_graph`; all paired optimization rows
  passed their cosine and greedy gates.
