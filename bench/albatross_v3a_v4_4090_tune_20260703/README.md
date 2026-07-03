# Albatross v3a vs v4 4090 tune smoke — 2026-07-03

GPU: `NVIDIA GeForce RTX 4090, sm_89, driver 570.124.06, 24 GiB`.

Model: `rwkv7-g1d-0.4b-20260210-ctx8192.pth`.

Purpose: make the acceptance reference explicit instead of treating the earlier
`faster3a_2605` result as the only Albatross speed ceiling.  This is a
reference-tuning smoke, not a replacement for full MATH500 accuracy evaluation.

## Commands

```bash
# v3a Python reference
cd /workspace/projects/Albatross/faster3a_2605
python3 rwkv7_fast_v3a.py \
  --model /workspace/models/rwkv7/raw/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --cases "1x1,1x512,64x1,4x128,8x64" \
  --warmup 3 --iters 10

# v4 C++/CUDA reference
cd /workspace/projects/Albatross/faster4_2605_cpp
./bin/rwkv7_fast_v4 \
  --model /workspace/models/rwkv7/raw/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --model-forward \
  --cases "1x1,1x512,64x1,4x128,8x64" \
  --graph-bench --warmup 3 --iters 10
```

## Result

| Case | v3a tok/s | v4 tok/s | v4/v3a | Note |
|---|---:|---:|---:|---|
| B1T1 | 837.53 | 855.73 | 1.022x | v4 faster |
| B1T512 | 48311.51 | 58933.80 | 1.220x | v4 faster |
| B64T1 | 25130.68 | 25183.30 | 1.002x | v4 faster |
| B4T128 | 81847.50 | 89226.80 | 1.090x | v4 faster |
| B8T64 | 94940.28 | 96756.70 | 1.019x | v4 faster |

## Interpretation

- On this RTX 4090 smoke, v4 is faster for all tested cases; the largest win is
  the prompt-prefill acceptance-relevant `B1T512` case (`1.220x` vs v3a).
- The earlier PR #104 MATH500 reference remains the committed full accuracy
  reference because v4 has only been smoke-benchmarked here, not run through the
  full MATH500 avg@64 evaluation path.
- Final speed claims should compare against two references when available:
  1. committed full-eval v3a reference from PR #104;
  2. fastest tuned per-GPU Albatross backend/config for the exact shape.

## `linear_orig_layout_launch` tuning note

The v4 speed path keeps selected weights in original layout and routes them
through `linear_orig_layout_launch(...)`.  The launch policy is hard-coded by
`rows`, `K`, and group (`AttC2C`, `FfnKey`, `Head`).  For the 0.4B model
(`C=1024`) the acceptance-relevant `B1T512` prefill path uses `rows=512` for the
body and `head_rows=1` for the final logits.  The current v4 binary therefore
uses the built-in choices from `faster4_2605_cpp/src/rwkv7_fast_v4.cu` rather
than a per-GPU search.

Next tuning step: add an Albatross-side micro sweep that varies the cublasLt
algorithm/workspace and exact-row kernels for each `(GPU, C, rows, group)` bucket,
then store the winning policy beside this artifact.  Do not use a single global
layout policy across V100, Ada/4090, Hopper, Blackwell, and AMD backends.

## Artifacts

- `v3a_bench.log`
- `v4_bench.log`
- `summary.json`
- `summary.md`
- `gpu.txt`
- `python.txt`
