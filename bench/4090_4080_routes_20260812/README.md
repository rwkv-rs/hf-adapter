# RTX 4090 reproduction of RTX 4080 routes (2026-08-12)

**Status: PASS.** This artifact tests three exact RTX 4080 optimization ideas
on a physical desktop RTX 4090 instead of inheriting them by the shared
`sm_89` capability. Two routes clear the promotion gate for current
0.4B/1.5B/2.9B checkpoints: B8 grouped W/A/V projection BMM and block-scoped
FP16 GEMM accumulation during native Prefill. The 7.2B/B8 FP16 recurrent-state
idea is tracked separately until its exact 4090 A/B completes.

## B8 grouped W/A/V BMM

Each checkpoint was independently loaded three times. The baseline is the
current ungrouped fallback and the candidate enables both grouped W/A/G/V and
the tensor-core W/A/V BMM. Each run uses prompt 64, 128 greedy correctness
steps, 16 warmups, and 512 fixed-token timing steps.

| Model | Baseline | Candidate | Candidate throughput | Median speedup | VRAM delta | Greedy |
|---|---:|---:|---:|---:|---:|---:|
| 0.4B | 2.4707 ms | 2.0585 ms | 3,886.3 tok/s | **1.2002x** | +27.0 MiB | 3,072/3,072 |
| 1.5B | 5.2881 ms | 4.6287 ms | 1,728.3 tok/s | **1.1426x** | +280.8 MiB | 3,072/3,072 |
| 2.9B | 9.4113 ms | 8.3588 ms | 957.1 tok/s | **1.1259x** | +709.5 MiB | 3,072/3,072 |

The dispatch remains exactly B8 and accepts only hidden sizes
1024/2048/2560 with low-rank width at most 512. The generic grouped fallback
remains rows<=4 on RTX 4090, so hidden-4096 7.2B cannot enter this route.

## Block-scoped FP16 accumulation

The same loaded checkpoint measures `off`, process-global FP16 accumulation,
and block-only FP16 accumulation in forward and reverse order. The selected
block-only boundary accelerates transformer-block GEMMs while retaining FP32
accumulation for the final norm and vocabulary head.

- Matrix: 0.4B/1.5B/2.9B, B1/B8, P128/P512/P2048.
- Screening: **108/108 rows pass** route, prompt, cache-handoff, cosine, and
  greedy gates.
- All 18 selected exact shapes are faster in both orders. Minimum per-order
  speedup is **1.005662x**; per-shape median speedups span
  **1.010997x-1.300661x**.
- Minimum selected-route prompt/decode cosine is **0.999994/0.999991**.

## Default-policy validation

The patched policy was then loaded without accumulation environment overrides
and compared against the independent direct-native recurrent oracle. All
**18/18** rows select block-only accumulation, reject global accumulation,
preserve prompt and cache-handoff greedy tokens, and pass with minimum
prompt/decode cosine **0.999998/0.999997**.

## Environment and scope

- GPU: NVIDIA GeForce RTX 4090, `sm_89`, driver 550.142.
- Candidate: PyTorch 2.7.1+cu126, CUDA 12.6, Triton 3.3.1,
  Transformers 5.12.1, FP16 inputs with FP32 recurrent state.
- The checkpoints are the latest official g1d/g1i 0.4B/1.5B/2.9B sources
  converted through this repository.
- This is an exact-card, exact-model, exact-batch/prompt claim. RTX 4090
  variants, other Ada cards, B2/B4, BF16, and unlisted shapes retain their
  prior policies.

## Reproduction

The paired BMM command is:

```bash
PYTHONPATH=. python bench/bench_native_graph_ada_wagv_lora.py \
  --hf-dir /models/rwkv7-1.5b-hf --code-source repo \
  --dtype fp16 --device cuda --attn-mode fused_recurrent \
  --axis ada_wagv_bmm_from_default --batch-size 8 --prompt-tokens 64 \
  --correctness-steps 128 --warmup 16 --steps 512 --fixed-token \
  --num-warps 8 --results /tmp/b8_wagv_bmm_from_default.jsonl
```

The paired Prefill screen is:

```bash
PYTHONPATH=. python bench/bench_native_prefill_accum_ab.py \
  --model /models/rwkv7-1.5b-hf --device cuda --dtype fp16 \
  --batch-sizes 1 8 --prompt-tokens 128 512 2048 --orders both \
  --warmup 5 --steps 15 --min-cosine 0.9999 --code-source repo \
  --results /tmp/accum_all_shapes_screen.jsonl
```

Recalculate the fail-closed summary with
`bench/summarize_4090_4080_routes.py`.

## Files

- `b8_wagv_bmm_from_default.jsonl`: nine independent BMM A/B rows.
- `accum_all_shapes_screen.jsonl`: 108 same-process forward/reverse rows.
- `policy_default_prefill_validation.jsonl`: 18 default-policy oracle rows.
- `summary.json`, `summary.md`: coverage and gate output.
- `environment.json`: exact runtime contract.
- `SHA256SUMS`: artifact integrity.
