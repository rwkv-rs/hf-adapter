# RTX 5090 production BN/TN Tensor Core W4 — 2026-07-16

Status: **production gate passed for the exact 7.2B BF16/W4 B1/B8 lane**.

This is the production successor to the scalar
[`../5090_bn_tn_20260716/`](../5090_bn_tn_20260716/README.md) experiment. The
scalar probe remains useful negative evidence, but it is not the dispatched
kernel. Production BN/TN runs inside the vendored Marlin Tensor Core kernel.

## Physical grid

| Internal launch rows | BN: CTA output columns | TN: BF16 columns per epilogue writer | K tile | CUDA threads | Stages |
|---:|---:|---:|---:|---:|---:|
| `1..16` | 128 | 8 | 128 | 256 | 4 |
| `>16` | 256 | 8 | 64 | 256 | 4 |

`TN=8` is physical: one vectorized CUDA `int4` store is 16 bytes, or eight
BF16 outputs. Marlin's historical variable named `thread_n` is a CTA tile
width and is **not** this TN. The Python route enables a per-launch production
sentinel in CUDA; the launcher aborts if auto-dispatch selects a different
BN/TN after each internal M segment has been formed.

A logical GEMM can use both grids. For example, 65 rows are split into a
64-row `BN=256` launch and a 1-row `BN=128` tail. The extended contract sweep
covers 35 row counts from 1 through 8192 for both FFN shapes: **70/70 pass**, all
70 are bit-exact against the same unguarded Marlin calculation, all 70 reject
an intentionally wrong BN, and 10 exercise mixed-grid tails. This closes the
non-aligned prompt/dynamic-batch boundary that a whole-GEMM BN assertion would
incorrectly reject.

## End-to-end acceptance

Official `rwkv7-g1h-7.2b-hf`, prompt 128, decode 128, five timing repeats. The
table uses the fresh post-boundary-audit rerun:

| Batch | Footprint / BF16 | Prefill / hot BF16 | Decode / BF16 | Quant prefill | Quant decode | Final cosine | Same token |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `0.5298x` | `1.0010x` | `1.5068x` | 1789.5 tok/s | 133.5 tok/s | `0.99963713` | yes |
| 8 | `0.5298x` | `1.1561x` | `1.4978x` | 10812.3 tok/s | 886.1 tok/s | `0.99954909` | yes |

Because B1 prefill is the tight gate, an independent nine-repeat confirmation
was also run after the audit. It passes at `1.0024x` prefill and `1.5062x`
decode with the same `0.99963713` final cosine and next token. The summary above
retains the lower `1.0010x` post-audit prefill result rather than cherry-picking
the confirmation.

The route packs 32 `ffn.key` and 32 `ffn.value` matrices as symmetric
group-128 W4, keeps the asymmetric TorchAO W4 head, and leaves 4096-square
attention projections dense. The benchmark records
`native_mm_kernel=bntn_marlin_bf16_w4` and 32 fused FFN-key modules.

## Fused FFN-key epilogue

The key projection fuses ReLU-square into the final BF16 epilogue. Against
Marlin plus the existing standalone Triton ReLU-square, output is bit-exact at
rows 1, 8, 128 and 1024. Measured speedups are:

| Rows | Fused / unfused speedup | Max abs difference |
|---:|---:|---:|
| 1 | `2.1471x` | 0 |
| 8 | `2.1255x` | 0 |
| 128 | `1.0247x` | 0 |
| 1024 | `1.0262x` | 0 |

The ABI is explicit: generic `MarlinW4Linear.forward()` remains a plain Linear.
Only `rwkv7_forward_relu2()` enables the epilogue. Recognized FLA RWKV7 FFNs,
native prefill and native graph call that method directly. This prevents
generic HF/FLA callers from applying ReLU-square twice.

## Rejected schedules

- Manual K/BN/thread schedules: only 1 of 32 supported non-auto rows was a
  micro win (`1.015x`, M128 down projection); the rest did not justify a
  production override.
- Explicit persistent-CTA/SM counts: 0/144 non-auto rows beat auto; observed
  range was `0.9472x–0.9907x`.
- Two-stage pipeline: 0/32 rows beat four stages; observed range was
  `0.9233x–0.9510x`.

Production therefore keeps Marlin's auto tile selection, asserts its physical
BN/TN result, uses all physical SMs, and retains four stages.

## Scope and fallback

Promotion is fail-closed to RTX 5090, SM120, BF16, group 128, exact FFN roles,
and `(16384,4096)/(4096,16384)` weight shapes. Other GPUs and shapes retain the
previous TorchAO/dense/native dispatch. This artifact is not evidence for
5070, Hopper, Ampere, Volta, ROCm, W8, or the still-dense square projections.

## Reproduce

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export PYTHONPATH=$PWD

python bench/bench_marlin_relu2.py \
  --shapes 4096x16384 --rows 1 8 128 1024 \
  --warmup 10 --runs 50 --repeats 7 \
  --output /tmp/bntn_relu2.jsonl

python bench/bench_marlin_bn_tn_contract.py \
  --rows 1 2 3 4 5 7 8 9 15 16 17 24 31 32 33 63 64 65 96 \
         127 128 129 255 256 257 511 512 513 1023 1024 1025 \
         1536 2048 4096 8192 \
  --output /tmp/bntn_grid_contract.jsonl

python bench/bench_native_quant_e2e_decode.py \
  --hf-dir /path/to/rwkv7-g1h-7.2b-hf \
  --model-size-label 7.2b --dtype bf16 --device cuda \
  --attn-mode fused_recurrent --fast-cache true \
  --fast-token-backend native_graph \
  --single-quantization torchao_w4 --min-params 1 --policy speed \
  --batch-size 8 --prompt-tokens 128 --decode-tokens 128 \
  --warmup 1 --timing-repeats 5 --paired-baseline \
  --results /tmp/bntn_e2e_b8.jsonl
```

Raw files in this directory retain the CUDA compile log, exact environment,
70-row grid contract, microkernel rows, B1/B8 end-to-end rows, and rejected
schedule sweeps.
