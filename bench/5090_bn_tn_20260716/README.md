# RTX 5090 explicit BN/TN quant-kernel sweep

Status: **correctness pass; negative performance result; no production
dispatch promoted**.

Start with the terminology and ratio definitions in
[`../../docs/performance/BN_TN_TUNING.md`](../../docs/performance/BN_TN_TUNING.md).

## One-minute interpretation

- `BN` is output columns per CUDA block.
- `TN` is output columns per CUDA thread.
- block size is exactly `BN / TN` threads.
- `speedup_vs_current > 1` means faster than the old native quant kernel.
- `speedup_vs_fp16 > 1` means faster than same-shape dense FP16.
- `TN` is not Triton `num_warps` and BN/TN is unrelated to bitsandbytes.

The sweep validates the terminology and correctness of nine explicit launch
configurations. It does not validate this scalar kernel as a production fast
path: only 4/32 case winners beat the old quant kernel and 0/32 beat FP16.

## Environment and matrix

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32GB, `sm_120` |
| Driver | `595.58.03` |
| Torch / CUDA / Triton | `2.11.0+cu128` / `12.8` / `3.6.0` |
| Activation dtype | FP16 |
| Quant formats | native affine MM8 and packed MM4 |
| Batch | B1 and B8 |
| Candidate tiles | nine legal pairs from `BN={64,128,256}`, `TN={1,2,4,8}` |
| Timing | 10 warmups, 50 CUDA-event runs per row |

Shape notation is `KxN` for `X[B,K] @ W[K,N] -> Y[B,N]`:

- 1.5B: `2048x2048`, `2048x8192`, `8192x2048`, `2048x65536`;
- 7.2B: `4096x4096`, `4096x16384`, `16384x4096`, `4096x65536`.

The complete matrix is 2 modes x 2 batches x 8 shapes x 9 configurations =
**288 rows**.

## Metric definitions

```text
speedup_vs_current = current_ms / candidate_ms
speedup_vs_fp16    = fp16_ms / candidate_ms
```

Greater than 1 is faster. Correctness compares the candidate with the current
quantized output through `cosine_vs_current` and `max_abs_vs_current`.

## Aggregate result

| Metric | Result |
|---|---:|
| Correct rows | `288/288` |
| Minimum cosine vs current quant | `0.999999642` |
| Maximum absolute error vs current quant | `0.000977` |
| Best candidate beats current quant | `4/32` cases |
| Best candidate beats dense FP16 | `0/32` cases |
| Winning TN distribution | TN1 `28/32`, TN2 `3/32`, TN4 `1/32` |
| Winning BN distribution | BN64 `27/32`, BN128 `3/32`, BN256 `2/32` |

## The four wins against current quant

All four occur in W4/B8 square or FFN-up projections. The last column is the
reason they are not production wins.

| Shape `KxN` | BN | TN | Candidate ms | vs current quant | vs dense FP16 |
|---|---:|---:|---:|---:|---:|
| `2048x2048` | 64 | 1 | `0.183247` | `2.603268x` | `0.059335x` |
| `2048x8192` | 64 | 1 | `0.187442` | `2.527187x` | `0.063006x` |
| `4096x4096` | 64 | 1 | `0.370535` | `1.293249x` | `0.052171x` |
| `4096x16384` | 64 | 1 | `0.379322` | `1.257639x` | `0.225539x` |

Representative rejected W4/B8 boundaries:

| Role | 1.5B vs current / FP16 | 7.2B vs current / FP16 |
|---|---:|---:|
| FFN-down | `0.652025x / 0.021411x` | `0.322505x / 0.059411x` |
| lm-head | `0.959715x / 0.354894x` | `0.395833x / 0.269087x` |

Every W8 case and every W4/B1 case is slower than the current quant kernel.
All 32 case winners are slower than dense FP16.

## Decision and later production route

No result from this artifact changes runtime dispatch. The scalar kernel
performs per-element dequantization and accumulation without the tensor-core
dataflow needed to compete with dense GEMM across the full model.

The later RTX 5090 W4 close uses Marlin for the 7.2B FFN pair and TorchAO for
the head. That is a separate implementation and does not retroactively promote
this BN/TN experiment:
[`../5090_marlin_w4_hybrid_20260716/`](../5090_marlin_w4_hybrid_20260716/README.md).

## Reproduce

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH=$PWD

python bench/bench_quant_bn_tn.py \
  --modes mm8 mm4 --batch-sizes 1 8 \
  --shapes 2048x2048 2048x8192 8192x2048 2048x65536 \
  --block-n 64 128 256 --thread-n 1 2 4 8 \
  --warmup 10 --runs 50 \
  --output bench/5090_bn_tn_20260716/bn_tn_1p5b.jsonl

python bench/bench_quant_bn_tn.py \
  --modes mm8 mm4 --batch-sizes 1 8 \
  --shapes 4096x4096 4096x16384 16384x4096 4096x65536 \
  --block-n 64 128 256 --thread-n 1 2 4 8 \
  --warmup 10 --runs 50 \
  --output bench/5090_bn_tn_20260716/bn_tn_7p2b.jsonl
```

## Raw evidence

- [`bn_tn_1p5b.jsonl`](bn_tn_1p5b.jsonl): 1.5B square/up/down rows;
- [`bn_tn_1p5b_lm_head.jsonl`](bn_tn_1p5b_lm_head.jsonl): 1.5B head rows;
- [`bn_tn_7p2b.jsonl`](bn_tn_7p2b.jsonl): all 7.2B rows;
- `*_summary.log`: selected winner for each of the 32 cases.

Raw JSONL and logs are immutable evidence; this README is the interpretation
layer.
