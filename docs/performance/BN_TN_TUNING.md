# BN/TN CUDA kernel tuning

This page is the public reference for explicit BN/TN tuning. Read it before
interpreting card-local JSONL files or proposing a production policy.

## Current conclusion

**BN and TN are useful, independent launch parameters, but promotion is
implementation- and card-specific.** The RTX 5090 scalar CUDA W8/W4 probe is
negative performance evidence: 288/288 rows are correct, but no case beats
same-shape dense FP16. The separate V100 packed-W4 A16/DP4A implementation now
has an exact-shape BN/TN table and passes three end-to-end decode profiles.

The later RTX 5090 7.2B W4 production route uses BN/TN as physical Tensor Core
CTA/epilogue contracts inside Marlin; it does not promote or reuse the scalar
kernel. See the production section below and its canonical artifact.

## Terminology

For an input `X[B,K]`, weight `W[K,N]`, and output `Y[B,N]`:

| Term | Meaning in this probe | CUDA launch consequence |
|---|---|---|
| `BN` / block-N | Output columns assigned to one CUDA thread block | grid-x is `ceil(N / BN)` |
| `TN` / thread-N | Output columns accumulated by one CUDA thread | block size is `BN / TN` threads |
| `threads` | Physical threads in the block | exactly `BN / TN` |

For output tile `tile = blockIdx.x * BN`, thread `lane` computes columns
`tile + lane + j * threads` for `j = 0..TN-1`. Therefore BN controls output
tiling while TN trades thread-level work against block occupancy. They must not
be collapsed into one label.

BN/TN is unrelated to **bitsandbytes**. `TN` is also not Triton
`num_warps`: Triton's physical lane/MMA mapping is selected by the compiler,
whereas this handwritten CUDA probe assigns an explicit per-thread output
tile.

## Legal configurations

The default Cartesian product is `BN={64,128,256}` and `TN={1,2,4,8}`. A
candidate is retained only when `BN / TN` produces 32–1024 threads in complete
32-thread warps.

| BN | TN | Threads | Warps | Legal |
|---:|---:|---:|---:|---|
| 64 | 1 | 64 | 2 | yes |
| 64 | 2 | 32 | 1 | yes |
| 64 | 4 / 8 | 16 / 8 | — | no: fewer than 32 threads |
| 128 | 1 | 128 | 4 | yes |
| 128 | 2 | 64 | 2 | yes |
| 128 | 4 | 32 | 1 | yes |
| 128 | 8 | 16 | — | no: fewer than 32 threads |
| 256 | 1 | 256 | 8 | yes |
| 256 | 2 | 128 | 4 | yes |
| 256 | 4 | 64 | 2 | yes |
| 256 | 8 | 32 | 1 | yes |

This yields nine candidates per mode/batch/shape case.

The V100 implementation uses a separate legal set because each warp or
half-warp owns an output tile:
`(1,1),(2,1),(4,1),(4,2),(4,4),(8,1),(8,2),(8,4),(16,1),(16,2),
(16,4),(32,1),(32,2)`. These values must not be projected onto the RTX 5090
scalar probe or another GPU family.

## What the benchmark measures

[`../../bench/bench_quant_bn_tn.py`](../../bench/bench_quant_bn_tn.py) JIT
compiles handwritten CUDA kernels and measures each legal pair against two
same-run baselines:

1. `current_ms`: the existing Triton `mm8_matmul_triton` or
   `mm4_matmul_triton` implementation;
2. `fp16_ms`: same-shape dense `X @ W` in FP16.

The JSONL ratios are deliberately baseline-over-candidate:

```text
speedup_vs_current = current_ms / candidate_ms
speedup_vs_fp16    = fp16_ms / candidate_ms
```

For both fields, **greater than 1 means the BN/TN candidate is faster**;
less than 1 means it is slower. For example, `speedup_vs_current=2.60` and
`speedup_vs_fp16=0.059` means “2.60x faster than the old quant kernel, but only
5.9% of dense-FP16 speed.” It is not an FP16 speed win.

Correctness is checked against the current quantized kernel, not against dense
FP16:

- `cosine_vs_current` must meet `--min-cosine` (default `0.999`);
- `max_abs_vs_current` is retained as diagnostic telemetry;
- the lowest-latency correct row is selected as the case winner.

## Required measurement matrix

A card-local sweep should include:

- true batch 1 and 8;
- square, FFN-up, FFN-down, and output-head projections;
- W8 and W4;
- all nine legal default BN/TN pairs;
- exact GPU name, compute capability, driver, Torch and CUDA versions;
- warmup count, timed-run count, raw JSONL, and per-case winners.

The probe is a synthetic projection microbenchmark. It does **not** measure
model prefill, recurrent decode, full generation, peak model memory, or task
quality. A microkernel result cannot establish an end-to-end production claim.

## RTX 5090 result (2026-07-16)

Environment: RTX 5090 32GB (`sm_120`), driver `595.58.03`, Torch
`2.11.0+cu128`, CUDA 12.8, FP16 activations, 10 warmups and 50 CUDA-event
runs. The matrix contains 32 mode/batch/shape cases and nine candidates per
case: **288/288 rows passed correctness**.

| Gate | Result |
|---|---:|
| Correct candidate rows | `288/288` |
| Minimum cosine vs current quant | `0.999999642` |
| Maximum absolute error vs current quant | `0.000977` |
| Case winners faster than current quant | `4/32` |
| Case winners faster than dense FP16 | `0/32` |
| Winner `(BN,TN)=(64,1)` | `27/32` |

The four wins against the old quant kernel are all W4, B8, square/FFN-up
projections:

| Shape `KxN` | BN/TN | vs current quant | vs dense FP16 |
|---|---:|---:|---:|
| `2048x2048` | `64/1` | `2.603268x` | `0.059335x` |
| `2048x8192` | `64/1` | `2.527187x` | `0.063006x` |
| `4096x4096` | `64/1` | `1.293249x` | `0.052171x` |
| `4096x16384` | `64/1` | `1.257639x` | `0.225539x` |

FFN-down and output-head cases regress, and every W8/B1 winner is slower than
the existing quant path. The experiment therefore rejects this scalar design
as a universal kernel even though it validates BN/TN as separate tuning axes.

Full evidence:
[`../../bench/5090_bn_tn_20260716/`](../../bench/5090_bn_tn_20260716/README.md).

## V100 packed-W4 result (2026-07-16)

Environment: Tesla V100-PCIE-32GB (`sm_70`), driver `580.159.03`, Torch
`2.5.1+cu124`, CUDA 12.4, Triton 3.3.0 and fp16. The decode kernel uses A16 at
B1 and dynamic A8 plus DP4A at B2/B4/B8. Rowwise and groupwise output tiles
have independent exact `(rows,K,N)` dispatch tables.

The tuning evidence contains 432 rowwise rows, 108 group128 head rows and 52
group256 head rows. Best correct candidates beat same-shape fp16 in `35/48`,
`11/12` and `4/4` cases respectively. Microbenchmarks are only used to choose
tiles; promotion requires the paired end-to-end matrix below.

| Model | Single config | Decode vs fp16 | Footprint | Min final cosine | Gate |
|---|---|---:|---:|---:|---:|
| 1.5B | memory + group128 head + fused epilogue | `1.0255x-1.1837x` | `0.5395x` | `0.99828702` | `7/7` |
| 2.9B | speed + group256 head | `1.0111x-1.0346x` | `0.9573x` | `0.99965668` | `7/7` |
| 7.2B | memory + group128 head | `1.0810x-1.8422x` | `0.3013x` | `0.99903870` | `7/7` |

All 21 current-main cells preserve the complete timed greedy sequence and
repeat SHA256. The weakest 1.5B B4, 2.9B B8 and 7.2B B8 cells use five
repeats. The 1.5B route explicitly enables the fused
ReLU-squared/residual epilogues; they remain default-off for every other route.
Group128 was rejected for the 2.9B profile after its independent B8 result fell
to `0.9984x`; group256 plus `(BN,TN)=(32,1)` records `1.0111x` end to end on
the rebased code.

This is primarily a cached-decode promotion. Full-memory prefill remains slow
(`0.0716x-0.3192x` fp16 across the 1.5B/7.2B profiles), so the result must not
be described as universal W4 or prefill production. The head-only 2.9B speed
profile separately passes all seven prefill cells at `1.0006x-1.0603x`.
Evidence:
[`../../bench/v100_sm70_mm4_bntn_20260716/`](../../bench/v100_sm70_mm4_bntn_20260716/README.md).

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
  --output /tmp/bn_tn_1p5b.jsonl
```

Run the 7.2B shape set separately:

```bash
python bench/bench_quant_bn_tn.py \
  --modes mm8 mm4 --batch-sizes 1 8 \
  --shapes 4096x4096 4096x16384 16384x4096 4096x65536 \
  --block-n 64 128 256 --thread-n 1 2 4 8 \
  --warmup 10 --runs 50 \
  --output /tmp/bn_tn_7p2b.jsonl
```

The first run JIT-compiles a CUDA extension and is not part of the timed
region. Use an idle GPU and preserve clock/power/process telemetry when making
new performance claims.

## Production promotion rule

A BN/TN candidate may enter card policy only when all of the following pass:

1. quantized-output correctness for every promoted shape;
2. repeated same-card improvement over the current production quant kernel;
3. FP16 equivalence or speed according to the declared gate;
4. paired end-to-end prefill and decode with no regression;
5. footprint/peak-memory and quality gates;
6. exact-card/shape/dtype dispatch with fail-closed fallback elsewhere.

The scalar probe passes item 1 only. Its correct status is **negative
performance evidence; no scalar production dispatch is promoted**.

The V100 packed-W4 implementation passes the named cached-decode profiles;
its group selection remains explicit and its full-memory prefill path remains
open.

## RTX 5090 production Tensor Core route

The scalar result did not end BN/TN work; it established that a serial-K scalar
kernel was the wrong architecture. The promoted implementation keeps Marlin's
Tensor Core accumulation and gives BN/TN explicit physical contracts:

- each internal launch segment with rows `<=16`: `BN=128`, `TN=8`, K tile
  128, 256 threads, 4 stages;
- each internal launch segment with rows `>16`: `BN=256`, `TN=8`, K tile 64,
  256 threads, 4 stages;
- `TN=8` is one 16-byte `int4` epilogue store containing eight BF16 values;
- CUDA validates expected BN/TN after forming every scheduler segment and
  fails closed on disagreement.

The per-segment distinction is required for non-aligned M. A 65-row logical
GEMM is a 64-row `BN=256` launch plus a 1-row `BN=128` tail. The production
contract sweep checks 35 row counts through 8192 on both 7.2B FFN shapes:
70/70 pass, 70/70 are bit-exact against unguarded Marlin, 70/70 intentionally
wrong BN checks fail closed, and 10 rows exercise mixed-grid tails.

Historical Marlin `thread_n` means CTA output-tile width, not per-writer TN.
Manual tile, SM-count and two-stage sweeps did not beat auto broadly, so
production retains auto schedule selection and validates the selected grid.

The exact RTX 5090 7.2B B1/B8 route also fuses FFN-key ReLU-square through an
explicit ABI. It passes paired hot-BF16 prefill and decode, uses `0.5298x`
footprint, preserves same-next-token output, and keeps other cards on their
old fallback. Canonical evidence:
[`../../bench/5090_bn_tn_tensorcore_20260716/`](../../bench/5090_bn_tn_tensorcore_20260716/README.md).
