# RTX 3090 self-fused RWKV-7 long-prefill close (2026-07-13)

This artifact records the first promoted end-to-end result for the vendored
RWKV-7 sequence-mode DPLR prefill kernel on RTX 3090.  It compares the official
RWKV-7 7.2B HF checkpoint with Qwen3.5-9B through the repository's common HF
speed harness.  The measured production route is `native_graph`; FLA is not
the selected prefill or decode backend.

## Environment and contract

- GPU: NVIDIA GeForce RTX 3090, 24,576 MiB, `sm_86`.
- PyTorch 2.6.0+cu124, CUDA runtime 12.4, Transformers 5.12.1,
  bitsandbytes 0.49.2, Triton 3.2.0.
- Candidate: RWKV-7 G1G 7.2B HF; reference: Qwen3.5-9B HF fast path.
- fp16, prompt 2048, decode 128, batch 1/2, three warmups and three measured
  runs; medians are reported.
- Dense acceptance gates: prefill and decode must each be at least `1.05x` the
  same-shape Qwen row; logits must be finite.

## Official HF harness result

| Bsz | RWKV prefill tok/s | Qwen prefill tok/s | Ratio | RWKV decode tok/s | Qwen decode tok/s | Ratio | RWKV/Qwen footprint MiB | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4,536.404 | 4,113.174 | **1.1029x** | 49.922 | 23.660 | **2.1100x** | 13,731.3 / 17,078.0 | PASS |
| 2 | 4,579.237 | 4,349.864 | **1.0527x** | 89.369 | 45.329 | **1.9716x** | 13,731.3 / 17,078.0 | PASS |

Peak VRAM is 14,592.3/15,260.6 MiB for RWKV at batch 1/2, versus
17,610.2/18,002.3 MiB for Qwen.  All four raw rows report `status=pass` and
finite logits.  Raw harness output is in [`results.jsonl`](results.jsonl).

## Kernel correctness and delta

At batch 2, prompt 2048, a seven-repeat direct A/B probe records:

| Route | Median ms | Prefill tok/s | Min cosine | Max abs | Same greedy | Finite |
|---|---:|---:|---:|---:|---:|---:|
| self16 | 887.792 | 4,613.695 | 0.99999547 | reference | yes | yes |
| recurrent | 899.849 | 4,551.874 | 0.99999547 | 0.0625 | yes | yes |

The self-contained sequence route is `1.0136x` the optimized recurrent route
inside the same loaded process.  Raw probe output is in
[`correctness.jsonl`](correctness.jsonl).  CPU-capable policy/fallback tests and
CUDA-gated sequence tests pass as `11 passed, 1 skipped` on the development
host; the GPU probe supplies the live CUDA correctness evidence.

## Implemented path

The promoted path:

1. computes RWKV-7 DPLR `a=-kk` and `b=kk*a_gate` in the intra-chunk Triton
   kernel instead of materializing two full-sequence tensors;
2. accepts and emits the native `[B,H,V,K]` recurrent-state layout directly,
   removing per-layer state transposes;
3. fuses deferred A/V-gate sigmoid work into state preparation;
4. uses GEMM `beta=1`-style in-place residual projection epilogues;
5. keeps the numerically sensitive intra-chunk dot products in float32/TF32;
6. uses the measured shape policy: cuBLASLt below 4,096 rows and cuBLAS from
   4,096 rows, so batch 1 and batch 2 both retain their faster backend.

Chunk sizes 16/32/64 are supported by the vendored implementation.  The
production RTX 3090 policy remains chunk 16 because it is the measured winner.
Other GPUs retain capability-conservative defaults.

## Reproduction

```bash
python bench/run_qwen35_speed_matrix.py \
  --pair 'rwkv-7.2b__qwen3.5-9b=/models/rwkv7-g1g-7.2b-hf::/models/Qwen3.5-9B' \
  --prompt-tokens 2048 --decode-tokens 128 --batch-sizes 1 2 \
  --quantizations none --dtype fp16 --benchmark-matrix qwen35_3090_hf \
  --qwen-backend auto --require-qwen-fast-path \
  --model-roles candidate reference --warmup 3 --runs 3 \
  --results results.jsonl
```

## Claim boundary

This closes the measured 7.2B/9B dense long-prefill batch-1 and batch-2 cells.
It is not a claim that the full 216-cell 3090 dense/W8/W4 matrix is complete.
The remaining matrix and strict quantized `>=1.00x` gates stay open.
