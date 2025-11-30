# RTX 5090 explicit BN/TN quant-kernel sweep

Status: **correctness pass; no production dispatch promoted**.

This experiment measures `BN` (output columns per CUDA block) and `TN`
(output columns accumulated per CUDA thread) as distinct physical launch
parameters. It does not relabel Triton `num_warps` as TN and is unrelated to
bitsandbytes.

## Environment and matrix

- GPU: NVIDIA GeForce RTX 5090, `sm_120`, 32GB;
- driver: `595.58.03`;
- PyTorch/CUDA/Triton: `2.11.0+cu128` / `12.8` / `3.6.0`;
- dtype: fp16 activations with native affine MM8 or packed MM4 weights;
- batches: B1 and B8;
- 1.5B shapes: `2048x2048`, `2048x8192`, `8192x2048`, and
  `2048x65536` lm-head;
- 7.2B shapes: `4096x4096`, `4096x16384`, `16384x4096`, and
  `4096x65536` lm-head;
- candidates: the nine legal whole-warp combinations from
  `BN={64,128,256}` and `TN={1,2,4,8}`;
- timing: 10 warmups and 50 CUDA-event runs per row.

The artifact contains 32 cases x 9 configurations = **288/288 passing rows**.
Minimum cosine against the current native quantized output is `0.999999642`;
the largest recorded maximum-absolute difference is `0.000977`.

## Result

| Model shape family | W8 conclusion | W4 B1 conclusion | W4 B8 internal up/square | W4 B8 down / lm-head | Promotion |
|---|---|---|---|---|---|
| 1.5B | every winner slower than current | every winner slower than current | `2.527x / 2.603x` current quant | `0.652x / 0.960x` current quant | **No** |
| 7.2B | every winner slower than current | every winner slower than current | `1.258x / 1.293x` current quant | `0.323x / 0.396x` current quant | **No** |

`TN=1` wins 28/32 cases, `TN=2` wins 3/32, and `TN=4` wins one 1.5B
lm-head case. This confirms that BN and TN cannot be collapsed into one tuning
label, but it also rejects the tested scalar CUDA design as a universal speed
path: **all 32 best candidates are slower than paired dense fp16**.

The W4 B8 internal-projection wins are insufficient for integration because
FFN-down and lm-head regress, and the original requirement is end-to-end
fp16-or-faster. The current Triton/native dispatch therefore remains unchanged.

## Reproduce

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH=$PWD

python bench/bench_quant_bn_tn.py \
  --modes mm8 mm4 --batch-sizes 1 8 \
  --shapes 2048x2048 2048x8192 8192x2048 2048x65536 \
  --warmup 10 --runs 50 \
  --output bench/5090_bn_tn_20260716/bn_tn_1p5b.jsonl

python bench/bench_quant_bn_tn.py \
  --modes mm8 mm4 --batch-sizes 1 8 \
  --shapes 4096x4096 4096x16384 16384x4096 4096x65536 \
  --warmup 10 --runs 50 \
  --output bench/5090_bn_tn_20260716/bn_tn_7p2b.jsonl
```

Raw evidence:

- [`bn_tn_1p5b.jsonl`](bn_tn_1p5b.jsonl)
- [`bn_tn_1p5b_lm_head.jsonl`](bn_tn_1p5b_lm_head.jsonl)
- [`bn_tn_7p2b.jsonl`](bn_tn_7p2b.jsonl)
- corresponding `*_summary.log` files preserve per-case winners.
