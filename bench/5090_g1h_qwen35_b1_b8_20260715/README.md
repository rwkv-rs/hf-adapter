# RTX 5090 RWKV-7 vs full-FLA Qwen3.5 B1/B8

Status: **PASS, complete 8/8 batch-pairs on current main**.

This artifact compares:

- RWKV-7 g1d 0.4B vs official Qwen3.5 0.8B
- RWKV-7 g1h 1.5B vs official Qwen3.5 2B
- RWKV-7 g1h 2.9B vs official Qwen3.5 4B
- RWKV-7 g1h 7.2B vs official Qwen3.5 9B

The official small checkpoint used by this matrix is g1d 0.4B; the current
g1h release starts at 1.5B. This is an inference speed, memory, and numerical
alignment result, not a model-quality comparison.

## Contract

- Code base: upstream main `5f193610c7be62db7301045f870b2fd84e6bc8f9`
  plus the RTX 5090 changes in this PR
- GPU: NVIDIA GeForce RTX 5090, `sm_120`
- Runtime: PyTorch `2.11.0+cu128`, CUDA `12.8`, Triton `3.6.0`,
  Transformers `5.12.1`, FLA `0.5.1`, bitsandbytes `0.49.2`
- Dtype: fp16
- Shapes per batch-pair: prompt `128/512/2048` x decode `128/512`
- Batch sizes: B1 and B8 are reported separately
- RWKV route: repository `native_prefill_graph` plus native-graph decode
- Qwen route: FLA chunk prefill, fused-recurrent decode, fused gated norm, and
  the repository FLA Triton causal-convolution bridge
- Quant lanes: dense, W8, and W4 with paired-fp16 total-latency, footprint,
  peak-VRAM, logits, and greedy-token gates

All 144 Qwen performance rows verify the full fused operator contract. Torch
attention or Torch convolution fallback cannot satisfy the summary gate.

## Results

Each value below is the minimum across six prompt/decode cells. Quant speed is
minimum total-latency speedup against paired RWKV fp16; footprint is the
maximum model-footprint ratio against paired RWKV fp16.

| Batch | Pair | Dense prefill vs Qwen | Dense decode vs Qwen | Active-work decode | W8 total / footprint | W4 total / footprint |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.4B / 0.8B | `4.1807x` | `10.8611x` | `6.5071x` | `1.0076x / 0.9258x` | `1.0180x / 0.8907x` |
| 8 | 0.4B / 0.8B | `1.3814x` | `7.1688x` | `4.2949x` | `1.0070x / 0.9258x` | `1.0089x / 0.8907x` |
| 1 | 1.5B / 2B | `2.8112x` | `6.6869x` | `5.4274x` | `1.0965x / 0.5607x` | `1.0241x / 0.9342x` |
| 8 | 1.5B / 2B | `1.0226x` | `4.5588x` | `3.7002x` | `1.1565x / 0.6046x` | `1.0123x / 0.9355x` |
| 1 | 2.9B / 4B | `2.3681x` | `5.0057x` | `3.5084x` | `1.1292x / 0.5447x` | `1.3077x / 0.3597x` |
| 8 | 2.9B / 4B | `1.3003x` | `3.7833x` | `2.6516x` | `1.2147x / 0.5731x` | `1.0070x / 0.9573x` |
| 1 | 7.2B / 9B | `1.1739x` | `2.8934x` | `2.3263x` | `1.4049x / 0.5339x` | `1.5666x / 0.3288x` |
| 8 | 7.2B / 9B | `1.0309x` | `2.8130x` | `2.2618x` | `1.3720x / 0.5526x` | `1.0063x / 0.9726x` |

Coverage is 8/8 batch-pairs, 144 candidate rows, 144 joined Qwen reference
rows, and 144/144 verified full-FLA reference contracts. All 32 correctness
reports pass greedy-token equality; the minimum recorded logits cosine is
`0.99943173`. Every pair passes all 18 dense/W8/W4 cells.

Raw prefill and decode lead Qwen in every checked cell. Tokens/s per active
billion parameters also lead in every cell. The stricter active-parameter work
rate, defined as throughput multiplied by active parameter count, is below
`1.0x` for some prefill cells, including all B8 pair minima, so this artifact
does not claim an all-cell active-work prefill lead.

Dense RWKV model footprint is lower in all cells. Peak VRAM is not universally
lower: B8 1.5B/2B reaches `1.0801x` and B8 7.2B/9B reaches `1.0418x` of Qwen.
The W8/W4 claims in the table are RWKV-local selected-route footprint and
total-latency results, not universal full-memory quantization claims.

## Exact-shape evidence

The 7.2B B8 prompt128 baseline exposed one `0.987x` prefill cell. The retained
15-way A/B sweep selected stacked R/K/V alone; clampw and sequence-FFN
combinations were slower. The formal prompt128/decode512 confirmation measured
`80.957 ms` for RWKV and `82.986 ms` for full-FLA Qwen (`1.0251x`). The full
B8 matrix then passed with a `1.0309x` minimum. This policy is restricted to
`(hidden=4096, layers=32, B=8, T=128)` on an exact RTX 5090.

The first Qwen3.5-9B conv-oracle probe at prompt128 had cosine `0.9998268` but
diverged after the third greedy token. Prompt512 passes all eight greedy tokens
with prompt/final cosine `0.9998738/0.9994317`; 9B correctness therefore uses
prompt512 while performance still covers prompt128/512/2048. The four official
9B shards are independently size-checked, indexed, opened as safetensors, and
SHA256-recorded under `model_verification/`.

## Reproduce the summary

```bash
python bench/summarize_5090_qwen35_acceptance.py \
  bench/5090_g1h_qwen35_b1_b8_20260715 \
  --output bench/5090_g1h_qwen35_b1_b8_20260715/summary.json
```

The summarizer is fail-closed over all eight batch-pairs.
