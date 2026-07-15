# RTX 5090 RWKV-7 vs full-FLA Qwen3.5 B1/B8 (staged)

Status: **PASS for the six checked batch-pairs; full 8-pair matrix incomplete**.

This staged artifact records the completed B1 and B8 comparisons for:

- RWKV-7 g1d 0.4B vs official Qwen3.5 0.8B
- RWKV-7 g1h 1.5B vs official Qwen3.5 2B
- RWKV-7 g1h 2.9B vs official Qwen3.5 4B

The remaining g1h 7.2B vs Qwen3.5 9B pair and the separate official g1h
13.3B validation are still running and are not represented as passing here.

## Contract

- GPU: NVIDIA GeForce RTX 5090, `sm_120`
- Runtime: PyTorch `2.11.0+cu128`, CUDA `12.8`, Triton `3.6.0`,
  Transformers `5.12.1`, FLA `0.5.1`, bitsandbytes `0.49.2`
- Dtype: fp16
- Shapes per batch-pair: prompt `128/512/2048` x decode `128/512`
- Batch sizes: B1 and B8 are reported separately
- RWKV route: `native_prefill_graph` plus native-graph decode
- Qwen route: FLA chunk prefill, fused-recurrent decode, fused gated norm, and
  the FLA Triton causal-convolution bridge; Torch convolution fallback is
  rejected by the summarizer
- Quant lanes: dense, W8 and W4, with explicit route manifests, footprint,
  peak VRAM, logits and greedy-token checks

The benchmark run was collected from base `4fec6f1084dce518845aafb2512225759650f89b`
plus the RTX 5090 changes in this PR. The branch is now rebased onto upstream
`5f193610c7be62db7301045f870b2fd84e6bc8f9`; a fresh final rerun is in progress.
The intervening `native_jit.py` upstream change is RTX 4090-specific, but the
fresh rerun remains required before promoting the complete matrix.

## Staged results

Every row below is the minimum across the six prompt/decode shapes for that
batch-pair. Quant speed is the minimum total-latency ratio against paired fp16;
footprint is the maximum model-footprint ratio against paired fp16.

| Batch | Pair | Dense prefill vs Qwen | Dense decode vs Qwen | Active-work decode | W8 total / footprint | W4 total / footprint |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.4B / 0.8B | `4.1130x` | `10.7340x` | `6.4308x` | `1.0096x / 0.9258x` | `1.0158x / 0.8907x` |
| 8 | 0.4B / 0.8B | `1.3901x` | `7.1024x` | `4.2551x` | `1.0089x / 0.9258x` | `1.0133x / 0.8907x` |
| 1 | 1.5B / 2B | `2.7722x` | `6.5326x` | `5.3023x` | `1.0962x / 0.5607x` | `1.2623x / 0.4069x` |
| 8 | 1.5B / 2B | `1.0191x` | `4.4752x` | `3.6324x` | `1.1553x / 0.6046x` | `1.0124x / 0.9355x` |
| 1 | 2.9B / 4B | `2.3892x` | `5.2100x` | `3.6515x` | `1.1303x / 0.5447x` | `1.3067x / 0.3597x` |
| 8 | 2.9B / 4B | `1.3021x` | `3.9869x` | `2.7943x` | `1.2126x / 0.5731x` | `1.0048x / 0.9612x` |

Coverage is 6/6 selected batch-pairs, 108 candidate rows, 108 joined Qwen
reference rows, and 108/108 verified full-FLA operator contracts. All 24
correctness reports pass greedy-token equality. The B8 active-parameter-normalized
prefill ratio falls below `1.0x` in some cells; this staged artifact therefore
does not claim that every active-normalized prefill cell leads Qwen.

This is an inference speed, memory and numerical-alignment result. It is not a
claim of instruction, reasoning, math, code, multilingual or long-context model
quality superiority.

## Reproduce the staged summary

```bash
python bench/summarize_5090_qwen35_acceptance.py \
  bench/5090_g1h_qwen35_b1_b8_20260715 \
  --pair rwkv-0.4b__qwen3.5-0.8b \
  --pair rwkv-1.5b__qwen3.5-2b \
  --pair rwkv-2.9b__qwen3.5-4b \
  --output bench/5090_g1h_qwen35_b1_b8_20260715/partial_summary.json
```

The summarizer remains fail-closed over all eight batch-pairs when no `--pair`
arguments are supplied.
