# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: FAIL

Coverage: `216/216` cells.

| Metric | Minimum | Median |
|---|---:|---:|
| Prefill RWKV/Qwen | 1.246x | 1.936x |
| Decode RWKV/Qwen | 0.947x | 1.317x |

## Red cells

| Pair | Prompt | Decode | Bsz | Quant | Prefill | Decode | Candidate | Reference |
|---|---:|---:|---:|---|---:|---:|---|---|
| rwkv-2.9b__qwen3.5-4b | 128 | 128 | 4 | bnb4 | 1.942x | 1.027x | pass | pass |
| rwkv-2.9b__qwen3.5-4b | 128 | 512 | 2 | bnb4 | 1.893x | 1.017x | pass | pass |
| rwkv-2.9b__qwen3.5-4b | 128 | 512 | 4 | bnb4 | 1.956x | 1.038x | pass | pass |
| rwkv-2.9b__qwen3.5-4b | 2048 | 128 | 1 | bnb4 | 1.996x | 0.955x | pass | pass |
| rwkv-2.9b__qwen3.5-4b | 512 | 128 | 2 | bnb4 | 2.026x | 0.998x | pass | pass |
| rwkv-2.9b__qwen3.5-4b | 512 | 512 | 2 | bnb4 | 2.087x | 1.016x | pass | pass |
| rwkv-7.2b__qwen3.5-9b | 128 | 512 | 4 | bnb4 | 1.563x | 1.015x | pass | pass |
| rwkv-7.2b__qwen3.5-9b | 2048 | 128 | 1 | bnb4 | 1.362x | 0.947x | pass | pass |
| rwkv-7.2b__qwen3.5-9b | 2048 | 128 | 4 | bnb4 | 1.427x | 1.003x | pass | pass |

Missing candidate rows: `0`.
Missing reference rows: `0`.
