# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.059x | 1.064x | 1.126x | 6/6 |
| Decode RWKV/Qwen | 1.788x | 1.808x | 1.822x | 6/6 |
| Model footprint RWKV/Qwen | 0.804x | 0.804x | 0.804x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.850x | 0.897x | 0.995x | 6/6 |
| Runtime working set RWKV/Qwen | 1.842x | 2.045x | 2.234x | 0/6 |
| Active parameters RWKV/Qwen | 0.804x | 0.804x | 0.804x | 6/6 |
| Prefill tok/s per active-B | 1.317x | 1.323x | 1.400x | 6/6 |
| Decode tok/s per active-B | 2.224x | 2.249x | 2.266x | 6/6 |
| Prefill active-param work rate | 0.851x | 0.855x | - | 6/6 |
| Decode active-param work rate | 1.438x | 1.454x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.059x / 1.064x | 1.788x / 1.808x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
