# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 2.389x | 2.716x | 5.955x | 6/6 |
| Decode RWKV/Qwen | 5.210x | 5.243x | 5.324x | 6/6 |
| Model footprint RWKV/Qwen | 0.701x | 0.701x | 0.701x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.703x | 0.713x | 0.717x | 6/6 |
| Runtime working set RWKV/Qwen | 0.747x | 1.489x | 1.626x | 1/6 |
| Active parameters RWKV/Qwen | 0.701x | 0.701x | 0.701x | 6/6 |
| Prefill tok/s per active-B | 3.409x | 3.875x | 8.496x | 6/6 |
| Decode tok/s per active-B | 7.433x | 7.481x | 7.597x | 6/6 |
| Prefill active-param work rate | 1.675x | 1.904x | - | 6/6 |
| Decode active-param work rate | 3.652x | 3.675x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 2.389x / 2.716x | 5.210x / 5.243x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
