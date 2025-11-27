# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.031x | 1.041x | 1.112x | 6/6 |
| Decode RWKV/Qwen | 2.813x | 2.830x | 3.552x | 6/6 |
| Model footprint RWKV/Qwen | 0.804x | 0.804x | 0.804x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.997x | 1.028x | 1.042x | 1/6 |
| Runtime working set RWKV/Qwen | 2.539x | 4.119x | 6.729x | 0/6 |
| Active parameters RWKV/Qwen | 0.804x | 0.804x | 0.804x | 6/6 |
| Prefill tok/s per active-B | 1.282x | 1.295x | 1.383x | 6/6 |
| Decode tok/s per active-B | 3.499x | 3.520x | 4.417x | 6/6 |
| Prefill active-param work rate | 0.829x | 0.837x | - | 6/6 |
| Decode active-param work rate | 2.262x | 2.275x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.031x / 1.041x | 2.813x / 2.830x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
