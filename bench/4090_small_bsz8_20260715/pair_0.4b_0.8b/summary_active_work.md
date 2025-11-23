# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.370x | 1.750x | 3.930x | 6/6 |
| Decode RWKV/Qwen | 12.102x | 12.148x | 12.191x | 6/6 |
| Model footprint RWKV/Qwen | 0.599x | 0.599x | 0.599x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.750x | 0.825x | 0.856x | 6/6 |
| Runtime working set RWKV/Qwen | 0.996x | 1.483x | 2.052x | 1/6 |
| Active parameters RWKV/Qwen | 0.599x | 0.599x | 0.599x | 6/6 |
| Prefill tok/s per active-B | 2.287x | 2.921x | 6.560x | 6/6 |
| Decode tok/s per active-B | 20.200x | 20.277x | 20.349x | 6/6 |
| Prefill active-param work rate | 0.821x | 1.049x | - | 6/6 |
| Decode active-param work rate | 7.250x | 7.278x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.370x / 1.750x | 12.102x / 12.148x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
