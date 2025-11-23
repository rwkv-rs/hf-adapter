# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `false`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.369x | 1.753x | 4.048x | 18/18 |
| Decode RWKV/Qwen | 12.102x | 12.351x | 12.709x | 18/18 |
| Model footprint RWKV/Qwen | 0.532x | 0.555x | 0.599x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.709x | 0.800x | 0.856x | 18/18 |
| Runtime working set RWKV/Qwen | 0.996x | 1.483x | 2.052x | 3/18 |
| Active parameters RWKV/Qwen | 0.510x | 0.599x | 0.599x | 18/18 |
| Prefill tok/s per active-B | 2.286x | 2.932x | 7.930x | 18/18 |
| Decode tok/s per active-B | 20.200x | 21.179x | 24.924x | 18/18 |
| Prefill active-param work rate | 0.701x | 1.049x | - | reported 18/18 |
| Decode active-param work rate | 6.267x | 7.260x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.370x / 1.750x | 12.102x / 12.148x | - | - | - | - | - |
| w4 | 6 | 1.369x / 1.757x | 12.605x / 12.677x | - | - | - | 0.891x | 0.946x |
| w8 | 6 | 1.375x / 1.750x | 12.291x / 12.351x | - | - | - | 0.926x | 0.963x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
