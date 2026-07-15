# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 4.181x | 5.470x | 10.366x | 6/6 |
| Decode RWKV/Qwen | 10.861x | 10.901x | 10.913x | 6/6 |
| Model footprint RWKV/Qwen | 0.599x | 0.599x | 0.599x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.628x | 0.655x | 0.668x | 6/6 |
| Runtime working set RWKV/Qwen | 1.410x | 1.632x | 1.864x | 0/6 |
| Active parameters RWKV/Qwen | 0.599x | 0.599x | 0.599x | 6/6 |
| Prefill tok/s per active-B | 6.978x | 9.130x | 17.303x | 6/6 |
| Decode tok/s per active-B | 18.129x | 18.196x | 18.215x | 6/6 |
| Prefill active-param work rate | 2.505x | 3.277x | - | 6/6 |
| Decode active-param work rate | 6.507x | 6.531x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 4.181x / 5.470x | 10.861x / 10.901x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
