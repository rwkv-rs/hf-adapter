# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.023x | 1.190x | 2.586x | 6/6 |
| Decode RWKV/Qwen | 4.559x | 4.594x | 5.766x | 6/6 |
| Model footprint RWKV/Qwen | 0.812x | 0.812x | 0.812x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.891x | 1.014x | 1.080x | 2/6 |
| Runtime working set RWKV/Qwen | 1.810x | 1.994x | 2.775x | 0/6 |
| Active parameters RWKV/Qwen | 0.812x | 0.812x | 0.812x | 6/6 |
| Prefill tok/s per active-B | 1.260x | 1.467x | 3.186x | 6/6 |
| Decode tok/s per active-B | 5.617x | 5.660x | 7.104x | 6/6 |
| Prefill active-param work rate | 0.830x | 0.966x | - | 6/6 |
| Decode active-param work rate | 3.700x | 3.729x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.023x / 1.190x | 4.559x / 4.594x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
