# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.381x | 1.612x | 4.003x | 6/6 |
| Decode RWKV/Qwen | 7.169x | 7.199x | 7.251x | 6/6 |
| Model footprint RWKV/Qwen | 0.599x | 0.599x | 0.599x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.630x | 0.682x | 0.709x | 6/6 |
| Runtime working set RWKV/Qwen | 0.682x | 0.983x | 1.132x | 4/6 |
| Active parameters RWKV/Qwen | 0.599x | 0.599x | 0.599x | 6/6 |
| Prefill tok/s per active-B | 2.306x | 2.690x | 6.682x | 6/6 |
| Decode tok/s per active-B | 11.966x | 12.016x | 12.103x | 6/6 |
| Prefill active-param work rate | 0.828x | 0.966x | - | 6/6 |
| Decode active-param work rate | 4.295x | 4.313x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.381x / 1.612x | 7.169x / 7.199x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
