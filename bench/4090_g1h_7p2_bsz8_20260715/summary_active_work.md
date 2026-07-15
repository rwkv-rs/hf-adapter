# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.024x | 1.119x | 1.184x | 6/6 |
| Decode RWKV/Qwen | 2.210x | 2.223x | 2.230x | 6/6 |
| Model footprint RWKV/Qwen | 0.804x | 0.804x | 0.804x | 6/6 |
| Peak VRAM RWKV/Qwen | 1.156x | 1.200x | 1.209x | 0/6 |
| Runtime working set RWKV/Qwen | 3.970x | 6.464x | 11.394x | 0/6 |
| Active parameters RWKV/Qwen | 0.804x | 0.804x | 0.804x | 6/6 |
| Prefill tok/s per active-B | 1.274x | 1.391x | 1.472x | 6/6 |
| Decode tok/s per active-B | 2.749x | 2.765x | 2.774x | 6/6 |
| Prefill active-param work rate | 0.823x | 0.900x | - | 6/6 |
| Decode active-param work rate | 1.777x | 1.787x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.024x / 1.119x | 2.210x / 2.223x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
