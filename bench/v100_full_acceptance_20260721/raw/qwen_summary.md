# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `72/72` cells.

Required Qwen backend: `fla`; verified: `72/72` cells.

Required Qwen full fusion: `true`; verified: `72/72` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.194x | 1.435x | 3.245x | 72/72 |
| Decode RWKV/Qwen | 1.840x | 4.859x | 11.253x | 72/72 |
| Model footprint RWKV/Qwen | 0.701x | 0.804x | 0.812x | 72/72 |
| Peak VRAM RWKV/Qwen | 0.454x | 0.803x | 0.991x | 72/72 |
| Runtime working set RWKV/Qwen | 0.254x | 0.788x | 5.763x | 42/72 |
| Active parameters RWKV/Qwen | 0.701x | 0.804x | 0.812x | 72/72 |
| Prefill tok/s per active-B | 1.486x | 1.785x | 3.998x | 72/72 |
| Decode tok/s per active-B | 2.288x | 6.687x | 13.864x | 72/72 |
| Prefill active-param work rate | 0.960x | 1.150x | - | reported 72/72 |
| Decode active-param work rate | 1.479x | 3.825x | - | reported 72/72 |

Strict speed cells: `72/72`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 72 | 1.194x / 1.435x | 1.840x / 4.859x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
