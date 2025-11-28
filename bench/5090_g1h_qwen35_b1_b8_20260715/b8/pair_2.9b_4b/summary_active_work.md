# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.300x | 1.481x | 2.133x | 6/6 |
| Decode RWKV/Qwen | 3.783x | 3.815x | 3.892x | 6/6 |
| Model footprint RWKV/Qwen | 0.701x | 0.701x | 0.701x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.705x | 0.751x | 0.766x | 6/6 |
| Runtime working set RWKV/Qwen | 0.722x | 1.159x | 1.330x | 2/6 |
| Active parameters RWKV/Qwen | 0.701x | 0.701x | 0.701x | 6/6 |
| Prefill tok/s per active-B | 1.855x | 2.113x | 3.044x | 6/6 |
| Decode tok/s per active-B | 5.398x | 5.442x | 5.553x | 6/6 |
| Prefill active-param work rate | 0.911x | 1.038x | - | 6/6 |
| Decode active-param work rate | 2.652x | 2.674x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.300x / 1.481x | 3.783x / 3.815x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
