# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.390x | 1.613x | 5.021x | 6/6 |
| Decode RWKV/Qwen | 7.102x | 7.129x | 7.138x | 6/6 |
| Model footprint RWKV/Qwen | 0.599x | 0.599x | 0.599x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.630x | 0.682x | 0.709x | 6/6 |
| Runtime working set RWKV/Qwen | 0.682x | 0.983x | 1.132x | 4/6 |
| Active parameters RWKV/Qwen | 0.599x | 0.599x | 0.599x | 6/6 |
| Prefill tok/s per active-B | 2.320x | 2.693x | 8.381x | 6/6 |
| Decode tok/s per active-B | 11.855x | 11.899x | 11.915x | 6/6 |
| Prefill active-param work rate | 0.833x | 0.967x | - | 6/6 |
| Decode active-param work rate | 4.255x | 4.271x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.390x / 1.613x | 7.102x / 7.129x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
