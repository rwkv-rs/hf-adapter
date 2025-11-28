# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.342x | 1.614x | 4.396x | 18/18 |
| Decode RWKV/Qwen | 6.905x | 7.252x | 7.356x | 18/18 |
| Model footprint RWKV/Qwen | 0.398x | 0.555x | 0.599x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.555x | 0.648x | 0.709x | 18/18 |
| Runtime working set RWKV/Qwen | 0.682x | 0.983x | 1.132x | 12/18 |
| Active parameters RWKV/Qwen | 0.510x | 0.599x | 0.599x | 18/18 |
| Prefill tok/s per active-B | 2.306x | 2.720x | 8.622x | 18/18 |
| Decode tok/s per active-B | 11.525x | 12.191x | 14.329x | 18/18 |
| Prefill active-param work rate | 0.684x | 0.967x | - | reported 18/18 |
| Decode active-param work rate | 3.684x | 4.305x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.381x / 1.612x | 7.169x / 7.199x | - | - | - | - | - |
| w4 | 6 | 1.342x / 1.615x | 7.253x / 7.304x | 0.972x | 1.012x | 1.009x | 0.891x | 0.936x |
| w8 | 6 | 1.386x / 1.936x | 6.905x / 7.255x | 1.000x | 0.961x | 1.007x | 0.926x | 0.956x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
