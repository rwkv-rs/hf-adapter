# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `false`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.000x | 1.184x | 2.099x | 18/18 |
| Decode RWKV/Qwen | 2.210x | 2.274x | 3.027x | 18/18 |
| Model footprint RWKV/Qwen | 0.429x | 0.782x | 0.804x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.490x | 1.159x | 1.209x | 6/18 |
| Runtime working set RWKV/Qwen | 1.308x | 4.149x | 11.394x | 0/18 |
| Active parameters RWKV/Qwen | 0.774x | 0.789x | 0.804x | 18/18 |
| Prefill tok/s per active-B | 1.244x | 1.472x | 2.711x | 18/18 |
| Decode tok/s per active-B | 2.749x | 2.879x | 3.910x | 18/18 |
| Prefill active-param work rate | 0.778x | 0.917x | - | reported 18/18 |
| Decode active-param work rate | 1.750x | 1.810x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.024x / 1.119x | 2.210x / 2.223x | - | - | - | - | - |
| w4 | 6 | 1.000x / 1.118x | 2.261x / 2.274x | 0.977x | 1.023x | 1.013x | 0.973x | 0.983x |
| w8 | 6 | 1.509x / 1.943x | 3.002x / 3.017x | 1.473x | 1.357x | 1.360x | 0.534x | 0.456x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
