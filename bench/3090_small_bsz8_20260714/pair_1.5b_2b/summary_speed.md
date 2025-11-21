# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.014x | 1.097x | 1.762x | 18/18 |
| Decode RWKV/Qwen | 3.383x | 3.531x | 4.086x | 18/18 |
| Model footprint RWKV/Qwen | 0.491x | 0.758x | 0.812x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.584x | 0.905x | 1.176x | 14/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.031x / 1.085x | 3.383x / 3.415x | - | - | - | - | - |
| w4 | 6 | 1.014x / 1.075x | 3.483x / 3.531x | 0.983x | 1.028x | 1.011x | 0.934x | 0.969x |
| w8 | 6 | 1.474x / 1.623x | 3.998x / 4.064x | 1.373x | 1.182x | 1.193x | 0.605x | 0.817x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
