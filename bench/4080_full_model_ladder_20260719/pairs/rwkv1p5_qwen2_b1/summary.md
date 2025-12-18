# RTX 4080 rwkv-1.5b__qwen3.5-2b B1 acceptance

Status: **pass**

| Axis | Measured range |
|---|---:|
| Dense prefill / full-FLA Qwen | 1.0123x - 2.3899x |
| Dense decode / full-FLA Qwen | 1.9029x - 1.9170x |
| Active-work decode ratio | 2.3444x - 2.3618x |
| a8w8 paired prefill/decode/total | 0.9978x / 1.0200x / 1.0201x minimum |
| torchao_w4 paired prefill/decode/total | 0.9923x / 1.0412x / 1.0369x minimum |
