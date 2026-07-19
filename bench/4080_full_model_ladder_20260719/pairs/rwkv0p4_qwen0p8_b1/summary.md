# RTX 4080 rwkv-0.4b__qwen3.5-0.8b B1 acceptance

Status: **pass**

| Axis | Measured range |
|---|---:|
| Dense prefill / full-FLA Qwen | 1.3852x - 4.0664x |
| Dense decode / full-FLA Qwen | 4.8595x - 4.9105x |
| Active-work decode ratio | 8.1112x - 8.1962x |
| a8w8 paired prefill/decode/total | 0.9913x / 1.0071x / 1.0049x minimum |
| torchao_w4 paired prefill/decode/total | 0.9734x / 1.0820x / 1.0701x minimum |
