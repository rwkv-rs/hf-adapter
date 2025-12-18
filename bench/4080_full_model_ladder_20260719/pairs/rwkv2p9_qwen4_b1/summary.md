# RTX 4080 rwkv-2.9b__qwen3.5-4b B1 acceptance

Status: **pass**

| Axis | Measured range |
|---|---:|
| Dense prefill / full-FLA Qwen | 1.0621x - 1.6654x |
| Dense decode / full-FLA Qwen | 1.6121x - 1.6513x |
| Active-work decode ratio | 2.3001x - 2.3561x |
| a8w8 paired prefill/decode/total | 0.9981x / 1.0138x / 1.0134x minimum |
| torchao_w4 paired prefill/decode/total | 0.9987x / 1.0246x / 1.0232x minimum |
