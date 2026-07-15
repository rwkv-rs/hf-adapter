# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: FAIL

Coverage: `72/72` cells.

Required Qwen backend: `fla`; verified: `72/72` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 0.508x | 1.040x | 2.593x | 35/72 |
| Decode RWKV/Qwen | 1.019x | 1.334x | 3.635x | 71/72 |
| Model footprint RWKV/Qwen | 0.729x | 0.772x | 0.812x | 72/72 |
| Peak VRAM RWKV/Qwen | 0.742x | 0.793x | 0.913x | 72/72 |

Strict speed cells: `35/72`.

## Red cells

| Pair | Prompt | Decode | Bsz | Quant | Prefill | Decode | Qwen backend | Candidate | Reference |
|---|---:|---:|---:|---|---:|---:|---|---|---|
| rwkv-1.5b__qwen3.5-2b | 128 | 128 | 1 | none | 0.935x | 3.146x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 128 | 2 | bnb4 | 0.585x | 1.019x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 128 | 2 | none | 0.900x | 2.777x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 128 | 4 | bnb4 | 0.683x | 1.053x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 128 | 4 | bnb8 | 1.029x | 1.282x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 128 | 8 | bnb4 | 0.857x | 1.152x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 128 | 8 | none | 0.844x | 2.837x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 512 | 2 | bnb8 | 0.508x | 1.229x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 512 | 4 | bnb4 | 0.976x | 1.257x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 512 | 4 | none | 1.033x | 2.847x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 512 | 8 | bnb4 | 0.874x | 1.281x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 512 | 8 | bnb8 | 0.923x | 1.315x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 128 | 512 | 8 | none | 0.889x | 2.388x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 2048 | 128 | 1 | bnb4 | 1.035x | 1.178x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 2048 | 128 | 1 | bnb8 | 0.983x | 1.258x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 128 | 1 | bnb4 | 0.764x | 1.126x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 128 | 1 | none | 0.986x | 2.745x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 128 | 2 | bnb4 | 0.867x | 1.159x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 128 | 2 | none | 0.864x | 2.658x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 128 | 4 | bnb4 | 0.873x | 1.103x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 128 | 4 | bnb8 | 0.893x | 1.320x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 128 | 4 | none | 0.868x | 2.372x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 128 | 8 | bnb4 | 0.947x | 1.184x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 128 | 8 | bnb8 | 0.955x | 1.410x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 128 | 8 | none | 0.921x | 2.407x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 1 | bnb4 | 0.793x | 1.177x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 1 | bnb8 | 0.793x | 1.320x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 1 | none | 1.029x | 2.705x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 2 | bnb4 | 0.866x | 1.237x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 2 | bnb8 | 1.002x | 1.273x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 2 | none | 0.855x | 2.616x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 4 | bnb4 | 0.933x | 1.118x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 4 | bnb8 | 0.918x | 1.340x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 4 | none | 0.914x | 2.481x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 8 | bnb4 | 1.044x | 1.240x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 8 | bnb8 | 0.971x | 1.386x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |
| rwkv-1.5b__qwen3.5-2b | 512 | 512 | 8 | none | 0.996x | 2.476x | qwen_fla_gated_delta_rule_torch_conv | pass | pass |

Missing candidate rows: `0`.
Missing reference rows: `0`.
