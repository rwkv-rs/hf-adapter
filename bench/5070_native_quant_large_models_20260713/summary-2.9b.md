# Native quant end-to-end matrix summary

- Completion: `42/42` rows
- Failed attempts: `0`
- Unresolved failures: `0`
- Execution complete: `yes`
- All quant paths accepted: `no`

| Model / quant / fusion | Rows | Speed >=fp16 | Decode/fp16 median | Min | Max | Footprint median | Min final cosine | Greedy match | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.9b/mm4/off | 7 | 7/7 | 1.2067 | 1.1012 | 1.3737 | 0.5310 | 0.96761703 | 0/7 | no |
| 2.9b/mm4/up | 7 | 7/7 | 1.2030 | 1.1518 | 1.3834 | 0.5310 | 0.96757555 | 0/7 | no |
| 2.9b/mm8/deep | 7 | 7/7 | 1.1542 | 1.1019 | 1.1918 | 0.6876 | 0.99995422 | 7/7 | yes |
| 2.9b/mm8/off | 7 | 7/7 | 1.1416 | 1.0870 | 1.1887 | 0.6876 | 0.99995452 | 7/7 | yes |
| 2.9b/mm8/up | 7 | 7/7 | 1.1363 | 1.0567 | 1.1906 | 0.6876 | 0.99995315 | 7/7 | yes |
| 2.9b/none/off | 7 | 7/7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.00000000 | 7/7 | n/a |

| Paired comparison | Cells | Right wins | >=0.99x | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| mm4_up_vs_off | 7 | 5 | 7 | 1.0071 | 0.9969 | 1.0460 |
| mm8_up_vs_off | 7 | 2 | 3 | 0.9891 | 0.9228 | 1.0095 |
| mm8_deep_vs_up | 7 | 5 | 6 | 1.0157 | 0.9762 | 1.0922 |
