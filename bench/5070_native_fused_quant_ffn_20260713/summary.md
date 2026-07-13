# Native quant end-to-end matrix summary

- Completion: `42/42` rows
- Failed attempts: `0`
- Unresolved failures: `0`
- Complete: `yes`

| Model / quant / fusion | Rows | Decode/fp16 median | Min | Max | Footprint median | Min final cosine | Greedy match |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5b/mm4/off | 7 | 0.8171 | 0.8025 | 0.9868 | 0.5394 | 0.99809140 | 7/7 |
| 1.5b/mm4/up | 7 | 0.8171 | 0.7870 | 0.9911 | 0.5394 | 0.99808919 | 7/7 |
| 1.5b/mm8/deep | 7 | 0.9671 | 0.9471 | 1.0893 | 0.6932 | 0.99995530 | 7/7 |
| 1.5b/mm8/off | 7 | 0.9551 | 0.9413 | 1.0820 | 0.6932 | 0.99995482 | 7/7 |
| 1.5b/mm8/up | 7 | 0.9620 | 0.9472 | 1.0852 | 0.6932 | 0.99995518 | 7/7 |
| 1.5b/none/off | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.00000000 | 7/7 |

| Paired comparison | Cells | Right wins | >=0.99x | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| mm4_up_vs_off | 7 | 4 | 6 | 1.0014 | 0.9770 | 1.0071 |
| mm8_up_vs_off | 7 | 7 | 7 | 1.0062 | 1.0028 | 1.0194 |
| mm8_deep_vs_up | 7 | 5 | 6 | 1.0059 | 0.9888 | 1.0162 |
