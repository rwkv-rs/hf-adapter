# Uncheatable logit compression alignment

Teacher-forced external-token NLL benchmark. Lower bits/token is better; candidate/reference ratio near `1.0` means the candidate compresses the fixed external text like the reference.

## Summary

| metric | value |
|---|---:|
| reference bits/token | `4.04740845` |
| candidate bits/token | `4.06661165` |
| candidate/reference bits ratio | `1.00474457` |
| candidate-reference bits/token | `0.01920320` |
| tokens scored | `950` |
| quality gate | `PASS` |

## Compression ratio vs token position

| token position bin | reference bpt | candidate bpt | cand/ref ratio | tokens |
|---|---:|---:|---:|---:|
| `1` | `1.42643312` | `1.36911950` | `0.95982032` | `8` |
| `2-8` | `6.85553189` | `6.91966850` | `1.00935545` | `56` |
| `9-16` | `5.51014433` | `5.50147119` | `0.99842597` | `64` |
| `17-32` | `3.79751678` | `3.79211484` | `0.99857751` | `128` |
| `33-64` | `4.06199817` | `4.07041434` | `1.00207193` | `256` |
| `65-128` | `3.75848831` | `3.79157052` | `1.00880200` | `377` |
| `129-256` | `2.52727423` | `2.55563206` | `1.01122072` | `61` |

## Worst texts by candidate/reference ratio

| index | unique_id | tokens | ref bpt | cand bpt | ratio | delta bpt |
|---:|---|---:|---:|---:|---:|---:|
| `0` | `0` | `120` | `4.32907817` | `4.39532267` | `1.01530222` | `0.06624451` |
| `2` | `2` | `110` | `2.84314104` | `2.87775241` | `1.01217364` | `0.03461137` |
| `7` | `7` | `99` | `6.31244328` | `6.37290631` | `1.00957839` | `0.06046303` |
| `4` | `4` | `115` | `2.31465428` | `2.33627213` | `1.00933956` | `0.02161786` |
| `1` | `1` | `189` | `3.25179101` | `3.26911143` | `1.00532643` | `0.01732042` |
| `5` | `5` | `111` | `4.83892327` | `4.83862855` | `0.99993909` | `-0.00029473` |
| `6` | `6` | `87` | `4.81816329` | `4.79349632` | `0.99488042` | `-0.02466697` |
| `3` | `3` | `119` | `4.62854466` | `4.60266011` | `0.99440763` | `-0.02588455` |
