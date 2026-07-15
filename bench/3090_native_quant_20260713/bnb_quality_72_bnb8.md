# Uncheatable logit compression alignment

Teacher-forced external-token NLL benchmark. Lower bits/token is better; candidate/reference ratio near `1.0` means the candidate compresses the fixed external text like the reference.

## Summary

| metric | value |
|---|---:|
| reference bits/token | `4.04740845` |
| candidate bits/token | `4.05337643` |
| candidate/reference bits ratio | `1.00147452` |
| candidate-reference bits/token | `0.00596799` |
| tokens scored | `950` |

## Compression ratio vs token position

| token position bin | reference bpt | candidate bpt | cand/ref ratio | tokens |
|---|---:|---:|---:|---:|
| `1` | `1.42643312` | `1.70286475` | `1.19379222` | `8` |
| `2-8` | `6.85553189` | `6.92853622` | `1.01064897` | `56` |
| `9-16` | `5.51014433` | `5.64582996` | `1.02462470` | `64` |
| `17-32` | `3.79751678` | `3.79017340` | `0.99806627` | `128` |
| `33-64` | `4.06199817` | `4.01206144` | `0.98770636` | `256` |
| `65-128` | `3.75848831` | `3.77846979` | `1.00531636` | `377` |
| `129-256` | `2.52727423` | `2.47607406` | `0.97974095` | `61` |

## Worst texts by candidate/reference ratio

| index | unique_id | tokens | ref bpt | cand bpt | ratio | delta bpt |
|---:|---|---:|---:|---:|---:|---:|
| `2` | `2` | `110` | `2.84314104` | `2.91893005` | `1.02665679` | `0.07578902` |
| `7` | `7` | `99` | `6.31244328` | `6.38507635` | `1.01150633` | `0.07263307` |
| `0` | `0` | `120` | `4.32907817` | `4.35805613` | `1.00669380` | `0.02897797` |
| `5` | `5` | `111` | `4.83892327` | `4.86435954` | `1.00525660` | `0.02543626` |
| `3` | `3` | `119` | `4.62854466` | `4.59739005` | `0.99326903` | `-0.03115460` |
| `6` | `6` | `87` | `4.81816329` | `4.77894049` | `0.99185939` | `-0.03922280` |
| `1` | `1` | `189` | `3.25179101` | `3.22403713` | `0.99146505` | `-0.02775388` |
| `4` | `4` | `115` | `2.31465428` | `2.28166822` | `0.98574904` | `-0.03298605` |
