# Uncheatable logit compression alignment

Teacher-forced external-token NLL benchmark. Lower bits/token is better; candidate/reference ratio near `1.0` means the candidate compresses the fixed external text like the reference.

## Summary

| metric | value |
|---|---:|
| reference bits/token | `4.04740845` |
| candidate bits/token | `4.05374033` |
| candidate/reference bits ratio | `1.00156443` |
| candidate-reference bits/token | `0.00633188` |
| tokens scored | `950` |

## Compression ratio vs token position

| token position bin | reference bpt | candidate bpt | cand/ref ratio | tokens |
|---|---:|---:|---:|---:|
| `1` | `1.42643312` | `3.11868140` | `2.18634955` | `8` |
| `2-8` | `6.85553189` | `6.73209363` | `0.98199436` | `56` |
| `9-16` | `5.51014433` | `5.42566445` | `0.98466830` | `64` |
| `17-32` | `3.79751678` | `3.83069164` | `1.00873593` | `128` |
| `33-64` | `4.06199817` | `4.06026556` | `0.99957346` | `256` |
| `65-128` | `3.75848831` | `3.77574391` | `1.00459110` | `377` |
| `129-256` | `2.52727423` | `2.43692162` | `0.96424899` | `61` |

## Worst texts by candidate/reference ratio

| index | unique_id | tokens | ref bpt | cand bpt | ratio | delta bpt |
|---:|---|---:|---:|---:|---:|---:|
| `4` | `4` | `115` | `2.31465428` | `2.36945774` | `1.02367674` | `0.05480347` |
| `7` | `7` | `99` | `6.31244328` | `6.44352337` | `1.02076535` | `0.13108009` |
| `0` | `0` | `120` | `4.32907817` | `4.38020020` | `1.01180899` | `0.05112203` |
| `5` | `5` | `111` | `4.83892327` | `4.87846416` | `1.00817142` | `0.03954088` |
| `6` | `6` | `87` | `4.81816329` | `4.84655937` | `1.00589355` | `0.02839608` |
| `3` | `3` | `119` | `4.62854466` | `4.57813372` | `0.98910869` | `-0.05041094` |
| `1` | `1` | `189` | `3.25179101` | `3.19052043` | `0.98115790` | `-0.06127058` |
| `2` | `2` | `110` | `2.84314104` | `2.76423978` | `0.97224856` | `-0.07890126` |
