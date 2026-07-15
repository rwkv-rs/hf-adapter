# RTX 3090 BnB quality gate

Fixed external-token teacher-forced NLL; lower bits/token is better.

| quant | ref bpt | candidate bpt | candidate/ref | tokens | gate |
|---|---:|---:|---:|---:|---|
| `bnb8` | `4.04740845` | `4.05337643` | `1.00147452` | `950` | `PASS` |
| `bnb4` | `4.04740845` | `4.05374033` | `1.00156443` | `950` | `PASS` |

Acceptance: ratio `<= 1.01` and tokens `>= 900`. Overall: **PASS**.
