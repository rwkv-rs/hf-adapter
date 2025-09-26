# 4090 MATH500 sampling/refill stochasticity — 2026-07-03

## Observed full-run gap

| Metric | HF | Albatross | HF - Albatross |
|---|---:|---:|---:|
| Correct generations | 4421 | 4670 | -249 |
| Rollout accuracy | 0.13815625 | 0.14593750 | -0.00778125 |
| Pass tasks | 179 | 185 | -6 |
| Pass@rollout | 0.358000 | 0.370000 | -0.012000 |

## Prefix pass curve

| k | HF pass@k | Albatross pass@k | HF - Albatross | HF correct | Albatross correct |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.144000 | 0.142000 | +0.002000 | 72 | 71 |
| 2 | 0.190000 | 0.182000 | +0.008000 | 145 | 148 |
| 4 | 0.218000 | 0.214000 | +0.004000 | 283 | 286 |
| 8 | 0.246000 | 0.248000 | -0.002000 | 558 | 575 |
| 16 | 0.274000 | 0.298000 | -0.024000 | 1121 | 1192 |
| 32 | 0.316000 | 0.334000 | -0.018000 | 2228 | 2363 |
| 64 | 0.358000 | 0.370000 | -0.012000 | 4421 | 4670 |

## Empirical stochasticity estimate

Method: `empirical per-task Binomial(n, observed_correct_rate) repeated-rollout bootstrap` with `20000` draws and seed `7`.

| Quantity | Value |
|---|---:|
| Expected HF pass tasks | 167.292 |
| Expected Albatross pass tasks | 174.014 |
| Expected delta | -6.722 |
| Delta draw p2.5 / p50 / p97.5 | -14.0 / -7.0 / 1.0 |
| P(delta >= 0) | 0.0546 |
| P(delta <= observed delta) | 0.6260 |
| P(abs(delta) >= abs(observed)) | 0.6271 |

## Interpretation

- The empirical bootstrap 95% interval for pass-task delta includes zero, so the observed pass@64 gap is not strong evidence of a large deterministic model-math mismatch by itself.
- Use this together with logits parity and targeted reruns; it is not a substitute for the final full MATH500 avg@64 acceptance run.
