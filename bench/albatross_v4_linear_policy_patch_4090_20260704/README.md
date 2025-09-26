# Albatross v4 linear policy patch smoke — 4090 — 2026-07-04

Patch: `att_c2c K=1024 rows512->orig rows64->cfg_t32_r3_o4; ffn_key K=1024 rows512/64->orig`.

| Case | baseline tok/s | tuned tok/s | tuned/baseline | baseline ms | tuned ms |
|---|---:|---:|---:|---:|---:|
| B1T1 | 856.63 | 832.96 | 0.972x | 1.167360 | 1.200540 |
| B1T512 | 59091.90 | 55594.50 | 0.941x | 8.664470 | 9.209550 |
| B64T1 | 25194.50 | 24366.50 | 0.967x | 2.540240 | 2.626560 |
| B4T128 | 89376.70 | 87919.80 | 0.984x | 5.728560 | 5.823490 |
| B8T64 | 96786.70 | 96796.10 | 1.000x | 5.289980 | 5.289470 |

Interpretation: this direct v4 binary patch did not improve the model-forward smoke despite faster isolated linear microbench buckets, so the microbench winners are not a safe full-model Albatross reference replacement yet. Keep the committed v4 smoke as the speed ceiling and treat deeper policy tuning as future Albatross-side work.
