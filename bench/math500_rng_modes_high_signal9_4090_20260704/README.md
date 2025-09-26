# 4090 MATH500 high-signal-9 RNG/refill probe — 2026-07-04

Subset original task IDs: `73,160,116,67,277,374,383,319,72`.

| Runner / RNG mode | Correct generations | Rollout accuracy | Pass@64 | Summary token/s |
|---|---:|---:|---:|---:|
| HF global full-batch RNG (default / Albatross-compatible) | 315/576 | 0.54687500 | 0.888889 | 6241.051 |
| HF active-row global RNG | 297/576 | 0.51562500 | 0.888889 | 6188.883 |
| HF deterministic per-sample RNG | 302/576 | 0.52430556 | 0.888889 | 6080.989 |
| Albatross v3a | 325/576 | 0.56423611 | 0.888889 | 3187.349 |

## Interpretation

- Default HF `global` mode remains best among the tested HF RNG/refill variants on this subset: `315/576` correct and `8/9` pass@64.
- `active_global` drops to `297/576`; deterministic `per_sample` drops to `302/576`; both keep pass@64 tied at `8/9`.
- Therefore the immediate fix is **not** to switch the acceptance benchmark away from Albatross-compatible global full-batch sampling. The remaining work should focus on seed/refill sensitivity measurement or a full avg@64 rerun only after a targeted variant beats the default on subsets.

## Artifacts

- `rng_mode_summary.json`
- `active_global_summary.json`, `active_global_run.log`, `active_global_gap_report.json`, `active_global_gap_report.md`
- `per_sample_summary.json`, `per_sample_run.log`, `per_sample_gap_report.json`, `per_sample_gap_report.md`
