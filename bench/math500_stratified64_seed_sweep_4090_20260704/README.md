# 4090 MATH500 stratified-64 HF seed sweep — 2026-07-04

Purpose: broader seed/refill-sensitivity probe after high-signal-9 RNG variants did not beat the default. This is not the final full MATH500 avg@64 gate.

## Subset construction

- Selected tasks: `64`
- Buckets: `{'albatross_only_pass': 16, 'both_pass_albatross_adv': 16, 'both_pass_hf_adv': 16, 'hf_only_pass': 16}`
- Source: full HF and Albatross generation artifacts from PR #104 / 2026-07-03.
- Completed fresh HF seeds: `[42, 43]`

## Source full-run reference restricted to selected tasks

| Reference | Correct generations | Pass tasks | Pass@64 |
|---|---:|---:|---:|
| HF full run selected rows | 893/4096 | 48/64 | 0.750000 |
| Albatross full run selected rows | 1062/4096 | 48/64 | 0.750000 |

## Fresh HF global RNG seed sweep on this subset

| HF seed | Correct generations | Rollout accuracy | Pass tasks | Pass@64 | Summary token/s |
|---:|---:|---:|---:|---:|---:|
| 42 | 938/4096 | 0.22900391 | 46/64 | 0.718750 | 5926.511 |
| 43 | 981/4096 | 0.23950195 | 46/64 | 0.718750 | 5812.783 |

## Interpretation

- Best fresh HF seed by pass@64 then correct generations: seed `43` with `46/64` pass tasks and `981/4096` correct generations.
- The disagreement-enriched subset has selected-task pass parity in the source full-run reference (`48/64` vs `48/64`), despite Albatross having more correct generations.
- Fresh HF seeds `42` and `43` both reach `46/64`; seed `43` improves correct generations (`981/4096`) over seed `42` (`938/4096`) but does not close selected-task pass parity.
- This supports seed sensitivity but does not justify changing the final acceptance seed/path yet; full avg@64 remains required for completion.
