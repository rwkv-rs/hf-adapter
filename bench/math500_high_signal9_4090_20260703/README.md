# 4090 MATH500 high-signal-9 rollout64 subset — 2026-07-03

This subset reruns the nine high-signal tasks identified by the full generation gap report, starting from a fresh RNG stream and using the same rollout64 / bsz64 / fake_think / top-k/top-p policy.

Original full-run task IDs, in subset order: `73, 160, 116, 67, 277, 374, 383, 319, 72`.

## Summary

| Metric | HF adapter dynamic | Albatross | Delta / ratio |
|---|---:|---:|---:|
| Correct generations | 315/576 | 325/576 | -10 |
| Rollout accuracy | 0.54687500 | 0.56423611 | -0.01736111 |
| Pass@64 | 0.888889 | 0.888889 | +0.000000 |
| Summary token/s | 6241.051 | 3187.349 | 1.958x |

## Interpretation

- On this fresh high-signal subset, pass@64 is tied: `8/9` vs `8/9`.
- The correct-generation gap shrinks to `-10/576`, much smaller than the full-run net `-249/32000`.
- This supports the logits-parity finding: the large full-run task-level differences are likely driven by stochastic sampling / RNG stream / dynamic refill history, not a large model-math mismatch on these prompts.
- Subset throughput is not the final speed metric because only 9 tasks were run; use the full PR #104 benchmark for final speed.

## Artifacts

- `hf_summary.json`, `hf_run.log`
- `albatross_summary.json`, `albatross_run.log`
- `comparison.json`, `comparison.txt`
- `gap_report.json`, `gap_report.md`
