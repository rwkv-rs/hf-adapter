# 4090 MATH500 stratified-64 parity subset — 2026-07-04

## Selection summary

- Selected tasks: `64`
- Rollout per task in source runs: `64`
- Buckets: `{'albatross_only_pass': 16, 'both_pass_albatross_adv': 16, 'both_pass_hf_adv': 16, 'hf_only_pass': 16}`

## Full-run reference restricted to selected tasks

| Metric | HF full run | Albatross full run | HF - Albatross |
|---|---:|---:|---:|
| Correct generations | 893 | 1062 | -169 |
| Pass tasks | 48 | 48 | 0 |
| Pass@64 | 0.750000 | 0.750000 | +0.000000 |

## Selected original task IDs

- `albatross_only_pass`: `73, 160, 67, 24, 444, 104, 123, 229, 335, 499, 21, 100, 114, 130, 151, 194`
- `both_pass_albatross_adv`: `116, 277, 417, 440, 414, 296, 255, 146, 487, 150, 472, 176, 85, 37, 132, 373`
- `both_pass_hf_adv`: `383, 319, 72, 212, 350, 102, 111, 280, 65, 438, 28, 91, 327, 276, 120, 449`
- `hf_only_pass`: `374, 213, 81, 226, 295, 353, 6, 80, 94, 244, 262, 304, 371, 409, 421, 473`

## Artifacts

- `dataset.jsonl`: subset dataset for fresh reruns
- `subset_tasks.json`: local->original mapping and source full-run counts
- Large remapped reference generation files were generated on the 4090 host for gap reports but are intentionally not committed here; regenerate them with `bench/make_math500_stratified_subset.py` when needed.
