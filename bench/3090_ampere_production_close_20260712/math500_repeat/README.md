# MATH500 final acceptance benchmark

This artifact follows the BlinkDL/Albatross MATH500 avg@64 evaluation shape and adds best-bsz speed selection plus uncheatable teacher-forced compression/logit alignment.

## Best-bsz sweep

| rank | requested bsz | effective bsz | generation tok/s | decode sec | decoded tokens | pass@rollout |
|---:|---:|---:|---:|---:|---:|---:|

Selected bsz: `128`.

## Full avg@64 summary

| metric | value |
|---|---:|
| `num_tasks` | `500` |
| `rollout` | `64` |
| `total_generations` | `32000` |
| `correct_generations` | `4582` |
| `rollout_accuracy` | `0.1431875` |
| `pass_at_rollout_accuracy` | `0.368` |
| `generation_token_per_sec` | `8770.457674831505` |
| `wall_token_per_sec` | `8441.39948748651` |
| `decode_sec` | `1953.8933468349987` |
| `decoded_token_events` | `19517397` |

## HF vs Albatross comparison

Overall gate: `FAIL`.

## Uncheatable compression alignment

| metric | value |
|---|---:|
| reference bits/token | `1.92407828` |
| candidate bits/token | `1.92407828` |
| candidate/reference bits ratio | `1.00000000` |
| tokens scored | `43865` |

See `compression_alignment/compression_alignment.md` for ratio vs token position.

## Files

- `manifest.json`: top-level machine-readable manifest.
- `bsz_sweep_summary.json`: sorted best-bsz speed rows.
- `full_avg64/summary.json`: full MATH500 result when enabled.
- `comparison/comparison.json`: HF-vs-Albatross gates when an Albatross summary is provided.
- `compression_alignment/compression_alignment.json`: external-token compression/NLL report.
