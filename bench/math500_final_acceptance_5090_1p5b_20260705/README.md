# MATH500 final acceptance benchmark

This artifact follows the BlinkDL/Albatross MATH500 avg@64 evaluation shape and adds best-bsz speed selection plus uncheatable teacher-forced compression/logit alignment.

## Best-bsz sweep

| rank | requested bsz | effective bsz | generation tok/s | decode sec | decoded tokens | pass@rollout |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `128` | `128` | `4855.721` | `9.103` | `65517` | `0.250000` |
| `2` | `96` | `96` | `4412.250` | `10.443` | `65517` | `0.250000` |
| `3` | `64` | `64` | `3970.580` | `12.018` | `65517` | `0.250000` |
| `4` | `192` | `192` | `3731.191` | `13.145` | `65517` | `0.250000` |
| `5` | `32` | `32` | `2463.453` | `22.138` | `65513` | `0.250000` |

Selected bsz: `128`.

## Full avg@64 summary

| metric | value |
|---|---:|
| `num_tasks` | `500` |
| `rollout` | `64` |
| `total_generations` | `32000` |
| `correct_generations` | `12756` |
| `rollout_accuracy` | `0.398625` |
| `pass_at_rollout_accuracy` | `0.662` |
| `generation_token_per_sec` | `5918.905856165877` |
| `wall_token_per_sec` | `5854.032836587377` |
| `decode_sec` | `2494.73530734994` |
| `decoded_token_events` | `18486256` |

## HF vs Albatross comparison

Overall gate: `FAIL`.

## Uncheatable compression alignment

| metric | value |
|---|---:|
| reference bits/token | `1.76191866` |
| candidate bits/token | `1.76191866` |
| candidate/reference bits ratio | `1.00000000` |
| tokens scored | `43865` |

See `compression_alignment/compression_alignment.md` for ratio vs token position.

## Files

- `manifest.json`: top-level machine-readable manifest.
- `bsz_sweep_summary.json`: sorted best-bsz speed rows.
- `full_avg64/summary.json`: full MATH500 result when enabled.
- `comparison/comparison.json`: HF-vs-Albatross gates when an Albatross summary is provided.
- `compression_alignment/compression_alignment.json`: external-token compression/NLL report.
