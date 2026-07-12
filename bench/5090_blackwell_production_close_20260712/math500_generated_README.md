# MATH500 final acceptance benchmark

This artifact follows the BlinkDL/Albatross MATH500 avg@64 evaluation shape and adds best-bsz speed selection plus uncheatable teacher-forced compression/logit alignment.

## Best-bsz sweep

| rank | requested bsz | effective bsz | generation tok/s | decode sec | decoded tokens | pass@rollout |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `128` | `128` | `9026.824` | `4.786` | `57457` | `0.500000` |
| `2` | `192` | `192` | `8618.606` | `5.053` | `57133` | `0.500000` |
| `3` | `96` | `96` | `8331.163` | `5.454` | `57987` | `0.500000` |
| `4` | `64` | `64` | `7381.335` | `6.348` | `57799` | `0.500000` |

Selected bsz: `128`.

## Full avg@64 summary

| metric | value |
|---|---:|
| `num_tasks` | `500` |
| `rollout` | `64` |
| `total_generations` | `32000` |
| `correct_generations` | `4559` |
| `rollout_accuracy` | `0.14246875` |
| `pass_at_rollout_accuracy` | `0.38` |
| `generation_token_per_sec` | `16925.605451653635` |
| `wall_token_per_sec` | `16118.14358625907` |
| `decode_sec` | `1014.438516525086` |
| `decoded_token_events` | `19618782` |

## HF vs Albatross comparison

Overall gate: `PASS`.

## Uncheatable compression alignment

| metric | value |
|---|---:|
| reference bits/token | `1.92410150` |
| candidate bits/token | `1.92410150` |
| candidate/reference bits ratio | `1.00000000` |
| tokens scored | `43865` |

See `compression_alignment/compression_alignment.md` for ratio vs token position.

## Files

- `manifest.json`: top-level machine-readable manifest.
- `bsz_sweep_summary.json`: sorted best-bsz speed rows.
- `full_avg64/summary.json`: full MATH500 result when enabled.
- `comparison/comparison.json`: HF-vs-Albatross gates when an Albatross summary is provided.
- `compression_alignment/compression_alignment.json`: external-token compression/NLL report.
