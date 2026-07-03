# MATH500 bsz sweep with deferred verification + text decode on RTX 4090
This short sweep estimates the best dynamic batch size before launching the full seed43 avg@64 speed-gate run.

## Command shape

- `--limit 4 --rollout 64 --max-new-tokens 256`
- `--seed 43 --rng-mode global`
- `--prefill-backend native --decode-backend fast_token`
- `--defer-verification --verify-workers 4 --summary-speed-timing generation --defer-text-decode`

## Results

| bsz | generation token/s | decode sec | decoded tokens | correct | pass@64 |
|---:|---:|---:|---:|---:|---:|
| `32` | `3459.559` | `14.737` | `56517` | `48` | `0.750000` |
| `64` | `5391.690` | `9.130` | `57799` | `34` | `0.500000` |
| `96` | `6099.680` | `7.895` | `57966` | `32` | `0.500000` |
| `128` | `7131.751` | `6.514` | `57535` | `36` | `0.500000` |
| `192` | `6302.463` | `7.481` | `57133` | `38` | `0.500000` |

## Conclusion

Best short-run speed was bsz `128` at `7131.751` generation token/s.  The full seed43 avg@64 run was therefore launched with `--bsz 128 --defer-verification --summary-speed-timing generation --defer-text-decode`.
