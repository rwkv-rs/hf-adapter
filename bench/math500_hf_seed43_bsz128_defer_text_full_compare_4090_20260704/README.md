# MATH500 seed43 bsz128 deferred-text HF vs Albatross comparison on RTX 4090

This artifact records the full HF dynamic MATH500 avg@64 rerun with seed `43`, `bsz=128`, deferred verification, generation-timed throughput, and deferred text decode.  It is compared to the committed full Albatross reference from `2026-07-03`.

## Inputs

- HF summary: `/tmp/math500_hf_dynamic_full_avg64_seed43_bsz128_defer_text_20260704/summary.json`
- HF log: `/tmp/math500_hf_dynamic_full_avg64_seed43_bsz128_defer_text_20260704.log`
- HF generations: `/tmp/math500_hf_dynamic_full_avg64_seed43_bsz128_defer_text_20260704/generations.jsonl` (not committed)
- Albatross summary: `/tmp/albatross_math500_full_avg64_20260703/summary.json`
- Albatross log: `/tmp/albatross_math500_full_avg64_20260703.log`
- Albatross generations: `/tmp/albatross_math500_full_avg64_20260703/generations.jsonl` (not committed)

## Result

| Metric | HF seed43 bsz128 deferred-text | Albatross full reference | Delta / ratio |
|---|---:|---:|---:|
| Correct generations | `4489/32000` | `4670/32000` | `-181` |
| Rollout accuracy | `0.14028125` | `0.14593750` | `-0.00565625` |
| Pass@64 | `0.380000` | `0.370000` | `+0.010000` |
| Summary token/s | `10426.943` | `3903.633` | `2.671x` |
| Wall token/s | `10053.618` | `3903.633` | `2.575x` |
| Decode token/s | `11588.182` | `3970.135` | `2.919x` |

## Acceptance status

- Accuracy primary gate (`pass@64 >= 0.370`): **passed**. HF reached `0.380000`, `+0.010000` vs Albatross.
- Correct-generation count: **not matched**, but improved acceptance is pass@64-based. HF is `181` correct generations below Albatross (`4489` vs `4670`).
- Speed gate (`>= 2x` throughput vs committed Albatross reference): **passed**. Generation-timed summary throughput is `2.671x`; wall throughput including deferred verification is `2.575x`; steady decode throughput is `2.919x`.

## HF speed details

- `speed_timing`: `generation`
- `prefill_sec`: `189.814`
- `decode_sec`: `1704.369`
- `generation_elapsed_sec`: `1894.183`
- `verification_sec`: `69.695`
- `decoded_token_events`: `19750537`
- `dynamic_bsz`: `128`

## Gap-analysis highlights

- Rows compared: `32000/32000`.
- Prompt-token diff rate: `0.000000`.
- Completion diff rows: `26353`.
- Correctness disagreement rows: `2263`.
- HF-only correct generations: `1041`.
- Albatross-only correct generations: `1222`.
- HF-only pass tasks: `17`.
- Albatross-only pass tasks: `12`.

## Conclusion

This run satisfies the current G1 acceptance gates against the committed full Albatross reference: `pass@64 >= 0.370` and `>=2x` throughput.  The final report should still keep the tuned-Albatross caveat from the v3a/v4 smoke: v4 is a higher prefill-speed ceiling on RTX 4090, but no full avg@64 v4 accuracy/speed reference is available in the committed artifacts.
