# MATH500 seed43 full HF vs Albatross comparison on RTX 4090

This artifact records the full HF dynamic MATH500 avg@64 rerun with seed `43`, compared to the committed full Albatross reference from `2026-07-03`.

## Inputs

- HF summary: `/tmp/math500_hf_dynamic_full_avg64_seed43_20260704/summary.json`
- HF log: `/tmp/math500_hf_dynamic_full_avg64_seed43_20260704.log`
- HF generations: `/tmp/math500_hf_dynamic_full_avg64_seed43_20260704/generations.jsonl` (not committed)
- Albatross summary: `/tmp/albatross_math500_full_avg64_20260703/summary.json`
- Albatross log: `/tmp/albatross_math500_full_avg64_20260703.log`
- Albatross generations: `/tmp/albatross_math500_full_avg64_20260703/generations.jsonl` (not committed)

## Result

| Metric | HF seed43 dynamic | Albatross full reference | Delta / ratio |
|---|---:|---:|---:|
| Correct generations | `4600/32000` | `4670/32000` | `-70` |
| Rollout accuracy | `0.14375000` | `0.14593750` | `-0.00218750` |
| Pass@64 | `0.372000` | `0.370000` | `+0.002000` |
| Summary token/s | `6275.770` | `3903.633` | `1.608x` |
| Decode token/s | `6693.283` | `3970.135` | `1.686x` |

## Acceptance status

- Accuracy primary gate (`pass@64 >= 0.370`): **passed**. HF seed43 reached `0.372000`, which is `+0.002000` vs the Albatross full reference.
- Correct-generation count: **not fully matched**. HF seed43 is `70` correct generations below Albatross (`4600` vs `4670`).
- Speed gate (`>= 2x` throughput vs committed Albatross reference): **not passed in this run**. Summary throughput is `1.608x`, and decode-token throughput is `1.686x`. This is lower than the earlier PR #104 HF full run (`9161.229` token/s, about `2.347x` vs Albatross).

## Gap-analysis highlights

- Rows compared: `32000/32000`.
- Prompt-token diff rate: `0.000000`.
- Completion diff rows: `26475`.
- Correctness disagreement rows: `2318`.
- HF-only correct generations: `1124`.
- Albatross-only correct generations: `1194`.
- HF-only pass tasks: `15`.
- Albatross-only pass tasks: `14`.

## Conclusion

Seed `43` is the first full HF dynamic avg@64 rerun in this branch that clears the primary pass@64 acceptance threshold against the committed Albatross reference. G1 should remain open because the same run does not preserve the `>=2x` speed gate; the next work should restore fair generation-speed accounting, most likely by deferring expensive verification out of the decode loop or by running the previous fast benchmark path with the accepted seed/settings.
