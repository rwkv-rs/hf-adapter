# MATH500 per_sample RNG vs Albatross high-signal gap

## Shape

- Rows: HF `576`, Albatross `576`, common `576`
- Tasks: HF `9`, Albatross `9`, common `9`

## Accuracy summary

| Metric | HF | Albatross | HF - Albatross |
|---|---:|---:|---:|
| Correct generations | 302 | 325 | -23 |
| Rollout accuracy | 0.52430556 | 0.56423611 | -0.03993056 |
| Pass@rollout | 0.888889 | 0.888889 | +0.000000 |
| Pass tasks | 8 | 8 | 0 |

## Row-level disagreement

- Completion differs: `399` / `576` (`69.2708%`)
- Token counts differ: `370` / `576` (`64.2361%`)
- Correctness disagreement rows: `199` / `576` (`34.5486%`)
- Prompt token diffs: `0` / `576` (`0.0000%`)
- Both correct: `214`; HF-only correct: `88`; Albatross-only correct: `111`; both wrong: `163`

## Pass-task deltas

- HF-only pass tasks (0): `[]`
- Albatross-only pass tasks (0): `[]`

## Top Albatross task advantages

| Task | Advantage | HF correct | Albatross correct | Problem |
|---:|---:|---:|---:|---|
| 6 | 12 | 31 | 43 | For what values of $x$ is it true that $x^2 - 5x - 4 \le 10$? Express your answer in interval not... |
| 4 | 8 | 18 | 26 | What is $\sqrt{53}$ in simplest radical form? |
| 3 | 5 | 15 | 20 | The lengths of two opposite sides of a square are decreased by $40\%$ while the lengths of the ot... |
| 8 | 5 | 48 | 53 | What is the length, in units, of the radius of a sphere whose volume and surface area, in cubic u... |
| 7 | -2 | 57 | 55 | In the equation $|x-4| -10 = 2$, what is the product of all possible values of $x$? |
| 1 | -5 | 31 | 26 | The product of two consecutive positive even integers is 288. What is the greater of the two inte... |

## Top HF task advantages

| Task | Advantage | HF correct | Albatross correct | Problem |
|---:|---:|---:|---:|---|
| 1 | 5 | 31 | 26 | The product of two consecutive positive even integers is 288. What is the greater of the two inte... |
| 7 | 2 | 57 | 55 | In the equation $|x-4| -10 = 2$, what is the product of all possible values of $x$? |
| 3 | -5 | 15 | 20 | The lengths of two opposite sides of a square are decreased by $40\%$ while the lengths of the ot... |
| 8 | -5 | 48 | 53 | What is the length, in units, of the radius of a sphere whose volume and surface area, in cubic u... |
| 4 | -8 | 18 | 26 | What is $\sqrt{53}$ in simplest radical form? |
| 6 | -12 | 31 | 43 | For what values of $x$ is it true that $x^2 - 5x - 4 \le 10$? Express your answer in interval not... |

## Stop reasons

- HF: `{'eod': 512, 'max_tokens': 64}`
- Albatross: `{'eod': 512, 'max_tokens': 64}`

## Verify errors

- HF: `{'': 576}`
- Albatross: `{'': 576}`

## Interpretation

- Prompt token counts match for all common rows; the gap is unlikely to be caused by prompt length/BOS truncation differences.
- Both runs report empty verifier errors for all rows; the gap is unlikely to be caused by verifier exceptions.
- Large completion divergence means the next probe should compare logits/state parity, not just final verifier outputs.
