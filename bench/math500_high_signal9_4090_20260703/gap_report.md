# 4090 MATH500 high-signal-9 rollout64 gap analysis

## Shape

- Rows: HF `576`, Albatross `576`, common `576`
- Tasks: HF `9`, Albatross `9`, common `9`

## Accuracy summary

| Metric | HF | Albatross | HF - Albatross |
|---|---:|---:|---:|
| Correct generations | 315 | 325 | -10 |
| Rollout accuracy | 0.54687500 | 0.56423611 | -0.01736111 |
| Pass@rollout | 0.888889 | 0.888889 | +0.000000 |
| Pass tasks | 8 | 8 | 0 |

## Row-level disagreement

- Completion differs: `328` / `576` (`56.9444%`)
- Token counts differ: `289` / `576` (`50.1736%`)
- Correctness disagreement rows: `140` / `576` (`24.3056%`)
- Prompt token diffs: `0` / `576` (`0.0000%`)
- Both correct: `250`; HF-only correct: `65`; Albatross-only correct: `75`; both wrong: `186`

## Pass-task deltas

- HF-only pass tasks (0): `[]`
- Albatross-only pass tasks (0): `[]`

## Top Albatross task advantages

| Task | Advantage | HF correct | Albatross correct | Problem |
|---:|---:|---:|---:|---|
| 4 | 8 | 18 | 26 | What is $\sqrt{53}$ in simplest radical form? |
| 7 | 4 | 51 | 55 | In the equation $|x-4| -10 = 2$, what is the product of all possible values of $x$? |
| 1 | 1 | 25 | 26 | The product of two consecutive positive even integers is 288. What is the greater of the two inte... |
| 6 | -1 | 44 | 43 | For what values of $x$ is it true that $x^2 - 5x - 4 \le 10$? Express your answer in interval not... |
| 2 | -2 | 40 | 38 | Solve for $x$: $2^{2x} = 256^\frac{1}{2}$. |

## Top HF task advantages

| Task | Advantage | HF correct | Albatross correct | Problem |
|---:|---:|---:|---:|---|
| 2 | 2 | 40 | 38 | Solve for $x$: $2^{2x} = 256^\frac{1}{2}$. |
| 6 | 1 | 44 | 43 | For what values of $x$ is it true that $x^2 - 5x - 4 \le 10$? Express your answer in interval not... |
| 1 | -1 | 25 | 26 | The product of two consecutive positive even integers is 288. What is the greater of the two inte... |
| 7 | -4 | 51 | 55 | In the equation $|x-4| -10 = 2$, what is the product of all possible values of $x$? |
| 4 | -8 | 18 | 26 | What is $\sqrt{53}$ in simplest radical form? |

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
