# MATH500 accuracy parity exploration

Branch: `wangyue/math500-accuracy-parity`

Base: `wangyue/math500-acceptance-clean` / PR #104.

## Goal

Close the current full MATH500 avg@64 accuracy gap against BlinkDL/Albatross while preserving the HF adapter dynamic-speed advantage.

Current acceptance baseline from `bench/math500_acceptance_4090_20260703`:

| Metric | HF adapter dynamic | Albatross | Gap |
|---|---:|---:|---:|
| Pass@64 | `0.358` | `0.370` | `-0.012` |
| Correct generations | `4421/32000` | `4670/32000` | `-249` |
| Summary token/s | `9161.229` | `3903.633` | `2.347x` |

## Execution goal G1: accuracy parity with tuned Albatross reference

**Objective.** Turn the current benchmark result into a passable final acceptance result by closing the MATH500 avg@64 gap while keeping the HF dynamic route speed-competitive against a tuned Albatross reference.

### Done criteria

1. **Reference tuning is explicit**
   - Run or document an Albatross `v3a` vs `v4` reference check on the target GPU.
   - Sweep / record `linear_orig_layout_launch` choices per GPU instead of assuming a fixed value.
   - Store the winning reference config and command beside the benchmark artifact.

2. **Accuracy gap is closed**
   - Primary gate: full MATH500 avg@64 `pass@64 >= 0.370` under the PR #104 benchmark shape.
   - Stretch gate: match or exceed the best tuned Albatross reference accuracy if the tuned reference changes the acceptance number.
   - Track both pass@64 and correct generations; current gaps are `-0.012` pass@64 and `-249/32000` generations.

3. **Speed advantage is preserved**
   - Keep `>= 2x` throughput vs the committed `v3a` PR #104 reference while iterating.
   - Final claim must also compare against the best-tuned per-GPU Albatross reference (`v3a`/`v4` + tuned `linear_orig_layout_launch`).

4. **Root cause is identified before broad changes**
   - Produce a parity report showing whether the gap starts at prefill logits, teacher-forced decode logits, recurrent state update, sampler/refill order, or stop/verification handling.
   - Any code fix should include the smallest targeted subset run before another full 500-task run.

### Immediate work order

1. Add a lightweight gap-analysis script for HF/Albatross `generations.jsonl` artifacts. **Done:**
   `bench/analyze_math500_gap.py` and `bench/math500_gap_4090_20260703/{README.md,gap_report.json}`.
   Result: prompt token counts match on all `32000` rows and verifier errors are empty on both sides, but completions differ on
   `26265/32000` rows, so the next probe should compare logits/state parity rather than prompt or verifier plumbing.
2. Add a logits-parity probe for selected tasks and fixed token continuations. **Done:**
   `bench/compare_albatross_logits.py` and `bench/math500_logits_parity_4090_20260703/{README.md,logits_parity_report.json}`.
   Result on high-signal tasks: prompt and continuation token IDs match; prefill argmax matches on all tasks;
   teacher-forced dynamic-path argmax match rate is `1.0`; cosine is effectively `~1.0`.  This shifts the
   likely root cause away from model math/prefill/state update and toward sampler RNG / dynamic refill order /
   stochastic variance under near-parity logits.
3. Run targeted rollout64 subsets on the high-signal tasks listed below. **Done:**
   `bench/math500_high_signal9_4090_20260703/`.  On the fresh 9-task subset, pass@64 ties at
   `8/9` vs `8/9`, while correct generations are `315/576` HF vs `325/576` Albatross.  The
   full-run net gap `-249/32000` shrinks to `-10/576`, supporting the logits-parity conclusion
   that sampling/RNG/refill history is more likely than a large model-math mismatch.
4. Run Albatross `v4` / `linear_orig_layout_launch` tuning checks and update the reference table. **Partial done:**
   `bench/albatross_v3a_v4_4090_tune_20260703/` shows v4 is faster than v3a on this RTX 4090
   smoke (`B1T512` is `58,933.8` tok/s vs `48,311.5`, `1.220x`).  The remaining reference-tuning
   task is an Albatross-side micro sweep of `linear_orig_layout_launch` choices per `(GPU, C, rows, group)`.
5. Add a sampler/refill stochasticity report before changing model math. **Done:**
   `bench/math500_sampling_variance_4090_20260703/` shows the prefix curve starts near parity (`pass@1` HF
   `0.144` vs Albatross `0.142`) and the empirical repeated-rollout bootstrap delta interval includes zero
   (`p2.5/p50/p97.5 = -14/-7/+1` pass tasks).
6. Probe deterministic/active RNG refill variants on high-signal subset. **Done:**
   `bench/math500_rng_modes_high_signal9_4090_20260704/` shows default full-batch global RNG remains
   strongest among tested HF modes (`315/576` correct, `8/9` pass@64); `active_global` drops to
   `297/576` and `per_sample` to `302/576`, so do not switch final acceptance away from
   Albatross-compatible global sampling yet.
7. Only after a targeted variant beats the default or the reference tuning is complete, run full MATH500 avg@64 again.

## Working hypothesis

The speed path is already strong.  The next work should focus on numerical / sampling parity:

1. Compare HF vs Albatross prefill logits for identical prompts.
2. Compare teacher-forced decode logits over fixed tokens.
3. Compare native prefill + fast-token vs forward prefill + forward decode.
4. Isolate whether the gap comes from native prefill, recurrent state update, logits dtype/cast, sampler RNG/refill order, or verifier/stop handling.

## First gap report

Artifact: `bench/math500_gap_4090_20260703/README.md` and `gap_report.json`.

Key findings from the full HF/Albatross generation diff:

- Rows match: `32000/32000`; tasks match: `500/500`.
- Prompt token diffs: `0/32000`, so the current gap is not explained by prompt length/BOS truncation differences.
- Verify errors: empty on both sides for all rows, so the current gap is not explained by verifier exceptions.
- Completion divergence: `26265/32000` rows (`82.08%`).
- Correctness disagreement rows: `2395/32000` (`7.48%`).
- HF-only correct generations: `1073`; Albatross-only correct generations: `1322`; net `-249`.
- HF-only pass tasks: `17`; Albatross-only pass tasks: `23`; net pass@64 gap `-6` tasks.

Conclusion: continue with logits/state parity probes.

## First logits parity report

Artifact: `bench/math500_logits_parity_4090_20260703/README.md` and `logits_parity_report.json`.

Probe setup:

- Tasks: `73,160,116,67,277,374,383,319,72`.
- Continuation source: Albatross full-run `generations.jsonl`, `sample_id=0`.
- Teacher-forced steps: first `64` continuation tokens.
- Implementations: HF adapter vs Albatross `rwkv7_fast_v3a` / `fp32io16`.

Key findings:

- Prompt ID mismatches: `0`; continuation ID mismatches: `0`.
- Prefill forward vs Albatross: argmax match rate `1.0`, cosine mean `0.99999977`, mean abs diff `0.02456`, max abs diff `0.15625`.
- Prefill native vs Albatross: argmax match rate `1.0`, cosine mean `0.99999974`, mean abs diff `0.02310`, max abs diff `0.21875`.
- Teacher-forced all-logits: argmax match rate mean `0.99826`, cosine mean `0.99999998`.
- Teacher-forced dynamic path: argmax match rate mean `1.0`, cosine mean `0.99999997`, max abs max `0.4375`.

Conclusion: the high-signal gap is not explained by a large prefill/decode logits or state-update mismatch.  Next work should focus on sampler RNG, dynamic refill ordering, seed sensitivity, and targeted rollout subset reproducibility before changing model math.

## Initial high-signal tasks

From the full-run diff, prioritize small rollout64 subsets before another full 500-task run:

- Albatross advantage: `73`, `160`, `116`, `67`, `277`.
- HF advantage: `374`, `383`, `319`, `72`.

## Targeted high-signal subset rerun

Artifact: `bench/math500_high_signal9_4090_20260703/`.

Subset task IDs: `73,160,116,67,277,374,383,319,72`.

| Metric | HF adapter dynamic | Albatross | Delta / ratio |
|---|---:|---:|---:|
| Correct generations | `315/576` | `325/576` | `-10` |
| Rollout accuracy | `0.54687500` | `0.56423611` | `-0.01736111` |
| Pass@64 | `0.888889` | `0.888889` | `0` |
| Summary token/s | `6241.051` | `3187.349` | `1.958x` |

Interpretation: when the high-signal tasks are rerun from a fresh RNG stream, the pass@64 gap disappears on this subset.  The remaining correct-generation delta is small relative to the full-run gap.  Combined with logits parity, this points to sampler RNG / dynamic refill order / seed sensitivity as the next investigation target.



## Sampling / refill stochasticity report

Artifact: `bench/math500_sampling_variance_4090_20260703/`.

| k | HF pass@k | Albatross pass@k | HF - Albatross |
|---:|---:|---:|---:|
| `1` | `0.144000` | `0.142000` | `+0.002000` |
| `2` | `0.190000` | `0.182000` | `+0.008000` |
| `4` | `0.218000` | `0.214000` | `+0.004000` |
| `8` | `0.246000` | `0.248000` | `-0.002000` |
| `16` | `0.274000` | `0.298000` | `-0.024000` |
| `32` | `0.316000` | `0.334000` | `-0.018000` |
| `64` | `0.358000` | `0.370000` | `-0.012000` |

Empirical repeated-rollout bootstrap from observed per-task correct rates (`20,000` draws, seed `7`): expected pass-task delta is `-6.722`; delta quantiles are `p2.5=-14`, `p50=-7`, `p97.5=+1`; `P(delta >= 0)=0.0546`.  This does not prove the full result is accepted, but together with logits parity and the high-signal rerun it supports treating the remaining pass@64 gap as sampling/refill-order sensitive before changing recurrence math.

Follow-up sampler/refill task was run in the RNG/refill mode probe below. Since the variants did not beat the default, keep the default global Albatross-compatible RNG for acceptance unless a broader seed/refill sensitivity sweep finds a stronger targeted variant.


## RNG/refill mode probe

Artifact: `bench/math500_rng_modes_high_signal9_4090_20260704/`.

The HF dynamic evaluator now has opt-in RNG modes for probing only; default remains `global`, matching the Albatross full-batch sampling behavior.

| Runner / RNG mode | Correct generations | Rollout accuracy | Pass@64 | Summary token/s |
|---|---:|---:|---:|---:|
| HF `global` full-batch RNG | `315/576` | `0.54687500` | `0.888889` | `6241.051` |
| HF `active_global` active-row RNG | `297/576` | `0.51562500` | `0.888889` | `6188.883` |
| HF `per_sample` deterministic RNG | `302/576` | `0.52430556` | `0.888889` | `6080.989` |
| Albatross v3a | `325/576` | `0.56423611` | `0.888889` | `3187.349` |

Conclusion: deterministic per-row/per-sample RNG and active-row-only global RNG do not improve this high-signal subset.  The final acceptance path should keep the default Albatross-compatible global full-batch RNG unless a later targeted variant beats it.  The next useful probe is seed/refill sensitivity over a broader stratified subset or a full avg@64 rerun with the current default once reference tuning is finalized.

## Albatross v3a/v4 reference smoke on RTX 4090

Artifact: `bench/albatross_v3a_v4_4090_tune_20260703/`.

| Case | v3a tok/s | v4 tok/s | v4/v3a |
|---|---:|---:|---:|
| `B1T1` | `837.53` | `855.73` | `1.022x` |
| `B1T512` | `48311.51` | `58933.80` | `1.220x` |
| `B64T1` | `25130.68` | `25183.30` | `1.002x` |
| `B4T128` | `81847.50` | `89226.80` | `1.090x` |
| `B8T64` | `94940.28` | `96756.70` | `1.019x` |

Interpretation: v4 is a higher speed ceiling than the committed v3a reference on this RTX 4090 smoke, especially for prompt-prefill.  The committed PR #104 full MATH500 v3a run remains the accuracy reference until v4 has a full avg@64 runner/result.  For final speed claims, report both the committed full-eval v3a reference and the fastest tuned per-GPU Albatross reference.

`linear_orig_layout_launch` status: checked but not fully tuned.  v4 hard-codes launch choices by `rows`, `K`, and group.  For 0.4B `C=1024`, `B1T512` uses `rows=512` in the body and `head_rows=1`; the next Albatross-reference task is a micro sweep over the cublasLt algorithm/workspace and exact-row kernel choices per `(GPU, C, rows, group)`.

## Albatross reference tuning notes

Do not treat the current Albatross `v3a` reference as the only possible speed ceiling:

- Some Albatross `v4` configurations may be faster than `v3a` on the same model / MATH500 shape.
- `linear_orig_layout_launch` is GPU-sensitive and should be tuned per GPU instead of hard-coded globally.
- The benchmark harness should therefore record the exact Albatross backend/config used for the reference (`v3a` vs `v4`, layout launch policy, GPU name/SM, CUDA/Torch versions).
- For acceptance comparisons, keep two numbers when available:
  1. **fixed-current reference**: the committed `v3a` run used by PR #104;
  2. **best-tuned reference**: the fastest valid Albatross config for that GPU after `linear_orig_layout_launch` tuning.
- A HF speed win should eventually be claimed against the **best-tuned per-GPU Albatross reference**, not only against the first `v3a` baseline.

## Acceptance gate

A parity fix should satisfy:

- `pass@64 >= 0.370` on the full MATH500 benchmark, or statistically clear evidence on targeted subsets before full rerun.
- HF dynamic throughput remains `>= 2x` Albatross on the same 4090 acceptance benchmark.
