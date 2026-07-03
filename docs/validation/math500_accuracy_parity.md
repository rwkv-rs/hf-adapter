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
2. Add a logits-parity probe for selected tasks and fixed token continuations.
3. Run targeted rollout64 subsets on the high-signal tasks listed below.
4. Run Albatross `v4` / `linear_orig_layout_launch` tuning checks and update the reference table.
5. Only after the above, run full MATH500 avg@64 again.

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

## Initial high-signal tasks

From the full-run diff, prioritize small rollout64 subsets before another full 500-task run:

- Albatross advantage: `73`, `160`, `116`, `67`, `277`.
- HF advantage: `374`, `383`, `319`, `72`.

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
