# Temporary TODO: DPLR/WY Compiled Prefill

This is a short-lived working TODO for the current `wangyue/native-fused-fp16-kernel` branch. Keep the default HF path unchanged unless a benchmark explicitly opts in.

## Current checkpoint

- Branch: `wangyue/native-fused-fp16-kernel`
- Dense3 scaffold checkpoint commit: `a8f76a6 Add dense three-stage DPLR Triton scaffold`
- Current state:
  - `triton_wy`: fast P0 bridge through existing fused recurrent scan.
  - `triton_dense3`: explicit dense three-stage scaffold: summary -> prefix -> apply/output.
  - 4090 synthetic target passes correctness and is faster than sequential.
  - Dense3 is correctness-first and still slower than `triton_wy` because it materializes dense `[N,N]` summaries.

## P0 next actions

- [x] Push current branch so the dense3 scaffold is backed up.
  - Done: pushed `wangyue/native-fused-fp16-kernel` through `a8f76a6` to `origin`.
- [x] Add/extend benchmark rows that separately time:
  - [x] dense chunk summary
  - [x] dense prefix combine
  - [x] dense chunk apply/output
  - [x] full dense3 end-to-end
  - Done: `bench/bench_dplr_prefill_scan.py --stage-probe` emits `axis="dplr_dense3_stage_proto"`; analyzer prints the new section.
- [x] Run the staged timing split on RTX 4090 target shape:
  - `B=1,T=512,H=16,N=64,chunk_size=64,fp16`
  - Done: `/tmp/dplr_dense3_stage_probe.jsonl`, warmup=1, steps=5.
- [x] Identify whether the main dense3 bottleneck is summary, prefix, apply, or memory traffic.
  - Current split: summary `~0.144 ms`, prefix `~0.092 ms`, apply/output `~0.065 ms`, full dense3 `~0.264-0.269 ms`. Dense summary/prefix `[N,N]` traffic is the first compact-WY target.

## P1 compact WY path

- [x] Add torch reference for compact chunk summary factors instead of dense `[N,N]` summaries.
  - [x] transition diag / low-rank factors
  - [x] additive low-rank factors
  - [x] final-state reconstruction oracle for correctness
  - Done: `dplr_compact_wy_chunk_summary_torch`, `dplr_compact_wy_summary_to_dense`, and `dplr_compact_wy_apply_summaries_torch` added. 4090 target oracle: transition diff `~4.6e-14`, additive diff `~5.96e-08`, final state diff `~1.13e-04`.
- [x] Add Triton compact summary kernel for the target shape first.
  - Done: `dplr_compact_wy_chunk_summary_triton` and availability helper added. First kernel is target-constrained to `N<=64, chunk_size<=64`; 4090 target `B=1,T=512,H=16,N=64,chunk=64,fp16` matches torch compact factors with max factor diff `<=5.96e-08`, final state diff `~1.13e-04`, and summary time `~0.155 ms`.
- [x] Add compact prefix combine using factors instead of materialized transition/additive matrices.
  - Done: `dplr_compact_wy_prefix_combine_torch`, `dplr_compact_wy_prefix_combine_triton`, and availability helper added. 4090 target prefix combine: starts diff vs dense `~5.96e-08`, final state diff vs ref `~1.13e-04`, time `~0.067 ms`.
- [x] Reuse current chunk apply/output kernel initially, then fuse/optimize only after correctness is stable.
  - Done: `dplr_compact_wy_three_stage_triton` now runs compact summary -> compact prefix -> existing chunk apply/output. 4090 target correctness: `out_min_cosine~=0.9999999`, state diff `~1.13e-04`; current full compact path time `~0.501 ms`, so it is correctness-stable but still needs algorithm/benchmark exposure and later fusion/optimization.
- [x] Add benchmark algorithm name for the compact path, e.g. `triton_wy_compact` or replace internal `triton_dense3` route once it is clearly better.
  - Done: `RWKV7_DPLR_PREFILL_ALGORITHM=triton_wy_compact` and `bench_dplr_prefill_scan.py --algorithms triton_wy_compact` now route to `dplr_compact_wy_three_stage_triton`. 4090 target benchmark: `~0.241 ms`, `~2.12M tok/s`, `out_min_cosine=1.0`; HF repo-code smoke 0.4B/prompt512/bsz1 passes greedy/cache at `~17.5k tok/s`.

## Correctness gates

- [x] Local no-CUDA checks:
  - `python -m py_compile rwkv7_hf/dplr_prefill_triton.py rwkv7_hf/dplr_prefill.py bench/bench_dplr_prefill_scan.py tests/test_dplr_prefill_triton.py tests/test_dplr_prefill_scan.py`
  - `git diff --check`
- [x] 4090 unit tests:
  - `PYTHONPATH=. python tests/test_dplr_prefill_scan.py`
  - `PYTHONPATH=. python tests/test_dplr_prefill_triton.py`
- [x] 4090 synthetic fp16 target:
  - `out_min_cosine >= 0.9999`
  - state diff comparable to current `triton_wy` / dense3 rows
  - Latest dense3 stage-probe full row: `out_min_cosine=1.0`, `state_max_abs_diff=0.0001257062`.
- [x] HF repo-code smoke:
  - `RWKV7_NATIVE_PREFILL_DPLR_SCAN=1`
  - `RWKV7_DPLR_PREFILL_ALGORITHM=<candidate>`
  - 0.4B / prompt512 / bsz1
  - greedy/cache smoke must pass
  - Done for current scaffold: `triton_wy`, `triton_dense3`, and `triton_wy_compact` passed 4090 / 0.4B / prompt512 / bsz1 smoke.

## Performance targets

- Baseline evidence from latest 4090 synthetic target:
  - `sequential`: about `55.63 ms`, `9.2k tok/s` in the earlier mixed run
  - `triton_wy`: about `0.233 ms`, `2.20M tok/s`
  - `triton_dense3`: latest stage-probe full row about `0.264-0.269 ms`, `~1.9M tok/s`
- Short-term compact target:
  - [x] compact path `< 0.4 ms` on synthetic target
    - Done: `triton_wy_compact` benchmark row is `~0.241 ms`.
  - [x] then approach or beat current `triton_wy` P0 `~0.233 ms`
    - Done/approached: same run `triton_wy_compact ~0.241 ms` vs `triton_wy ~0.228 ms`; close but not yet faster, so later fusion remains useful.
- HF target:
  - [ ] 4090 / 0.4B / prompt512 / bsz1 moves toward `>=0.45x` Albatross
  - [ ] stretch: `>=0.60x` Albatross

## Guardrails

- Do not default-enable dense3 in the HF path.
- Do not claim Albatross-level performance from dense3 alone.
- Do not start vLLM/SGLang work in this repository.
- Do not optimize Python loops instead of compiled kernel/factor work.
