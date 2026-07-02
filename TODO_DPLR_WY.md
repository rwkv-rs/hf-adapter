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
- [ ] Add/extend benchmark rows that separately time:
  - [ ] dense chunk summary
  - [ ] dense prefix combine
  - [ ] dense chunk apply/output
  - [ ] full dense3 end-to-end
- [ ] Run the staged timing split on RTX 4090 target shape:
  - `B=1,T=512,H=16,N=64,chunk_size=64,fp16`
- [ ] Identify whether the main dense3 bottleneck is summary, prefix, apply, or memory traffic.

## P1 compact WY path

- [ ] Add torch reference for compact chunk summary factors instead of dense `[N,N]` summaries.
  - [ ] transition diag / low-rank factors
  - [ ] additive low-rank factors
  - [ ] final-state reconstruction oracle for correctness
- [ ] Add Triton compact summary kernel for the target shape first.
- [ ] Add compact prefix combine using factors instead of materialized transition/additive matrices.
- [ ] Reuse current chunk apply/output kernel initially, then fuse/optimize only after correctness is stable.
- [ ] Add benchmark algorithm name for the compact path, e.g. `triton_wy_compact` or replace internal `triton_dense3` route once it is clearly better.

## Correctness gates

- [ ] Local no-CUDA checks:
  - `python -m py_compile rwkv7_hf/dplr_prefill_triton.py rwkv7_hf/dplr_prefill.py bench/bench_dplr_prefill_scan.py tests/test_dplr_prefill_triton.py tests/test_dplr_prefill_scan.py`
  - `git diff --check`
- [ ] 4090 unit tests:
  - `PYTHONPATH=. python tests/test_dplr_prefill_scan.py`
  - `PYTHONPATH=. python tests/test_dplr_prefill_triton.py`
- [ ] 4090 synthetic fp16 target:
  - `out_min_cosine >= 0.9999`
  - state diff comparable to current `triton_wy` / dense3 rows
- [ ] HF repo-code smoke:
  - `RWKV7_NATIVE_PREFILL_DPLR_SCAN=1`
  - `RWKV7_DPLR_PREFILL_ALGORITHM=<candidate>`
  - 0.4B / prompt512 / bsz1
  - greedy/cache smoke must pass

## Performance targets

- Baseline evidence from latest 4090 synthetic target:
  - `sequential`: about `55.63 ms`, `9.2k tok/s`
  - `triton_wy`: about `0.233 ms`, `2.20M tok/s`
  - `triton_dense3`: about `0.584 ms`, `877k tok/s`
- Short-term compact target:
  - [ ] compact path `< 0.4 ms` on synthetic target
  - [ ] then approach or beat current `triton_wy` P0 `~0.233 ms`
- HF target:
  - [ ] 4090 / 0.4B / prompt512 / bsz1 moves toward `>=0.45x` Albatross
  - [ ] stretch: `>=0.60x` Albatross

## Guardrails

- Do not default-enable dense3 in the HF path.
- Do not claim Albatross-level performance from dense3 alone.
- Do not start vLLM/SGLang work in this repository.
- Do not optimize Python loops instead of compiled kernel/factor work.
