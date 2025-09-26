# Temporary TODO: DPLR/WY Compiled Prefill

This is a short-lived working TODO for the current `wangyue/native-fused-fp16-kernel` branch. Keep the default HF path unchanged unless a benchmark explicitly opts in.

## Temporary TODO: G1 MATH500 speed gate after seed43 accuracy pass

- [x] Preserve the full seed43 accuracy evidence:
  - `bench/math500_hf_seed43_full_compare_4090_20260704/`
  - HF seed43 `pass@64=0.372` vs Albatross `0.370`
  - current blocker: same run speed is only `1.608x` summary token/s and
    `1.686x` decode token/s vs Albatross, below the `>=2x` gate.
- [x] Add an opt-in deferred verifier path so CPU `math_verify` does not stall
  the dynamic GPU decode/refill loop.
  - Implementation: `bench/eval_math500_hf.py --defer-verification`
  - Default behavior stays unchanged.
  - Optional speed denominator: `--summary-speed-timing generation`.
- [x] Validate deferred verification on a 4090 dynamic smoke.
  - Artifact: `bench/math500_defer_verification_smoke_4090_20260704/`
  - Completion mismatches: `0`
  - Correctness mismatches: `0`
  - Small smoke token/s improved from `358.850` inline to `411.810`
    generation-timed deferred.
- [x] Add and validate opt-in deferred text decode.
  - Implementation: `bench/eval_math500_hf.py --defer-text-decode`
  - Artifact: `bench/math500_defer_text_decode_smoke_4090_20260704/`
  - Completion / correctness / stop mismatches: `0`
  - This removes per-token `tokenizer.decode(...)` from the dynamic
    decode/refill loop while keeping the default path unchanged.
- [x] Run a short dynamic-batch sweep before the full speed-gate run.
  - Artifact: `bench/math500_bsz_sweep_defer_text_4090_20260704/`
  - Shape: `--limit 4 --rollout 64 --max-new-tokens 256`
  - Best short-run row: `bsz=128`, `7131.751` generation token/s.
- [x] Finish the full seed43 avg@64 deferred-verification + deferred-text
  decode run and compare it with the committed Albatross reference.
  - Remote output: `/tmp/math500_hf_dynamic_full_avg64_seed43_bsz128_defer_text_20260704`
  - Remote log: `/tmp/math500_hf_dynamic_full_avg64_seed43_bsz128_defer_text_20260704.log`
  - Artifact:
    `bench/math500_hf_seed43_bsz128_defer_text_full_compare_4090_20260704/`
  - Result: `pass@64=0.380` vs Albatross `0.370`, generation token/s
    `10426.943` vs `3903.633` (`2.671x`), wall token/s `10053.618`
    (`2.575x`), decode token/s `11588.182` vs `3970.135` (`2.919x`).
  - Current G1 gates against the committed full Albatross reference are met.

## Temporary TODO: next 4090 push

Use this as the current scratch checklist until the Albatross-ratio item below
is either checked off or replaced with a more precise kernel task.

- [x] Reproduce the latest 4090 HF/native prefill baseline on repo code:
  - 0.4B checkpoint
  - prompt512
  - bsz1
  - fp16
  - `triton_wy`, `triton_dense3`, `triton_wy_compact`
  - Done: `/tmp/native_4090_todo_sweep_20260702_103919.jsonl`
    reproduced repo-code DPLR rows:
    - `triton_wy`: `20,421.7 tok/s`, `0.3916x` Albatross
    - `triton_dense3`: `18,546.0 tok/s`, `0.3556x` Albatross
    - `triton_wy_compact`: `17,970.5 tok/s`, `0.3446x` Albatross
- [x] Treat the current Albatross reference as:
  - `albatross_speed` / faster3a / 4090 / 0.4B / bsz1 / prompt512:
    `52148.52 tok/s`
  - `0.45x` target: `>=23467 tok/s`
  - `0.60x` stretch target: `>=31289 tok/s`
  - Done: sweep rows now record `albatross_ref_tokps_total`,
    `albatross_ratio`, and `target_0_45_met`.
- [x] Inspect the native prefill env toggles before changing code:
  - `RWKV7_NATIVE_PREFILL_DPLR_SCAN`
  - `RWKV7_DPLR_PREFILL_ALGORITHM`
  - fused output / WAVG-LoRA / shift-mix / state-prep toggles
  - scan block / warp tuning knobs
  - Done: inspected `rwkv7_hf/native_jit.py`,
    `rwkv7_hf/dplr_prefill_triton.py`, and
    `bench/bench_native_prefill_scan.py`. Benchmark telemetry now records the
    DPLR algorithm, DPLR Triton knobs, fused WAVG-LoRA knobs, and prefill
    fused-output-project knobs.
- [x] Run a small 4090 env/autotune sweep and record the fastest passing row:
  - pass greedy/cache smoke
  - pass correctness gates
  - record tok/s, latency, peak VRAM if available
  - Done: `/tmp/native_4090_todo_sweep_20260702_103919.jsonl`.
    Fastest sweep row was `fused_scan_state_bm8_w1`: pass, `22,777.0 tok/s`,
    `22.4788 ms`, `0.4368x` Albatross, `991.2 MiB` peak VRAM.
    Confirmation run `/tmp/native_4090_todo_confirm_20260702_104202.jsonl`
    with warmup=3/steps=9: pass, `22,292.0 tok/s`, `22.9679 ms`,
    `0.4275x`, `991.2 MiB`.
- [x] Check whether a no-code setting reaches `>=0.45x`.
  - Result: no. Best observed setting remained below `23,467 tok/s`, so the
    HF target below is not checked off.
- [x] If no setting reaches `>=0.45x`, create the next concrete kernel task:
  - profile which HF layer/kernel dominates the gap
  - likely targets: compact apply/output fusion, fused fp16 output path, or
    launch-count reduction around DPLR prefill
  - do not continue wrapper-only micro-optimization as the main route
  - Done: `/tmp/native_4090_todo_breakdown_20260702_104126.jsonl` identifies
    the top profiled components for the best fused-scan configuration:
    recurrent scan `7.4571 ms` / `26.34%`, FFN `4.0836 ms` / `14.42%`,
    attention norm+shift-mix `3.8040 ms` / `13.44%`, fused state prep
    `3.2982 ms` / `11.65%`. Next real performance task is launch-count and
    recurrent-scan/state-prep fusion work; DPLR compact apply/output fusion
    remains the DPLR-specific route.
  - First opt-in fused-output-project experiment added behind
    `RWKV7_NATIVE_PREFILL_FUSED_OUTPUT_PROJECT=1` and measured slower:
    `/tmp/native_4090_output_project_20260702_104430.jsonl`, pass,
    `18,228.8 tok/s`; keep it disabled by default.
- [x] Keep default HF behavior unchanged; all experimental paths must stay
  opt-in through env/benchmark flags.
  - Done: DPLR, fused scan, fused output, and fused output-project paths are
    all env/benchmark opt-ins. Default HF path remains unchanged.

## Next concrete kernel TODO from the 4090 sweep

- [x] Complete the RTX 4090 / Ada HF adapter validation issue checklist.
  - GitHub issue: `#66` (`[card] RTX 4090 / Ada — HF 适配验证`).
  - Final artifact: `bench/results_4090_issue66_final_20260702_113804.jsonl`
    and appended rows in `bench/results.jsonl`.
  - Remote log: `/tmp/issue66_4090_final_20260702_113804.log`.
  - Environment recorded: RTX 4090 sm_89, Python `3.12.3`, PyTorch
    `2.11.0+cu128`, CUDA `12.8`, Transformers `5.12.1`, PEFT `0.19.1`,
    TRL `1.7.0`, bitsandbytes `0.49.2`, DeepSpeed `0.19.2`, Accelerate
    `1.14.0`.
  - Passed: `smoke_hf_generate`, `test_hf_api_contract` fp16/bf16,
    `test_quantized_inference` W8/W4, `bench_speed`, `bench_batch_sweep`,
    `test_peft_lora`, `test_hf_training_smoke` Trainer/TRL SFT, and
    `test_hf_rl_training_smoke` DPO.
  - Note: quantized W8/W4 fast-forward now safely falls back to FLA when a
    global native fast-token backend is requested, because bitsandbytes packed
    int8/int4 weights are not dense-native-runner compatible yet.

- [x] Run current 4090 adaptation validation pass before more kernel work.
  - Unit/correctness on RTX 4090 passed:
    - `python -m py_compile ...`
    - `python tests/test_native_prefill_scan.py`
    - `python tests/test_dplr_prefill_scan.py`
    - `python tests/test_dplr_prefill_triton.py`
  - DPLR synthetic validation:
    `/tmp/verify_4090_dplr_20260702_111046.jsonl`
    - `triton_wy`: pass, `0.226 ms`, `2.265M tok/s`
    - `triton_dense3`: pass, `0.27073 ms`, `1.891M tok/s`
    - `triton_wy_compact`: pass, `0.24142 ms`, `2.121M tok/s`
  - HF repo-code validation:
    `/tmp/verify_4090_native_prefill_20260702_111055.jsonl`
    - fused-scan best config: pass, `22,116.7 tok/s`, `0.4241x`
    - cache-view experiment: pass but slower, `22,081.8 tok/s`, not kept
    - DPLR compact HF smoke: pass, `17,663.3 tok/s`, `0.3387x`
  - Conclusion: 4090 correctness is stable; no new row reaches `0.45x`.
- [x] Close the remaining `0.45x` Albatross gap on 4090 / 0.4B / prompt512 /
  bsz1:
  - previous confirmed best: `22,292.0 tok/s` (`0.4275x`)
  - target: `>=23,467 tok/s`
  - Done: opt-in fused state-prep + recurrent scan path
    (`RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN=1` plus
    `RWKV7_NATIVE_PREFILL_FUSED_OUTPUT=1`) confirmed at
    `/tmp/native_4090_fused_state_scan_confirm_20260702_111924.jsonl`:
    pass, `25,663.2 tok/s`, `19.9507 ms`, `0.4921x` Albatross,
    greedy/cache smoke passing, `989.2 MiB` peak VRAM.
- [x] Profile and reduce recurrent-scan/state-prep launch count:
  - profile target row: `fused_scan_state_bm8_w1`
  - first target was saving at least `0.7 ms` from recurrent scan + state prep
    without breaking greedy/cache smoke
  - Done: `fused_recurrent_scan_state_prep(...)` fuses raw W/K/V state prep,
    KK normalization, optional V-first interpolation, recurrent scan, and
    adjusted K/V materialization in one full-head Triton kernel. Compared with
    the prior confirmed best, latency moved from `22.9679 ms` to `19.9507 ms`
    (about `3.02 ms` saved) and throughput rose about `15.1%`.
- [x] Re-test DPLR compact only after apply/output fusion or launch-count
  reduction is implemented; current HF repo-code compact row is correctness
  passing but slower than the fused recurrent scan path.
  - Done: `/tmp/native_4090_dplr_compact_retest_20260702_111924.jsonl`
    remains correctness-passing but slower: `16,863.4 tok/s`, `30.3616 ms`,
    `0.3234x` Albatross. Keep DPLR compact as the high-upside line, but its
    next useful task is DPLR-specific apply/output fusion rather than
    wrapper-level changes.

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
  - [x] 4090 / 0.4B / prompt512 / bsz1 moves toward `>=0.45x` Albatross
    - Done: fused state-scan confirmation row is `25,663.2 tok/s` (`0.4921x`).
  - [ ] stretch: `>=0.60x` Albatross
    - Current confirmed fused state-scan row is still below the stretch target
      `31,289 tok/s` by about `5,626 tok/s` (`~21.9%` relative uplift).

## Big TODO routing note

- [ ] Keep the FLA/PyTorch path as the compatibility and correctness fallback,
  not the main Albatross-gap optimization target. Native-unsupported, training,
  PEFT/TRL, and generic quantized paths may still fall back to FLA/PyTorch.
- [ ] Keep two performance tracks active:
  - short-term: native fused fp16 prefill/decode kernels, starting from the
    confirmed fused state-scan row and pushing 4090 0.4B/prompt512/bsz1 from
    `0.4921x` to `>=0.60x` Albatross;
  - high-upside math: DPLR/WY compact chunk prefill, with next work on
    apply/output fusion, less dense `[N,N]` traffic/materialization, and later
    fused W8/W4 kernels.
- [x] Native MM8/MM4 lm_head bridge for fast-token decode:
  - `native_jit` and `native_graph` no longer assume `lm_head.weight` exists;
    they route the final projection through the module when `lm_head` is a
    packed `MM8Linear` or `MM4Linear`.
  - 4090 validation on `/workspace/models/rwkv7/rwkv7-g1d-0.4b-hf` using an
    effective repo-code model passed:
    - MM8: `1` layer quantized, e2e cosine `0.999991`, native_jit fast-token
      cosine vs quantized-FLA `0.999993`, native_graph `0.999992`.
    - MM4: `1` layer quantized, e2e cosine `0.998870`, native_jit fast-token
      cosine vs quantized-FLA `0.999994`, native_graph `0.999993`.
  - This closes the immediate native fast-token crash for size-gated
    native-quant `lm_head`; full model-level W8/W4 speed claims still require
    the remaining larger projection/FFN fused quant path.
- [x] Prior-art check: search official RWKV-LM, Albatross, FLA, VKWR/rwkv.cpp,
  wind_rwkv, and vLLM/SGLang RWKV work before inventing another kernel
  boundary. Current conclusion: there are strong references, but no merged
  drop-in HF Transformers solution that satisfies our full PEFT/TRL/training +
  native fused performance target. Borrow ideas rather than replacing this repo:
  Albatross/faster3a layout and benchmarks, FLA chunk-DPLR math, wind_rwkv
  H100/MI300X kernels, vLLM closed PR state/scheduler design, VKWR continuous
  batching, and rwkv.cpp quant formats.

## Guardrails

- Do not default-enable dense3 in the HF path.
- Do not claim Albatross-level performance from dense3 alone.
- Do not start vLLM/SGLang work in this repository.
- Do not optimize Python loops instead of compiled kernel/factor work.
