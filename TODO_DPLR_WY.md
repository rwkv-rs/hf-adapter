# Temporary TODO: DPLR/WY Compiled Prefill

This is a short-lived working TODO for the current native-prefill performance branches. Keep the default HF path unchanged unless a benchmark explicitly opts in.

## Temporary TODO: 0.60x Albatross experiment branch

Branch: `wangyue/native-prefill-060-albatross`

- [x] Start from merged `origin/main` after PR #90.
- [x] Fix the first experiment blocker:
  - `extract()` now appends the optional `RKVw` pack item used by the
    VKWR/RKV policy path; native prefill and its profiler were still unpacking
    the old 40-field pack.
  - Updated `rwkv7_hf/native_jit.py` and
    `bench/bench_native_prefill_breakdown.py` to accept both 40-field legacy
    packs and 41-field current packs.
- [x] Re-run 4090 fused-state-scan fine/layer breakdown after PR #90:
  - result file: `bench/results_4090_prefill060_experiments_20260702_120602.jsonl`
  - remote row source: `/tmp/native_4090_060_breakdown_20260702_120418.jsonl`
  - pass, greedy match vs native prefill, max diff `0.0`
  - top components in profiled row:
    - `recurrent_scan_state_prep_fused`: `13.3723 ms`, `52.69%`
    - `ffn`: `2.2262 ms`, `8.77%`
    - LoRA path sum (`w/a/v_gate/g`): about `6.1581 ms`
    - `attn_norm_shift_mix`: `1.2186 ms`, `4.8%`
  - profiling row is slower than the end-to-end benchmark because it measures
    per-component CUDA events; use it for attribution, not headline tok/s.
- [x] Run prior-art direction experiments instead of guessing:
  - baseline fused state-scan + fused output:
    - `/tmp/native_4090_060_wavg_ab_20260702_120500.jsonl` baseline:
      `26,197.2 tok/s`, `19.5441 ms`, about `0.5024x` Albatross
    - `/tmp/native_4090_060_shiftmix_ab_20260702_120602.jsonl` baseline:
      `26,487.4 tok/s`, `19.3300 ms`, about `0.5079x` Albatross
  - fused WAVG-LoRA prefill experiment:
    - pass/correct, but slower: `25,552.3 tok/s`, `20.0374 ms`,
      about `0.4900x` Albatross
    - conclusion: do not promote the current WAVG-LoRA prefill kernel; it
      reduces launch count but loses enough inside the Triton kernel to be a
      net negative at 4090 / 0.4B / prompt512 / bsz1.
  - fused shift-mix prefill experiment:
    - pass/correct, but slower: `25,784.5 tok/s`, `19.8569 ms`,
      about `0.4944x` Albatross
    - conclusion: do not promote the current standalone shift-mix kernel;
      Albatross-style norm+mix is still a good boundary, but it must be fused
      deeper with norm/projection instead of adding a standalone launch.
- [x] Wire and verify fused state-scan warp specialization:
  - `fused_recurrent_scan_state_prep(...)` now accepts a validated
    `num_warps` argument, and native prefill/profiler pass
    `RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS` through the existing tuning helper.
  - result file:
    `bench/results_4090_prefill060_state_scan_warps_20260702_121224.jsonl`
  - remote row source:
    `/tmp/native_4090_060_state_scan_warps_20260702_121224.jsonl`
  - all rows pass greedy/cache smoke with `max_abs_diff=0.0625`,
    `min_cosine=1.0`, and `989.2 MiB` peak VRAM:
    - `num_warps=1`: `18,679.0 tok/s`, `27.4104 ms`, about `0.3582x`
    - `num_warps=2`: `23,025.4 tok/s`, `22.2363 ms`, about `0.4415x`
    - `num_warps=4`: `25,816.9 tok/s`, `19.8320 ms`, about `0.4951x`
    - `num_warps=8`: `26,758.3 tok/s`, `19.1343 ms`, about `0.5131x`
  - conclusion: keep Ada/4090 default at `8` warps for this shape; lower warp
    counts are valid but slower and should not be promoted as the default.
- [x] Try split-row fused state-scan as the next direct state-scan experiment:
  - added an opt-in `block_m < head_dim` path inside
    `fused_recurrent_scan_state_prep(...)`, reusing
    `RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M` and
    `RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS`.
  - sweep result file:
    `bench/results_4090_prefill060_state_scan_rows_20260702_202132.jsonl`
  - confirmation result file:
    `bench/results_4090_prefill060_state_scan_rows_confirm_20260702_202538.jsonl`
  - remote row sources:
    `/tmp/native_4090_060_state_scan_rows_20260702_202132.jsonl` and
    `/tmp/native_4090_060_state_scan_rows_confirm_20260702_202538.jsonl`
  - sweep best row was split-row `block_m=8,num_warps=4`: pass,
    `27,004.0 tok/s`, `18.9601 ms`, about `0.5178x` Albatross.
  - confirmation did not support promoting split-row on 4090: `block_m=8,w4`
    fell to `26,299.1 tok/s`, while full-head `block_m=64,w8` confirmed at
    `27,173.3 tok/s`, `18.8420 ms`, about `0.5211x`.
  - conclusion: keep the split-row state-scan path as an experimental knob for
    other cards/shapes, but do not make it the 4090 default. Current best
    confirmed path remains full-head fused state-scan with `8` warps.
- [x] Correct the 4090 harness and re-run current-branch rows:
  - issue found: `/workspace/activate_rwkv7.sh` runs
    `cd /workspace/rwkv7-hf-adapter` and prepends that older checkout to
    `PYTHONPATH`. Correct 4090 commands must now use:
    `source /workspace/activate_rwkv7.sh && cd /workspace/projects/rwkv7-hf-adapter-060 && export PYTHONPATH=.`
  - corrected result files:
    - `bench/results_4090_prefill060_corrected_state_scan_output_20260702_203712.jsonl`
    - `bench/results_4090_prefill060_corrected_confirm_20260702_203852.jsonl`
  - remote row sources:
    - `/tmp/native_4090_060_corrected_state_scan_output_20260702_203712.jsonl`
    - `/tmp/native_4090_060_corrected_confirm_20260702_203852.jsonl`
  - corrected current-branch rows, all pass greedy/cache smoke:
    - full-head state-scan + fused output: best confirm
      `26,395.7 tok/s`, `19.3971 ms`, about `0.5062x` Albatross
    - split-row `block_m=8,w4`: `25,324.1 tok/s`, about `0.4856x`
    - larger state-scan+output-prep fusion `w4`: `23,810.4 tok/s`,
      about `0.4566x`
    - larger state-scan+output-prep fusion `w8`: `23,609.2 tok/s`,
      about `0.4527x`
  - conclusion: the larger fused state-scan+output-prep kernel is correct but
    slower on 4090, so keep it opt-in and do not promote it. The corrected
    current-branch performance evidence supersedes rows produced with the
    wrong activate/cwd order.
- [x] Next experiment should target the real dominant path:
  - first choice: optimize/specialize `fused_recurrent_scan_state_prep` itself
    for 4090/Ada 0.4B `H=16,N=64,T=512`, because it is now over half of the
    profiled component time;
  - second choice: one larger fused attention-prep kernel that combines
    norm/shift-mix + dense R/K/V + W/A/G/V LoRA + state-scan boundary, rather
    than enabling standalone fused WAVG-LoRA or standalone fused shift-mix.
- [x] Corrected-harness breakdown and route sweep:
  - profile the corrected full-head `block_m=64,w8` current-branch path again
    to refresh the real top components after the activate/cwd fix;
  - corrected breakdown result file:
    `bench/results_4090_prefill060_corrected_breakdown_20260702_204229.jsonl`
  - remote row source:
    `/tmp/native_4090_060_corrected_breakdown_20260702_204229.jsonl`
  - top corrected components:
    - `recurrent_scan_state_prep_fused`: `12.8968 ms`, `50.27%`
    - `ffn`: `2.1907 ms`, `8.54%`
    - `attn_lora_w`: `1.7671 ms`, `6.89%`
    - `attn_lora_a`: `1.6815 ms`, `6.55%`
    - `attn_lora_v_gate`: `1.4949 ms`, `5.83%`
    - `attn_norm_shift_mix`: `1.4366 ms`, `5.60%`
    - `attn_lora_g`: `1.3364 ms`, `5.21%`
  - corrected route sweep result file:
    `bench/results_4090_prefill060_corrected_route_sweep_20260702_204349.jsonl`
  - remote row source:
    `/tmp/native_4090_060_corrected_route_sweep_20260702_204349.jsonl`
  - route sweep rows:
    - current state-scan + fused output: pass, `26,206.0 tok/s`,
      `19.5375 ms`, about `0.5025x`
    - state-scan without fused output: pass, `25,440.6 tok/s`, about `0.4878x`
    - old scan+output fusion: pass, `22,769.4 tok/s`, about `0.4366x`
    - separate fused state-prep + scan + output: pass, `22,494.3 tok/s`,
      about `0.4314x`
    - clampw/KV-prep route: correctness pass but not effective/usable here,
      `220.5 tok/s`
  - conclusion: corrected evidence keeps the same engineering direction:
    current full-head `fused_recurrent_scan_state_prep` remains the only viable
    4090 route and is still the dominant cost; older/shallow fusion routes are
    worse and should stay disabled.
- [x] Wire and test `fused_recurrent_scan_state_prep` `num_stages` scheduling:
  - added `RWKV7_NATIVE_PREFILL_SCAN_NUM_STAGES` plumbing through native
    prefill, profiler telemetry, analyzer keys, and
    `fused_recurrent_scan_state_prep(...)`.
  - sweep result file:
    `bench/results_4090_prefill060_state_scan_num_stages_20260702_205036.jsonl`
  - confirmation result file:
    `bench/results_4090_prefill060_state_scan_num_stages_confirm_20260702_205315.jsonl`
  - remote row sources:
    - `/tmp/native_4090_060_state_scan_num_stages_20260702_205036.jsonl`
    - `/tmp/native_4090_060_state_scan_num_stages_confirm_20260702_205315.jsonl`
  - sweep rows:
    - `num_stages=2`: pass, `26,399.4 tok/s`, `19.3943 ms`, about `0.5062x`
    - `num_stages=5`: pass, `26,310.0 tok/s`, about `0.5045x`
    - `num_stages=4`: pass, `25,953.4 tok/s`, about `0.4977x`
    - `num_stages=6`: pass, `25,704.0 tok/s`, about `0.4929x`
    - `num_stages=3`: pass, `25,443.6 tok/s`, about `0.4879x`
    - `num_stages=1`: pass, `24,887.6 tok/s`, about `0.4772x`
  - confirmation rows:
    - `num_stages=3` (current default): pass, `26,745.8 tok/s`,
      `19.1432 ms`, about `0.5129x`
    - `num_stages=2`: pass, `25,988.3 tok/s`, `19.7011 ms`, about `0.4984x`
  - conclusion: expose the knob for future card/shape tuning, but keep the
    default at `3` on 4090 because confirmation did not support promoting
    stage `2`.
- [x] Try reduced K/V writeback via scan-emitted correction:
  - added opt-in `RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_CORRECTION=1`.
  - implementation:
    - `fused_recurrent_scan_state_prep_correction(...)` computes the RWKV
      correction `sum(r * k_adj * r_k) * v_adj` inside the full-head
      state-prep scan and returns `(recurrent, final_state, correction)` rather
      than full adjusted K/V tensors.
    - `fused_attn_output_prepare_from_correction(...)` consumes that correction
      with fused per-head group norm and G gate.
    - default HF/native paths stay unchanged unless the env flag is set.
  - validation result file:
    `bench/results_4090_prefill060_state_scan_correction_confirm_20260702_152407.jsonl`
  - remote row source:
    `/tmp/native_4090_state_scan_correction_confirm_20260702_152407.jsonl`
  - confirmation rows, both pass greedy/cache smoke:
    - no-KV correction path: `26,164.0 tok/s`, `19.5689 ms`,
      about `0.5017x` Albatross, peak `990.2 MiB`
    - current baseline full-head state-scan + fused output:
      `27,051.0 tok/s`, `18.9272 ms`, about `0.5187x` Albatross,
      peak `989.2 MiB`
  - conclusion: correctness is good and the memory-writeback hypothesis is now
    represented by an opt-in experiment, but it is slower on 4090 because the
    correction reduction increases pressure in the already-dominant scan
    kernel. Keep it disabled by default and do not promote it.
- [ ] Next corrected-harness experiment:
  - target `fused_recurrent_scan_state_prep` internal cost directly beyond
    shallow scheduling knobs; reduced K/V writeback is tested and negative, so
    the next candidate is a deeper pre-scan projection/LoRA boundary that does
    not add a standalone launch.
- [ ] Stretch target remains `>=0.60x` Albatross (`>=31,289 tok/s`) for
  4090 / 0.4B / prompt512 / bsz1. Best current confirmed row on this branch is
  `26,745.8 tok/s` (`~0.5129x`), still about `17.0%` short of the stretch.

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
