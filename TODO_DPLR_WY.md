# Temporary TODO: DPLR/WY Compiled Prefill

This is a short-lived working TODO for the current native-prefill performance branches. Keep the default HF path unchanged unless a benchmark explicitly opts in.

## Temporary TODO: 0.60x Albatross experiment branch

Branch: `wangyue/native-prefill-060-albatross`

- [x] Add the speed+accuracy acceptance overlay from Albatross MATH500 eval:
  - Reference:
    `https://github.com/BlinkDL/Albatross/blob/main/faster3a_2605/eval_math500.py`.
  - Speed is not only bsz1 prefill tok/s.  The benchmark target must find the
    fastest GPU setting by sweeping bsz / batch policy and report the best
    passing throughput.
  - Accuracy must include MATH500 `avg@64` / rollout-style math verification,
    not only greedy/cache smoke.
  - Different RWKV checkpoints can have different FFN `relu^2` sparsity
    patterns; any sparse-FFN or relusq-related shortcut must be checked per
    model for both speed and accuracy.
  - Logit alignment must use uncheatable compression ratio, including
    compression ratio vs token position.  Max-diff/cosine/greedy checks remain
    smoke tests only and are too weak for final acceptance.
- [x] Add first acceptance-test harnesses and run the first RTX 4090 overlay
  smoke:
  - Added `bench/eval_math500_hf.py`, an HF-adapter MATH500 runner that emits
    Albatross-style fields (`rollout_accuracy`, `pass_at_rollout_accuracy`,
    `sample_per_sec`, `token_per_sec`, `generations_jsonl`, and config).
  - Added `bench/bench_logit_compression_alignment.py`, an uncheatable
    logits/NLL compression-ratio harness over fixed external tokens, including
    `candidate_vs_ref_bits_ratio` and compression ratio vs token-position bins.
  - Fixed `bench/bench_batch_sweep.py` timed prefill to run under
    `torch.inference_mode()`; without this the HF fast-prefill guard sees
    gradients enabled, silently falls back to the slow FLA path, and can OOM at
    high bsz.
  - 4090 native-prefill fastest-bsz sweep:
    `bench/results_4090_accept_speed_bsz_20260703_193117.jsonl`.
    Current 0.4B / prompt512 best passing row in that sweep is `bsz=16`,
    `55,874.9 tok/s`, `146.6132 ms`, peak `1975.6 MiB`; bsz1 in the same
    noisy run is `25,925.3 tok/s`, while the older strict bsz1 best remains
    `28,780.6 tok/s`.
  - High-bsz shift-WAVG MAX_M check:
    `bench/results_4090_accept_speed_bsz_shiftwavg_maxm2_20260703.jsonl`.
    Forcing shift-WAVG at bsz>=4 is slower; best is `50,206.9 tok/s` at
    `bsz=32`, so keep the default MAX_M behavior for now.
  - Serving-style batch sweep after the benchmark fix:
    `bench/results_4090_accept_serving_batch2_20260703.jsonl`.
    Best prefill row is `bsz=16`, `55,702.8 tok/s`; best recurrent fast-token
    decode row is `bsz=64`, `15,043.6 tok/s`, peak `4487.0 MiB`.
  - Compression-ratio alignment:
    `bench/results_4090_accept_logit_compression_20260703.jsonl` over 8
    MATH500 prompts / 561 scored target tokens.  Native-prefill candidate vs
    reference bits ratio is `0.9999498`, argmax match rate `1.0`,
    reference/candidate bits-per-token `1.632204/1.632123`, max abs diff
    `0.25`; position bins are stored in the result row.
  - HF MATH500 smoke:
    `bench/math500_hf_accept_smoke_20260703/summary.json` used 8 tasks,
    rollout 4, max_new_tokens 128: `0/32` correct, truncation `96.875%`,
    `91.81 tok/s`; this validates the harness but is not an accuracy result.
  - HF MATH500 long smoke:
    `bench/math500_hf_accept_long_smoke_20260703/summary.json` used 2 tasks,
    rollout 4, Albatross-style max_new_tokens 1500: `0/8` correct, truncation
    `12.5%`, mean generated tokens `660.5`, `93.26 tok/s`.
  - Superseded by the dynamic runner below: the original bsz1 HF MATH500
    harness remains useful as a correctness smoke, but final speed/accuracy
    acceptance should use `bench/eval_math500_hf.py --dynamic-batching --bsz 64`
    with rollout `64`, max_new_tokens `1500`, and the full 500-task MATH500
    dataset.
- [x] Implement and validate Albatross-style HF MATH500 dynamic rollout:
  - `bench/eval_math500_hf.py` now supports `--dynamic-batching --bsz N` with:
    prompt-state prefill cache, fixed-size decode slots, dynamic slot refill,
    Albatross-compatible `temperature -> top_k -> top_p` sampler, optional
    `--add-bos`, native prefill, and `rwkv7_forward_token` fast-token decode.
  - 4090 dynamic smoke:
    `bench/math500_hf_dynamic_accept_smoke_20260703/summary.json`, 2 tasks,
    rollout 4, bsz 4, max_new_tokens 128: pass, `1024` decoded token events,
    `decode_sec=1.7144`, overall `396.13 tok/s` including prefill/verify.
  - 4090 dynamic long smoke:
    `bench/math500_hf_dynamic_accept_long_20260703/summary.json`, 2 tasks,
    rollout 4, bsz 4, max_new_tokens 1500: `2/8` correct,
    `pass_at_rollout_accuracy=0.5`, `truncated_rate=0.0`, `4572` decoded
    token events, `decode_sec=8.1083`, overall `509.66 tok/s`.
  - 4090 dynamic avg@64 smoke:
    `bench/math500_hf_dynamic_accept_rollout64_smoke_20260703/summary.json`,
    2 tasks, rollout 64, bsz 64, max_new_tokens 1500: `5/128` correct,
    `pass_at_rollout_accuracy=0.5`, `truncated_rate=0.25`, `89309` decoded
    token events, `decode_sec=20.3724`, overall `4201.63 tok/s`.
  - Same 2-task / rollout64 / bsz64 reference through Albatross:
    `bench/math500_albatross_rollout64_head2_20260703/summary.json`.
    It produced the same `5/128` correct and `pass_at_rollout_accuracy=0.5`;
    run log records `decode_s=28.614` for `88727` tokens.  Its summary
    `elapsed_sec=111.565` includes one-time CUDA extension loading/compile
    (`~72.8s`), so use the log's decode line for steady-state comparison.
  - Added reproducibility/comparison helpers for the final acceptance path:
    `scripts/run_math500_acceptance.sh`,
    `bench/compare_math500_summaries.py`, and
    `docs/validation/math500_acceptance.md`.  These are CPU/lightweight except
    for the actual requested eval run, and let the final Albatross result be
    compared without hand calculation.  This benchmark is marked as the current
    final evaluation standard because it follows the requester/bounty-owner
    command: Albatross MATH500 avg@64, fastest practical GPU batch policy, and
    same-policy speed+accuracy comparison.
  - Full 500-task MATH500 avg@64 HF dynamic run completed on the 4090:
    `bench/math500_hf_dynamic_full_avg64_20260703/summary.json` and
    `bench/math500_hf_dynamic_full_avg64_20260703/run.log`.  This used full
    MATH500 (`500` tasks), rollout `64`, dynamic bsz `64`, max_new_tokens
    `1500`, native prefill, and `native_graph` fast-token decode.  Result:
    `4421/32000` correct generations, `rollout_accuracy=0.13815625`,
    `pass_at_rollout_accuracy=0.358`, `truncated_rate=0.21609375`, mean
    generated tokens `612.2159`, `decoded_token_events=19,615,994`,
    `decode_sec=2128.4963`, `elapsed_sec=2141.1968`,
    `token_per_sec=9161.2290`, `sample_per_sec=14.9449`, and prefill cache
    build time `12.1229s`.
  - Full 500-task MATH500 avg@64 Albatross reference completed on the same
    4090 with matching rollout/sampling/stop policy:
    `bench/math500_albatross_full_avg64_20260703/summary.json` and
    `bench/math500_albatross_full_avg64_20260703/run.log`.  Result:
    `4670/32000` correct generations, `rollout_accuracy=0.1459375`,
    `pass_at_rollout_accuracy=0.37`, `truncated_rate=0.21575`, mean generated
    tokens `612.84375`, `decode_s=4945.952` for `19,636,096` decoded tokens,
    `elapsed_sec=5030.2108`, `token_per_sec=3903.6328`, and
    `sample_per_sec=6.3616`.  Direct comparison: HF dynamic is about
    `2.347x` faster by summary token/s (`9161.2290 / 3903.6328`) and about
    `2.320x` faster by steady decode token/s (`19615994/2128.4963` vs
    `19636096/4945.952`), while Albatross accuracy is higher by `249/32000`
    generations and `+0.012` absolute pass@64 (`0.37` vs `0.358`).  The formal
    comparison artifact is now stored at
    `bench/math500_acceptance_4090_20260703/{README.md,comparison.json,comparison.txt}`.
    Next benchmark gate: close the `-0.012` pass@64 gap while keeping the HF
    dynamic route above `2x` Albatross throughput.

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
- [x] Try deeper pre-scan projection/LoRA boundary:
  - added opt-in `RWKV7_NATIVE_PREFILL_FUSED_PROJECTION=1`.
  - implementation:
    - prefill layer 0 can use existing `fused_rkv_wag_projection(...)` for
      R/K/V dense projections plus W/A/G LoRA.
    - prefill layers after 0 can use existing
      `fused_rkv_wavg_projection(...)` for R/K/V plus W/A/G/V-gate LoRA.
    - telemetry records request/effective/max rows and block M/R/K knobs.
    - default HF/native path stays unchanged unless the env flag is set.
  - validation result files:
    - `bench/results_4090_prefill060_fused_projection_smoke_20260702_153144.jsonl`
    - `bench/results_4090_prefill060_fused_projection_sweep_20260702_153224.jsonl`
  - remote row sources:
    - `/tmp/native_4090_prefill_fused_projection_smoke_20260702_153144.jsonl`
    - `/tmp/native_4090_prefill_fused_projection_sweep_20260702_153224.jsonl`
  - rows all pass greedy/cache smoke, but are much slower than the current
    full-head state-scan + fused-output baseline:
    - `block_m=64,block_r=64,block_k=64`: `20,204.5 tok/s`,
      about `0.3874x` Albatross.
    - `block_m=128,block_r=64,block_k=64`: `19,793.3 tok/s`,
      about `0.3796x` Albatross.
    - `block_m=64,block_r=64,block_k=128`: `17,699.2 tok/s`,
      about `0.3394x` Albatross.
    - `block_m=128,block_r=64,block_k=128`: `17,317.0 tok/s`,
      about `0.3321x` Albatross.
    - `block_m=128,block_r=128,block_k=128`: `16,732.3 tok/s`,
      about `0.3209x` Albatross.
  - conclusion: this deeper boundary is now represented by an opt-in
    experiment, but it is negative on 4090 because the Triton dense-projection
    replacement loses badly to cuBLAS for the prefill matrix shapes. Keep it
    disabled by default and do not promote it.
- [x] Try algebraically expanded recurrent output inside the full-head
  state-scan:
  - added opt-in `RWKV7_NATIVE_PREFILL_SCAN_ALGEBRAIC_OUTPUT=1`.
  - implementation:
    - `fused_recurrent_scan_state_prep(...)` can dispatch to an alternate
      full-head Triton kernel that computes
      `sum((state * w + v*k - (state@kk)*kk*a) * r)` as three dot products
      before updating the state, instead of materializing the updated state
      before the recurrent-output reduction.
    - benchmark/profiler/analyzer telemetry now records
      `scan_algebraic_output`.
    - default HF/native behavior stays unchanged unless the env flag is set.
  - validation result files:
    - `bench/results_4090_prefill060_state_scan_algebraic_smoke_20260702_234644.jsonl`
    - `bench/results_4090_prefill060_state_scan_algebraic_confirm_20260702_234722.jsonl`
  - remote row sources:
    - `/tmp/native_4090_state_scan_algebraic_smoke_20260702_234644.jsonl`
    - `/tmp/native_4090_state_scan_algebraic_confirm_20260702_234722.jsonl`
  - confirmation rows, both pass greedy/cache smoke:
    - current baseline full-head state-scan + fused output:
      `26,252.8 tok/s`, `19.5026 ms`, about `0.5034x` Albatross.
    - algebraic-output scan: `25,222.8 tok/s`, `20.2991 ms`, about
      `0.4837x` Albatross, max diff `0.125`.
  - conclusion: correctness is acceptable, but the rewrite is slower on 4090
    because the extra dot products increase arithmetic/register pressure inside
    the already-dominant scan kernel. Keep it disabled by default and do not
    promote it.
- [x] Try no-K/V-writeback scan plus raw-K/V output-prep recompute:
  - added opt-in `RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_RAW_OUTPUT=1`.
  - implementation:
    - `fused_recurrent_scan_state_prep_nokv(...)` runs the full-head
      state-prep scan and returns only `(recurrent, final_state)`, skipping
      adjusted K/V global writeback from the dominant scan kernel.
    - `fused_attn_output_prepare_raw_kv(...)` recomputes adjusted K and
      interpolated V from raw K/V/A plus `k_a`/`v_gate` during output prep, so
      the correction still matches the baseline path.
    - benchmark/profiler/analyzer telemetry now records
      `prefill_fused_state_scan_raw_output_*`.
    - default HF/native behavior stays unchanged unless the env flag is set.
  - validation result files:
    - `bench/results_4090_prefill060_state_scan_raw_output_smoke_20260703_000043.jsonl`
    - `bench/results_4090_prefill060_state_scan_raw_output_confirm_20260703_000125.jsonl`
  - remote row sources:
    - `/tmp/native_4090_state_scan_raw_output_smoke_20260703_000043.jsonl`
    - `/tmp/native_4090_state_scan_raw_output_confirm_20260703_000125.jsonl`
  - confirmation rows, both pass greedy/cache smoke:
    - current baseline full-head state-scan + fused output:
      `26,056.9 tok/s`, `19.6493 ms`, about `0.4997x` Albatross.
    - no-K/V scan + raw-K/V output recompute: `25,941.7 tok/s`,
      `19.7366 ms`, about `0.4975x` Albatross, max diff `0.125`.
  - conclusion: correctness is acceptable and this directly tested whether K/V
    writeback was the bottleneck, but the recompute path is only parity/slightly
    slower on 4090. Keep it disabled by default; the remaining gap is inside
    the state update/readout math and likely needs a dedicated CUDA/persistent
    scan/layout rewrite rather than more output-boundary movement.
- [x] Try head-dim-64 no-mask specialization inside the full-head scan:
  - added opt-in `RWKV7_NATIVE_PREFILL_SCAN_NOMASK64=1`.
  - implementation:
    - `fused_recurrent_scan_state_prep(...)` can dispatch to a specialized
      full-head Triton kernel for `N=64, block_n=64` that removes all
      per-vector masks and masked load/store paths from the dominant scan loop.
    - benchmark/profiler/analyzer telemetry now records `scan_nomask64`.
    - default HF/native behavior stays unchanged unless the env flag is set.
  - validation result files:
    - `bench/results_4090_prefill060_state_scan_nomask64_smoke_20260703_000958.jsonl`
    - `bench/results_4090_prefill060_state_scan_nomask64_confirm_20260703_001039.jsonl`
  - remote row sources:
    - `/tmp/native_4090_state_scan_nomask64_smoke_20260703_000958.jsonl`
    - `/tmp/native_4090_state_scan_nomask64_confirm_20260703_001039.jsonl`
  - confirmation rows, both pass greedy/cache smoke:
    - current baseline full-head state-scan + fused output:
      `26,291.0 tok/s`, `19.4743 ms`, about `0.5042x` Albatross.
    - no-mask N64 scan: `25,764.3 tok/s`, `19.8725 ms`, about `0.4941x`
      Albatross, max diff `0.0625`.
  - conclusion: removing generic mask overhead is correctness-safe but slower
    on 4090; Triton likely already optimizes much of the `N=64` masking or the
    specialized variant changes scheduling/register allocation unfavorably.
    Keep it disabled by default. This further narrows the remaining path to a
    real state-layout/CUDA-persistent rewrite instead of another small Triton
    full-head specialization.
- [x] Start the dedicated CUDA state-scan path with a minimal shared-state
  prototype:
  - added opt-in `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN=1`.
  - implementation:
    - `rwkv7_hf/cuda_state_scan.py` JIT-builds a CUDA extension through
      `torch.utils.cpp_extension.load_inline`.
    - current kernel target is deliberately narrow: fp16, `head_dim=64`, one
      CUDA block per `(batch, head)`, 64 threads, full `[64,64]` state kept in
      shared memory, and raw W/K/V/A state prep plus recurrent scan in CUDA.
    - native prefill can route the existing fused state-scan branch through
      this CUDA prototype; benchmark/profiler/analyzer telemetry records
      `prefill_cuda_state_scan_*`.
    - default HF/native behavior stays unchanged unless the env flag is set.
  - validation result files:
    - `bench/results_4090_prefill060_cuda_state_scan_smoke_20260703_002059.jsonl`
    - `bench/results_4090_prefill060_cuda_state_scan_confirm_20260703_002141.jsonl`
  - remote row sources:
    - `/tmp/native_4090_cuda_state_scan_smoke_20260703_002059.jsonl`
    - `/tmp/native_4090_cuda_state_scan_confirm_20260703_002141.jsonl`
  - confirmation rows, both pass greedy/cache smoke:
    - current Triton baseline full-head state-scan + fused output:
      `25,883.4 tok/s`, `19.7810 ms`, about `0.4963x` Albatross.
    - CUDA shared-state prototype: `10,547.6 tok/s`, `48.5417 ms`, about
      `0.2023x` Albatross, max diff `0.0625`.
  - conclusion: the CUDA route now has a correctness-passing repo scaffold, but
    the naive one-block/shared-state implementation is far slower than Triton.
    This is still useful because it validates the integration/build path and
    gives the next CUDA task a concrete target: increase parallelism and reduce
    per-token global/shared synchronization, not promote this first kernel.
- [x] Add the first card-specific sm70/V100 tuning rule:
  - V100 server validation target: `Tesla V100-PCIE-32GB`, sm70, fp16,
    0.4B / prompt512 / bsz1, current repo code on branch
    `wangyue/native-prefill-060-albatross`.
  - Created a clean V100 checkout at
    `/home/data/wangyue/projects/rwkv7-hf-adapter-prefill060-v100` instead
    of touching the dirty legacy main checkout.
  - Environment: `/home/data/wangyue/envs/rwkv7` with PyTorch
    `2.5.1+cu124`; the older cu118 env exposes a FLA import incompatibility
    with `torch.distributed.tensor.Replicate` and is not the preferred FLA
    benchmark env.
  - Result files:
    - `bench/results_v100_prefill060_sm70_block_sweep_20260703_003614.jsonl`
    - `bench/results_v100_prefill060_sm70_auto_default_20260703_004617.jsonl`
  - Remote row sources:
    - `/tmp/native_v100_prefill060_sm70_block_sweep_20260703_003614.jsonl`
    - `/tmp/native_v100_prefill060_sm70_auto_default_v2_20260703_004617.jsonl`
  - Sweep rows all pass greedy/cache smoke. Best V100 row is split-row
    `block_m=16,num_warps=4,num_stages=3`: `16,379.5 tok/s`,
    `31.2586 ms`, peak `1144.2 MiB`, max diff `0.0625`.
  - Full-head `block_m=64,num_warps=8` is slower on V100: `14,053.3 tok/s`,
    so sm70 should not inherit the Ada/4090 full-head default.
  - Implemented an architecture-aware default: for CUDA sm70 and
    `head_dim=64`, `_native_prefill_scan_block_m(...)` defaults to `16`,
    which then defaults `_native_prefill_scan_num_warps(...)` to `4`; explicit
    `RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M` / `RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS`
    still override this for reproducible sweeps.
  - Fixed benchmark/profiler telemetry to report the effective default
    scan block size instead of always reporting raw `head_dim` when the env
    override is absent.
  - Confirmation with no block/warp env after the default change: pass,
    reported `scan_block_m=16,scan_num_warps=4`, `16,187.9 tok/s`,
    `31.6286 ms`, peak `1144.2 MiB`.

- [x] Try row-parallel CUDA state-scan inside the dedicated CUDA scaffold:
  - added `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_LANES` for the experimental
    CUDA state-scan path. The rowgroup layout supports `1`, `2`, `4`, `8`,
    and `16`; the later row-block layout adds `64`. The default remains `1`,
    so the CUDA path is still opt-in and unchanged unless requested.
  - implementation:
    - new `rwkv7_state_scan_prep_n64_rowgroup_kernel` keeps the same narrow
      fp16 / `head_dim=64` / shared-state CUDA scaffold but assigns multiple
      CUDA threads to each state row, parallelizing the two per-row dot/reduce
      loops that were serial in the first CUDA prototype.
    - benchmark/profiler telemetry now records `prefill_cuda_state_scan_lanes`.
  - validation result files:
    - `bench/results_4090_prefill060_cuda_state_scan_lanes_smoke_20260702_165445.jsonl`
    - `bench/results_4090_prefill060_cuda_state_scan_lanes_sweep2_20260702_165828.jsonl`
  - remote row sources:
    - `/tmp/native_4090_cuda_state_scan_lanes_smoke_20260702_165445.jsonl`
    - `/tmp/native_4090_cuda_state_scan_lanes_sweep2_20260702_165828.jsonl`
  - rows all pass greedy/cache smoke, max diff `0.0625`:
    - Triton baseline row in the same smoke: `25,007.6 tok/s`, `20.4738 ms`.
    - CUDA lanes=2: `9,935.7 tok/s`, slower than the original one-lane
      scaffold.
    - CUDA lanes=4: `14,342.8 tok/s`.
    - CUDA lanes=8: best CUDA row, `17,062.3 tok/s`, `30.0077 ms`, about
      `0.327x` Albatross and about `0.68x` of the same-run Triton baseline.
    - CUDA lanes=16: `15,086.6 tok/s`.
  - conclusion: row-level parallelism materially improves the CUDA scaffold
    versus the first one-block/64-thread kernel (`~10.5k -> ~17.1k tok/s`),
    but it is still well below Triton because every token still performs
    block-wide synchronization around shared state. Keep it disabled by
    default. The next CUDA step must change the state layout / persistent
    schedule rather than only increasing row parallelism.
- [x] Try row-block register-state CUDA layout inside the dedicated CUDA
  scaffold:
  - added `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_LANES=64` as a separate
    row-block layout experiment. The prior lanes `1/2/4/8/16` rowgroup paths
    remain unchanged.
  - implementation:
    - new `rwkv7_state_scan_prep_n64_rowblock_kernel` launches one CUDA block
      per `(batch, head, state_row)` for the narrow fp16 / `head_dim=64`
      prototype.
    - each row keeps its `state[row, col]` element in a register across the
      whole sequence and only writes the final state at the end, avoiding the
      full `[64,64]` shared-state layout used by the first CUDA scaffold.
    - the first row-block version used serial thread-0 reductions; the followup
      version adds warp-shuffle block reductions for the row dot products.
  - validation result files:
    - `bench/results_4090_prefill060_cuda_state_scan_rowblock_smoke_20260702_170519.jsonl`
    - `bench/results_4090_prefill060_cuda_state_scan_rowblock_confirm_20260702_170620.jsonl`
    - `bench/results_4090_prefill060_cuda_state_scan_rowblock_reduce_smoke_20260702_170948.jsonl`
    - `bench/results_4090_prefill060_cuda_state_scan_rowblock_reduce_confirm_20260702_171050.jsonl`
  - remote row sources:
    - `/tmp/native_4090_cuda_state_scan_rowblock_smoke_20260702_170519.jsonl`
    - `/tmp/native_4090_cuda_state_scan_rowblock_confirm_20260702_170620.jsonl`
    - `/tmp/native_4090_cuda_state_scan_rowblock_reduce_smoke_20260702_170948.jsonl`
    - `/tmp/native_4090_cuda_state_scan_rowblock_reduce_confirm_20260702_171050.jsonl`
  - rows all pass greedy/cache smoke:
    - first row-block smoke: `25,361.2 tok/s` versus same-run CUDA lanes=1
      baseline `24,966.5 tok/s`.
    - first row-block confirm: `25,053.1 tok/s` versus same-run baseline
      `26,217.6 tok/s`.
    - warp-shuffle row-block smoke: `26,600.9 tok/s` versus same-run baseline
      `25,489.0 tok/s`.
    - warp-shuffle row-block confirm: `26,069.7 tok/s` versus same-run
      baseline `26,184.1 tok/s`, max diff `0.09375`.
  - conclusion: the register-state row-block rewrite is much more promising
    than the shared-state rowgroup CUDA scaffold and reaches near-parity with
    the current Triton path in a corrected 4090 harness, but confirmation still
    does not beat the Triton baseline and remains below the `0.60x` Albatross
    stretch target. Keep it opt-in. The next useful CUDA step is to remove
    duplicated vector prep / K normalization across the 64 row blocks, e.g.
    split into a head-level vector-precompute stage plus register-row apply, or
    move to a cooperative persistent head-level schedule.
- [x] Try two-stage CUDA vector precompute plus row-block register-state apply:
  - added opt-in `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_PRECOMPUTE=1`, valid
    only with `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN=1` and
    `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_LANES=64`.
  - implementation:
    - new `rwkv7_state_scan_prep_n64_vector_precompute_kernel` runs once per
      `(batch, token, head)` and computes W decay, normalized KK, adjusted K,
      and adjusted V once instead of recomputing them in each of the 64
      row-blocks.
    - new `rwkv7_state_scan_prep_n64_rowblock_precomputed_kernel` keeps the
      row state in registers and consumes the precomputed vectors. K/V returned
      to the rest of the HF path stay fp16, while the row-block update consumes
      fp32 temp vectors for correctness parity with the non-precompute
      row-block path.
    - benchmark/profiler/analyzer telemetry now records
      `prefill_cuda_state_scan_precompute`.
  - validation result files:
    - `bench/results_4090_prefill060_cuda_state_scan_precompute_smoke_20260703_011900.jsonl`
    - `bench/results_4090_prefill060_cuda_state_scan_precompute_confirm_20260703_012150.jsonl`
  - remote row sources:
    - `/tmp/native_4090_cuda_state_scan_precompute_smoke_20260703_011900.jsonl`
    - `/tmp/native_4090_cuda_state_scan_precompute_confirm_20260703_012150.jsonl`
  - rows all pass greedy/cache smoke:
    - smoke baseline Triton: `24,017.2 tok/s`.
    - smoke row-block no-precompute: `27,122.9 tok/s`.
    - smoke two-stage precompute: `26,194.8 tok/s`, peak `994.2 MiB`.
    - confirm baseline Triton: `24,720.9 tok/s`.
    - confirm row-block no-precompute: `26,251.0 tok/s`.
    - confirm two-stage precompute: `26,395.6 tok/s`, peak `994.2 MiB`,
      max diff `0.09375`.
  - conclusion: removing duplicated vector prep/K normalization is correctness
    safe and gives a small confirm win over the same-run row-block CUDA path,
    but the extra precompute kernel and four fp32 temp vectors keep it below
    the historical best confirmed branch row (`27,051.0 tok/s`) and below the
    `0.60x` Albatross stretch. Keep it opt-in. The useful signal is that vector
    prep duplication is real, but it needs a lower-traffic/persistent schedule
    rather than a naive temp-tensor precompute stage.
- [x] Try reduced-temp CUDA precompute mode for the row-block register-state
  path:
  - added `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_PRECOMPUTE_MODE=wk` as a
    second precompute variant. `full` keeps the previous four-fp32-temp
    implementation; `wk` precomputes only W decay and normalized KK as fp32
    temps, while adjusted K/V are written to and consumed from the existing
    fp16 output tensors.
  - implementation:
    - new `rwkv7_state_scan_prep_n64_vector_precompute_wk_kernel` computes
      W decay, normalized KK, fp16 adjusted K, and fp16 adjusted V once per
      `(batch, token, head)`.
    - new `rwkv7_state_scan_prep_n64_rowblock_precomputed_wk_kernel` keeps
      row state in registers and consumes fp32 W/KK plus fp16 K/V. This tests
      whether the prior full-precompute path was mainly losing to temp tensor
      traffic.
    - telemetry now records `prefill_cuda_state_scan_precompute_mode`.
  - validation result files:
    - `bench/results_4090_prefill060_cuda_state_scan_precompute_wk_smoke_20260703_014100.jsonl`
    - `bench/results_4090_prefill060_cuda_state_scan_precompute_wk_confirm_20260703_014420.jsonl`
  - remote row sources:
    - `/tmp/native_4090_cuda_state_scan_precompute_wk_smoke_20260703_014100.jsonl`
    - `/tmp/native_4090_cuda_state_scan_precompute_wk_confirm_20260703_014420.jsonl`
  - rows all pass greedy/cache smoke:
    - smoke baseline Triton: `24,781.9 tok/s`.
    - smoke row-block no-precompute: `25,897.6 tok/s`.
    - smoke full precompute: `25,057.0 tok/s`.
    - smoke reduced-temp `wk`: `25,361.3 tok/s`, peak `990.2 MiB`.
    - confirm baseline Triton: `25,218.6 tok/s`.
    - confirm row-block no-precompute: `26,667.1 tok/s`.
    - confirm reduced-temp `wk`: `25,568.4 tok/s`, peak `990.2 MiB`,
      max diff `0.0625`.
  - conclusion: reduced-temp `wk` is correctness-safe and uses less extra VRAM
    than full precompute, but it is still slower than recomputing vectors
    inside the row-block path on this 4090 shape. The bottleneck is not fixed
    by materializing fewer temp tensors; the next useful step is not another
    global-memory precompute variant, but a cooperative/persistent schedule
    that shares vector prep without writing it to global memory.
- [x] Try cooperative rows-per-block CUDA row-block schedule:
  - added `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_ROWS_PER_BLOCK` with supported
    values `1`, `2`, `4`, and `8`. Values above `1` are valid only for the
    `lanes=64`, no-precompute row-block path.
  - implementation:
    - new `rwkv7_state_scan_prep_n64_rowblock_coop_kernel` handles multiple
      state rows per CUDA block. The block computes R/W/K/A/normalized-KK once
      into shared memory, then multiple row workers consume those shared
      vectors while keeping each row's state element in a register.
    - this tests the next TODO's idea of sharing vector prep without writing
      global temp tensors. It is still a CTA-local prototype, not a real
      persistent/cluster schedule.
    - telemetry now records `prefill_cuda_state_scan_rows_per_block`.
  - validation result files:
    - `bench/results_4090_prefill060_cuda_state_scan_coop_rows_smoke_20260703_020000.jsonl`
    - `bench/results_4090_prefill060_cuda_state_scan_coop_rows_confirm_20260703_020500.jsonl`
  - remote row sources:
    - `/tmp/native_4090_cuda_state_scan_coop_rows_smoke_20260703_020000.jsonl`
    - `/tmp/native_4090_cuda_state_scan_coop_rows_confirm_20260703_020500.jsonl`
  - rows all pass greedy/cache smoke and the small CUDA oracle matches
    row-block `rows_per_block=1` exactly for outputs, state, K, and V:
    - smoke baseline Triton: `24,877.6 tok/s`.
    - smoke row-block `rpb=1`: `25,958.9 tok/s`.
    - smoke cooperative `rpb=2`: `25,641.6 tok/s`.
    - smoke cooperative `rpb=4`: `25,532.0 tok/s`.
    - smoke cooperative `rpb=8`: `24,935.9 tok/s`.
    - confirm baseline Triton: `25,470.3 tok/s`.
    - confirm row-block `rpb=1`: `26,596.1 tok/s`.
    - confirm cooperative `rpb=2`: `25,685.1 tok/s`.
  - conclusion: CTA-local sharing is correctness-safe but slower than one
    row per block. The loss likely comes from lower parallelism / occupancy and
    heavier per-block synchronization overwhelming the saved vector prep. Keep
    it opt-in. A useful persistent schedule probably needs finer-grained
    producer/consumer overlap or warp-specialized vector prep, not simply more
    rows per CTA.
- [x] Try warp-specialized producer/worker CUDA row-block schedule:
  - added `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE=warp_specialized`,
    valid only for the opt-in CUDA state-scan row-block path
    (`RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN=1`,
    `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_LANES=64`,
    no vector precompute).
  - implementation:
    - new `rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel`
      uses one producer warp to compute R/W/K/A/normalized-KK for the current
      token and one worker warp per row to keep row state in registers.
    - each worker lane owns two state columns (`N=64`), reducing the per-row
      worker width from two warps to one warp while sharing vector prep within
      a CTA.
    - telemetry now records `prefill_cuda_state_scan_schedule`.
  - validation result files:
    - `bench/results_4090_prefill060_cuda_state_scan_warpspec_smoke_20260703_022500.jsonl`
    - `bench/results_4090_prefill060_cuda_state_scan_warpspec_confirm_20260703_023000.jsonl`
  - remote row sources:
    - `/tmp/native_4090_cuda_state_scan_warpspec_smoke_20260703_022500.jsonl`
    - `/tmp/native_4090_cuda_state_scan_warpspec_confirm_20260703_023000.jsonl`
  - rows all pass greedy/cache smoke. The small CUDA oracle matches the
    default row-block path with max diffs `[0.0, 9.54e-07, 0.0, 0.0]` for
    `(out, state, K, V)` across `rpb=1/2/4/8`.
  - smoke rows:
    - Triton baseline: `25,459.3 tok/s`.
    - default CUDA row-block `rpb=1`: `26,066.8 tok/s`.
    - warp-specialized `rpb=1`: `26,435.5 tok/s`.
    - warp-specialized `rpb=2`: `25,573.3 tok/s`.
    - warp-specialized `rpb=4`: `25,202.8 tok/s`.
    - warp-specialized `rpb=8`: `26,328.8 tok/s`.
  - confirmation rows:
    - Triton baseline: `25,491.9 tok/s`.
    - default CUDA row-block `rpb=1`: `26,280.6 tok/s`.
    - warp-specialized `rpb=1`: `25,560.6 tok/s`.
  - conclusion: the producer/worker layout is correctness-safe and gave a
    smoke win, but confirmation did not beat the simpler two-warp row-block
    path. Keep it opt-in and do not promote it. The likely issue is that the
    one-warp worker loses enough per-row parallelism / ILP to offset saved
    vector prep, and the intra-CTA schedule still cannot overlap producer and
    worker phases across tokens.
- [x] Add a narrow CUDA row-block micro/profiler path before another large
  rewrite:
  - added `bench/bench_cuda_state_scan_micro.py` and a synthetic
    `cuda_state_scan_rowblock_phase(...)` wrapper around new CUDA phase
    kernels.
  - the phase kernels are cumulative and intentionally model the current
    row-block grid outside the HF layer:
    - phase 0: duplicated vector prep + K normalization;
    - phase 1: phase 0 plus state-dot-KK reduction;
    - phase 2: phase 1 plus state update;
    - phase 3: phase 2 plus recurrent-output reduction.
  - result file:
    `bench/results_4090_prefill060_cuda_state_scan_micro_20260703_024500.jsonl`
  - remote row source:
    `/tmp/cuda_state_scan_micro_4090_confirm_20260703_024500.jsonl`
  - 4090 / synthetic `B=1,T=512,H=16,N=64` rows:
    - phase 0 `prep_norm`: `0.412672 ms`.
    - phase 1 `prep_norm_state_dot`: `0.420864 ms`.
    - phase 2 `prep_norm_state_dot_update`: `0.412704 ms`.
    - phase 3 full cumulative: `0.510976 ms`.
    - delta estimate:
      `{duplicated_vector_prep_norm: 0.412672 ms, state_dot: 0.008192 ms,
      state_update: -0.008160 ms, recurrent_output: 0.098272 ms}`.
    - full default row-block micro: `0.486400 ms`.
    - full warp-specialized `rpb=1` micro: `0.422912 ms`.
    - full warp-specialized `rpb=8` micro: `0.433120 ms`.
  - conclusion:
    - the row-block micro path now confirms the per-layer kernel scale
      (`~0.49-0.51 ms`, consistent with the HF breakdown's
      `~12.9 ms / 24 layers` state-scan cost).
    - the coarse phase deltas show duplicated vector prep / K normalization as
      the largest standalone cumulative block, while recurrent output is the
      next visible cost. State-dot/update deltas are too small/noisy in this
      synthetic phase kernel to over-interpret.
    - standalone micro prefers warp-specialized kernels, but full HF
      confirmation did not, so micro rows are direction evidence only and not
      promotion proof.
    - since global-temp precompute, CTA-local sharing, and producer/worker
      sharing all failed in full HF, another small row-block sharing variant is
      unlikely to close the remaining `~15.7%` stretch gap. A true
      persistent/inter-CTA CUDA rewrite remains possible but is a larger design
      item; the next bounded experiment should pivot back to the high-upside
      DPLR/WY compact apply/output fusion track.
- [x] Next corrected-harness experiment:
  - CUDA rowgroup, row-block, full precompute, reduced-temp precompute, and
    CTA-local cooperative rows-per-block plus warp-specialized producer/worker
    testing, plus the new row-block micro/profiler, narrowed the remaining
    credible path: the row-register CUDA layout is viable but bounded sharing
    attempts are not enough in full HF. Return to the earlier DPLR/WY compact
    track and implement the next bounded apply/output fusion experiment for
    `triton_wy_compact`, using the existing synthetic correctness oracle and
    then the corrected 4090 HF smoke. Do not promote wrapper/projection fusion
    or the current rowgroup, row-block, full-precompute, reduced-temp,
    CTA-local cooperative, warp-specialized, or micro-only scaffolds.
  - Done: added the opt-in compact apply/output experiment
    `RWKV7_DPLR_TRITON_COMPACT_OUTPUT_ONLY=1`.
    - implementation:
      - new `dplr_dense_chunk_apply_output_triton(...)` mirrors the existing
        dense chunk apply/output stage but omits dense `chunk_end_state`
        materialization;
      - `dplr_compact_wy_three_stage_triton(..., output_only=True)` now uses
        compact prefix-combine's `prefix_final` as the final state, so stage 3
        only emits recurrent outputs;
      - `bench/bench_dplr_prefill_scan.py --compact-stage-probe` records
        compact summary/prefix/apply/full timings plus output-only timings;
      - default HF/DPLR behavior is unchanged unless the env flag is set.
    - correctness:
      - 4090 unit gate: `python tests/test_dplr_prefill_triton.py` passes.
      - synthetic oracle file:
        `bench/results_4090_prefill060_dplr_compact_output_probe_20260703_030000.jsonl`
      - HF corrected smoke file:
        `bench/results_4090_prefill060_native_dplr_compact_output_probe_20260703_030000.jsonl`
      - both synthetic and HF rows pass with `out_min_cosine=1.0`; HF greedy
        and decode-after-prefill smoke pass.
    - 4090 synthetic stage probe, `B=1,T=512,H=16,N=64,chunk=64,fp16`:
      - compact summary: `0.14335 ms`;
      - compact prefix: `0.05637 ms`;
      - normal compact apply/output: `0.05847 ms`;
      - output-only apply/output: `0.05330 ms` (`~8.8%` faster for the apply
        stage);
      - full compact path: `0.22795 ms`;
      - full output-only compact path: `0.22985 ms` in the stage-probe row
        and `0.25851 ms` in the separate env row, so synthetic end-to-end does
        not justify promoting the flag.
    - 4090 HF corrected smoke, 0.4B / prompt512 / bsz1:
      - normal `triton_wy_compact`: pass, `17,329.4 tok/s`, `29.5452 ms`,
        peak `1038.5 MiB`;
      - output-only compact: pass, `18,345.6 tok/s`, `27.9085 ms`, peak
        `996.2 MiB`.
      - conclusion: this bounded fusion is useful and memory-positive in the
        HF path (`~5.9%` faster than same-run compact baseline and `~42 MiB`
        less peak VRAM), but still far below the main fused recurrent scan
        line and far below the `0.60x` Albatross stretch. Keep it opt-in.
- [x] Next compact-WY task:
  - The output-only apply experiment shows chunk-end writeback is worth
    removing in HF, but stage timings now make `compact_chunk_summary` +
    `compact_prefix_combine` the larger remaining compact path. The next
    bounded DPLR/WY experiment should reduce dense `start_states`
    materialization/readback or fuse compact prefix metadata more deeply with
    apply/output, while preserving chunk-level parallelism. Do not switch back
    to wrapper-only optimization.
  - Done: added opt-in compact fp16 start-state materialization:
    `RWKV7_DPLR_TRITON_COMPACT_START_STATE_DTYPE=fp16`.
    - implementation:
      - `dplr_compact_wy_prefix_combine_triton(..., start_dtype=...)` can now
        store dense chunk `start_states` as fp32/fp16/bf16; default remains
        fp32;
      - `dplr_dense_chunk_apply_triton(...)` and the output-only apply helper
        now read fp16 starts back into fp32 inside the Triton kernel when the
        recurrent vectors are fp16;
      - benchmark/HF telemetry records the compact start-state dtype.
    - result files:
      - synthetic:
        `bench/results_4090_prefill060_dplr_compact_fp16_starts_20260703_034500.jsonl`
      - HF corrected smoke:
        `bench/results_4090_prefill060_native_dplr_compact_fp16_starts_20260703_034500.jsonl`
    - 4090 synthetic, `B=1,T=512,H=16,N=64,chunk=64,fp16`,
      output-only compact with fp16 starts:
      - pass, `out_min_cosine=0.99999988`;
      - `start_states_max_abs_diff=0.00012207`;
      - peak benchmark VRAM drops from the previous compact probe's
        `60.7 MiB` to `53.4 MiB`;
      - prefix time regresses from `0.05637 ms` to `0.05947 ms`;
      - output-only apply regresses from `0.05330 ms` to `0.05474 ms`;
      - full output-only compact is essentially flat/slightly worse:
        `0.22847 ms` versus prior `0.22985 ms` stage-probe row and prior
        `0.23091 ms` normal algorithm row, within noise.
    - 4090 HF corrected smoke, 0.4B / prompt512 / bsz1:
      - output-only fp32 starts: pass, `17,978.2 tok/s`, `28.4789 ms`,
        peak `996.2 MiB`;
      - output-only fp16 starts: pass, `17,549.1 tok/s`, `29.1754 ms`,
        peak `995.2 MiB`, max diff `0.125`;
      - conclusion: fp16 start states are correctness-safe and slightly reduce
        memory, but they do not speed the compact path on 4090 and should stay
        opt-in / not promoted.
- [x] Next compact-WY task:
  - Do not spend another iteration on lossy `start_states` storage. The next
    useful compact experiment should reduce or bypass dense start-state traffic
    without adding lossy conversion, e.g. a compact-prefix/apply fusion that
    computes each chunk's start state inside the apply boundary from compact
    factors, or a lower-traffic prefix representation that still preserves
    chunk-level parallelism. Keep output-only apply as the best compact HF
    flag so far, but do not default-enable it until a same-run synthetic/HF
    confirmation beats the current fused recurrent scan line.
  - Done: added opt-in compact recompute-starts apply/output fusion:
    `RWKV7_DPLR_TRITON_COMPACT_RECOMPUTE_STARTS=1`.
    - implementation:
      - new `dplr_compact_wy_recompute_apply_output_triton(...)` launches the
        apply/output stage per `(batch, chunk, head, row_block)` and
        recomputes each chunk's dense start state from compact WY prefix
        factors inside the kernel;
      - this avoids global dense `start_states` materialization/readback and
        writes only the final state's last chunk, preserving fp32 start-state
        math instead of using lossy fp16 starts;
      - stage probe now records `compact3_recompute_starts_full`; HF telemetry
        records `prefill_dplr_compact_recompute_starts`.
    - result files:
      - stage probe:
        `bench/results_4090_prefill060_dplr_compact_recompute_starts_stage_20260703_044500.jsonl`
      - synthetic algorithm confirm:
        `bench/results_4090_prefill060_dplr_compact_recompute_starts_confirm_20260703_043500.jsonl`
      - HF corrected smoke:
        `bench/results_4090_prefill060_native_dplr_compact_recompute_starts_20260703_043000.jsonl`
    - 4090 synthetic, `B=1,T=512,H=16,N=64,chunk=64,fp16`:
      - correctness passes, `out_min_cosine=0.99999988`,
        `state_max_abs_diff=0.0001257`;
      - normal compact full in the same stage probe: `0.22799 ms`;
      - output-only compact full: `0.22986 ms`;
      - recompute-starts full: `0.50250 ms`;
      - algorithm confirm with env flag: `0.63908 ms`.
      - conclusion: recomputing starts removes the dense start-state traffic
        but duplicates too much compact prefix math for synthetic throughput.
    - 4090 HF corrected smoke, 0.4B / prompt512 / bsz1:
      - same-run output-only compact baseline: pass, `17,696.9 tok/s`,
        `28.9316 ms`, peak `996.2 MiB`;
      - recompute-starts compact: pass, `18,620.9 tok/s`, `27.4960 ms`,
        peak `994.2 MiB`;
      - conclusion: the non-lossy recompute route is mildly better in full HF
        than the same-run output-only compact row and saves a little memory,
        but it is still far below the main fused recurrent scan line and is
        strongly negative on synthetic. Keep it opt-in; do not promote.
- [x] Next compact-WY task:
  - Stop testing duplicated-prefix variants at the current chunk count. The
    useful next compact experiment should preserve the prefix-combine work
    efficiency while reducing dense start-state traffic, e.g. store a smaller
    prefix representation or add a true segmented/persistent prefix+apply
    schedule that shares compact prefix work across chunks without lossy
    starts. If that is too large for one turn, first add a micro/probe that
    estimates how much of recompute-starts time is duplicated prefix vs token
    apply so the next kernel boundary is chosen from evidence.
  - Done: added a timing-only compact recompute-starts phase probe:
    - new helper: `dplr_compact_wy_recompute_phase_probe_triton(...)`;
    - new stage rows:
      `compact_recompute_prefix_only_probe` and
      `compact_recompute_token_apply_only_probe`;
    - analyzer now preserves the phase probe fields.
  - Validation:
    - local no-torch gate: py_compile, `git diff --check`, and
      `python tests/test_dplr_prefill_triton.py` skip/pass;
    - 4090 gate: py_compile and `python tests/test_dplr_prefill_triton.py`
      pass.
  - Result file:
    `bench/results_4090_prefill060_dplr_compact_recompute_phase_probe_20260703_073554.jsonl`
    copied from remote
    `/tmp/dplr_compact_recompute_phase_probe_4090_20260703_073554.jsonl`.
  - 4090 synthetic `B=1,T=512,H=16,N=64,chunk=64,fp16`, warmup `3`,
    steps `9`:
    - normal compact full: `0.22843 ms`;
    - output-only compact full: `0.23002 ms`;
    - recompute-starts full: `0.50274 ms`;
    - duplicated-prefix-only probe: `0.34365 ms`
      (`0.6835x` of recompute full);
    - token-apply-only probe: `0.07983 ms`
      (`0.1588x` of recompute full);
    - prefix+token probe sum: `0.42348 ms`
      (`0.8423x` of recompute full).
  - conclusion: recompute-starts is dominated by duplicated compact-prefix
    math, not by token apply. Do not spend another iteration on recomputing
    prefix per chunk. The next compact boundary must share prefix work once
    while avoiding dense start-state traffic, or compact should stay research
    only while the main fused fp16 line chases the `0.60x` stretch.
- [x] Next compact-WY task:
  - Prototype a true prefix-shared schedule instead of another duplicated
    recompute variant. The bounded experiment should either:
    - fuse prefix+apply in a segmented/persistent schedule that computes each
      chunk start once and immediately consumes it, without materializing dense
      `start_states`; or
    - prove with a small benchmark that such a schedule loses too much
      chunk-level parallelism, then route the remaining `0.60x` work back to
      the main fused fp16 recurrent-scan/output line.
  - Promotion gate: same-run synthetic and HF corrected smoke must beat the
    current output-only compact route and must not regress the main fused
    recurrent scan line.
  - Done: added opt-in prefix-shared compact apply/output schedule:
    `RWKV7_DPLR_TRITON_COMPACT_PREFIX_SHARED=1`.
    - new helper:
      `dplr_compact_wy_prefix_shared_apply_output_triton(...)`;
    - new stage rows:
      `compact_prefix_shared_apply_output` and
      `compact3_prefix_shared_full`;
    - HF telemetry now records
      `prefill_dplr_compact_prefix_shared`;
    - default HF/native behavior is unchanged unless the env flag is set.
  - Validation:
    - local no-torch gate: py_compile, `git diff --check`, and
      `python tests/test_dplr_prefill_triton.py` skip/pass;
    - 4090 gate: py_compile and `python tests/test_dplr_prefill_triton.py`
      pass.
  - Result files:
    - synthetic:
      `bench/results_4090_prefill060_dplr_compact_prefix_shared_20260703_074611.jsonl`
      from remote `/tmp/dplr_compact_prefix_shared_4090_20260703_074611.jsonl`;
    - HF corrected smoke:
      `bench/results_4090_prefill060_native_dplr_prefix_shared_20260703_074742.jsonl`
      from remote `/tmp/native_4090_dplr_prefix_shared_20260703_074742.jsonl`.
  - 4090 synthetic `B=1,T=512,H=16,N=64,chunk=64,fp16`:
    - output-only compact full, same run: `0.22998 ms`;
    - prefix-shared apply/output stage: `0.16429 ms`;
    - compact3 prefix-shared full: `0.16565 ms`;
    - env-routed `triton_wy_compact` with prefix-shared:
      `0.22027 ms`, `2.324M tok/s`, pass.
  - 4090 HF corrected smoke, 0.4B / prompt512 / bsz1:
    - same-run output-only compact baseline: pass, `17,649.4 tok/s`,
      `29.0095 ms`, peak `996.2 MiB`;
    - prefix-shared compact: pass, `20,205.0 tok/s`, `25.3402 ms`,
      about `0.3875x` Albatross, peak `991.2 MiB`;
    - correctness gates pass: greedy/cache smoke pass, decode-after-prefill
      greedy match pass, max diff `0.125`.
  - conclusion: prefix sharing is useful and removes the duplicated-prefix
    loss without dense `start_states`, improving the compact HF line by about
    `14.5%` versus same-run output-only compact. It still remains far below
    the main fused recurrent scan line (`27,051 tok/s`, `~0.5187x`) and far
    below the `0.60x` stretch, so keep it opt-in. The next compact work must
    recover chunk-level parallelism without reintroducing dense starts; if that
    is not viable, route the Albatross gap back to the main fused fp16 line.
- [x] Next compact-WY/mainline routing task:
  - Try one bounded prefix-shared parallelism recovery experiment only. The
    useful direction is a persistent/segmented producer-consumer schedule that
    shares chunk starts without global dense `start_states` and without
    serializing all chunks per row block. If a small prototype cannot beat the
    prefix-shared `20,205 tok/s` HF row or approach the main `27,051 tok/s`
    line, freeze compact-WY as a research/quantization track and move the next
    `0.60x` experiment back to main fused fp16 recurrent-scan/output.
  - Done: added opt-in grouped prefix-shared schedule:
    `RWKV7_DPLR_TRITON_COMPACT_PREFIX_SHARED_GROUP_SIZE=<1|2|4>`.
    - new helper:
      `dplr_compact_wy_grouped_prefix_shared_apply_output_triton(...)`;
    - stage probe rows:
      `compact_grouped_prefix_shared_g1`,
      `compact_grouped_prefix_shared_g2`, and
      `compact_grouped_prefix_shared_g4`;
    - HF telemetry now records
      `prefill_dplr_compact_prefix_shared_group_size`;
    - default HF/native behavior remains unchanged.
  - Validation:
    - local no-torch gate: py_compile, `git diff --check`, and
      `python tests/test_dplr_prefill_triton.py` skip/pass;
    - 4090 gate: py_compile and `python tests/test_dplr_prefill_triton.py`
      pass.
  - Result files:
    - synthetic:
      `bench/results_4090_prefill060_dplr_compact_grouped_prefix_shared_20260703_075658.jsonl`
      from remote
      `/tmp/dplr_compact_grouped_prefix_shared_4090_20260703_075658.jsonl`;
    - HF corrected smoke:
      `bench/results_4090_prefill060_native_dplr_grouped_prefix_shared_20260703_075802.jsonl`
      from remote
      `/tmp/native_4090_dplr_grouped_prefix_shared_20260703_075802.jsonl`.
  - 4090 synthetic `B=1,T=512,H=16,N=64,chunk=64,fp16`:
    - serial prefix-shared stage: `0.16423 ms`;
    - grouped `g1`: `0.50133 ms`;
    - grouped `g2`: `0.34034 ms`;
    - grouped `g4`: `0.26055 ms`;
    - env-routed grouped `g4`: `0.40192 ms`, `1.274M tok/s`, pass.
  - 4090 HF corrected smoke, 0.4B / prompt512 / bsz1:
    - grouped `g4`: pass, `18,171.0 tok/s`, `28.1768 ms`, peak
      `994.2 MiB`;
    - this is below the previous prefix-shared compact row
      `20,205.0 tok/s` and far below the main fused recurrent scan row
      `27,051.0 tok/s`.
  - conclusion: the bounded parallelism-recovery prototype did not beat
    serial prefix sharing. Compact-WY is now frozen as an opt-in
    research/quantization track for this branch. The next `0.60x` Albatross
    work moves back to the main fused fp16 recurrent-scan/output path.
- [x] Next main fused-fp16 task:
  - Resume the main Albatross-gap line instead of compact-WY. Start from the
    strict confirmed 4090 row `27,051.0 tok/s` and profile/reduce the largest
    remaining native prefill costs in the fused state-scan + fused-output path.
    The next bounded experiment should target a real launch/memory boundary
    on the main path and must compare against the same-run fused recurrent
    scan baseline. Promotion gate: move toward `>=31,289 tok/s` without
    breaking greedy/cache/decode smoke.
  - Done: refreshed the corrected 4090 mainline profile and tried two bounded
    main-path experiments.
    - refreshed baseline/result files:
      - `bench/results_native_4090_main_baseline_20260703_000836.jsonl`
        from remote `/tmp/native_4090_main_baseline_20260703_000836.jsonl`;
      - `bench/results_native_4090_main_breakdown_20260703_000836.jsonl`
        from remote `/tmp/native_4090_main_breakdown_20260703_000836.jsonl`.
    - refreshed 4090 / 0.4B / prompt512 / bsz1 baseline:
      - e2e row: pass, `25,891.5 tok/s`, `19.7748 ms`, peak
        `989.2 MiB`;
      - breakdown row: `recurrent_scan_state_prep_fused` remains dominant:
        `13.4225 ms`, `51.66%`; next are FFN `2.2284 ms`, W/A/V/G LoRA
        about `6.37 ms` combined, and norm/shift-mix `1.2908 ms`.
    - Experiment A: opt-in W-decay precompute before fused state-scan:
      - added `RWKV7_NATIVE_PREFILL_SCAN_PRECOMPUTE_W=1` and
        `RWKV7_NATIVE_PREFILL_SCAN_PRECOMPUTE_W_DTYPE={fp32,input}`.
      - implementation keeps default behavior unchanged; when enabled,
        native prefill materializes `exp(-0.606531 * sigmoid(w_raw))` before
        calling `fused_recurrent_scan_state_prep(...)`, and the Triton scan
        kernel skips its internal sigmoid/exp via `W_PRECOMPUTED`.
      - telemetry/analyzer now record `scan_precompute_w` and
        `scan_precompute_w_dtype`; profiler records
        `attn_w_decay_precompute`.
      - smoke result file:
        `bench/results_native_4090_prew_smoke_20260703_002028.jsonl`;
        breakdown result file:
        `bench/results_native_4090_prew_breakdown_20260703_002205.jsonl`.
      - smoke rows all pass greedy/cache/decode:
        - same-run baseline: `25,477.1 tok/s`, `20.0965 ms`;
        - precompute W fp32: `24,089.8 tok/s`, `21.2538 ms`;
        - precompute W input/fp16: `24,248.3 tok/s`, `21.1149 ms`,
          max diff `0.125`.
      - breakdown explains the loss: state-scan time does not improve
        (`13.2082 ms` baseline vs `13.2148 ms` precompute-W), while the new
        `attn_w_decay_precompute` component adds `1.6128 ms`.
      - conclusion: do not promote W precompute. The bottleneck is not the
        scan-loop W sigmoid/exp boundary.
    - Experiment B: tune the existing opt-in fused WAVG-LoRA prefill kernel
      after the refreshed profile showed LoRA as the next largest aggregate
      cost outside state-scan.
      - single-process sweep result file:
        `bench/results_native_4090_wavg_lora_blocks_single_20260703_002450.jsonl`;
      - confirmation result file:
        `bench/results_native_4090_wavg_lora_best_confirm_20260703_002557.jsonl`.
      - best sweep setting was `block_m=64,block_r=32,block_k=64`: pass,
        `24,842.6 tok/s` vs same-run baseline `24,630.8 tok/s`.
      - confirmation stayed correctness-clean but the gain was small/noisy:
        - pass 1: baseline `24,893.6 tok/s`, tuned WAVG `25,206.7 tok/s`;
        - pass 2: baseline `25,464.2 tok/s`, tuned WAVG `25,539.5 tok/s`.
      - conclusion: keep fused WAVG-LoRA opt-in and tuned setting recorded,
        but do not promote it as a default because it does not approach the
        strict best mainline row (`27,051.0 tok/s`) or the `0.60x` target.
- [x] Next main fused-fp16 task:
  - Stop spending iterations on W precompute or shallow WAVG-LoRA tuning.
    The next `0.60x` attempt should target the actual remaining scan
    state-layout/readout work: either a stronger full-head state-scan phase
    probe that isolates state-dot/update/readout costs in the Triton path, or
    a CUDA/persistent row-register schedule that improves on the previous
    row-block near-parity result without adding global temp precompute.
  - Done: added a stronger full-head Triton state-scan phase probe for the
    current main fused fp16 path.
    - implementation:
      - new synthetic helper
        `fused_recurrent_scan_state_prep_phase_probe(...)` times cumulative
        phases of the full-head `fused_recurrent_scan_state_prep` loop;
      - new benchmark `bench/bench_fused_state_scan_micro.py` records
        `triton_state_scan_micro` rows;
      - analyzer now preserves the Triton full-head micro rows and adds their
        component estimates to `next_focus`;
      - default HF/native behavior is unchanged.
    - result files:
      - Triton micro:
        `bench/results_triton_state_scan_micro_4090_20260703_003443.jsonl`
        from remote `/tmp/triton_state_scan_micro_4090_20260703_003443.jsonl`;
      - HF smoke after adding the probe:
        `bench/results_native_4090_after_triton_micro_smoke_20260703_003717.jsonl`
        from remote
        `/tmp/native_4090_after_triton_micro_smoke_20260703_003717.jsonl`.
    - 4090 synthetic full-head scan micro, `B=1,T=512,H=16,N=64,fp16`,
      `num_warps=8,num_stages=3`:
      - phase 0 prep/K-normalization/KV/W: `0.334848 ms`;
      - phase 1 plus state-dot-KK: `0.476160 ms`;
      - phase 2 plus state update: `0.504832 ms`;
      - phase 3 plus recurrent output: `0.549792 ms`;
      - normal full fused helper: `0.515072 ms`;
      - component estimate:
        `{prep_norm_kv_w: 0.334848 ms, state_dot: 0.141312 ms,
        state_update: 0.028672 ms, recurrent_output: 0.044960 ms,
        phase3_vs_full: 0.034720 ms}`.
    - correctness:
      - phase 3 matches the normal full-head helper exactly for synthetic
        outputs, final state, adjusted K, and adjusted V (`max_abs_diff=0.0`
        for all four).
      - corrected 4090 HF smoke still passes after adding the probe:
        `25,008.6 tok/s`, `20.4730 ms`, peak `989.2 MiB`,
        `max_abs_diff=0.0625`, greedy/cache/decode smoke pass.
    - conclusion: the earlier W-only precompute boundary was too narrow.  The
      actual full-head Triton scan signal says combined vector prep,
      K-normalization, K/V writeback, and W decay dominate the synthetic
      kernel, with state-dot-KK as the next visible cost; update/readout are
      smaller.  The next mainline experiment should reduce this prep/norm/KV
      phase without global temp tensors, or revisit the CUDA row-register
      path with a persistent/cooperative schedule that shares those vectors
      while retaining enough parallelism.
- [x] Next main fused-fp16 task:
  - Use the new full-head phase evidence rather than another W-only or shallow
    LoRA boundary.  The next bounded experiment should target the dominant
    `prep_norm_kv_w` + state-dot region directly.  Candidate directions:
    - a Triton variant that keeps adjusted K/V local and fuses the needed
      correction/output consumer so K/V global writeback does not dominate; or
    - a CUDA row-register/persistent schedule that shares R/W/K/A/KK/V prep
      once per token/head without materializing fp32 temp tensors or losing the
      row-block parallelism that made the previous CUDA row-block near parity.
  - Done: extended the full-head Triton phase probe to isolate adjusted K/V
    global writeback cost.
    - implementation:
      - `fused_recurrent_scan_state_prep_phase_probe(..., write_kv=False)`
        disables adjusted K/V stores while keeping the same prep/update/output
        math live;
      - `bench/bench_fused_state_scan_micro.py --include-no-kv-write` records
        both normal and no-K/V-write phase rows plus a delta summary;
      - analyzer now preserves `write_kv` micro rows and reports the no-K/V
        delta in `next_focus`; default HF/native behavior is unchanged.
    - result file:
      - `bench/results_triton_state_scan_micro_nokv_4090_20260703_004626.jsonl`
        from remote
        `/tmp/triton_state_scan_micro_nokv_4090_20260703_004626.jsonl`.
    - 4090 synthetic full-head scan micro, `B=1,T=512,H=16,N=64,fp16`,
      `num_warps=8,num_stages=3`:
      - normal write-K/V summary: phase 3 `0.548864 ms`, normal full helper
        `0.514048 ms`, phase 3 exact vs helper for outputs/state/K/V;
      - no-K/V-write summary: phase 3 `0.460800 ms`;
      - total adjusted K/V writeback estimate: `0.088064 ms`, about
        `16.0%` of the normal phase-3 probe;
      - phase-0 prep/K-normalization/KV/W delta: `0.068736 ms`
        (`0.349184 ms` with K/V stores vs `0.280448 ms` without).
    - conclusion: adjusted K/V writeback is a real cost inside the synthetic
      full-head scan, but the prior full HF no-K/V/raw-output recompute path
      was only parity/slower because it paid the savings back in downstream
      recompute/output prep.  The next main experiment should not repeat raw
      recompute; it should either consume the local adjusted K/V in a fused
      correction/output boundary, or move to a CUDA/persistent row-register
      schedule that shares vector prep and keeps enough row parallelism.
- [x] Next main fused-fp16 task:
  - Convert the no-K/V-write signal into an end-to-end candidate instead of
    stopping at the micro result.  The bounded route should avoid the already
    negative raw-output recompute path.  Preferred next experiment:
    - fused no-K/V state-scan plus correction/output-prep boundary that keeps
      adjusted K/V local for the RWKV correction, or an equivalent CUDA
      persistent row-register schedule that shares per-token vectors without
      global temp tensors;
    - compare against same-run fused recurrent scan + fused output baseline on
      4090 / 0.4B / prompt512 / bsz1;
    - preserve greedy/cache/decode smoke and only promote if it moves toward
      `>=31,289 tok/s`.
  - Done: added the opt-in sk-scale no-K/V end-to-end route
    `RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_SK_OUTPUT=1`.
    - implementation:
      - `fused_recurrent_scan_state_prep_sk(...)` runs the full-head
        state-prep scan without adjusted K/V writeback and emits only
        recurrent output, final state, and per-token/head
        `sk=sum(r * k_adj * r_k)`;
      - `fused_attn_output_prepare_from_sk_raw_v(...)` consumes recurrent
        output + `sk` + raw V, recomputing only V interpolation for later
        layers instead of recomputing adjusted K or writing a full correction
        tensor;
      - native prefill and benchmark telemetry now record
        `prefill_fused_state_scan_sk_output_*`; default HF/native behavior is
        unchanged unless the env flag is set.
    - correctness:
      - remote synthetic oracle on 4090: state-scan recurrent output and final
        state match the normal full K/V-writeback route exactly
        (`out_diff=0.0`, `state_diff=0.0`), and output-prep diff is within
        fp16 tolerance (`prep_diff=0.0625`).
      - all HF rows below pass greedy/cache/decode smoke with
        `max_abs_diff=0.0625`, `min_cosine=1.0`, and peak `989.2 MiB`.
    - result files:
      - smoke:
        `bench/results_native_4090_state_scan_sk_output_20260703_005756.jsonl`
        from remote `/tmp/native_4090_state_scan_sk_output_20260703_005756.jsonl`;
      - warp/stage sweep:
        `bench/results_native_4090_state_scan_sk_output_sweep_20260703_005904.jsonl`
        from remote
        `/tmp/native_4090_state_scan_sk_output_sweep_20260703_005904.jsonl`;
      - confirmation:
        `bench/results_native_4090_state_scan_sk_output_confirm_20260703_010145.jsonl`
        from remote
        `/tmp/native_4090_state_scan_sk_output_confirm_20260703_010145.jsonl`.
    - 4090 / 0.4B / prompt512 / bsz1 rows:
      - smoke same-run baseline: `25,019.9 tok/s`, `20.4637 ms`;
      - smoke sk-scale route: `26,187.6 tok/s`, `19.5512 ms`;
      - sweep baseline: `25,051.9 tok/s`;
      - sweep best sk-scale row `num_warps=8,num_stages=3`:
        `26,702.7 tok/s`, `19.1741 ms`;
      - confirmation baseline: `24,757.5 tok/s`, `20.6806 ms`;
      - confirmation sk-scale `num_warps=8,num_stages=3`:
        `25,785.6 tok/s`, `19.8560 ms`.
    - conclusion: the sk-scale route is the first no-K/V-derived end-to-end
      route in this sequence that repeatedly beats its same-run baseline, but
      it still does not beat the strict historical best row `27,051.0 tok/s`
      and remains below the `0.60x` target.  Keep it opt-in for now.
- [x] Next main fused-fp16 task:
  - Use the positive sk-scale route as the new bounded branch, but do not
    promote it yet.  Next experiment should profile or micro-split the sk path
    to locate the remaining overhead: added `sk` reduction inside scan, raw-V
    interpolation in output prep, or launch/memory traffic between scan and
    output prep.  Promotion gate remains same-run 4090 / 0.4B / prompt512 /
    bsz1 correctness plus movement beyond the strict `27,051.0 tok/s` row and
    toward `>=31,289 tok/s`.
  - Done: added and ran a synthetic state-scan/output micro-split benchmark.
    - implementation:
      - new benchmark `bench/bench_state_scan_output_micro.py` times scan-only,
        output-only, and scan+output route pairs for full K/V-writeback,
        no-K/V raw recompute, sk-scale raw-V, and full-correction variants;
      - analyzer now preserves `state_scan_output_micro` rows and adds their
        component/delta summary to `next_focus`.
    - result file:
      - `bench/results_state_scan_output_micro_4090_20260703_011009.jsonl`
        from remote `/tmp/state_scan_output_micro_4090_20260703_011009.jsonl`.
    - 4090 synthetic `B=1,T=512,H=16,N=64,fp16`, `num_warps=8`,
      `num_stages=3`, all correctness checks pass within fp16 tolerance
      (`out/state` diffs `0.0`; output-prep diffs up to `0.125`):
      - scan-only:
        - full K/V-writeback: `0.515968 ms`;
        - no-K/V: `0.455680 ms`;
        - sk-scale: `0.502784 ms`;
        - full correction-vector: `0.574464 ms`;
      - output-only:
        - full K/V output prep: `0.080896 ms`;
        - raw-K/V recompute output prep: `0.099328 ms`;
        - sk-scale raw-V output prep: `0.081856 ms`;
        - correction-vector output prep: `0.067680 ms`;
      - route totals:
        - full K/V baseline route: `0.521216 ms`;
        - no-K/V raw route: `0.462848 ms`;
        - sk-scale route: `0.509952 ms`;
        - correction-vector route: `0.580608 ms`.
    - conclusion: the sk-scale route's remaining overhead is primarily inside
      the scan kernel, not output prep.  `scan_sk - scan_nokv` is about
      `0.047104 ms`, while `output_sk_raw_v` is only `0.000960 ms` slower
      than full K/V output prep.  Writing a full correction vector is clearly
      negative.  The next bounded experiment should reduce or move the sk
      reduction cost, or revisit the no-K/V raw route with a full-HF profiler
      because synthetic route timing favors it even though the earlier HF raw
      route was only parity/slower.
- [x] Next main fused-fp16 task:
  - Do not promote sk-scale yet.  Use the micro-split evidence to target the
    scan-side `sk` reduction overhead directly.  Candidate bounded directions:
    - add a scan phase/probe or kernel variant that computes `sk` with lower
      register/reduction pressure, then re-test the sk-scale HF route; or
    - run a corrected HF breakdown for sk-scale and no-K/V raw routes to
      resolve why synthetic `route_nokv_raw` is much faster while prior HF raw
      was not.
  - Same-run gate remains 4090 / 0.4B / prompt512 / bsz1 correctness plus a
    confirmed row beyond the strict `27,051.0 tok/s` before default promotion.
  - Done: added corrected HF profiler support for the sk-scale route, ran the
    raw/sk corrected breakdown, and tried the narrow sk no-mask plus
    sk+WAVG-LoRA combination experiments.
    - implementation:
      - `bench/bench_native_prefill_breakdown.py` now routes
        `RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_SK_OUTPUT=1` through
        `fused_recurrent_scan_state_prep_sk(...)` and
        `fused_attn_output_prepare_from_sk_raw_v(...)`, with profiler
        components `recurrent_scan_state_prep_sk_fused` and
        `attn_output_prep_sk_raw_v_fused`;
      - added an opt-in N=64 no-mask sk kernel under the existing
        `RWKV7_NATIVE_PREFILL_SCAN_NOMASK64=1` knob for the sk-scale route;
      - default HF/native behavior remains unchanged unless the benchmark env
        enables these experimental flags.
    - corrected HF breakdown result file:
      `bench/results_native_4090_sk_raw_breakdown_20260703_011513.jsonl`
      from remote `/tmp/native_4090_sk_raw_breakdown_20260703_011513.jsonl`.
    - 4090 breakdown rows, all pass with `max_abs_diff_vs_native_prefill=0.0`:
      - baseline full K/V scan+output: profiled total `29.4134 ms`,
        `17,407.0 tok/s`; scan component
        `recurrent_scan_state_prep_fused=13.2268 ms`, output prep
        `0.1875 ms`;
      - raw no-K/V output route: profiled total `29.5475 ms`,
        `17,328.0 tok/s`; scan component
        `recurrent_scan_state_prep_nokv_fused=11.9508 ms`, output prep
        `0.1906 ms`;
      - sk-scale route: profiled total `28.7642 ms`, `17,799.9 tok/s`;
        scan component `recurrent_scan_state_prep_sk_fused=13.3087 ms`,
        output prep `0.1874 ms`.
    - end-to-end sk/no-mask result file:
      `bench/results_native_4090_sk_nomask64_sweep_20260703_012451.jsonl`
      from remote `/tmp/native_4090_sk_nomask64_sweep_20260703_012451.jsonl`.
      Rows all pass greedy/cache/decode smoke:
      - baseline: `25,309.6 tok/s`, `20.2295 ms`;
      - raw no-K/V output: `24,574.1 tok/s`, `20.8349 ms`;
      - sk-scale: `25,812.9 tok/s`, `19.8350 ms`;
      - sk-scale + no-mask N64: `25,314.8 tok/s`, `20.2253 ms`.
    - combination result file:
      `bench/results_native_4090_sk_wavg_combo_20260703_012714.jsonl`
      from remote `/tmp/native_4090_sk_wavg_combo_20260703_012714.jsonl`.
      Rows pass greedy/cache/decode smoke:
      - tuned WAVG-LoRA only: `25,457.9 tok/s`, `20.1117 ms`;
      - sk-scale + tuned WAVG-LoRA: `26,375.2 tok/s`, `19.4122 ms`.
    - conclusion:
      - corrected HF profiling confirms the raw no-K/V scan kernel is faster
        inside the scan (`~1.276 ms` lower than full K/V), but that saving
        still does not transfer to e2e because the downstream raw route loses
        it back and is slower in confirmed prefill rows;
      - sk-scale remains the best no-K/V-derived route and combines cleanly
        with tuned WAVG-LoRA, but the best new combined row still stays below
        the historical strict `27,051.0 tok/s` and far below the
        `31,289 tok/s` 0.60x target;
      - the sk no-mask N64 variant is negative on 4090, so do not promote it.
- [x] Next main fused-fp16 task:
  - Stop repeating shallow raw/sk/no-mask/WAVG combinations unless a new
    component-level reason appears.  The next bounded experiment must attack a
    larger remaining cost boundary:
    - either a real fused layer-prep boundary that reduces the aggregate
      norm/shift + LoRA/projection launch/memory cost without replacing
      cuBLAS dense matmuls with slower Triton matmul; or
    - a deeper CUDA/persistent state-scan schedule that shares vector prep
      without global temp tensors and still preserves row-level parallelism.
  - Promotion gate remains unchanged: same-run 4090 / 0.4B / prompt512 / bsz1
    correctness plus a confirmed row beyond `27,051.0 tok/s`, moving toward
    `>=31,289 tok/s`.
  - Done: tried the deeper CUDA/persistent-side branch by adding a
    two-worker-warp row-block schedule that shares vector prep across multiple
    rows while preserving the two-warp-per-row state/recurrent reductions.
    - implementation:
      - added `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE=warp2` to the
        opt-in CUDA N=64 state-scan scaffold;
      - `warp2` uses one producer warp per CTA to compute shared R/W/K/A/KK
        prep and two worker warps per row, with `ROWS_PER_BLOCK` variants
        `1/2/4/8`;
      - `bench/bench_cuda_state_scan_micro.py` now records the `warp2`
        schedule rows; default HF/native behavior remains unchanged unless the
        CUDA state-scan env flags are set.
    - micro result file:
      `bench/results_cuda_state_scan_warp2_micro_4090_20260703_013407.jsonl`
      from remote `/tmp/cuda_state_scan_warp2_micro_4090_20260703_013407.jsonl`.
      4090 synthetic `B=1,T=512,H=16,N=64,fp16`:
      - default row-block `rpb=1`: `0.518592 ms`;
      - prior warp-specialized `rpb=1`: `0.449088 ms`;
      - prior warp-specialized `rpb=8`: `0.460800 ms`;
      - new `warp2 rpb=1`: `0.498224 ms`;
      - new `warp2 rpb=2`: `0.472064 ms`;
      - new `warp2 rpb=4`: `0.480256 ms`;
      - new `warp2 rpb=8`: `0.502784 ms`.
    - HF confirmation result file:
      `bench/results_native_4090_cuda_warp2_confirm_20260703_013526.jsonl`
      from remote `/tmp/native_4090_cuda_warp2_confirm_20260703_013526.jsonl`.
      Rows all pass greedy/cache/decode smoke:
      - Triton baseline: `25,297.9 tok/s`, `20.2388 ms`;
      - CUDA row-block `rpb=1`: `26,076.1 tok/s`, `19.6348 ms`;
      - CUDA warp-specialized `rpb=1`: `26,186.4 tok/s`, `19.5522 ms`;
      - CUDA `warp2 rpb=2`: `25,882.8 tok/s`, `19.7815 ms`.
    - Combination result file:
      `bench/results_native_4090_cuda_wavg_combo_20260703_013719.jsonl`
      from remote `/tmp/native_4090_cuda_wavg_combo_20260703_013719.jsonl`.
      Rows all pass greedy/cache/decode smoke:
      - Triton baseline: `25,733.8 tok/s`, `19.8960 ms`;
      - tuned WAVG-LoRA only: `25,428.5 tok/s`, `20.1349 ms`;
      - CUDA row-block + tuned WAVG-LoRA: `25,308.0 tok/s`,
        `20.2307 ms`;
      - CUDA warp-specialized + tuned WAVG-LoRA: `26,404.8 tok/s`,
        `19.3904 ms`.
    - conclusion:
      - `warp2` is correctness-safe and improves on the default row-block
        micro, but it does not beat the existing one-worker-warp
        warp-specialized CUDA schedule in micro or HF;
      - CUDA warp-specialized + tuned WAVG-LoRA is the best same-run
        combination in this experiment, but still below the strict historical
        `27,051.0 tok/s` row and below the `31,289 tok/s` stretch target;
      - keep `warp2` opt-in and do not promote it.
- [x] Next main fused-fp16 task:
  - Move to the other larger-boundary option: implement or probe a real
    layer-prep fusion that reduces norm/shift + LoRA/projection launch and
    memory cost while preserving cuBLAS for the big dense R/K/V/O matmuls.
    The bounded experiment should compare same-run:
    - current full-head Triton state-scan baseline;
    - best opt-in CUDA state-scan/WAVG combination;
    - the new layer-prep boundary.
  - Promotion gate remains 4090 / 0.4B / prompt512 / bsz1 correctness plus a
    confirmed row beyond `27,051.0 tok/s`, moving toward `>=31,289 tok/s`.
  - Done: added an opt-in prefill attention norm+shift/time-mix boundary:
    `RWKV7_NATIVE_PREFILL_FUSED_NORM_MIX=1`.
    - implementation:
      - new Triton helper `fused_attn_norm_shift_mix_prefill(...)` computes
        optional pre-attention layernorm, attention layernorm, previous-token
        `h`, and the six RWKV time-mix tensors in one kernel;
      - native prefill and the profiler route through it only when the env
        flag is enabled;
      - telemetry/analyzer now record
        `prefill_fused_norm_mix_requested/effective`;
      - dense R/K/V/O matmuls remain cuBLAS-backed; default HF/native behavior
        is unchanged.
    - correctness:
      - 4090 random CUDA oracle passed for both `has_pre_norm=False` and
        `has_pre_norm=True` with max diffs within fp16 tolerance;
      - all HF rows below pass greedy/cache/decode smoke.
    - result files:
      - e2e:
        `bench/results_native_4090_norm_mix_layerprep_20260703_014616.jsonl`
        from remote
        `/tmp/native_4090_norm_mix_layerprep_20260703_014616.jsonl`;
      - fine-attention breakdown:
        `bench/results_native_4090_norm_mix_breakdown_20260703_014616.jsonl`
        from remote
        `/tmp/native_4090_norm_mix_breakdown_20260703_014616.jsonl`.
    - 4090 / 0.4B / prompt512 / bsz1 e2e rows:
      - current Triton baseline: pass, `24,821.1 tok/s`, `20.6276 ms`,
        peak `989.2 MiB`;
      - best CUDA state-scan + tuned WAVG comparison:
        pass, `26,256.9 tok/s`, `19.4997 ms`;
      - fused norm-mix only: pass, `24,311.1 tok/s`, `21.0603 ms`;
      - fused norm-mix + WAVG: pass, `25,719.4 tok/s`, `19.9072 ms`;
      - fused norm-mix + CUDA warp-specialized state-scan + WAVG:
        pass, `25,552.1 tok/s`, `20.0375 ms`.
    - breakdown signal:
      - `attn_norm_shift_mix` drops from `1.2575 ms` to `0.6043 ms`;
      - profiled total is roughly flat/slightly better
        (`29.6110 ms` -> `29.5373 ms`), but the e2e benchmark is slower and
        the combined rows do not beat the same-run CUDA/WAVG comparison.
    - conclusion: the standalone recompute-prev norm+mix boundary is
      correctness-safe and reduces the targeted profiled component, but it
      does not improve end-to-end 4090 throughput.  The extra launch/traffic
      and previous-token norm recompute erase the saved Python/PyTorch
      operations. Keep it opt-in and do not promote it; it does not beat the
      strict historical `27,051.0 tok/s` row or the `31,289 tok/s` stretch.
- [x] Next main fused-fp16 task:
  - Do not spend another iteration on standalone recompute-prev norm-mix.  The
    next bounded performance experiment should either:
    - make layer-prep larger without previous-token recompute, e.g.
      compute/cache `h` once then fuse time-mix with the surrounding LoRA
      inputs while keeping big dense projections on cuBLAS; or
    - return to a deeper CUDA/persistent state-scan schedule that shares
      per-token vectors without global temp tensors and without losing
      row-level parallelism.
  - Same-run gate remains 4090 / 0.4B / prompt512 / bsz1 correctness plus a
    confirmed row beyond `27,051.0 tok/s`, moving toward `>=31,289 tok/s`.
  - Done: added the no-prev-recompute layer-prep route
    `RWKV7_NATIVE_PREFILL_FUSED_SHIFT_WAVG_LORA=1`.
    - implementation:
      - new `fused_shift_wavg_lora(...)` keeps the regular cached `h` and
        `prev_h` path, materializes only `xr/xk/xv` for the large cuBLAS
        R/K/V projections, and computes W/A/G/V-gate LoRA from on-the-fly
        time-mixed vectors inside the fused Triton down kernel;
      - this avoids the negative recompute-prev layernorm from the previous
        norm-mix experiment and avoids writing/loading `xw/xa/xg`;
      - native prefill, scan telemetry, analyzer keys, and profiler telemetry
        now record `prefill_fused_shift_wavg_lora_*`;
      - default HF/native behavior stays unchanged unless the env flag is set.
    - correctness:
      - remote CUDA oracle passed against the torch fallback for random fp16
        tensors;
      - all HF rows below pass greedy/cache/decode smoke with `max_abs_diff`
        `0.0625` or better.
    - result files:
      - initial e2e:
        `bench/results_native_4090_shift_wavg_layerprep_20260703_021100.jsonl`
        from remote
        `/tmp/native_4090_shift_wavg_layerprep_20260703_021100.jsonl`;
      - block sweep:
        `bench/results_native_4090_shift_wavg_blocks_20260703_021900.jsonl`
        and
        `bench/results_native_4090_shift_wavg_blocks2_20260703_022600.jsonl`;
      - confirmation:
        `bench/results_native_4090_shift_wavg_confirm_20260703_023200.jsonl`
        from remote
        `/tmp/native_4090_shift_wavg_confirm_20260703_023200.jsonl`;
      - default-block smoke after making `128/64/64` the route default:
        `bench/results_native_4090_shift_wavg_default_smoke_20260703_024600.jsonl`
        from remote
        `/tmp/native_4090_shift_wavg_default_smoke_20260703_024600.jsonl`;
      - refreshed breakdown:
        `bench/results_native_4090_shift_wavg_breakdown_20260703_023900.jsonl`
        from remote
        `/tmp/native_4090_shift_wavg_breakdown_20260703_023900.jsonl`.
    - 4090 / 0.4B / prompt512 / bsz1 e2e rows:
      - same-run baseline: `25,438.5 tok/s`, `20.1270 ms`;
      - tuned WAVG only: `25,718.3 tok/s`, `19.9080 ms`;
      - shift-WAVG only: `26,823.7 tok/s`, `19.0876 ms`;
      - CUDA warp-specialized state-scan + tuned WAVG:
        `26,092.5 tok/s`, `19.6225 ms`;
      - CUDA warp-specialized state-scan + shift-WAVG:
        `28,071.9 tok/s`, `18.2389 ms`.
    - block sweep best:
      - `block_m=128,block_r=64,block_k=64` reached `28,564.7 tok/s`,
        `17.9242 ms`; nearby `bk128`, `br128`, and smaller/larger tiles were
        slower. This tile is now the shift-WAVG route default, while the old
        standalone WAVG-LoRA defaults stay unchanged.
    - confirmation row:
      - baseline: `25,180.6 tok/s`, `20.3331 ms`;
      - CUDA warp-specialized + old tuned WAVG: `25,781.7 tok/s`,
        `19.8591 ms`;
      - CUDA warp-specialized + shift-WAVG `bm128/br64/bk64`:
        `28,780.6 tok/s`, `17.7898 ms`, peak `988.2 MiB`, about `0.5519x`
        Albatross and `+6.4%` over the previous strict best
        `27,051.0 tok/s`.
    - breakdown signal for the new best route:
      - `recurrent_scan_state_prep_cuda`: `11.3919 ms`, `47.51%`;
      - `attn_shift_wavg_lora_fused`: `5.4047 ms`, `22.54%`;
      - `ffn`: `2.6432 ms`, `11.02%`;
      - the layer-prep fusion is now the largest non-scan component, so the
        next stretch step needs either split/tune that fused kernel or return
        to state-scan/FFN fusion.
- [x] Split/profile the current shift-WAVG layer-prep route before spending
  more time on blind tuning:
  - Added `bench/bench_shift_wavg_lora_micro.py` and exposed
    `RWKV7_NATIVE_PREFILL_FUSED_SHIFT_WAVG_LORA_DOWN_WARPS` /
    `RWKV7_NATIVE_PREFILL_FUSED_SHIFT_WAVG_LORA_UP_WARPS` so this fused
    LoRA/time-mix section can be measured and tuned separately.
  - Micro split on 4090 / layer1 / `T=512,H=1024` / `bm128/br64/bk64`:
    - default `down_warps=4,up_warps=4`: full `0.196608 ms`, down
      `0.083968 ms`, up `0.088064 ms`, `max_abs_diff_vs_fallback=0.0625`;
    - sweep best also stayed at `4/4` (`0.196608 ms`); larger warp counts hurt
      the micro-kernel, especially `down_warps=8`.
  - E2E warp sweep on 4090 / 0.4B / prompt512 / bsz1 / CUDA state-scan +
    shift-WAVG showed no new strict best:
    - `dw8/uw4`: `28,195.8 tok/s`, `18.1587 ms`;
    - `dw4/uw4`: `28,059.5 tok/s`, `18.2469 ms`;
    - `dw4/uw2`: `27,961.9 tok/s`, `18.3106 ms`;
    - `dw2/uw4`: `27,920.8 tok/s`, `18.3376 ms`;
    - `dw4/uw8`: `27,795.8 tok/s`, `18.4200 ms`.
  - Conclusion: down/up work is roughly balanced and simple warp retuning does
    not close the remaining Albatross gap. Keep `4/4` as the portable default
    unless a full same-run sweep proves otherwise.
- [x] Try CUDA state-scan SK/no-KV-writeback as the first structural
  state-scan/output experiment:
  - Added opt-in `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SK=1`, implemented a
    CUDA warp-specialized SK-emitting state-scan path, paired it with the
    existing `fused_attn_output_prepare_from_sk_raw_v`, and added telemetry plus
    micro rows.  The route avoids writing full adjusted K/V from the CUDA scan
    and emits only per-token/head `sk=sum(r*k_adj*r_k)`.
  - Micro result on 4090 / synthetic `B=1,T=512,H=16,N=64`:
    - existing full warp-specialized rpb1: `0.450560 ms`;
    - SK rpb1/rpb2/rpb4/rpb8: `0.508928 / 0.497664 / 0.504832 /
      0.510976 ms`;
    - best SK micro is rpb2, but it is still slower than existing full K/V
      writeback, so the extra SK reduction outweighs removed K/V writes at this
      shape.
  - E2E confirm with current shift-WAVG route on 4090 / 0.4B / prompt512 / bsz1:
    - same-run baseline CUDA state-scan + shift-WAVG: `27,815.6 tok/s`,
      `18.4069 ms`, max diff `0.0625`;
    - best CUDA-SK rpb2: `27,354.5 tok/s`, `18.7172 ms`, max diff `0.125`;
    - rpb4/rpb1/rpb8: `27,110.2 / 27,022.5 / 26,915.2 tok/s`.
  - Conclusion: keep the CUDA-SK path as telemetry-only / opt-in; do not promote
    it to the main route. It does not beat the previous strict best
    `28,780.6 tok/s`, and it also loses to the same-run baseline.
- [x] Try cheaper cached vector-prep for the CUDA row-block scan:
  - Added opt-in `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_PRECOMPUTE_MODE=wk_half`
    to precompute only W-decay and normalized KK into fp16 temporaries, while
    reusing the existing fp16 adjusted K/V outputs.  This tests the
    split-vector-prep branch with less temporary bandwidth than the old fp32
    `full` / `wk` precompute modes.
  - Micro result on 4090 / synthetic `B=1,T=512,H=16,N=64`:
    - current warp-specialized rpb1: `0.449536 ms`;
    - precompute `full`: `0.334848 ms`;
    - precompute `wk`: `0.330752 ms`;
    - new precompute `wk_half`: `0.326656 ms` (best micro).
  - E2E confirm with current shift-WAVG route on 4090 / 0.4B / prompt512 / bsz1:
    - same-run baseline warp-specialized CUDA state-scan + shift-WAVG:
      `28,288.0 tok/s`, `18.0995 ms`, max diff `0.0625`;
    - `wk_half`: `27,814.5 tok/s`, `18.4077 ms`, max diff `0.125`;
    - `full`: `27,743.6 tok/s`, `18.4547 ms`, max diff `0.125`;
    - `wk`: `26,758.2 tok/s`, `19.1343 ms`, max diff `0.0625`.
  - Conclusion: cached vector-prep is a real micro-kernel win, and `wk_half` is
    the best cached form, but the extra launch / temp traffic still loses e2e
    to warp-specialized in the current HF prefill route. Keep `wk_half` as an
    opt-in probe only.
- [x] Try wider single-launch vector-prep sharing inside CUDA state-scan:
  - Added `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_ROWS_PER_BLOCK=16` for the
    single-launch row-block CUDA paths.  The warp-specialized rpb16 path uses
    one producer warp for 16 row-worker warps, so it reuses one vector-prep /
    norm result across more rows without adding the cached-prep extra launch.
  - Micro result on 4090 / synthetic `B=1,T=512,H=16,N=64`:
    - warp-specialized rpb1: `0.447488 ms`;
    - warp-specialized rpb8: `0.458752 ms`;
    - new warp-specialized rpb16: `0.519168 ms`;
    - default/coop rpb16: `0.826368 ms`;
    - SK rpb16: `0.570368 ms`.
  - E2E confirm with current shift-WAVG route on 4090 / 0.4B / prompt512 / bsz1:
    - same-run rpb1: `28,349.5 tok/s`, `18.0603 ms`, max diff `0.0625`;
    - rpb8: `28,312.8 tok/s`, `18.0837 ms`, max diff `0.0625`;
    - rpb16: `26,303.0 tok/s`, `19.4655 ms`, max diff `0.0625`.
  - Conclusion: sharing vector prep across 16 rows in one launch over-reduces
    row parallelism / occupancy.  rpb8 is near rpb1, but neither beats the
    previous strict best `28,780.6 tok/s`. Keep rpb16 as an opt-in probe only;
    do not promote.
- [x] Try a bounded FFN activation boundary after state-scan sharing variants lost:
  - Added opt-in `RWKV7_NATIVE_PREFILL_FFN_FUSED_ACT=1` for the prefill FFN
    middle activation, keeping both large FFN GEMMs on cuBLAS.  Modes:
    - `RWKV7_NATIVE_PREFILL_FFN_FUSED_ACT_MODE=triton`: one Triton kernel for
      in-place `relu(x)^2`;
    - `RWKV7_NATIVE_PREFILL_FFN_FUSED_ACT_MODE=torch_inplace`: PyTorch in-place
      relu plus square, avoiding extra allocation but not reducing launches.
  - Correctness:
    - 4090 fp16/bf16/fp32 activation oracle passed with max diff `0.0`;
    - all HF rows below pass greedy/cache/decode smoke with max diff `0.0625`.
  - Result files:
    - `bench/results_native_4090_ffn_fused_act_20260703_112242.jsonl` from
      remote `/tmp/native_4090_ffn_fused_act_20260703_112242.jsonl`;
    - `bench/results_native_4090_ffn_act_torch_inplace_20260703_112706.jsonl`
      from remote `/tmp/native_4090_ffn_act_torch_inplace_20260703_112706.jsonl`.
  - 4090 / 0.4B / prompt512 / bsz1, current CUDA state-scan + shift-WAVG route:
    - same-run baseline: `27,794.6 tok/s`, `18.4208 ms`;
    - Triton activation block512: `26,331.2 tok/s`, `19.4446 ms`;
    - Triton activation block1024: `26,657.4 tok/s`, `19.2067 ms`;
    - Triton activation block2048: `25,905.3 tok/s`, `19.7643 ms`;
    - Triton activation block4096: `26,990.0 tok/s`, `18.9700 ms`;
    - second same-run baseline: `28,232.2 tok/s`, `18.1353 ms`;
    - PyTorch in-place activation: `27,115.8 tok/s`, `18.8820 ms`.
  - Conclusion: this small FFN activation boundary is correctness-safe but
    negative end-to-end.  Do not promote it.  The remaining FFN opportunity, if
    any, must be a larger boundary than standalone activation; otherwise return
    to persistent/two-level state-scan scheduling.
- [x] Try a larger FFN norm+shift boundary without replacing cuBLAS GEMMs:
  - Added opt-in `RWKV7_NATIVE_PREFILL_FFN_FUSED_NORM_SHIFT=1` for the prefill
    FFN key-input boundary.  It fuses FFN layernorm, previous-token alignment,
    and `fx_k` shift/mix into one Triton kernel, returns the shifted key input
    plus the final FFN cache, and keeps both large FFN GEMMs on cuBLAS.
  - Correctness:
    - 4090 synthetic oracle passed for fp16/bf16/fp32.  Max diffs:
      fp16 `0.015625`, bf16 `0.125`, fp32 `2.86e-06`;
    - all HF rows below pass greedy/cache/decode smoke, with max diff
      `0.0625` for `block_h=1024` and `0.125` for `block_h=2048`.
  - Result files:
    - `bench/results_native_4090_ffn_norm_shift_20260703_113515.jsonl` from
      remote `/tmp/native_4090_ffn_norm_shift_20260703_113515.jsonl`;
    - `bench/results_native_4090_ffn_norm_shift_confirm_20260703_113639.jsonl`
      from remote `/tmp/native_4090_ffn_norm_shift_confirm_20260703_113639.jsonl`.
  - 4090 / 0.4B / prompt512 / bsz1, current CUDA state-scan + shift-WAVG route:
    - sweep same-run baseline: `27,643.3 tok/s`, `18.5217 ms`, peak
      `988.2 MiB`;
    - FFN norm+shift `block_h=1024`: `27,676.5 tok/s`, `18.4995 ms`, peak
      `964.2 MiB`;
    - FFN norm+shift `block_h=2048`: `27,127.6 tok/s`, `18.8738 ms`, peak
      `964.2 MiB`;
    - confirm baseline #1: `27,247.0 tok/s`, `18.7911 ms`, peak
      `988.2 MiB`;
    - confirm `block_h=1024`: `27,708.6 tok/s`, `18.4781 ms`, peak
      `964.2 MiB`;
    - confirm baseline #2: `26,749.1 tok/s`, `19.1409 ms`, peak
      `988.2 MiB`.
  - Conclusion: `block_h=1024` is correctness-safe, consistently lowers peak
    memory by about `24 MiB`, and can beat same-run baselines modestly.  It
    still does not beat the strict historical best `28,780.6 tok/s`, so keep it
    opt-in for now rather than default-promoting.  The next performance push
    should return to the larger remaining state-scan schedule gap.
- [x] Try shift-WAVG-emitted W-decay for the CUDA state-scan boundary:
  - Motivation: the previous CUDA micro rows showed vector prep / K-normalization
    as the biggest single scan-side cost, and the current best route already
    runs `fused_shift_wavg_lora(...)` before the CUDA scan.  This experiment
    moves `exp(-0.606531 * sigmoid(w_raw))` into the shift-WAVG LoRA up kernel
    and lets the warp-specialized CUDA scan consume precomputed W decay in the
    same single-launch row-block schedule.
  - Implementation:
    - `fused_shift_wavg_lora(..., output_w_decay=True)` can now emit W-decay
      instead of raw W while preserving the default raw-W behavior;
    - `cuda_state_scan_prep(..., w_precomputed=True)` dispatches the narrow
      fp16 / `head_dim=64` / `lanes=64` / `warp_specialized` path with W
      already in decay form;
    - native prefill wires this only when
      `RWKV7_NATIVE_PREFILL_FUSED_SHIFT_WAVG_LORA_W_DECAY=1` and the CUDA
      warp-specialized scan route is active; default HF/native behavior remains
      unchanged.
  - Correctness:
    - 4090 synthetic CUDA oracle passed; baseline raw-W scan vs precomputed-W
      scan diffs were output `0.0625`, state `0.0031514`, adjusted K `0.0`,
      adjusted V `0.0`;
    - all HF rows below pass greedy/cache/decode smoke.
  - Result files:
    - micro:
      `bench/results_cuda_state_scan_wpre_micro_4090_20260703_034911.jsonl`
      from remote
      `/tmp/cuda_state_scan_wpre_micro_4090_20260703_034911.jsonl`;
    - HF e2e:
      `bench/results_native_4090_shift_wdecay_20260703_034945.jsonl`
      from remote `/tmp/native_4090_shift_wdecay_20260703_034945.jsonl`.
  - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
    - full warp-specialized rpb1: `0.447488 ms`, `1,144,164.7 tok/s`;
    - W-precomputed warp-specialized rpb1: `0.446464 ms`,
      `1,146,789.0 tok/s`;
    - full warp-specialized rpb8: `0.458752 ms`, `1,116,071.4 tok/s`;
    - W-precomputed warp-specialized rpb8: `0.456704 ms`,
      `1,121,076.3 tok/s`.
  - 4090 / 0.4B / prompt512 / bsz1 HF rows:
    - same-run baseline: `26,857.2 tok/s`, `19.0638 ms`, peak
      `988.2 MiB`;
    - shift-WAVG W-decay rpb1: `26,329.2 tok/s`, `19.4461 ms`, peak
      `988.2 MiB`;
    - shift-WAVG W-decay rpb8: `26,858.8 tok/s`, `19.0627 ms`, peak
      `988.2 MiB`;
    - shift-WAVG W-decay + FFN norm-shift: `27,147.5 tok/s`,
      `18.8599 ms`, peak `964.2 MiB`.
  - Conclusion: precomputing W-decay inside shift-WAVG is correctness-safe and
    gives a tiny scan-kernel micro win (`~0.2-0.4%`), but it does not transfer
    into a meaningful HF e2e gain and remains below the strict best
    `28,780.6 tok/s`.  Keep it opt-in; do not promote.  The useful signal is
    that removing only W sigmoid/exp is too small once the cost is paid in the
    shift-WAVG boundary; the remaining `0.60x` gap still needs a larger
    persistent/two-level scan schedule or a bigger fused boundary.
- [x] Try head-level register-state CUDA schedule for two-level sharing:
  - Motivation: test a real alternative to row-block producer duplication.
    `head_reg16` launches one CTA per `(batch, head)`, uses 1024 threads, maps
    16 lanes to each state row, and keeps four state columns per thread in
    registers.  R/W/K/A/normalized-KK/adjusted-KV are computed once per
    token/head in shared memory, then half-warp row workers perform
    state-dot/update/recurrent output.  This directly tests whether head-level
    vector sharing can beat duplicated row-block prep without global temp
    tensors.
  - Implementation:
    - added `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE=head_reg16`;
    - added `rwkv7_state_scan_prep_n64_head_reg16_kernel`;
    - kept the route opt-in and left default HF/native behavior unchanged;
    - analyzer now preserves the W-precomputed and shift-WAVG W-decay
      telemetry fields.
  - Correctness:
    - 4090 synthetic oracle vs warp-specialized row-block passed with diffs
      `[0.0004883, 1.907e-06, 0.0, 0.0]` for output/state/K/V;
    - HF e2e row passed greedy/cache/decode smoke with `max_abs_diff=0.0625`.
  - Result files:
    - micro: `bench/results_cuda_state_scan_head_reg16_micro_4090_20260703.jsonl`
      from remote `/tmp/cuda_state_scan_head_reg16_micro_4090_.jsonl`;
    - HF e2e: `bench/results_native_4090_head_reg16_20260703.jsonl`
      from remote `/tmp/native_4090_head_reg16_20260703.jsonl`.
  - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
    - warp-specialized rpb1: `0.447488 ms`, `1,144,164.7 tok/s`;
    - warp-specialized rpb8: `0.459776 ms`, `1,113,585.7 tok/s`;
    - `head_reg16`: `0.688128 ms`, `744,047.6 tok/s`.
  - 4090 / 0.4B / prompt512 / bsz1 HF rows:
    - same-run warp-specialized + shift-WAVG baseline: `26,879.8 tok/s`,
      `19.0478 ms`;
    - `head_reg16` + shift-WAVG: `21,563.7 tok/s`, `23.7436 ms`;
    - both rows peak at `988.2 MiB`.
  - Conclusion: `head_reg16` is correctness-safe but clearly slower in both
    micro and HF.  Full head-level sharing over-reduces CTA parallelism and
    makes a 1024-thread CTA too heavy for this shape.  Keep the route opt-in
    as a negative schedule probe; do not promote.
- [x] Try double-buffered warp-pipelined CUDA row-block schedule:
  - Motivation: keep the proven warp-specialized row-block occupancy while
    reducing per-token synchronization.  The new `warp_pipelined` schedule
    double-buffers shared R/W/K/A/normalized-KK/V-row vectors: producer warp
    prefetches token `t+1` while row-worker warps consume token `t`, reducing
    the row-block inner loop from two `__syncthreads()` per token to one after
    the initial preload.
  - Implementation:
    - added `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE=warp_pipelined`;
    - added `rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel`;
    - supported rpb `1/2/4/8/16`;
    - kept the route opt-in and left default HF/native behavior unchanged.
  - Correctness:
    - 4090 synthetic oracle vs warp-specialized rpb1 passed exactly for
      rpb `1/2/4/8/16`, with output/state/K/V diffs all `0.0`;
    - all HF rows below pass greedy/cache/decode smoke with `max_abs_diff=0.0625`.
  - Result files:
    - micro:
      `bench/results_cuda_state_scan_warppipe_micro_4090_20260703_121500.jsonl`
      from remote
      `/tmp/cuda_state_scan_warppipe_micro_4090_20260703_121500.jsonl`;
    - HF sweep:
      `bench/results_native_4090_warppipe_sweep_20260703_122000.jsonl`
      from remote `/tmp/native_4090_warppipe_sweep_20260703_122000.jsonl`;
    - HF confirm:
      `bench/results_native_4090_warppipe_confirm_20260703_123000.jsonl`
      from remote `/tmp/native_4090_warppipe_confirm_20260703_123000.jsonl`.
  - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
    - warp-specialized rpb1: `0.448512 ms`, `1,141,552.5 tok/s`;
    - warp-specialized rpb8: `0.460800 ms`, `1,111,111.1 tok/s`;
    - warp-pipelined rpb1: `0.379904 ms`, `1,347,708.9 tok/s`;
    - warp-pipelined rpb8: `0.347200 ms`, `1,474,654.4 tok/s`;
    - warp-pipelined rpb16: `0.369664 ms`, `1,385,041.5 tok/s`.
  - 4090 / 0.4B / prompt512 / bsz1 HF sweep rows:
    - same-run warp-specialized rpb1 + shift-WAVG baseline:
      `27,223.1 tok/s`, `18.8076 ms`;
    - warp-pipelined rpb1: `26,920.9 tok/s`, `19.0187 ms`;
    - warp-pipelined rpb8: `27,079.5 tok/s`, `18.9073 ms`;
    - warp-pipelined rpb16: `27,739.6 tok/s`, `18.4574 ms`.
  - 4090 HF confirmation rows:
    - baseline #1: `27,591.2 tok/s`, `18.5566 ms`;
    - warp-pipelined rpb16: `27,033.4 tok/s`, `18.9395 ms`;
    - warp-pipelined rpb16 + FFN norm-shift: `27,705.4 tok/s`,
      `18.4802 ms`, peak `964.2 MiB`;
    - baseline #2: `27,634.9 tok/s`, `18.5273 ms`.
  - Conclusion: the double-buffered schedule is the strongest CUDA micro win so
    far and proves sync reduction is real inside the isolated scan kernel, but
    the HF confirmation does not beat the strict best `28,780.6 tok/s` and only
    shows same-run parity/slight wins when combined with FFN norm-shift.  Keep
    it opt-in; do not promote.  The next step should profile the HF path with
    warppipe enabled to see whether the scan component improves but total time
    is hidden by launch/model overhead, or whether the micro win is lost inside
    the real layer mix.
- [x] Run corrected HF breakdown for the warp-pipelined route:
  - Compared baseline warp-specialized rpb1, warp-pipelined rpb16, and
    warp-pipelined rpb16 + FFN norm-shift on the current shift-WAVG route.
  - Result file:
    `bench/results_native_4090_warppipe_breakdown_20260703_124500.jsonl`
    from remote `/tmp/native_4090_warppipe_breakdown_20260703_124500.jsonl`.
  - 4090 / 0.4B / prompt512 / bsz1 profiled rows, all pass against native
    prefill with max diff `0.0`:
    - baseline warp-specialized + shift-WAVG:
      profiled total `25.7310 ms`, `19,898.2 tok/s`, component sum
      `22.4311 ms`, peak `1005.2 MiB`;
    - warp-pipelined rpb16 + shift-WAVG:
      profiled total `26.0547 ms`, `19,651.0 tok/s`, component sum
      `22.1882 ms`, peak `1005.2 MiB`;
    - warp-pipelined rpb16 + shift-WAVG + FFN norm-shift:
      profiled total `25.0931 ms`, `20,404.0 tok/s`, component sum
      `21.3850 ms`, peak `957.3 MiB`.
  - Component signal:
    - `recurrent_scan_state_prep_cuda` drops from `11.2925 ms` baseline to
      `10.2545 ms` with warppipe rpb16 and `10.1245 ms` with warppipe+FFN
      norm-shift (`~1.04-1.17 ms` scan-side win);
    - FFN norm-shift lowers `ffn` from `2.2250 ms` to `1.9957 ms` and peak
      VRAM by about `48 MiB`;
    - the largest remaining non-scan component is still
      `attn_shift_wavg_lora_fused` at about `5.08-5.20 ms`.
  - Conclusion: the warppipe scan improvement is real in component profiling,
    but it is not enough to beat the strict best `28,780.6 tok/s` in e2e.  The
    next bounded experiment should target the adjacent shift-WAVG/state-scan
    boundary or a larger FFN memory boundary rather than another pure scan-only
    micro retile.
- [x] Extend shift-WAVG W-decay into the warp-pipelined scan boundary:
  - Implementation:
    - `rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel` now accepts
      precomputed W-decay through the existing `w_precomputed` CUDA extension
      path;
    - `RWKV7_NATIVE_PREFILL_FUSED_SHIFT_WAVG_LORA_W_DECAY=1` is now allowed
      for both `warp_specialized` and `warp_pipelined`;
    - benchmark telemetry records the effective `prefill_cuda_state_scan_w_precomputed`
      path; default HF/native behavior remains unchanged.
  - Correctness:
    - 4090 synthetic oracle vs raw-W warp-pipelined passed for rpb
      `1/2/4/8/16`, with output diff `0.0625`, state diff `0.00560`, and
      adjusted K/V diffs `0.0`;
    - all HF rows below pass greedy/cache/decode smoke.
  - Result files:
    - micro:
      `bench/results_cuda_state_scan_warppipe_wpre_micro_4090_20260703_130000.jsonl`
      from remote
      `/tmp/cuda_state_scan_warppipe_wpre_micro_4090_20260703_130000.jsonl`;
    - HF sweep:
      `bench/results_native_4090_warppipe_wpre_20260703_131500.jsonl`
      from remote `/tmp/native_4090_warppipe_wpre_20260703_131500.jsonl`.
  - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
    - warp-pipelined rpb8: `0.347136 ms`, `1,474,926.3 tok/s`;
    - warp-pipelined W-precomputed rpb8: `0.333824 ms`,
      `1,533,742.3 tok/s`;
    - warp-pipelined rpb16: `0.393216 ms`, `1,302,083.3 tok/s`;
    - warp-pipelined W-precomputed rpb16: `0.379904 ms`,
      `1,347,708.9 tok/s`.
  - 4090 / 0.4B / prompt512 / bsz1 HF rows:
    - warppipe rpb8 + shift-WAVG: `27,035.2 tok/s`, `18.9383 ms`;
    - warppipe W-precomputed rpb8 + shift-WAVG: `27,258.7 tok/s`,
      `18.7830 ms`;
    - warppipe rpb16 + shift-WAVG: `27,745.7 tok/s`, `18.4533 ms`;
    - warppipe W-precomputed rpb16 + shift-WAVG: `27,166.4 tok/s`,
      `18.8468 ms`.
  - Conclusion: W-precompute remains a micro win even with warppipe, but it does
    not transfer into a strict HF improvement and remains below the best
    `28,780.6 tok/s`.  Keep it opt-in; do not promote.  The remaining gap is
    no longer likely to close through W-only or pure scan synchronization
    retile.
- [x] Try the existing larger FFN boundary combination on the current best
  shift-WAVG/CUDA-scan route:
  - Motivation: after small W-only/scan-retile gains stopped transferring to
    HF e2e, test whether the already-correct FFN norm-shift and activation
    probes stack into a larger useful FFN memory boundary.
  - Result file:
    `bench/results_native_4090_ffn_combo_20260703_142500.jsonl`
    from remote `/tmp/native_4090_ffn_combo_20260703_142500.jsonl`.
  - 4090 / 0.4B / prompt512 / bsz1, CUDA warp-specialized state-scan +
    shift-WAVG `bm128/br64/bk64`, rows all pass greedy/cache/decode smoke:
    - baseline: `27,582.0 tok/s`, `18.5628 ms`, peak `988.2 MiB`;
    - FFN norm-shift only: `28,020.4 tok/s`, `18.2724 ms`, peak
      `964.2 MiB`;
    - FFN activation Triton block4096 only: `26,472.2 tok/s`, `19.3411 ms`,
      peak `986.2 MiB`;
    - FFN norm-shift + activation Triton block4096: `27,128.4 tok/s`,
      `18.8732 ms`, peak `963.2 MiB`;
    - FFN norm-shift + activation torch-inplace: `27,599.0 tok/s`,
      `18.5514 ms`, peak `963.2 MiB`.
  - Conclusion: FFN norm-shift is still memory-positive and can be a same-run
    small win, but activation fusion destroys the gain and no FFN combination
    beats the strict best `28,780.6 tok/s`.  This supports the current
    hypothesis that small standalone fusion boundaries now have high marginal
    cost; the next useful experiment needs a wider attention/shift-WAVG
    boundary or a genuinely persistent state-scan design, not another
    one-kernel activation tweak.
- [x] Try packed R/K/V bmm as a larger attention projection boundary:
  - Motivation: the repo already carries VKWR-inspired packed `RKVw` weights
    for native-graph decode, while prefill still uses three separate
    `F.linear(xr/xk/xv)` calls.  This experiment exposes
    `RWKV7_NATIVE_PREFILL_RKV_BMM=1`, makes `extract()` pack `RKVw` when that
    prefill flag is requested, and routes prefill R/K/V through one packed
    `torch.bmm` group without touching the default path.
  - Result file:
    `bench/results_native_4090_prefill_rkv_bmm_20260703_150000.jsonl`
    from remote `/tmp/native_4090_prefill_rkv_bmm_20260703_150000.jsonl`.
  - 4090 / 0.4B / prompt512 / bsz1, CUDA warp-specialized state-scan +
    shift-WAVG `bm128/br64/bk64`, rows all pass greedy/cache/decode smoke:
    - baseline: `27,964.9 tok/s`, `18.3087 ms`, peak `988.2 MiB`;
    - packed prefill RKV-bmm: effective, `27,204.3 tok/s`, `18.8206 ms`,
      peak `1134.2 MiB`;
    - packed prefill RKV-bmm + FFN norm-shift: effective,
      `27,072.1 tok/s`, `18.9125 ms`, peak `1111.0 MiB`.
  - Conclusion: packed RKV-bmm is correctness-safe but negative for this
    prefill shape.  It increases peak memory by carrying packed RKV weights and
    loses to the three cuBLAS linear calls, so keep it opt-in/telemetry only.
    This rules out the simple "borrow VKWR packed RKV" projection boundary for
    the 4090 bsz1/prompt512 path; the next attention-side attempt must reduce
    shift-WAVG intermediate traffic or fuse with the state-scan/output
    consumer, not just repackage the dense R/K/V matmuls.
- [x] Try a lean shift-WAVG down-kernel traffic retile:
  - Motivation: the current shift-WAVG down kernel launches one program per
    rank block.  For the 0.4B layer shape (`w/a=64`, `g=128`, `v=32`,
    `block_r=64`) the second rank block only needs `g` work, but the original
    kernel still loaded all six mix vectors and formed all mixed tensors.
    This experiment adds opt-in
    `RWKV7_NATIVE_PREFILL_FUSED_SHIFT_WAVG_LORA_LEAN_DOWN=1` so non-needed
    rank blocks mask off unused mix-vector loads while preserving the default
    path.
  - Result files:
    - micro:
      `bench/results_shift_wavg_lean_micro_4090_20260703_153000.jsonl`
      from remote `/tmp/shift_wavg_lean_micro_4090_20260703_153000.jsonl`;
    - HF e2e:
      `bench/results_native_4090_shift_wavg_lean_20260703_153000.jsonl`
      from remote `/tmp/native_4090_shift_wavg_lean_20260703_153000.jsonl`.
  - 4090 shift-WAVG micro, layer 1, rows512, hidden1024:
    - baseline down/full: `0.083968 ms` / `0.196608 ms`;
    - lean-down down/full: `0.082944 ms` / `0.196608 ms`;
    - correctness passes with max diff `0.0625`.
  - 4090 / 0.4B / prompt512 / bsz1 HF rows, all pass greedy/cache/decode:
    - baseline CUDA warp-specialized state-scan + shift-WAVG:
      `27,971.7 tok/s`, `18.3042 ms`, peak `988.2 MiB`;
    - lean-down only: `27,398.5 tok/s`, `18.6872 ms`, peak `988.2 MiB`;
    - lean-down + FFN norm-shift: `28,364.2 tok/s`, `18.0509 ms`, peak
      `964.2 MiB`.
  - Conclusion: masking unused mix-vector loads gives only a tiny down-phase
    micro win and does not improve the full shift-WAVG micro or HF e2e.  The
    FFN norm-shift combination is again memory-positive but still below the
    strict best `28,780.6 tok/s`.  Keep lean-down opt-in only.  This rules out
    another shallow shift-WAVG traffic tweak; the next useful attention-side
    route must remove or consume larger intermediates across the state-scan /
    output boundary rather than only pruning per-rank mix loads.
- [x] Try a lean shift-WAVG up-kernel rank retile:
  - Motivation: after the down-kernel traffic pruning failed to move HF e2e,
    test the symmetric up-projection side.  For the 0.4B shape with
    `block_r=64`, rank block 1 is G-only (`w/a/v` ranks are already
    exhausted), so this opt-in route skips the W/A/V mid and up-weight loads
    plus reductions for those exhausted rank ranges.
  - Implementation:
    - added `LEAN_UP` to `_wavg_lora_up_kernel`;
    - exposed `RWKV7_NATIVE_PREFILL_FUSED_SHIFT_WAVG_LORA_LEAN_UP=1`;
    - benchmark telemetry now records
      `prefill_fused_shift_wavg_lora_lean_up`;
    - default HF/native behavior is unchanged.
  - Result files:
    - micro:
      `bench/results_shift_wavg_lean_up_micro_4090_20260703_170000.jsonl`
      from remote `/tmp/shift_wavg_lean_up_micro_4090_20260703_170000.jsonl`;
    - rank32 micro:
      `bench/results_shift_wavg_rank32_micro_4090_20260703_171500.jsonl`
      from remote `/tmp/shift_wavg_rank32_micro_4090_20260703_171500.jsonl`;
    - HF e2e:
      `bench/results_native_4090_shift_wavg_lean_up_20260703_170000.jsonl`
      from remote `/tmp/native_4090_shift_wavg_lean_up_20260703_170000.jsonl`.
  - 4090 shift-WAVG micro, layer 1, rows512, hidden1024:
    - baseline `bm128/br64/bk64`: full `0.19968 ms`, down `0.086016 ms`,
      up `0.090112 ms`;
    - lean-up: full `0.198656 ms`, up `0.089088 ms`;
    - lean-down+lean-up: full `0.196608 ms`, up `0.088064 ms`;
    - correctness passes with max diff `0.0625`.
  - 4090 rank-retile micro:
    - `block_r=32` baseline: full `0.221184 ms`, down `0.106496 ms`,
      up `0.091136 ms`;
    - `block_r=32` + lean-up: full `0.2048 ms`, up `0.072704 ms`;
    - conclusion: lean-up helps small-rank up projection, but `block_r=32`
      increases down-kernel work enough to lose versus the current
      `block_r=64` route.
  - 4090 / 0.4B / prompt512 / bsz1 HF rows, all pass greedy/cache/decode:
    - same-run baseline: `26,928.3 tok/s`, `19.0135 ms`, peak `988.2 MiB`;
    - lean-up: `27,725.4 tok/s`, `18.4668 ms`, peak `988.2 MiB`;
    - lean-down+lean-up: `27,698.8 tok/s`, `18.4846 ms`, peak `988.2 MiB`;
    - lean-up + FFN norm-shift: `26,523.4 tok/s`, peak `964.2 MiB`;
    - lean-down+lean-up + FFN norm-shift: `27,285.1 tok/s`, peak
      `964.2 MiB`.
  - Conclusion: lean-up is correctness-safe and can beat the same-run
    baseline, but it does not exceed the strict best `28,780.6 tok/s`, and the
    rank32 retile loses overall.  Keep lean-up opt-in only.  This exhausts the
    shallow rank-traffic retile line; the next step needs a wider
    shift-WAVG/state-scan/output consumer boundary or a persistent/two-level
    scan design that changes more than per-rank pruning.
- [x] Try a G-mid output-prep consumer boundary:
  - Motivation: move beyond per-rank pruning by removing a full hidden-size
    tensor boundary.  The shift-WAVG route normally writes `g_out`
    `[B*T,hidden]`, then output-prep reads it as the attention gate.  This
    opt-in experiment skips G up-projection materialization in shift-WAVG,
    keeps only `g_mid` `[B*T,g_rank]`, and computes the G gate inside
    output-prep.
  - Implementation:
    - added `SKIP_G_OUT` to `_wavg_lora_up_kernel` and
      `fused_shift_wavg_lora(..., output_g_mid=True)`;
    - added `fused_attn_output_prepare_from_g_mid(...)`;
    - exposed
      `RWKV7_NATIVE_PREFILL_FUSED_SHIFT_WAVG_LORA_G_MID_OUTPUT=1`;
    - benchmark/analyzer telemetry now records the requested/effective flag;
    - default HF/native behavior is unchanged.
  - Result files:
    - micro:
      `bench/results_shift_wavg_gmid_micro_4090_20260703_183000.jsonl`
      from remote `/tmp/shift_wavg_gmid_micro_4090_20260703_183000.jsonl`;
    - HF e2e:
      `bench/results_native_4090_shift_wavg_gmid_20260703_183000.jsonl`
      from remote `/tmp/native_4090_shift_wavg_gmid_20260703_183000.jsonl`.
  - 4090 shift-WAVG micro, layer 1, rows512, hidden1024:
    - baseline: full `0.200704 ms`, down `0.084992 ms`, up
      `0.092160 ms`;
    - output-G-mid: full `0.185344 ms`, up `0.077824 ms`;
    - output-G-mid + lean-up: full `0.171008 ms`, up `0.063488 ms`;
    - correctness passes with max diff `0.0625`.
  - 4090 / 0.4B / prompt512 / bsz1 HF rows, all pass greedy/cache/decode:
    - same-run baseline: `26,808.8 tok/s`, `19.0982 ms`, peak `988.2 MiB`;
    - G-mid output-prep: `26,032.7 tok/s`, `19.6675 ms`, peak `987.3 MiB`;
    - G-mid + lean-up: `26,627.1 tok/s`, `19.2285 ms`, peak `987.3 MiB`;
    - G-mid + FFN norm-shift: `26,600.4 tok/s`, `19.2478 ms`, peak
      `963.3 MiB`.
  - Conclusion: the boundary is correctness-safe and does remove work from
    the shift-WAVG micro-kernel, but output-prep recomputing the G projection
    is slower end-to-end than reading the materialized gate.  Keep it opt-in
    only.  This rules out the simple "consume G mid in output" tensor-boundary
    route; the next attempt needs either a more integrated state-scan/output
    consumer or a persistent/two-level scan schedule, not moving one LoRA up
    projection into an output-prep tile.
- [x] Try the larger FFN norm+shift two-pass memory boundary:
  - Motivation: the previous recompute-style FFN norm+shift probe lowered peak
    memory but did not beat the strict best.  This bounded follow-up keeps both
    FFN GEMMs on cuBLAS and tests whether avoiding previous-token layernorm
    recomputation plus PyTorch `cat`/pointwise temporaries is enough to make the
    FFN memory boundary profitable.
  - Implementation:
    - added `RWKV7_NATIVE_PREFILL_FFN_FUSED_NORM_SHIFT_MODE=two_pass`;
    - `fused_ffn_norm_shift_prefill(..., mode="two_pass")` first writes the
      normalized `h` sequence with a Triton layernorm kernel, then builds `fk`
      from adjacent normalized rows in a second Triton shift/mix kernel;
    - benchmark telemetry now records
      `prefill_ffn_fused_norm_shift_mode`; default HF/native behavior and the
      existing recompute mode remain unchanged.
  - Correctness:
    - 4090 synthetic fp16 oracle vs PyTorch fallback passed for both modes:
      `fk_diff=0.0078125`, `h_last_diff=0.0`, cosine `1.0`;
    - all HF rows below pass greedy/cache/decode smoke.
  - Result file:
    `bench/results_native_4090_ffn_twopass_20260703_064419.jsonl` from remote
    `/tmp/native_4090_ffn_twopass_20260703_064419.jsonl`.
  - 4090 / 0.4B / prompt512 / bsz1, current CUDA state-scan + shift-WAVG route:
    - same-run baseline: `26,833.2 tok/s`, `19.0809 ms`, peak `988.2 MiB`;
    - recompute norm+shift `block_h=1024`: `26,431.7 tok/s`, `19.3707 ms`,
      peak `964.2 MiB`;
    - two-pass `block_h=1024`: `26,020.7 tok/s`, `19.6767 ms`, peak
      `964.2 MiB`;
    - two-pass `block_h=2048`: `26,269.8 tok/s`, `19.4901 ms`, peak
      `964.2 MiB`.
  - Conclusion: the two-pass FFN boundary is correctness-safe and preserves the
    memory win, but the extra launch and full `h` write/read lose more than the
    saved recompute.  Keep it opt-in only; do not promote.  This closes the
    cheap FFN-memory branch for the current shape.
- [x] Try CUDA warp-pair two-row worker schedule:
  - Motivation: test a middle point between row-block duplication and heavy
    head-level sharing.  The `warp_pair` schedule shares one producer/vector
    prep across an even row block, but each worker warp carries two state rows
    in registers.  This preserves more row-block CTAs than `head_reg16` while
    reducing worker-warps/CTA versus `warp_specialized rpb8/rpb16`.
  - Implementation:
    - added `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE=warp_pair`;
    - added `rwkv7_state_scan_prep_n64_rowblock_warp_pair_kernel`;
    - supported rpb `2/4/8/16`, kept it opt-in, and left default HF/native
      behavior unchanged;
    - `bench_cuda_state_scan_micro.py` now includes warp-pair rows.
  - Correctness:
    - 4090 synthetic oracle vs warp-specialized rpb1 passed exactly for rpb
      `2/4/8/16` (`out/state/K/V` diffs all `0.0`) on a short compile check;
    - all HF rows below pass greedy/cache/decode smoke.
  - Result files:
    - micro:
      `bench/results_cuda_state_scan_warppair_micro_20260703_065936.jsonl`
      from remote `/tmp/cuda_state_scan_warppair_micro_20260703_065936.jsonl`;
    - HF e2e:
      `bench/results_native_4090_warppair_20260703_070015.jsonl` from remote
      `/tmp/native_4090_warppair_20260703_070015.jsonl`.
  - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
    - warp-specialized rpb1: `0.448512 ms`, `1,141,552.5 tok/s`;
    - warp-pair rpb2: `0.512000 ms`, `1,000,000.0 tok/s`;
    - warp-pair rpb4: `0.518144 ms`, `988,142.3 tok/s`;
    - warp-pair rpb8: `0.523264 ms`, `978,473.6 tok/s`;
    - warp-pair rpb16: `0.529408 ms`, `967,118.0 tok/s`;
    - same-run warp-pipelined rpb8 remains much faster at `0.326656 ms`.
  - 4090 / 0.4B / prompt512 / bsz1 HF rows:
    - same-run warp-specialized rpb1 + shift-WAVG baseline:
      `27,273.8 tok/s`, `18.7726 ms`, peak `988.2 MiB`;
    - warp-pair rpb2: `26,294.5 tok/s`, `19.4717 ms`;
    - warp-pair rpb4: `26,624.5 tok/s`, `19.2304 ms`;
    - warp-pair rpb8: `26,162.7 tok/s`, `19.5699 ms`.
  - Conclusion: the schedule is correctness-safe but negative.  Serializing two
    rows inside each worker warp loses more than reducing CTA worker warps, so
    this row-pair middle point should stay telemetry-only and not be promoted.
- [x] Try allowed warp-pipelined + shift-WAVG tile/FFN combo sweep:
  - Motivation: consume a larger adjacent boundary with the proven
    `warp_pipelined` CUDA state-scan while avoiding the now-negative row-pair,
    head-level, raw no-K/V, SK, G-mid, W-decay, FFN two-pass, and lean routes.
  - Result file:
    `bench/results_native_4090_warppipe_allowed_combo_20260703_070650.jsonl`
    from remote
    `/tmp/native_4090_warppipe_allowed_combo_20260703_070650.jsonl`.
  - 4090 / 0.4B / prompt512 / bsz1 rows all pass greedy/cache/decode smoke:
    - same-run warp-specialized rpb1 baseline: `26,943.3 tok/s`,
      `19.0028 ms`;
    - warp-specialized rpb1 + FFN norm-shift recompute:
      `27,418.8 tok/s`, `18.6733 ms`, peak `964.2 MiB`;
    - warp-pipelined rpb8 baseline: `27,418.6 tok/s`, `18.6734 ms`;
    - warp-pipelined rpb8 + FFN norm-shift recompute:
      `26,690.7 tok/s`, `19.1827 ms`, peak `964.2 MiB`;
    - warp-pipelined rpb16 baseline: `27,068.6 tok/s`, `18.9149 ms`;
    - warp-pipelined rpb8 with shift-WAVG `(block_m,block_r,block_k)`:
      `(64,64,64)` -> `26,966.3 tok/s`,
      `(128,128,64)` -> `27,441.9 tok/s`,
      `(128,64,128)` -> best sweep row `27,583.2 tok/s`,
      `(128,64,256)` -> `26,963.2 tok/s`,
      `(128,128,128)` -> `26,903.1 tok/s`,
      `(64,64,128)` -> `26,917.2 tok/s`;
    - warp-specialized rpb1 with shift-WAVG `block_k=128`:
      `27,310.7 tok/s`;
    - warp-pipelined rpb1 with shift-WAVG `block_k=128`:
      `27,039.0 tok/s`.
  - Conclusion: the best allowed-combo row is
    `pipe_rpb8_bk128` at `27,583.2 tok/s` (`18.5620 ms`), which is
    correctness-safe but still below the strict branch best `28,780.6 tok/s`.
    Keep `warp_pipelined` and FFN norm-shift as opt-in tuning branches for
    card/shape variance, but do not promote them on the current 4090 target.
- [x] Try producer-side `kk*a` premultiply inside CUDA state-scan:
  - Motivation: test whether moving the recurrent update product
    `normalized_kk * a` from each row worker into the producer warp helps
    row-block schedules with more than one state row per CTA.  This preserves
    row parallelism and does not change any Python/wrapper boundary.
  - Result file:
    `bench/results_cuda_state_scan_kka_micro_20260703_071850.jsonl` from remote
    `/tmp/cuda_state_scan_kka_micro_20260703_071850.jsonl`.
  - Implementation was tested as a temporary source patch only and then
    reverted because the micro signal was negative/too small to justify
    carrying another opt-in schedule.
  - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` rows, all correctness-close vs
    default row-block (`out diff 0.03125`, state diff `1.9e-06`, K/V diff
    `0.0`):
    - premul warp-specialized rpb1: `0.456704 ms`, slower than the prior
      non-premul rpb1 `~0.4485 ms`;
    - premul warp-specialized rpb8: `0.456640 ms`, only a noise-level change
      versus prior `~0.4608 ms`;
    - premul warp-specialized rpb16: `0.508832 ms`, still slow;
    - premul warp-pipelined rpb8: `0.355328 ms`, slower than prior
      `~0.3472 ms`;
    - premul warp-pipelined rpb16: `0.391168 ms`, not competitive.
  - Conclusion: shifting one multiply from row workers to the producer warp is
    not enough and slightly hurts the strongest warppipe path.  Do not add the
    schedule.  The next attempt still needs a wider state-scan/shift-WAVG
    boundary or a genuinely two-level/persistent scan design.
- [x] Try two-level precomputed-warp CUDA state-scan:
  - Motivation: test a genuine two-stage schedule that preserves row
    parallelism without serializing multiple state rows inside one worker.
    Stage 1 precomputes W decay, normalized KK, adjusted K, and adjusted V once
    per token/head using the existing `wk_half` vector precompute.  Stage 2 is
    a new row-parallel warp-worker scan (`precomputed_warp`) with no producer
    warp and no per-token `__syncthreads()`, using one worker warp per state
    row and optional row blocks `1/2/4/8/16`.
  - Implementation:
    - added `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE=precomputed_warp`;
    - added
      `rwkv7_state_scan_prep_n64_rowblock_precomputed_wk_half_warp_kernel`;
    - relaxed CUDA/Python validation only for
      `precompute_mode=wk_half + schedule=precomputed_warp`;
    - added precomputed-warp rows to `bench/bench_cuda_state_scan_micro.py`;
    - default HF/native behavior remains unchanged unless the env flags are
      requested.
  - Correctness:
    - 4090 short synthetic compile check passed for rpb `1/2/4/8/16` vs the
      existing `wk_half` precompute scan (`out diff 0.0009766`, state diff
      `1.9e-06`, K/V diff `0.0`);
    - all HF rows below pass greedy/cache/decode smoke.
  - Result files:
    - micro:
      `bench/results_cuda_state_scan_precomputed_warp_micro_4090_20260703_073000.jsonl`
      from remote
      `/tmp/cuda_state_scan_precomputed_warp_micro_4090_20260703_073000.jsonl`;
    - HF sweep:
      `bench/results_native_4090_precomputed_warp_20260703_074200.jsonl`
      from remote `/tmp/native_4090_precomputed_warp_20260703_074200.jsonl`;
    - HF breakdown:
      `bench/results_native_4090_precomputed_warp_breakdown_20260703_075000.jsonl`
      from remote
      `/tmp/native_4090_precomputed_warp_breakdown_20260703_075000.jsonl`;
    - FFN-combo sweep:
      `bench/results_native_4090_precomputed_warp_ffn_20260703_075500.jsonl`
      from remote `/tmp/native_4090_precomputed_warp_ffn_20260703_075500.jsonl`.
  - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
    - warp-specialized rpb1: `0.447488 ms`;
    - warp-pipelined rpb8: `0.348128 ms`;
    - precomputed-warp rpb1: `0.201728 ms`;
    - precomputed-warp rpb4: `0.203776 ms`;
    - precomputed-warp rpb8: best micro row `0.197632 ms`,
      `2,590,673.6 tok/s`;
    - precomputed-warp rpb16: `0.222208 ms`.
  - 4090 / 0.4B / prompt512 / bsz1 HF sweep rows:
    - same-run warp-specialized + shift-WAVG baseline:
      `27,193.1 tok/s`, `18.8283 ms`;
    - existing default `wk_half` precompute row:
      `27,422.0 tok/s`, `18.6711 ms`;
    - precomputed-warp rpb1: `26,586.8 tok/s`, `19.2577 ms`;
    - precomputed-warp rpb4: `26,142.9 tok/s`, `19.5847 ms`;
    - precomputed-warp rpb8: `26,392.8 tok/s`, `19.3992 ms`;
    - precomputed-warp rpb16: `26,710.1 tok/s`, `19.1688 ms`.
  - Breakdown signal:
    - recurrent scan/state-prep component drops from `11.2995 ms` to
      `5.5550 ms` with precomputed-warp rpb8;
    - profiled total still gets worse (`25.3635 ms` -> `26.9793 ms`) because
      the extra precompute/global-temp route shifts time into the rest of the
      layer path and does not transfer into end-to-end throughput.
  - FFN norm-shift combo rows:
    - baseline + FFN norm-shift: `27,332.4 tok/s`, `18.7323 ms`;
    - default `wk_half` + FFN norm-shift: `26,930.4 tok/s`;
    - precomputed-warp rpb8 + FFN norm-shift: `26,398.7 tok/s`;
    - precomputed-warp rpb16 + FFN norm-shift: `26,323.0 tok/s`.
  - Conclusion: the two-level worker scan is the strongest isolated
    state-scan micro result so far and proves the worker-only schedule itself
    is viable, but the separate vector-precompute/global-temp boundary loses in
    full HF.  Keep `precomputed_warp` opt-in for future wider-boundary or
    card/shape experiments, but do not promote it on the current 4090 target.
    To turn this signal into an Albatross-gap win, the precomputed vectors need
    to be produced by the adjacent shift-WAVG/state-prep boundary rather than
    by an extra standalone precompute launch.
- [ ] Next persistent/two-level state-scan experiment:
  - Continue from the state-scan/shift-WAVG boundary, but avoid the now-negative
    row-pair, head-level, raw no-K/V, SK, G-mid, W-decay, FFN two-pass, and
    lean per-rank routes.  The next candidate should either consume a larger
    adjacent boundary with the proven warp-pipelined scan, or implement a
    genuinely different persistent/two-level schedule that preserves row
    parallelism without per-worker row serialization.
  - Same-run gate remains 4090 / 0.4B / prompt512 / bsz1 correctness plus a
    confirmed row beyond `28,780.6 tok/s`, moving toward `>=31,289 tok/s`.
  - [x] Try reusable precompute-temp buffers for CUDA `wk_half` vector
    precompute:
    - Motivation: the previous `precomputed_warp` micro result was strong, but
      full HF lost to the extra temp allocation/global-temp boundary.  This
      probe keeps the same math and schedule while reusing one preallocated
      fp16 `W` temp and one fp16 normalized-`KK` temp across layers.
    - Implementation:
      - added opt-in
        `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_REUSE_PRECOMPUTE=1`;
      - extended the CUDA extension entrypoint with optional `w_temp` /
        `kk_temp` tensors, validated only for `precompute_mode=wk_half`;
      - native prefill now preallocates and passes those buffers only when
        CUDA state-scan + `wk_half` precompute are requested;
      - benchmark rows now expose
        `prefill_cuda_state_scan_reuse_precompute(_effective)`.
    - Correctness/compile:
      - local `py_compile` and `git diff --check` passed;
      - 4090 direct CUDA wrapper check passed with no-temp vs temp-reuse diffs
        `[0.0, 0.0, 0.0, 0.0]`.
    - Result file:
      `bench/results_native_4090_precompute_reuse_20260703_083814.jsonl`
      from remote `/tmp/native_4090_precompute_reuse_20260703_083814.jsonl`.
    - 4090 / 0.4B / prompt512 / bsz1 HF rows, all pass greedy/cache smoke:
      - warp-specialized baseline: `26,633.1 tok/s`, `19.2242 ms`;
      - default `wk_half` no reuse: `26,038.8 tok/s`, `19.6629 ms`;
      - default `wk_half` reuse: `27,407.1 tok/s`, `18.6813 ms`;
      - precomputed-warp rpb8 no reuse: `26,606.7 tok/s`, `19.2433 ms`;
      - precomputed-warp rpb8 reuse: `26,737.2 tok/s`, `19.1493 ms`;
      - precomputed-warp rpb16 reuse: `26,639.4 tok/s`, `19.2197 ms`.
    - Conclusion: temp reuse fixes part of the allocation overhead and is a
      correctness-safe opt-in knob, but it still does not beat the strict
      branch best `28,780.6 tok/s`.  Keep it disabled by default.  The next
      Albatross-gap attempt still needs the precomputed vectors produced by a
      larger adjacent shift-WAVG/state-prep boundary or a different persistent
      schedule, not merely reused standalone precompute storage.
  - [x] Try in-place adjusted K/V output reuse for CUDA precompute routes:
    - Motivation: after temp-buffer reuse, the standalone CUDA vector-precompute
      boundary still allocates and writes separate adjusted K/V outputs even
      though the raw dense-projection K/V tensors are dead after state prep.
      This tests a narrower "consume the adjacent projection/state-prep
      boundary" idea by letting the precompute kernel overwrite those raw K/V
      tensors and return them as adjusted K/V.
    - Implementation:
      - added opt-in `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_INPLACE_KV=1`;
      - extended the CUDA extension entrypoint with an `inplace_kv` flag;
      - only allows in-place K/V when `write_kv=True` and
        `precompute_mode!=none`, avoiding the unsafe row-block no-precompute
        race where one row CTA could overwrite raw K before another row CTA
        has read it;
      - native prefill passes the flag only for CUDA precompute routes, and
        benchmark/profiler rows expose
        `prefill_cuda_state_scan_inplace_kv(_effective)`;
      - microbench now emits `wk_half + inplace_kv` rows for default and
        precomputed-warp schedules.
    - Correctness:
      - local `py_compile` and `git diff --check` passed;
      - 4090 direct CUDA wrapper check passed for default `wk_half` and
        `precomputed_warp rpb8`: no-inplace vs inplace diffs were
        `[0.0, 0.0, 0.0, 0.0]`, and returned K/V tensors alias the input K/V
        buffers.
    - Result files:
      - micro:
        `bench/results_cuda_state_scan_inplace_kv_micro_4090_20260703_085839.jsonl`
        from remote
        `/tmp/cuda_state_scan_inplace_kv_micro_20260703_085839.jsonl`;
      - HF smoke:
        `bench/results_native_4090_inplace_kv_20260703_085921.jsonl`
        from remote `/tmp/native_4090_inplace_kv_hf_20260703_085921.jsonl`.
    - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
      - default `wk_half` no in-place: `0.360960 ms`;
      - default `wk_half` in-place K/V: `0.329216 ms`
        (`~8.8%` faster in the CUDA component);
      - precomputed-warp rpb8 no in-place: `0.230400 ms`;
      - precomputed-warp rpb8 in-place K/V: `0.207360 ms`
        (`~10.0%` faster in the CUDA component);
      - precomputed-warp rpb16 no in-place: `0.256512 ms`;
      - precomputed-warp rpb16 in-place K/V: `0.230400 ms`.
    - HF e2e smoke rows all passed greedy/cache/decode, but the worker was in
      an anomalously slow launch-overhead state during this run: same-run
      warp-specialized baseline was only `15,711.0 tok/s` versus the usual
      `26k-28k` band.  Within that noisy run, default `wk_half` + temp reuse +
      in-place K/V reached `19,541.7 tok/s`, while precomputed-warp rpb8 +
      temp reuse + in-place K/V was `15,131.5 tok/s`.
    - Conclusion: in-place K/V is a real micro win and a correctness-safe
      opt-in for precompute routes, but the HF confirmation needs a normal
      low-noise 4090 rerun before it can be considered for promotion.  Keep it
      disabled by default; next work should either re-test this under a clean
      e2e timing window or fuse more of the adjacent shift-WAVG/state-prep
      producer boundary so the micro win survives full HF.
  - [x] Try in-place `normalized_kk * a` (KKA) precompute for CUDA `wk_half`
    routes:
    - Motivation: after in-place K/V made the vector-precompute component
      faster, test one more adjacent-state-prep boundary by overwriting the
      dead prefill `a` tensor with `normalized_kk * a`, so row workers can
      consume the recurrent-update product directly instead of multiplying
      `kk * a` in every row update.
    - Implementation:
      - added opt-in `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_INPLACE_KKA=1`;
      - extended the CUDA extension entrypoint and Python wrapper with an
        `inplace_kka` flag;
      - only allows KKA in-place when `precompute_mode=wk_half` and
        `w_precomputed=false`, because the vector precompute kernel must own
        the safe producer point for the overwritten `a` buffer;
      - default HF/native behavior is unchanged; benchmark/profiler rows expose
        `prefill_cuda_state_scan_inplace_kka(_effective)`.
    - Correctness:
      - local `py_compile` and `git diff --check` passed;
      - 4090 direct CUDA wrapper check passed for default `wk_half` and
        `precomputed_warp rpb8`: no-KKA vs in-place-KKA diffs were
        `[0.015625, 0.00040245, 0.0, 0.0]` for output/state/K/V.  This is
        correctness-close but not bit-exact because KKA is stored in fp16
        before row update.
    - Result file:
      `bench/results_cuda_state_scan_inplace_kka_micro_4090_20260703_091158.jsonl`
      from remote `/tmp/cuda_state_scan_inplace_kka_micro_20260703_091158.jsonl`.
    - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
      - default `wk_half`: `0.322560 ms`;
      - default `wk_half` + in-place K/V: `0.324608 ms`;
      - default `wk_half` + in-place K/V + in-place KKA: `0.330288 ms`;
      - precomputed-warp rpb8: `0.200704 ms`;
      - precomputed-warp rpb8 + in-place K/V: `0.202752 ms`;
      - precomputed-warp rpb8 + in-place K/V + in-place KKA: `0.202752 ms`;
      - precomputed-warp rpb16: `0.224768 ms`;
      - precomputed-warp rpb16 + in-place K/V: `0.226304 ms`;
      - precomputed-warp rpb16 + in-place K/V + in-place KKA: `0.225280 ms`.
    - Conclusion: KKA in-place is correctness-close but not a performance win;
      it slows the default `wk_half` route and is flat/noise for
      precomputed-warp.  Skip HF promotion rerun because the micro gate already
      failed.  Keep this as opt-in/telemetry only; the next Albatross-gap work
      should return to a wider shift-WAVG/state-scan/output boundary or a
      different persistent/two-level schedule, not more tiny vector-precompute
      products.
  - [x] Re-test in-place K/V precompute under a normal 4090 HF timing window:
    - Motivation: the earlier HF smoke for in-place K/V ran while the host was
      in an anomalously slow launch-overhead state (`~15.7k tok/s` baseline),
      so the micro win needed a clean e2e rerun before any promotion decision.
    - Result file:
      `bench/results_native_4090_inplace_kv_clean_20260703_092500.jsonl`
      from remote `/tmp/native_4090_inplace_kv_clean_20260703_092500.jsonl`.
    - 4090 / 0.4B / prompt512 / bsz1, current shift-WAVG route, all rows pass
      greedy/cache/decode smoke:
      - baseline warp-specialized: `26,486.1 tok/s`, `19.3309 ms`, peak
        `988.2 MiB`;
      - `wk_half` + temp reuse, no in-place: `26,360.6 tok/s`,
        `19.4229 ms`, peak `990.2 MiB`;
      - `wk_half` + temp reuse + in-place K/V: `26,336.8 tok/s`,
        `19.4405 ms`, peak `990.2 MiB`;
      - precomputed-warp rpb8 + temp reuse + in-place K/V:
        `26,020.8 tok/s`, `19.6766 ms`;
      - precomputed-warp rpb16 + temp reuse + in-place K/V:
        `26,392.8 tok/s`, `19.3992 ms`;
      - repeat baseline: `26,135.3 tok/s`, `19.5904 ms`.
    - Conclusion: the clean rerun confirms the CUDA micro win does not transfer
      into HF e2e for this shape.  The best precompute/in-place row stays in
      the same noisy band as baseline and below the strict branch best
      `28,780.6 tok/s`.  Keep in-place K/V disabled by default.  Stop spending
      iterations on standalone vector-precompute storage/reuse; the next
      Albatross-gap experiment should be a wider producer/consumer boundary or
      a genuinely different two-level/persistent schedule.
  - [x] Re-test the existing fused output-project boundary on the current
    CUDA-scan + shift-WAVG route:
    - Motivation: after standalone vector-precompute variants stopped
      transferring to HF e2e, try a wider attention output consumer boundary
      that fuses output-prep and `Ow` projection instead of only changing the
      scan-side vector producer.
    - Result file:
      `bench/results_native_4090_output_project_current_20260703_094500.jsonl`
      from remote `/tmp/native_4090_output_project_current_20260703_094500.jsonl`.
    - 4090 / 0.4B / prompt512 / bsz1, current shift-WAVG route, all rows pass
      greedy/cache/decode smoke:
      - baseline warp-specialized: `25,970.7 tok/s`, `19.7145 ms`;
      - output-project `block_m=32`: `19,336.3 tok/s`, `26.4787 ms`;
      - output-project `block_m=64`: `23,180.9 tok/s`, `22.0872 ms`;
      - output-project `block_m=128`: `23,670.3 tok/s`, `21.6305 ms`;
      - warp-pipelined rpb16: `26,015.0 tok/s`, `19.6809 ms`;
      - warp-pipelined rpb16 + output-project `block_m=128`:
        `24,917.5 tok/s`, `20.5478 ms`;
      - repeat baseline: `26,341.6 tok/s`, `19.4369 ms`.
    - Conclusion: the current output-project kernel is still negative even on
      the current CUDA/shift-WAVG route, because it recomputes output prep per
      output tile and loses to cuBLAS `Ow`. Keep it disabled and do not spend
      another iteration on this existing output-project design.
  - [x] Try warp-pipelined half-shared CUDA state-scan:
    - Motivation: test a persistent/two-level schedule variant that preserves
      row parallelism but reduces shared-memory traffic/footprint by storing
      the warp-pipelined per-token vectors (`r/w/k/a/kk/v_row`) as fp16 in
      shared memory while keeping row state and reductions in fp32.
    - Implementation:
      - added opt-in schedule
        `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE=warp_pipelined_half`;
      - added
        `rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_half_shared_kernel`;
      - supported rpb `1/2/4/8/16`; default HF/native behavior is unchanged;
      - microbench now records the half-shared rpb rows.
    - Correctness:
      - 4090 direct oracle against normal warp-pipelined rpb8 passed with
        output/state/K/V diffs `[0.03125, 0.0036721, 0.0, 0.0]`;
      - all HF rows below pass greedy/cache/decode smoke.  Some rows have
        expected fp16-shared drift up to `max_abs_diff=0.15625` while cosine
        remains `1.0`.
    - Result files:
      - micro:
        `bench/results_cuda_state_scan_warppipe_half_micro_4090_20260703_100000.jsonl`
        from remote
        `/tmp/cuda_state_scan_warppipe_half_micro_4090_20260703_100000.jsonl`;
      - HF sweep:
        `bench/results_native_4090_warppipe_half_20260703_100500.jsonl`
        from remote `/tmp/native_4090_warppipe_half_20260703_100500.jsonl`;
      - combo sweep:
        `bench/results_native_4090_warppipe_half_combo_20260703_101500.jsonl`
        from remote `/tmp/native_4090_warppipe_half_combo_20260703_101500.jsonl`.
    - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
      - warp-specialized rpb1: `0.458256 ms`;
      - normal warp-pipelined rpb1/rpb8/rpb16:
        `0.397216 / 0.358992 / 0.404480 ms`;
      - half-shared warp-pipelined rpb1/rpb8/rpb16:
        `0.388544 / 0.362512 / 0.405984 ms`.
    - 4090 HF rows, current shift-WAVG route:
      - same-run baseline: `26,128.4 tok/s`, `19.5955 ms`;
      - normal warppipe rpb1/rpb8:
        `25,461.4 / 24,669.5 tok/s`;
      - half-shared rpb1/rpb8:
        `26,616.5 / 26,583.9 tok/s`;
      - repeat baseline: `26,338.1 tok/s`.
    - Combo sweep:
      - baseline: `26,144.6 tok/s`, `19.5834 ms`, peak `988.2 MiB`;
      - baseline + FFN norm-shift: `26,432.0 tok/s`, peak `964.2 MiB`;
      - half rpb1 + FFN norm-shift: `26,018.5 tok/s`;
      - half rpb8 + FFN norm-shift: `26,317.9 tok/s`;
      - half rpb1/rpb8 with shift-WAVG `block_k=128`:
        `26,305.7 / 26,218.3 tok/s`;
      - half rpb8 + `block_k=128` + FFN norm-shift:
        `25,812.6 tok/s`.
    - Conclusion: half-shared is correctness-safe and can beat the noisy
      same-run normal warppipe rows, but it does not beat the strict branch
      best `28,780.6 tok/s` and does not stack with FFN norm-shift or
      `block_k=128`. Keep it opt-in only; no default promotion.
  - [x] Try a lighter head-level register-state CUDA schedule (`head_reg8`):
    - Motivation: the first one-CTA-per-head schedule (`head_reg16`) shared
      vector prep once per token/head but used a very heavy 1024-thread CTA.
      This probe keeps the same head-level/persistent idea but halves the CTA
      to 512 threads: each row has 8 lanes and each lane carries 8 state
      columns in registers.  It also parallelizes the KK norm reduction rather
      than using the older head_reg16 serial thread-0 norm loop.
    - Implementation:
      - added opt-in schedule
        `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE=head_reg8`;
      - added `rwkv7_state_scan_prep_n64_head_reg8_kernel`;
      - added quarter-warp row reductions and microbench coverage;
      - default HF/native behavior is unchanged.
    - Correctness:
      - 4090 direct oracle vs warp-specialized row-block passed with output /
        state / K / V diffs `[0.00024414, 4.77e-07, 0.0, 0.0]`;
      - HF rows below pass greedy/cache/decode smoke.
    - Result files:
      - micro:
        `bench/results_cuda_state_scan_head_reg8_micro_4090_20260703_103000.jsonl`
        from remote
        `/tmp/cuda_state_scan_head_reg8_micro_4090_20260703_103000.jsonl`;
      - HF:
        `bench/results_native_4090_head_reg8_20260703_103500.jsonl`
        from remote `/tmp/native_4090_head_reg8_20260703_103500.jsonl`.
    - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
      - warp-specialized rpb1: `0.458704 ms`;
      - head_reg16: `0.655360 ms`;
      - new head_reg8: `0.642048 ms`;
      - warp-pipelined rpb8: `0.337920 ms`;
      - warp-pipelined half rpb1: `0.365568 ms`.
    - 4090 / 0.4B / prompt512 / bsz1 HF rows:
      - baseline warp-specialized: `25,883.4 tok/s`, `19.7810 ms`;
      - head_reg8: `21,783.2 tok/s`, `23.5044 ms`;
      - head_reg16: `22,856.0 tok/s`, `22.4011 ms`.
    - Conclusion: reducing head-level CTA size and parallelizing the norm made
      micro only slightly better than head_reg16, but the route is still far
      slower than row-block/warppipe schedules and loses badly in HF e2e. Keep
      it opt-in only. Head-level one-CTA-per-head sharing remains a negative
      direction for this 4090 shape.
  - [x] Try moving A sigmoid into the shift-WAVG LoRA producer boundary:
    - Motivation: after scan-side persistent/head-level variants stayed below
      the strict best, test a small adjacent producer/consumer boundary that
      removes the separate PyTorch `sigmoid(A)` tensor op before state scan.
      This stays on the current shift-WAVG + CUDA row-block path and leaves the
      default HF path unchanged.
    - Implementation:
      - added opt-in
        `RWKV7_NATIVE_PREFILL_FUSED_SHIFT_WAVG_LORA_A_SIGMOID=1`;
      - `_wavg_lora_up_kernel` can now emit post-sigmoid `A` directly;
      - native prefill consumes `a2_out` directly only when the flag is
        effective, otherwise it keeps the old `torch.sigmoid(a2_out)` path;
      - scan and breakdown benchmark telemetry now records
        `prefill_fused_shift_wavg_lora_a_sigmoid_(requested|effective)`.
    - Correctness:
      - local and 4090 `py_compile` plus local `git diff --check` passed;
      - 4090 shift-WAVG micro row passed with
        `max_abs_diff_vs_fallback=0.031738` and
        `phase_max_abs_diff_vs_full=0.0`;
      - HF e2e row passed greedy/cache/decode smoke with `max_abs_diff=0.125`
        and cosine `1.0`.
    - Result files:
      - micro:
        `bench/results_shift_wavg_asigmoid_micro_4090_20260703_120000.jsonl`
        from remote `/tmp/shift_wavg_asigmoid_micro_20260703_1.jsonl`;
      - HF:
        `bench/results_native_4090_shift_wavg_asigmoid_20260703_120500.jsonl`
        from remote `/tmp/native_4090_shift_wavg_asigmoid_20260703_1.jsonl`.
    - 4090 micro rows, layer 1 / rows 512 / fp16:
      - baseline raw-A full helper: `0.202592 ms`;
      - A-sigmoid full helper: `0.207872 ms`.
    - 4090 / 0.4B / prompt512 / bsz1 HF rows:
      - baseline shift-WAVG + CUDA warp-specialized:
        `26,280.8 tok/s`, `19.4819 ms`;
      - A-sigmoid shift-WAVG:
        `25,970.2 tok/s`, `19.7149 ms`.
    - Conclusion: correctness is acceptable, but fusing this sigmoid into the
      Triton up-kernel is a small performance loss in both micro and HF e2e.
      Keep the flag opt-in only and do not promote it. The next useful stretch
      attempt still needs a wider shift-WAVG/state-scan/output boundary or a
      genuinely different persistent/two-level schedule, not another single
      activation move.
  - [x] Try half-warp paired CUDA row-block schedule (`halfwarp_pair`):
    - Motivation: test a genuinely different persistent/two-level schedule that
      preserves row parallelism without serializing two rows in one full worker
      warp.  The new worker warp is split into two independent half-warps: lanes
      `0..15` process one state row and lanes `16..31` process the next row,
      with each half-warp lane owning four state columns.
    - Implementation:
      - added opt-in schedule
        `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE=halfwarp_pair`;
      - added
        `rwkv7_state_scan_prep_n64_rowblock_halfwarp_pair_kernel`;
      - supported row blocks `2/4/8/16` and rejected `rows_per_block=1` for
        this schedule;
      - kept the producer warp shared-vector path and changed only the row
        worker schedule;
      - benchmark micro coverage now records half-warp-pair rows.
    - Correctness:
      - local `py_compile` and `git diff --check` passed;
      - 4090 direct oracle vs `warp_specialized rpb2` passed for rpb
        `2/4/8/16` with output/state/K/V diffs
        `[0.0009766, 1.9e-06, 0.0, 0.0]`;
      - all HF rows below pass greedy/cache/decode smoke.
    - Result files:
      - micro:
        `bench/results_cuda_state_scan_halfwarp_pair_micro_4090_20260703_121500.jsonl`
        from remote
        `/tmp/cuda_state_scan_halfwarp_pair_micro_20260703_1.jsonl`;
      - HF:
        `bench/results_native_4090_halfwarp_pair_20260703_122000.jsonl`
        from remote `/tmp/native_4090_halfwarp_pair_20260703_1.jsonl`.
    - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
      - warp-specialized rpb1/rpb8/rpb16:
        `0.457728 / 0.470016 / 0.528384 ms`;
      - full-warp-pair rpb2/rpb4/rpb8/rpb16:
        `0.521248 / 0.527360 / 0.533408 / 0.575488 ms`;
      - half-warp-pair rpb2/rpb4/rpb8/rpb16:
        `0.427904 / 0.432128 / 0.437248 / 0.470016 ms`;
      - warp-pipelined rpb1/rpb8/rpb16 remains faster at
        `0.388096 / 0.357376 / 0.403328 ms`.
    - 4090 / 0.4B / prompt512 / bsz1 HF rows, current shift-WAVG route:
      - baseline warp-specialized: `26,017.6 tok/s`, `19.6790 ms`;
      - half-warp-pair rpb2: `26,517.9 tok/s`, `19.3077 ms`;
      - half-warp-pair rpb4: `26,005.3 tok/s`, `19.6883 ms`;
      - half-warp-pair rpb8: `26,344.9 tok/s`, `19.4345 ms`;
      - repeat baseline: `26,562.6 tok/s`, `19.2753 ms`.
    - Conclusion: half-warp pairing improves the older full-warp-pair and
      warp-specialized micro schedules, but it still trails the stronger
      warp-pipelined micro rows and does not beat the strict branch best
      `28,780.6 tok/s` in HF e2e. Keep it opt-in only; the next stretch
      attempt should move to a wider shift-WAVG/state-scan/output producer-
      consumer boundary or a new schedule that beats warp-pipelined, not a
      smaller row-pair variant.
  - [x] Try shift-WAVG prev-cache producer boundary:
    - Motivation: remove the materialized `prev_h = cat(xpa, h[:-1])` tensor
      from the current shift-WAVG layer-prep route.  The new opt-in path lets
      the shift-WAVG down kernel read token `t-1` directly from the normalized
      `h` sequence and read token `0` from the attention cache `xpa`, so this
      tests a wider adjacent producer boundary without changing the default HF
      path.
    - Implementation:
      - added opt-in
        `RWKV7_NATIVE_PREFILL_FUSED_SHIFT_WAVG_LORA_PREV_CACHE=1`;
      - extended `fused_shift_wavg_lora(...)` with `prev_cache` / `seq_len`
        mode;
      - `_shift_wavg_lora_down_kernel` now supports `USE_PREV_CACHE` and
        computes `prev_h` from `(h, xpa)` internally;
      - native prefill avoids constructing `prev_h` only when the shift-WAVG
        route and the new flag are both effective;
      - scan benchmark/analyzer telemetry records the requested/effective flag.
    - Correctness:
      - local and 4090 `py_compile` plus local `git diff --check` passed;
      - shift-WAVG micro correctness passed with `max_abs_diff_vs_fallback=0.0625`
        and phase diff `0.0`;
      - HF e2e rows pass greedy/cache/decode smoke with `max_abs_diff=0.0625`
        and cosine `1.0`.
    - Result files:
      - micro:
        `bench/results_shift_wavg_prevcache_micro_4090_20260703_123000.jsonl`
        from remote `/tmp/shift_wavg_prevcache_micro_20260703_1.jsonl`;
      - HF:
        `bench/results_native_4090_shift_wavg_prevcache_20260703_123500.jsonl`
        from remote `/tmp/native_4090_shift_wavg_prevcache_20260703_1.jsonl`.
    - 4090 shift-WAVG micro, layer 1 / rows512 / fp16:
      - materialized-prev baseline: full `0.203776 ms`, down `0.090112 ms`,
        up `0.092160 ms`;
      - prev-cache mode: full `0.216064 ms`, down `0.104448 ms`,
        up `0.091136 ms`.
    - 4090 / 0.4B / prompt512 / bsz1 HF rows, current shift-WAVG + CUDA
      warp-specialized state-scan route:
      - baseline: `24,622.2 tok/s`, `20.7942 ms`;
      - prev-cache: `26,430.2 tok/s`, `19.3718 ms`;
      - repeat baseline: `24,917.2 tok/s`, `20.5481 ms`.
    - Conclusion: the opt-in path is correctness-safe and avoids the Python
      `prev_h` materialization boundary, but the down-kernel itself is slower
      and the HF run was in a low-throughput/noisy window.  It does not beat
      the strict branch best `28,780.6 tok/s`, so keep it disabled by default.
      The next stretch attempt still needs a larger shift-WAVG/state-scan/output
      producer-consumer boundary or a schedule that beats warp-pipelined in both
      micro and HF, not only a prev-cache indexing tweak.
  - [x] Try native prefill tail final-norm slicing:
    - Motivation: test a cheap tail boundary on the current HF native prefill
      route: when `logits_to_keep` is smaller than the prompt length, normalize
      only the retained tail tokens before `lm_head` instead of applying the
      final layernorm to all prompt tokens.  This is independent per token, so
      it is correctness-preserving for `logits_to_keep=1`, and stays opt-in.
    - Implementation:
      - added opt-in `RWKV7_NATIVE_PREFILL_TAIL_NORM_SLICE=1`;
      - native prefill keeps the old full-sequence final norm unless the flag is
        effective and `keep < T`;
      - scan benchmark/analyzer telemetry records the requested/effective flag.
    - Correctness:
      - local and 4090 `py_compile` plus local `git diff --check` passed;
      - HF e2e rows pass greedy/cache/decode smoke with `max_abs_diff=0.0625`
        and cosine `1.0`.
    - Result file:
      `bench/results_native_4090_tail_norm_slice_20260703_124500.jsonl`
      from remote `/tmp/native_4090_tail_norm_slice_20260703_1.jsonl`.
    - 4090 / 0.4B / prompt512 / bsz1 HF rows, current shift-WAVG + CUDA
      warp-specialized state-scan route:
      - baseline: `26,086.5 tok/s`, `19.6270 ms`;
      - tail final-norm slice: `25,678.1 tok/s`, `19.9392 ms`;
      - repeat baseline: `26,476.3 tok/s`, `19.3381 ms`.
    - Conclusion: the optimization is correctness-safe but slower in same-run
      HF because the saved final-norm work is too small and the extra slice path
      does not move the dominant attention/scan costs. Keep it opt-in only; do
      not promote. The stretch target still needs a wider attention
      producer-consumer boundary or a stronger persistent/two-level scan.
  - [x] Try in-place adjusted-V output reuse for no-precompute CUDA state scan:
    - Motivation: after K/V in-place was limited to precompute routes because
      raw K has cross-row CTA read hazards, test the safe half of that idea for
      the no-precompute row-block path.  V is only consumed by the row that
      writes adjusted V, so the CUDA scan can reuse the raw V projection tensor
      as `v_out` while still allocating a separate adjusted-K output.
    - Implementation:
      - added opt-in `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_INPLACE_V=1`;
      - extended the CUDA extension entrypoint and Python wrapper with
        `inplace_v`;
      - allowed it only when `write_kv=True` and `inplace_kv=False`;
      - native prefill enables it only for CUDA no-precompute state-scan routes;
      - scan benchmark/analyzer telemetry records
        `prefill_cuda_state_scan_inplace_v(_effective)`;
      - microbench now emits no-precompute in-place-V rows for
        warp-specialized and warp-pipelined schedules.
    - Correctness:
      - local and 4090 `py_compile` plus local `git diff --check` passed;
      - 4090 direct CUDA oracle passed for `warp_specialized` rpb1/rpb8 and
        `warp_pipelined` rpb8/rpb16, both layer-0/no-gate and gated cases:
        output/state/K/V diffs were all `0.0`, and returned `v_out` aliases the
        supplied V scratch buffer.
    - Result files:
      - micro:
        `bench/results_cuda_state_scan_inplace_v_micro_4090_20260703_140000.jsonl`
        from remote `/tmp/cuda_state_scan_inplace_v_micro_20260703_1.jsonl`;
      - HF:
        `bench/results_native_4090_inplace_v_20260703_140500.jsonl`
        from remote `/tmp/native_4090_inplace_v_20260703_1.jsonl`.
    - 4090 synthetic `B=1,T=512,H=16,N=64,fp16` micro rows:
      - warp-specialized rpb1: `0.459776 ms`;
      - warp-specialized rpb1 + in-place V: `0.462848 ms`;
      - warp-specialized rpb8: `0.472064 ms`;
      - warp-specialized rpb8 + in-place V: `0.455680 ms`;
      - warp-pipelined rpb8: `0.359424 ms`;
      - warp-pipelined rpb8 + in-place V: `0.339840 ms`;
      - warp-pipelined rpb16: `0.405504 ms`;
      - warp-pipelined rpb16 + in-place V: `0.380928 ms`.
    - 4090 / 0.4B / prompt512 / bsz1 HF rows, current shift-WAVG route:
      - baseline warp-specialized rpb1:
        `25,676.3 tok/s`, `19.9405 ms`;
      - in-place V warp-specialized rpb1:
        `25,724.8 tok/s`, `19.9030 ms`;
      - warp-pipelined rpb8:
        `26,281.0 tok/s`, `19.4817 ms`;
      - in-place V warp-pipelined rpb8:
        `25,337.2 tok/s`, `20.2074 ms`;
      - repeat baseline:
        `25,887.9 tok/s`, `19.7776 ms`.
    - Conclusion: V output aliasing is bit-exact and saves memory traffic in
      some CUDA micro rows, but it does not transfer into a strict HF e2e win
      and remains below the branch best `28,780.6 tok/s`. Keep it opt-in only
      and do not promote.  This closes another small storage-reuse boundary;
      the next stretch attempt should return to a wider integrated
      shift-WAVG/state-scan/output producer-consumer kernel or a schedule that
      improves both micro and full HF.
  - [x] Run a strongest-existing-boundary stack sweep before another kernel
    rewrite:
    - Motivation: the remaining unchecked gate is still `>=0.60x` Albatross,
      and several opt-in boundaries had only been tested alone or in older
      noisy windows.  Before writing another schedule, re-test the strongest
      existing knobs around the strict best route:
      shift-WAVG `bm128/br64/bk64` + CUDA warp-specialized rpb1 state scan +
      fused output.
    - Result files:
      - broad sweep:
        `bench/results_native_4090_existing_combo_sweep_20260703_150000.jsonl`
        from remote
        `/tmp/native_4090_existing_combo_sweep_20260703_1.jsonl`;
      - focused prev-cache stack:
        `bench/results_native_4090_prevcache_stack_20260703_151000.jsonl`
        from remote `/tmp/native_4090_prevcache_stack_20260703_1.jsonl`.
    - Broad 4090 / 0.4B / prompt512 / bsz1 rows, all pass
      greedy/cache/decode smoke:
      - baseline: `26,061.9 tok/s`, `19.6455 ms`;
      - FFN norm-shift: `26,197.7 tok/s`, `19.5437 ms`;
      - lean-down: `26,414.3 tok/s`, `19.3834 ms`;
      - lean-down + FFN norm-shift: `26,707.2 tok/s`, `19.1708 ms`;
      - in-place V: `26,267.5 tok/s`, `19.4918 ms`;
      - tail norm slice: `26,653.5 tok/s`, `19.2095 ms`;
      - prev-cache: `27,471.0 tok/s`, `18.6378 ms`;
      - warp-pipelined rpb16: `26,304.5 tok/s`, `19.4644 ms`;
      - warp-pipelined rpb16 + W-decay: `25,410.6 tok/s`, `20.1491 ms`;
      - raw no-K/V output: `26,522.5 tok/s`, `19.3043 ms`;
      - repeat baseline: `26,281.5 tok/s`, `19.4814 ms`.
    - Focused prev-cache stack rows:
      - prev-cache repeat: `27,221.5 tok/s`, `18.8087 ms`;
      - prev-cache + FFN norm-shift: `26,921.2 tok/s`, `19.0185 ms`;
      - prev-cache + lean-down: `26,833.2 tok/s`, `19.0808 ms`;
      - prev-cache + lean-down + FFN norm-shift:
        `26,899.0 tok/s`, `19.0342 ms`;
      - prev-cache + tail norm slice: `26,149.0 tok/s`, `19.5801 ms`;
      - prev-cache + in-place V: `26,937.1 tok/s`, `19.0072 ms`;
      - prev-cache + raw no-K/V output: `26,578.8 tok/s`, `19.2635 ms`.
    - Conclusion: the existing opt-in stack does not close the gap.  The best
      current broad row (`prev-cache`, `27,471.0 tok/s`) and focused repeat
      (`27,221.5 tok/s`) remain below the strict branch best
      `28,780.6 tok/s`, and every stack with FFN norm-shift, lean-down, tail
      norm slicing, in-place V, raw no-K/V, or warp-pipelined W-decay regresses.
      Keep all of these opt-in only.  The next stretch attempt should not be
      another flag-combo rerun; it needs a new integrated producer-consumer
      boundary or a different scan schedule.
- [ ] Stretch target remains `>=0.60x` Albatross (`>=31,289 tok/s`) for
  4090 / 0.4B / prompt512 / bsz1. Best current confirmed row on this branch is
  `28,780.6 tok/s` (`~0.5519x`), still about `2,508 tok/s` (`~8.7%`
  relative uplift) short of the stretch.

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
    - Current strict confirmed row on this branch is `28,780.6 tok/s`
      (`~0.5519x`), still below the stretch target `31,289 tok/s` by about
      `2,508 tok/s` (`~8.7%` relative uplift).

## Big TODO routing note

- [ ] Keep the FLA/PyTorch path as the compatibility and correctness fallback,
  not the main Albatross-gap optimization target. Native-unsupported, training,
  PEFT/TRL, and generic quantized paths may still fall back to FLA/PyTorch.
- [ ] Keep two performance tracks active:
  - short-term: native fused fp16 prefill/decode kernels, starting from the
    confirmed fused state-scan + shift-WAVG row and pushing 4090
    0.4B/prompt512/bsz1 from the current strict `0.5519x` row to `>=0.60x`
    Albatross;
  - high-upside math: DPLR/WY compact chunk prefill, with next work on
    prefix-shared apply/output scheduling, less dense `[N,N]`
    traffic/materialization, and later fused W8/W4 kernels.
- [x] Validate the new raw-output CUDA state-scan no-K/V-write probe on 4090:
  pair `RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_RAW_OUTPUT=1` with
  `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN=1`,
  `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_LANES=64`,
  `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE=warp_specialized`, and
  `RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_PRECOMPUTE=0`. The benchmark rows now
  expose `prefill_cuda_state_scan_raw_nokv_effective` and
  `prefill_cuda_state_scan_write_kv`; keep the probe only if it beats the
  current strict 4090 best `28,780.6 tok/s` and moves toward P1
  `31,289 tok/s`.
  - Result file: `bench/results_native_4090_raw_nokv_cuda_20260703_062833.jsonl`
    from remote `/tmp/native_4090_raw_nokv_cuda_20260703_062833.jsonl`.
  - Same-run baseline shift-WAVG + CUDA warp-specialized state scan:
    `26,741.0 tok/s`, `19.1466 ms`, pass.
  - Raw-output CUDA no-K/V-write rows all passed correctness/cache smoke:
    - `rows_per_block=1`: `27,359.6 tok/s`, `18.7137 ms`;
    - `rows_per_block=2`: `27,342.9 tok/s`, `18.7252 ms`;
    - `rows_per_block=4`: `18,755.3 tok/s`, `27.2989 ms`;
    - `rows_per_block=8`: `26,346.1 tok/s`, `19.4336 ms`;
    - `rows_per_block=16`: `26,548.1 tok/s`, `19.2857 ms`.
  - Conclusion: the CUDA no-K/V raw-output path is correctness-safe and can
    beat the same-run baseline in a noisy run, but it does not exceed the
    strict branch best `28,780.6 tok/s`; keep it opt-in and do not promote.
    This reinforces the next mainline direction: a larger integrated
    shift-WAVG/state-scan/output consumer or a stronger persistent/two-level
    scan schedule, not another raw-output recompute boundary.
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

- HF Transformers adapter is the 30w scope in this repo; vLLM/SGLang work is
  out of scope for this branch and should not consume the Albatross-gap loop.
- Do not default-enable dense3 in the HF path.
- Do not claim Albatross-level performance from dense3 alone.
- Do not start vLLM/SGLang work in this repository.
- Do not optimize Python loops instead of compiled kernel/factor work.
