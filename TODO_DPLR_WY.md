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
- [ ] Next compact-WY task:
  - Do not spend another iteration on lossy `start_states` storage. The next
    useful compact experiment should reduce or bypass dense start-state traffic
    without adding lossy conversion, e.g. a compact-prefix/apply fusion that
    computes each chunk's start state inside the apply boundary from compact
    factors, or a lower-traffic prefix representation that still preserves
    chunk-level parallelism. Keep output-only apply as the best compact HF
    flag so far, but do not default-enable it until a same-run synthetic/HF
    confirmation beats the current fused recurrent scan line.
- [ ] Stretch target remains `>=0.60x` Albatross (`>=31,289 tok/s`) for
  4090 / 0.4B / prompt512 / bsz1. Best current confirmed row on this branch is
  `27,051.0 tok/s` (`~0.5187x`), still about `15.7%` relative uplift short of
  the stretch.

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
