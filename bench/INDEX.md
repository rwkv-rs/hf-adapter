# Bench inventory

Generated inventory of benchmark scripts and evidence directories. Keep this file lightweight: it is an orientation map, not the source of truth for benchmark conclusions.

## Promoted production-close artifacts

| Platform | Artifact | Current conclusion |
|---|---|---|
| V100 32GB | [`v100_acceptance_20260716/`](v100_acceptance_20260716/README.md) | Historical 2026-07-16 fail-closed snapshot; newer V100 conclusions remain in current status and MM4 artifacts |
| V100 32GB | [`v100_production_close_20260711/`](v100_production_close_20260711/README.md) | Dense Albatross P1 and native W8/W4 speed lane pass |
| V100 32GB | [`v100_active_b1b8_20260715/`](v100_active_b1b8_20260715/README.md) | 1.5B vs full-FLA Qwen3.5-2B B1/B8 raw and active-parameter work gates pass |
| V100 32GB / RTX 4080 | [`4080_v100_decode_tuning_20260808/`](4080_v100_decode_tuning_20260808/README.md) | Fail-closed exact-B8 decode tuning passes paired speed, policy and greedy gates on both products |
| V100 32GB | [`v100_exact_card_20260811/`](v100_exact_card_20260811/README.md) | Exact 0.4B/1.5B B8 FP16 state passes both A/B orders, saves allocated VRAM, and preserves greedy traces |
| RTX 3090 | [`3090_g1h_7p2_bsz8_20260714/`](3090_g1h_7p2_bsz8_20260714/README.md) | Latest g1h 7.2B/9B bsz8 dense, active-work, W8/W4 speed and memory gates pass 18/18 |
| RTX 3090 | [`3090_g1i_qwen35_maxperf_20260812/`](3090_g1i_qwen35_maxperf_20260812/README.md) | Max-performance g1d/g1i 0.4B-7.2B B1/B8 strict parameter-adjusted prefill passes 24/24; correctness passes 25/25 |
| RTX 3090 | [`3090_self_fused_20260713/`](3090_self_fused_20260713/README.md) | 7.2B/9B prompt-2048 batch-1/2 self-fused dense gates pass |
| RTX 4090 | [`4090_g1h_7p2_bsz8_20260715/`](4090_g1h_7p2_bsz8_20260715/README.md) | Latest g1h 7.2B/9B bsz8 dense, active-work, W8/W4 speed and quant-local memory gates pass 18/18 |
| RTX 4090 | [`4090_small_bsz8_20260715/`](4090_small_bsz8_20260715/README.md) | 0.4B/0.8B, 1.5B/2B and 2.9B/4B bsz8 dense, active-work, W8/W4 speed and physical-memory gates pass 54/54 |
| RTX 4090 | [`4090_validation_summary.md`](4090_validation_summary.md) | Measured dense decode/current-session prefill and quant speed lanes pass |
| RTX 4080 | [`4080_full_model_ladder_20260719/`](4080_full_model_ladder_20260719/README.md) | Native HF 0.4B/1.5B/2.9B B1/B8 full-FLA-Qwen matrices pass 6/6; 7.2B/13.3B capacity and quant routes recorded |
| RTX 4080 | [`4080_b8_projection_bmm_20260809/`](4080_b8_projection_bmm_20260809/README.md) | Exact-B8 grouped W/A/V projection route improves 0.4B/1.5B/2.9B decode by `1.1267x/1.0942x/1.0809x` with exact logits and greedy `4,608/4,608` |
| RTX 4080 | [`4080_7p2b_fp16_state_20260809/`](4080_7p2b_fp16_state_20260809/README.md) | 7.2B/B8 FP16-state decode reaches `344.39 tok/s`, `1.0301x` FP32-state, `-123.88 MiB`, and greedy `12,288/12,288` |
| RTX 5070 Laptop | [`5070_max_perf_20260811/`](5070_max_perf_20260811/README.md) | Exact 0.4B/1.5B Native prefill/decode schedules pass correctness and paired speed gates; negative candidates remain disabled |
| RTX 5090 | [`5090_blackwell_production_close_20260712/`](5090_blackwell_production_close_20260712/README.md) | Quant pressure, 13.3B conversion and full MATH500 pass |
| RTX 5090 | [`5090_g1h_qwen35_b1_b8_20260715/`](5090_g1h_qwen35_b1_b8_20260715/README.md) | Historical under the superseding protocol; the original 0.4B/0.8B through 7.2B/9B B1/B8 matrix passes 8/8 batch-pairs and 144/144 full-FLA cells |
| RTX 5090 | [`5090_g1h_13p3_20260715/`](5090_g1h_13p3_20260715/README.md) | Latest official g1h 13.3B load/generate plus B8 paired-fp16 MM8/MM4 speed-policy gate pass |
| RTX 5090 | [`5090_bntn_all_models_20260716/`](5090_bntn_all_models_20260716/README.md) | g1h 1.5B/2.9B/7.2B/13.3B B1/B8 BN/TN Tensor Core W4 passes all-phase speed, `0.5298x–0.6250x` footprint and correctness gates |
| RTX 5090 | [`5090_train_temp_alignment_20260717/`](5090_train_temp_alignment_20260717/README.md) | Official vs opt-in HF train_temp BF16 12x768 backward/step exact; 3-seed x 1,000-step cohort passes |
| RTX 5090 | [`5090_native_train_temp_b16_20260718/`](5090_native_train_temp_b16_20260718/README.md) | Native/no-FLA B16/T512 exact tensors, 3-seed x 1,000-step, 500+500 resume and steady-memory gates pass; training speed remains `0.9499x` official |
| RTX 5090 | [`5090_native_official_fp16_production_20260718/`](5090_native_official_fp16_production_20260718/README.md) | Native default-policy fp16-state decode and 2.9B/13.3B sequence prefill pass pinned official v3a tensor, greedy and speed gates |
| RTX 5090 | [`5090_native_train_temp_real_minipile_20260718/`](5090_native_train_temp_real_minipile_20260718/README.md) | Native B16/T512 exact step, paired real-MiniPile 3-seed, continuous 5,000-step and 2,500+2,500 recovery pass at or above official throughput |
| Apple M5 | [`../docs/hardware/APPLE_PRODUCTION_CLOSE.md`](../docs/hardware/APPLE_PRODUCTION_CLOSE.md) | Selected MLX/Qwen3.5 production pairs pass |

Canonical cross-platform summary: [`../BENCHMARK.md`](../BENCHMARK.md) and
[`../docs/HARDWARE_MATRIX.md`](../docs/HARDWARE_MATRIX.md).

## Promoted paired comparison artifacts

| Platform | Artifact | Current conclusion |
|---|---|---|
| RTX 3090 | [`3090_qwen35_paired_pd_v2_20260816/`](3090_qwen35_paired_pd_v2_20260816/README.md) | Frozen optimized-Qwen reference plus fresh RWKV: raw and parameter-adjusted Prefill/Decode pass all 48 cells; adjusted minima `1.208324x/1.017763x`; 8/8 long-horizon correctness checks pass. Speed only, not quality/E2E |
| RTX 4090 | [`4090_qwen35_paired_pd_v2_20260815/`](4090_qwen35_paired_pd_v2_20260815/README.md) | Frozen optimized-Qwen reference plus fresh RWKV: raw and parameter-adjusted Prefill/Decode pass all 48 cells; adjusted minima `1.148668x/1.026173x`; 8/8 long-horizon correctness checks pass. Speed only, not quality/E2E |
| RTX 4080 | [`4080_qwen35_paired_pd_v1_20260814/`](4080_qwen35_paired_pd_v1_20260814/README.md) | Same-runtime raw and parameter-adjusted Prefill/Decode pass all 36 cells; adjusted minima `1.051333x/1.022115x`; 6/6 512-token native-graph/FLA probes pass. Speed only, not quality/E2E |
| Tesla V100 | [`v100_qwen35_paired_pd_v1_20260814/`](v100_qwen35_paired_pd_v1_20260814/README.md) | Frozen-reference raw and parameter-adjusted Prefill/Decode pass all 48 cells; adjusted minima `1.808536x/1.120373x`; 8/8 FLA/native probes plus 7.2B/B8 graph closure pass. Speed only, not quality/E2E |
| RTX 5090 | [`5090_qwen35_paired_decode_v1_20260813/`](5090_qwen35_paired_decode_v1_20260813/README.md) | Frozen-reference parameter-adjusted Decode passes 48/48 at minimum `1.029966x`; raw Decode 48/48 is supporting telemetry. Decode-only, not quality/Prefill/E2E |

## Promoted reference-only artifacts

| Platform | Artifact | Current conclusion |
|---|---|---|
| RTX 5090 | [`5090_qwen35_best_optimized_hf_v1_20260813/`](5090_qwen35_best_optimized_hf_v1_20260813/README.md) | Qwen3.5 0.8B/2B/4B/9B official-operator reference passes 48/48 with fixed per-model optimized Graph Decode; source remains reference-only and paired Decode is reported separately |

## Promoted exact-card validation artifacts

| Platform | Artifact | Current conclusion |
|---|---|---|
| Tesla T4 15GB | [`t4_production_close_20260720/`](t4_production_close_20260720/README.md) | 0.1B–2.9B HF/cache/prefill/decode, exact-T4 W8/W4 and training integration validated; dense Albatross and broad all-phase quant gaps remain, so this is not production-close |
| AMD gfx1100 | [`amd_gfx1100_native_20260727/`](amd_gfx1100_native_20260727/README.md) | Post-split fully native HF load/generate, PEFT, cache/chunked prefill, bf16 Trainer and B1/B2/B4/B8 baseline pass; fused/quantized production performance remains open |

## Apple M5 production-close evidence

The `apple_production_close_*_m5_20260711.jsonl` top-level files contain the
checked Qwen3.5 0.8B/2B baselines, RWKV-7 0.4B compiled W4 rows, RWKV-7 1.5B
W4/W8 compile rows, full-context RWKV draft speculation, and the final
two-pair conservative gate. Conclusions and reproduction commands are in
[`../docs/hardware/APPLE_PRODUCTION_CLOSE.md`](../docs/hardware/APPLE_PRODUCTION_CLOSE.md).

## Evidence directories

| Directory | Title / purpose | JSONL | Logs |
| --- | --- | --- | --- |
| 3090_qwen35_paired_pd_v2_20260816 | RTX 3090 frozen-reference strict paired Prefill/Decode 48/48, routes and 8/8 FLA correctness | 23 | 17 |
| 4080_qwen35_paired_pd_v1_20260814 | RTX 4080 same-runtime strict paired Prefill/Decode 36/36, routes and 6/6 FLA correctness | 24 | 16 |
| v100_qwen35_paired_pd_v1_20260814 | Tesla V100 frozen-reference strict paired Prefill/Decode 48/48, routes, 8/8 FLA correctness, and 7.2B/B8 graph closure | 33 | 1 |
| amd_gfx1100_native_20260727 | AMD gfx1100/ROCm 7.2.1 fully native HF compatibility and baseline | 1 | 13 |
| 3090_g1h_7p2_bsz8_20260714 | RTX 3090 latest-g1h 7.2B vs Qwen3.5-9B bsz8 acceptance | 5 | 5 |
| 3090_g1i_qwen35_maxperf_20260812 | RTX 3090 exact-shape max-performance B1/B8 Qwen3.5 matrix, correctness and promotion probes | 10 | 0 |
| 3090_g1i_qwen35_prefill_pd_20260812 | RTX 3090 latest-checkpoint B1/B8 strict parameter-adjusted prefill and FP16-accumulation correctness | 2 | 0 |
| 3090_self_fused_20260713 | RTX 3090 self-fused RWKV-7 7.2B long-prefill close | 2 | 0 |
| 4090_g1h_7p2_bsz8_20260715 | RTX 4090 latest-g1h 7.2B vs Qwen3.5-9B bsz8 acceptance | 6 | 7 |
| 4090_small_bsz8_20260715 | RTX 4090 0.4B/1.5B/2.9B vs Qwen3.5 bsz8 acceptance | 5 | 1 |
| 4080_ada_validation_20260719 | RTX 4080 Native HF, full-FLA Qwen3.5 and quant acceptance | 21 | 9 |
| 4080_v100_decode_tuning_20260808 | RTX 4080 grouped B8 decode and exact-V100 B8 WAVG launch tuning | 7 | 0 |
| 4080_b8_projection_bmm_20260809 | RTX 4080 0.4B/1.5B/2.9B exact-B8 grouped W/A/V projection acceptance | 2 | 0 |
| 4080_7p2b_fp16_state_20260809 | RTX 4080 7.2B/B8 FP16-state decode acceptance | 2 | 2 |
| 4080_bnb8_refactor_20260728 | RTX 4080 BnB W8 helper-split correctness, memory and B1/B8 A/B regression | 5 | 3 |
| 4080_native_jit_split_20260728 | RTX 4080 complete native-JIT facade split B1/B8 correctness, memory and performance A/B | 4 | 3 |
| 4080_full_model_ladder_20260719 | RTX 4080 B1/B8 full-FLA-Qwen pair matrix and large-model capacity ladder | 28 | 0 |
| 5070_native_memory_loading_20260716 | RTX 5070 Laptop CPU-first native MM8/MM4 memory-loading smoke | 1 | 0 |
| 5090_g1h_13p3_20260715 | RTX 5090 latest official g1h 13.3B conversion, smoke, and speed-policy acceptance | 2 | 0 |
| 5090_g1h_qwen35_b1_b8_20260715 | RTX 5090 complete B1/B8 RWKV-7 vs full-FLA Qwen3.5 acceptance | 107 | 0 |
| 5090_qwen35_best_optimized_hf_v1_20260813 | RTX 5090 Qwen3.5 0.8B/2B/4B/9B best-optimized official-operator reference-only matrix | 6 | 12 |
| 5090_qwen35_paired_decode_v1_20260813 | RTX 5090 frozen-Qwen versus RWKV strict parameter-adjusted Decode 48/48, route A/B and 8/8 FLA correctness | 30 | 22 |
| 5090_blackwell_hf_matrix_20260704 | RTX 5090 Blackwell HF validation matrix (2026-07-04) | 2 | 12 |
| 5090_blackwell_native_prefill_smoke_20260704 | RTX 5090 Blackwell native-prefill validation (2026-07-04) | 2 | 3 |
| 5090_blackwell_native_quant_20260704 | RTX 5090 native quant benchmark (2026-07-04) | 1 | 2 |
| 5090_blackwell_quant_matrix_20260705 | RTX 5090 Blackwell native MM8/MM4 fresh-process quant matrix (2026-07-05) | 1 | 0 |
| 5090_blackwell_quant_policy_20260705 | RTX 5090 native MM8/MM4 quantization policy sweep (2026-07-05) | 7 | 0 |
| 5090_bntn_all_models_20260716 | RTX 5090 g1h 1.5B/2.9B/7.2B/13.3B production BN/TN W4 model matrix plus 0.4B rejection, automatic-profile smokes and grid/autotune evidence | 13 | 0 |
| 5090_native_hf_gradio_train_temp_20260718 | RTX 5090 real Native HF Gradio UI, official-v3a comparison and unchanged official-shell B16/T512/ZeRO-2 evidence | 4 | 3 |
| 5090_native_official_fp16_production_20260718 | RTX 5090 Native default-policy fp16-state decode and exact 2.9B/13.3B sequence-prefill evidence | 4 | 68 |
| 5090_native_train_temp_real_minipile_20260718 | RTX 5090 Native real-MiniPile train_temp exact-step, multi-seed, 5,000-step and resume evidence | 0 | 28 |
| 5090_native_train_temp_b16_20260718 | RTX 5090 Native/no-FLA B16/T512 official train_temp tensor, convergence, resume and memory-stability evidence | 0 | 1 |
| 5090_train_temp_alignment_20260717 | RTX 5090 official train_temp versus opt-in HF CUDA numerical and convergence alignment | 0 | 1 |
| 5090_bn_tn_20260716 | RTX 5090 explicit CUDA block-N/thread-N W8/W4 sweep; 288/288 correct, 4/32 winners beat old quant, 0/32 beat FP16, no production promotion | 3 | 3 |
| 5090_bn_tn_tensorcore_20260716 | RTX 5090 production BN/TN Tensor Core W4; B1/B8 all-phase close plus 70/70 per-launch contract checks | 10 | 2 |
| 5090_blackwell_smoke_20260704 | RTX 5090 Blackwell smoke (2026-07-04) | 0 | 3 |
| albatross_linear_orig_layout_tune_4090_20260704 | Albatross linear_orig_layout 4090 tuning | 0 | 1 |
| albatross_v3a_v4_4090_tune_20260703 | Albatross v3a vs v4 4090 tune smoke — 2026-07-03 | 0 | 4 |
| albatross_v4_linear_policy_patch_4090_20260704 | Albatross v4 linear policy patch smoke — 4090 — 2026-07-04 | 0 | 3 |
| apple_coreml_state_contract_m5_20260707 | Apple CoreML stateful contract evidence | 3 | 0 |
| apple_decode_direct_step_m5_20260708 | Apple M5 decode direct-step experiment (2026-07-08) | 3 | 0 |
| apple_decode_eval_interval_m5_20260708 | Apple M5 decode eval-interval experiment (2026-07-08) | 1 | 0 |
| apple_e2e_scan_prefill_m5_20260707 | Apple M5 MLX WKV scan prefill end-to-end evidence (2026-07-07) | 2 | 0 |
| apple_e2e_scan_prefill_m5_20260708 | Apple M5 MLX WKV scan prefill second evidence batch (2026-07-08) | 4 | 0 |
| apple_e2e_smoke_m5_20260707 | Evidence directory; add README.md when promoting results. | 2 | 0 |
| apple_fast_group_norm_m5_20260708 | Apple M5 fast group norm experiment (2026-07-08) | 3 | 0 |
| apple_fast_layer_norm_m5_20260708 | Apple M5 fast layer norm experiment (2026-07-08) | 2 | 0 |
| apple_mlx_chunked_state_only_m5_20260707 | Apple M5 MLX chunked-prefill state-only seam | 2 | 0 |
| apple_mlx_component_profile_m5_20260707 | Apple MLX RWKV-7 component profile — 2026-07-07 | 1 | 0 |
| apple_mlx_decode_sync_m5_20260707 | Apple M5 MLX decode synchronization cleanup and attn-mix probe | 2 | 0 |
| apple_mlx_fused_ffn_relu2_m5_20260707 | Apple MLX fused FFN key relu² smoke — Apple M5, 2026-07-07 | 2 | 0 |
| apple_mlx_wkv_scan_m5_20260707 | Apple M5 MLX multi-token WKV scan prototype | 2 | 0 |
| apple_qwen35_08b_longctx_m5_20260707 | Apple M5 Qwen3.5 0.8B long-context comparison | 5 | 0 |
| apple_qwen35_08b_tokenonly_m5_20260707 | Apple Qwen3.5 0.8B MLX-VLM token-only vs RWKV 0.4B expanded smoke — 2026-07-07 | 3 | 0 |
| apple_qwen35_2b_tokenonly_m5_20260707 | Apple Qwen3.5 2B MLX-VLM token-only vs RWKV-7 1.5B MLX — 2026-07-07 | 7 | 0 |
| apple_qwen35_compare_scan_auto_m5_20260708 | Apple M5 Qwen3.5 comparison refresh with RWKV scan-prefill auto (2026-07-08) | 5 | 0 |
| apple_qwen35_goal_audit_m5_20260707 | Apple/Qwen3.5 goal audit — Apple M5, 2026-07-07 | 1 | 0 |
| apple_qwen35_live_m5_20260707 | Apple/Qwen3.5 live smoke — 2026-07-07 | 3 | 0 |
| apple_qwen35_mlx_vlm_group_m5_20260707 | Apple Qwen3.5 MLX-VLM vs RWKV group-quant pass smoke — 2026-07-07 | 3 | 0 |
| apple_qwen35_mlx_vlm_m5_20260707 | Apple Qwen3.5 MLX-VLM baseline smoke — 2026-07-07 | 3 | 0 |
| apple_rkv_quant_min_m5_20260707 | Apple MLX R/K/V quant-min activation smoke — 2026-07-07 | 5 | 0 |
| apple_scan_prefill_auto_m5_20260708 | Apple M5 MLX scan-prefill auto policy evidence (2026-07-08) | 4 | 0 |
| apple_step_eval_interval_15b_m5_20260707 | Apple MLX step-eval interval sweep — 1.5B/mm4 fused FFN, Apple M5, 2026-07-07 | 3 | 0 |
| apple_step_eval_interval_m5_20260707 | Apple MLX step eval interval smoke — 2026-07-07 | 5 | 0 |
| math500_acceptance_4090_20260703 | 4090 MATH500 avg@64 acceptance comparison — 2026-07-03 | 0 | 1 |
| math500_albatross_full_avg64_20260703 | Evidence directory; add README.md when promoting results. | 0 | 1 |
| math500_bsz_sweep_defer_text_4090_20260704 | MATH500 bsz sweep with deferred verification + text decode on RTX 4090 | 0 | 0 |
| math500_defer_text_decode_smoke_4090_20260704 | MATH500 deferred text-decode smoke on RTX 4090 | 0 | 0 |
| math500_defer_verification_smoke_4090_20260704 | MATH500 deferred verification smoke on RTX 4090 | 0 | 0 |
| math500_final_acceptance_5090_1p5b_20260705 | MATH500 final acceptance benchmark | 0 | 0 |
| math500_gap_4090_20260703 | 4090 MATH500 avg@64 HF vs Albatross gap analysis | 0 | 0 |
| math500_hf_dynamic_full_avg64_20260703 | Evidence directory; add README.md when promoting results. | 0 | 1 |
| math500_hf_seed43_bsz128_defer_text_full_compare_4090_20260704 | MATH500 seed43 bsz128 deferred-text HF vs Albatross comparison on RTX 4090 | 0 | 3 |
| math500_hf_seed43_full_compare_4090_20260704 | MATH500 seed43 full HF vs Albatross comparison on RTX 4090 | 0 | 3 |
| math500_high_signal9_4090_20260703 | 4090 MATH500 high-signal-9 rollout64 subset — 2026-07-03 | 0 | 3 |
| math500_logits_parity_4090_20260703 | HF vs Albatross logits parity probe | 0 | 0 |
| math500_rng_modes_high_signal9_4090_20260704 | 4090 MATH500 high-signal-9 RNG/refill probe — 2026-07-04 | 0 | 2 |
| math500_sampling_variance_4090_20260703 | 4090 MATH500 sampling/refill stochasticity — 2026-07-03 | 0 | 0 |
| math500_stratified64_seed_sweep_4090_20260704 | 4090 MATH500 stratified-64 HF seed sweep — 2026-07-04 | 0 | 2 |
| v100_native_jit_split_20260728 | Tesla V100 complete native-JIT facade split CUDA, W8/W4 and B1/B8 A/B regression | 6 | 5 |

## Top-level benchmark scripts

| Script | Category |
| --- | --- |
| analyze_math500_gap.py | analysis |
| analyze_math500_sampling_variance.py | analysis |
| analyze_results.py | analysis |
| audit_qwen35_apple_goal.py | analysis |
| bench.py | utility |
| bench_albatross.py | benchmark |
| bench_albatross_linear_orig_layout.py | benchmark |
| bench_albatross_projection_layout.py | benchmark |
| bench_batch.py | benchmark |
| bench_batch_sweep.py | benchmark |
| bench_chunked_prefill.py | benchmark |
| bench_decode_breakdown.py | benchmark |
| bench_decode_components.py | benchmark |
| bench_decode_micro.py | benchmark |
| bench_dplr_prefill_scan.py | benchmark |
| bench_dynamic_batch.py | benchmark |
| bench_fast_token_warmup.py | benchmark |
| bench_forward_fast_path.py | benchmark |
| bench_fused_attn_output.py | benchmark |
| bench_fused_attn_output_project.py | benchmark |
| bench_fused_ffn.py | benchmark |
| bench_fused_projection.py | benchmark |
| bench_fused_recurrent.py | benchmark |
| bench_fused_recurrent_output.py | benchmark |
| bench_fused_recurrent_scan.py | benchmark |
| bench_fused_rkv_wag_projection.py | benchmark |
| bench_fused_shift_mix.py | benchmark |
| bench_fused_wa_lora.py | benchmark |
| bench_fused_wag_lora.py | benchmark |
| bench_fused_wavg_lora.py | benchmark |
| bench_generate_fast_path.py | benchmark |
| bench_larger_model_smoke.py | benchmark |
| bench_marlin_bn_tn.py | benchmark |
| bench_marlin_bn_tn_contract.py | benchmark |
| bench_marlin_relu2.py | benchmark |
| bench_logit_compression_alignment.py | benchmark |
| bench_native_decode.py | benchmark |
| bench_native_graph_fused_output.py | benchmark |
| bench_native_graph_fused_output_project.py | benchmark |
| bench_native_graph_fused_projection.py | benchmark |
| bench_native_graph_fused_recurrent.py | benchmark |
| bench_native_graph_fused_recurrent_output.py | benchmark |
| bench_native_graph_fused_wag_lora.py | benchmark |
| bench_native_graph_fused_wavg_lora.py | benchmark |
| bench_native_graph_overhead.py | benchmark |
| bench_native_graph_vkwr_rkv_policy.py | benchmark |
| bench_native_mm_quant_decode.py | benchmark |
| bench_native_model_decode.py | benchmark |
| bench_native_prefill_breakdown.py | benchmark |
| bench_native_prefill_scan.py | benchmark |
| bench_native_quant_e2e_decode.py | benchmark |
| bench_native_quant_gemv.py | benchmark |
| bench_native_quant_mm4.py | benchmark |
| bench_native_quant_mm8.py | benchmark |
| bench_native_quant_rkv.py | benchmark |
| bench_native_quant_rkv_sweep.py | benchmark |
| bench_native_quant_w4_gemv.py | benchmark |
| bench_native_quant_w4_rkv.py | benchmark |
| bench_projection_lora.py | benchmark |
| bench_quantization.py | benchmark |
| bench_quant_bn_tn.py | benchmark |
| bench_speculative_decode.py | benchmark |
| bench_speed.py | benchmark |
| bench_ttft_tpot.py | benchmark |
| check_results.py | analysis |
| compare_albatross_logits.py | analysis |
| compare_fast_token_layouts.py | analysis |
| compare_math500_summaries.py | analysis |
| compare_qwen35_apple_baseline.py | analysis |
| eval_math500_hf.py | utility |
| make_math500_stratified_subset.py | utility |
| plot_train_temp_alignment.py | analysis |
| profile_decode.py | profile |
| profile_mlx_components.py | profile |
| run_blackwell_quant_matrix.py | orchestrator |
| run_coreml_apple_baseline.py | orchestrator |
| run_math500_final_acceptance.py | orchestrator |
| run_qwen35_apple_baseline.py | orchestrator |
| score_qwen35_quality.py | analysis |
| summarize_blackwell_quant_matrix.py | analysis |
| summarize_results.py | analysis |

## Notes

- `results*.jsonl` files at `bench/` root are legacy aggregate streams. Prefer creating a dated evidence directory for new work.
- `__pycache__/` is local runtime noise and should not be committed.
- Apple optimization experiments with mixed results should remain opt-in and documented as negative/mixed evidence.
