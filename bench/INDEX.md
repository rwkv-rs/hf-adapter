# Bench inventory

Generated inventory of benchmark scripts and evidence directories. Keep this file lightweight: it is an orientation map, not the source of truth for benchmark conclusions.

## Promoted production-close artifacts

| Platform | Artifact | Current conclusion |
|---|---|---|
| V100 32GB | [`v100_production_close_20260711/`](v100_production_close_20260711/README.md) | Dense Albatross P1 and native W8/W4 speed lane pass |
| RTX 3090 | [`3090_g1h_7p2_bsz8_20260714/`](3090_g1h_7p2_bsz8_20260714/README.md) | Latest g1h 7.2B/9B bsz8 dense, active-work, W8/W4 speed and memory gates pass 18/18 |
| RTX 3090 | [`3090_self_fused_20260713/`](3090_self_fused_20260713/README.md) | 7.2B/9B prompt-2048 batch-1/2 self-fused dense gates pass |
| RTX 4090 | [`4090_g1h_7p2_bsz8_20260715/`](4090_g1h_7p2_bsz8_20260715/README.md) | Latest g1h 7.2B/9B bsz8 dense, active-work, W8/W4 speed and quant-local memory gates pass 18/18 |
| RTX 4090 | [`4090_validation_summary.md`](4090_validation_summary.md) | Measured dense decode/current-session prefill and quant speed lanes pass |
| RTX 5090 | [`5090_blackwell_production_close_20260712/`](5090_blackwell_production_close_20260712/README.md) | Quant pressure, 13.3B conversion and full MATH500 pass |
| Apple M5 | [`../docs/hardware/APPLE_PRODUCTION_CLOSE.md`](../docs/hardware/APPLE_PRODUCTION_CLOSE.md) | Selected MLX/Qwen3.5 production pairs pass |

Canonical cross-platform summary: [`../BENCHMARK.md`](../BENCHMARK.md) and
[`../docs/HARDWARE_MATRIX.md`](../docs/HARDWARE_MATRIX.md).

## Apple M5 production-close evidence

The `apple_production_close_*_m5_20260711.jsonl` top-level files contain the
checked Qwen3.5 0.8B/2B baselines, RWKV-7 0.4B compiled W4 rows, RWKV-7 1.5B
W4/W8 compile rows, full-context RWKV draft speculation, and the final
two-pair conservative gate. Conclusions and reproduction commands are in
[`../docs/hardware/APPLE_PRODUCTION_CLOSE.md`](../docs/hardware/APPLE_PRODUCTION_CLOSE.md).

## Evidence directories

| Directory | Title / purpose | JSONL | Logs |
| --- | --- | --- | --- |
| 3090_g1h_7p2_bsz8_20260714 | RTX 3090 latest-g1h 7.2B vs Qwen3.5-9B bsz8 acceptance | 5 | 5 |
| 3090_self_fused_20260713 | RTX 3090 self-fused RWKV-7 7.2B long-prefill close | 2 | 0 |
| 4090_g1h_7p2_bsz8_20260715 | RTX 4090 latest-g1h 7.2B vs Qwen3.5-9B bsz8 acceptance | 6 | 7 |
| 5090_blackwell_hf_matrix_20260704 | RTX 5090 Blackwell HF validation matrix (2026-07-04) | 2 | 12 |
| 5090_blackwell_native_prefill_smoke_20260704 | RTX 5090 Blackwell native-prefill validation (2026-07-04) | 2 | 3 |
| 5090_blackwell_native_quant_20260704 | RTX 5090 native quant benchmark (2026-07-04) | 1 | 2 |
| 5090_blackwell_quant_matrix_20260705 | RTX 5090 Blackwell native MM8/MM4 fresh-process quant matrix (2026-07-05) | 1 | 0 |
| 5090_blackwell_quant_policy_20260705 | RTX 5090 native MM8/MM4 quantization policy sweep (2026-07-05) | 7 | 0 |
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
