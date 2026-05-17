# Current benchmark inventory

This index lists only current retained evidence and current benchmark entry
points. The machine-readable source of truth is
[`CURRENT_ARTIFACTS.json`](CURRENT_ARTIFACTS.json). Historical tuning sweeps and
superseded RWKV FLA performance matrices are intentionally not retained.

## Current evidence

| Platform | Line | Artifact |
|---|---|---|
| RTX 3090 | Native quantization | [`3090_native_quant_20260713/`](3090_native_quant_20260713/README.md) |
| RTX 3090 | Paired Prefill/Decode | [`3090_qwen35_paired_pd_v2_20260816/`](3090_qwen35_paired_pd_v2_20260816/README.md) |
| RTX 4080 | 7.2B FP16-state Decode | [`4080_7p2b_fp16_state_20260809/`](4080_7p2b_fp16_state_20260809/README.md) |
| RTX 4080 | B8 projection kernel | [`4080_b8_projection_bmm_20260809/`](4080_b8_projection_bmm_20260809/README.md) |
| RTX 4080 | Paired Prefill/Decode | [`4080_qwen35_paired_pd_v1_20260814/`](4080_qwen35_paired_pd_v1_20260814/README.md) |
| RTX 4090 | Native route tuning | [`4090_4080_routes_20260812/`](4090_4080_routes_20260812/README.md) |
| RTX 4090 | 7.2B B8 quantization | [`4090_g1h_7p2_bsz8_20260715/`](4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 4090 | Paired Prefill/Decode | [`4090_qwen35_paired_pd_v2_20260815/`](4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 4090 | Small-model B8 quantization | [`4090_small_bsz8_20260715/`](4090_small_bsz8_20260715/README.md) |
| RTX 5070 Laptop | Native exact-card | [`5070_max_perf_20260811/`](5070_max_perf_20260811/README.md) |
| RTX 5070 Laptop | Native quant loading | [`5070_native_memory_loading_20260716/`](5070_native_memory_loading_20260716/README.md) |
| RTX 5090 | W4 BN/TN | [`5090_bntn_all_models_20260716/`](5090_bntn_all_models_20260716/README.md) |
| RTX 5090 | Native versus official | [`5090_native_official_fp16_production_20260718/`](5090_native_official_fp16_production_20260718/README.md) |
| RTX 5090 | Official training-math alignment | [`5090_train_temp_alignment_20260717/`](5090_train_temp_alignment_20260717/README.md) |
| RTX 5090 | Native B16 shell-shape training | [`5090_native_train_temp_b16_20260718/`](5090_native_train_temp_b16_20260718/README.md) |
| RTX 5090 | Native real-MiniPile training | [`5090_native_train_temp_real_minipile_20260718/`](5090_native_train_temp_real_minipile_20260718/README.md) |
| RTX 5090 | Qwen reference | [`5090_qwen35_best_optimized_hf_v1_20260813/`](5090_qwen35_best_optimized_hf_v1_20260813/README.md) |
| RTX 5090 | Paired Decode | [`5090_qwen35_paired_decode_v1_20260813/`](5090_qwen35_paired_decode_v1_20260813/README.md) |
| AMD gfx1100 | Native exact-card close | [`amd_gfx1100_full_close_20260730/`](amd_gfx1100_full_close_20260730/README.md) |
| AMD gfx1100 | Quantization | [`amd_gfx1100_quant_20260728/`](amd_gfx1100_quant_20260728/README.md) |
| Apple M5 | B1 paired comparison | [`apple_bsz1_active_m5_20260715/`](apple_bsz1_active_m5_20260715/README.md) |
| Apple M5 | B8 paired comparison | [`apple_bsz8_active_m5_20260714/`](apple_bsz8_active_m5_20260714/README.md) |
| RTX 5090 | MATH500 final acceptance | [`math500_final_acceptance_5090_1p5b_20260705/`](math500_final_acceptance_5090_1p5b_20260705/README.md) |
| Moore Threads S70 | Shift-mix kernel | [`musa_s70_shift_mix_20260728/`](musa_s70_shift_mix_20260728/README.md) |
| Moore Threads S70 | Native compatibility | [`musa_s70_validation_20260728/`](musa_s70_validation_20260728/README.md) |
| Tesla T4 | Native exact-card | [`t4_production_close_20260720/`](t4_production_close_20260720/README.md) |
| Tesla V100 | FP16 recurrent state | [`v100_exact_card_20260811/`](v100_exact_card_20260811/README.md) |
| Tesla V100 | Dense/Albatross and serving | [`v100_production_close_20260711/`](v100_production_close_20260711/README.md) |
| Tesla V100 | Paired Prefill/Decode | [`v100_qwen35_paired_pd_v1_20260814/`](v100_qwen35_paired_pd_v1_20260814/README.md) |
| Tesla V100 | Packed MM4 Decode | [`v100_sm70_mm4_bntn_20260716/`](v100_sm70_mm4_bntn_20260716/README.md) |
| Tesla V100 | W4 Prefill | [`v100_sm70_prefill_dequant_20260723/`](v100_sm70_prefill_dequant_20260723/README.md) |
| 2x Tesla V100 | Transformers tensor parallel | [`v100_transformers_tp_20260726/`](v100_transformers_tp_20260726/README.md) |

## Current entry points

- General Native performance: `bench_batch_sweep.py`,
  `bench_native_prefill_scan.py`, `bench_native_graph_overhead.py`.
- Cross-model matrices: `bench_cross_model_speed.py` and
  `bench_cross_model_speed_resident.py`.
- Current paired protocols: `run_3090_rwkv_paired_pd_v2.sh`,
  `run_4080_rwkv_paired_pd_v1.sh`, `run_4090_rwkv_paired_pd_v2.sh`,
  `run_5090_rwkv_paired_decode_v1.sh`, and
  `run_v100_rwkv_paired_pd_v1.sh`.
- Validation: `validate_qwen35_3090_paired_pd_v2.py`,
  `validate_qwen35_4090_paired_pd_v2.py`,
  `validate_qwen35_paired_pd_v1.py`,
  `validate_qwen35_paired_decode_v1.py`, and
  `validate_qwen35_v100_paired_pd_v1.py`.
- Quantization: `bench_native_quant_e2e_decode.py`,
  `bench_native_mm_quant_decode.py`, and `bench_sm70_w4_bn_tn.py`.
- Training and quality: `bench_train_temp_alignment.py` and
  `run_math500_final_acceptance.py`.

FLA remains available as an explicit compatibility/reference backend and as a
correctness oracle. It is not the retained RWKV performance route.
