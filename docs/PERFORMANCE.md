# Production performance status

This page contains only promoted current conclusions. Exploratory tuning and
historical rows remain in platform documents and `bench/` artifacts.

The `v0.6.0` HF deliverable is complete for the promoted profiles below.
Additional cards, models, and shapes are post-release matrix expansion rather
than blockers for the released adapter.

## Current promoted lanes

| Platform | Dense fp16/bf16 | Quant speed lane | Quality/correctness | Status |
|---|---|---|---|---|
| RTX 3090 | g1h 7.2B vs full-FLA Qwen3.5-9B bsz8 prefill/decode minimum `1.058907x/1.788418x`; decode active-work minimum `1.437946x` | W8 total/decode minimum `1.098658x/1.084305x`; W4 `1.014527x/1.025666x`; footprint and peak lower in 18/18 | finite logits, 24/24 Qwen FLA bindings and fail-closed route checks; task quality not measured | Production-close for measured bsz8 lane |
| RTX 5070 Laptop | 1.5B/full-FLA-Qwen B8 minimum `1.082707x/1.795119x`; exact Native 0.4B/1.5B graph+scan, raw recurrent `1.0272x-1.1265x`, promoted norm/mix `1.0373x-1.1630x` | full-FLA fp16/W8/W4 pass; Native B8 FP16 state saves `16.875-58.125 MiB` | full-FLA bindings pass; Native prefill handoff and decode greedy/cosine gates pass; 1.5B/B4 norm/mix fails closed | Production-close for measured full-FLA and Native lanes |
| RTX 4080 | 0.4B/0.8B, 1.5B/2B and 2.9B/4B B1/B8 full-FLA-Qwen matrices; all six parameter-adjusted Prefill/Decode medians pass at `1.06x–1.43x` / `1.15x–2.93x`; adjusted E2E medians are `1.14x–2.88x`; exact-B8 grouped projection and 7.2B/B8 FP16-state lanes remain | output-head A8W8/W4 complete-cell minima `1.003101x/1.015996x`; full-model BNB routes lower footprint; 7.2B/B8 FP16 state saves median `123.88 MiB`; 13.3B MM8/MM4 fit | exact-shape FP16-accumulation promotion passes greedy `15/15` for both Prefill and Decode with cosine above `0.9999`; Qwen full-FLA contracts, 7.2B state, and output-head gates pass | Production-close for measured Native HF B1/B8 and capacity lanes |
| V100 | Albatross/full-FLA-Qwen lanes remain; exact-B8 WAVG adds `1.0114x-1.0312x`; exact 0.4B/1.5B B8 FP16 state adds `1.0216x-1.0288x` | W8/W4 lanes remain; FP16 state saves `16.875-58.125 MiB` | Existing greedy/cache gates pass; FP16-state processes match `4,096/4,096` greedy tokens per model across both orders | Production-close for measured lanes |
| RTX 4090 | g1h 7.2B vs full-FLA Qwen3.5-9B bsz8 prefill/decode minimum `1.023951x/2.210065x`; decode active-work minimum `1.776961x` | W8 total/decode minimum `1.360072x/1.356914x`; W4 `1.013273x/1.022724x`; selected quant footprint and peak lower in 12/12 | finite logits, 24/24 Qwen optimized bindings, BNB8/MM4 cosine+greedy probes; task quality not measured | Production-close for measured bsz8 lane |
| RTX 4090 small models | 0.4B/0.8B, 1.5B/2B, 2.9B/4B bsz8 dense prefill minima `1.370369x/1.041959x/1.305103x`; decode minima `12.101818x/5.636846x/4.214362x` | W8 total minima `1.011441x/1.131672x/1.176050x`; W4 `1.029994x/1.027211x/1.014959x`; footprint and peak lower in 36/36 selected quant cells | finite logits, full-Qwen-FLA dense contract, active-work and fail-closed route gates; task quality not measured | Production-close for measured bsz8 lanes |
| RTX 5090 Qwen matrix | 0.4B/0.8B through 7.2B/9B at B1/B8; raw prefill/decode minima `1.0226x/2.8130x`; per-active-B throughput leads in 144/144 cells | W8/W4 exact-cell total-latency and footprint gates pass in all measured cells | 144/144 full-FLA Qwen contracts and 32/32 greedy reports pass; active-work prefill and dense peak-VRAM are not universal wins | Production-close for measured B1/B8 lanes |
| RTX 5090 BF16/W4 | g1h 1.5B/2.9B/7.2B/13.3B paired BF16 at B1/B8, prompt128/decode128 | all-phase prefill/decode minima `1.0010x/1.1854x`; footprint `0.5298x–0.6250x`; model-level head/final-layer policy is automatic | prompt/final cosine `>=0.9995`, same-next 8/8; 280/280 group-128 grid contract | Production-close for measured all-phase W4 matrix |
| RTX 5090 MATH500 / 13.3B | 0.4B MATH500 generation `16,925.6 tok/s`, steady decode `19,339.5 tok/s`; latest g1h 13.3B load/generate passes | 13.3B selected speed-policy MM8/MM4 decode `1.0013x/0.9845x` paired fp16 with footprint `0.9899x/0.9848x` | MATH500 pass@64 `0.38`; 13.3B cosine above `0.99985` and same-next pass | Production-close artifacts |
| RTX 5090 Native fp16-state | official g1h 7.2B cached decode B1/B8 is `1.0010x/1.0104x` pinned v3a; 2.9B/13.3B B1/B8 prompt128/512/2048 prefill passes 12/12 at `1.0029x–1.5690x` | not a quant lane; no cross-harness memory-parity claim | decode logits/state/xpa/xpf and greedy pass; prefill logits/layers/state/xpa/xpf/first tokens pass | Exact-card default-policy pass |
| RTX 5090 Native train_temp | L12/D768/FFN3072 BF16 B16/T512 paired real-MiniPile median is `1.00049x` official; continuous 5,000-step is `1.00255x` | not a quant lane; steady allocated/reserved deltas are `-1.375/-188 MiB` | exact 399 gradients/deltas; 3-seed, 5,000-step and 2,500+2,500 resume gates pass | Exact single-card lane pass |
| Apple M5 | Tiled DPLR and guarded compiled decode close selected same-device Qwen3.5 gates | W4 lowers memory; selected production pair gates pass | target-greedy oracle and state/session checks pass | Production-close for measured MLX pairs |
| AMD gfx1100 | Exact-card fused decode remains positive through 13.3B (`1.2131x-2.0402x` conservative native across measured B1/B8 rows) | Output-head W8/W4 beats fp16 cached decode in 40/40 0.4B-13.3B B1/B2/B4/B8 rows; full-model quant remains below fp16 | Dense and head-quant greedy parity pass; 2.9B full-model W4 quality fails | Validated named decode lanes |

V100 optimized-Qwen evidence:
[`v100_active_b1b8_20260715`](../bench/v100_active_b1b8_20260715/README.md),
plus exact-B8 launch evidence:
[`4080_v100_decode_tuning_20260808`](../bench/4080_v100_decode_tuning_20260808/README.md),
and exact B8 state evidence:
[`v100_exact_card_20260811`](../bench/v100_exact_card_20260811/README.md).

RTX 5070 exact Native evidence:
[`5070_max_perf_20260811`](../bench/5070_max_perf_20260811/README.md).

RTX 4080 Native HF and optimized-Qwen evidence:
[`4080_adjusted_pd_20260811`](../bench/4080_adjusted_pd_20260811/README.md),
[`4080_full_model_ladder_20260719`](../bench/4080_full_model_ladder_20260719/README.md)
and exact-B8 grouped projection evidence:
[`4080_b8_projection_bmm_20260809`](../bench/4080_b8_projection_bmm_20260809/README.md),
plus 7.2B/B8 FP16-state evidence:
[`4080_7p2b_fp16_state_20260809`](../bench/4080_7p2b_fp16_state_20260809/README.md).

RTX 5090 evidence:
[`5090_g1h_qwen35_b1_b8_20260715`](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md)
[`5090_g1h_13p3_20260715`](../bench/5090_g1h_13p3_20260715/README.md), and
[`5090_bntn_all_models_20260716`](../bench/5090_bntn_all_models_20260716/README.md).
Native HF decode and UI/official-shell evidence:
[`5090_native_decode_fused_20260718`](../bench/5090_native_decode_fused_20260718/README.md) and
[`5090_native_hf_gradio_train_temp_20260718`](../bench/5090_native_hf_gradio_train_temp_20260718/README.md).
Native B16 train_temp evidence:
[`5090_native_train_temp_real_minipile_20260718`](../bench/5090_native_train_temp_real_minipile_20260718/README.md).
Native same-precision inference evidence:
[`5090_native_official_fp16_production_20260718`](../bench/5090_native_official_fp16_production_20260718/README.md).

## Interpretation rules

1. Compare the same model/checkpoint, dtype, prompt/decode shape and device.
2. Prefer paired same-process timing for quant-vs-fp comparisons.
3. Preserve both current-session and historical high-water references.
4. A load/generate smoke is not a performance result.
5. Aggregate batch throughput and per-sequence latency must not be conflated.
6. MATH500 speed claims must retain shape, seed, rollout count and accuracy gates.

## Post-release performance expansion

- Extend the RTX 5090 Marlin W4 matrix from selected FFN pairs to still-dense
  square projections and the rejected 0.4B full-FFN shape; reproduce an
  all-phase large-payload win for W8 and the remaining declared cards.
- Extend P2/P3 Albatross matrices to larger models and more hardware. The
  existing V100/4090/5090 reference lanes and 3090/4080/4090/5070/5090 Qwen
  pair lanes are already promoted and should not be listed as missing.
- Extend the exact RTX 5090 Native official-reference matrix beyond the closed
  7.2B B1/B8 decode and 2.9B/13.3B 12-cell prefill scopes, especially 7.2B
  sequence prefill, additional checkpoints and same-harness memory telemetry.
- Add the still-missing RTX 4080 7.2B/B8 same-card Qwen/Albatross comparison
  and full-model W8/W4 lane. Its internal FP16-state decode speed, memory and
  correctness gate is already closed.
- Recover the retained 0.4B RTX 4090 historical prompt-512 prefill high-water
  mark; the separate g1h 7.2B/Qwen3.5 bsz8 lane is closed.
- Add H100 production evidence. AMD/ROCm already has fused decode and 40/40
  output-head W8/W4 wins; remaining AMD work is fused prefill, full-model quant
  quality/speed, MI-series and same-card reference evidence.
- Broaden Apple results beyond M5 and complete CoreML/ANE production telemetry.

## Reproduction entrypoints

- General speed: `bench/bench_speed.py`, `bench/bench_batch_sweep.py`
- TTFT/TPOT: `bench/bench_ttft_tpot.py`
- Albatross ingestion/comparison: `bench/bench_albatross.py`
- Native quant matrix: `bench/run_blackwell_quant_matrix.py`
- Native cached decode: `bench/bench_native_model_decode.py`
- Native fused correctness: `bench/bench_native_model_decode_alignment.py`
- MATH500 final runner: `bench/run_math500_final_acceptance.py`
- Apple same-device runner: `scripts/run_qwen35_apple_acceptance.sh`

Numeric summary: [`../BENCHMARK.md`](../BENCHMARK.md). Kernel roadmap:
[`performance/FUSED_BACKEND.md`](performance/FUSED_BACKEND.md).
