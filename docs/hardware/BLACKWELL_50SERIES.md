# Blackwell / NVIDIA 50-series

The Blackwell family is supported through the Native HF backend. Exact-card
policy is intentionally conservative: RTX 5090 and RTX 5070 Laptop routes are
selected only for the model, dtype, batch, prompt, and decode shapes recorded
in `rwkv7_hf/kernel_policy.py`. Adjacent 50-series products do not inherit an
exact-card route without their own evidence.

## Current evidence

| Card | Line | Result | Evidence |
|---|---|---|---|
| RTX 5070 Laptop | Native FP16 Prefill/Decode | 0.4B/1.5B, B1/B2/B4/B8 exact-card routes pass | [`5070_max_perf_20260811`](../../bench/5070_max_perf_20260811/README.md) |
| RTX 5070 Laptop | Native quantized loading | Accepted bounded memory/loading scope | [`5070_native_memory_loading_20260716`](../../bench/5070_native_memory_loading_20260716/README.md) |
| RTX 5090 | Native versus official RWKV | 7.2B cached Decode and 2.9B/13.3B sequence Prefill pass | [`5090_native_official_fp16_production_20260718`](../../bench/5090_native_official_fp16_production_20260718/README.md) |
| RTX 5090 | Qwen3.5 paired Decode | Parameter-adjusted Decode passes 48/48 | [`5090_qwen35_paired_decode_v1_20260813`](../../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |
| RTX 5090 | Frozen Qwen3.5 reference | 48/48 optimized reference rows pass | [`5090_qwen35_best_optimized_hf_v1_20260813`](../../bench/5090_qwen35_best_optimized_hf_v1_20260813/README.md) |
| RTX 5090 | W4 BN/TN | g1h 1.5B/2.9B/7.2B/13.3B B1/B8 routes pass | [`5090_bntn_all_models_20260716`](../../bench/5090_bntn_all_models_20260716/README.md) |
| RTX 5090 | Native training | Official-math, B16 shell-shape, and real-MiniPile gates pass | [`5090_train_temp_alignment_20260717`](../../bench/5090_train_temp_alignment_20260717/README.md), [`5090_native_train_temp_b16_20260718`](../../bench/5090_native_train_temp_b16_20260718/README.md), [`5090_native_train_temp_real_minipile_20260718`](../../bench/5090_native_train_temp_real_minipile_20260718/README.md) |

## Runtime policy

- Native/no-FLA is the performance route.
- Cached Decode uses `native_graph` only for exact accepted shapes.
- Prefill graph, fused scan/state preparation, norm/mix, recurrent output,
  projection, sparse FFN, and FP16 recurrent-state choices are controlled by
  `rwkv7_hf/kernel_policy.py` and explicit benchmark overrides.
- RTX 5090 and RTX 5070 Laptop have separate policies; a shared `sm_120`
  capability is not sufficient to reuse an exact-card tuning row.
- Unsupported shapes fall back to the compatible Native/PyTorch path. A formal
  performance run must fail closed when its requested optimized route is not
  selected or effective.

FLA remains optional for compatibility and cross-backend correctness checks.
Superseded RWKV FLA performance artifacts are not part of the current result
set.

## Quantization

The promoted RTX 5090 W4 line uses exact model profiles, BF16 activations,
group-128 Marlin FFN weights, and a dense or TorchAO-W4 output head selected per
model. The measured 1.5B/2.9B/7.2B/13.3B B1/B8 profiles reduce footprint to
`0.5298x-0.6250x` of BF16 and pass their Prefill, Decode, cosine, and next-token
gates. The 0.4B full-FFN candidate remains rejected.

Use the artifact command rather than forcing a profile on another model or
card:

```bash
python bench/bench_native_quant_e2e_decode.py \
  --hf-dir /path/to/rwkv7-g1h-2.9b-hf --model-size-label 2.9b \
  --dtype bf16 --device cuda --attn-mode fused_recurrent \
  --fast-cache true --fast-token-backend native_graph \
  --single-quantization torchao_w4 --min-params 1 --policy speed \
  --batch-size 8 --prompt-tokens 128 --decode-tokens 128 \
  --warmup 1 --timing-repeats 5 --paired-baseline \
  --results /tmp/rtx5090-w4.jsonl
```

## Validation rules

Before promoting a new 50-series card or route, record:

1. exact GPU name, compute capability, driver, CUDA, PyTorch, Triton, model
   hash, dtype, and shape matrix;
2. requested, selected, and effective backend telemetry with no fallback;
3. finite logits, greedy/cache continuity, and the route-specific cosine gate;
4. repeated Prefill/Decode timing and peak VRAM;
5. an exact-card policy entry and a new retained evidence bundle in
   [`../../bench/CURRENT_ARTIFACTS.json`](../../bench/CURRENT_ARTIFACTS.json).

For current numbers, use [`../QWEN35_LATEST_P_D_TOKPS.md`](../QWEN35_LATEST_P_D_TOKPS.md)
and [`../../BENCHMARK.md`](../../BENCHMARK.md).
