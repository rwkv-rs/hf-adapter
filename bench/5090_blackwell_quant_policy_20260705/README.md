# RTX 5090 native MM8/MM4 quantization policy sweep (2026-07-05)

This artifact validates the `native_mm_policy` split added for repository-native
MM8/MM4 quantization:

- `memory`: historical size-gated behavior; quantizes every `nn.Linear` with
  `weight.numel() >= min_params`.
- `speed`: quantizes only `lm_head` after the same size gate, keeping recurrent
  and FFN decode projections dense until fused quantized RWKV block kernels are
  available.

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32GB
- Driver: 610.43.02
- Torch: 2.6.0a0+ecf3bae40a.nv25.01, CUDA 12.8
- Transformers: 4.56.2
- Backend: `RWKV7_NATIVE_MODEL=1`, `attn_mode=fused_recurrent`, fp16
- Benchmark: `bench/bench_native_mm_quant_decode.py`, prompt 128, cached decode

## Key result

The old size-gated policy is good for footprint but bad for cached decode once
per-layer FFN/projection matrices cross the threshold. On RTX 5090, replacing
49/65 large modules cuts 1.5B/2.9B decode speed to about 0.47x fp16. The new
`speed` policy replaces only `lm_head` and restores decode to ~0.98-1.00x fp16
while still reducing the model footprint.

The e2e logits check also keeps prompt/final logits aligned with fp16. On
1.5B/2.9B, MM8 cosine stays above `0.99999` and MM4 stays above `0.99976`.
The 7.2B speed rows also pass on RTX 5090: MM8 decode is slightly above fp16,
MM4 is effectively equal to fp16, and both keep the same greedy next token.

### 1.5B min-params sweep

| min_params | quantization | policy equivalent | replaced modules | decode tok/s | ratio vs fp16 | footprint MB |
|---:|---|---|---:|---:|---:|---:|
| 8,000,000 | fp16 | dense | 0 | 108.3 | 1.000 | 2913.3 |
| 8,000,000 | native_mm8 | memory | 49 | 51.6 | 0.477 | 2019.4 |
| 8,000,000 | native_mm4 | memory | 49 | 50.8 | 0.469 | 1571.4 |
| 50,000,000 | fp16 | dense | 0 | 104.8 | 1.000 | 2913.3 |
| 50,000,000 | native_mm8 | lm_head-only | 1 | 103.2 | 0.985 | 2785.6 |
| 50,000,000 | native_mm4 | lm_head-only | 1 | 102.8 | 0.981 | 2721.6 |
| 100,000,000 | fp16 | dense | 0 | 107.6 | 1.000 | 2913.3 |
| 100,000,000 | native_mm8 | lm_head-only | 1 | 106.9 | 0.994 | 2785.6 |
| 100,000,000 | native_mm4 | lm_head-only | 1 | 107.6 | 1.000 | 2721.6 |

### 1.5B explicit speed policy

| quantization | policy | replaced modules | decode tok/s | ratio vs fp16 | footprint MB |
|---|---|---:|---:|---:|---:|
| fp16 | speed | 0 | 106.8 | 1.000 | 2913.3 |
| native_mm8 | speed | 1 | 105.3 | 0.986 | 2785.6 |
| native_mm4 | speed | 1 | 105.4 | 0.987 | 2721.6 |

### 2.9B memory vs speed policy

| quantization | policy | replaced modules | decode tok/s | ratio vs fp16 | footprint MB |
|---|---|---:|---:|---:|---:|
| fp16 | memory | 0 | 80.6 | 1.000 | 5622.4 |
| native_mm8 | memory | 65 | 38.2 | 0.474 | 3865.7 |
| native_mm4 | memory | 65 | 37.9 | 0.470 | 2985.7 |
| fp16 | speed | 0 | 80.9 | 1.000 | 5622.4 |
| native_mm8 | speed | 1 | 80.1 | 0.990 | 5462.6 |
| native_mm4 | speed | 1 | 79.2 | 0.979 | 5382.6 |

### 7.2B explicit speed policy

7.2B was converted and run on the same RTX 5090. Because the 32GB card is close
to the limit, W8/W4 rows were run in fresh processes. The first combined
none/mm8/mm4 attempt showed mm4 OOM only after a prior mm8 run had already
peaked at ~31GB; fresh-process mm4 passes with peak ~17.3GB.

| quantization | policy | replaced modules | decode tok/s | ratio vs fp16 | footprint MB |
|---|---|---:|---:|---:|---:|
| fp16 | speed | 0 | 81.3 | 1.000 | 13731.3 |
| native_mm8 | speed | 1 | 81.7 | 1.005 | 13475.5 |
| native_mm4 | speed | 1 | 81.7 | 1.005 | 13347.5 |

### Speed-policy e2e logits check

This uses `bench/bench_native_quant_e2e_decode.py --policy speed` and records
footprint ratio, decode-speed ratio, prompt/final logits cosine vs fp16, and
greedy next-token equality.

| model | quantization | footprint ratio | speed ratio | prompt cosine | final cosine | same next token |
|---|---|---:|---:|---:|---:|---|
| 1.5B | mm8 | 0.9562 | 0.9841 | 0.99999499 | 0.99999452 | true |
| 1.5B | mm4 | 0.9342 | 0.9860 | 0.99982727 | 0.99983704 | true |
| 2.9B | mm8 | 0.9716 | 0.9975 | 0.99999589 | 0.99999553 | true |
| 2.9B | mm4 | 0.9573 | 0.9706 | 0.99983841 | 0.99976456 | true |
| 7.2B | mm8 | 0.9814 | 1.0074 | 0.99999332 | 0.99999321 | true |
| 7.2B | mm4 | 0.9720 | 0.9988 | 0.99946028 | 0.99944884 | true |

`peak_vram_mb` in the raw rows includes temporary quantization/packing buffers;
`model_footprint_mb` is the stable model footprint used for memory comparison.

## Reproduction

```bash
export RWKV7_NATIVE_MODEL=1
export HF_MODULES_CACHE=/tmp/hf_modules_rwkv7_5090_policy
PYTHONPATH=. python bench/bench_native_mm_quant_decode.py \
  --hf-dir /path/to/rwkv7_g1g_29b_hf \
  --model-size-label 2.9b \
  --dtype fp16 --device cuda \
  --quantizations none mm8 mm4 \
  --min-params 8000000 \
  --policy speed \
  --prompt-tokens 128 --decode-tokens 24 --warmup 1 --runs 1 \
  --results bench/results.jsonl
```

Raw rows:

- `results_5090_quant_minparams_15b.jsonl`
- `results_5090_quant_policy_speed_15b.jsonl`
- `results_5090_quant_policy_29b.jsonl`
- `results_5090_quant_policy_e2e_logits.jsonl`
- `results_5090_quant_policy_72b_speed_decode8.jsonl`
- `results_5090_quant_policy_72b_e2e_logits.jsonl`
- `results_5090_quant_policy_72b_e2e_logits_mm4_fresh.jsonl`
