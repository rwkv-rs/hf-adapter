# Apple Silicon production-close evidence

This page records the bounded Apple M5 production gate for the RWKV-7 HF
adapter. It is deliberately based on same-machine, same-prompt, repeated live
runs rather than theoretical FLOP or package-size comparisons.

## Scope

- Host: Apple M5, 16 GB unified memory, macOS 26.5.
- Runtime: MLX 0.32.0, Python 3.11.
- Shape: batch 1, 512 prompt characters, 64 generated tokens, one warmup and
  three measured repeats.
- Baselines: local MLX Qwen3.5 0.8B 4-bit and Qwen3.5 2B 4-bit snapshots.
- RWKV pairs: RWKV-7 0.4B W4 and RWKV-7 1.5B W4.
- Aggregation is conservative: minimum observed prefill/decode throughput,
  maximum TTFT, and maximum peak memory.

This is an M5 gate, not a claim that every M-series generation or every prompt
shape has already been measured.

The original table below is the batch-1 production-close run. A newer,
separately collected batch-8 target-only gate is recorded in the next section;
the two thermal sessions must not be combined into a synthetic comparison.

## Result

| Pair | Decode ratio | Prefill ratio | TTFT ratio | Peak-memory ratio | Gate |
|---|---:|---:|---:|---:|---|
| RWKV-7 0.4B W4 / Qwen3.5 0.8B W4 | 2.374x | 1.546x | 0.679x | 0.520x | PASS |
| RWKV-7 1.5B W4 + 0.1B draft / Qwen3.5 2B W4 | 1.733x | 1.307x | 0.804x | 0.774x | PASS |

Lower is better for TTFT and memory. The checked gate requires decode and
prefill ratios >=1.0, TTFT ratio <=1.1, and memory ratio <=1.0.

Absolute conservative values:

| Runtime | Decode tok/s | Prefill tok/s | TTFT s | Peak bytes |
|---|---:|---:|---:|---:|
| Qwen3.5 0.8B MLX W4 | 81.24 | 1,071.73 | 0.1185 | 980,667,349 |
| RWKV-7 0.4B MLX W4 | 192.83 | 1,657.00 | 0.0804 | 509,478,732 |
| Qwen3.5 2B MLX W4 | 87.06 | 968.73 | 0.1311 | 2,193,081,277 |
| RWKV-7 1.5B MLX W4 + draft | 150.90 | 1,266.34 | 0.1054 | 1,697,520,708 |

Canonical evidence is in
[`bench/apple_production_close_qwen35_gate_m5_20260711.jsonl`](../../bench/apple_production_close_qwen35_gate_m5_20260711.jsonl).
The comparison summary has `status="pass"`, two passing comparisons, and no gap
actions.

## Batch-8 1.5B target-only close (2026-07-15)

This is the stricter no-assistance lane for the 1.5B model:

- true batch 8, 512 prompt characters, 133 RWKV target tokens, and 64 decoded
  tokens per sequence;
- RWKV-7 1.5B group-128 W4 versus Qwen3.5 2B MLX group-64 W4;
- isolated child processes, one warmup and three retained repeats in ABBA
  order, with an initial 60-second cooldown and 30 seconds between engines;
- no draft model, no speculative acceptance, and no prefix-state coalescing;
- throughput normalized as aggregate tok/s multiplied by active text parameter
  count. Raw peak memory is used for the memory gate.

| Metric | RWKV-7 1.5B | Qwen3.5 2B | Active-normalized ratio | Gate |
|---|---:|---:|---:|---|
| Prefill | 2,249.15 tok/s | 1,600.50 tok/s | 1.1406x | PASS |
| Decode | 185.59 tok/s | 132.20 tok/s | 1.1394x | PASS |
| Raw peak memory | 1,790,200,768 bytes | 2,151,577,894 bytes | n/a | PASS |

The closing change is a specialized B8/T1 NAX W4 FFN-key kernel using
`BM32/BK64/BN64/WM2/WN2` and fusing ReLU-squared into the quantized matmul.
A same-process alternating A/B isolates that change at `1.1549x` median decode
speedup, with exact generated tokens. The post-change fidelity suite passes W4
versus fp16 greedy equality, fused-versus-generic greedy equality, prefix-cache
exactness, and a real 1.5B-target/0.1B-draft mismatch oracle.

Canonical evidence and reproduction instructions are in
[`bench/apple_bsz8_active_m5_20260714/README.md`](../../bench/apple_bsz8_active_m5_20260714/README.md).
The fail-closed entry point is:

```bash
scripts/run_apple_bsz8_target_only_acceptance.sh
```

This closes only the checked M5/B8/T133/decode64 target-only profile. It does
not establish the same ratio on other M-series chips, batch sizes, prompt
lengths, decode lengths, or thermal conditions.

## What changed

### Native MLX groupwise W8/W4

`--rwkv-quant-backend groupwise` stores weights in MLX's native packed affine
layout and dispatches `mx.quantized_matmul`. It does not materialize a persistent
dense dequantized model. Unit coverage checks W8/W4 output shape, finite output,
bounded dense-reference error, packed-storage reduction, dispatch telemetry,
and model integration.

The direct 1.5B compile rows also show that quantization is not only a memory
feature:

- W4 compiled: minimum 55.81 tok/s, peak about 1.77 GB.
- W8 compiled: minimum 46.12 tok/s, peak about 2.24 GB.
- Both 64-token compile validations preserve greedy tokens and pass bounded
  logits/state gates.
- The earlier 1.5B fp16 fast path was about 33.64 tok/s with about 3.1 GB peak,
  so both packed modes improve decode speed while reducing memory on this M5.

Evidence:

- [`bench/apple_production_close_groupwise_w4_15_m5_20260711.jsonl`](../../bench/apple_production_close_groupwise_w4_15_m5_20260711.jsonl)
- [`bench/apple_production_close_groupwise_w8_15_m5_20260711.jsonl`](../../bench/apple_production_close_groupwise_w8_15_m5_20260711.jsonl)

### DPLR prefill tuning

The layer-major tiled DPLR path now supports `collect_all=True`, which lets one
target graph verify a block of draft tokens. For the 1.5B W4 M5 shape, chunk 12
with no intermediate layer materialization is the selected target-prefill
policy. For 0.4B, chunk 64 with four-layer materialization remains faster.
These are runtime policy values, not changes to RWKV-7 model math.

### Guarded compiled decode

The 0.4B W4 route uses compiled decode after a 32-token promotion gate. The
retained run has exact generated tokens, logits max-abs 0.125, state max-abs
0.0625, and a passing bounded validation. Against the retained 0.4B fp16 row,
W4 lowers peak memory from about 1.03 GB to 0.51 GB and raises conservative
decode from 59.20 to 192.83 tok/s.

### RWKV draft speculative decode

`rwkv7_hf.mlx_speculative.speculative_decode_greedy` uses a smaller RWKV model
as draft and the target's DPLR collect-all path as verifier. The implementation
includes:

- immutable shallow recurrent-state forks;
- target block verification and rejection replay;
- adaptive target-only fallback when observed acceptance falls below 25%;
- target-first streaming TTFT accounting;
- separate draft-prefill telemetry;
- exact target-greedy validation after measured performance rows.

In the retained 1.5B/0.1B W4 run, proposal width is 32, acceptance is 100%, two
target verification calls produce 64 tokens, fallback is not used, and the
post-run 64-token target-greedy oracle passes. Draft prefill is charged to decode
latency, because the first target token is already streamable after target
prefill; it is still exposed separately in every row.

Evidence:
[`bench/apple_production_close_speculative_w4_15_m5_20260711_final.jsonl`](../../bench/apple_production_close_speculative_w4_15_m5_20260711_final.jsonl).

## Correctness and serving coverage

This performance lane composes with the existing Apple tests rather than
replacing them:

- recurrent state select/reorder/drop/compact and session reuse;
- chunked prefill and bounded DPLR windows;
- interleaved dynamic session batching;
- compiled decode promotion/fallback;
- stateful CoreML prefill/decode export, state transfer, chunk split, and HF
  greedy checks on 0.1B/0.4B;
- native PyTorch MPS Trainer, PEFT LoRA, and TRL SFT/DPO/GRPO coverage.

See [APPLE_SILICON.md](APPLE_SILICON.md) for the full hardware and training
matrix.

## Reproduce the final gate

Run the Qwen and RWKV collectors with the same prompt seed, then compare the
combined JSONL:

```bash
PYTHONPATH=. python bench/compare_qwen35_apple_baseline.py \
  --results bench/apple_production_close_qwen35_gate_m5_20260711.jsonl \
  --pair '/path/qwen35-0.8b-mlx-4bit=rwkv7-g1d-0.4b-hf' \
  --pair '/path/qwen35-2b-mlx-4bit=rwkv7-g1g-1.5b-hf' \
  --min-decode-ratio 1.0 \
  --min-prefill-ratio 1.0 \
  --max-ttft-ratio 1.1 \
  --max-memory-ratio 1.0 \
  --require-prefill --require-ttft --require-memory \
  --diagnostics --fail-on-gate
```

The 1.5B RWKV collector uses:

```bash
RWKV7_MLX_WKV_SCAN_PREFILL=0 PYTHONPATH=. \
python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 512 --decode-lengths 64 \
  --repeat 3 --warmup-repeats 1 --qwen-models '' \
  --rwkv-mlx-models /path/rwkv7-g1g-1.5b-hf \
  --rwkv-draft-model /path/rwkv7-g1d-0.1b-hf \
  --rwkv-speculative-proposal-tokens 32 \
  --rwkv-dtype fp16 --rwkv-quantization mm4 \
  --rwkv-quant-min-params 1000000 \
  --rwkv-quant-backend groupwise --rwkv-wkv-backend metal \
  --rwkv-prefill-backend auto --rwkv-dplr-chunk-size 12 \
  --rwkv-dplr-summary-implementation tiled \
  --rwkv-dplr-layer-eval-interval 0 \
  --rwkv-dplr-layer-eval-min-tokens 1 \
  --rwkv-dplr-window-tokens 512
```

## Runtime maintainability

The production paths are now separated into model math, recurrent state, session/dynamic batching, and dependency-free policy modules while preserving the historical `rwkv7_hf.mlx_model` imports. See [MLX_RUNTIME_ARCHITECTURE.md](../reference/MLX_RUNTIME_ARCHITECTURE.md).

## Remaining portability work

The M5 gate is closed. Production portability still requires repeating the same
checked matrix on M1/M2/M3/M4 generations, additional memory tiers, longer
contexts, and sustained multi-session pressure. CoreML INT4 quality and proven
ANE occupancy also remain separate from the MLX GPU pass above.
