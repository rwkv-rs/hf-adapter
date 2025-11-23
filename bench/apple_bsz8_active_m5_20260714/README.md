# Apple M5 B8 W4 active-parameter acceptance

## Device and software

- MacBook Air, Apple M5 (10 CPU cores, 8 GPU cores), 16 GB unified memory
- macOS 26.5 (25F71), Metal 4
- Python 3.11.15; package versions are recorded in `environment.txt`
- RWKV candidates: `rwkv7-g1d-0.4b-hf` and `rwkv7-g1g-1.5b-hf`
- Qwen controls: MLX-LM W4 `qwen35-0.8b-mlx-4bit` and
  `qwen35-2b-mlx-4bit`

## Acceptance contract

Each runtime runs in an isolated child process. The measured workload is true
batch 8, a 512-character prompt, 64 generated tokens per sequence, one warmup,
and three measured repeats. RWKV uses group-128 affine W4; the compared
published MLX-LM Qwen checkpoints use group-64 affine W4. Throughput is
aggregate tokens per second. The requested active-parameter comparison is:

```text
normalized throughput = aggregate tok/s * active text parameter count
```

Higher is better. The speed gate requires both normalized prefill and decode
ratios to be at least 1.0. The memory gate compares raw child-process peak
memory, not memory normalized by parameter count. The compared tokenizers do
not emit identical prompt-token counts, so the JSONL records both counts and
uses each runtime's measured aggregate throughput.

Cold prefill and prefix-state-cache prefill are deliberately separate rows.
The 1.5B cache row has one unique prompt in an eight-request batch (one miss,
seven hits, 87.5% hit rate). It is a serving/cache acceptance scenario and is
not evidence that cold 1.5B prefill beats Qwen2.

## Results

| Scenario | RWKV prefill | Qwen prefill | Active-normalized prefill | RWKV decode | Qwen decode | Active-normalized decode | Raw peak RWKV / Qwen | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.4B vs 0.8B, cold | 7942.62 | 4276.23 | 1.1128x | 781.66 | 379.17 | 1.2351x | 1.253 / 1.644 GB | PASS |
| 1.5B vs 2B, 87.5%-hit prefix cache | 13677.81 | 2113.13 | 5.2537x | 686.01 | 174.40 | 3.1928x | 1.858 / 2.152 GB | PASS |
| 1.5B vs 2B, cold | 2478.03 | 2235.16 | 0.8999x | 695.14 | 184.59 | 3.0567x | 2.112 / 2.152 GB | FAIL (prefill) |

The historical cold 1.5B pair retained validation/candidate allocations and
reported 2.468 GB. The current route releases validation state and evicts the
W/A source matrices after building their mathematically equivalent packed
LoRA-down cache. The cold speed gap remains real: current normalized prefill
is `0.8999x`, so this repository does
not make a blanket cold 1.5B victory claim.

Both final rows were taken with the normal desktop applications left running;
all three samples for each engine are retained rather than reporting only a
best sample. Earlier cross-process experiments produced contradictory results
when unrelated desktop load changed between candidates. Same-process,
order-balanced A/B evidence resolves that ambiguity: the fused scan/post path
is `1.0938x` the split path, and the two-GEMM LoRA-down path is `1.0189x` the
direct path. See `ab_scan_post_same_process.json` and
`ab_lora_down_same_process.json`.

## Correctness and quantization

- 0.4B W4 versus fp16: exact B8 x 64 greedy token equality (100% match).
- Fused scan/post versus generic W4: exact greedy tokens; prefill logits/state
  max-abs 0.0625; final-state max-abs 0.046875.
- Mixed-prefix cache versus cold W4: exact greedy tokens with two genuinely
  different equal-length prefixes, six hits (75%), and exact reorder/compact.
  The 0.4B max-abs bounds are 0.0625 logits/state and 0.046875 final state.
  The 1.5B bounds are 0.09375 logits, 0.137917 prefill state, and 0.191262
  final state.
- 1.5B W4 versus fp16: exact B8 x 64 greedy equality (100% match). The
  prefill-logit max-abs is 6.125, so this is a token-level gate.
- Real 1.5B target / 0.1B draft rejection: acceptance rate `0.116369`, 56
  verifier/replay calls, and exact target-greedy output for all 512 B8 tokens.
- Compiled zero-state prefill versus eager: exact logits and state on both
  validated concrete shapes.
- Quantized embedding and linear payload ratios are both 0.265625 of their
  dense fp16 equivalents.

The 0.4B W4-vs-fp16 prefill-logit max-abs is 7.4375 even though all measured
greedy tokens match. This is token-level acceptance, not a claim of close
logit equivalence to fp16.

## Reproduce

Configure model paths if they are not under `../models`, then run:

```bash
PYTHON_BIN=/path/to/python \
MODEL_ROOT=/path/to/models \
COOLDOWN_SECONDS=30 INITIAL_COOLDOWN_SECONDS=60 \
scripts/run_apple_bsz8_active_acceptance.sh
```

The one-click script removes only its four named prior JSONL outputs before a
run, validates fidelity, and then executes the isolated comparisons. It
enables W/A LoRA-down double-GEMM fusion for both model sizes and evicts the
now-redundant original W/A down-projection matrices.

## Candidate and rejected A/B routes

The accepted candidate combines the thread-local Metal WKV scan, fused
scan+GroupNorm+bonus+gate post-processing, groupwise W4 Metal embedding lookup,
guarded compiled zero-state prefill, fast norms, prefix-state coalescing where
explicitly requested, and exact lockstep B8 speculative decode with a 0.1B
RWKV draft. Both profiles enable two-GEMM W/A LoRA-down fusion. The 1.5B route
releases 18,874,368 bytes of redundant source matrices after packing.

Rejected on this M5: a concatenated single-GEMM LoRA-down path, adding G/V to
the fused down path, fused LoRA-up, grouped RKV W4, blanket rank-3 flattening,
group sizes 32/64, SIMD/two-lane WKV state ownership, MXFP4/NVFP4, nested local
FFN compilation, fp16 recurrent state, and threadgroup-resident full state.
Selective flattening is enabled only for exact wide-to-narrow groupwise FFN
value projections; the end-to-end same-process B8 gain is `1.0066x` with zero
logit/state difference. The fused scan/post and two-GEMM LoRA-down candidates
also remain accepted.

The remaining 1.5B cold-prefill work is a native W4 FFN/projection or block
megakernel. Cache/wrapper tuning alone cannot close that acceptance row.
