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
and three measured repeats. Both model families use group-128 W4. Throughput is
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
| 0.4B vs 0.8B, cold | 9636.69 | 5162.62 | 1.1183x | 940.86 | 456.21 | 1.2356x | 1.261 / 1.644 GB | PASS |
| 1.5B vs 2B, 87.5%-hit prefix cache | 17113.84 | 2841.44 | 4.8886x | 872.01 | 232.10 | 3.0494x | 1.838 / 2.152 GB | PASS |
| 1.5B vs 2B, cold reference | 3202.66 | 2943.15 | 0.8832x | 886.73 | 235.82 | 3.0520x | see note | FAIL (prefill) |

The historical cold 1.5B pair retained validation/candidate allocations and
reported 2.468 GB. After releasing validation state and disabling the
model-size-regressing LoRA-down packed cache, an isolated RWKV row is about
2.09 GB versus the recorded Qwen 2.15 GB. The cold speed gap remains real:
current normalized prefill is approximately 0.88-0.90x, so this repository does
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
- Prefix cache versus cold W4: exact greedy tokens. The 0.4B max-abs bounds are
  0.0625 logits, 0.0625 prefill state, and 0.125 final state. The 1.5B bounds
  are 0.078125 logits and 0.133675 state.
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
run, validates fidelity, and then executes the isolated comparisons. It enables
LoRA-down double-GEMM fusion for 0.4B and disables it for 1.5B.

## Candidate and rejected A/B routes

The accepted candidate combines the thread-local Metal WKV scan, fused
scan+GroupNorm+bonus+gate post-processing, groupwise W4 Metal embedding lookup,
guarded compiled zero-state prefill, fast norms, prefix-state coalescing where
explicitly requested, and exact lockstep B8 speculative decode with a 0.1B
RWKV draft. The 0.4B profile enables two-GEMM W/A LoRA-down fusion; the 1.5B
profile disables it because the duplicate packed cache does not pay back at
that model size.

Rejected on this M5: a concatenated single-GEMM LoRA-down path, adding G/V to
the fused down path, fused LoRA-up, grouped RKV W4, flattened rank-3 quantized
matmul, group sizes 32/64, two-lane WKV state ownership, fp16 recurrent state,
and threadgroup-resident full state. These remain disabled by default. The
fused scan/post and two-GEMM LoRA-down candidates are not in this rejected
list: order-balanced same-process measurements confirmed both are positive.

The remaining 1.5B cold-prefill work is a native W4 FFN/projection or block
megakernel. Cache/wrapper tuning alone cannot close that acceptance row.
