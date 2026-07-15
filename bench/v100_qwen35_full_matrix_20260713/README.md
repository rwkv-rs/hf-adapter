# V100 RWKV-7 vs Qwen3.5 full HF speed matrix — 2026-07-13

This artifact closes the V100 Qwen3.5 comparison lane across three model pairs,
three prompt lengths, two decode lengths, four batch sizes, and dense/W8/W4
loads. All rows were collected on the same two-V100 host; each individual row
uses one V100. Candidate reruns are append-only and the comparator selects the
latest row for a cell.

## Result

`summary.md` reports **PASS** with `216/216` cells:

- overall prefill RWKV/Qwen: minimum `1.246x`, median `1.936x`;
- overall decode RWKV/Qwen: minimum `1.003x`, median `1.324x`;
- missing candidate/reference rows: `0/0`;
- red cells: `0`.

Pair coverage and latest-row ratios:

| Pair | Cells | Prefill min / median | Decode min / median |
|---|---:|---:|---:|
| RWKV-7 1.5B / Qwen3.5 2B | 72 | `1.412x / 2.023x` | `1.054x / 1.312x` |
| RWKV-7 2.9B / Qwen3.5 4B | 72 | `1.751x / 2.174x` | `1.016x / 1.347x` |
| RWKV-7 7.2B / Qwen3.5 9B | 72 | `1.246x / 1.533x` | `1.003x / 1.312x` |

Dense rows use a strict `>=1.05x` prefill and decode gate. W8/W4 rows use a
non-regression `>=1.00x` gate, matching the project requirement that quantized
inference not be slower than fp16-class comparison throughput while reducing
the quantized model payload.

## Scope and comparison lane

- Matrix: `3 pairs × 3 prompts × 2 decode lengths × 4 batch sizes × 3 modes`.
- Prompts: `128`, `512`, `2048` tokens.
- Decode lengths: `128`, `512` tokens.
- Batch sizes: `1`, `2`, `4`, `8`.
- Modes: dense fp16, bitsandbytes W8, bitsandbytes W4.
- Device: Tesla V100-PCIE-32GB (`sm_70`).
- Qwen backend: the explicitly recorded Transformers torch-fallback lane.

The Qwen rows are **not** presented as an FLA/causal-conv optimized Qwen
production maximum. The backend is pinned and recorded as
`transformers_torch_fallback`/`qwen_backend_requested=torch`, so this artifact
is reproducible and does not silently mix Qwen implementations.

## Final W4 close

Three original W4 decode cells were below the non-regression gate. The
`prefill_hot` storage policy retains the latency-sensitive projections in dense
form while quantizing the remaining payload. The append-only reruns close all
three cells without changing their reference rows:

- RWKV-7 2.9B, prompt 2048, decode 128, batch 1;
- RWKV-7 2.9B, prompt 512, decode 128, batch 2;
- RWKV-7 7.2B, prompt 2048, decode 128, batch 1.

## Reproduce the gate

```bash
python bench/compare_qwen35_speed_matrix.py \
  --results bench/v100_qwen35_full_matrix_20260713/results.jsonl \
  --expected-cells 216 \
  --min-prefill-speedup 1.05 \
  --min-decode-speedup 1.05 \
  --min-quant-prefill-speedup 1.00 \
  --min-quant-decode-speedup 1.00 \
  --required-reference-backend torch \
  --json-output /tmp/v100-qwen35-summary.json \
  --markdown-output /tmp/v100-qwen35-summary.md \
  --fail-on-gate
```

Files:

- `results_original.jsonl`: immutable 432-row full candidate/reference run;
- `prefill_hot_reruns.jsonl`: three append-only candidate reruns;
- `results.jsonl`: gate input combining the two files above;
- `summary.json` / `summary.md`: machine-readable and human-readable verdict;
- `environment.txt`: software and GPU inventory captured on the benchmark host.
