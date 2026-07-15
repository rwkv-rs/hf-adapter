# V100 acceptance evidence index

Date: 2026-07-16

This directory is the canonical orientation point for V100 evidence already
committed to the repository. It does not replace or rewrite raw benchmark
rows. The fail-closed summarizer reads the promoted production-close bundle,
the two-cell full-FLA Qwen comparison, and the historical torch-fallback Qwen
matrix with their correct, separate contracts.

## Current verdict

Evidence validation is **PASS**. Overall V100 work remains **PARTIAL** because
the promoted lanes do not cover every model, shape, quantization path, quality
axis, or Albatross tier.

| Lane | Status | Exact boundary |
|---|---|---|
| Dense/serving/training production close | PASS | 0.1B/0.4B/1.5B measured lanes; selected-module W8/W4 only |
| Full-FLA Qwen3.5 | PASS 2/2 | RWKV-7 1.5B vs Qwen3.5-2B, P512/D64, B1/B8, dense fp16 |
| Historical Qwen3.5 matrix | PASS for its explicit torch lane | 216/216 joined cells, but every Qwen reference is Transformers torch fallback |
| Full-memory native MM8/MM4 | OPEN | Draft PR #21 is not promoted: MM4 has greedy mismatches and MM8 misses every speed cell |

The full-FLA rows have minimum raw prefill/decode ratios
`2.815921x/5.270432x` and minimum active-parameter work ratios
`2.285574x/4.277804x`. The B1 peak-VRAM ratio is `1.024885x`, so this is not a
universal memory win.

The historical append-only matrix has 435 raw rows and 216 joined cells. With
the backend pinned to `torch`, its current comparator gate passes with minimum
prefill/decode ratios `1.246447x/1.002879x`. Memory is not a gate: model
footprint is no larger in 213/216 final cells and peak VRAM is no larger in
189/216.

## Sources

- [`../v100_production_close_20260711/`](../v100_production_close_20260711/README.md)
- [`../v100_active_b1b8_20260715/`](../v100_active_b1b8_20260715/README.md)
- [`../v100_qwen35_full_matrix_20260713/`](../v100_qwen35_full_matrix_20260713/README.md)
- [`../../docs/validation/V100_HF_VALIDATION.md`](../../docs/validation/V100_HF_VALIDATION.md)

`summary.json` is the machine-readable verdict and `summary.md` is the compact
human report.

## Reproduce without a GPU

```bash
python bench/check_v100_production_close.py

python bench/compare_qwen35_speed_matrix.py \
  --results bench/v100_qwen35_full_matrix_20260713/results.jsonl \
  --expected-cells 216 \
  --min-prefill-speedup 1.05 --min-decode-speedup 1.05 \
  --min-quant-prefill-speedup 1.00 --min-quant-decode-speedup 1.00 \
  --required-reference-backend torch --fail-on-gate

python bench/summarize_v100_acceptance.py \
  --json-output bench/v100_acceptance_20260716/summary.json \
  --markdown-output bench/v100_acceptance_20260716/summary.md
```

## Work that still requires GPU access

- Expand the full-FLA Qwen matrix beyond one model pair, one prompt/decode
  shape and B1/B8.
- Develop and remeasure quality-safe full-memory MM4 plus a real Volta W8A16
  or deeper fused MM8 path.
- Extend larger/longer training, ZeRO resume and Albatross P2/P3 rows.

No new GPU result is synthesized by this index. Model-quality superiority over
Qwen3.5 is not claimed.
