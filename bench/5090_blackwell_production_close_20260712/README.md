# RTX 5090 Blackwell HF acceptance close (2026-07-12)

This artifact records the Blackwell follow-up for the RWKV-7 Hugging Face
adapter. It closes the historical 7.2B pressure-row regression, validates the
new batched MM8/MM4 kernels, and removes the 48GB-host blocker for converting
and loading the official 13.3B checkpoint.

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, sm_120.
- Driver: 610.43.02; PyTorch: `2.6.0a0+ecf3bae40a.nv25.01`; CUDA: 12.8.
- Quant matrix runtime: Transformers 4.57.6 and Triton 3.1.0, FLA-free native model/JIT path.
- MATH500 wrapper runtime: Triton 3.3.1 and flash-linear-attention 0.5.1.
- Repository base: `dcd8a0f` plus this change.

## What changed

- MM8 and MM4 now have one-launch batched GEMV kernels for small decode batches.
- Blackwell fp16 decode batches of four or more rows use tensor-core `tl.dot`
  kernels; MM4 loads each packed byte once and computes both nibbles.
- Pre-Blackwell dispatch remains on its previously validated route, so this
  exact-card optimization does not silently replace V100/Ampere policy.
- Fresh-process paired baselines and median timing repeats remove cross-process
  clock-state noise from quant speed gates.
- The matrix summarizer now has a fail-closed acceptance mode.
- Conversion supports `--low-memory`: mmap source checkpoint, meta template,
  exact translated-key/shape/dtype validation, and bounded safetensors shards.
- The remote-code entrypoint directly declares its complete transitive kernel
  closure. Fresh Transformers caches no longer require manually pre-copying
  `ada_sparse_ffn.py`, `sm70_quant.py`, or related helpers. The fresh-cache
  import proof is in [`fresh_remote_code_cache.log`](fresh_remote_code_cache.log).

## 1.5B / 2.9B / 7.2B quant pressure matrix

Raw rows: [`quant_gap_close.jsonl`](quant_gap_close.jsonl). Full generated table:
[`quant_gap_close_summary.md`](quant_gap_close_summary.md).

Shape: model × `{fp16, MM8, MM4}` × prompt `{128, 2048}` × decode
`{128, 512}` × batch `8`, for 36 rows. Quant rows use an in-process paired fp16
baseline. The sole noisy 7.2B W4 row was repeated three times and the median run
is retained.

| Model | Quant | Rows | Min speed ratio | Min footprint ratio | Min prompt cosine | Min final cosine | Same next |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.5B | MM8 | 4 | 0.9841 | 0.9562 | 0.99996281 | 0.99995750 | 4/4 |
| 1.5B | MM4 | 4 | 0.9932 | 0.9342 | 0.99980468 | 0.99979019 | 4/4 |
| 2.9B | MM8 | 4 | 0.9925 | 0.9716 | 0.99995792 | 0.99995804 | 4/4 |
| 2.9B | MM4 | 4 | 0.9967 | 0.9573 | 0.99982035 | 0.99980414 | 4/4 |
| 7.2B | MM8 | 4 | 0.9913 | 0.9814 | 0.99996138 | 0.99996281 | 4/4 |
| 7.2B | MM4 | 4 | 0.9919 | 0.9720 | 0.99944615 | 0.99948436 | 4/4 |

Result:

- All 24 quant rows reduce model footprint and preserve the fp16 greedy next token.
- All 2.9B/7.2B rows pass the strict `>=0.99x` paired speed-equivalence gate.
- The combined 36-row matrix passes a conservative `>=0.98x` gate. One 1.5B
  MM8 row is `0.9841x`; it is the only row below `0.99x`, so universal strict
  no-slower-than-fp16 is not claimed for that shape.
- The previous 7.2B large-pressure regressions (`0.7619x` MM8 / `0.6695x` MM4
  in the old cross-process matrix) are closed: prompt2048/decode512/bsz8 is now
  `0.9913x` MM8 / `0.9919x` MM4 with paired measurement.

Gate command:

```bash
python bench/summarize_blackwell_quant_matrix.py \
  bench/5090_blackwell_production_close_20260712/quant_gap_close.jsonl \
  --gate --expected-rows 36 --min-speed-ratio 0.98
```

## 13.3B low-memory conversion and boundary

Official source:

- `rwkv7-g1g-13.3b-20260523-ctx8192.pth`
- size: `26,540,868,485` bytes
- SHA256: `0aa686d3ca4bb486e83e3071f4798a210f960e1fc1f5042e6cb418cc463814d6`

The previous 48GB/no-swap attempt could not hold both the checkpoint and a
second initialized fp16 template. The new low-memory path produced six
safetensors shards on the same host. The 0.1B normal-vs-low-memory validation
had identical 399-key sets, shapes and dtypes, with tensor `max_abs=0`; see
[`low_memory_parity.log`](low_memory_parity.log).

The generated 13.3B directory then passed load, forward, and four-token HF
`generate` on the 5090. See [`13p3_smoke.jsonl`](13p3_smoke.jsonl):

- model footprint: 25,309.1 MiB;
- peak VRAM: 25,536.6 MiB;
- generated tail decodes to `Hello! How can`;
- 61 layers, hidden 4096, head dimension 64.

Speed-policy boundary rows are in
[`quant_13p3_boundary.jsonl`](quant_13p3_boundary.jsonl):

| Quant | Decode ratio | Footprint MiB | Footprint ratio | Prompt cosine | Final cosine | Same next |
|---|---:|---:|---:|---:|---:|---:|
| MM8 | 0.9912 | 25,053.3 | 0.9899 | 0.99997813 | 0.99996901 | yes |
| MM4 | 0.9889 | 24,925.3 | 0.9848 | 0.99985284 | 0.99985945 | yes |

These are `speed`-policy rows (lm_head only), not full-memory quantization
claims. Full-memory W8/W4 still needs deeper fused projection kernels to beat
fp16 while delivering the much larger payload reduction.

## MATH500 / Albatross gate

The final runner selected bsz128 after a bsz64/96/128/192 sweep, then completed
the full `500 x 64 = 32,000` generation workload with deferred text decode and
four-worker answer verification:

| Metric | HF adapter | Albatross reference | Result |
|---|---:|---:|---:|
| pass@64 | 0.38 | 0.37 | PASS (`+0.01`) |
| rollout accuracy | 0.142469 | 0.145937 | delta `-0.003469` |
| generation token/s | 16,925.6 | 3,903.6 | PASS (`4.336x`) |
| steady decode token/s | 19,339.5 | 3,970.1 | PASS (`4.871x`) |

All four fail-closed gates pass: compatible shape, pass@64 `>=0.37`, summary
speed `>=2x`, and steady decode speed `>=2x`. The HF run produced 4,559
correct generations, ended 25,137 at EOD, and truncated 6,863 (`21.45%`).
The separate 500-text compression check scored 43,865 external tokens and
matched the HF reference exactly: `1.9241015 bits/token`, ratio `1.0`.

This is a live RTX 5090 HF run compared against the repository's committed
Albatross full-run reference. It is not a fresh same-card Albatross rerun, so
the result closes the committed acceptance gate but is not presented as a
same-session kernel microbenchmark. Generated manifest, summaries, comparison,
and compression-alignment evidence are retained in this directory:
[`math500_manifest_compact.json`](math500_manifest_compact.json),
[`math500_full_summary.json`](math500_full_summary.json),
[`math500_comparison.json`](math500_comparison.json), and
[`math500_compression_alignment.md`](math500_compression_alignment.md).

## Regression gates

- Full repository pytest on the 5090 host: `165 passed, 3 skipped`.
- MM8 CLI correctness: per-layer cosine `0.999023`, bsz2/8 fused max-abs
  `0.006592/0.005859`, end-to-end cosine `0.999991`, PASS.
- MM4 CLI correctness: per-layer cosine `0.983398`, bsz2/8 fused max-abs
  `0.003906/0.009766`, end-to-end cosine `0.998870`, PASS.
- Native Transformers contract, remote-code direct-import closure, acceptance
  scripts, batch-conversion manifest, and Python compilation: PASS.

Logs: [`native_quant_mm8_test.log`](native_quant_mm8_test.log),
[`native_quant_mm4_test.log`](native_quant_mm4_test.log), and
[`pytest.log`](pytest.log).

## Reproduction

```bash
python scripts/convert_rwkv7_to_hf.py \
  --input /data/models/official/rwkv7-g1g-13.3b-20260523-ctx8192.pth \
  --output /root/models/rwkv7-g1g-13.3b-hf \
  --vocab-file /root/rwkv_vocab_v20230424.txt \
  --precision fp16 --attn-mode fused_recurrent --no-fuse-norm \
  --max-shard-size 5GB --low-memory

python bench/run_blackwell_quant_matrix.py \
  --model 2.9b=/root/models/rwkv7-g1g-2.9b-hf \
  --model 7.2b=/root/models/rwkv7-g1g-7.2b-hf \
  --prompt-tokens 128 2048 --decode-tokens 128 512 \
  --batch-sizes 8 --quantizations none mm8 mm4 \
  --policy speed --warmup 16 --paired-baseline
```
