# V100 PR58 Native final regression

Date: 2026-07-19
GPU: Tesla V100-PCIE-32GB (sm70), GPU 0
Runtime: PyTorch 2.5.1+cu124, CUDA 12.4
Model: RWKV7-G1G 1.5B, fp16 activations, fp32 recurrent state
Candidate base: `8434510` plus the balanced-quantization and benchmark fixes in this artifact's commit.

The competing process on GPU 0 was stopped with `SIGSTOP` only for each timed
region and resumed by an EXIT/INT/TERM trap. GPU 1 and its serving workload were
not interrupted. Albatross and Native were measured fresh on the same V100.

## Result

The **primary V100 HF acceptance lane passes**:

- fresh dense Native decode is `0.9105x-1.0616x` fresh Albatross end to end for B1/B2/B4/B8;
- fresh dense Native prefill is `0.9205x-1.1110x` fresh Albatross for prompt 128/512 and B1/B2/B4/B8;
- prompt and continuation top-1 match, with minimum cosine at least `0.99999988` in the dense rows;
- dynamic batching B8 -> B2 has a `1.0` Native graph-cache hit rate;
- the W4 `balanced` deployment profile reduces model footprint and peak VRAM while retaining fp16-parity-or-better speed;
- W8 `speed` reduces model footprint and is at fp16 speed, with cosine above `0.99996`.

This does **not** convert the full-memory W4 route into a prefill-speed claim.
That optional route remains below fp16 prefill and is recorded as negative/open
evidence.

## Fresh Albatross comparison

### Cached decode

| Batch | Albatross tok/s | Native graph replay tok/s | ratio | Native end-to-end tok/s | ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 234.02 | 232.38 | 0.9930x | 230.39 | 0.9845x |
| 2 | 418.36 | 383.45 | 0.9166x | 380.92 | 0.9105x |
| 4 | 599.95 | 614.72 | 1.0246x | 610.50 | 1.0176x |
| 8 | 865.27 | 921.88 | 1.0654x | 918.59 | 1.0616x |

### Prefill

| Prompt | Batch | Albatross tok/s | Native tok/s | Native / Albatross |
|---:|---:|---:|---:|---:|
| 128 | 1 | 7,894.90 | 7,267.86 | 0.9206x |
| 128 | 2 | 11,080.82 | 12,311.03 | 1.1110x |
| 128 | 4 | 16,706.48 | 16,068.39 | 0.9618x |
| 128 | 8 | 19,202.70 | 18,842.33 | 0.9812x |
| 512 | 1 | 12,599.69 | 11,597.97 | 0.9205x |
| 512 | 2 | 16,357.38 | 16,843.81 | 1.0297x |
| 512 | 4 | 20,166.37 | 20,125.79 | 0.9980x |
| 512 | 8 | 22,006.19 | 20,670.87 | 0.9393x |

The remaining dense gap is now concentrated in B2 decode and B1 prefill,
not a broad Native/wrapper regression. B4/B8 decode and B2 prefill exceed the
fresh Albatross reference.

## Chunked prefill and state continuation

Prompt length is 512. Every row preserves sequence length and the next-token
continuation; maximum measured absolute logit difference is `0.15625`.

| Batch | chunk64 | chunk128 | chunk256 | chunk512 |
|---:|---:|---:|---:|---:|
| 1 | 0.4192x | 0.6279x | 0.8705x | 1.0017x |
| 8 | 0.7715x | 0.9095x | 0.9677x | 0.9937x |

Ratios are relative to a full prefill in the same process. Small B1 chunks are
valid bounded-memory/latency choices, not whole-prompt throughput wins.

## Dynamic batching and state cache

The repository Native model was warmed for B2-B8 and then decoded while rows
were reordered every four steps and dropped every eight steps:

- initial/final batch: `8 -> 2`;
- 64 decode steps, 16 reorders, 6 drops;
- throughput: `594.4 tok/s` total;
- graph-cache requests/hits: `64/64`, hit rate `1.0`;
- cache-copy and bind fast-skip rate: `0.8906`;
- effective backend: `native_graph`.

## Quantized production profiles

The new `balanced` native policy packs `lm_head` plus the first FFN key/value
pair by default. `RWKV7_NATIVE_MM_BALANCED_FFN_LAYERS` can change the FFN layer
count. This gives V100 a real speed-and-memory profile instead of calling the
slower full-memory route production-speed complete.

### W4 balanced (recommended V100 speed lane)

| Batch | prefill / fp16 | decode / fp16 | model footprint / fp16 | peak VRAM / fp16 | prompt/final cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0108x | 1.0345x | 0.9183x | 0.9579x | 0.99984682 / 0.99985588 |
| 8 | 0.9986x | 1.0037x | 0.9183x | 0.9514x | 0.99980372 / 0.99981284 |

B8 prefill is within 0.14% of fp16 and passes the repository's ±1% production
parity rule. Greedy equality and repeat determinism pass. CPU-first packing now
accepts `balanced` and uses `RWKV7_SM70_TARGET_PACK=1` automatically on V100;
the fresh packed B1/B8 rows use less peak VRAM and retain the exact sm70 layout.

### W8 speed

| Batch | prefill / fp16 | decode / fp16 | model footprint / fp16 | prompt/final cosine |
|---:|---:|---:|---:|---:|
| 1 | 1.0040x | 1.0004x | 0.9562x | 0.99999505 / 0.99999565 |
| 8 | 1.0005x | 1.0007x | 0.9562x | 0.99996191 / 0.99996340 |

The paired CUDA-graph peak is `1.0341x/1.0379x`; therefore this is a model
footprint win, not a claimed peak-VRAM win.

### Full-memory diagnostic routes

| Route | Batch | prefill / fp16 | decode / fp16 | footprint / fp16 | peak / fp16 |
|---|---:|---:|---:|---:|---:|
| W4 memory | 1 | 0.8992x | 1.1736x | 0.5395x | 0.6631x |
| W4 memory | 8 | 0.9533x | 1.0270x | 0.5395x | 0.6109x |
| W8 memory | 1 | 1.0061x | 1.0004x | 0.6932x | 1.2397x |
| W8 memory | 8 | 1.0010x | 1.0002x | 0.6932x | 1.2754x |

Full-memory W4 remains memory/decode specialized. Full-memory W8 closes the
speed ratio but retains large graph workspaces, so it is not a peak-memory
profile.

## Broader V100 regression retained

`python bench/check_v100_production_close.py` still passes the canonical
0.1B/0.4B/1.5B matrix: 12 dense decode, 12 dense prefill, 24 quant decode, 24
quant prefill, chunked serving, dynamic cache, 2-GPU device-map, Trainer/TRL,
and ZeRO-2/3 resume evidence. The current changes are inference policy and
benchmark changes; training math and distributed code were not changed, so the
validated dual-V100 training evidence was not rerun while GPU 1 was occupied.

The existing 1.5B/2.9B/7.2B 21-cell W4 cached-decode evidence remains under
`../v100_sm70_mm4_bntn_20260716/`. This artifact adds fresh 1.5B prefill,
Albatross, dynamic-cache, and balanced-profile closure rather than replacing
those model-size rows.

## Reproduction

All commands use `PYTHONPATH=.` and the repository code path. Representative
commands:

```bash
python bench/bench_native_model_decode.py \
  --hf-dir "$HF" --dtype fp16 --device cuda --prompt-tokens 128 \
  --decode-steps 64 --warmup 5 --repetitions 3 --batch-sizes 1 2 4 8 \
  --fast-token-api --timing-scope end_to_end --backends native_graph \
  --results native_decode_15b_e2e.jsonl

python bench/bench_native_default_v100_regression.py \
  --hf-dir "$HF" --model-label 1.5b --batch-sizes 1 2 4 8 \
  --prompt-tokens 512 --correctness-tokens 64 --warmup 3 --repeats 7 \
  --output native_prefill_15b_p512.json

RWKV7_NATIVE_MM_BALANCED_FFN_LAYERS=1 \
RWKV7_SM70_W4_FUSED_EPILOGUE=1 \
python bench/bench_native_quant_e2e_decode.py \
  --hf-dir "$HF" --code-source repo --model-size-label 1.5b \
  --single-quantization mm4 --mm4-group-size 128 \
  --mm4-group-policy lm_head --policy balanced --paired-baseline \
  --batch-size 8 --prompt-tokens 128 --decode-tokens 16 \
  --warmup 3 --timing-repeats 7 --results quant_mm4_balanced_b8.jsonl
```

Machine-readable aggregate: `summary.json`.
