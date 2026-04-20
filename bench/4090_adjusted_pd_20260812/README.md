# RTX 4090 latest-checkpoint parameter-adjusted Prefill/Decode

**Status: PASS — 36/36 adjusted Prefill cells and 36/36 adjusted Decode
cells exceed `1.0x`.** This artifact compares the latest RWKV-7 g1d/g1i
0.4B/1.5B/2.9B checkpoints with Qwen3.5 0.8B/2B/4B on the same desktop RTX
4090 in dense FP16.

The matrix covers B1/B8, prompt 128/512/2048, decode 128/512, three warmups,
and seven measured runs per cell. Qwen fails closed unless FLA chunk Gated
DeltaNet, fused-recurrent decode, fused gated normalization, and the repository
Triton causal-convolution Prefill/update kernels are all live. All 36 reference
rows report `qwen_fast_path_verified=true`,
`qwen_full_fused_contract_pass=true`, and effective backend
`qwen_fla_gated_delta_rule_fla_triton_conv`.

The adjustment is:

```text
adjusted ratio = (RWKV tok/s / Qwen tok/s)
                 * (RWKV active parameters / Qwen active parameters)
```

| Pair | Batch | Adjusted Prefill min / median | Adjusted Decode min / median |
|---|---:|---:|---:|
| 0.4B / 0.8B | B1 | `4.315233x / 4.734654x` | `12.191128x / 12.289724x` |
| 0.4B / 0.8B | B8 | `1.273087x / 1.331289x` | `10.629066x / 10.692834x` |
| 1.5B / 2B | B1 | `1.196063x / 3.344352x` | `7.015246x / 7.045034x` |
| 1.5B / 2B | B8 | `1.108265x / 1.252091x` | `6.333830x / 6.363889x` |
| 2.9B / 4B | B1 | `2.129590x / 2.549867x` | `4.616954x / 4.644845x` |
| 2.9B / 4B | B8 | `1.205831x / 1.333566x` | `4.158943x / 4.175175x` |

## Closing the 1.5B/B1/P2048 red cell

The first complete matrix correctly failed closed at 1.5B/B1/P2048:
adjusted Prefill was `0.963992x` for D128 and `0.953292x` for D512. The recent
RTX 4080 self-chunk and stacked-R/K/V work supplied the candidate route, but
tiles were re-measured on this exact RTX 4090 instead of copied blindly.

Three independent loads used an interleaved route order. Each load measured
both D128 and D512; the table first reduces those two readings to a per-process
median, then reports the median across three processes.

| Route | Prefill tok/s, three processes | Median vs control |
|---|---:|---:|
| Control | `40,452 / 40,311 / 40,314` | `1.0000x` |
| self-chunk 16 | `50,033 / 49,940 / 49,929` | `1.2388x` |
| self-chunk 16 + stacked R/K/V | `50,518 / 50,549 / 50,562` | **`1.2539x`** |
| self-chunk 32 | `49,243 / 49,132 / 49,049` | `1.2187x` |
| self-chunk 32 + stacked R/K/V | `49,770 / 49,483 / 49,714` | `1.2332x` |

The selected tile-16 + stacked route then passed a same-process forward/reverse
A/B. It measured `1.248372x` and `1.251061x`; minimum Prompt/Decode cosine was
`0.99999475/0.99999404`, and both greedy comparisons matched. The default
policy rerun reached `50,631.193/50,346.248 tok/s` for D128/D512, moving the
two adjusted Prefill cells to `1.212204x/1.196063x`.

Promotion is exact-shape-only: desktop RTX 4090, hidden 2048, 24 layers, B1,
P2048, FP16. Adjacent Ada cards and every unmeasured shape retain the previous
fallback. This is a performance comparison; it is not a model-quality claim.

## Reproduction

Run the full matrix:

```bash
RWKV_PYTHON_BIN=/path/to/rwkv-python \
QWEN_PYTHON_BIN=/path/to/qwen-python \
RWKV_04_MODEL=/models/rwkv7-g1d-0.4b-hf \
RWKV_15_MODEL=/models/rwkv7-g1i-1.5b-hf \
RWKV_29_MODEL=/models/rwkv7-g1i-2.9b-hf \
QWEN_08_MODEL=/models/Qwen3.5-0.8B \
QWEN_2_MODEL=/models/Qwen3.5-2B \
QWEN_4_MODEL=/models/Qwen3.5-4B \
OUT_DIR=/tmp/4090-adjusted-pd \
  bash bench/run_4090_adjusted_pd.sh
```

Recheck the selected route against its local control:

```bash
python bench/bench_native_prefill_self_chunk_ab.py \
  --model /models/rwkv7-g1i-1.5b-hf \
  --batch-size 1 --prompt-tokens 2048 \
  --chunk-size 16 --stacked-rkv \
  --warmup 3 --steps 10 --min-cosine 0.9999 \
  --results /tmp/selfchunk-ab.jsonl
```

## Artifacts

- `candidate.jsonl`: 36 final RWKV rows; the two P2048 red cells are the
  default-policy corrected rerun.
- `qwen_reference.jsonl`: 36 verified FLA + Triton-conv Qwen rows.
- `summary.json`, `summary.md`: every joined cell, formula, medians, and gate.
- `selfchunk_sweep.jsonl`: 30 route-selection rows across three loads.
- `selfchunk_ab.jsonl`: two forward/reverse correctness and speed rows.
- `corrected_cells.jsonl`: the two final default-policy red-cell reruns.
- `environment.json`: exact hardware/runtime identity.
- `SHA256SUMS`: integrity hashes for every evidence file except itself.
