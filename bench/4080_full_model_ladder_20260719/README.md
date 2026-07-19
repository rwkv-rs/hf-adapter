# RTX 4080 full HF model ladder

Date: 2026-07-19 to 2026-07-20

Hardware: one `NVIDIA GeForce RTX 4080` 16GB (`15923.2 MiB` reported),
`sm_89`, driver `595.71.05`.

Software: Linux, Python `3.11.15`, PyTorch `2.6.0+cu124`, CUDA runtime
`12.4`, Transformers `5.12.1`, Triton `3.2.0`, bitsandbytes `0.49.2`,
TorchAO `0.16.0`, flash-linear-attention `0.5.1`, fla-core `0.5.1`, and
causal-conv1d `1.5.0.post8`.

The implementation and runners start from upstream `main` commit
`8945395a165c497c2e3eb5f1b6e9284176b48872`. Private host paths were replaced
with `<PRIVATE_ROOT>` before committing the evidence.

## Accepted B1/B8 pairs

The six matrices compare official RWKV-7 checkpoints with official Qwen3.5
checkpoints on the same RTX 4080, dtype, batch size and token shape. Each pair
contains prompt lengths `128/512/2048` and decode lengths `128/512`. Qwen rows
fail closed unless FLA chunk prefill, fused-recurrent decode,
causal-conv1d prefill/update and fused gated normalization are all live.

| RWKV / Qwen pair | Batch | Dense prefill min-max | Dense decode min-max | Decode active-work min-max |
|---|---:|---:|---:|---:|
| 0.4B / 0.8B | B1 | `1.385151x-4.066360x` | `4.859517x-4.910462x` | `8.111217x-8.196243x` |
| 0.4B / 0.8B | B8 | `1.376179x-1.653856x` | `3.550845x-3.587359x` | `5.926843x-5.987795x` |
| 1.5B / 2B | B1 | `1.012285x-2.389928x` | `1.902894x-1.916997x` | `2.344436x-2.361824x` |
| 1.5B / 2B | B8 | `1.024180x-1.122658x` | `1.435296x-1.472913x` | `1.768344x-1.814687x` |
| 2.9B / 4B | B1 | `1.062111x-1.665402x` | `1.612078x-1.651326x` | `2.300076x-2.356070x` |
| 2.9B / 4B | B8 | `1.243909x-1.478998x` | `1.537228x-1.775644x` | `2.193280x-2.533441x` |

All six summaries report `status=pass`. This is an engine throughput and
active-parameter-work comparison. It is not a model-quality comparison.

## Quantized pair lanes

Every pair also contains 12 full-model BNB memory rows and 12 same-process
paired output-head speed rows. BNB8/BNB4 are accepted only as finite-logit,
lower-footprint routes. A8W8 and TorchAO-W4 require lower footprint, greedy
equality, minimum logits cosine `0.999`, cached decode no slower than fp16 and
complete-cell `prefill + decode` latency no slower than fp16. Phase prefill is
reported but not independently gated.

Across the six summaries:

- A8W8 complete-cell minima are `1.003101x-1.020060x`, with footprint ratios
  `0.9258x-0.9716x` and greedy equality in every cell.
- TorchAO-W4 complete-cell minima are `1.015996x-1.070146x`, with footprint
  ratios `0.8907x-0.9612x` and greedy equality in every cell.
- BNB8 footprint ratios are `0.573136x-0.665038x`; BNB4 ratios are
  `0.359704x-0.497558x`. No full-model speed claim is attached to these rows.

## Exact-card tuning

The 1.5B B1 P512/P2048 cells use an exact RTX 4080 self-chunk route with
chunk 32. P2048 additionally uses stacked R/K/V. The five-warmup,
eleven-sample P2048 feature factorial records `25224.732 tok/s` for the
baseline and `25669.724 tok/s` for stacked R/K/V. These defaults are guarded by
exact model, batch and prompt shapes and do not apply to another Ada card.

## Larger-model capacity

| Model/route | Shape | Result | Footprint / peak | Throughput and correctness |
|---|---|---|---:|---|
| RWKV-7 7.2B fp16 | B1, P128/D128 | fits | `13731.3 / 13919.4 MiB` | `360.7 / 44.5 tok/s` prefill/decode; deterministic greedy |
| RWKV-7 7.2B fp16 | B2/B4, P128/D1 | fits | peak `13942.6 / 14145.4 MiB` | finite logits and one generated token |
| RWKV-7 7.2B fp16 | B8, P128/D1 | expected capacity limit | peak `14550.8 MiB` | controlled CUDA OOM after successful load/forward |
| RWKV-7 7.2B MM8 | B1, P128/D128 | fits | `7340.5 / 7760.0 MiB` | `331.3 / 34.0 tok/s`; cosine `0.99998426`; greedy equal |
| RWKV-7 7.2B MM4 | B1, P128/D128 | fits | `4140.5 / 4816.0 MiB` | `256.0 / 27.1 tok/s`; cosine `0.99728274`; greedy equal |
| RWKV-7 13.3B fp16 | B1, P8/D1 | expected capacity limit | peak `14774.2 MiB` | controlled load-time CUDA OOM |
| RWKV-7 13.3B MM8 | B1, P128/D128 | fits | `13358.5 / 13894.9 MiB` | `175.2 / 18.4 tok/s`; deterministic greedy |
| RWKV-7 13.3B MM4 | B1, P128/D128 | fits | `7374.5 / 8166.9 MiB` | `149.8 / 15.4 tok/s`; deterministic greedy |
| Qwen3.5-9B fp16 | B1, P8/D1 | expected capacity limit | peak `14740.3 MiB` | controlled load-time CUDA OOM |

The 7.2B MM8/MM4 rows save memory and preserve the recorded fp16 greedy
sequence, but are slower than fp16. The 13.3B fp16 baseline cannot fit on this
card, so its quantized rows establish capacity and deterministic execution
only; they do not establish speed or logits parity against fp16.

## Repository tests

The final Linux run in an isolated worktree on the RTX 4080 host completes with
`561 passed`, `9 skipped`, and exit code 0. The full output is retained in
`full_tests_linux.log`.

## Reproduction

Install the CUDA dependencies and prepare local converted RWKV and official
Qwen directories. Run each pair twice, once with B1 and once with B8:

```bash
python -m pip install -e ".[train,quant]"

BATCH_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=python \
  bash bench/run_4080_qwen35_pair_acceptance.sh \
  rwkv-1.5b__qwen3.5-2b \
  /path/to/rwkv7-g1g-1.5b-hf \
  /path/to/Qwen3.5-2B \
  /tmp/rtx4080-rwkv1p5-qwen2-b1
```

Supported pair labels are `rwkv-0.4b__qwen3.5-0.8b`,
`rwkv-1.5b__qwen3.5-2b`, and `rwkv-2.9b__qwen3.5-4b`.

Observable success is exit code 0, `matrix_failures.txt=0`,
`pipeline_exit_code.txt=0`, `summary.json.status=pass`, exact coverage
`6/6/12/12`, and an empty `summary.json.errors` list. To recover a failed run,
retain its JSONL and logs, diagnose the exact cell, then rerun into a fresh
output directory. For Qwen failures, check the FLA and causal-conv1d bindings
first.

For AI-assisted setup, use the repository's single entry point:
[`docs/AI_ASSISTED_SETUP.md`](../../docs/AI_ASSISTED_SETUP.md).

## Files

- `pairs/`: accepted dense, memory and paired-quant rows plus fail-closed
  summaries for all six pair/batch combinations.
- `large_boundaries/`: 7.2B and 13.3B fit/quant rows plus Qwen3.5-9B capacity.
- `tuning/`: the accepted 1.5B B1 self-chunk and stacked-R/K/V A/B probes.
- `summary.json`: machine-readable aggregate of the promoted results.
- `full_tests_linux.log` / `full_tests_linux_exit_code.txt`: final Linux suite.
- `SHA256SUMS`: integrity manifest for this artifact.
