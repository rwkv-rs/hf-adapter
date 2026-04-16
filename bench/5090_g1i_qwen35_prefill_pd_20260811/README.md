# RTX 5090 latest-checkpoint prefill PD close

This artifact closes the dense-FP16, B1/B8 prefill comparison for the latest
available RWKV-7 checkpoints against official Qwen3.5. Every measured cell
passes the parameter-adjusted throughput gate introduced by PR #107.

## Contract

- Device: one NVIDIA GeForce RTX 5090 (`sm_120`), driver `595.58.03`.
- Runtime: PyTorch `2.11.0+cu128`, CUDA runtime `12.8`, Triton `3.6.0`,
  Transformers `5.12.1`, FLA `0.5.1`, bitsandbytes `0.49.2`.
- Candidate checkpoints: RWKV-7 g1d 0.4B plus the 2026-08-05 g1i
  1.5B/2.9B/7.2B checkpoints, converted to HF FP16.
- References: official Qwen3.5 0.8B/2B/4B/9B.
- Matrix: B1/B8, prompt 128/512/2048, decode 128, prefill chunk 512,
  warmup 2, measured runs 5.
- Qwen gate: requested FLA, effective FLA Triton causal convolution, live
  fused-operator bindings, and full-fused contract in all 24 reference rows.
- Correctness gate: prefill logits cosine `>=0.9999`, exact greedy token, and
  exact greedy token after recurrent-cache handoff.

The primary metric is:

```text
prefill PD = (RWKV prefill tok/s / Qwen prefill tok/s)
             * (RWKV active parameters / Qwen active parameters)
```

Each cell must independently be at least `1.0`; medians do not hide a red
cell.

## Result

The machine-readable summary passes 24/24 joined cells with no red cells.
Raw RWKV/Qwen prefill is `1.320082x` minimum and `1.760289x` median.
Parameter-adjusted prefill PD is `1.028427x` minimum and `1.269319x` median.

| RWKV / Qwen pair | Active-param ratio | Minimum prefill PD | Passing cells |
|---|---:|---:|---:|
| 0.4B / 0.8B | `0.599112` | `1.028427x` | 6/6 |
| 1.5B / 2B | `0.811661` | `1.125352x` | 6/6 |
| 2.9B / 4B | `0.700882` | `1.100428x` | 6/6 |
| 7.2B / 9B | `0.804032` | `1.061388x` | 6/6 |

The 21 changed-route correctness rows all pass. Their minimum prefill or
post-handoff cosine is `0.99999994`; prefill and post-handoff greedy tokens
match in every row. The final 7.2B routes peak at `17.4-18.6 GiB`, below the
32 GiB device limit.

## Promoted routing

- Exact 0.4B/1.5B B1 and missing 0.4B B8 shapes use the already established
  native prefill CUDA graph to remove launch overhead.
- Exact 0.4B/1.5B B8 P512 chunks and 2.9B B8 P128/P512 chunks use scoped
  full-prefill FP16 GEMM accumulation.
- Exact 7.2B B1 uses CUDA graph plus scoped FP16 accumulation. Exact 7.2B B8
  keeps eager native prefill plus scoped FP16 accumulation because graph-only
  A/B was neutral or slower.
- Exact 7.2B B1/B8 P128/P512 chunks also enable the measured shift-mix,
  state-prep, output-prep, and stacked-RKV fusion combination.
- The FP16 accumulation switch is restored after each prefill call, is
  disabled for BF16/FP32, fails closed on multi-GPU unless explicitly allowed,
  and is empty on every non-5090 policy.

## Files

- [`results.jsonl`](results.jsonl): 24 candidate plus 24 full-FLA reference
  rows.
- [`correctness.jsonl`](correctness.jsonl): changed-route logits, greedy, and
  cache-handoff gates.
- [`summary.json`](summary.json): fail-closed machine-readable comparison.
- [`summary.md`](summary.md): rendered comparison table.

## Reproduce

Run each RWKV/Qwen pair with the same shape contract, changing the model paths
and pair labels as needed:

```bash
python bench/bench_cross_model_speed_resident.py \
  --model /path/to/rwkv7-hf \
  --model-kind rwkv --model-role candidate \
  --model-pair rwkv-1.5b__qwen3.5-2b --model-size-label 1.5b \
  --benchmark-matrix qwen35_5090_g1i_pd_final \
  --dtype fp16 --quantization none --device cuda \
  --batch-sizes 1 8 --prompt-tokens 128 512 2048 \
  --decode-tokens 128 --prefill-chunk-size 512 \
  --warmup 2 --runs 5 --rwkv-code-source repo \
  --results /tmp/5090-g1i-pd.jsonl

python bench/bench_cross_model_speed_resident.py \
  --model /path/to/Qwen3.5-2B \
  --model-kind qwen35 --model-role reference \
  --model-pair rwkv-1.5b__qwen3.5-2b --model-size-label 2b \
  --benchmark-matrix qwen35_5090_g1i_pd_final \
  --dtype fp16 --quantization none --device cuda \
  --batch-sizes 1 8 --prompt-tokens 128 512 2048 \
  --decode-tokens 128 --prefill-chunk-size 512 \
  --warmup 2 --runs 5 --qwen-backend fla \
  --qwen-conv-backend fla_triton --require-qwen-fast-path \
  --results /tmp/5090-g1i-pd.jsonl

python bench/compare_qwen35_speed_matrix.py \
  --results /tmp/5090-g1i-pd.jsonl --expected-cells 6 \
  --require-qwen-fast-path --require-qwen-full-fused \
  --required-reference-backend fla \
  --min-prefill-active-parameter-throughput-ratio 1.0 \
  --fail-on-gate
```

The benchmark was run from base commit
`82d31a86502d0df12b8a9fff29ae721a6e383cb4` plus the policy/runtime changes in
this artifact's pull request.
