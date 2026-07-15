# V100 RWKV-7 vs full-FLA Qwen3.5 active-parameter acceptance

Date: 2026-07-15

This artifact is the exact-card, target-only HF inference comparison requested
for batch sizes 1 and 8. It compares RWKV-7 1.5B with Qwen3.5-2B on one
`Tesla V100-PCIE-32GB` (`sm_70`). It measures inference implementation speed,
active-parameter work rate and memory; it is not a model-quality comparison.

## Contract

- Models: RWKV-7 g1g 1.5B HF and the official text-only Qwen3.5-2B HF
  checkpoint.
- Shape: prompt 512, decode 64, batch 1 and 8, dense fp16.
- Timing: two warmups and three measured fresh-process runs; reported timing is
  the median and throughput is aggregate tokens/s.
- RWKV route: repository `native_prefill_graph` plus `native_graph` cached
  decode.
- Qwen route: FLA chunk Gated DeltaNet prefill, FLA fused-recurrent decode,
  fused gated RMS norm and the repository Triton causal-convolution
  prefill/update kernels. Both rows report
  `qwen_full_fused_contract_pass=true` and the effective backend
  `qwen_fla_gated_delta_rule_fla_triton_conv`.
- No draft model, speculative acceptance, prefix-state reuse or hidden cache
  warm-start is used.

The active-parameter work metric is:

```text
work_rate = aggregate_tokens_per_second * active_text_parameters
```

The measured active counts are 1,527,404,544 for RWKV and 1,881,825,088 for
Qwen, a ratio of `0.811661x`. Therefore RWKV must reach at least
`1 / 0.811661 = 1.232041x` Qwen's raw throughput merely to tie the normalized
work-rate gate. The fail-closed comparator requires both raw throughput and
active-parameter work rate to be `>=1.0x` in prefill and decode.

## Result

Overall: **PASS — 2/2 joined cells, 2/2 full-FLA Qwen references, no missing
or red cells.**

| Bsz | Phase | RWKV tok/s | Qwen tok/s | Raw ratio | RWKV active TOPS | Qwen active TOPS | Active-work ratio |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Prefill | 10,425.596 | 3,702.375 | `2.815921x` | 15.924103 | 6.967222 | `2.285574x` |
| 1 | Decode | 151.357 | 25.596 | `5.913307x` | 0.231183 | 0.048168 | `4.799514x` |
| 8 | Prefill | 20,729.017 | 3,833.197 | `5.407762x` | 31.661594 | 7.213407 | `4.389270x` |
| 8 | Decode | 816.606 | 154.941 | `5.270432x` | 1.247288 | 0.291572 | `4.277804x` |

Memory is reported independently and was not used to hide or override a speed
failure:

| Bsz | RWKV/Qwen model footprint | RWKV/Qwen peak VRAM | Conclusion |
|---:|---:|---:|---|
| 1 | 2,913.3 / 3,589.3 MiB (`0.811662x`) | 3,776.7 / 3,685.0 MiB (`1.024885x`) | RWKV footprint is lower; peak is 2.49% higher |
| 8 | 2,913.3 / 3,589.3 MiB (`0.811662x`) | 3,592.8 / 4,291.2 MiB (`0.837248x`) | RWKV footprint and peak are lower |

This dense comparison does not make a new quantized-versus-Qwen claim. The
separate canonical V100 artifact retains the card-local native W8/W4 versus
fp16 B1/B2/B4/B8 speed, payload and token-parity gates.

## Correctness probes

- Qwen full-FLA plus Triton-conv versus the same FLA core with the Transformers
  convolution oracle: 32/32 greedy tokens match; prompt/final logit cosine is
  `0.9999890/0.9999919` (gate `>=0.999`).
- RWKV native graph versus the FLA-backed HF wrapper: 32/32 greedy tokens
  match; prompt/final logit cosine is `0.9999952/0.9999954` (gate
  `>=0.9999`).
- Every formal speed row reports finite logits.

## Exact environment

- GPU: Tesla V100-PCIE-32GB, `sm_70`, driver `580.159.03`.
- PyTorch `2.5.1+cu124`, CUDA runtime `12.4`, Transformers `5.12.1`, Triton
  `3.4.0`, bitsandbytes `0.49.2`.
- FLA is imported from the source checkout recorded in the reproduction
  command. Checkpoint-config and benchmark-source hashes are retained in
  `environment.txt`.

## Reproduce

```bash
cd /path/to/rwkv7-hf-adapter
export CUDA_VISIBLE_DEVICES=1
export FLA_FLASH_QLA=0
export PYTHONPATH=/path/to/flash-linear-attention:$PWD

python bench/run_qwen35_speed_matrix.py \
  --pair 'rwkv-1.5b__qwen3.5-2b=/models/rwkv7-g1g-1.5b-hf::/models/Qwen3.5-2B' \
  --prompt-tokens 512 --decode-tokens 64 --batch-sizes 1 8 \
  --quantizations none --benchmark-matrix v100_active_b1b8_20260715 \
  --dtype fp16 --qwen-backend fla --qwen-conv-backend fla_triton \
  --require-qwen-fast-path --rwkv-fast-token-backend native_graph \
  --warmup 2 --runs 3 --results /tmp/v100-active/results_dense.jsonl \
  --fail-fast

python bench/compare_qwen35_speed_matrix.py \
  --results /tmp/v100-active/results_dense.jsonl --expected-cells 2 \
  --min-prefill-speedup 1.0 --min-decode-speedup 1.0 \
  --required-reference-backend fla --require-qwen-fast-path \
  --require-qwen-full-fused \
  --min-prefill-active-parameter-throughput-ratio 1.0 \
  --min-decode-active-parameter-throughput-ratio 1.0 \
  --fail-on-gate
```

## Artifacts

- `results_dense.jsonl`: four raw candidate/reference speed rows.
- `summary_dense.{json,md}`: fail-closed joined comparison and verdict.
- `qwen-full-fla-smoke.jsonl`, `qwen-conv-oracle-smoke.jsonl` and
  `qwen-full-fla-vs-oracle.json`: Qwen route and correctness proof.
- `rwkv-fla-reference-smoke.jsonl`, `rwkv-native-graph-smoke.jsonl` and
  `rwkv-native-graph-vs-fla.json`: RWKV route and correctness proof.
- `environment.txt`, `matrix_dense.log` and `SHA256SUMS`: stack, raw runner log
  and retained artifact hashes.
