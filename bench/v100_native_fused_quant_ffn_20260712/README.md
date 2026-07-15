# V100 native fused quant FFN evidence

Date: 2026-07-12

Branch: `wangyue/native-fused-mm8-mm4-ffn`

Validated source commit: `e2c9ee5`

Hardware: one `Tesla V100-PCIE-32GB` (`sm_70`) on GPU0. GPU1 ran an
independent Qwen3.5 matrix and is not part of these rows.

## Scope

This artifact validates the opt-in
`RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN=1` route. It compares the existing packed
MM8/MM4 FFN key projection plus a separate ReLU-square against the fused
projection epilogue. The flag remains disabled by default.

The raw rows are:

- [`micro_results.jsonl`](micro_results.jsonl): MM8/MM4, fp16,
  `2048 -> 8192`, bsz 1/2/4/8, 10 warmups and 50 measured runs.
- [`e2e_results.jsonl`](e2e_results.jsonl): 1.5B, fp16, memory policy,
  prompt 128, decode 128, bsz1, three timing repeats, and a fresh paired fp16
  baseline for each quantized row.
- [`environment.txt`](environment.txt): exact runtime and GPU identity.

Pre-fix rows that reused stale MM8 dense packs or crashed MM4 prefill were
retained only on the benchmark host for diagnosis. They are invalid and are
not included here.

## Isolated FFN epilogue

All eight rows pass the `0.999` cosine gate.

| Quant | bsz | separate ms | fused ms | speedup | cosine |
|---|---:|---:|---:|---:|---:|
| MM8 | 1 | 0.218195 | 0.198555 | `1.0989x` | 0.99999994 |
| MM8 | 2 | 0.323456 | 0.304448 | `1.0624x` | 0.99999988 |
| MM8 | 4 | 0.495586 | 0.479423 | `1.0337x` | 0.99999988 |
| MM8 | 8 | 0.822629 | 0.816481 | `1.0075x` | 0.99999988 |
| MM4 | 1 | 0.104098 | 0.060374 | `1.7242x` | 1.00000012 |
| MM4 | 2 | 0.113901 | 0.075960 | `1.4995x` | 1.00000000 |
| MM4 | 4 | 0.113902 | 0.083238 | `1.3684x` | 0.99999988 |
| MM4 | 8 | 0.113619 | 0.098451 | `1.1541x` | 0.99999988 |

## End-to-end decode

| Quant | Fused FFN | decode tok/s | ratio vs fp16 | footprint MiB | footprint ratio | peak MiB | final cosine | same next |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MM8 | off | 94.4 | `0.4110x` | 2019.4 | 0.6932 | 3086.4 | 0.99998724 | yes |
| MM8 | on | 95.2 | `0.4145x` | 2019.4 | 0.6932 | 3086.4 | 0.99998689 | yes |
| MM4 | off | 263.4 | `1.1462x` | 1569.9 | 0.5389 | 2636.9 | 0.99837250 | yes |
| MM4 | on | 272.7 | `1.1867x` | 1569.9 | 0.5389 | 2636.9 | 0.99837714 | yes |

The fused epilogue improves the quantized path by about `1.0085x` for MM8 and
`1.0353x` for MM4. MM4 closes the fp16 speed and footprint gate for this exact
1.5B/bsz1/decode shape. MM8 remains a memory-saving path at only `0.4145x`
fp16 speed. This mixed result is not sufficient to enable the flag by default;
larger batch sizes, prompts, checkpoints, and other GPU families still need
end-to-end rows.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python bench/bench_native_quant_fused_ffn.py \
  --device cuda --dtype fp16 --input-size 2048 --output-size 8192 \
  --batch-sizes 1 2 4 8 --quantizations mm8 mm4 \
  --warmup 10 --runs 50 --results micro_results.jsonl
```

For each `quant` in `mm8 mm4`, run the end-to-end command once without and
once with `--fused-quant-ffn`:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python bench/bench_native_quant_e2e_decode.py \
  --hf-dir /path/to/rwkv7-g1g-1.5b-hf --code-source repo \
  --model-size-label 1.5b --dtype fp16 --device cuda \
  --attn-mode fused_recurrent --fast-cache true \
  --fast-token-backend native_graph --single-quantization mm4 \
  --policy memory --batch-size 1 --prompt-tokens 128 --decode-tokens 128 \
  --warmup 2 --timing-repeats 3 --paired-baseline \
  --fused-quant-ffn --results e2e_results.jsonl
```
