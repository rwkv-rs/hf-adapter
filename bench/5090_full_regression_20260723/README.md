# RTX 5090 full HF/native regression — 2026-07-23

Status: **pass for the tested 0.1B functional matrix and official g1h 1.5B
BF16/W4 B1/B8 lanes**.  The run started from `main` at
`3788176cb77710ba723ffe5940cc4a692cc1b28f`.

## Environment

- GPU: NVIDIA GeForce RTX 5090, SM120, 32 GiB
- PyTorch: `2.10.0+cu128`
- CUDA runtime/toolkit: `12.8`
- Triton: `3.6.0`
- Transformers: `5.14.1`
- TorchAO: `0.17.0`
- bitsandbytes: `0.49.2`

See [`environment.log`](environment.log).  The TorchAO warning about its
optional C++ extension requiring PyTorch 2.11 does not affect the vendored
Marlin BF16/W4 CUDA backend exercised here.

## Regressions found and closed

1. Three unit tests inherited physical RTX 5090 policy defaults instead of
   exercising the generic fallback.  They now pin the generic policy; exact
   card defaults remain covered by policy tests.
2. The 5090 low-memory sparse-FFN packer treated BnB/Marlin/TorchAO packed
   modules as dense FP16 `nn.Linear` weights.  Relayout is now limited to its
   exact dense CUDA-FP16 contract.
3. BnB W8/W4 prefill can return an FP16 recurrent cache while its deliberately
   eager decode fallback has an FP32 recurrence contract.  The fallback now
   promotes the cache once before the recurrence.
4. The four-way deterministic sparse FFN route was selected for g1d 0.1B
   (`H=768`, `I=3072`) even though its CUDA ABI requires `H % 512 == 0` and
   `I % 2048 == 0`.  Runtime and prewarm now share the exact shape predicate.
5. After newer dense prefill fusions, official g1h 1.5B W4 B8/P128 was only
   `0.9287x` dense in eager mode despite both W4 FFN GEMMs winning in isolation.
   The loss was Python/custom-op launch overhead.  Exact-card CUDA graph replay
   raises W4 prefill to `1.0634x` dense and is now the default only for
   `(H=2048, L=24, B=8, T=128)` on an exact RTX 5090.

## Test results

### Repository suite

`616 passed, 12 skipped, 37 warnings in 60.29s` on the physical RTX 5090.
See [`full_pytest.log`](full_pytest.log).

### 0.1B HF matrix

The one-command 5090 validation completed successfully and covered:

- remote-code load/generate and HF API/beam contract;
- native Trainer + PEFT, six optimizer steps, `72/72` trainable parameters
  changed;
- BnB 8-bit and 4-bit load/forward/generate;
- native-graph decode at B1/B2/B4/B8;
- chunked prefill at 64/128/256 tokens with greedy/decode match;
- fused-output and fused-recurrent-output A/B with `32/32` greedy match;
- dynamic B8 generation with 512 decoded token events.

Native-graph decode throughput was 1545.8, 1868.8, 3810.3 and 7556.0 tok/s
at B1/B2/B4/B8 respectively.  The complete rows are in
[`hf_matrix_0p1b.jsonl`](hf_matrix_0p1b.jsonl); individual functional logs are
stored beside it.

The g1d shape also verifies the deterministic sparse reducer fallback rather
than entering an unsupported four-way CUDA layout.

### Live BnB regression

Both BnB modes pass `forward + cached decode + generate` after the FP16-cache
fix.  W8 used 283.4 MiB model storage and W4 used 242.9 MiB; both generated the
same two-token tail `[4171, 1184]`.  Eager decode is expected for these external
quant modules.  See [`bnb_w8_w4_live.log`](bnb_w8_w4_live.log).

### Official g1h 1.5B BF16/W4

Prompt 128, decode 128, symmetric group-128 Marlin W4, 46 FFN modules packed,
one final layer and the head dense:

| Batch | Footprint / BF16 | Prefill / BF16 | Decode / BF16 | Prompt cosine | Final cosine | Greedy stream |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `0.6250x` | `1.2860x` | `1.2692x` | `0.99981397` | `0.99984372` | identical |
| 8 | `0.6250x` | `1.0634x` | `1.2572x` | `0.99972332` | `0.99961078` | identical, 7/7 deterministic |

The B8 policy-selected backend is `native_prefill_graph`; no environment
override was set.  Raw rows: [`g1h_1p5b_w4_b1.jsonl`](g1h_1p5b_w4_b1.jsonl)
and [`g1h_1p5b_w4_b8.jsonl`](g1h_1p5b_w4_b8.jsonl).

The per-layer B8 microbenchmark measured 0.3829 ms for the fused dense FFN and
0.3250 ms for W4 (`1.178x`), confirming that the previous end-to-end loss was
dispatch overhead rather than the BN/TN kernel.  The automatic Marlin schedule
also remained the best supported candidate in the exact 2048/8192 shape sweep.

## Reproduce

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export PYTHONPATH=$PWD

pytest -q

bash bench/run_5090_hf_validation.sh \
  HF_DIR=/path/to/rwkv7-g1d-0.1b-hf \
  OUT_DIR=/tmp/rwkv7-5090-full \
  DTYPE=fp16 BATCH_SIZES='1 2 4 8'

python bench/bench_native_quant_e2e_decode.py \
  --hf-dir /path/to/rwkv7-g1h-1.5b-hf \
  --code-source repo --model-size-label 1.5b \
  --dtype bf16 --device cuda --attn-mode fused_recurrent \
  --fast-cache true --fast-token-backend native_graph \
  --single-quantization torchao_w4 --min-params 1 \
  --group-size 128 --policy speed \
  --batch-size 8 --prompt-tokens 128 --decode-tokens 128 \
  --warmup 3 --timing-repeats 7 --paired-baseline \
  --results /tmp/rwkv7-5090-w4-b8.jsonl
```

The B8 command must run on an otherwise idle GPU.  Concurrent CUDA work was
discarded during this audit because it changed the paired ratio materially.
