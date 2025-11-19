# RTX 5070 Laptop Qwen3.5 FLA Matrix

Date: 2026-07-13

Hardware: one `NVIDIA GeForce RTX 5070 Laptop GPU`, `sm_120`, 8151 MiB,
driver `582.05`.

Software: Windows, PyTorch `2.11.0+cu128`, Triton Windows
`3.7.1.post27`, Transformers `5.12.1`, flash-linear-attention `0.5.1`,
bitsandbytes `0.49.2`. `causal-conv1d` is not installed on this Windows
environment.

## Scope

This artifact compares RWKV-7 1.5B against official Qwen3.5 2B using exact
matched tensor shapes: prompt 128/512/2048, decode 128/512, bsz 1/2/4/8, and
fp16/bnb8/bnb4. It contains 144 raw rows forming 72 comparison cells. Every
raw row passed, every cell joined, and every Qwen reference row verified the
FLA core operator contract.

The Qwen path binds all 18 linear-attention layers to FLA chunk prefill,
FLA fused-recurrent decode, and FLA fused gated normalization. Convolution
uses the Transformers Torch fallback because the Windows CUDA
`causal-conv1d` extension is absent. The effective backend is therefore
`qwen_fla_gated_delta_rule_torch_conv`, not a full-fusion claim.

## Correctness gate

The isolated Qwen fp16 prompt128/decode8/bsz1 FLA and forced-Torch probes use
the same input and eight greedy steps:

- greedy tokens: 8/8 match;
- prompt logits cosine: `0.99999076`;
- final logits cosine: `0.99999213`;
- prompt/final max absolute difference: `0.021484375` / `0.0234375`.

The same smoke measured FLA/Torch speedups of `3.105x` prefill and `1.148x`
decode. This is a single smoke shape, not the matrix-wide speed claim.

## Matrix result

The strict gate requires both prefill and decode to be at least `1.05x` Qwen
in every cell. Overall result: **FAIL**, with `35/72` strict cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 0.508x | 1.040x | 2.593x | 35/72 at >=1.05x; 41/72 at >=1.0x |
| Decode RWKV/Qwen | 1.019x | 1.334x | 3.635x | 71/72 at >=1.05x; 72/72 at >=1.0x |
| Model footprint RWKV/Qwen | 0.729x | 0.772x | 0.812x | 72/72 no larger |
| Peak VRAM RWKV/Qwen | 0.742x | 0.793x | 0.913x | 72/72 no larger |

Per quantization, prefill/decode `>=1.05x` counts are fp16 `11/24` and
`24/24`, bnb8 `14/24` and `24/24`, and bnb4 `10/24` and `23/24`.
Static footprint is 2913.3/3589.3 MiB for fp16, 1761.3/2280.2 MiB for bnb8,
and 1185.3/1625.6 MiB for bnb4 (RWKV/Qwen).

These rows use one warmup and one measured run in a fresh process per raw row.
That preserves process isolation and peak-memory attribution but leaves laptop
power-state and single-sample variance visible. In particular, prefill is
remeasured for both decode axes and should not be treated as a precision
microbenchmark. The strict per-cell report intentionally retains every red
row rather than replacing it with a median claim.

## Reproduction

Run from the repository root in Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File bench/run_5070_qwen35_fla_matrix.ps1
```

The Qwen weight is the official `Qwen/Qwen3.5-2B` safetensors file,
4548221488 bytes, SHA256
`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.

Canonical files:

- `results.jsonl`: all 144 raw rows;
- `summary.json` / `summary.md`: 72-cell strict comparison;
- `fla-smoke.jsonl` / `torch-smoke.jsonl`: isolated backend smokes;
- `fla-vs-torch-probe.json`: logits and greedy-token gate;
- `synthetic-fla-vs-torch.json`: direct kernel oracle check;
- `environment.json`, `model-verify.json`, and `exit-codes.json`: provenance.
