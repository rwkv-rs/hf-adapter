# RTX 4080 B8 grouped projection BMM (2026-08-09)

This artifact records the exact-card RTX 4080 batch-8 decode optimization for
RWKV-7's low-rank W/A/G/V projections. It advances the previously promoted
B8 grouped graph route without changing checkpoints, public APIs, or fallback
behavior on other batch sizes and devices.

## Implementation

The candidate route is deliberately narrow: CUDA sm89, FP16 inference, exactly
B8, hidden size at least 1024, and low-rank width at most 512. It:

- pads and caches the nearby W/A/V ranks once per layer;
- runs W/A/V rank-in and rank-out as two tensor-core batched matrix multiplies;
- leaves the larger G projection on its existing GEMMs to avoid padding every
  group to G's rank;
- emits W/A/V from fused norm/mix in one backing allocation and reuses it as a
  zero-copy BMM view;
- applies private W/A/G/V activations in place;
- uses a smaller W/A-only pack for layer zero, where no V gate is required;
- invalidates a cached pack after a parameter version, shape, stride, dtype, or
  device change; the cache is not serialized in the state dict.

Unsupported shapes, dtypes, training/autograd execution, non-sm89 devices, and
environment-disabled runs retain the existing grouped/dense implementation.
The default is enabled only for an exact desktop RTX 4080 policy match.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4080, 16,376 MiB, sm89 |
| Driver | 595.71.05 |
| PyTorch / CUDA | 2.6.0+cu124 / 12.4 |
| Triton | 3.2.0 |
| Dtype | FP16 |
| Backend | repository `native_graph` implementation |

Models:

- `rwkv7-g1d-0.4b-hf`
- `rwkv7-g1g-1.5b-hf`
- `rwkv7-g1g-2.9b-hf-converted`

## Paired A/B result

Each checkpoint was loaded independently three times. Every run used B8,
prompt 64, 16 warmup steps, 512 fixed-token timing steps, and 64 untimed greedy
correctness steps. The baseline keeps the existing grouped W/A/G/V route on and
changes only `RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=0`; the candidate changes it to
`1`. All other native-graph fusions are held constant.

| Model | Baseline median | Candidate median | Candidate throughput | Median speedup | Speedup range | Peak-memory delta |
|---|---:|---:|---:|---:|---:|---:|
| 0.4B | 2.7904 ms/step | 2.4798 ms/step | 3,226.1 tok/s | **1.1267x** | 1.1240x-1.1288x | +27.0 MiB (+2.39%) |
| 1.5B | 6.4396 ms/step | 5.8853 ms/step | 1,359.3 tok/s | **1.0942x** | 1.0941x-1.0944x | +65.0 MiB (+1.90%) |
| 2.9B | 11.8160 ms/step | 10.9309 ms/step | 731.9 tok/s | **1.0809x** | 1.0805x-1.0819x | +100.2 MiB (+1.55%) |

Correctness across the three loads per model:

| Model | Greedy match | Minimum first-step cosine | Maximum first-step absolute difference |
|---|---:|---:|---:|
| 0.4B | 1,536 / 1,536 | 1.0000001 | 0.0 |
| 1.5B | 1,536 / 1,536 | 1.0000002 | 0.0 |
| 2.9B | 1,536 / 1,536 | 1.0000000 | 0.0 |

Raw rows: `4080_b8_projection_bmm_ab.jsonl`.

## Current-policy smoke

A separate current-policy run used prompt 128, decode 256, 16 warmup calls and
three runs. Telemetry confirmed `native_graph_ada_wagv_bmm=true` and the
effective `native_graph` token backend:

| Model | Decode API | Decode throughput | Decode ms/step |
|---|---|---:|---:|
| 0.4B | `rwkv7_forward_token` | 3,210.9 tok/s | 2.49 |
| 1.5B | `rwkv7_forward_token` | 1,356.2 tok/s | 5.90 |

Raw rows: `4080_b8_current_policy.jsonl`. The JSON `fuse_norm` field reflects
the converted checkpoint configuration; native-graph norm/mix policy telemetry
is the authoritative decode-path field and is true for these rows.

## Reproduction

Paired A/B:

```bash
PYTHONPATH=. python bench/bench_native_graph_ada_wagv_lora.py \
  --hf-dir /path/to/model \
  --code-source repo --dtype fp16 --device cuda \
  --attn-mode fused_recurrent --axis ada_wagv_bmm \
  --batch-size 8 --prompt-tokens 64 \
  --correctness-steps 64 --warmup 16 --steps 512 --fixed-token \
  --num-warps 4 --results /tmp/4080_b8_projection_bmm_ab.jsonl
```

Current policy:

```bash
PYTHONPATH=. python bench/bench_batch_sweep.py \
  --hf-dir /path/to/model --model-size-label 1.5b \
  --code-source repo --dtype fp16 --device cuda \
  --attn-mode fused_recurrent --fast-token-backend native_graph \
  --fast-decode-api true --batch-sizes 8 \
  --prompt-tokens 128 --decode-tokens 256 --warmup 16 --runs 3
```

## Regression result

- Focused policy, kernel, native-graph, cache, benchmark, and module-split suite:
  **87 passed** on RTX 4080.
- CUDA numerical tests cover both W/A/V and layer-zero W/A grouped paths.
- B1/B2/B4 and non-RTX-4080 devices cannot enter this exact-B8 policy gate and
  continue through their existing measured routes.
