# RTX 3090 Qwen3.5 paired Prefill/Decode v2

**Status: PASS — all 48 cells strictly exceed the optimized Qwen3.5
reference in raw and parameter-adjusted Prefill and Decode throughput.**

This artifact compares the official RWKV-7 g1d/g1i
0.4B/1.5B/2.9B/7.2B checkpoints with Qwen3.5 0.8B/2B/4B/9B on one
`NVIDIA GeForce RTX 3090`. The matrix is B1/B8 × prompt 128/512/2048 ×
decode 128/512, with three warmups and seven measured samples per cell.
Every gate uses unrounded raw values and requires strict `> 1.0`.

The parameter adjustment is:

```text
adjusted ratio = (RWKV tok/s / Qwen tok/s)
                 * (RWKV active parameters / Qwen active parameters)
```

## Result

| Gate | Passed | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| Raw Prefill | 48/48 | `1.574925x` | `2.182651x` | `8.428073x` |
| Parameter-adjusted Prefill | 48/48 | `1.208324x` | `1.535161x` | `5.049362x` |
| Raw Decode | 48/48 | `1.253926x` | `1.740275x` | `3.094400x` |
| Parameter-adjusted Decode | 48/48 | `1.017763x` | `1.207730x` | `1.853893x` |

The narrowest adjusted Prefill cell is 0.4B/0.8B, B8/P512/D512:
`1.208324x` (`79,021.219` versus `39,180.376` raw Prefill tok/s). The
narrowest adjusted Decode cell is 1.5B/2B, B1/P128/D128: `1.017763x`
(`231.809` versus `184.867` raw Decode tok/s), leaving a `+1.7763%`
adjusted margin.

Per pair and batch, the values below are minimum / median across the six P/D
cells:

| Pair | Batch | Raw Prefill | Adjusted Prefill | Raw Decode | Adjusted Decode |
|---|---:|---:|---:|---:|---:|
| 0.4B / 0.8B | B1 | `3.695604x / 4.311974x` | `2.214082x / 2.583356x` | `1.742779x / 1.938406x` | `1.044128x / 1.161323x` |
| 0.4B / 0.8B | B8 | `2.016857x / 2.394471x` | `1.208324x / 1.434557x` | `1.832772x / 2.155534x` | `1.098004x / 1.291407x` |
| 1.5B / 2B | B1 | `2.299866x / 2.312865x` | `1.866712x / 1.877263x` | `1.253926x / 1.322339x` | `1.017763x / 1.073291x` |
| 1.5B / 2B | B8 | `1.581052x / 1.607761x` | `1.283279x / 1.304957x` | `1.298576x / 1.433855x` | `1.054003x / 1.163805x` |
| 2.9B / 4B | B1 | `1.892200x / 1.930958x` | `1.326209x / 1.353374x` | `1.737759x / 1.788087x` | `1.217963x / 1.253238x` |
| 2.9B / 4B | B8 | `2.181448x / 2.293401x` | `1.528937x / 1.607403x` | `1.816888x / 2.009969x` | `1.273424x / 1.408751x` |
| 7.2B / 9B | B1 | `1.574925x / 1.629266x` | `1.266289x / 1.309982x` | `1.366759x / 1.396962x` | `1.098917x / 1.123201x` |
| 7.2B / 9B | B8 | `1.683825x / 1.767953x` | `1.353849x / 1.421490x` | `1.296517x / 1.384378x` | `1.042441x / 1.113084x` |

## Correctness and routes

Eight independent long-horizon comparisons cover all four model pairs at B1
and B8, P2048/D512. Each comparison uses the repository HF wrapper with the
official FLA operator path and `RWKV7StateCache` as the mathematical reference,
and `native_model + native_graph + NativeRWKV7Cache` as the candidate. Every
comparison checks 512 autoregressive steps, requires finite Decode logits at
every step, exact greedy-token agreement, matching input IDs, and prompt/final
logits cosine `>= 0.9999`. All 8/8 pass; the global minimum cosine is
`0.999987364`. B8 correctness probes use eight distinct prompts.

Qwen fixes one passing StaticCache Graph route per model: 0.8B/2B use
Inductor CUDA Graph and 4B/9B use raw CUDA Graph. RWKV uses the exact
`sm86_qwen_alignment` route profile without per-cell fallback. B1 uses the
verified WAGV extension path. Small-model B8 uses 24/24-layer projection BMM,
G epilogue and compiled FFN telemetry; 2.9B/B8 uses 32/32-layer BMM; 7.2B/B8
keeps the measured manual-RKV route. Requested, selected and effective route
fields are all checked by the validator.

## Locked environment

- GPU: `NVIDIA GeForce RTX 3090`, SM 8.6, 24,576 MiB, driver `550.142`.
- Python `3.10.12`; PyTorch `2.7.1+cu126`; CUDA runtime `12.6`; Triton
  `3.3.1`.
- Transformers `5.12.1`; FLA `0.5.1`; causal-conv1d `1.6.2.post1`.
- Qwen reference commit: `8f98046dc274d24441d6b3b8eb1578496fee6b2d`.
- RWKV candidate commit: `c0c743af251763a8be7669f8b7fa319435b4c766`.
- Qwen reference SHA256:
  `cb5d86e7a4a8cabad1f7dd86c0187dbf514b92cfafccdb0699026dc9e9696c81`.
- RWKV candidate SHA256:
  `b10eb856acf4d58867947d44aff44506fcf616cf0f86338b553dd5e6b95fb1c9`.

Model hashes were captured before and after all GPU work and are
byte-identical. The validator reports `status=pass`,
`paired_pd_table_eligible=true`, `errors=[]`, and exits zero. All remote work
was contained under `/home/ubuntu/rwkv-3090-alignment/`.

## Reproduce validation

From the repository root, with the 16 external `.pt` probes restored beside
the text artifacts:

```bash
python -m bench.validate_qwen35_3090_paired_pd_v2 \
  --candidate bench/3090_qwen35_paired_pd_v2_20260816/rwkv_candidate.jsonl \
  --reference bench/3090_qwen35_paired_pd_v2_20260816/qwen_reference.jsonl \
  --reference-contract bench/3090_qwen35_paired_pd_v2_20260816/reference-contract.json \
  --correctness-manifest bench/3090_qwen35_paired_pd_v2_20260816/rwkv_native_graph_fla_correctness.json \
  --runtime-lock bench/3090_qwen35_paired_pd_v2_20260816/runtime-lock.json \
  --model-hashes bench/3090_qwen35_paired_pd_v2_20260816/model_hashes.sha256 \
  --model-hashes-after bench/3090_qwen35_paired_pd_v2_20260816/model_hashes.after.sha256 \
  --system bench/3090_qwen35_paired_pd_v2_20260816/system.csv \
  --summary /tmp/3090-validation.json \
  --paired-table /tmp/3090-paired.jsonl \
  --markdown /tmp/3090-summary.md
```

## Artifacts

- `qwen_reference.jsonl` and `qwen_{0p8,2b,4b,9b}.jsonl`: frozen optimized
  Qwen reference rows.
- `rwkv_candidate.jsonl` and `rwkv_{0p4,1p5,2p9,7p2}_b{1,8}.jsonl`: formal
  RWKV rows.
- `validation.json`, `paired_pd_table.jsonl`, and `summary.md`: validator
  output and all 48 joined cells.
- `rwkv_native_graph_fla_correctness.json`, eight `*_compare.json` files and
  eight FLA row files: long-horizon correctness evidence.
- `runtime-lock.json`, `pip-freeze.txt`, `system.csv`, and
  `model_hashes*.sha256`: runtime, hardware and checkpoint identity.
- `probe_artifact_sha256.txt`: hashes for the 16 tensor probes retained in the
  external full evidence directory. The `.pt` tensors are intentionally not
  committed, so a fresh clone alone cannot recompute tensor cosine.
- `logs/`: 17 formal lane and validation logs.
- `artifact_sha256.txt`: hashes for every committed artifact file except
  itself.

## Claim boundary

This proves that, for this exact RTX 3090, runtime, checkpoints, routes and
48-cell matrix, RWKV strictly exceeds the optimized Qwen baseline in both raw
and parameter-adjusted Prefill and Decode throughput. It is an inference-engine
speed result, not a model-quality result, and does not establish TTFT,
continuous end-to-end serving, scheduler or cache-handoff latency superiority.
The Qwen reference and RWKV candidate use separate clean commits and were
captured sequentially, not as an interleaved A/B. Performance B8 inputs
replicate one prompt across the batch; only the correctness probes use eight
distinct prompts.
