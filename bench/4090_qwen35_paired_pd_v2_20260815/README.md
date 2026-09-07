# RTX 4090 Qwen3.5 paired Prefill/Decode v2

**Status: PASS — all 48 cells strictly exceed the optimized Qwen3.5 baseline in raw and parameter-adjusted Prefill and Decode throughput.**

This artifact compares the official RWKV-7 g1d/g1i 0.4B/1.5B/2.9B/7.2B checkpoints with Qwen3.5 0.8B/2B/4B/9B on one `NVIDIA GeForce RTX 4090`. The matrix is B1/B8 × prompt 128/512/2048 × decode 128/512, with three warmups and seven measured samples per cell. Every gate uses unrounded raw values and requires strict `> 1.0`.

The adjustment is:

```text
adjusted ratio = (RWKV tok/s / Qwen tok/s)
                 * (RWKV active parameters / Qwen active parameters)
```

## Result

| Gate | Passed | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| Raw Prefill | 48/48 | `1.415206x` | `2.449410x` | `12.686421x` |
| Parameter-adjusted Prefill | 48/48 | `1.148668x` | `1.695334x` | `7.600590x` |
| Raw Decode | 48/48 | `1.276285x` | `1.770640x` | `3.116990x` |
| Parameter-adjusted Decode | 48/48 | `1.026173x` | `1.323737x` | `1.867427x` |

The narrowest adjusted Prefill cell is RWKV-1.5B/Qwen-2B, B8/P512/D128: `1.148668x` (`58,134.180` versus `41,078.237` raw Prefill tok/s). The narrowest adjusted Decode cell is RWKV-7.2B/Qwen-9B, B8/P128/D128: `1.026173x` (`448.713` versus `351.578` raw Decode tok/s). Thus the weakest formal cell still has a `2.6173%` adjusted Decode margin.

Per pair and batch, the values below are minimum / median across the six P/D cells:

| Pair | Batch | Raw Prefill | Adjusted Prefill | Raw Decode | Adjusted Decode |
|---|---:|---:|---:|---:|---:|
| 0.4B / 0.8B | B1 | `6.755416x / 7.186935x` | `4.047252x / 4.305781x` | `1.759737x / 1.937215x` | `1.054280x / 1.160609x` |
| 0.4B / 0.8B | B8 | `2.015169x / 2.248138x` | `1.207313x / 1.346887x` | `1.881462x / 2.153583x` | `1.127207x / 1.290238x` |
| 1.5B / 2B | B1 | `3.921957x / 4.021414x` | `3.183301x / 3.264026x` | `1.577906x / 1.647296x` | `1.280725x / 1.337046x` |
| 1.5B / 2B | B8 | `1.415206x / 1.567633x` | `1.148668x / 1.272387x` | `1.352989x / 1.467867x` | `1.098169x / 1.191411x` |
| 2.9B / 4B | B1 | `2.749972x / 2.783136x` | `1.927406x / 1.950650x` | `2.174049x / 2.265716x` | `1.523752x / 1.587999x` |
| 2.9B / 4B | B8 | `1.771597x / 1.942331x` | `1.241680x / 1.361345x` | `1.667381x / 1.833305x` | `1.168637x / 1.284930x` |
| 7.2B / 9B | B1 | `1.543417x / 1.583476x` | `1.240956x / 1.273165x` | `1.637199x / 1.677814x` | `1.316360x / 1.349016x` |
| 7.2B / 9B | B8 | `1.495272x / 1.618440x` | `1.202246x / 1.301277x` | `1.276285x / 1.355142x` | `1.026173x / 1.089577x` |

## Correctness and routes

Eight independent long-horizon comparisons cover all four model pairs at B1 and B8, P2048/D512. Each comparison uses the official FLA wrapper as the mathematical reference and native Graph as the candidate, checks 512 autoregressive steps, requires finite Decode logits at every step, exact greedy-token agreement, matching input IDs, and prompt/final logits cosine `>= 0.9999`. All 8/8 pass; the global minimum prompt/final cosine is `0.9999929667`. B8 correctness probes use eight distinct prompts.

The FLA correctness reference runs eagerly (`TORCH_COMPILE_DISABLE=1`, `TORCHDYNAMO_DISABLE=1`) because PyTorch 2.7.1 generated Inductor workers import an obsolete Triton descriptor with Triton 3.3.1. This affects only the correctness oracle; no FLA timing enters the performance table. Two failed compile attempts are retained in `logs/` before the successful eager rerun.

Qwen selected one route per model, with no per-cell mixing:

- 0.8B and 2B: `StaticCache + Inductor CUDA Graph`, `max-autotune`.
- 4B and 9B: `StaticCache + raw CUDA Graph`; their Inductor probes failed the same-cache `0.9999` cosine gate and were rejected before formal timing.

RWKV uses `native_model + native_graph + NativeRWKV7Cache` for all 48 rows. Small-model B8 enables the exact RTX 4090 fused W/A/G/V projection route and compiled dense FFN route, both with complete 24/24-layer telemetry, one reused compiled graph, and zero graph breaks. Other shapes retain their measured exact-card route without silent fallback.

## Locked environment

- GPU: `NVIDIA GeForce RTX 4090`, SM 8.9, driver `550.142`.
- Python `3.12.8`; PyTorch `2.7.1+cu126`; CUDA runtime `12.6`; Triton `3.3.1`.
- Transformers `5.12.1`; FLA `0.5.1`; causal-conv1d `1.6.2.post1`.
- Qwen reference commit: `bdb0e4a66aa0b97d06d7fee5a4b304aeaf8923d0`.
- RWKV candidate commit: `9b2bde5060c92d39acbb6ef58706dfd7c8a84264`.
- Qwen reference SHA256: `7274b4ba3c549320740a4ea3bf7d72ce4dcafb1a671e6ab01e4fa1c1ba1db24f`.
- RWKV candidate SHA256: `42ced74ad19a3aca28405e5acb640f05f993a36e0172153d2b91dfebd3dedf80`.

Model hashes were captured before and after all GPU work and are byte-identical. The paired validator reports `status=pass`, `paired_pd_table_eligible=true`, and `errors=[]`.

## Artifacts

- `qwen_reference.jsonl` and `qwen_{0p8,2b,4b,9b}.jsonl`: frozen optimized Qwen reference rows.
- `rwkv_candidate.jsonl` and `rwkv_{0p4,1p5,2p9,7p2}_b{1,8}.jsonl`: formal RWKV rows.
- `paired_validation.json`, `paired_pd_table.jsonl`, `paired_pd.md`: strict validator output and all 48 joined cells.
- `rwkv_native_graph_fla_correctness_final.json` and eight `*_compare_final.json` files: correctness manifest and comparison results.
- `runtime-lock.json`, `pip-freeze.txt`, `system.csv`, `model_hashes*.sha256`: runtime, hardware, and checkpoint identity.
- `probe_artifact_sha256.txt`: hashes for the 16 tensor probes retained in the external full evidence directory. The `.pt` tensors are intentionally not committed; a fresh clone alone cannot recompute tensor cosine without those external files.
- `logs/`: formal lane logs and correctness-oracle retry history.
- `artifact_sha256.txt`: hashes for every committed artifact file except itself.

## Claim boundary

This proves that, for this exact RTX 4090, runtime, checkpoints, routes, and 48-cell matrix, RWKV strictly exceeds the optimized Qwen baseline in both raw and parameter-adjusted Prefill and Decode throughput. It is not a model-quality result and does not establish TTFT, continuous end-to-end serving, scheduler, or cache-handoff latency superiority. Performance B8 inputs replicate one prompt across the batch; only the correctness probes use eight distinct prompts.

