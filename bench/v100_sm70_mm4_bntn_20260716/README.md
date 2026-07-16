# V100 sm70 MM4 BN/TN production decode evidence

Status: **production decode gate passes for three exact V100 deployment
profiles**. This is not a universal W4 or prefill-speed claim.

## Contract

Every promoted profile covers these seven paired-fp16 cells:

```text
(batch,prompt,decode) =
(1,128,128), (2,128,128), (4,128,128), (8,128,128),
(1,512,128), (1,2048,128), (1,128,512)
```

Each cell runs in a fresh process with two warmups and three timing repeats.
All three seven-cell matrices were completed on the candidate rebased onto
`e333534`; the weakest 1.5B B4, 2.9B B8 and 7.2B B8 cells use five repeats. A
cell passes only when decode speed is at
least fp16, model footprint is lower, final-logit cosine is at least `0.998`,
the complete timed greedy sequence matches fp16, and all repeat greedy SHA256
values are identical.

## Environment

| Item | Value |
|---|---|
| GPU | Tesla V100-PCIE-32GB, `sm_70` |
| Driver | `580.159.03` |
| Torch / CUDA / Triton | `2.5.1+cu124` / `12.4` / `3.3.0` |
| Transformers | `5.12.1` |
| Base main commit | `e333534` |
| Matrix scope | current-main 1.5B, 2.9B and 7.2B full seven-cell matrices |
| Dtype / backend | fp16 / `native_graph`, `fused_recurrent` |

GPU0 retained an unrelated long-running process. All measurements used idle
GPU1; no benchmark was started on GPU0.

## Promoted decode profiles

| Model | One deployment config | Replaced modules | Decode vs fp16 | Footprint | Min final cosine | Gate |
|---|---|---:|---:|---:|---:|---:|
| 1.5B | memory, group128 on `lm_head`, fused epilogue | 49 | `1.0255x-1.1837x` | `0.5395x` | `0.99828702` | `7/7` |
| 2.9B | speed, group256 on `lm_head`, unfused | 1 | `1.0111x-1.0346x` | `0.9573x` | `0.99965668` | `7/7` |
| 7.2B | memory, group128 on `lm_head`, unfused | 193 | `1.0810x-1.8422x` | `0.3013x` | `0.99903870` | `7/7` |

All 21 current-main cells pass complete greedy equality and repeat determinism.
The three weakest cells record `1.0255x`, `1.0111x` and `1.0810x` decode. The
2.9B group128 confirmation remains rejected at `0.9984x`.

The claim is primarily cached decode. Full-memory prefill remains open: 1.5B
prefill is `0.1276x-0.3192x` fp16 and 7.2B is `0.0716x-0.1516x`. The head-only
2.9B speed profile separately passes all seven paired prefill cells at
`1.0006x-1.0603x`.

## Kernel decision

- B1 uses an A16 packed-W4 kernel; B2/B4/B8 use activation quantization and
  DP4A. Exact `(rows,K,N)` tables select independent BN/TN values.
- Group128 improves head quality enough for the full-memory 1.5B/7.2B lanes.
- Group256 halves scale groups for the 2.9B head. Its B8 `(BN,TN)=(32,1)` row
  reaches `1.132506x` same-shape fp16 in the 100-run microbenchmark and closes
  the end-to-end stability gap.
- Group size and group policy remain explicit configuration. Unsupported
  devices and shapes fall back; the existing default config is unchanged.
- Fused ReLU-squared/residual epilogues remain globally default-off. The exact
  1.5B profile enables them and moves rebased B4 from an unfused `0.9997x` to
  `1.0255x`. The 2.9B fused B4 row remains rejected at `0.9997x`.

The rejected WMMA prefill prototype reached only `0.054305x-0.115137x` fp16
on the measured shapes. Its runtime prototype was removed; the raw negative
rows are retained here.

## Reproduce

```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/home/data/wangyue/cuda-12.4-shim
export PATH="$CUDA_HOME/bin:/home/data/wangyue/envs/rwkv7/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export PYTHONPATH=$PWD

python bench/run_v100_sm70_mm4_production_matrix.py \
  --model 2.9b=/path/to/rwkv7-g1g-2.9b-hf \
  --policy speed --group-size 256 --group-policy lm_head \
  --fused-epilogue false \
  --output-dir /tmp/v100-2p9b-mm4
```

For 1.5B use `--policy memory --group-size 128 --fused-epilogue true`. For
7.2B use the same memory/group settings with `--fused-epilogue false`. The
runner exits nonzero if a row is missing, has the wrong policy, group or fused
configuration, or fails any production gate.

## Raw evidence

- [`production_1p5b_memory_fused.jsonl`](production_1p5b_memory_fused.jsonl)
- [`production_2p9b_group256_speed.jsonl`](production_2p9b_group256_speed.jsonl)
- [`production_7p2b_memory.jsonl`](production_7p2b_memory.jsonl)
- [`post_rebase_2p9b_b8_confirm.jsonl`](post_rebase_2p9b_b8_confirm.jsonl)
- [`post_rebase_7p2b_b8_confirm.jsonl`](post_rebase_7p2b_b8_confirm.jsonl)
- [`rowwise_bn_tn_screen.jsonl`](rowwise_bn_tn_screen.jsonl): 432 rows
- [`group128_head_screen.jsonl`](group128_head_screen.jsonl): 108 rows
- [`group256_head_screen.jsonl`](group256_head_screen.jsonl): 52 rows
- [`negative_2p9b_memory.jsonl`](negative_2p9b_memory.jsonl)
- [`negative_group128_2p9b_b8_confirm.jsonl`](negative_group128_2p9b_b8_confirm.jsonl)
- [`negative_wmma_prefill.jsonl`](negative_wmma_prefill.jsonl)
- [`negative_fused_epilogue_2p9b_b4.jsonl`](negative_fused_epilogue_2p9b_b4.jsonl)
- [`negative_pre_rebase_1p5b_unfused_matrix.jsonl`](negative_pre_rebase_1p5b_unfused_matrix.jsonl)
- [`negative_post_rebase_1p5b_b4_unfused.jsonl`](negative_post_rebase_1p5b_b4_unfused.jsonl)

The JSONL files are immutable measurements; this README defines their scope.
