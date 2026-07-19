# V100 Native-default regression close — 2026-07-18

This artifact validates the PR57 Native-Transformers migration on two exact
`Tesla V100-PCIE-32GB` cards (`sm_70`). It covers dense inference, recurrent
cache behavior, public chunked prefill, native W8/W4, and two-GPU DeepSpeed
checkpoint resume. It is a migration/regression artifact, not a claim that the
complete cross-card or Albatross performance project is finished.

## Verdict

- **Dense cached decode:** Native and the current historical wrapper are at
  parity for RWKV-7 1.5B at B1 and B8.
- **Dynamic batching/cache:** correctness passed; warmed graph-cache hit rate
  is `100%` across B2-B8 (`97.65625%` including the three compulsory misses).
- **Chunked prefill:** public Native cache continuation is correct and now uses
  vectorized Native prefill rather than a Python token loop.
- **W4 cached decode:** all 21 exact-card cells pass speed, footprint, logits,
  complete-greedy and repeat-determinism gates after repairing CUDA CCCL header
  discovery.
- **W8 cached decode:** effectively equal to FP16 (`0.9996x-1.0000x`); memory
  policy reduces loaded payload to `0.6932x` FP16. It is parity, not a material
  speed win.
- **Training:** two-card ZeRO2 and ZeRO3 save/resume/optimizer continuation pass.
- **Remaining performance work:** B1 short-prefill, small chunk scheduling, and
  full-memory W4/W8 prefill remain below the dense prefill route.

## CUDA extension failure found and closed

The first W4 matrix was catastrophically slow even though correctness and
memory passed. The reason was environmental, not Native-vs-wrapper dispatch:

```text
CUDA_HOME=/home/wzu/cuda-12.4-local
fatal error: nv/target: No such file or directory
```

The minimal CUDA tree contained `cuda_fp16.h`, but CUDA 12.4's CCCL header
`nv/target` lived under the active Conda prefix. The lazy sm70 extension failed
to compile and silently used the dequantized PyTorch fallback. The fallback
measured about `0.07x` FP16 on representative linears.

`rwkv7_hf/sm70_quant.py` now searches explicit, CUDA, Conda and NVIDIA Python
package include roots for `nv/target`, passes the discovered roots to the
extension builder, and builds extension revision `v20`. The benchmark records
`sm70_extension_build_error`, and the production matrix rejects a non-empty
error. After the repair, representative B1 linears reach `1.74x-3.04x` FP16;
see `current_rowwise_micro_b1.log`.

## Dense Native vs wrapper

Shape: prompt 128, decode 128, FP16, five timing repeats.

| Route | B | Prefill tok/s | Decode tok/s | Decode backend |
|---|---:|---:|---:|---|
| current wrapper | 1 | 5,831.4 | 150.7 | wrapper native_graph |
| Native model | 1 | 2,516.8 | 150.7 | native_graph |
| current wrapper | 8 | 17,495.5 | 897.6 | wrapper native_graph |
| Native model | 8 | 17,630.8 | 897.0 | native_graph |

The B1 short-prefill regression is real; decode is not regressed. With an exact
512-token prompt, Native prefill reaches `10,377.53` tok/s at B1 and
`20,139.77` tok/s at B8. Exact-512 Native decode is `150.69` tok/s at B1 and
`952.12` aggregate tok/s at B8. The decode benchmark now repeats its seed IDs
when necessary instead of silently truncating a requested prompt.

## Public chunked prefill

0.1B and 0.4B correctness passed for B2 and chunk sizes 1/2/4/8/16. Sequence
lengths match, continuation top-1 matches, and maximum absolute differences are
at most `0.09375` in that correctness run.

The 0.4B prompt-512 performance rows are:

| B | Mode | Chunk | tok/s | Ratio vs full | Peak/full | Max abs | Decode max abs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | full | 512 | 10,553.0 | 1.0000 | 1.0000 | — | — |
| 1 | chunked | 64 | 1,309.7 | 0.1241 | 1.2175 | 0.1250 | 0.0625 |
| 1 | chunked | 128 | 2,624.9 | 0.2487 | 1.2280 | 0.0625 | 0.0625 |
| 1 | chunked | 256 | 5,136.2 | 0.4867 | 1.2366 | 0.0625 | 0.0625 |
| 8 | full | 512 | 57,990.2 | 1.0000 | 1.0000 | — | — |
| 8 | chunked | 64 | 9,560.3 | 0.1649 | 0.9277 | 0.0625 | 0.0625 |
| 8 | chunked | 128 | 19,318.4 | 0.3331 | 0.9943 | 0.0625 | 0.0625 |
| 8 | chunked | 256 | 38,485.3 | 0.6637 | 1.0431 | 0.0625 | 0.0625 |

Chunking is therefore functional and vectorized, but small chunks still pay a
large launch/scheduling cost. Chunked mode is a serving/memory capability, not
yet a claim of full-prefill throughput parity.

## W4 exact-card production matrix

Each cell is a fresh-process paired FP16/MM4 run. The seven shapes are B1/B2/
B4/B8 at prompt/decode 128/128 plus B1 512/128, 2048/128 and 128/512.

| Model/profile | Cells | Decode/FP16 range | Footprint/FP16 | Final cosine range | Verdict |
|---|---:|---:|---:|---:|---|
| 1.5B memory, group128, fused epilogue | 7/7 | `1.0459x-1.3426x` | `0.5395x` | `0.99825090-0.99852973` | PASS |
| 2.9B speed, group256 | 7/7 | `1.0125x-1.0318x` | `0.9573x` | `0.99965668-0.99974990` | PASS |
| 7.2B memory, group128 | 7/7 | `1.0775x-2.2574x` | `0.3013x` | `0.99904591-0.99925733` | PASS |

Every row also passes complete greedy equality and repeat SHA determinism. The
memory profiles do **not** pass dense prefill speed: 1.5B is `0.1275x-0.7648x`
and 7.2B is `0.0722x-0.1799x`. The 2.9B head-only speed profile passes prefill
at `1.0099x-1.2154x`.

## W8

| Policy | B | Decode/FP16 | Prefill/FP16 | Footprint/FP16 | Final cosine |
|---|---:|---:|---:|---:|---:|
| speed (head only) | 1 | `1.0000x` | `0.9948x` | `0.9562x` | 0.99999547 |
| speed (head only) | 8 | `0.9998x` | `0.9222x` | `0.9562x` | 0.99996209 |
| memory (49 modules) | 1 | `1.0000x` | `0.8469x` | `0.6932x` | 0.99999487 |
| memory (49 modules) | 8 | `0.9996x` | `0.6093x` | `0.6932x` | 0.99996221 |

The `0.9996x-0.9998x` values are within 0.04% of FP16 and treated as measured
parity, not as evidence of a speedup.

## Two-GPU DeepSpeed resume

Model: Native-default RWKV-7 0.1B, FP32, two V100s, world size 2, first run one
step, resume to step two.

| ZeRO | First loss | Resume loss | First max delta | Resume max delta | Global step |
|---:|---:|---:|---:|---:|---:|
| 2 | 5.94135 | 2.52967 | 0.0718703 | 0.0719911 | 2 |
| 3 | 5.94135 | 2.55970 | 0.0001000 | 0.0001000 | 2 |

Both stages save, reload optimizer/model state, update trainable parameters
after resume, and end at the requested global step.

## Tests

- CPU regression selection: `453 passed, 21 skipped, 7 deselected`.
- Changed-path V100 selection: `56 passed`.
- Exact sm70 extension plus Native chunk/decode units: `19 passed`.

## Artifact map

- `environment.txt`: exact cards, software stack, CCCL location and source hashes.
- `native_vs_wrapper_15b_exact.jsonl`: dense same-card A/B.
- `native_{prefill,decode}_15b_idle_exact512.jsonl`: clean exact-512 probes.
- `dynamic_batch_{01b,04b}.jsonl`: graph-cache and batching evidence.
- `chunked_prefill_*`: public chunk correctness and performance.
- `mm4_{15b,29b,72b}/`: repaired 21-cell matrices and summaries.
- `native_mm8_{speed,memory}.jsonl`: post-repair W8 profiles.
- `native_mm8_15b_pre_cccl.jsonl`: retained negative evidence from the failed
  extension environment.
- `zero_resume_native_01b.jsonl`: rank-local ZeRO2/3 resume evidence.
