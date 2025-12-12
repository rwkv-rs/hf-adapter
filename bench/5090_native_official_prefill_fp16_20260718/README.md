# RTX 5090 Native prefill FP16-state close

This artifact compares the Native/no-FLA HF sequence-prefill backend with the
pinned official RWKV-Gradio v3a implementation on one RTX 5090 and the same
g1h 7.2B checkpoint. Both sides use FP16 weights, FP16 recurrent state and the
same B1/B8 prompt tensors. The matrix captures the final prompt logits,
first-decode logits, every layer output, recurrent state, `xpa`, `xpf` and both
greedy tokens before applying a throughput gate.

## Result

All six exact-shape cells pass quality and Native throughput is at least the
official throughput in every measured cell:

| Batch | Prompt | Native ms | Official ms | Native/reference | Native/official peak MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | `14.6880` | `15.0948` | `1.0277x` | `13808.8 / 14232.0` |
| 1 | 512 | `46.8325` | `50.9794` | `1.0885x` | `13879.9 / 14355.0` |
| 1 | 2048 | `172.2062` | `183.6339` | `1.0664x` | `14218.3 / 14577.3` |
| 8 | 128 | `69.5166` | `77.7371` | `1.1183x` | `14163.2 / 14633.3` |
| 8 | 512 | `292.7793` | `300.0388` | `1.0248x` | `14788.7 / 15166.5` |
| 8 | 2048 | `1184.3094` | `1209.0208` | `1.0209x` | `17498.3 / 17295.6` |

The lowest final-logits cosine is `0.9999997622`; the largest final-logits
absolute difference is `0.09375`. The lowest recurrent-state cosine is
`0.9999910179`, the largest state absolute difference is `2.5859375`, and the
largest state mean absolute difference is `0.00029429`. Prompt and first-decode
greedy tokens match in all cells.

The exact RTX 5090 g1h-7.2B B8/P128 policy enables reduced FP16 accumulation
only for the FFN key GEMM. It is the only matrix cell reporting
`fp16_accum_ffn_key_effective=true`; the other five cells prove the ordinary
FP16-state route. This is an exact-card, exact-model, exact-shape policy, not a
global CUDA default.

Native peak allocation is lower in five cells. B8/P2048 is the explicit
exception: Native uses about `202.7 MiB` more peak VRAM while remaining faster
and quality-passing. The result therefore closes the measured speed/quality
lane, not a blanket memory-parity claim.

## Environment

- RTX 5090, SM120, driver `595.58.03`
- PyTorch `2.11.0+cu128`, CUDA toolkit/runtime `12.8`
- Transformers `5.12.1`, Triton `3.6.0`, Python `3.10.12`
- official source commit `cc57df475465c6cacd42ecd4f2f05a588ee5473b`
- checkpoint `rwkv7-g1h-7.2b-20260710-ctx10240.pth`
- 10 warmups and 21 timed repeats per backend and cell

[`official_source_manifest.json`](official_source_manifest.json) pins the
official Python source hashes. Every report also records the source revision,
precision contract, thresholds, all 21 timings and route telemetry.

## Reproduce and recover

Refresh a converted checkpoint with the current remote-code manifest first:

```bash
python scripts/sync_hf_adapter_code.py /absolute/path/to/rwkv7-g1h-7.2b-hf
```

Then run the fail-closed matrix from a checkout of this repository:

```bash
PYTHONPATH=. python bench/run_official_native_prefill_matrix.py \
  --hf-dir /absolute/path/to/rwkv7-g1h-7.2b-hf \
  --official-dir /absolute/path/to/pinned/RWKV-Gradio-3 \
  --official-model /absolute/path/to/rwkv7-g1h-7.2b-20260710-ctx10240.pth \
  --official-source-manifest /absolute/path/to/official_source_manifest.json \
  --output-dir /absolute/path/to/prefill-results \
  --cases 1x128,1x512,1x2048,8x128,8x512,8x2048 \
  --warmup 10 --repeats 21
```

A successful run writes `exit_code.txt` containing `0`, `summary.json` with
`status=pass`, `quality_pass_cases=6` and `performance_pass_cases=6`, and six
reports with `status=pass`. If interrupted, rerun the same command with
`--skip-existing`; completed captures are verified and retained. If a CUDA
extension build failed, fix the matching PyTorch/CUDA toolchain before
resuming. Do not combine captures from another card, checkpoint or official
source revision.

The `.pt` tensor captures are intentionally omitted from Git because they total
about 1.1 GiB. The committed reports contain their complete quality metrics,
timing samples, route selection and pass/fail decisions. AI-assisted execution
uses the single repository entry point
[`docs/AI_ASSISTED_SETUP.md`](../../docs/AI_ASSISTED_SETUP.md).

## Boundary

This evidence covers prefill plus the first decode handoff at B1/B8 and prompt
128/512/2048 on one RTX 5090. It does not prove another model, another card,
arbitrary batch/prompt shapes, decode throughput, training, quantization or a
global default. Exact-card evidence is required before extending the
FFN-key accumulation policy.
