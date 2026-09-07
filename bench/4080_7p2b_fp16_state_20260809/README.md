# RTX 4080 RWKV-7 7.2B/B8 FP16-state decode acceptance

This directory records the exact-card acceptance for the native-graph Triton
FP16 recurrent-state route.  The production default is deliberately limited
to desktop RTX 4080, RWKV-7 hidden size 4096, 32 layers, FP16 weights, and
batch size 8.  Adjacent cards, model shapes, dtypes, and batch sizes retain the
existing state path unless explicitly overridden for benchmarking.

## Change

- Reuse the existing fused raw recurrent/output Triton kernel with FP16 state.
- Avoid the separate JIT-built CUDA FP16 recurrent extension on this route.
- Cut recurrent-state traffic and storage in half.
- Preserve FP32 and native-extension fallbacks for every unmeasured shape.
- Exclude 7.2B (`hidden=4096`) from the grouped W/A/G/V BMM route after its
  paired probe measured `0.9994x` speed with about `+200 MiB` peak VRAM.

## Paired result

Three independent process runs used B8, prompt length 64, 512 greedy
correctness steps, 16 timing warmups, and 512 fixed-token timing steps.
One run reversed candidate/baseline order.

| Metric | FP32-state baseline | Triton FP16-state | Result |
| --- | ---: | ---: | ---: |
| Median latency | 23.9283 ms/step | 23.2292 ms/step | **1.0301x** |
| Median throughput | 334.33 tok/s | 344.39 tok/s | **+3.01%** |
| Median peak allocated VRAM | 14548.85 MiB | 14424.98 MiB | **-123.88 MiB** |
| Run-to-run speedup | - | - | 1.0298x–1.0305x |
| Greedy decode | - | - | **12288 / 12288 exact** |
| Minimum first-step cosine | - | - | 0.99999475 |
| Maximum first-step absolute difference | - | - | 0.0625 |
| Graph-cache hit rate | 99.9039% | 99.9039% | unchanged |

The standard repository batch sweep, with no state-related environment
overrides, confirmed that the default policy selected `torch.float16` state,
the Triton route, and the native-graph backend.  Its serving-style B8 decode
row measured 344.1 tok/s and 23.25 ms/step.

## Reproduction

```bash
export PYTHONPATH="$PWD"
python bench/bench_native_graph_state_dtype.py \
  --hf-dir /path/to/rwkv7-g1h-7.2b-hf-converted \
  --batch-size 8 \
  --prompt-tokens 64 \
  --correctness-steps 512 \
  --warmup 16 \
  --steps 512 \
  --results results.jsonl

python bench/bench_batch_sweep.py \
  --hf-dir /path/to/rwkv7-g1h-7.2b-hf-converted \
  --model-size-label 7.2b \
  --dtype fp16 \
  --device cuda \
  --code-source repo \
  --attn-mode fused_recurrent \
  --fast-token-backend native_graph \
  --batch-sizes 8 \
  --prompt-tokens 64 \
  --decode-tokens 256 \
  --warmup 4 \
  --runs 3 \
  --results policy_smoke.jsonl
```

## Regression

- Focused native-graph/kernel suite: `91 passed` plus raw CUDA recurrent test
  `PASS` (`tests.log`).
- Full Linux/RTX 4080 suite excluding the pre-existing environment-rendered
  Apple acceptance document check: `802 passed, 9 skipped, 1 deselected`
  (`full_tests.log`).
- The A8W8 fused-FFN CUDA functional test now explicitly enables its kernel,
  separating numerical coverage from exact-card performance policy.

Artifacts:

- `results.jsonl`: three paired A/B rows.
- `policy_smoke.jsonl`: standard batch-sweep default-policy row.
- `environment.json`: hardware and software identity.
- `tests.log`: focused regression output.
- `full_tests.log`: broad Linux/RTX 4080 regression output.
