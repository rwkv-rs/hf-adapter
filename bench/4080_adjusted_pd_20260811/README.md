# RTX 4080: all parameter-adjusted Prefill and Decode medians exceed Qwen3.5

**Status: PASS.** All six RWKV-7/Qwen3.5 model-pair and batch groups exceed
`1.00x` in both parameter-size-adjusted Prefill and Decode median throughput.
Each median covers the six `P128/512/2048 × D128/512` cells on the same
physical RTX 4080 in dense FP16.

| Pair | Batch | Raw Prefill / Decode | Adjusted Prefill / Decode | E2E raw / adjusted |
|---|---:|---:|---:|---:|
| 0.4B / 0.8B | B1 | **1.82x / 4.89x** | **1.09x / 2.93x** | **4.81x / 2.88x** |
| 0.4B / 0.8B | B8 | **1.77x / 4.16x** | **1.06x / 2.49x** | **3.93x / 2.36x** |
| 1.5B / 2B | B1 | **1.54x / 1.90x** | **1.25x / 1.55x** | **1.90x / 1.54x** |
| 1.5B / 2B | B8 | **1.76x / 1.77x** | **1.43x / 1.44x** | **1.77x / 1.44x** |
| 2.9B / 4B | B1 | **1.75x / 1.63x** | **1.22x / 1.15x** | **1.63x / 1.14x** |
| 2.9B / 4B | B8 | **1.99x / 1.75x** | **1.40x / 1.23x** | **1.77x / 1.24x** |

The adjustment is:

```text
adjusted ratio = (RWKV tok/s / Qwen tok/s)
                 × (RWKV active parameters / Qwen active parameters)
```

Exact parameter counts and every cell ratio are retained in `summary.json`.

## Promoted route and parity

The exact desktop RTX 4080 policy enables PyTorch native FP16 GEMM
accumulation only inside a locked native-Prefill scope and restores the
process-global setting afterward. An 18-shape FP16-versus-FP32 accumulation
sweep kept the 15 promoted shapes that passed both prefill and decode greedy
parity with cosine above `0.9999`. Three boundary shapes retain FP32
accumulation. See `parity/promoted_summary.json` and the per-model JSONL files.

The candidate uses the validated native RWKV runtime, while the Qwen reference
uses its validated full-FLA + Triton causal-conv runtime. Both runtime stacks,
backend telemetry, active parameter counts, shapes, and device identity are
recorded rather than inferred. The Qwen rows are the optimized reference rows
from `4080_full_model_ladder_20260719`; the new candidate rows were measured on
the same RTX 4080.

## Live GPU reproduction

Create the two exact environments listed in `environment.json`, place the six
local HF model directories, then run:

```bash
RWKV_PYTHON_BIN=/path/to/rwkv-torch2.11/bin/python \
QWEN_PYTHON_BIN=/path/to/qwen-full-fla/bin/python \
RWKV_04_MODEL=/models/rwkv7-0.4b-hf \
RWKV_15_MODEL=/models/rwkv7-1.5b-hf \
RWKV_29_MODEL=/models/rwkv7-2.9b-hf \
QWEN_08_MODEL=/models/Qwen3.5-0.8B \
QWEN_2_MODEL=/models/Qwen3.5-2B \
QWEN_4_MODEL=/models/Qwen3.5-4B \
OUT_DIR=/tmp/4080-adjusted-pd \
  bash bench/run_4080_adjusted_pd.sh
```

The runner checks the exact desktop RTX 4080 and both runtime contracts, then
loads all six models on the GPU, performs 3 warmups and 7 timed runs per cell,
verifies the Qwen fast path, emits 36 RWKV and 36 Qwen rows, and exits nonzero
unless every parameter-adjusted Prefill and Decode group median is above
`1.00x`.

To recalculate the committed evidence without rerunning the GPU:

```bash
python bench/summarize_4080_adjusted_pd.py \
  bench/4080_adjusted_pd_20260811/candidate.jsonl \
  bench/4080_adjusted_pd_20260811/qwen_reference.jsonl \
  --output /tmp/summary.json --markdown-output /tmp/summary.md
```

## Artifacts

- `candidate.jsonl`: 36 new native-RWKV GPU measurements.
- `qwen_reference.jsonl`: 36 optimized full-FLA Qwen GPU measurements.
- `summary.json`, `summary.md`: all formulas, cell ratios, medians, and gate.
- `parity/`: numerical promotion evidence.
- `environment.json`: exact device, runtime, and benchmark contract.
- `SHA256SUMS`: artifact integrity.
