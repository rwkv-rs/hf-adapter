# Qwen3.5 Apple / mobile baseline

This document defines the first reproducible gate for the goal: **RWKV-7 HF /
Apple MLX / CoreML should beat Qwen3.5 on Apple/mobile deployment metrics**.

The current repository does **not** claim that this gate is complete.  The point
of this lane is to make "beat Qwen3.5" measurable before deeper MLX fused and
CoreML/ANE optimization work starts.

## Public comparison targets

Use same-device, same-prompt-text runs against the public Qwen3.5 MLX/mobile
packages.  The initial public size classes are:

| Baseline | Runtime | Public package size | Use in gate |
|---|---|---:|---|
| `qwen3.5:0.8b-mlx` | Ollama / MLX | 1.2GB | tiny/mobile floor |
| `qwen3.5:2b-mlx` | Ollama / MLX | 3.1GB | 1.5B-ish speed/memory comparison |
| `qwen3.5:4b-mlx` | Ollama / MLX | 4.0GB | 2.9B-ish comparison |
| `qwen3.5:9b-mlx` | Ollama / MLX | 8.9GB | upper mobile/local comparison |

Reference pages:

- [Ollama qwen3.5](https://ollama.com/library/qwen3.5)
- [MollySophia/rwkv-mobile](https://github.com/MollySophia/rwkv-mobile)

## Metrics that count

A row only supports the goal when it records all relevant fields in JSONL:

| Area | Required fields |
|---|---|
| Device | platform, machine, macOS version, memory when available |
| Prompt | prompt case name, prompt character target, actual tokenizer prompt token count when available |
| Generation | requested generated tokens, actual generated tokens, response preview |
| Speed | TTFT if available, prefill tok/s, decode tok/s, wall time |
| Memory | MLX active/peak/cache memory or runtime-native memory telemetry |
| Quant | W8/W4/LUT/INT4 mode, backend, quant min params, fallback/Metal counts when available |
| State cache | chunked prefill max diff, seen-token checks, batch/session backend where applicable |
| Evidence | append-only JSONL path plus command line used to produce it |

## Harness

The shared baseline runner is:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py --help
```

For end-to-end local collection, use the one-command wrapper:

```bash
# Dry-run the full Qwen/RWKV/CoreML plan without contacting runtimes.
DRY_RUN=1 \
RWKV_MLX_MODELS=/path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
COREML_EXPORT_MODELS=/path/to/rwkv7-g1g-1.5b-hf \
scripts/run_qwen35_apple_acceptance.sh

# Live same-device acceptance. Set PULL_QWEN=1 only when you want the wrapper
# to run `ollama pull` before collecting rows.
PULL_QWEN=1 \
RWKV_MLX_MODELS=/path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
COREML_EXPORT_MODELS=/path/to/rwkv7-g1g-1.5b-hf \
RESULTS=bench/results_qwen35_apple_baseline.jsonl \
scripts/run_qwen35_apple_acceptance.sh
```

The wrapper runs `bench/run_qwen35_apple_baseline.py`, optionally runs
`scripts/export_rwkv7_coreml.py`, optionally emits CoreML runtime/plan rows via
`bench/run_coreml_apple_baseline.py`, optionally scores response quality via
`bench/score_qwen35_quality.py`, then appends
`bench/compare_qwen35_apple_baseline.py` gate rows.  The default comparison
pairs cover the currently available 0.4B/1.5B RWKV classes; override `PAIRS`,
`QWEN_MODELS`, and `RWKV_MLX_MODELS` for 4B/9B or distilled-mobile gates.

For reproducible prefill rows, the wrapper defaults to `OLLAMA_THINK=0`,
`OLLAMA_KEEP_ALIVE=0`, and `OLLAMA_CACHE_PROMPT=0`. This keeps short thinking
traces out of `response_text` and unloads Ollama after each row so a completed
prompt cannot be reported as near-zero prefill. The runner records both steady
`ttft_s` (load duration removed) and load-inclusive `cold_ttft_s`. Override
these defaults only when deliberately measuring a shared prompt-cache service.
With the default isolated policy it temporarily keeps the model alive long
enough to query official `/api/ps`, records `ollama_loaded_memory_bytes`, and
then explicitly unloads it. This is loaded-runtime memory, not peak memory, so
the strict peak-to-peak gate remains unknown.

The wrapper also defaults `RWKV_PREFILL_EVAL_INTERVAL=2`. This batches two
lazy MLX recurrent prompt steps between graph evaluations. The reusable model
API keeps the safer interval-1 default. Before changing this value on a new
model/device, run `scripts/mlx_prefill_eval_interval_bench.py`; it treats
logits, all recurrent/cache tensors, seen-token count, and next-token parity as
a hard gate rather than inferring correctness from throughput alone.

It emits rows with `axis=qwen35_apple_baseline` and can run:

1. Qwen3.5 through a local Ollama server using the streaming `/api/generate`
   endpoint.
2. RWKV-7 through this repository's optional MLX recurrent backend.
3. RWKV-7 through the stateful CoreML multifunction runner; confirmed ANE
   placement remains a separate gate.

The companion export entry point is `scripts/export_rwkv7_coreml.py`; the companion runtime row generator is `bench/run_coreml_apple_baseline.py`.  It writes
a reproducible CoreML export manifest in `--dry-run` mode on any machine. With
`--export-kind stateful-multifunction` it exports masked `prefill` and one-token
`decode` functions with packed RWKV Core ML state. The runtime records state
transfer, chunk-boundary drift, HF greedy parity, TTFT, throughput, package
bytes, and peak process memory.

Dry-run the matrix without contacting runtimes:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --dry-run \
  --prompt-target-chars 1024,4096 \
  --decode-lengths 128,512 \
  --qwen-models qwen3.5:0.8b-mlx,qwen3.5:2b-mlx,qwen3.5:4b-mlx \
  --rwkv-mlx-models /path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Run Qwen3.5 baselines after pulling models into Ollama:

```bash
ollama pull qwen3.5:0.8b-mlx
ollama pull qwen3.5:2b-mlx
ollama pull qwen3.5:4b-mlx

PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 1024,4096,8192 \
  --decode-lengths 128,512 \
  --qwen-models qwen3.5:0.8b-mlx,qwen3.5:2b-mlx,qwen3.5:4b-mlx \
  --rwkv-mlx-models '' \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Run RWKV-7 MLX rows against the same prompt text:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 1024,4096,8192 \
  --decode-lengths 128,512 \
  --qwen-models '' \
  --rwkv-mlx-models /path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization none \
  --rwkv-wkv-backend metal \
  --rwkv-chunk-size 2048 \
  --rwkv-prefill-eval-interval 2 \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Run RWKV-7 W4/Metal rows:

```bash
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 1024,4096,8192 \
  --decode-lengths 128,512 \
  --qwen-models '' \
  --rwkv-mlx-models /path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization mm4 \
  --rwkv-quant-min-params 4000000 \
  --rwkv-quant-backend auto \
  --rwkv-wkv-backend metal \
  --rwkv-chunk-size 2048 \
  --rwkv-prefill-eval-interval 2 \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Run CoreML runtime rows from an export manifest:

```bash
# Plan rows only; safe without CoreMLTools or an .mlpackage.
PYTHONPATH=. python bench/run_coreml_apple_baseline.py \
  --manifest exports/rwkv7-g1g-1.5b-coreml/coreml_export_manifest.json \
  --dry-run \
  --prompt-target-chars 1024,4096 \
  --decode-lengths 128,512 \
  --results bench/results_qwen35_apple_baseline.jsonl

# Live stateful runtime + correctness gates.
PYTHONPATH=. python bench/run_coreml_apple_baseline.py \
  --manifest exports/rwkv7-g1g-1.5b-coreml/coreml_export_manifest.json \
  --compute-units cpu-and-ne \
  --verify-chunked-prefill \
  --verify-hf-parity \
  --require-hf-greedy-match \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Plan and smoke a CoreML package export:

```bash
# Import-safe plan: no CoreMLTools required.
PYTHONPATH=. python scripts/export_rwkv7_coreml.py \
  /path/to/rwkv7-g1g-1.5b-hf \
  exports/rwkv7-g1g-1.5b-coreml \
  --dry-run \
  --export-kind stateful-multifunction \
  --chunks 4 \
  --prefill-seq-length 16 \
  --sample-seq-length 128 \
  --state-mode wkv-coreml \
  --quantization none \
  --results bench/results_qwen35_apple_baseline.jsonl

# Live correctness-first stateful export when CoreMLTools is installed.
PYTHONPATH=. python scripts/export_rwkv7_coreml.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  exports/rwkv7-g1d-0.1b-coreml \
  --export-kind stateful-multifunction \
  --prefill-seq-length 16 \
  --deployment-target iOS18 \
  --compute-units cpu-and-ne \
  --coreml-compute-precision auto \
  --quantization none \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Stateful TorchScript prefill is statically unrolled, so the exported chunk is
intentionally small (default `16`, maximum `128`). Longer prompts do not require
a larger package: the runtime streams them through repeated masked chunks.

The export row uses `axis=rwkv7_coreml_export`.  `status=plan` only records the
manifest/contract; `status=pass` means a `.mlpackage` was produced.  A CoreML
export row alone is **not** a Qwen3.5 performance win. Only live stateful runtime
rows with TTFT, prefill/decode tok/s, memory, and correctness fields enter the
`qwen35_apple_baseline` matrix.

Summarize an existing result file:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --summarize bench/results_qwen35_apple_baseline.jsonl
```

Score quality rows after collecting responses:

```bash
# Collect full response text/token ids first.
STORE_RESPONSES=1 \
QUALITY_RUBRIC=docs/hardware/qwen35_quality_rubric.example.json \
scripts/run_qwen35_apple_acceptance.sh

# Or score an existing JSONL file directly.
PYTHONPATH=. python bench/score_qwen35_quality.py \
  --results bench/results_qwen35_apple_baseline.jsonl \
  --rubric docs/hardware/qwen35_quality_rubric.example.json \
  --pair qwen3.5:0.8b-mlx=rwkv7-g1d-0.4b-hf \
  --append bench/results_qwen35_apple_baseline.jsonl
```

Quality rows use `axis=qwen35_apple_quality`; pairwise quality comparisons use
`axis=qwen35_apple_quality_comparison`.  Missing full `response_text` is
reported as `unknown`, so quality parity cannot be claimed from truncated
previews.

Compare RWKV rows against Qwen3.5 rows and emit explicit gate results:

```bash
PYTHONPATH=. python bench/compare_qwen35_apple_baseline.py   --results bench/results_qwen35_apple_baseline.jsonl   --pair qwen3.5:0.8b-mlx=rwkv7-g1d-0.4b-hf   --pair qwen3.5:2b-mlx=rwkv7-g1g-1.5b-hf   --min-decode-ratio 1.0   --require-prefill   --require-ttft   --max-ttft-ratio 1.1   --diagnostics   --append bench/results_qwen35_apple_baseline.jsonl
```

The comparison rows use `axis=qwen35_apple_baseline_comparison`; optional
`--diagnostics` rows use `axis=qwen35_apple_baseline_gap_diagnostic`; the
summary row uses `axis=qwen35_apple_baseline_comparison_summary`.  Missing
required metrics produce `status=unknown`, not `pass`, so a PR cannot claim a
Qwen3.5 win from an incomplete row.  Diagnostic rows translate missing/failing
gates into concrete actions such as `collect_qwen_baseline_rows`,
`collect_memory_telemetry`, `optimize_decode_kernel_or_batching`, or
`reduce_peak_memory_or_quantize_more`; `scripts/run_qwen35_apple_acceptance.sh`
enables these rows by default with `COMPARE_DIAGNOSTICS=1`.

## Initial acceptance matrix

The first M5/16GB live 0.8B-vs-0.4B matrix is now present. At 128/512 prompt
characters and 32 generated tokens, the retained conservative RWKV fp16 decode
rows reach about `0.82x/0.92x` Qwen, while prefill reaches only `0.090x/0.049x`. RWKV W4 lowers
its own peak memory from about `929MB` to `528MB`, but decode falls to about
`0.62x/0.60x` Qwen and prefill to `0.064x/0.030x`. Qwen `/api/ps` loaded memory
is about `1.09-1.11GB`, but peak memory is not yet captured. W4 does not
preserve fp16 tokens on every prompt, so neither the peak-memory nor quality
gate is complete. See the two
`bench/results_qwen35_apple_m5_20260710_*.jsonl` files.

| RWKV target | Qwen3.5 comparator | Runtime gate | Current status |
|---|---|---|---|
| RWKV-7 0.4B fp16/W4 MLX | `qwen3.5:0.8b-mlx` | lower memory and higher decode tok/s at prompt 1k/4k/8k, decode 128/512 | first short same-device rows landed; decode/prefill/TTFT gates fail, Qwen loaded memory is recorded, and Qwen peak memory is unknown |
| RWKV-7 1.5B W4/MLX | `qwen3.5:2b-mlx` | lower memory and higher or equal decode tok/s; TTFT no worse by >10% | needs same-device rows |
| RWKV-7 2.9B W4/MLX/CoreML | `qwen3.5:4b-mlx` | lower memory and higher decode tok/s | 0.1B stateful CoreML correctness passes; 2.9B quantized/ANE rows not landed |
| RWKV-7 larger / distilled mobile | `qwen3.5:9b-mlx` | mobile-useful memory envelope plus quality eval | requires model/quality work |

## CoreML / ANE follow-up

`rwkv-mobile` shows the right production direction for mobile Apple devices:

- separate `decode` and `prefill` CoreML functions
- chunked model export
- CoreML state / tensor state / WKV-CoreML state variants
- async prefill loading
- int8 / int4 / LUT quantization

The repository now has a live CoreML bridge. It records chunking, state mode,
quantization, deployment target, and compute precision, and exports deduplicated
stateful prefill/decode functions. On M5, the 0.1B fp32-compute short row passes
MLState transfer, alternate chunk split, and HF greedy-token parity. fp16
stateful compute remains opt-in because its first live row mismatched HF tokens.

The next repository lane should add:

1. Extend live correctness rows to prefill chunks 16/64 and long prompts/decode.
2. Add 0.4B/1.5B and CoreML W4/LUT/INT4 rows in the same schema.
3. Fix/selectively preserve recurrent precision in the fp16/ANE lane.
4. Record confirmed runtime placement rather than treating `CPU_AND_NE` as
   proof of ANE use.
5. Add iPhone/iPad rows once device access is available.

## Non-goals for the first baseline PR

- It does not claim final quality superiority over Qwen3.5.
- It does not claim the short 0.1B CoreML correctness row as production ANE performance.
- It does not mark W8/W4 as fp16-beating until JSONL evidence proves it.
- It does not replace the existing Apple MLX session and quant regression tests.
