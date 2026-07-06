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

It emits rows with `axis=qwen35_apple_baseline` and can run:

1. Qwen3.5 through a local Ollama server using the streaming `/api/generate`
   endpoint.
2. RWKV-7 through this repository's optional MLX recurrent backend.
3. CoreML/ANE rows in the same schema once the CoreML runner lands.

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
  --results bench/results_qwen35_apple_baseline.jsonl
```

Summarize an existing result file:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --summarize bench/results_qwen35_apple_baseline.jsonl
```

Compare RWKV rows against Qwen3.5 rows and emit explicit gate results:

```bash
PYTHONPATH=. python bench/compare_qwen35_apple_baseline.py   --results bench/results_qwen35_apple_baseline.jsonl   --pair qwen3.5:0.8b-mlx=rwkv7-g1d-0.4b-hf   --pair qwen3.5:2b-mlx=rwkv7-g1g-1.5b-hf   --min-decode-ratio 1.0   --require-prefill   --require-ttft   --max-ttft-ratio 1.1   --append bench/results_qwen35_apple_baseline.jsonl
```

The comparison rows use `axis=qwen35_apple_baseline_comparison`; the summary row
uses `axis=qwen35_apple_baseline_comparison_summary`.  Missing required metrics
produce `status=unknown`, not `pass`, so a PR cannot claim a Qwen3.5 win from an
incomplete row.

## Initial acceptance matrix

| RWKV target | Qwen3.5 comparator | Runtime gate | Current status |
|---|---|---|---|
| RWKV-7 0.4B W4/MLX | `qwen3.5:0.8b-mlx` | lower memory and higher decode tok/s at prompt 1k/4k/8k, decode 128/512 | needs same-device rows |
| RWKV-7 1.5B W4/MLX | `qwen3.5:2b-mlx` | lower memory and higher or equal decode tok/s; TTFT no worse by >10% | needs same-device rows |
| RWKV-7 2.9B W4/MLX/CoreML | `qwen3.5:4b-mlx` | lower memory and higher decode tok/s | CoreML/ANE path not landed |
| RWKV-7 larger / distilled mobile | `qwen3.5:9b-mlx` | mobile-useful memory envelope plus quality eval | requires model/quality work |

## CoreML / ANE follow-up

`rwkv-mobile` shows the right production direction for mobile Apple devices:

- separate `decode` and `prefill` CoreML functions
- chunked model export
- CoreML state / tensor state / WKV-CoreML state variants
- async prefill loading
- int8 / int4 / LUT quantization

The next repository lane should add:

1. HF RWKV-7 -> Torch traced decode/prefill -> CoreML multifunction export.
2. CoreML correctness checks against HF/MLX logits and state.
3. CoreML W4/LUT/INT4 export rows in the same `qwen35_apple_baseline` schema.
4. iPhone/iPad rows once device access is available.

## Non-goals for the first baseline PR

- It does not claim final quality superiority over Qwen3.5.
- It does not implement CoreML export yet.
- It does not mark W8/W4 as fp16-beating until JSONL evidence proves it.
- It does not replace the existing Apple MLX session and quant regression tests.
