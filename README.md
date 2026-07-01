# RWKV-7 HF Adapter

First-stage Hugging Face adapter for official RWKV-7 `.pth` checkpoints.

This repository converts RWKV-7 weights to a Hugging Face-style directory and provides remote-code wrappers so the result can be loaded with:

- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
- `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
- `model.generate(..., use_cache=True)`
- PEFT LoRA smoke tests
- HF Trainer, TRL SFTTrainer, DPOTrainer, and GRPOTrainer one-step smoke tests
- HF `device_map` multi-GPU generate smoke for the pipeline-parallel direction

The current backend uses the FLA (`flash-linear-attention`) RWKV-7 implementation. The next milestone is a native Transformers implementation without the FLA runtime dependency.

## Layout

```text
rwkv7_hf/
  configuration_rwkv7.py
  modeling_rwkv7.py
  tokenization_rwkv7.py
scripts/
  convert_rwkv7_to_hf.py
  batch_convert_rwkv7_to_hf.py
tests/
  smoke_hf_generate.py
  test_official_alignment.py
  test_reload_roundtrip.py
  test_fast_cache.py
  test_fast_decode_api.py
  test_chunked_prefill.py
  test_batch_cache.py
  test_dynamic_batch_cache.py
  test_peft_lora.py
  test_hf_training_smoke.py
  test_device_map_generate.py
  test_result_tools.py
bench/
  bench_speed.py
  bench_decode_breakdown.py
  bench_batch_sweep.py
  bench_dynamic_batch.py
  bench_chunked_prefill.py
  bench_decode_micro.py
  bench_forward_fast_path.py
  bench_generate_fast_path.py
  bench_fast_token_warmup.py
  bench_native_graph_overhead.py
  bench_decode_components.py
  bench_projection_lora.py
  compare_fast_token_layouts.py
  analyze_results.py
  check_results.py
  profile_decode.py
NEXT_STEPS.md
BENCHMARK.md
```

## Convert an official checkpoint

```bash
export PYTHONPATH=/path/to/flash-linear-attention:/path/to/rwkv7-hf-adapter:$PYTHONPATH

python scripts/convert_rwkv7_to_hf.py \
  --input /path/to/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --output /path/to/rwkv7-g1d-0.1b-hf \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --attn-mode chunk \
  --no-fuse-norm
```

For multiple downloaded checkpoints, use the batch wrapper. It writes a
reproducible manifest with source path, output path, size, SHA256, conversion
options, status, and the exact command for each model:

```bash
python scripts/batch_convert_rwkv7_to_hf.py \
  --input-dir /path/to/rwkv7-pth-files \
  --output-root /path/to/hf-models \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --attn-mode fused_recurrent \
  --no-fuse-norm \
  --manifest /path/to/hf-models/manifest.json

# Enumerate and hash without loading torch/FLA or writing model directories.
python scripts/batch_convert_rwkv7_to_hf.py \
  --input-dir /path/to/rwkv7-pth-files \
  --output-root /path/to/hf-models \
  --dry-run
```

## Inference smoke test

```bash
export PYTHONPATH=/path/to/flash-linear-attention:$PYTHONPATH

python tests/smoke_hf_generate.py \
  --model /path/to/rwkv7-g1d-0.1b-hf
```

Minimal usage:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/path/to/rwkv7-g1d-0.1b-hf"

tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda",
).eval()

x = tok("User: Hello!\n\nAssistant:", return_tensors="pt").to("cuda")
y = model.generate(**x, max_new_tokens=32, do_sample=False, use_cache=True)
print(tok.decode(y[0], skip_special_tokens=True))
```

## PEFT LoRA smoke test

On the current V100 test box, FLA backward is more reliable with Dynamo disabled:

```bash
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=/path/to/flash-linear-attention:$PYTHONPATH

python tests/test_peft_lora.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --attn-mode fused_recurrent
```

HF Trainer / TRL SFTTrainer one-step smoke:

```bash
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=/path/to/flash-linear-attention:$PYTHONPATH

python tests/test_hf_training_smoke.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --attn-mode fused_recurrent \
  --backend both
```

TRL DPO / GRPO LoRA one-step smoke:

```bash
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=/path/to/flash-linear-attention:$PYTHONPATH

python tests/test_hf_rl_training_smoke.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --attn-mode fused_recurrent \
  --backend dpo

python tests/test_hf_rl_training_smoke.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --attn-mode fused_recurrent \
  --backend grpo \
  --grpo-max-completion-length 2
```

DeepSpeed ZeRO preset validation:

```bash
python tests/test_deepspeed_configs.py
```

HF multi-GPU `device_map` generate smoke, for the pipeline-parallel direction:

```bash
CUDA_VISIBLE_DEVICES=0,1 python tests/test_device_map_generate.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --max-new-tokens 4 \
  --compare-single-device
```

Fast recurrent cache equivalence test:

```bash
python tests/test_fast_cache.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --fuse-norm false
```

Inference-only fast decode API equivalence test:

```bash
python tests/test_fast_decode_api.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --fuse-norm false \
  --batch-sizes 1 2 4
```

Batched recurrent cache smoke test:

```bash
python tests/test_batch_cache.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --fuse-norm false \
  --batch-sizes 1 2 4
```

Dynamic-batch cache reorder smoke test:

```bash
python tests/test_dynamic_batch_cache.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --fuse-norm false \
  --batch-size 3
```


## Correctness and benchmark tests

Official alignment including greedy 64-token equality:

```bash
python tests/test_official_alignment.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --pth /path/to/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --official-strategy 'cpu fp32' \
  --greedy-window 64 \
  --fuse-norm false
```

Save/reload roundtrip:

```bash
python tests/test_reload_roundtrip.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --dtype fp16
```

Serving-style speed/memory benchmark:

```bash
python bench/bench_speed.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --pth /path/to/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --backend both \
  --dtype fp16 \
  --hf-logits-to-keep 1 \
  --fuse-norm false \
  --fast-cache true
```


Full V100 fast-decode validation bundle:

```bash
./bench/run_v100_fast_decode_validation.sh
python bench/summarize_results.py --device V100 --last 12
```


Fast-token layout A/B benchmark, for opt-in 2D hot-path experiments after the baseline is stable:

```bash
./bench/run_v100_fast_token_layout_ab.sh
# Resume only the candidate side after an interrupted run:
LAYOUTS=2d SPEED_BACKEND=hf ./bench/run_v100_fast_token_layout_ab.sh
python bench/compare_fast_token_layouts.py --results bench/results.jsonl --device V100 --dtype fp16 --require-candidate --min-speedup 1.0
```

Serving-style speed/memory benchmark using the one-token fast decode API:

```bash
python bench/bench_speed.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --pth /path/to/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --backend both \
  --dtype fp16 \
  --hf-logits-to-keep 1 \
  --fuse-norm false \
  --fast-cache true \
  --hf-decode-api rwkv7_forward_token
```

Native-JIT / native-graph backends for the HF fast-token path. `auto` is the
serving default for `rwkv7_forward_token`: it picks `native_graph` when CUDA
graph replay is available for the active batch size, falls back to `native_jit`,
then to the FLA tensor path. Benchmark rows record both the requested backend
and `fast_token_backend_effective`. Normal HF one-token inference calls with
`past_key_values` also use this path by default, so `model.generate(...,
use_cache=True)` benefits without changing caller code; set
`RWKV7_FAST_FORWARD=0` to force the reference HF recurrent forward baseline.

```bash
python bench/bench_speed.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --pth /path/to/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --backend both \
  --dtype fp16 \
  --hf-logits-to-keep 1 \
  --fuse-norm false \
  --fast-cache true \
  --hf-decode-api rwkv7_forward_token \
  --fast-token-backend auto

python bench/bench_speed.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --backend hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --hf-logits-to-keep 1 \
  --fuse-norm false \
  --fast-cache true \
  --hf-decode-api rwkv7_forward_token \
  --fast-token-backend native_graph
```

Batch-size sweep for serving-style prefill and recurrent decode:

```bash
python bench/bench_batch_sweep.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-decode-api auto \
  --batch-sizes 1 2 4 8
```

Dynamic-batch decode benchmark with cache reorder/drop simulation:

```bash
python bench/bench_dynamic_batch.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --decode-apis forward rwkv7_forward_token \
  --batch-size 8 \
  --min-batch-size 2
```

Decode bottleneck breakdown:

```bash
python bench/bench_decode_breakdown.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --pth /path/to/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --dtype fp16 \
  --attn-modes chunk fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-decode-api auto
```

Decode microbench for stable per-component timings:

```bash
python bench/bench_decode_micro.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-decode-api auto
```

Production-facing HF forward fast-path benchmark:

```bash
python bench/bench_forward_fast_path.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-token-backend auto
```

Production-facing HF `generate()` fast-path benchmark:

```bash
python bench/bench_generate_fast_path.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-token-backend auto \
  --batch-size 2 \
  --max-new-tokens 16
```

Native JIT / CUDA graph decode prototype benchmark:

```bash
python bench/bench_native_decode.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --prompt-tokens 32 \
  --decode-tokens 64 \
  --greedy-check-tokens 16
```

Fast-token component timing benchmark:

```bash
python bench/bench_decode_components.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fixed-token
```

Attention projection/LoRA microbenchmark:

```bash
python bench/bench_projection_lora.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --layers 0 1 11
```

Larger converted-model smoke benchmark:

```bash
python bench/bench_larger_model_smoke.py \
  --hf-dir /path/to/rwkv7-g1d-0.4b-hf \
  --model-size-label 0.4b \
  --checkpoint-path /path/to/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 4

python bench/bench_larger_model_smoke.py \
  --hf-dir /path/to/rwkv7-g1g-1.5b-hf \
  --model-size-label 1.5b \
  --checkpoint-path /path/to/rwkv7-g1g-1.5b-20260526-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 2

python bench/bench_larger_model_smoke.py \
  --hf-dir /path/to/rwkv7-g1g-2.9b-hf \
  --model-size-label 2.9b \
  --checkpoint-path /path/to/rwkv7-g1g-2.9b-20260526-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 2

python bench/bench_larger_model_smoke.py \
  --hf-dir /path/to/rwkv7-g1g-7.2b-hf \
  --model-size-label 7.2b \
  --checkpoint-path /path/to/rwkv7-g1g-7.2b-20260523-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 2

python bench/bench_larger_model_smoke.py \
  --hf-dir /path/to/rwkv7-g1g-13.3b-hf \
  --model-size-label 13.3b \
  --checkpoint-path /path/to/rwkv7-g1g-13.3b-20260523-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend native_jit \
  --max-new-tokens 2
```

Benchmark gap report against current targets:

```bash
python bench/analyze_results.py \
  --results bench/results.jsonl \
  --device V100 \
  --dtype fp16
```

Benchmark regression/target gate:

```bash
# Current regression floor: should pass on the committed V100 rows.
python bench/check_results.py --results bench/results.jsonl --device V100 --dtype fp16

# Current V100 target gate: should pass on the committed native-JIT/native-graph rows.
python bench/check_results.py --results bench/results.jsonl --device V100 --dtype fp16 --target
```

Profiler for one-token decode hotspots:

```bash
python bench/profile_decode.py \
  --backend hf \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --attn-mode chunk \
  --fuse-norm false \
  --fixed-token \
  --fast-cache true \
  --hf-decode-api forward

# Profile the fast one-token decode API instead:
python bench/profile_decode.py \
  --backend hf \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fixed-token \
  --fast-cache true \
  --hf-decode-api rwkv7_forward_token
```

## Current validation

For `rwkv7-g1d-0.1b-20260129-ctx8192`:

- HF `generate()` works.
- Converter infers layer count, hidden size, head dimension, per-layer value
  dimensions, and low-rank dimensions from checkpoint tensor shapes instead of
  hard-coding the 0.1B layout; offline tests cover non-64 head dims and shape
  validation errors.
- Batch conversion wrapper writes a SHA256 manifest and supports dry-run
  enumeration for downloaded 0.4B+ checkpoints before launching heavyweight
  conversions; the 0.4B, 1.5B, 2.9B, 7.2B, and 13.3B checkpoints have now been converted and
  smoke-tested from generated HF directories on V100.
- HF API contract smoke covers fixed-vocab `resize_token_embeddings` handling,
  `prepare_inputs_for_generation`, beam cache reorder, and
  `gradient_checkpointing_enable`.
- PEFT LoRA forward/loss/backward works.
- HF Trainer and TRL SFTTrainer one-step LoRA smoke runs work.
- TRL DPOTrainer and GRPOTrainer one-step LoRA smoke scripts are available for
  preference/RL compatibility checks, and `configs/deepspeed/zero2.json` plus
  `configs/deepspeed/zero3.json` provide HF Trainer-compatible ZeRO presets.
- Fast recurrent cache matches the default FLA cache exactly on prefill and recurrent decode.
- `RWKV7StateCache` exposes serving-friendly `select_batch` / `batch_select`,
  `clone`, `detach`, `to`, and `get_batch_size` helpers so dynamic batching can
  reorder/drop active rows and temporarily CPU-offload inactive states without
  relying on beam-search-only cache hooks. `rwkv7_cache_metrics()` reports
  update/select/reorder/offload counters plus current layer, token, and batch
  sizes for HF serving telemetry.
- `rwkv7_prefill_chunks` provides an inference-only chunked prefill helper that
  preserves HF `forward` as the source of truth while carrying
  `RWKV7StateCache` across prompt chunks.
- Inference-only `rwkv7_forward_token` API supports one-token decode for
  batched serving experiments; normal eval/no-grad HF `forward` and
  `generate()` automatically route one-token cached decode through it unless
  `RWKV7_FAST_FORWARD=0` is set. `rwkv7_forward_one` remains as the bsz=1
  compatibility entrypoint.
- Initial HF-compatible `rwkv7_speculative_generate()` supports greedy bsz=1
  speculative decoding with a RWKV/HF draft model. It verifies draft spans with
  block HF forwards, reports accepted/proposed/corrected tokens and acceptance
  rate, and falls back to cache resync on mismatch.
- Batched recurrent cache smoke coverage exists for repeated prompts across bsz=1/2/4; benchmark sweep records total/per-sequence throughput for bsz=1/2/4/8 and includes the fast token API when available.
- Dynamic-batch cache reorder coverage exists for heterogeneous prompts; benchmark simulation records reorder/drop counts and total decoded tokens/s.
- Chunked prefill coverage compares full vs chunked logits/cache and records
  throughput/memory tradeoffs for multiple chunk sizes.
- Decode microbench coverage records stable timing for reference HF recurrent
  forward, ordinary HF forward with fast-forward enabled, the direct fast token
  API, `lm_head`, argmax, embedding, and empty-loop overhead.
- `bench_forward_fast_path.py` records the production-facing ordinary HF
  cached `forward()` path against both `RWKV7_FAST_FORWARD=0` reference forward
  and direct `rwkv7_forward_token`, and `check_results.py` gates correctness,
  speedup, and direct-fast parity.
- `bench_generate_fast_path.py` records the production-facing
  `model.generate(..., use_cache=True)` path with `RWKV7_FAST_FORWARD=0/1`,
  gates greedy token equality, backend selection, bsz>=2 coverage, and
  end-to-end generation speedup. V100 prompt=8/new=16 bsz=2 runs show
  reference generate at `75.3 tok/s` aggregate and fast-forward generate at
  `303.5 tok/s` aggregate (`4.03x`) with all `32/32` generated tokens
  identical and effective backend `native_graph`.
- `rwkv7_warmup_fast_token()` pre-initializes native fast-token resources for
  requested serving batch sizes, and
  `rwkv7_native_graph_cache_batch_sizes()` reports the native-graph LRU contents.
  `rwkv7_native_graph_cache_stats()` reports graph-runner requests, hits,
  misses, evictions, retained batch sizes, and hit rate; the counters can be
  reset with `rwkv7_reset_native_graph_cache_stats()`.
  `bench_fast_token_warmup.py` records `axis=fast_token_warmup`; the default gate
  requires bsz=1/2/4/8 to resolve to `native_graph` and be present in the graph
  cache.
- Native-graph replay overhead coverage records runner-vs-public-API equality,
  cache-copy/token-copy/graph-replay/cache-bind timing, public API tok/s, and
  graph-runner cache requests/hits/misses/hit-rate, so wrapper overhead and
  state-cache reuse are gated separately from model math.
- Decode component benchmark coverage times the fast-token layer path by projection, recurrent, norm/output, FFN, and layer totals.
- Projection/LoRA benchmark coverage times the largest component and compares simple PyTorch bmm fusion candidates.
- Benchmark analysis coverage reports speed/memory ratios and next optimization focus from `bench/results.jsonl`.
- Benchmark check coverage provides passing regression and target gates for the current native-JIT HF fast-token rows; native-graph rows are reported as an optional reduced-launch speed path.
- `RWKV7_FAST_TOKEN_BACKEND=auto` now chooses the fastest available dense
  fast-token backend per active batch (`native_graph` -> `native_jit` -> FLA)
  and exposes the chosen value through `rwkv7_last_fast_token_backend()`.
  Generic bitsandbytes 8-bit/4-bit loads intentionally stay on the FLA
  fast-token path until a dedicated quantized native projection path is added.
- `RWKV7_FAST_FORWARD=1` (default) lets standard HF cached one-token
  `forward()` / `generate()` use the same fast-token path in eval/no-grad mode;
  tests and benchmarks can set it to `0` when they need the slower reference
  recurrent forward baseline. A short V100 microbench with prompt=64/steps=8
  records reference HF forward at about `40 tok/s`, ordinary HF forward with
  fast-forward at about `251 tok/s`, and direct `rwkv7_forward_token` at about
  `252 tok/s`, all resolving to `native_graph`. For HF `device_map` placements
  that span multiple CUDA devices, the adapter skips this single-device
  fast-token shortcut so Accelerate's normal hooks can move tensors across the
  split.
  Quantized loads use the same fast-forward hook by default through the FLA
  fallback; set `RWKV7_FAST_FORWARD_QUANT=0` to force the slower reference
  path for debugging.
- Latest V100 fast-token results: FLA bsz=1 decode `59.2 tok/s` vs official `92.1 tok/s`; native-JIT bsz=1 decode reaches `92.1 tok/s` vs official `92.1 tok/s`; HF `native_graph` bsz=1 reaches `255.5 tok/s` in speed_mem. Batched native-graph reaches `253.9` / `434.3` / `852.6` / `1539.1` aggregate tok/s for bsz=1/2/4/8, and warmup pre-captures those graph runners in `1.389s` with cache sizes `[1,2,4,8]`. Native-graph replay overhead rows for bsz=1/2/4/8 show public API `255.1` / `449.8` / `857.2` / `1548.1` aggregate tok/s, runner/API diff `0.0`, cache-copy share `0.052` / `0.032` / `0.030` / `0.028`, and graph-runner cache hit rate `0.9737` after skipping graph-buffer self-copy. Dynamic-batch simulation with native-graph reorder/drop through `select_batch` reaches `1209.3` total tok/s. The converted 0.4B, 1.5B, 2.9B, 7.2B, and 13.3B HF directories load and generate on V100: 0.4B has hidden=1024/layers=24, checkpoint SHA256 `947cb9b8013224e06b112b72204256bec65096cc935a7767ce63d8e3ddef83bb`, peak VRAM `1124.5 MB`; 1.5B has hidden=2048/layers=24, checkpoint SHA256 `441f70b096ad62442b5c33128bfe717c5d8529915c45a9709d4482016e8a0482`, peak VRAM `3178.6 MB`; 2.9B has hidden=2560/layers=32, checkpoint SHA256 `3d118ed77fe94e63e6fc0a6afd5a4fac49fe70da4e3d9d91b628951bb55dd798`, peak VRAM `5888.0 MB`; 7.2B has hidden=4096/layers=32, checkpoint SHA256 `425fc9bda2d12d4ce3b6bfe5c3b3f355be8b14d85960cf40fcca58a19d632630`, peak VRAM `13997.8 MB`; 13.3B has hidden=4096/layers=61, checkpoint SHA256 `0aa686d3ca4bb486e83e3071f4798a210f960e1fc1f5042e6cb418cc463814d6`, peak VRAM `25575.6 MB`, and uses `native_jit` for the V100 smoke because native-graph capture can reserve too much extra memory on 32GB cards. Chunked prefill bsz=2 prompt=512 preserves logits/cache within fp16 tolerance and reduces peak VRAM to about `0.60x` / `0.62x` / `0.63x` of full prefill for chunk sizes 64/128/256, trading throughput to `0.13x` / `0.25x` / `0.50x`. Component timing identifies `attn_linears_lora` as the largest group at about `9.87 ms/token`; naive PyTorch bmm projection/LoRA candidates are not enough, so the next implementation needs custom fusion/reduced launch count.
- HF `device_map` smoke on 2 x V100 manually splits 12 layers at layer 6,
  keeps `RWKV7_FAST_FORWARD=1`, skips the single-device fast-token backend,
  and matches the single-device greedy tail `[36786, 34, 308, 459]`.
- Bitsandbytes quantization smoke now loads and generates for both 8-bit and
  4-bit on V100, and cached decode can use the HF fast-forward hook through
  the FLA fallback. Short benchmark rows show model footprint dropping from
  `364.4 MB` fp16 to `278.4 MB` 8-bit and `235.3 MB` 4-bit; generic bnb
  fast-forward improves decode from `7.9 -> 8.4 tok/s` for 8-bit and
  `22.5 -> 27.1 tok/s` for 4-bit, but production quantization still needs a
  fused/native quantized projection path to beat fp16 fast decode.
- Native JIT / CUDA graph prototype: V100 fp16 native logits match HF logits (`cosine≈1.00000024`, max_abs `0.03125`), graph-vs-JIT greedy decode is `16/16` identical, native JIT reaches `103.52 tok/s`, and native CUDA graph reaches `254.33 tok/s`. The same reduced-launch idea is now available through HF `rwkv7_forward_token` via `RWKV7_FAST_TOKEN_BACKEND=native_graph` for fixed bsz and dynamic active-batch sizes; captured runners are retained in a per-model LRU controlled by `RWKV7_NATIVE_GRAPH_CACHE_SIZE` and can be released with `rwkv7_clear_native_graph_cache()`.
- Save/reload roundtrip works with exact logit equality.
- Official `rwkv` alignment includes prompt logits and 64-token greedy equality.
- Official `rwkv` logits comparison on smoke prompts:
  - top-5 token IDs match
  - cosine similarity ≈ `0.999998` on V100 fp16
  - fp16 max absolute difference ≈ `0.072` on V100 with native norm; fp32 reference ≈ `0.030`

## Known limitations

- This is a wrapper-based first stage, not yet a native upstream Transformers implementation.
- The backend currently requires FLA.
- The remote config uses a unique `rwkv7_hf_adapter` model type so `AutoModelForCausalLM` reliably loads this adapter instead of a locally registered FLA `rwkv7` class.
- V100 serving-style memory is now near parity with official for 0.1B when using `logits_to_keep=1`.
- V100 native-norm + fast-cache HF decode is about 41 tok/s; FLA `rwkv7_forward_token` improves this to about 59 tok/s; native-JIT `rwkv7_forward_token` reaches official parity for bsz=1 and supports batched/dynamic serving; native-graph `rwkv7_forward_token` reaches about 255 tok/s for bsz=1 and 1539 aggregate tok/s for bsz=8 with extra captured graph buffers.
- Generic bnb 8-bit/4-bit loading reduces model footprint and now benefits
  from HF fast-forward through the FLA fallback, but it is still slower than
  fp16 native-graph decode on the current V100 path; next performance work is
  a fused/native quantized serving path for higher bsz and larger models.
