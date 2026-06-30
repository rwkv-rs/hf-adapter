# RWKV-7 HF Adapter

First-stage Hugging Face adapter for official RWKV-7 `.pth` checkpoints.

This repository converts RWKV-7 weights to a Hugging Face-style directory and provides remote-code wrappers so the result can be loaded with:

- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
- `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
- `model.generate(..., use_cache=True)`
- PEFT LoRA smoke tests
- HF Trainer and TRL SFTTrainer one-step smoke tests

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
  test_result_tools.py
bench/
  bench_speed.py
  bench_decode_breakdown.py
  bench_batch_sweep.py
  bench_dynamic_batch.py
  bench_chunked_prefill.py
  bench_decode_micro.py
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

Native-JIT / native-graph backends for the HF fast-token path:

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
  --fast-token-backend native_jit

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

# Final acceptance target: expected to fail until decode reaches >=0.9x official.
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
  conversions.
- HF API contract smoke covers fixed-vocab `resize_token_embeddings` handling,
  `prepare_inputs_for_generation`, beam cache reorder, and
  `gradient_checkpointing_enable`.
- PEFT LoRA forward/loss/backward works.
- HF Trainer and TRL SFTTrainer one-step LoRA smoke runs work.
- Fast recurrent cache matches the default FLA cache exactly on prefill and recurrent decode.
- `rwkv7_prefill_chunks` provides an inference-only chunked prefill helper that
  preserves HF `forward` as the source of truth while carrying
  `RWKV7StateCache` across prompt chunks.
- Inference-only `rwkv7_forward_token` API supports one-token decode for batched serving experiments without changing HF `forward`/`generate`; `rwkv7_forward_one` remains as the bsz=1 compatibility entrypoint.
- Batched recurrent cache smoke coverage exists for repeated prompts across bsz=1/2/4; benchmark sweep records total/per-sequence throughput for bsz=1/2/4/8 and includes the fast token API when available.
- Dynamic-batch cache reorder coverage exists for heterogeneous prompts; benchmark simulation records reorder/drop counts and total decoded tokens/s.
- Chunked prefill coverage compares full vs chunked logits/cache and records
  throughput/memory tradeoffs for multiple chunk sizes.
- Decode microbench coverage records stable timing for HF recurrent forward, the fast token API, `lm_head`, argmax, embedding, and empty-loop overhead.
- Decode component benchmark coverage times the fast-token layer path by projection, recurrent, norm/output, FFN, and layer totals.
- Projection/LoRA benchmark coverage times the largest component and compares simple PyTorch bmm fusion candidates.
- Benchmark analysis coverage reports speed/memory ratios and next optimization focus from `bench/results.jsonl`.
- Benchmark check coverage provides passing regression and target gates for the current native-JIT HF fast-token rows; native-graph rows are reported as an optional reduced-launch speed path.
- Latest V100 fast-token results: FLA bsz=1 decode `59.2 tok/s` vs official `92.1 tok/s`; native-JIT bsz=1 decode reaches `92.1 tok/s` vs official `92.1 tok/s`; HF `native_graph` bsz=1 reaches `255.5 tok/s` in speed_mem. Batched native-graph reaches `253.9` / `434.3` / `852.6` / `1539.1` aggregate tok/s for bsz=1/2/4/8. Dynamic-batch simulation with native-graph reorder/drop reaches `524.7` total tok/s. Chunked prefill bsz=2 prompt=512 preserves logits/cache within fp16 tolerance and reduces peak VRAM to about `0.60x` / `0.62x` / `0.63x` of full prefill for chunk sizes 64/128/256, trading throughput to `0.13x` / `0.25x` / `0.50x`. Component timing identifies `attn_linears_lora` as the largest group at about `9.87 ms/token`; naive PyTorch bmm projection/LoRA candidates are not enough, so the next implementation needs custom fusion/reduced launch count.
- Bitsandbytes quantization smoke now loads and generates for both 8-bit and
  4-bit on V100. Short benchmark rows show model footprint dropping from
  `364.4 MB` fp16 to `278.4 MB` 8-bit and `235.3 MB` 4-bit; current generic bnb
  decode is slower (`40.4` -> `9.5` / `27.1 tok/s`), so production quantization
  still needs a faster custom path.
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
- Generic bnb 8-bit/4-bit loading reduces model footprint but is slower than
  fp16 on the current V100 path; next performance work is CUDA graph / lower
  launch count plus a faster quantized serving path for higher bsz and larger
  models.
