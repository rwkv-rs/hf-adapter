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
tests/
  smoke_hf_generate.py
  test_official_alignment.py
  test_reload_roundtrip.py
  test_fast_cache.py
  test_fast_decode_api.py
  test_batch_cache.py
  test_dynamic_batch_cache.py
  test_peft_lora.py
  test_hf_training_smoke.py
bench/
  bench_speed.py
  bench_decode_breakdown.py
  bench_batch_sweep.py
  bench_dynamic_batch.py
  bench_decode_micro.py
  bench_decode_components.py
  bench_projection_lora.py
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
- PEFT LoRA forward/loss/backward works.
- HF Trainer and TRL SFTTrainer one-step LoRA smoke runs work.
- Fast recurrent cache matches the default FLA cache exactly on prefill and recurrent decode.
- Inference-only `rwkv7_forward_token` API supports one-token decode for batched serving experiments without changing HF `forward`/`generate`; `rwkv7_forward_one` remains as the bsz=1 compatibility entrypoint.
- Batched recurrent cache smoke coverage exists for repeated prompts across bsz=1/2/4; benchmark sweep records total/per-sequence throughput for bsz=1/2/4/8 and includes the fast token API when available.
- Dynamic-batch cache reorder coverage exists for heterogeneous prompts; benchmark simulation records reorder/drop counts and total decoded tokens/s.
- Decode microbench coverage records stable timing for HF recurrent forward, the fast token API, `lm_head`, argmax, embedding, and empty-loop overhead.
- Decode component benchmark coverage times the fast-token layer path by projection, recurrent, norm/output, FFN, and layer totals.
- Projection/LoRA benchmark coverage times the largest component and compares simple PyTorch bmm fusion candidates.
- Benchmark analysis coverage reports speed/memory ratios and next optimization focus from `bench/results.jsonl`.
- Benchmark check coverage provides a passing regression gate and a failing final-target gate until decode reaches >=0.9x official.
- Latest V100 fast-token results: bsz=1 decode `58.0 tok/s` vs official `90.0 tok/s`; batch sweep fast-token per-seq decode is about `55 tok/s` for bsz=1/2/4/8; dynamic-batch simulation improves from `205.2` to `345.7` total tok/s; component timing identifies `attn_linears_lora` as the largest group at about `9.87 ms/token`; naive PyTorch bmm projection/LoRA candidates are not enough, so the next implementation needs custom fusion/reduced launch count.
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
- V100 native-norm + fast-cache HF decode is about 41 tok/s; `rwkv7_forward_token` improves this to about 58 tok/s, but official is still about 90 tok/s for 0.1B fp16.
