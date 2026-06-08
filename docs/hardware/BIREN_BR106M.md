# Biren BR106M / BIRENSUPA

The canonical native/no-FLA Hugging Face model supports the Biren BR106M
through the private-use PyTorch `supa` device registered by `torch_br`. The
validated BR10x runtime supports BF16 and FP32 matrix multiplication, but not
the RWKV-7 FP16 GEMM path. The adapter therefore selects BF16 model math,
retains FP32 recurrent state, rejects FP16 before dispatch, and uses eager
execution with a decomposed GroupNorm implementation.

This integration ports the HF patch and acceptance contract from
[`yyqdbngt/rwkv7-biren-br106m` at `47322bf`](https://github.com/yyqdbngt/rwkv7-biren-br106m/tree/47322bfaffc2e662fa989863c3fda4d74f02fc32).
The standalone repository remains the source for raw environment/HF evidence
and the independent SGLang/vLLM patch sets. This HF repository does not import
either serving engine.

## Current boundary

| Item | Current boundary |
|---|---|
| Device | one exact Biren106M, 32,512 MiB |
| Validated stack | BIRENSUPA SDK 1.11.0.0.rc2, driver/BR-SMI 1.11.0, SUPA 1.11, PyTorch 2.9.0+cu128, torch_br 1.10.0.20900+br1xx |
| Backend | native eager/no-FLA; JIT, graph, `torch.compile`, CUDA and Triton routes fail closed |
| Dtypes | BF16 model weights/activations, FP32 recurrent state; FP16 is rejected |
| Models | standalone exact-card auto-load/forward/cached-generate evidence for 0.1B, 0.4B, 1.5B, 2.9B, 7.2B and 13.3B |
| HF ecosystem | 0.1B cache/chunked-prefill/dynamic-batch/save-reload, PEFT LoRA and Trainer evidence |
| Quantization | no production W8/W4 route promoted |
| Performance | recorded timings are compatibility telemetry; this is **not a performance** closure claim |

The standalone patches were based on HF adapter commit `22237b629` and include
the GroupNorm fallback plus a low-memory conversion fix for mmap
partial-storage views. Both changes are now integrated into the canonical
model. A full **current-main** BR106M rerun remains required before the evidence
provenance can be promoted from “ported” to “current-main rerun”.

## Install

Use the vendor container or source its environment before Python starts:

```bash
source /usr/local/birensupa/sdk/latest/scripts/brsw_set_env.sh
python -m pip install "rwkv7-hf[biren]==0.8.1"
python -c 'from rwkv7_hf import biren_available; print(biren_available())'
```

The last command must print `True`. The BIRENSUPA and `torch_br` wheels are
vendor/runtime-specific and are intentionally not declared as universal PyPI
dependencies. Use `pip install -e '.[biren]'` only inside a cloned source tree
for repository tests or conversion.

`enable_biren()` checks the exact product and validated software row. An
unknown stack fails closed. `RWKV7_ALLOW_UNVALIDATED_BIREN=1` permits an
explicitly reported experimental run but does not promote that stack.

## Convert a checkpoint

BR106M checkpoints should be converted to BF16 without fused normalization:

```bash
git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
python -m pip install -e '.[biren]'
```

```bash
rwkv7-hf convert \
  --input /path/to/rwkv7-model.pth \
  --output /path/to/rwkv7-model-hf \
  --precision bf16 --adapter-layout thin --no-fuse-norm
```

Large checkpoints can use the converter's low-memory mode. The converter now
materializes mmap tensors that are views into a larger storage before
safetensors export, while ordinary tensors remain zero-copy.

## Transformers usage

```python
import torch
import torch_br  # registers the supa private-use device
from transformers import AutoModelForCausalLM, AutoTokenizer
from rwkv7_hf import enable_biren

info = enable_biren("supa:0")
model_dir = "/path/to/converted-rwkv7-hf"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).eval().to(info.device)
inputs = tokenizer("User: Hello!\n\nAssistant:", return_tensors="pt").to(info.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=8, use_cache=True)
print(tokenizer.decode(output[0]))
```

The user-facing example selects BF16 automatically:

```bash
python examples/generate.py \
  --model /path/to/converted-rwkv7-hf \
  --prompt 'User: Hello! Assistant:' \
  --device biren --backend native --dtype auto
```

For HF Trainer/Accelerate on one card:

```bash
export ACCELERATE_TORCH_DEVICE=supa
export RWKV7_NATIVE_MODEL_BACKEND=eager
```

Keep `TrainingArguments(fp16=False, bf16=False)` with the validated Accelerate
stack; the model itself remains BF16. The recorded fused AdamW path falls back
to CPU in the vendor runtime, so current training evidence is functional rather
than an optimized-training claim.

## Current-main smoke

Tiny BF16 forward, FP32 cache state, chunked prefill, cache selection,
generation, backward, and FP16 fail-closed checks:

```bash
PYTHON_BIN=python RESULTS=bench/local_biren/results.json \
  bash scripts/run_biren_br106m_smoke.sh
```

Add one converted real checkpoint, optionally with a CPU oracle and
save/reload:

```bash
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
CPU_ORACLE=1 SAVE_RELOAD=1 \
PYTHON_BIN=python RESULTS=bench/local_biren/results-0.1b.json \
  bash scripts/run_biren_br106m_smoke.sh
```

Run larger checkpoints without the CPU-oracle/save-reload switches when host
RAM or temporary disk is constrained.

## Ported evidence

The pinned standalone repository records:

- BF16/FP32 GEMM and FP32-state recurrence alignment on one exact BR106M;
- 0.1B and 0.4B CPU/SUPA greedy alignment with final-logit cosine above
  `0.999996`;
- all released 0.1B–13.3B checkpoints loading, producing finite logits,
  retaining FP32 recurrent state, and completing cached greedy generation;
- 0.1B chunked prefill, B2 cache reorder, save/reload, PEFT LoRA
  save/load/merge, and one-step Trainer update;
- low-memory BF16 conversion and safe serialization through 13.3B.

The PEFT merged logits had a non-zero BF16 max-absolute difference while the
greedy token matched. That evidence is compatibility, not strict FP32 merge
parity. Recorded latency includes eager/private-backend overhead and is not a
same-card baseline comparison.

## Open gates

- rerun tiny and all six released checkpoints against current HF main;
- run B1/B2/B4/B8 prompt/decode and memory sweeps;
- same-card RWKV-LM and Albatross throughput/memory comparison;
- TRL SFT/DPO/GRPO, checkpoint resume and real ZeRO-2/ZeRO-3;
- W8/W4 footprint, quality, operator coverage and no-slower speed gates;
- optimize GroupNorm, recurrent and projection paths only after exact-card A/B;
- multi-BR106M PP/TP and longer stability runs.
