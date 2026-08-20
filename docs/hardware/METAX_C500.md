# MetaX C500 / MXMACA

The canonical Native/no-FLA Hugging Face model supports MetaX C500 through
MXMACA's PyTorch `torch.cuda` compatibility API. The C500 reports CUDA-like
capability 8.0, but it is **not** an NVIDIA Ampere GPU. The adapter therefore
identifies the exact product name before capability-family routing and selects
a conservative native eager path instead of inheriting NVIDIA kernels.

This integration ports the HF compatibility work and acceptance contract from
[`123123213weqw/rwkv7-metax-c500` at `f2653e2`](https://github.com/123123213weqw/rwkv7-metax-c500/tree/f2653e20250821ec48534e5e08b07d59effb985c).
The standalone repository remains the source for raw environment JSON, logs,
checksums and the separate vLLM/SGLang tracks. Those artifacts are not copied
into this HF-only repository.

## Current boundary

| Item | Current boundary |
|---|---|
| Device | exact MetaX C500 64GB |
| Validated stack | MXMACA 3.5.3.20, driver 3.8.30, PyTorch 2.8.0+metax3.5.3.9, CUDA compatibility string 11.6 |
| HF backend | native eager/no-FLA; JIT, CUDA graph and Triton fusions fail closed by default |
| Dtypes | tiny-model FP32, FP16 and BF16 inference/backward; real 0.4B FP16 inference and BF16 training evidence |
| HF API | forward, cached generation, cache selection, chunked prefill, save/reload, Trainer and PEFT LoRA evidence |
| Quantization | no production W8/W4 route promoted |
| Performance | compatibility telemetry only; no RWKV-LM/Albatross performance-close claim |

The standalone evidence was produced against an older HF adapter commit plus
the FP32 key-normalization patch now included here. A full **current-main** C500
rerun is still required before the evidence provenance can be changed from
“ported” to “current-main rerun”.

## Why FP32 key normalization is required

RWKV-7 normalizes a per-head key vector with `torch.nn.functional.normalize`.
Its default epsilon is `1e-12`, which cannot be represented in FP16. A zero or
near-zero key norm can therefore become a division by zero on the C500 native
path. The eager implementation now performs this reduction and epsilon clamp
in FP32, then casts the normalized vector back to the projection dtype. This is
a numerical-stability correction, not a MetaX-only operator fork.

## Install

Start from the official C500/MXMACA runtime image and install its matched
PyTorch build first. The vendor wheels are not declared as universal PyPI
dependencies.

```bash
export MACA_PATH=/opt/maca
export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUDA_PATH=/opt/maca/tools/cu-bridge
python -m pip install "rwkv7-hf[metax]==0.8.0"
python -c 'from rwkv7_hf import metax_available; print(metax_available())'
```

The last command must print `True`. `enable_metax()` additionally checks the
exact device, MXMACA, PyTorch and compatibility-version row. An unknown stack
fails closed. `RWKV7_ALLOW_UNVALIDATED_METAX=1` permits an explicitly reported
experimental run but does not promote that stack. Use editable
`pip install -e '.[metax]'` only inside a cloned source tree for repository
tests or development.

## Transformers usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rwkv7_hf import enable_metax

info = enable_metax("cuda:0")
model_dir = "/path/to/converted-rwkv7-hf"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    dtype=torch.float16,
).eval().to(info.device)
inputs = tokenizer("User: Hello!\n\nAssistant:", return_tensors="pt").to(info.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=8, use_cache=True)
print(tokenizer.decode(output[0]))
```

The command-line example accepts `--device metax`. Automatic selection also
recognizes an exact C500 before the generic CUDA branch:

```bash
python examples/generate.py \
  --model /path/to/converted-rwkv7-hf \
  --prompt 'User: Hello! Assistant:' \
  --device metax --backend native --dtype fp16
```

## Reproduce the current smoke

Tiny FP32/FP16/BF16 forward, cache, chunked-prefill, generation and backward:

```bash
PYTHON_BIN=python RESULTS=bench/local_metax/results.json \
  bash scripts/run_metax_c500_smoke.sh
```

Add a real converted checkpoint:

```bash
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
PYTHON_BIN=python RESULTS=bench/local_metax/results-0.4b.json \
  bash scripts/run_metax_c500_smoke.sh
```

For training-ecosystem reruns, use the repository's existing HF acceptance
scripts after `enable_metax()`/the conservative environment has been selected:

```bash
RWKV7_NATIVE_MODEL_BACKEND=eager RWKV7_NATIVE_MODEL_JIT=0 \
python tests/test_peft_lora.py --model /path/to/model --device cuda --attn-mode chunk

RWKV7_NATIVE_MODEL_BACKEND=eager RWKV7_NATIVE_MODEL_JIT=0 \
python tests/test_hf_training_smoke.py --model /path/to/model \
  --device cuda --attn-mode chunk --train-dtype bf16 --max-steps 3
```

## Ported evidence

The pinned standalone repository records:

- exact C500 environment and synchronized FP16/BF16 matmul correctness;
- tiny Native HF FP32/FP16/BF16 forward, cache, generation and backward;
- real 0.4B FP16 CPU-oracle alignment, split/chunked prefill, B8-to-B3 cache
  selection, generation and save/reload;
- real 0.4B BF16 Trainer + LoRA three-step training;
- strict FP32 PEFT save/load/merge roundtrip.

The recorded eager latency values include compatibility-path overhead and are
**not a performance benchmark**. The FP16 PEFT merge diagnostic exceeded its
strict max-absolute-logit threshold; FP32 merge passed. Both facts remain part
of the acceptance boundary.

## Open gates

- rerun tiny and real 0.4B evidence against current HF main;
- validate all released model sizes and B1/B2/B4/B8 prompt/decode cells;
- same-card RWKV-LM and Albatross throughput/memory comparison;
- TRL SFT/DPO/GRPO, checkpoint resume and real ZeRO-2/ZeRO-3;
- W8/W4 footprint, quality and no-slower speed gates;
- native graph/JIT/fused-kernel work only after exact C500 A/B evidence;
- multi-C500 PP/TP and longer stability runs.
