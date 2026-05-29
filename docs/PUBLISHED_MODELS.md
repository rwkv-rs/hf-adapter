# Published RWKV7-G1 Hugging Face Models

Chinese version: [`PUBLISHED_MODELS_ZH.md`](PUBLISHED_MODELS_ZH.md)

The ready-to-load FP16 family is grouped in the
[`RWKV7-G1 Transformers` collection](https://huggingface.co/collections/wangyue114514/rwkv7-g1-transformers-6a85b04191034d4c2d1896f1).
Each model is an independent Transformers repository pinned to
`rwkv7-hf==0.7.0`.

## Model matrix

| Model | Parameters | FP16 weight files | Weight bytes | Conservative available-memory starting point |
|---|---:|---:|---:|---:|
| [`rwkv7-g1d-0.1b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1d-0.1b-hf) | 191,034,624 | 1 | 0.38 GB | 1 GB |
| [`rwkv7-g1d-0.4b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1d-0.4b-hf) | 450,767,872 | 1 | 0.90 GB | 2 GB |
| [`rwkv7-g1g-1.5b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1g-1.5b-hf) | 1,527,404,544 | 6 | 3.05 GB | 6 GB |
| [`rwkv7-g1g-2.9b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1g-2.9b-hf) | 2,947,735,040 | 13 | 5.90 GB | 10 GB |
| [`rwkv7-g1g-7.2b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1g-7.2b-hf) | 7,199,141,888 | 4 | 14.40 GB | 20 GB |
| [`rwkv7-g1g-13.3b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1g-13.3b-hf) | 13,269,245,952 | 7 | 26.54 GB | 32 GB |

The memory column is a first-load guideline, not a universal peak-VRAM claim.
Backend, dtype, batch size, prompt length, training, quantization, and device
placement can all change the actual requirement. Start with 0.1B when checking
a new environment.

## Install and load directly

```bash
python -m pip install "rwkv7-hf==0.7.0"
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "wangyue114514/rwkv7-g1d-0.1b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = tokenizer("User: Hello! Assistant:", return_tensors="pt")
inputs = {name: value.to(device) for name, value in inputs.items()}
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=16)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The model repositories contain weights, config, tokenizer assets, and three
small remote-code entrypoints. The maintained implementation and optimized
operators come from the pinned PyPI package instead of being copied into every
model repository.

## One-command release verification

From a checkout of this repository:

```bash
python scripts/verify_hf_release.py \
  --model wangyue114514/rwkv7-g1d-0.1b-hf
```

Success ends with `RESULT: PASS`. The command verifies the Hub revision,
conversion manifest, remote LFS sizes and SHA256 values, config, tokenizer,
loading keys, parameter count, finite logits, and a real generation.

For large repositories, verify metadata without downloading the weights:

```bash
python scripts/verify_hf_release.py \
  --model wangyue114514/rwkv7-g1g-13.3b-hf \
  --metadata-only
```

## Published models versus source checkpoints

[`BlinkDL/rwkv7-g1`](https://huggingface.co/BlinkDL/rwkv7-g1) remains the
authoritative multi-checkpoint archive. The repositories above are the
productized Transformers form: one independently loadable model per repository,
grouped by a Hub Collection. Every repository includes a
`conversion_manifest.json` that records the source revision, source and output
hashes, parameter count, runtime version, and validation result.
