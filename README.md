# RWKV-7 HF Adapter

First-stage Hugging Face adapter for official RWKV-7 `.pth` checkpoints.

This repository converts RWKV-7 weights to a Hugging Face-style directory and provides remote-code wrappers so the result can be loaded with:

- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
- `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
- `model.generate(..., use_cache=True)`
- PEFT LoRA smoke tests

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
  test_peft_lora.py
NEXT_STEPS.md
```

## Convert an official checkpoint

```bash
export PYTHONPATH=/path/to/flash-linear-attention:/path/to/rwkv7-hf-adapter:$PYTHONPATH

python scripts/convert_rwkv7_to_hf.py \
  --input /path/to/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --output /path/to/rwkv7-g1d-0.1b-hf \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --attn-mode chunk
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

## Current validation

For `rwkv7-g1d-0.1b-20260129-ctx8192`:

- HF `generate()` works.
- PEFT LoRA forward/loss/backward works.
- Official `rwkv` logits comparison on a smoke prompt:
  - top-5 token IDs match
  - cosine similarity ≈ `0.999996`
  - fp16 max absolute difference ≈ `0.047`

## Known limitations

- This is a wrapper-based first stage, not yet a native upstream Transformers implementation.
- The backend currently requires FLA.
- V100 `chunk` prefill/backward kernel behavior needs further optimization and compatibility work.
