# Checkpoint conversion

```bash
rwkv7-hf convert \
  --input /absolute/model.pth \
  --output /absolute/model-hf \
  --vocab-file /absolute/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --low-memory
```

The output contains at least:

```text
config.json
configuration_rwkv7.py
cache_rwkv7.py
ops_rwkv7.py
modeling_rwkv7.py
tokenization_rwkv7.py
chat_template.jinja
rwkv_vocab_v20230424.txt
tokenizer_config.json
model.safetensors
```

Large models may use sharded safetensors plus an index. The directory loads
without a source checkout or package installation:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "/absolute/model-hf", trust_remote_code=True
)
```

The low-memory path builds the shape template on the meta device and avoids a
second full dense model allocation. `--max-shard-size` controls sharding.
There is one output layout: the complete self-contained reference model. The
converter has no package-backed or legacy layout mode.
