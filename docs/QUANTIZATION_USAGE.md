# W8/W4 usage tutorial

RWKV-7 exposes three quantization families: standard HF bitsandbytes, native
MM8/MM4, and Apple MLX packed W8/W4. They have different hardware support and
performance behavior. This tutorial separates three claims:

1. **functional:** loading, finite logits, cache decode, and generation pass;
2. **memory:** model footprint is lower than the same dense baseline;
3. **speed:** paired end-to-end timing is no slower on the exact card/shape.

Chinese version: [`QUANTIZATION_USAGE_ZH.md`](QUANTIZATION_USAGE_ZH.md)

![Choose and verify bitsandbytes, native MM8/MM4, or MLX W8/W4](assets/tutorials/09-quantization-paths.png)

## 1. Install and establish a dense baseline

```bash
python -m pip install -e ".[quant]"
python tests/test_quantized_inference.py --model MODEL \
  --device cuda --dtype fp16 --quantization none --max-new-tokens 4
```

Save the JSON row. It records `model_footprint_mb`, `peak_vram_mb`, generated
tokens, and timing telemetry. A quantized result without a matching dense row
cannot establish memory or speed improvement.

## 2. Standard HF bitsandbytes W8/W4

Run W8:

```bash
python tests/test_quantized_inference.py --model MODEL \
  --device cuda --dtype fp16 --quantization 8bit --max-new-tokens 4
```

Run NF4 W4 with double quantization:

```bash
python tests/test_quantized_inference.py --model MODEL \
  --device cuda --dtype fp16 --quantization 4bit \
  --bnb-4bit-quant-type nf4 --bnb-4bit-use-double-quant \
  --max-new-tokens 4
```

Each command must print a JSON row with `status: pass`, nonzero quantized module
counts, finite logits, generated tokens, and final `PASS`. Do not use
`--optional` for acceptance: a skipped backend is not a pass.

Direct HF loading uses the standard API:

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

qconfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "MODEL",
    trust_remote_code=True,
    quantization_config=qconfig,
    device_map="cuda",
)
```

bitsandbytes is primarily a CUDA compatibility/memory path here. Existing rows
show that it is often slower than dense fp16; do not publish a speed claim from
successful loading alone.

## 3. bitsandbytes with the native/no-FLA model

```bash
RWKV7_NATIVE_MODEL=1 python tests/test_native_bnb_quant_smoke.py \
  --model MODEL --device cuda --dtype fp16 --quantization both
```

Success emits pass JSON rows for 8-bit and 4-bit, verifies actual quantized
linear modules plus forward/decode/generate, and prints
`NATIVE BNB QUANT PASS`. Native bnb decode intentionally uses its compatible
eager route where packed native JIT operands are unavailable.

## 4. Native MM8/MM4

Native quantization does not require bitsandbytes. Set exactly one mode on the
loaded RWKV config:

```python
import os
os.environ["RWKV7_NATIVE_MODEL"] = "1"

from transformers import AutoConfig, AutoModelForCausalLM

path = "MODEL"
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
config.use_native_mm8 = True
config.use_native_mm4 = False
config.native_mm8_policy = "speed"       # or "memory"
config.native_mm8_min_params = 8_000_000

model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, config=config
).eval()
print(model._rwkv7_native_mm_quantization)
print(model._rwkv7_native_mm_replaced_modules)
```

For MM4, set `use_native_mm8=False`, `use_native_mm4=True`, and configure
`native_mm4_policy`/`native_mm4_min_params`. MM8 and MM4 are mutually
exclusive.

- `speed` keeps most dense blocks and quantizes selected expensive projections;
  it saves less memory but is the only route eligible for exact-card speed
  promotion.
- `memory` replaces many eligible linears; it usually saves more memory but is
  not a universal fp16-or-faster route.

Verify config round trips without a large checkpoint, then verify real MM8
persistence:

```bash
python tests/test_native_quant_config.py
python tests/test_native_mm8_persist.py --model MODEL
```

The first prints `NATIVE QUANT CONFIG PASS`; the second prints `PASS` and checks
reloaded MM8 modules plus a cosine floor. Persisted flags cause eligible
linears to be packed again when the model is reloaded.

## 5. Accept or reject a quantized path

For the exact model, card, dtype, batch size, prompt length, and decode length:

| Gate | Required evidence |
|---|---|
| Functional | quant modules exist; logits are finite; forward, cache decode, and generate exit 0 |
| Quality | dense/quant logits meet the declared cosine/error floor and greedy next tokens match |
| Memory | quant `model_footprint_mb` is lower; report peak VRAM separately |
| Speed | paired warm steady-state prefill and decode timing; do not substitute a microbenchmark |
| Reproducibility | exact policy, thresholds, replaced-module count, package versions, GPU, and command |

Use [`QUANTIZATION.md`](QUANTIZATION.md) and
[`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) to determine whether an exact-card
route has promoted evidence. If no matching row exists, describe the result as
a local experiment.

## 6. Apple MLX W8/W4

MLX uses a separate packed runtime and does not use bitsandbytes. Follow
[`APPLE_USAGE.md`](APPLE_USAGE.md#4-packed-mlx-w8w4) for conversion, generation,
sessions, and the M5 evidence boundary.

## 7. AI execution rule

An AI assistant must run the dense baseline first, then only one quantized
mode. It must report module counts, footprint, peak VRAM, logits/greedy gates,
exact GPU and shape. It may say “faster” only after paired end-to-end timing;
“loaded”, “passed”, or “used less memory” are different conclusions.
