# Huawei Ascend NPU / torch-npu

The canonical Native/no-FLA Hugging Face model supports Huawei Ascend through
ordinary PyTorch operators registered by `torch_npu`. The integration is
optional and import-safe: CPU, CUDA, ROCm, MPS and MUSA users do not need
`torch_npu`.

This main-repository integration ports the HF runtime, fixed-batch NPUGraph
runner, exact-stack W8 route, explicit W4 candidate and W4 calibration helper
from [`rwkv-rs/rwkv7-ascend-npu` at `b6391271f`](https://github.com/rwkv-rs/rwkv7-ascend-npu/tree/b6391271f0ddb606dad5e97a65fa4742e82fcd50).
The standalone repository remains the source of the complete raw logs, hashes,
serving-engine code and release artifacts. They are intentionally not duplicated
inside this HF-only repository.

## Support boundary

| Item | Current boundary |
|---|---|
| Device | exact Huawei Ascend 910B3 only for production-admitted routes |
| Software | CANN 8.5.0, PyTorch 2.9.0+cpu, torch-npu 2.9.0 |
| HF compatibility | Native eager/JIT load, forward, generate, recurrent cache, chunked prefill and save/reload |
| Graph decode | fixed-batch `torch.npu.NPUGraph`; graph-captured prefill is not claimed |
| W8 | exact 7.2B FP16 FFN shapes and B1/B4/B8 speed policy |
| W4 | candidate only; never selected by a production policy |
| Other Ascend cards/stacks | experimental only with an explicit override; no inherited performance claim |

The source repository validated the above stack on 2026-07-24. Its real 7.2B
HF gate includes BF16 forward/generation/cache alignment, fixed-batch FP16 graph
decode, and paired W8 checks. This port is based on that accepted source commit;
a fresh full 7.2B run against the current canonical HF main is still required
before changing the evidence provenance from “ported” to “current-main rerun”.

## Install

Install the CANN-matched PyTorch and `torch_npu` wheels from the Huawei Ascend
distribution first. There is no universal PyPI dependency because the wheel is
specific to CANN, Python and architecture.

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
python3.11 -m venv .venv
. .venv/bin/activate
# Install the exact Huawei torch + torch_npu wheel pair here.
python -m pip install "rwkv7-hf[ascend]==0.8.0"
python -c 'from rwkv7_hf import ascend_available; print(ascend_available())'
```

The last command must print `True`. The empty `ascend` extra is a stable adapter
entrypoint; it intentionally does not guess or install a CANN-specific vendor
wheel. Use `pip install -e '.[ascend]'` only after cloning the repository for
source development or validation scripts.

## Standard Transformers usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rwkv7_hf import enable_ascend

info = enable_ascend("npu:0", backend="eager")
model_dir = "/path/to/converted-rwkv7-hf"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).eval().to(info.device)
inputs = tokenizer("User: Hello. Assistant:", return_tensors="pt").to(info.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=8, use_cache=True)
print(tokenizer.decode(output[0]))
```

The command-line example also accepts `--device npu`; its automatic dtype is
BF16. `enable_ascend` imports `torch_npu` lazily, configures conservative native
flags, validates the exact stack before selecting the device, and returns the
observed runtime metadata.

`RWKV7_ALLOW_UNVALIDATED_ASCEND=1` permits an explicit experimental audit on an
unknown card or software stack. The returned status is
`unvalidated_override`; such a run is not production evidence.

## Fixed-batch NPUGraph decode

```python
info = enable_ascend("npu:0", backend="native_graph")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    dtype=torch.float16,
).eval().to(info.device)
model.rwkv7_warmup_fast_token((1, 4, 8), backend="native_graph")
print(model.rwkv7_native_graph_cache_stats())
```

Each graph runner owns fixed-address token, logits and recurrent-state buffers
for one batch size. `RWKV7_ASCEND_GRAPH_CACHE_SIZE` controls the per-model LRU
(default: 3). Cache selection/reorder stays available for fixed-size batches,
and changing packed quantization buffers changes the graph key so a dense graph
cannot be replayed after module replacement.

The standalone real-7.2B artifact reported FP16 graph decode at 29.8664,
70.3739 and 141.7044 output tok/s for B1/B4/B8. Those numbers describe the
exact source commit and machine only; use the standalone repository for raw
JSON, logs and hashes.

## W8 speed route and W4 candidate

```python
from rwkv7_hf import quantize_ascend_w8a16

replaced = quantize_ascend_w8a16(model, policy="speed", strict=True)
print(len(replaced))  # exact 7.2B route: all 64 FFN key/value projections
```

The W8 speed policy fails closed unless all of these match:

- Ascend 910B3 / CANN 8.5.0 / torch 2.9.0+cpu / torch-npu 2.9.0;
- FP16 activation and source weights;
- the 32-layer 7.2B layout with `hidden_size=4096` and
  `intermediate_size=16384`;
- `ffn.key` 4096 -> 16384 and `ffn.value` 16384 -> 4096;
- logical rows B1, B4 or B8.

The standalone paired graph gate reported W8/FP16 medians of 1.0241x, 1.0205x
and 1.0259x for B1/B4/B8, with model tensor payload at 70.18% of FP16 and exact
greedy output on its production prompt set.

W4 is deliberately not production-promoted. It can be packed only through the
explicit acceptance API:

```python
from rwkv7_hf import quantize_ascend_w4a16_candidate

quantize_ascend_w4a16_candidate(
    model,
    group_size=128,
    require_explicit_candidate=False,
)
```

The retained W4 graph candidate reduced memory and improved measured latency,
but failed the quality threshold. `should_quantize(...)` therefore returns
`False` for every W4 tuple. `rwkv7_hf.ascend_w4_cle` provides a calibration
candidate for future quality work without changing that admission boundary.

## Reproduce the hardware smoke

```bash
PYTHON_BIN=.venv/bin/python DEVICE=npu:0 DTYPE=bf16 BACKEND=eager \
  bash scripts/run_huawei_ascend_smoke.sh /path/to/converted-rwkv7-hf

PYTHON_BIN=.venv/bin/python DEVICE=npu:0 DTYPE=fp16 BACKEND=native_graph \
  RESULTS=bench/local_ascend/results_graph.jsonl \
  bash scripts/run_huawei_ascend_smoke.sh /path/to/converted-rwkv7-hf
```

A pass row covers forward, generate, recurrent-cache selection/continuation,
chunked prefill and save/reload. Add `TRAINING_SMOKE=1` for the tiny
loss/backward/update check. A tiny fixture is compatibility evidence, not an
official-checkpoint quality or throughput result.

## Open gates

- rerun the full standalone 7.2B acceptance matrix on the current HF main;
- real-checkpoint PEFT/Trainer/TRL and multi-NPU HCCL/device-map validation;
- dynamic-shape graph and graph-captured prefill;
- longer stability/soak tests in the HF repository;
- production W4 quality admission;
- independent evidence for other Ascend cards and software stacks.
