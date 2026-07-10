# RWKV7 NVIDIA backend-v2 frozen design

This document freezes the complete optional-backend boundary before the old
performance implementation is moved.  The migration is one implementation
change: public names and tensor/cache contracts below do not change while
individual CUDA paths are validated.

## Non-negotiable ownership

`rwkv7_hf/` owns the readable Hugging Face model, configuration, tokenizer,
loss, generation contract, and canonical `RWKV7Cache`.  It remains usable with
only Torch and Transformers.

`kernels/rwkv7_kernels/` owns every hardware decision and optimized
implementation: Triton/CUDA operators, model operand packing, CUDA Graph
runners, internal state pools, shape routing, quantized linear operators, and
training autograd extensions.  No device name, tile, threshold, environment
variable, graph runner, quantizer, or compiled-source loader is added to model
configuration or cache.

## Frozen public protocol

The optional package keeps the existing low-level recurrent protocol and adds
one model-forward protocol.  A single model-forward boundary is sufficient for
prefill, one-token decode, generation graph replay, and fused training without
adding a new public entry point for every kernel.

```python
RWKV7_KERNEL_API_VERSION = 2

probe_recurrent_v1(...canonical tensors...) -> Support
recurrent_v1(...canonical tensors...) -> (output, canonical_state)

probe_model_forward_v1(owner, request: Mapping[str, object]) -> Support
model_forward_v1(owner, request: Mapping[str, object]) -> ModelForwardResult
```

`owner` is an `RWKV7Model` or `RWKV7ForCausalLM`. The optional package uses
structural attributes (layers, projection modules, norms, embedding, and LM
head); it never imports `rwkv7_hf.modeling_rwkv7`.

`request` mirrors the stable HF call fields:

- `model_kind`: `base` or `causal_lm`
- normalized `hidden_states`, `attention_mask`, `past_key_values` at the base
  model boundary; raw `input_ids` / `inputs_embeds` remain owned by HF
- `labels`, `use_cache`, `output_hidden_states`, `return_dict`
- `cache_position`, `logits_to_keep`
- `training`, `gradient_checkpointing`, `grad_enabled`

The probe returns `supported`, `implementation`, `reason`, and `phase`, where
phase is `prefill`, `decode`, or `training`.  Unsupported inputs do not mutate
cache or model state.  `auto` falls back to the readable model; `optimized`
raises with the probe/failure reason.

`ModelForwardResult` is a plain mapping and cannot contain Transformers output
classes.  Allowed values are:

- `last_hidden_state`
- `logits`
- `loss`
- `past_key_values` (the same public `RWKV7Cache` type supplied by the model)
- `hidden_states`
- `implementation` and `phase`

The clean model validates shapes/types and constructs standard
`BaseModelOutputWithPast` or `CausalLMOutputWithPast` itself.

## Cache and layout contract

- Public recurrent state is always `[B,H,K,V]`, FP32 unless the reference
  contract explicitly changes.
- Attention and FFN shift states are `[B,C]` per layer.
- `seen_tokens` is updated once by the owning HF forward call.
- Internal `[V,K]`, packed, pooled, or graph-static buffers live only inside
  the optional package and are converted/bound at protocol entry and exit.
- Masked tokens do not update recurrent or shift state.
- Batch reorder/select/repeat remains owned by `RWKV7Cache`; graph runners
  must rebind after those operations and never replace the public cache.

## Complete NVIDIA implementation inventory

The following old performance families move together behind the frozen model
protocol.  They are not copied into the clean model package.

### Dense prefill and decode

- dense operand packing and stacked R/K/V projection
- fused norm/shift/mix, W/A/G/V LoRA and R/K/V projection
- recurrent FP16/FP32 update and fused recurrent output
- fused attention output projection
- fused FFN up/ReLU2/down/residual
- DPLR, self-chunk, and shape-selected prefill
- one-token dense step, state pool, CUDA Graph capture/replay and LRU runners

### NVIDIA generation policies

- SM70/V100/T4 routes
- Ada/RTX 4080/4090 routes
- Blackwell/RTX 5090 routes
- batch/sequence/hidden-size routing and prewarm

### Quantization

- native W8/W4, A8W8, Bn/Tn layouts and calibration metadata
- bitsandbytes W8 adapter
- Marlin/TorchAO adapters and prebuilt/lazy extension loading
- quantized graph-safety and dense fallback rules

### Training

- Mix6, KkPre, A-gate, V-residual gate, clamp-W, CMix and fused loss autograd
- input/state/weight gradient parity
- gradient-checkpointing, resume and adapter/PEFT safety

## Explicitly separate plugins

Apple/MLX/CoreML, Ascend, MUSA, Biren and MetaX are not NVIDIA backend-v2.
Their old implementations remain preserved on `perf/native-kernels-v0.8` and
will use separate optional distributions/protocol implementations.  Excluding
them avoids importing unrelated runtimes and licenses into the CUDA wheel; it
does not remove their history.

## Integration points in the readable model

Only two visible boundaries are allowed:

1. the existing `rwkv7_recurrent(...)` call inside `RWKV7TimeMix`;
2. one early `maybe_model_forward(...)` call in `RWKV7Model.forward` and
   `RWKV7ForCausalLM.forward`, followed by the complete existing PyTorch body.

There are no optimized subclasses, monkey patches, duplicate model layers, or
native cache classes.  Installing/uninstalling `rwkv7-kernels` changes only the
selected implementation, never the HF class names or checkpoint layout.

## One-shot acceptance

The code migration is completed before backend-v2 is published or made the
default.  Acceptance may run in smaller diagnostic jobs, but a release build
requires one immutable wheel hash to pass all of the following:

- operator output/state/all-gradient parity;
- 0.1B/0.4B/1.5B full logits, cache, padding, greedy and beam;
- AutoModel/AutoModelForCausalLM, save/reload and package-free fallback;
- Trainer/Accelerate/PEFT plus SFT/DPO/GRPO and resume;
- reference/optimized/FLA 144-unit lm_eval equivalence;
- whole-model prefill/decode/training speed against readable reference and
  pinned FLA;
- RTX 4080, V100 and RTX 4090, followed by larger-model loading smoke.

Reference and FLA results may be reused only when their source/model/task
hashes are unchanged.  Every changed optimized wheel reruns the optimized
lane; filenames and requested routes never substitute for actual route traces.
