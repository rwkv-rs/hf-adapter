# Architecture

The reference line copies four model files into every converted model repository:

1. `configuration_rwkv7.py` — architecture-only configuration.
2. `cache_rwkv7.py` — recurrent cache lifecycle and batch operations.
3. `ops_rwkv7.py` — one pure-PyTorch WKV recurrence boundary.
4. `modeling_rwkv7.py` — TMix, CMix, blocks, backbone and causal LM.

The installed source tree keeps these model modules in `rwkv7_hf/`. CLI,
conversion, manifest, and smoke-test code is isolated in the sibling
`rwkv7_hf_tools/` package and is never copied into model repositories.

`RWKV7TimeMix` computes shifts and R/W/K/V/A/G projections, calls
`rwkv7_recurrent`, applies GroupNorm and the direct RKV term, then projects
back to the residual width. `RWKV7ChannelMix` is the shifted squared-ReLU
FFN. `RWKV7Block` exposes both residual paths directly.

## State

For each layer `RWKV7Cache` stores recurrent state
`[batch, heads, key_dim, value_dim]`, attention shift
`[batch, hidden]` and FFN shift `[batch, hidden]`.

`v_first` is passed from layer zero to later layers during one forward call
and is never persisted. False positions in a 2-D attention mask do not update
any recurrent or shift state, so left and right padding are deterministic.

The cache implements sequence length, reorder, select, repeat, reset, detach and
device/dtype conversion without graph runners, counters, hardware policy or
layout routing.

## Optional high-performance boundary

Installing `rwkv7-kernels` does not change the four reference files or add a
second model class. The reference model has only two optional call boundaries:

1. `rwkv7_recurrent(...)` can delegate the canonical recurrent tensors to the
   versioned low-level kernel protocol;
2. the base/causal-LM forward methods can offer one normalized request to the
   versioned whole-model protocol before executing their unchanged readable
   PyTorch body.

The optional wheel sees the existing layer/Linear/norm structure, returns plain
tensors plus the same `RWKV7Cache`, and never imports or replaces
`modeling_rwkv7.py`. Unsupported devices, dtypes, shapes, adapter structures or
training modes fail closed to the reference implementation in `auto` mode.
There is no monkeypatch, optimized subclass, private checkpoint layout, or
hardware field in `RWKV7Config`.

The wheel owns all performance-specific code: recurrent Triton/CUDA routes;
dense/fused decode; DPLR and self-chunk fused prefill; projection, norm, FFN
and LoRA fusions; CUDA Graph/state pools; exact SM70, Ada and Blackwell policy;
W8/W4/A8W8/BN-TN/BitsAndBytes/Marlin/TorchAO adapters; and the train-temp
forward/backward autograd operators. Public recurrent state is converted back
to canonical `[B,H,K,V]` before the call returns.

`nvidia/MIGRATION_MANIFEST.json` verifies all 102 historical NVIDIA transfers:
100 remain byte-identical, while the graph-cache runtime and train-temp wrapper
are explicitly adapted to the canonical cache and non-monkeypatch protocol.
`nvidia/CAPABILITY_INVENTORY.json` maps every one of those payloads exactly once
to an executable runtime family and real device-policy fields.
`nvidia/SOURCE_SCOPE.json` independently classifies the entire 153-file
historical package tree and reconstructs its frozen Git tree object, so the
102-file NVIDIA set cannot hide an omitted source file.
`nvidia/RECURRENT_SOURCE_SCOPE.json` separately reconstructs the later v0.10
recurrent-wheel subtree and byte-verifies its Graph and Triton implementations.
The release-wheel audit checks all four files from the built ZIP, not merely from
the checkout. Full-model production `auto` is promoted only after the same
immutable wheel passes the RTX 4080 and RTX 4090 functional, HF/training,
speed, FLA and 144-unit `lm_eval` gates. V100 remains implemented and retains
historical evidence, but it is not a v1.0 release gate.

## Numerical reproducibility

The model still uses ordinary `torch.nn.functional.linear` and PyTorch matrix
multiplication. To make FP16 scores independent of how an evaluation or
training framework regroups the same examples, `modeling_rwkv7.py` tiles batch
and time into a fixed 128-row linear shape and `ops_rwkv7.py` evaluates the
direct recurrence independently per sample. Linear padding rows are discarded.
This changes neither the RWKV equations nor checkpoint keys and has no
device-specific route, environment variable, compiled extension, or custom
kernel.

## HF contract

`RWKV7ForCausalLM.forward` accepts `input_ids`, `inputs_embeds`,
a 2-D `attention_mask`, `past_key_values`, `labels`,
`use_cache`, `output_hidden_states`, `return_dict`,
`cache_position` and `logits_to_keep`. Labels are shifted internally
and -100 is ignored. Cache is disabled while training and gradient
checkpointing. `output_attentions=True` raises `NotImplementedError`
because RWKV has no Transformer attention matrix to return.
