# Architecture

The reference line copies four model files into every converted model repository:

1. `configuration_rwkv7.py` — architecture-only configuration.
2. `cache_rwkv7.py` — recurrent cache lifecycle and batch operations.
3. `ops_rwkv7.py` — one pure-PyTorch WKV recurrence boundary.
4. `modeling_rwkv7.py` — TMix, CMix, blocks, backbone and causal LM.

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

## Numerical reproducibility

The model still uses ordinary `torch.nn.functional.linear` and PyTorch matrix
multiplication. To make FP16 scores independent of how an evaluation or
training framework regroups the same examples, `modeling_rwkv7.py` tiles batch
and time into a fixed 128-row linear shape. Padding rows are discarded before
the recurrent boundary; `ops_rwkv7.py` remains the direct batched equation.
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
