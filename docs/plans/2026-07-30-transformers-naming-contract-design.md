# Transformers naming contract design

> Historical lifecycle note: this design records the compatibility decisions
> for the implementation. Current behavior is enforced by the public contract
> tests and the stable architecture documentation.

## Goal

Make the optimized FLA-backed wrapper expose the same inspectable public
argument names as ordinary Transformers causal language models, and make both
RWKV-7 configuration classes expose `num_attention_heads` without replacing
the established RWKV `num_heads` field.

## Forward signature

`RWKV7ForCausalLM.forward` will explicitly declare the common Transformers
inputs used by generation, Trainer, PEFT and downstream signature inspection:
`input_ids`, `attention_mask`, `inputs_embeds`, `past_key_values`, `labels`,
`shift_labels`, `use_cache`, output controls, `logits_to_keep`, `position_ids`
and `cache_position`. Extra version-specific arguments remain accepted through
`**kwargs`.

The existing wrapper dispatch remains unchanged. The explicit arguments are
normalized into keyword arguments and then follow the current native-prefill,
fast-token or upstream FLA `super().forward` route. The deprecated
`num_logits_to_keep` spelling remains accepted through `**kwargs` and is
normalized before dispatch.

## Configuration aliases

`num_heads` remains the canonical RWKV field because current checkpoints,
kernels and converted configs use it. `num_attention_heads` is a Transformers
compatibility alias with the same integer value. Construction accepts either
name, serializes both names, and rejects conflicting non-null values instead of
silently selecting one.

The rule applies to both `NativeRWKV7Config` and the FLA-backed
`RWKV7HFAdapterConfig`. It does not rename parameters, state-dict keys,
`model_type` or `auto_map` entries.

## Validation

CPU tests inspect the public signature, exercise both config spellings,
round-trip serialized configs, verify conflict errors, and retain the existing
native Transformers contract suite. No model mathematics, GPU policy or kernel
default changes, so new performance claims or exact-card reruns are not part of
this structural compatibility change.
