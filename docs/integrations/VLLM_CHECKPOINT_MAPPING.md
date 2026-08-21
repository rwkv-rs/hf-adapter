<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  reference_implementation: rwkv7_hf/converter.py
  metadata: ../reference/provenance.yaml
  license: MIT
-->

# RWKV-7 checkpoint and weight mapping

## Supported input

The preferred serving input is the native HF converted directory:

```text
config.json
generation_config.json
model*.safetensors
tokenizer_config.json
special_tokens_map.json
rwkv_vocab_v20230424.txt
remote-code files (for Transformers use)
```

A serving engine needs `config.json`, safetensors, and tokenizer assets. It
does not need to execute remote model code if it implements this specification.

The canonical installed conversion source is `rwkv7_hf/converter.py`.
`scripts/convert_rwkv7_to_hf.py` is a backward-compatible wrapper.

## Required configuration

| Field | Meaning |
|---|---|
| `model_type` | `rwkv7_native` in canonical converted models |
| `vocab_size` | embedding/output vocabulary |
| `hidden_size` | residual width `D` |
| `attention_hidden_size` | recurrent attention width `A` |
| `num_hidden_layers` | `L` |
| `num_heads` / `num_attention_heads` | `H` |
| `head_dim` | `N` |
| `intermediate_size` | FFN width `F` |
| `decay_low_rank_dim` | `Rw` |
| `a_low_rank_dim` | `Ra` |
| `gate_low_rank_dim` | `Rg` |
| `v_low_rank_dim` | `Rv` |
| `tie_word_embeddings` | must be false |
| `torch_dtype` | checkpoint/reference dtype |
| `pad_token_id`, `eos_token_id`, `bos_token_id` | tokenizer/generation metadata |

Validate:

```text
attention_hidden_size == num_heads * head_dim
```

Do not assume `attention_hidden_size == hidden_size`.

Quantization config fields in `NativeRWKV7Config` describe HF runtime
replacement policies. A vLLM integration should translate them into its own
explicit quantization configuration rather than executing HF replacement code.

## Top-level mapping

| Official RWKV-LM `.pth` | Converted safetensors |
|---|---|
| `emb.weight` | `model.embeddings.weight` |
| `ln_out.weight` | `model.norm.weight` |
| `ln_out.bias` | `model.norm.bias` |
| `head.weight` | `lm_head.weight` |

Shapes:

```text
embedding [vocab,D]
final norm weight/bias [D]
lm_head [vocab,D]
```

The output head is independent; do not tie it to embeddings.

## Per-layer normalization and FFN

For layer index `i`:

| Official | Converted |
|---|---|
| `blocks.i.ln0.weight/bias` | `model.layers.i.pre_norm.weight/bias` |
| `blocks.i.ln1.weight/bias` | `model.layers.i.attn_norm.weight/bias` |
| `blocks.i.ln2.weight/bias` | `model.layers.i.ffn_norm.weight/bias` |
| `blocks.i.ffn.x_k` | `model.layers.i.ffn.x_k` |
| `blocks.i.ffn.key.weight` | `model.layers.i.ffn.key.weight` |
| `blocks.i.ffn.value.weight` | `model.layers.i.ffn.value.weight` |

`pre_norm` is used only by layer 0 in the native architecture. The converter
allows a generated identity/default value when the official checkpoint does
not provide the expected layer-0 tensor.

Shapes:

```text
norms            [D]
ffn.x_k          D elements (converted to module shape)
ffn.key.weight   [F,D]
ffn.value.weight [D,F]
```

## Attention projections and parameters

| Official | Converted |
|---|---|
| `blocks.i.att.x_r` | `model.layers.i.attn.x_r` |
| `blocks.i.att.x_w` | `model.layers.i.attn.x_w` |
| `blocks.i.att.x_k` | `model.layers.i.attn.x_k` |
| `blocks.i.att.x_v` | `model.layers.i.attn.x_v` |
| `blocks.i.att.x_a` | `model.layers.i.attn.x_a` |
| `blocks.i.att.x_g` | `model.layers.i.attn.x_g` |
| `blocks.i.att.k_k` | `model.layers.i.attn.k_k` |
| `blocks.i.att.k_a` | `model.layers.i.attn.k_a` |
| `blocks.i.att.r_k` | `model.layers.i.attn.r_k` |
| `blocks.i.att.receptance.weight` | `model.layers.i.attn.r_proj.weight` |
| `blocks.i.att.key.weight` | `model.layers.i.attn.k_proj.weight` |
| `blocks.i.att.value.weight` | `model.layers.i.attn.v_proj.weight` |
| `blocks.i.att.output.weight` | `model.layers.i.attn.o_proj.weight` |
| `blocks.i.att.ln_x.weight/bias` | `model.layers.i.attn.g_norm.weight/bias` |

Shapes:

```text
x_*                  D elements, module usually [1,1,D]
k_k, k_a             [A]
r_k                  trailing [H,N], leading singleton axes accepted in .pth
r/k/v projection     [A,D]
output projection    [D,A]
group norm           [A]
```

## Low-rank mapping

For `p` in `w`, `a`, `g`, and layer-`i>0` `v`:

| Official suffix | Converted suffix | Conversion |
|---|---|---|
| `p0` | `p_lora.lora.2.bias` | reshape if needed |
| `p1` | `p_lora.lora.0.weight` | transpose |
| `p2` | `p_lora.lora.2.weight` | transpose |

Examples:

```text
blocks.i.att.w1
  -> model.layers.i.attn.w_lora.lora.0.weight

blocks.i.att.w2
  -> model.layers.i.attn.w_lora.lora.2.weight

blocks.i.att.w0
  -> model.layers.i.attn.w_lora.lora.2.bias
```

Destination shapes:

```text
w down/up/bias [Rw,D], [A,Rw], [A]
a down/up/bias [Ra,D], [A,Ra], [A]
g down/up      [Rg,D], [A,Rg]          # bias-free native G path
v down/up/bias [Rv,D], [A,Rv], [A]     # layers > 0
```

Official layer-0 `v0/v1/v2`, if present, are intentionally unused because
layer 0 defines `v_first` directly from its value projection.

When loading already converted safetensors, **do not transpose these weights
again**. Transposition applies only while converting official `.pth` names.

## vLLM loader plan

Recommended module hierarchy:

```text
model.embeddings
model.layers.{i}.pre_norm
model.layers.{i}.attn_norm
model.layers.{i}.attn.*
model.layers.{i}.ffn_norm
model.layers.{i}.ffn.*
model.norm
lm_head
```

Keeping canonical names simplifies:

- direct safetensors loading;
- PEFT/LoRA target mapping;
- conversion parity;
- per-module quantization policy;
- comparison with HF state dictionaries.

Loader procedure:

1. parse and validate config;
2. build meta/empty modules;
3. stream safetensor shards;
4. reject unknown/missing required keys;
5. verify every shape;
6. apply TP partitioning while loading;
7. optionally pack quantized weights;
8. release dense source tensor after successful pack;
9. compute/load a checkpoint fingerprint;
10. run a small golden-vector gate.

## Tensor-parallel partition suggestions

These are starting points, not an accepted distributed implementation:

| Weight | Suggested partition |
|---|---|
| embedding | vocabulary parallel or replicated |
| `r/k/v_proj` | column parallel over `A/H` |
| `o_proj` | row parallel over `A/H`, all-reduce residual output |
| W/A/G/V down | replicated or row/column plan with explicit collective |
| W/A/G/V up | shard output over local heads where possible |
| `ffn.key` | column parallel over `F` |
| `ffn.value` | row parallel over `F`, all-reduce |
| `lm_head` | vocabulary parallel |

Partition recurrent state by attention heads:

```text
S_local [slots,H/TP,N,N]
```

Require `H % TP == 0` for this plan.

The low-rank paths can make naive sharding communication-heavy. Validate
collective placement against a single-rank layer before optimization.

## Pipeline-parallel partition

Assign complete consecutive layers to stages. Each stage loads only its layer
weights and owns corresponding state-pool slices. First and last stages own
embedding and final norm/head respectively, subject to the engine's standard
layout.

Transfer `v_first` with the hidden activation for the current token.

## Tokenizer contract

Converted models use the official RWKV trie vocabulary. Do not call
`resize_token_embeddings`; vocabulary size is fixed.

Preserve:

```text
rwkv_vocab_v20230424.txt
token IDs without implicit special-token insertion
padding/eos conventions from tokenizer_config.json
```

Prefix-cache keys must contain token IDs, not only input strings.

## Quantized checkpoints

If the vLLM project introduces prepacked files, keep dense and packed
identities separate:

```text
source checkpoint hash
packer/layout version
quantization algorithm
group/row granularity
module allowlist
kernel compatibility
quality calibration/evaluation record
```

See
[`../quantization/VLLM_QUANTIZATION_PORTING.md`](../quantization/VLLM_QUANTIZATION_PORTING.md).
