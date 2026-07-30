# Architecture

RWKV-7 follows the same Mamba-style source separation used by clean state-space
Hugging Face implementations: configuration describes architecture, cache owns
public state, modeling exposes readable layers and HF outputs, and the operator
file contains the reference recurrence plus one optional implementation
boundary. "Mamba-style" describes the file and ownership boundaries only;
RWKV-7 keeps its own recurrence and checkpoint semantics.

## Canonical Hugging Face files

Every converted model repository is self-contained. Its four architecture
modules are:

1. `configuration_rwkv7.py` — architecture-only `RWKV7Config`;
2. `cache_rwkv7.py` — canonical recurrent cache lifecycle and batch operations;
3. `ops_rwkv7.py` — pure-PyTorch recurrence and the minimal API-v4 facade;
4. `modeling_rwkv7.py` — TMix, CMix, blocks, backbone, and causal LM.

`tokenization_rwkv7.py`, the vocabulary, tokenizer configuration, chat
template, `config.json`, and safetensors complete the Hub model. The installed
source tree keeps the modules under `rwkv7_hf/`; CLI, conversion, manifest, and
smoke-test code lives in the sibling `rwkv7_hf_tools/` package and is never
copied into model repositories.

The main package is package-free by design: Torch and Transformers are enough
to load a converted Hub directory. FLA, Triton, a compiler, `rwkv7-hf`, and
`rwkv7-kernels` are not runtime requirements for its reference path.

## Readable model program

`RWKV7TimeMix` computes shifts and R/W/K/V/A/G projections, calls
`rwkv7_recurrent`, applies GroupNorm and the direct RKV term, then projects
back to the residual width. `RWKV7ChannelMix` is the shifted squared-ReLU FFN.
`RWKV7Block` exposes both residual paths directly, and `RWKV7Model` retains the
ordinary Python layer loop. Loss shifting and the LM head remain visible in
`RWKV7ForCausalLM`.

For each model call, `RWKV7Model` resolves one frozen
`RWKV7ExecutionContext`. It contains semantic facts such as mask activity,
state provenance, autograd eligibility, token alignment, and a reserved
training-program certificate field. Modeling passes the same object
explicitly through blocks, TMix/CMix, recurrence, Mix6, the LM-head boundary,
and gradient-checkpoint replay. It contains no device policy, kernel object,
parameter, optimizer, cache tensor, or graph runner.

There are two narrow routing exceptions to keyword propagation.
`_execution_context_capture` transfers the resolved context from decoder to LM
head across the standard Transformers output boundary. Standard `nn.Linear`,
PEFT, and quantization modules must keep `forward(x)`, so the model publishes
the already-resolved context through one lexical `linear_execution_context`
`ContextVar` around the layer loop and LM head, and
republishes the same scope inside checkpoint replay. Neither routing bridge makes
a new decision. Two separate context-local values store only last-route and
last-execution-context evidence and never select a route.

## State

For each layer `RWKV7Cache` stores recurrent state
`[batch, heads, key_dim, value_dim]`, attention shift `[batch, hidden]`, and FFN
shift `[batch, hidden]`. `v_first` is passed from layer zero to later layers
during one forward call and is never persisted.

False positions in a 2-D attention mask do not update recurrent or shift state,
so left and right padding are deterministic. The cache implements sequence
length, reorder, select, repeat, reset, detach, and device/dtype conversion
without graph runners, counters, hardware policy, or layout routing.

## One optional API-v4 boundary

Installing `rwkv7-kernels` does not add a second model class. The HF package
knows only `RWKV7_KERNEL_API_VERSION = 4` and
`execute_optional_v4(kind, ...)`. Five operation kinds cover the complete
optional surface:

1. `training_program` — the reserved atomic training-preflight boundary;
2. `model_forward` — accepted fused prefill/decode requests;
3. `linear_training` — stateless flattened projections;
4. `mix6_training` — explicit-shift six-way input mixing;
5. `recurrent` — inference or training recurrence.

Every response has `api_version`, `kind`, `supported`, `implementation`,
`reason`, `result`, and `phase`. Unsupported operations must return
`result=None`. A negative probe is side-effect-free, so `ops_rwkv7.py` may
continue through unchanged reference math. `model_forward` instead passes the
caller's canonical cache directly, allowing zero-copy CUDA Graph binding. Once
positive execution begins, an exception or malformed payload fails closed even
in `auto`; reference recomputation is unsafe because that cache may already be
bound or updated.
Capability probing, execution ordering, environment parsing, device/shape
policy, trace accounting, quantization, and implementation errors are owned by
the companion wheel, not by modeling/config/cache.

Training does not dispatch an opaque replacement model. The readable model
loop remains authoritative. With `RWKV7_TRAINING_KERNEL_IMPL=adaptive`, API v4
preflights the model-owned shape, mask, initial-state, autograd, alignment, and
head-dimension facts once and issues an immutable program certificate only for
the certified dense B4/T128 domain. The recurrent, linear, and Mix6 leaves then
revalidate their concrete tensors against that certificate. Any unexpected
decline fails the certified call closed. Other explicitly adaptive requests
use the separately gated exact-matrix/reference leaves. If the model boundary
cannot prove autograd eligibility (for example, frozen PEFT embeddings), the
immutable context selects one complete reference program. Strict `optimized`
requires the atomic certificate and fails at the model boundary outside its
domain.

The wheel may internally use Triton/CUDA recurrence, fused decode, DPLR or
self-chunk prefill, projection/norm/FFN/LoRA fusion, CUDA Graph/state pools,
SM70/Ada/Blackwell policies, quantization adapters, and training autograd.
Internal `[V,K]`, packed, pooled, or graph-static state is converted back to
canonical `[B,H,K,V]` before the facade returns. There is no monkeypatch,
optimized subclass, duplicate layer stack, native cache class, or hardware
field in `RWKV7Config`.

## Migration inventory

`nvidia/MIGRATION_MANIFEST.json` verifies all 102 historical NVIDIA
destinations: 86 byte-identical transfers and 16 declared clean-boundary
adaptations. `nvidia/CAPABILITY_INVENTORY.json` maps them to 16 runtime
families. `nvidia/SOURCE_SCOPE.json` classifies the complete 153-file
historical source tree as 86 byte-migrated, 26 adapted, 7 canonical-reference,
6 relocated/retired tooling, 27 separate-hardware, and 1 retired non-kernel
file. `nvidia/RECURRENT_SOURCE_SCOPE.json` separately reconstructs the later
recurrent-only source line. The wheel audit recomputes these identities from
the built ZIP.

Production promotion is tied to one immutable wheel pair. The current changed
API-v4 bytes still require the RTX 4080 correctness, route, HF/training, FLA,
speed, and lm_eval gates, followed by RTX 4090. Historical results are retained
but are not relabelled as evidence for the new wheel. The complete local suite must be rerun after the current source settles; local
tests are not GPU acceptance evidence.

## Numerical reproducibility

The readable model uses ordinary `torch.nn.functional.linear` and PyTorch
matrix multiplication. To make FP16 scores independent of framework batch
regrouping, `modeling_rwkv7.py` tiles batch and time into a fixed 128-row
linear shape and `ops_rwkv7.py` evaluates the direct recurrence independently
per sample. Padding rows are discarded. This changes neither the RWKV
equations nor checkpoint keys and introduces no device-specific reference
route.

## Hugging Face contract

`RWKV7ForCausalLM.forward` accepts `input_ids`, `inputs_embeds`, a 2-D
`attention_mask`, `past_key_values`, `labels`, `use_cache`,
`output_hidden_states`, `return_dict`, `cache_position`, and `logits_to_keep`.
Labels are shifted internally and `-100` is ignored. Cache is disabled while
training with gradient checkpointing. `output_attentions=True` raises
`NotImplementedError` because RWKV has no Transformer attention matrix to
return.
