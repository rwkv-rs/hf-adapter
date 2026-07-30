# RWKV7 kernel plugin API

`rwkv7-hf` 1.0 freezes one optional-backend boundary. Model repositories own
configuration, model structure, tokenizer, public cache tensors, loss, and
Transformers outputs. An independently installed backend owns hardware policy
and optimized tensor execution.

The only package-level entrypoint is:

```python
rwkv7_kernels.execute_optional_v4(kind, *args, **kwargs)
```

API version 4 accepts exactly five operation kinds:

```text
training_program
model_forward
linear_training
mix6_training
recurrent
```

Every call returns exactly these envelope fields:

```text
api_version, kind, supported, implementation, reason, result, phase
```

An unsupported request returns `supported=False` and `result=None`. A negative
probe is side-effect free. `auto` may then execute the readable reference
operation; strict `optimized` reports an error. Once a positive operation
starts, malformed output or an execution exception fails closed and is never
recomputed against a possibly modified cache.

The public recurrent state remains `[B,H,K,V]`. Plugins may use private layouts
internally but may not replace the HF model class, configuration, tokenizer,
cache class, or output objects. Package-free Hub models remain valid when no
plugin is installed.

The machine-readable contract is shipped as
`rwkv7_kernels/KERNEL_PLUGIN_API.json`. API-v4 operation names, envelope fields,
cache layout, and failure semantics are immutable for the 1.0 release line.

The exact executable inputs for the validated 1.0.0 wheel pair are recorded in [`RELEASE_SOURCE_FREEZE.json`](../RELEASE_SOURCE_FREEZE.json). CI recomputes every listed SHA-256 and rejects added, removed, or modified package files. Changing those bytes requires an explicit thaw, a new versioned manifest, and the complete hardware acceptance matrix again. Third-party and future backends remain replaceable because only this contract—not any private implementation module—is imported by the HF core.
