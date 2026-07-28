# RWKV-7 HF 1.0 status and release gates

The 1.0 source line is a release candidate until the immutable GitHub, Hugging
Face Hub, and PyPI artifacts have passed the machine-verified gates below.
Publication status is determined by those tagged artifacts and their attached
`release-provenance.json`, not by manually changing this document.

## Current ownership and layout

The organization follows the same separation-of-responsibilities idea used by
clean state-space-model integrations such as Mamba. “Mamba-style” describes
only the package boundaries; RWKV-7 math, state semantics, names, and code are
implemented independently here.

`rwkv7_hf/` owns the readable Hugging Face model and contains six core Python
modules:

1. `__init__.py`
2. `configuration_rwkv7.py`
3. `cache_rwkv7.py`
4. `ops_rwkv7.py`
5. `modeling_rwkv7.py`
6. `tokenization_rwkv7.py`

`chat_template.jinja` is the accompanying tokenizer asset. Conversion and
maintenance commands live in the sibling `rwkv7_hf_tools/` package rather than
inside the model package. Optional accelerated inference, training, graph, and
quantization implementations live in the separately installable
`rwkv7-kernels` package under `kernels/`. Installing that package must not make
configuration, cache, or model ownership ambiguous; the readable PyTorch path
remains available without it.

The only plug-in entrypoint is the frozen API-v4
`rwkv7_kernels.execute_optional_v4` contract. Its operation names, exact result
envelope, canonical cache layout, and fallback/fail-closed behavior are defined
in [`docs/KERNEL_PLUGIN_API.md`](docs/KERNEL_PLUGIN_API.md) and shipped in the
kernel wheel as `KERNEL_PLUGIN_API.json`.

## Implemented in the candidate

- [x] readable Transformers config, cache, model, causal-LM, tokenizer, and
      PyTorch operator boundary
- [x] standard cache, padding, generation, loss, save/reload, and training
      interfaces
- [x] sibling converter/CLI/manifest/smoke tooling
- [x] optional `rwkv7-kernels` package with explicit dispatch and reference
      fallback boundaries
- [x] evaluation and reproducibility harnesses for inference, training,
      finetuning, FLA diagnostics, and `lm_eval`

## Required release evidence

The release workflow requires all of the following and fails closed when any
record is missing or inconsistent:

- one source revision and immutable `rwkv7-hf` / `rwkv7-kernels` artifacts,
  with matching recorded SHA256 values;
- final RTX 4080 and RTX 4090 acceptance bundles from the same artifacts;
- the validated 144-unit reference/optimized/FLA `lm_eval` matrix;
- final-wheel SFT, DPO, and GRPO reproducibility gates;
- six tagged Hugging Face model repositories and Hub re-download smoke tests;
- verified GitHub 1.0 release assets and both PyPI projects.

Existing development and diagnostic runs are useful evidence, but they do not
replace the immutable-artifact records above.

## Published history

Version 0.9.0 is the previously published reference release. Its six-model Hub
and PyPI evidence remains archived under
[`results/release/hf-v0.9.0-v100`](results/release/hf-v0.9.0-v100/README.md).
That evidence documents v0.9.0 only and must not be presented as completion of
the 1.0 candidate.
