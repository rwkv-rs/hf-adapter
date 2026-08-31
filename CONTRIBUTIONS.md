# Contribution history

The RWKV-7 Hugging Face 1.0 source line is considered published only when its
release claims point to source, tests, evaluation tooling, and immutable result
bundles that satisfy [`HF_STATUS.md`](HF_STATUS.md). Candidate implementation
work must not be described as a published release before those gates complete.

## 1.0 implementation

Wang Yue (`@123123213weqw`; historical aliases `wangyue`, `wy`, and
`dsadsasdaddas`) is the lead architect and primary implementer. The candidate
work includes:

- a readable RWKV-7 Transformers implementation with explicit configuration,
  canonical recurrent cache, PyTorch operator boundary, model, causal-LM, and
  tokenizer ownership;
- a clean six-module `rwkv7_hf/` runtime, with converter, CLI, manifest, and
  smoke utilities moved to the sibling `rwkv7_hf_tools/` package;
- an optional, separately installable `rwkv7-kernels` package for accelerated
  inference, training, graph, and quantization routes;
- compatibility and reproducibility harnesses for Transformers, Trainer,
  Accelerate, PEFT, TRL, SFT/DPO/GRPO, FLA diagnostics, and `lm_eval`;
- release tooling for immutable wheels, device evidence, GitHub, PyPI, and six
  Hugging Face model repositories.

The layout is “Mamba-style” only in its separation of model, cache, readable
operators, and optional optimized backends. It does not imply copied Mamba
math, source, state representation, or implementation.

The executable work is in [`tests/`](tests/), [`evaluation/`](evaluation/),
[`benchmarks/`](benchmarks/), and
[`examples/finetune/`](examples/finetune/). The final hardware, evaluation,
Hub, and publication evidence required for 1.0 is defined in
[`HF_STATUS.md`](HF_STATUS.md) and recorded by the release provenance.

## Published 0.9 history

Version 0.9.0 established the readable pure-PyTorch Hugging Face reference
line, checkpoint conversion, ecosystem tests, finetuning examples, and the
six-model Hub publication workflow. Its archived evidence remains valid for
that release, but it is not evidence that the current 1.0 candidate has passed
its final immutable-artifact gates.

## Attribution

AI assistants and review bots are tooling, not separate human reward
recipients. The aliases above refer to the same human contributor. Other named
contributors remain distinct; see [`CONTRIBUTORS.md`](CONTRIBUTORS.md).
