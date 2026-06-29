# Contribution history

This branch is the readable RWKV-7 Hugging Face reference line. Claims about
`main` must point to the reference source, tests, evaluation tooling, or a
published immutable result bundle.

## 0.9 reference line

Wang Yue (`@123123213weqw`; historical aliases `wangyue`, `wy`, and
`dsadsasdaddas`) is the lead architect and primary implementer. The 0.9 work
includes:

- the pure-PyTorch `RWKV7TimeMix`, `RWKV7ChannelMix`, block, model, causal-LM,
  and canonical `RWKV7Cache` implementation;
- official checkpoint conversion into self-contained Hugging Face model
  directories;
- Transformers cache, padding, loss, generation, save/reload, Trainer, PEFT,
  and TRL compatibility tests;
- non-blocking optimized-backend diagnostics, the formal `lm_eval` matrix, and direct
  LoRA SFT/DPO/GRPO reproducibility examples;
- release documentation and six-model Hub publication tooling.

The executable evidence for these claims is in [`tests/`](tests/),
[`evaluation/`](evaluation/), [`benchmarks/`](benchmarks/), and
[`examples/finetune/`](examples/finetune/).
Release readiness is tracked in [`HF_STATUS.md`](HF_STATUS.md).

## Preserved performance history

The pre-0.9 CUDA, Triton, JIT, CUDA Graph, quantization, platform backends,
benchmarks, and exact-card evidence are preserved without modification on
`perf/native-kernels-v0.8` at commit `1014acf`. They are deliberately not part
of the readable reference implementation and should be cited from that branch
or from their original Git tags.

## Attribution

AI assistants and review bots are tooling, not separate human reward
recipients. The identities listed above refer to the same human contributor.
Other named contributors remain distinct; see [`CONTRIBUTORS.md`](CONTRIBUTORS.md).
