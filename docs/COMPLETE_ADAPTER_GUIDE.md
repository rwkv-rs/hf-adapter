# Complete adapter tutorial index

This page is the teaching-coverage contract for the RWKV-7 Hugging Face
adapter. Every implemented or promoted user-facing capability must have a
copyable tutorial, an observable success gate, and an explicit boundary.

Chinese version: [`COMPLETE_ADAPTER_GUIDE_ZH.md`](COMPLETE_ADAPTER_GUIDE_ZH.md)

Start with [`USER_GUIDE.md`](USER_GUIDE.md). Use the table below only after a
normal eight-token generation succeeds.

## Tutorial map

| User goal | Tutorial | Acceptance evidence | Current boundary |
|---|---|---|---|
| Install, inspect the environment, download, convert, and generate | [`USER_GUIDE.md`](USER_GUIDE.md) | `RESULT: READY`, model-directory `PASS`, generated text | Start with 0.1B/0.4B; FLA is optional |
| Convert one, many, or a large checkpoint; save/reload; run offline | [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md) | conversion exit 0, manifest success, `test_reload_roundtrip.py` prints `PASS` | `--low-memory` reduces conversion RAM, not inference VRAM |
| Use `AutoModelForCausalLM`, loss, masks, and the native no-FLA backend | [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md) | API command exits 0; native smoke prints its documented pass marker | Native is the portable compatibility route; exact-card performance varies |
| Reuse recurrent state, run batch cache, dynamic batching, and chunked prefill | [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md) | each focused cache/prefill test prints `PASS` | These are HF serving primitives, not a complete serving engine |
| Run speculative decoding or align a smaller draft | [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md) | exact target-greedy parity and `PASS`; trained draft requires paired speed | Same-model draft and successful draft training do not prove speed |
| Run multi-GPU `device_map` inference | [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md) | sharded/single-device parity and `PASS` | Layer placement is not native tensor parallelism |
| Run PEFT LoRA, adapter save/load/merge, Trainer, and resume | [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md) | finite loss/gradients and workflow-specific `PASS` marker | Tiny smokes do not prove production convergence |
| Run TRL SFT, DPO, and GRPO | [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md) | `NATIVE SFT/DPO/GRPO PASS` | Fixed smoke datasets are not training recipes |
| Run DeepSpeed ZeRO-2/ZeRO-3 | [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md) | requested result rows are `PASS` | Requires Linux/WSL2 and 2+ CUDA GPUs; resume coverage is matrix-specific |
| Use bitsandbytes W8/W4 or native MM8/MM4 | [`QUANTIZATION_USAGE.md`](QUANTIZATION_USAGE.md) | generation, finite logits, parity, and footprint checks pass | A functional or smaller model is not automatically faster |
| Run on Apple MPS, MLX, packed W8/W4, sessions, or CoreML | [`APPLE_USAGE.md`](APPLE_USAGE.md) | workflow-specific JSON/pass output | Promoted performance is exact M5/shape evidence, not all Apple devices |
| Ask an AI coding assistant to execute a workflow | [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) | exact command, exit code, device/model path, and pass marker reported | Never provide passwords, private tokens, or SSH keys |
| Reproduce hardware/performance claims | [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md), [`PERFORMANCE.md`](PERFORMANCE.md), and [`../bench/INDEX.md`](../bench/INDEX.md) | same-card, same-shape raw evidence plus documented gates | Benchmark evidence is narrower than API compatibility |

## What is deliberately not presented as a completed adaptation

- Native vLLM and SGLang integrations are outside this repository's HF-only
  scope.
- `device_map` and DeepSpeed ZeRO do not establish production tensor or
  pipeline parallel engines.
- Turing, Hopper, and AMD policy entries without exact-card rows are routing
  preparation, not validated support claims.
- Universal full-memory W8/W4 speed is still open. Use an exact-card accepted
  policy or treat quantization as a compatibility/memory path.
- A smoke test proves that an interface executes and preserves its local
  contract. It does not prove model quality, convergence, capacity, or a speed
  win.

## The six fields every new tutorial must contain

When a new adaptation is added, update this index and document:

1. prerequisites and supported environment;
2. a smallest safe model or input;
3. a copyable user command or API example;
4. the exact observable success gate;
5. failure recovery and current limitations;
6. an AI-assistant instruction that forbids guessing and requires evidence.

An implementation hidden only in source, tests, benchmark logs, or a pull
request is not considered documented for ordinary users.
