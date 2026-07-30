# LoRA fine-tuning

The three examples are direct TRL programs rather than a private training
framework. They all use the 0.1B model, seed 42, sequence length 512, LoRA
r=8 / alpha=16 / dropout=0.05 and target
`r_proj,k_proj,v_proj,o_proj,key,value`.
The canonical reference run uses gradient accumulation 1 so the deliberately
unfused PyTorch recurrence remains bounded; use
`--gradient-accumulation-steps` to increase the effective batch.
The canonical environment uses `transformers==4.56.2` and `trl==0.20.0`
(`pip install -e '.[train]'`). This combination retains V100 support with the
validated PyTorch 2.5 CUDA build.
GRPO selects TRL's native Transformers generation path and masks an unrelated
system-wide vLLM installation before importing `GRPOTrainer`. This prevents an
incompatible optional vLLM from breaking a run that does not request vLLM.
Training from scratch works with PyTorch 2.5, while restoring a Trainer
checkpoint that contains optimizer state requires PyTorch 2.6 or newer under
this Transformers version. The requirement comes from Transformers' secure
`torch.load` gate; the canonical resume check uses PyTorch 2.6.0+cu124.

## SFT

Dataset: [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k),
revision `8049631c405ae6576f93f445c6b8166f76f5505a`.

```bash
python examples/finetune/sft_lora.py \
  --output-dir results/finetune/sft
```

## DPO

Dataset: [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized),
revision `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`.

```bash
python examples/finetune/dpo_lora.py \
  --output-dir results/finetune/dpo
```

## GRPO

Dataset: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k),
revision `740312add88f781978c0658806c59bc2815b9866`. The script includes
answer extraction, an exact correctness reward, and a small format/diversity
reward so a cold-start 0.1B model does not receive zero advantage forever.
The 512-token context reserves 64 tokens for each sampled completion by
default; change this explicitly with `--max-completion-length`.

```bash
python examples/finetune/grpo_lora.py \
  --output-dir results/finetune/grpo
```

All examples disable cache, enable gradient checkpointing, save checkpoints and
accept `--resume-from-checkpoint`. They fail if loss is non-finite, gradients
never become non-zero, trainable parameters do not change, or an adapter reload
changes logits. Every run stores deterministic sample indices, resolved config,
environment, JSONL metrics, final metrics, checkpoint hashes and W&B metadata.
The small parent launcher also records `stdout.log`, `stderr.log`, and
`exit_status.json`, including failed runs.

The examples select one explicit model dtype and disable a second Trainer AMP
layer. Gradient checkpointing uses PyTorch's non-reentrant implementation so
the recomputation forward remains an ordinary differentiable request. These
settings prevent autocast or a legacy no-grad checkpoint probe from silently
changing the optional training route. PEFT keeps adapters in FP32 by default;
pass `--torch-dtype bfloat16 --lora-dtype model` only when validating the
optional BF16 training leaves. PEFT may promote restored adapter matrices to
FP32 while loading a Trainer checkpoint; the reload check detects the actual
trained adapter dtype and recreates that same runtime before comparing logits.

W&B is off by default. Enable it with `--report-to wandb`; local artifacts
remain authoritative and no token is written to disk.

## Optional backend acceptance

The examples remain ordinary TRL programs when the optional kernel wheel is
installed. A formal backend run supplies the exact wheel paths so their hashes
are recorded with the dataset/model revisions:

```bash
export RWKV7_BACKEND=auto
export RWKV7_KERNEL_IMPL=auto
export RWKV7_MODEL_KERNEL_IMPL=auto
export RWKV7_TRAINING_KERNEL_IMPL=auto

python examples/finetune/sft_lora.py \
  --model /models/rwkv7-0.1b-hf \
  --output-dir results/backend-v2/finetune/sft \
  --code-sha "$(git rev-parse HEAD)" \
  --hf-wheel /artifacts/rwkv7_hf-1.0.0-py3-none-any.whl \
  --kernel-wheel /artifacts/rwkv7_kernels-1.0.0-py3-none-any.whl \
  --torch-dtype bfloat16 \
  --lora-dtype model
```

Run DPO and GRPO with the same artifact arguments, then validate all canonical
runs with:

```bash
python evaluation/validate_finetune_runs.py \
  --result-dir results/backend-v2/finetune \
  --require-backend-v2-routes \
  --require-training-candidate adaptive
```

One immutable `RWKV7ExecutionContext` selects the program at the differentiable
model boundary and is passed explicitly through non-linear layer boundaries
and checkpoint replay. Two narrow routing bridges carry the resolved value:
one transfers decoder context to the LM head across the standard output
boundary, while lexical `linear_execution_context` preserves the standard
`nn.Linear`/PEFT `forward(x)` contract and republishes the same value during
replay. Two evidence-only context-local snapshots and a versioned process-wide counter
preserve every optional leaf that actually executed. The counter matters for
DPO, whose differentiable policy pass is followed by a no-grad reference pass.
Standard HF training always retains the readable
`torch-reference-model-v1` layer loop; there is no whole-model training
dispatch. This keeps PEFT/TRL wrappers, hooks, gradient checkpointing, masking,
and the ordinary autograd graph visible to the framework.

The optional package can certify recurrent, flattened-linear, and Mix6 as one
adaptive program for the dense B4/T128 domain. This changes only tensor leaves:
PEFT/TRL still see the readable model loop, hooks, adapters, checkpoint replay,
and ordinary PyTorch autograd. Shapes outside the certificate use the
individually gated exact-matrix/reference leaves; frozen embeddings that make
autograd eligibility unprovable at model entry select one complete reference
program. Formal SFT/DPO/GRPO must record process-wide leaf counts as well as the
final route because DPO and GRPO may legitimately combine optimized
differentiable passes with reference no-grad or sampled-shape passes. Finite
loss/gradients, changed parameters, and adapter save/reload remain mandatory.
