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

W&B is off by default. Enable it with `--report-to wandb`; local artifacts
remain authoritative and no token is written to disk.
