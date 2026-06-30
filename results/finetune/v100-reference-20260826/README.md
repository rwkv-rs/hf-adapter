# RWKV-7 HF 0.9 canonical fine-tuning — V100

Release validation: **PASS** (`validation.json`).

| Run | Dataset | Steps | Result | Source revision |
|---|---|---:|---|---|
| LoRA SFT | `HuggingFaceH4/ultrachat_200k` | 100 | pass | `117a53a4257f537cd3b03d94c0bcb6448cf0d4bd` |
| LoRA DPO | `HuggingFaceH4/ultrafeedback_binarized` | 100 | pass | `117a53a4257f537cd3b03d94c0bcb6448cf0d4bd` |
| LoRA GRPO | `openai/gsm8k` | 100 | pass | `bf7919f62fba1688e0beb4d4f745637fe9907aea` |
| SFT resume | SFT checkpoint 100 -> 101 | 101 | pass | `ebdbb1d9451ac8568838b75d6a9e3d7f68e5b35f` |
| W&B offline smoke | UltraChat deterministic subset | 1 | pass | `117a53a4257f537cd3b03d94c0bcb6448cf0d4bd` |

All three canonical methods recorded finite loss, nonzero gradients, 144
changed LoRA parameters, adapter save/reload max-absolute difference `0.0`,
resolved dataset/model revisions, deterministic sample indices, environment,
metrics, exit status, and checkpoint SHA256 inventories. The offline W&B run
recorded run ID `wy328yot`; local files remain authoritative.

SFT, DPO, GRPO, and the W&B smoke used a Tesla V100 32 GB with PyTorch
2.5.1+cu124, Transformers 4.56.2, TRL 0.20.0, and PEFT 0.19.1. Exact optimizer
resume used PyTorch 2.6.0+cu124 because Transformers 4.56.2 intentionally
blocks `torch.load` checkpoint restoration on older PyTorch versions.

`pipeline_status.txt` and `repair_status.txt` retain the exploratory attempt
history rather than hiding it. The release runs in `validation.json` supersede
those interrupted attempts: an uncached 256-token GRPO rollout was stopped as
impractical, the first dense reward saturated and produced zero advantage, and
the first resume attempt hit the documented PyTorch safety gate. Their fixes
are the 64-token cached inference-only rollout, nonsaturating dense reward, and
PyTorch 2.6 resume environment validated here.

Git stores the complete top-level metadata and logs but omits adapter and
checkpoint payloads. `checkpoint_inventory.json` records their hashes; the
full directory is packaged as a GitHub release artifact.
