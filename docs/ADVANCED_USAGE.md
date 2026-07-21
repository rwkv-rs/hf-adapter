# RWKV-7 advanced workflow guide

This guide starts after the ordinary first-generation path in
[`USER_GUIDE.md`](USER_GUIDE.md) has passed. It provides copyable acceptance
commands for speculative decoding, single-GPU training, multi-GPU inference,
and DeepSpeed multi-GPU training.

Chinese version: [`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md)

For conversion/cache, the full training ecosystem, quantization, and Apple
workflows, use the complete map in
[`COMPLETE_ADAPTER_GUIDE.md`](COMPLETE_ADAPTER_GUIDE.md).

These commands are short compatibility smokes. A smoke pass does not by itself
prove a speedup, production convergence, tensor parallelism, or long-run
stability.

## Common preflight

Use a converted 0.1B or 0.4B model first. Activate the repository virtual
environment and check the installation:

```bash
python examples/check_environment.py --model /path/to/model-hf
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

Install training and distributed dependencies only when those sections need
them:

```bash
python -m pip install -e ".[train]"
```

## 1. Speculative decoding

RWKV speculative decoding uses a draft model to propose token blocks and the
target model to verify them. Greedy output must remain identical to normal
target generation.

First prove the API and correctness contract with the target also acting as the
draft:

```bash
python tests/test_speculative_decode.py \
  --model /path/to/target-model-hf \
  --device cuda \
  --dtype fp16 \
  --max-new-tokens 8 \
  --draft-tokens 4
```

Success prints `speculative_stats`, decoded text, and `PASS`. The same-model
run should have `acceptance_rate=1.0` and zero corrections. It is a correctness
check, not a speed claim.

Then provide a smaller converted RWKV-7 draft model using the same tokenizer
and adapter contract:

```bash
python tests/test_speculative_decode.py \
  --model /path/to/target-model-hf \
  --draft-model /path/to/smaller-draft-model-hf \
  --device cuda \
  --dtype fp16 \
  --max-new-tokens 32 \
  --draft-tokens 4
```

The smaller-draft run passes when output still equals target greedy generation.
A real performance claim additionally requires paired timing against normal
target generation on the same device and shape.

### Optional: align a smaller draft

Create a UTF-8 text file with one representative prompt per line. Train only
the smaller draft against a frozen target, then save the merged draft:

```bash
python scripts/train_spec_draft.py \
  --target /path/to/target-model-hf \
  --draft /path/to/smaller-draft-model-hf \
  --prompts /path/to/prompts.txt \
  --output /path/to/aligned-draft-hf \
  --device cuda --dtype fp16 --epochs 1 --gen-tokens 64
```

Training must exit 0, print finite loss telemetry, and print
`saved_aligned_draft`. That is traceability, not speculative acceptance.
Measure the resulting draft against normal target generation:

```bash
python bench/bench_speculative_decode.py \
  --target-model /path/to/target-model-hf \
  --draft-model /path/to/aligned-draft-hf \
  --draft-tag trained --device cuda --dtype fp16 \
  --max-new-tokens 32 --draft-tokens 4
```

Require `status: pass`, exact target-greedy equality, and a paired
`speedup_vs_target_generate > 1` before describing the trained draft as a
speed improvement. Keep the original off-the-shelf draft for an A/B control.

## 2. Single-GPU PEFT and Trainer smoke

Start with a short backward pass before using a real dataset. This catches
unsupported training paths and out-of-memory conditions cheaply.

Run the LoRA backward smoke:

```bash
python tests/test_peft_lora.py \
  --model /path/to/model-hf \
  --device cuda \
  --attn-mode fused_recurrent
```

Success requires a finite loss, `nonzero_grad_count` greater than zero, and exit
code 0. The canonical model is native. If the card/dtype-specific fused training
path is unavailable, use the portable native Trainer smoke:

```bash
python tests/test_native_trainer_smoke.py \
  --model /path/to/model-hf \
  --dtype fp32 \
  --max-steps 2 \
  --batch-size 2 \
  --length 32
```

Success prints `NATIVE TRAINER PASS`, a decreasing short loss history, and at
least one updated trainable parameter. If memory is insufficient, use the 0.1B
checkpoint or reduce batch size and length before changing precision.

These tests use fixed tiny prompts and do not save a production adapter. A real
fine-tune still needs a reviewed dataset, train/evaluation split, checkpoint
directory, resume policy, evaluation metrics, and retained loss logs. The
validated ecosystem scope is summarized in [`TRAINING.md`](TRAINING.md).

## 3. Multi-GPU inference with `device_map`

The repository provides an HF layer-sharding smoke for the pipeline-parallel
direction. It verifies generation across two visible CUDA devices and can
compare against a single-device reference.

Linux or WSL2:

```bash
CUDA_VISIBLE_DEVICES=0,1 python tests/test_device_map_generate.py \
  --model /path/to/model-hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --max-new-tokens 4 \
  --compare-single-device
```

Windows PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES="0,1"
python tests\test_device_map_generate.py --model C:\path\to\model-hf --dtype fp16 --attn-mode fused_recurrent --max-new-tokens 4 --compare-single-device
```

Success prints `PASS`. This proves HF device placement and output parity for
the tested model. It is not native tensor parallelism, and cross-device layer
handoff may be slower for small models.

Cross-GPU recurrent-state handoff defaults to CPU staging because some
virtualized CUDA hosts falsely advertise working peer access and silently
corrupt larger P2P copies. On a host where direct CUDA P2P/NVLink has been
validated independently, opt into the faster path with
`RWKV7_DEVICE_MAP_TRANSFER=p2p`. Use `RWKV7_DEVICE_MAP_TRANSFER=cpu` to force
the conservative path explicitly.

## 4. Multi-GPU training with DeepSpeed ZeRO

DeepSpeed ZeRO partitions training state. ZeRO-2 partitions optimizer state and
gradients; ZeRO-3 also partitions parameters. Run this path on Linux or WSL2
with at least two visible CUDA GPUs.

Validate the repository presets first:

```bash
python tests/test_deepspeed_configs.py
```

Run both one-step distributed smokes:

```bash
NPROC_PER_NODE=2 \
ZERO_STAGE=both \
MODEL=/path/to/model-hf \
TRAIN_DTYPE=fp16 \
RESULTS=bench/results.jsonl \
bash scripts/run_zero_training_smoke.sh
```

Success requires exit code 0, `PASS` rows for the requested stages, and result
rows written to `bench/results.jsonl`. ZeRO smoke evidence must not be described
as tensor-parallel inference evidence. One step also does not prove long-run
convergence, checkpoint continuity, or optimizer/scheduler/RNG resume fidelity.

## 5. Run this with an AI assistant

Use the single task template in [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md)
and select speculative decoding, multi-GPU inference, or DeepSpeed training.
This page intentionally does not maintain a second AI prompt.
