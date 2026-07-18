# HF training ecosystem status

Canonical summary for Trainer, PEFT, TRL and DeepSpeed validation.

For copyable PEFT, adapter round-trip, Trainer/resume, SFT/DPO/GRPO, and matrix
commands, read [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md) or
[`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md). Multi-GPU ZeRO remains
in [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md).

## Interface status

| Workflow | Status | Validation level |
|---|---|---|
| Labels and causal LM loss | **PASS** | finite loss and parameter-update smoke |
| HF Trainer | **PASS** | tiny and real-model smoke; checkpoint resume evidence |
| Official `train_temp` alignment | **PASS for exact RTX 5090 B1 and Native B16 lanes** | BF16 12x768 T512 backward/step exact; Native B16 also passes 3-seed x 1,000-step, 500+500 resume and steady-memory gates |
| PEFT LoRA | **PASS** | forward/loss/backward, save/load and merge |
| TRL SFTTrainer | **PASS** | CUDA and Apple/MPS smoke |
| TRL DPOTrainer | **PASS** | CUDA and Apple/MPS smoke |
| TRL GRPOTrainer | **PASS** | CUDA and Apple/MPS smoke |
| DeepSpeed ZeRO-2 | **PASS for current matrix** | base and resume evidence on multiple CUDA setups |
| DeepSpeed ZeRO-3 | **PASS for current smoke matrix** | base plus selected resume paths |
| PP/TP training | **Not a completed claim** | ZeRO/device-map evidence does not equal full TP training support |

## Hardware/model evidence

- **V100:** 0.4B/1.5B/2.9B training ecosystem rows; dual-card ZeRO base/resume,
  including selected ZeRO-3 resume smoke.
- **A100 40GB:** Trainer/SFT/DPO and checkpoint resume through 7.2B; dual-card
  ZeRO-2/3 base and ZeRO-2 resume evidence.
- **A800 80GB:** single/dual-card ZeRO-2/3 base and resume evidence plus
  large-model inference/quant smoke.
- **RTX A6000 48GB:** Trainer/SFT/DPO/resume through tested 7.2B lanes; dual-card
  ZeRO-2/3 base and resume through 2.9B.
- **RTX 5090:** opt-in official-kernel `train_temp_cuda` lanes on the 12x768
  FFN3072 model. The original B1/T512 lane remains exact. The Native/no-FLA
  B16/T512 lane matches 399 gradients and 399 FusedAdam parameter deltas
  exactly, passes three seeds x 1,000 steps, restores model/optimizer/RNG across
  a 500+500 resume, and has zero reserved-memory growth across 20 steady
  samples. Native median training throughput is `0.9499x` official. See
  [`TRAIN_TEMP_CUDA.md`](TRAIN_TEMP_CUDA.md) and
  [`../bench/5090_native_train_temp_b16_20260718/`](../bench/5090_native_train_temp_b16_20260718/README.md).
- **Apple M5:** tiny and real-model PEFT/Trainer/TRL smoke through tested 1.5B
  workflows. This is compatibility evidence, not high-throughput training.

Detailed matrices:

- [`validation/V100_HF_VALIDATION.md`](validation/V100_HF_VALIDATION.md)
- [`validation/A100_HF_VALIDATION.md`](validation/A100_HF_VALIDATION.md)
- [`validation/A800_HF_VALIDATION.md`](validation/A800_HF_VALIDATION.md)
- [`hardware/APPLE_SILICON.md`](hardware/APPLE_SILICON.md)

## Remaining production work

- Extend exact `train_temp` alignment to larger checkpoints, real datasets,
  multi-day runs and additional cards; current accepted B1/B16 lanes are on one
  RTX 5090.
- Larger ZeRO-3 checkpoint-resume matrix.
- Optimizer/scheduler/RNG continuity checks after distributed resume; the
  single-GPU Native 500+500 path is already covered.
- H100 and AMD/ROCm training validation.
- Clear separation between compatibility smoke and production convergence
  evidence in every future report.
