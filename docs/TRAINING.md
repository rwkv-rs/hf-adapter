# HF training ecosystem status

Canonical summary for Trainer, PEFT, TRL and DeepSpeed validation.

For copyable single-GPU and multi-GPU smoke commands with visual explanations,
read [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md) or
[`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md).

## Interface status

| Workflow | Status | Validation level |
|---|---|---|
| Labels and causal LM loss | **PASS** | finite loss and parameter-update smoke |
| HF Trainer | **PASS** | tiny and real-model smoke; checkpoint resume evidence |
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
- **Apple M5:** tiny and real-model PEFT/Trainer/TRL smoke through tested 1.5B
  workflows. This is compatibility evidence, not high-throughput training.

Detailed matrices:

- [`validation/V100_HF_VALIDATION.md`](validation/V100_HF_VALIDATION.md)
- [`validation/A100_HF_VALIDATION.md`](validation/A100_HF_VALIDATION.md)
- [`validation/A800_HF_VALIDATION.md`](validation/A800_HF_VALIDATION.md)
- [`hardware/APPLE_SILICON.md`](hardware/APPLE_SILICON.md)

## Remaining production work

- Longer runs with loss curves, throughput and memory telemetry.
- Larger ZeRO-3 checkpoint-resume matrix.
- Optimizer/scheduler/RNG continuity checks after distributed resume.
- H100 and AMD/ROCm training validation.
- Clear separation between compatibility smoke and production convergence
  evidence in every future report.
