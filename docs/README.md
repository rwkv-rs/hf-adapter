# Documentation

## Start here

- [Ordinary-user PyPI and public-model guide](USER_GUIDE.md) / [中文](USER_GUIDE_ZH.md)
- [Unified CLI reference](CLI.md)
- [Published RWKV7-G1 models](PUBLISHED_MODELS.md) / [中文](PUBLISHED_MODELS_ZH.md)
- [Runtime and kernel-policy doctor](KERNEL_DOCTOR.md) / [中文](KERNEL_DOCTOR_ZH.md)
- [Prebuilt CUDA kernel wheels](KERNEL_WHEELS.md) / [中文](KERNEL_WHEELS_ZH.md)
- [Project summary](PROJECT_SUMMARY.md)
- [Complete adapter guide](COMPLETE_ADAPTER_GUIDE.md)
- [Acceptance contract](ACCEPTANCE.md)
- [Hardware matrix](HARDWARE_MATRIX.md)
- [Performance guide](PERFORMANCE.md)
- [Current results index](RESULTS_INDEX.md)
- [Latest Qwen3.5 P/D table](QWEN35_LATEST_P_D_TOKPS.md)
- [Latest Qwen3.5 P/D table (English)](QWEN35_LATEST_P_D_TOKPS_EN.md)
- [Qwen3.5 GPU reproduction tutorial](QWEN35_SPEED_REPRODUCTION_ZH.md)
- [Qwen3.5 GPU reproduction tutorial (English)](QWEN35_SPEED_REPRODUCTION.md)

## Use and integration

- [Inference workflows](INFERENCE_WORKFLOWS.md)
- [Training](TRAINING.md)
- [Training workflows](TRAINING_WORKFLOWS.md)
- [Quantization](QUANTIZATION.md)
- [Quantization usage](QUANTIZATION_USAGE.md)
- [Advanced usage](ADVANCED_USAGE.md)
- [Transformers tensor parallel](integrations/HF_TENSOR_PARALLEL.md)

## Hardware

- [Apple Silicon](hardware/APPLE_SILICON.md)
- [Apple production close](hardware/APPLE_PRODUCTION_CLOSE.md)
- [Blackwell 50-series](hardware/BLACKWELL_50SERIES.md)
- [AMD ROCm](validation/AMD_ROCM_HF_VALIDATION.md)
- [Moore Threads MUSA](hardware/MUSA.md)
- [Huawei Ascend](hardware/HUAWEI_ASCEND.md)
- [Biren](hardware/BIREN_BR106M.md)
- [MetaX](hardware/METAX_C500.md)

## Development

The documents below assume a repository clone and may intentionally use
editable installs. Ordinary inference starts from the user guide above.

- [Repository layout](architecture/REPOSITORY_LAYOUT.md)
- [Native default architecture](architecture/NATIVE_DEFAULT_BACKEND.md)
- [Fused backend roadmap](performance/FUSED_BACKEND.md)
- [Apple validation contributor guide](contributing/APPLE_VALIDATION.md)
- [Contributing](../CONTRIBUTING.md)

Historical benchmark inventories and superseded plans are intentionally not
kept in the current repository. See
[`../bench/CURRENT_ARTIFACTS.json`](../bench/CURRENT_ARTIFACTS.json) for the
retained evidence boundary.
