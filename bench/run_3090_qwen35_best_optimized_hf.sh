#!/usr/bin/env bash
# Run one Qwen3.5 checkpoint through the exact RTX 3090 reference contract.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export EXPECTED_GPU_MODEL=3090
export TORCH_CUDA_ARCH_LIST=8.6
export QWEN_CROSS_CACHE_FULL_GREEDY_POLICY=informational
exec "${ROOT}/bench/run_5090_qwen35_best_optimized_hf.sh" "$@"
