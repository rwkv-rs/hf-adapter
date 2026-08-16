#!/usr/bin/env bash
# Capture the strict RTX 3090 RWKV matrix and independent FLA correctness probes.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROTOCOL=qwen35_3090_paired_pd_v2
export CORRECTNESS_PROTOCOL=rwkv_native_graph_fla_correctness_3090_v2
export EXPECTED_GPU_NAME="NVIDIA GeForce RTX 3090"
export EXPECTED_PYTHON=3.10.12
export TORCH_CUDA_ARCH=8.6
export ROUTE_PROFILE=sm86_qwen_alignment
export SMALL_B8_MODE=sm86_qwen_alignment
export SPLIT_7P2_B8=0
unset ADA_WAGV_BMM_OVERRIDE

exec bash "${ROOT}/bench/run_4090_rwkv_paired_pd_v2.sh" "$@"
