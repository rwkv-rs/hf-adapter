#!/usr/bin/env bash
# AMD ROCm compatibility and baseline-performance validation for the fully
# native HF adapter. Exact-architecture policy may enable measured Triton/HIP
# decode fusions, while prefill/quant and every unmeasured AMD architecture
# fail closed. Model sync must select the decoupled native class.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

for arg in "$@"; do
  case "${arg}" in
    *=*) export "${arg}" ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

HF_DIR="${HF_DIR:-/workspace/models/rwkv7-g1d-0.1b-hf}"
OUT_DIR="${OUT_DIR:-bench/amd_rocm_validation_$(date +%Y%m%d_%H%M%S)}"
RESULTS="${RESULTS:-${OUT_DIR}/results.jsonl}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
TRAIN_DTYPE="${TRAIN_DTYPE:-bf16}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
PROMPT_TOKENS="${PROMPT_TOKENS:-128}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
CHUNKED_PROMPT_TOKENS="${CHUNKED_PROMPT_TOKENS:-256}"
CHUNK_SIZES="${CHUNK_SIZES:-32 64 128}"
SYNC_ADAPTER_CODE="${SYNC_ADAPTER_CODE:-1}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

run() {
  echo "+ $*" >&2
  "$@"
}

mkdir -p "${OUT_DIR}"

python - <<'PY' | tee "${OUT_DIR}/environment.log"
import json
import torch
from rwkv7_hf.kernel_policy import current_kernel_policy

assert torch.version.hip, "ROCm PyTorch build required"
assert torch.cuda.is_available(), "ROCm GPU is not visible; check /dev/kfd and render-group permissions"
policy = current_kernel_policy(torch_module=torch)
assert policy.profile.family == "amd_hip", policy.profile
if policy.profile.architecture == "gfx1100":
    assert policy.fused_recurrent_output
    assert policy.fused_recurrent_raw
    assert policy.fused_output
    assert policy.fused_norm_mix
else:
    assert not policy.fused_recurrent_output
    assert not policy.fused_recurrent_raw
    assert not policy.fused_output
    assert not policy.fused_norm_mix
assert not policy.fused_prefill_scan
props = torch.cuda.get_device_properties(0)
print(json.dumps({
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device": torch.cuda.get_device_name(0),
    "vram_gib": round(props.total_memory / 2**30, 2),
    "profile_family": policy.profile.family,
    "architecture": policy.profile.architecture,
    "fused_recurrent_output": policy.fused_recurrent_output,
    "fused_recurrent_raw": policy.fused_recurrent_raw,
    "fused_output": policy.fused_output,
    "fused_norm_mix": policy.fused_norm_mix,
    "norm_mix_num_warps": policy.norm_mix_num_warps,
    "fused_prefill_scan": policy.fused_prefill_scan,
}, indent=2))
PY

if [[ "${SYNC_ADAPTER_CODE}" != "0" ]]; then
  run python scripts/sync_hf_adapter_code.py "${HF_DIR}"
fi

run python - "${HF_DIR}" <<'PY' | tee "${OUT_DIR}/native_metadata.log"
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
expected = {
    "architectures": ["NativeRWKV7ForCausalLM"],
    "model_type": "rwkv7_native",
    "auto_model": "native_model.NativeRWKV7ForCausalLM",
}
actual = {
    "architectures": config.get("architectures"),
    "model_type": config.get("model_type"),
    "auto_model": config.get("auto_map", {}).get("AutoModelForCausalLM"),
}
assert actual == expected, {"expected": expected, "actual": actual}
assert not (model_dir / "modeling_rwkv7.py").exists(), "legacy FLA wrapper was not removed"
print(json.dumps(actual, indent=2))
PY

run python -m py_compile \
  rwkv7_hf/kernel_policy.py \
  rwkv7_hf/model_config.py \
  rwkv7_hf/model_cache.py \
  rwkv7_hf/model_generation.py \
  rwkv7_hf/model_fast_api.py \
  rwkv7_hf/model_layers.py \
  rwkv7_hf/model_backbone.py \
  rwkv7_hf/model_prefill_graph.py \
  rwkv7_hf/model_quantization.py \
  rwkv7_hf/model_runtime.py \
  rwkv7_hf/model_speculative.py \
  rwkv7_hf/native_model.py \
  bench/bench_batch_sweep.py \
  bench/bench_chunked_prefill.py \
  bench/bench_native_graph_policy_ab.py

run python tests/test_kernel_policy.py | tee "${OUT_DIR}/kernel_policy.log"
run python -m pytest -q tests/test_native_model_module_split.py \
  | tee "${OUT_DIR}/native_module_split.log"
run python tests/test_native_fla_free_import.py --model "${HF_DIR}" | tee "${OUT_DIR}/fla_free_import.log"
run python tests/smoke_hf_generate.py --model "${HF_DIR}" --device "${DEVICE}" --max-new-tokens 8 \
  | tee "${OUT_DIR}/generate.log"
run python tests/test_hf_api_contract.py --model "${HF_DIR}" --device "${DEVICE}" --dtype "${DTYPE}" --beam-new-tokens 2 \
  | tee "${OUT_DIR}/hf_api_contract.log"
run python tests/test_peft_lora.py --model "${HF_DIR}" --device "${DEVICE}" --attn-mode fused_recurrent \
  | tee "${OUT_DIR}/peft_lora.log"
run python tests/test_dynamic_batch_cache.py \
  --model "${HF_DIR}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --batch-size 3 --prompt-tokens 32 --decode-steps 2 --modes forward \
  | tee "${OUT_DIR}/dynamic_batch_cache.log"
run python tests/test_chunked_prefill.py \
  --model "${HF_DIR}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --batch-size 2 --chunk-sizes 1 4 8 --max-diff 0.2 \
  | tee "${OUT_DIR}/chunked_prefill_correctness.log"
run python tests/test_native_trainer_smoke.py \
  --model "${HF_DIR}" --dtype "${TRAIN_DTYPE}" --max-steps 6 --batch-size 2 --length 16 \
  | tee "${OUT_DIR}/native_trainer.log"

run python bench/bench_batch_sweep.py \
  --hf-dir "${HF_DIR}" --dtype "${DTYPE}" --device "${DEVICE}" \
  --fast-cache true --fast-decode-api false --fast-token-backend auto \
  --batch-sizes ${BATCH_SIZES} --prompt-tokens "${PROMPT_TOKENS}" \
  --decode-tokens "${DECODE_TOKENS}" --warmup 1 --runs 2 \
  --results "${RESULTS}" | tee "${OUT_DIR}/batch_sweep.log"

run python bench/bench_chunked_prefill.py \
  --hf-dir "${HF_DIR}" --dtype "${DTYPE}" --device "${DEVICE}" \
  --batch-size 1 --prompt-tokens "${CHUNKED_PROMPT_TOKENS}" \
  --chunk-sizes ${CHUNK_SIZES} --warmup 0 --runs 1 --max-diff 0.2 \
  --results "${RESULTS}" | tee "${OUT_DIR}/chunked_prefill_bench.log"

GPU_ARCH="$(python - <<'PY'
import torch
print(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":", 1)[0])
PY
)"
if [[ "${GPU_ARCH}" == "gfx1100" ]]; then
  for batch in 1 8; do
    run python bench/bench_native_graph_policy_ab.py \
      --hf-dir "${HF_DIR}" --dtype "${DTYPE}" --device "${DEVICE}" \
      --batch-size "${batch}" --prompt-tokens "${PROMPT_TOKENS}" \
      --correctness-steps 32 --warmup 8 --steps 128 \
      --min-speedup 1.0 --require-accelerated-policy \
      --results "${OUT_DIR}/decode_policy_ab.jsonl" \
      | tee "${OUT_DIR}/decode_policy_ab_b${batch}.log"
  done
fi

echo "AMD ROCm HF VALIDATION PASS"
echo "wrote ${OUT_DIR}"
echo "wrote ${RESULTS}"
