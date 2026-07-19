#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/rwkv7-hf-native-width
export PATH=/home/ubuntu/venv-rwkv5090/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_EXTENSIONS_DIR=/home/ubuntu/bench/train_temp_alignment_20260717/integrated_backend/torch_extensions
export MAX_JOBS=2
OUT=/home/ubuntu/bench/native_train_temp_b16_alignment_20260718
INIT=/home/ubuntu/bench/train_temp_alignment_20260717/rwkv-init-12x768-bf16.pth
MODEL="$OUT/rwkv-init-12x768-native-exact"
SHA=$(sha256sum "$INIT" | awk '{print $1}')
COMMON=(--precision bf16 --device cuda --learning-rate 0.0006 --learning-rate-final 0.00006 --schedule-total-steps 182888 --warmup-steps 10 --weight-decay 0.001 --beta1 0.9 --beta2 0.99 --adam-eps 1e-18 --grad-clip 1.0 --eval-interval 50 --optimizer fused_adam)
validation="$OUT/validation_seed9131_b16_t512.safetensors"
for seed in 131 232 333; do
  sequence="$OUT/sequence_seed${seed}_b16_t512_s1000.safetensors"
  if [ ! -f "$sequence" ]; then
    python bench/bench_train_temp_alignment.py make-sequence --output "$sequence" --metadata "${sequence%.safetensors}.json" --vocab-size 65536 --batch-size 16 --seq-len 512 --steps 1000 --seed "$seed" --pattern increment --active-vocab-size 256
  fi
  official="$OUT/official_convergence_seed${seed}_b16_t512_s1000.json"
  if ! python - "$official" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1])); ok=d.get('status')=='pass' and d.get('steps_completed')==1000
except Exception: ok=False
raise SystemExit(0 if ok else 1)
PY
  then
    python bench/bench_train_temp_alignment.py converge-official --official-checkout /home/ubuntu/RWKV-LM-train-temp --official-config configs/train_temp_official_x070_12x768_b16.json --checkpoint "$INIT" --sequence "$sequence" --validation-batch "$validation" --output-json "$official" --seed "$seed" "${COMMON[@]}"
  fi
  native="$OUT/native_convergence_seed${seed}_b16_t512_s1000.json"
  if ! python - "$native" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1])); ok=d.get('status')=='pass' and d.get('steps_completed')==1000
except Exception: ok=False
raise SystemExit(0 if ok else 1)
PY
  then
    python bench/bench_train_temp_alignment.py converge-hf --model "$MODEL" --checkpoint-sha256 "$SHA" --native --train-temp-cuda --gradient-checkpointing --sequence "$sequence" --validation-batch "$validation" --output-json "$native" --seed "$seed" "${COMMON[@]}"
  fi
done
python bench/bench_train_temp_alignment.py compare-convergence-cohort \
  --reference-json "$OUT/official_convergence_seed131_b16_t512_s1000.json" \
  --reference-json "$OUT/official_convergence_seed232_b16_t512_s1000.json" \
  --reference-json "$OUT/official_convergence_seed333_b16_t512_s1000.json" \
  --candidate-json "$OUT/native_convergence_seed131_b16_t512_s1000.json" \
  --candidate-json "$OUT/native_convergence_seed232_b16_t512_s1000.json" \
  --candidate-json "$OUT/native_convergence_seed333_b16_t512_s1000.json" \
  --output "$OUT/compare_convergence_cohort_b16_t512_s1000.json"
sequence="$OUT/sequence_seed131_b16_t512_s1000.safetensors"
checkpoint="$OUT/native_long_resume_seed131.pt"
python bench/bench_train_temp_alignment.py converge-hf --model "$MODEL" --checkpoint-sha256 "$SHA" --native --train-temp-cuda --gradient-checkpointing --sequence "$sequence" --validation-batch "$validation" --output-json "$OUT/native_long_resume_partial_seed131_s500.json" --seed 131 --checkpoint-out "$checkpoint" --checkpoint-every 250 --stop-after-step 500 "${COMMON[@]}"
python bench/bench_train_temp_alignment.py converge-hf --model "$MODEL" --checkpoint-sha256 "$SHA" --native --train-temp-cuda --gradient-checkpointing --sequence "$sequence" --validation-batch "$validation" --output-json "$OUT/native_long_resume_final_seed131_s1000.json" --seed 131 --resume-from "$checkpoint" --checkpoint-out "$checkpoint" --checkpoint-every 250 --stop-after-step 1000 "${COMMON[@]}"
python bench/bench_train_temp_alignment.py compare-convergence --reference-json "$OUT/native_convergence_seed131_b16_t512_s1000.json" --candidate-json "$OUT/native_long_resume_final_seed131_s1000.json" --output "$OUT/compare_native_long_resume_seed131_s1000.json"
