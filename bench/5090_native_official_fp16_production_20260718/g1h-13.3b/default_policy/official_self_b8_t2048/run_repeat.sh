#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/ubuntu/venv-rwkv5090/bin:/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TORCH_EXTENSIONS_DIR=/home/ubuntu/bench/official_extensions_cu130
export CUDA_VISIBLE_DEVICES=0
repo=/home/ubuntu/rwkv7-hf-native-width
base=/home/ubuntu/bench/native_fp16_model_matrix_20260718/g1h-13.3b/prefill_default_policy_dcc53fc_cu130
self=$base/official_self_b8_t2048
python=/home/ubuntu/venv-rwkv5090/bin/python
mkdir -p "$self" "$base/envelopes"
cd "$repo"
for i in 1 2 3; do
  "$python" scripts/compare_official_native_prefill.py \
    --mode capture-official \
    --hf-dir /home/ubuntu/models/rwkv7/rwkv7-g1h-13.3b-hf \
    --batch-size 8 --prompt-tokens 2048 --warmup 10 --repeats 21 \
    --output "$self/repeat${i}.pt" \
    --official-dir /home/ubuntu/RWKV-Gradio-3-native-hf \
    --official-model /home/ubuntu/models/rwkv7/source-g1h/rwkv7-g1h-13.3b-20260710-ctx10240.pth \
    --official-commit cc57df475465c6cacd42ecd4f2f05a588ee5473b \
    --official-source-manifest "$repo/bench/5090_native_official_alignment_20260718/official_source_manifest.json" \
    --official-emb cpu --official-batched-rkv off --official-cmix-sparse no-fc \
    --official-lowrank-weight transpose --official-orig-linear-groups none \
    > "$self/repeat${i}.log" 2>&1
done
"$python" scripts/compare_official_prefill_self_repeats.py \
  --reference "$base/b8_t2048_official.pt" \
  --repeats "$self/repeat1.pt" "$self/repeat2.pt" "$self/repeat3.pt" \
  --output "$base/envelopes/b8_t2048_envelope.json" \
  > "$self/envelope.log" 2>&1
if [[ ! -f "$base/b8_t2048_pre_envelope_failed_report.json" ]]; then
  cp "$base/b8_t2048_report.json" "$base/b8_t2048_pre_envelope_failed_report.json"
  cp "$base/b8_t2048_compare.log" "$base/b8_t2048_pre_envelope_failed_compare.log"
fi
set +e
"$python" scripts/compare_official_native_prefill.py \
  --mode compare \
  --native-capture "$base/b8_t2048_native.pt" \
  --official-capture "$base/b8_t2048_official.pt" \
  --official-commit cc57df475465c6cacd42ecd4f2f05a588ee5473b \
  --official-self-envelope "$base/envelopes/b8_t2048_envelope.json" \
  --official-envelope-multiplier 1.25 \
  --output "$base/b8_t2048_report.json" \
  > "$base/b8_t2048_compare.log" 2>&1
compare_ec=$?
set -e
COMPARE_EC="$compare_ec" "$python" - <<'PY'
import json, os
from pathlib import Path
base=Path('/home/ubuntu/bench/native_fp16_model_matrix_20260718/g1h-13.3b/prefill_default_policy_dcc53fc_cu130')
report=json.loads((base/'b8_t2048_report.json').read_text())
summary=json.loads((base/'summary.json').read_text())
for row in summary['rows']:
    if row['batch_size']==8 and row['prompt_tokens']==2048:
        row.update(status='pass' if report['quality_pass'] and report['performance_gate_pass'] else 'fail', quality_pass=report['quality_pass'], performance_gate_pass=report['performance_gate_pass'], native_over_official_tokps=report['native_over_official_tokps'], compare_exit_code=int(os.environ['COMPARE_EC']))
summary['quality_pass_cases']=sum(bool(row['quality_pass']) for row in summary['rows'])
summary['performance_pass_cases']=sum(bool(row['performance_gate_pass']) for row in summary['rows'])
summary['status']='pass' if all(row['status']=='pass' for row in summary['rows']) else 'incomplete'
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
printf '%s\n' "$compare_ec" > "$self/exit_code.txt"
exit "$compare_ec"