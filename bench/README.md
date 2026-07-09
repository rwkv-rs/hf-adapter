# Bench workspace

This directory stores benchmark entrypoints, raw evidence, and comparison artifacts for RWKV-7 HF/Transformers, native fused paths, quantization, and Apple/Qwen3.5 work.

The goal is to keep benchmark evidence reproducible without making the repository root impossible to read. New benchmark work should add a timestamped evidence directory with a short `README.md`, keep raw `.jsonl` rows, and link promoted conclusions from `BENCHMARK.md` or the relevant docs page.

## Directory contract

```text
bench/
  bench_*.py                         # single-purpose benchmark entrypoints
  run_*.py / run_*.sh                 # hardware or acceptance orchestrators
  compare_*.py / analyze_*.py         # post-processing and comparison tools
  profile_*.py                        # local profiler probes
  results*.jsonl                      # legacy aggregate result streams
  <lane>_<hardware>_<date>/           # immutable evidence bundle
    README.md                         # what was run, command/env, conclusion
    results*.jsonl                    # machine-readable rows
    *.log / *.txt / *.json / *.md     # raw logs and summaries
```

Evidence directories should be treated as append-only records. If a result is superseded, create a new dated directory instead of editing old rows, except for fixing broken links or adding explanatory README text.

## Naming convention

Use lower-case, underscore-separated names:

```text
<topic>_<hardware>_<yyyymmdd>/
<topic>_<model_or_baseline>_<prompt>_<decode>_<variant>.jsonl
```

Examples:

- `apple_scan_prefill_auto_m5_20260708/`
- `results_rwkv15_512_64_fast_gn.jsonl`
- `5090_blackwell_quant_policy_20260705/`

## Required README fields for new evidence

Each promoted evidence directory should state:

1. hardware / software environment;
2. model paths or model IDs;
3. command lines or orchestrator env vars;
4. key metrics: prefill tok/s, decode tok/s, TTFT/TPOT, peak memory, correctness status;
5. decision: default-on, opt-in, negative evidence, or informational only.

## Current benchmark lanes

| Lane | Purpose | Main files/directories |
|---|---|---|
| Apple MLX/CoreML | Qwen3.5 comparison, MLX scan prefill, CoreML state contract, mobile-oriented memory/speed | `apple_*`, `bench/run_qwen35_apple_baseline.py`, `bench/compare_qwen35_apple_baseline.py`, `bench/audit_qwen35_apple_goal.py` |
| Native fused CUDA | Albatross gap, fused recurrent/output/projection experiments | `bench/bench_native_*`, `bench/bench_fused_*`, `albatross_*`, `5090_*` |
| Quantization | bitsandbytes W8/W4 and native mm8/mm4 policies | `bench/bench_quantization.py`, `bench/bench_native_quant_*`, `5090_blackwell_quant_*` |
| HF acceptance | generate/API/Trainer/PEFT/TRL/DeepSpeed and hardware smoke | `scripts/run_hf_acceptance.sh`, `scripts/run_hardware_smoke.sh`, `bench/run_*_hf_validation.sh` |
| MATH500 / quality | quality and sampling variance against Albatross/HF paths | `math500_*`, `bench/eval_math500_hf.py`, `bench/run_math500_final_acceptance.py` |

## Promotion rules

- **Default-on optimization**: correctness parity, at least two real-model rows, no material memory regression, and clear positive speed on the target lane.
- **Opt-in optimization**: correctness parity but mixed or hardware-specific speed.
- **Negative evidence**: keep it if it prevents repeating work; README should say why it stays default-off.
- **Comparison evidence**: keep combined raw rows and generated comparison rows together in the same directory.

## Quick validation before committing benchmark changes

```bash
PYTHONPATH=. python tests/test_markdown_links.py
python -m json.tool < some_result.json >/dev/null  # for JSON files
python - <<'PY'
import json, pathlib
for p in pathlib.Path('bench').rglob('*.jsonl'):
    for i, line in enumerate(p.read_text(errors='ignore').splitlines(), 1):
        if line.strip():
            json.loads(line)
print('JSONL OK')
PY
git diff --check
```

See `bench/INDEX.md` for the current generated inventory.
