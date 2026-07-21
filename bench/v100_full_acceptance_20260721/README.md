# V100 comprehensive fail-closed acceptance audit

Date: **2026-07-20 to 2026-07-21**  
Adapter branch: `wangyue/v100-full-acceptance`  
Final regression commit: `31d52ca81f28bf94409c7803bf91fa11f3cadc86`  
Hardware: **2 x Tesla V100-PCIE-32GB (`sm_70`)**

## Decision

The Native HF adapter's V100 compatibility, correctness, training, cache and
distributed matrix passes. The universal production-performance target does
**not** pass: the full same-card Albatross parity gate and the full-model W8/W4
all-phase speed gate remain red.

This is intentionally stricter than the older selected-lane V100 promotion.
That earlier promotion remains valid only for its named models, shapes and
quant policies; it must not be reported as universal V100 parity.

## Latest case status

`status_latest.tsv` keeps only the final attempt for each `(phase, case)` pair.
The unfiltered 275-attempt history is in `raw/status.tsv`.

| Phase | Pass | Fail | Total |
|---|---:|---:|---:|
| provenance | 1 | 0 | 1 |
| validation | 21 | 0 | 21 |
| official/Albatross numerical alignment | 21 | 0 | 21 |
| dense Native execution | 32 | 0 | 32 |
| HF/PEFT/TRL training | 38 | 0 | 38 |
| distributed PP and ZeRO | 11 | 0 | 11 |
| Qwen comparison | 3 | 0 | 3 |
| Albatross collection + strict gate | 7 | 1 | 8 |
| quant collection + strict gate | 53 | 1 | 54 |
| **Total** | **187** | **2** | **189** |

The two failures are exactly:

1. `albatross/native_vs_albatross_gate`
2. `quant/production_gate`

No API, numerical-alignment, training, cache or distributed failure is hidden
inside the performance decision.

## Environment

The post-fix environment is preserved in
[`raw/environment_postfix.log`](raw/environment_postfix.log):

- driver `580.159.03`, CUDA runtime `12.4`;
- PyTorch `2.5.1+cu124`, Transformers `5.12.1`, Triton `3.3.0`;
- FLA `0.5.2`, PEFT `0.19.1`, TRL `1.7.0`;
- bitsandbytes `0.49.2`, DeepSpeed `0.19.2`, Accelerate `1.14.0`.

`raw/environment_queue_start.log` is the original queue-start snapshot and
therefore names an earlier adapter commit. Post-fix logs include their exact
commit and supersede it for final regression claims.

## Correctness and Native decode

The final full GPU suite reports:

```text
616 passed, 8 skipped, 2 warnings in 53.07s
```

See [`raw/pytest_gpu_postfix.log`](raw/pytest_gpu_postfix.log).

The strict fast-token checker passes with no failures. Key post-fix rows are:

- HF fast forward `726.3 tok/s` versus direct Native `727.5 tok/s`, or
  `0.9984x`;
- generated decode `1002.6 tok/s` versus eager `260.4 tok/s`, or `3.8502x`,
  with 256/256 exact generated tokens;
- graph-cache hit rate 100% and bounded fast-skip rate 96.88% in the measured
  dynamic-batch run;
- fixed-batch Native graph decode reaches `432.0/869.2/1728.6/3443.1 tok/s`
  for B1/B2/B4/B8 in the measured 0.1B validation bundle.

The checker artifact is
[`raw/fast_decode_bundle_postfix.log`](raw/fast_decode_bundle_postfix.log).

## Two-card 13.3B pipeline execution

The accepted manual PP row uses all 61 layers with a split after layer 30. It:

- generates 8/8 requested tokens;
- returns finite logits on `cuda:0`;
- leaves `last_fast_token_backend=null` for the mixed-device recurrent cache;
- takes 4.0563 seconds (`1.97 tok/s`);
- peaks at 12,589.9 MiB and 12,930.8 MiB on the two cards.

The first raw attempt produced non-finite logits but the old harness still
marked it pass. The runtime and harness were then fixed so non-finite logits,
wrong token count, mismatched reference output or accidental fast-backend use
fail closed. `raw/device_map_13.3b_exact8.jsonl` preserves both attempts;
[`raw/device_map_13.3b_exact8_latest.json`](raw/device_map_13.3b_exact8_latest.json)
is the accepted row.

This host's direct CUDA peer copies are asymmetric and size-dependent: some
cross-device transfers silently return corrupt values. Commit `6e2682d`
therefore CPU-stages Native cross-GPU tensors by default. Direct P2P is an
explicit, host-validated opt-in through `RWKV7_DEVICE_MAP_TRANSFER=p2p`.

## Training and distributed execution

All 38 HF/PEFT/TRL training cases pass; the raw structured rows are in
[`raw/training_results.jsonl`](raw/training_results.jsonl).

All 11 distributed cases pass:

- 0.1B, 0.4B, 1.5B, 2.9B and 7.2B ZeRO-2/ZeRO-3 train;
- the same five model sizes with checkpoint resume;
- 13.3B two-card manual pipeline generation.

The structured ZeRO rows are under [`raw/distributed/`](raw/distributed/).
The 7.2B ZeRO-3 train/resume cases require CPU parameter offload through
`configs/deepspeed/zero3_offload.json`. GPU-only ZeRO-3 exceeded the two
32-GiB cards and is not claimed as passing.

## Qwen3.5 comparison

The strict optimized-reference matrix passes **72/72** cells. Every Qwen row
binds the required FLA/full-fusion path. Minimum ratios are:

| Metric | Minimum |
|---|---:|
| RWKV/Qwen prefill throughput | `1.194x` |
| RWKV/Qwen decode throughput | `1.840x` |
| prefill throughput per active parameter | `1.486x` |
| decode throughput per active parameter | `2.288x` |
| model footprint | `0.701x` RWKV/Qwen |
| peak VRAM | `0.454x` RWKV/Qwen; worst measured row `0.991x` |

See [`raw/qwen_summary.md`](raw/qwen_summary.md) and
[`raw/qwen35_dense_gate.log`](raw/qwen35_dense_gate.log).

## Same-card Albatross gate: not closed

The final comprehensive gate contains 88 cells: 22 decode cells and 66 prefill
cells across 0.1B, 0.4B, 1.5B, 2.9B, 7.2B and both 13.3B checkpoints. Decode
was rerun cleanly at commit `31d52ca`; prefill uses the complete same-card
matrix because the capability-cache change is decode-only. The current report
passes **34/88**:

| Phase | Pass / total | Minimum ratio |
|---|---:|---:|
| decode | 11/22 | `0.6009x` |
| prefill | 23/66 | `0.4152x` |
| total | 34/88 | `0.4152x` |

The median decode ratio is `0.9965x`, but the strict requirement is every cell
at `>=1.0x`; 11 decode cells still fail. The two 13.3B rows reach only
`0.6009x/0.6026x`, and prefill still has 43 failing cells. Therefore the broad
Albatross production gate remains red even though many small-model rows are at
or above parity.

The final gate is
[`raw/native_vs_albatross_gate_31d52ca.json`](raw/native_vs_albatross_gate_31d52ca.json).
The compressed source rows and run log are
[`raw/results_fixed_decode_31d52ca.jsonl.gz`](raw/results_fixed_decode_31d52ca.jsonl.gz)
and [`raw/fixed_decode_31d52ca.log.gz`](raw/fixed_decode_31d52ca.log.gz).
The earlier 32/88 report is retained as
[`raw/native_vs_albatross_gate_pre_31d52ca.json`](raw/native_vs_albatross_gate_pre_31d52ca.json)
to show the effect of removing repeated whole-model capability scans.

## Full-model quant gate: not closed

The quant audit has complete coverage: 70 full-model rows, 28 paired prefill
rows and two informational 13.3B capacity rows. The scored gate passes
**53/98** and fails **45/98**. There are no coverage failures.

| Lane | Pass / total | Prefill minimum | Decode minimum | Worst footprint ratio | Quality minimum |
|---|---:|---:|---:|---:|---:|
| MM4 full model | 18/35 | `0.0686x` | `1.0049x` | `0.9573x` | cosine `0.9982866` |
| MM8 full model | 11/35 | `0.5934x` | `0.6358x` | `0.9257x` | cosine `0.99986684` |
| paired prefill lanes | 24/28 | see raw rows | — | — | gate-aligned |

The dominant gaps are full-model MM4/MM8 prefill and MM8 batched decode.
Footprint reduction and numerical quality generally pass; universal
fp16-or-faster speed does not. The raw fail-closed report is
[`raw/production_gate.json`](raw/production_gate.json).

## Fixes produced by this audit

- `88bf0b7`: force true eager reference baselines in Native benchmark contexts
  and fix benchmark inference-tensor seed handling.
- `6e2682d`: harden mixed-device Native state/activation transfers, suppress
  unsafe redundant output relocation, and make the 13.3B PP test fail closed.
- `31d52ca`: cache multi-device and quantized-model capability checks and remove
  single-device Accelerate hook overhead from the forward benchmark.

## Reproduction entry points

The exact server paths are recorded only for provenance. Portable entry points
from the repository root are:

```bash
python -m pytest -q

RESULTS=bench/results_v100_fast_decode.jsonl \
LOG_DIR=bench/v100_fast_decode_logs \
PROMPT_TOKENS=512 DECODE_TOKENS=128 \
bash bench/run_v100_fast_decode_validation.sh

python bench/bench_native_model_decode.py \
  --hf-dir /path/to/rwkv7-g1g-1.5b-hf \
  --model-size-label 1.5b --dtype fp16 --device cuda \
  --prompt-tokens 128 --decode-steps 256 \
  --warmup 3 --repetitions 5 --batch-sizes 1 2 4 8 \
  --fast-token-api --require-active-extensions \
  --backends native_jit native_graph \
  --results bench/results_v100_native_decode.jsonl
```

Albatross and quant promotion require paired runs on an idle exact V100 with
unchanged clocks, model payloads and gate thresholds. Do not combine historical
high-water rows with a new candidate run or lower the required `>=1.0x`
throughput ratio.
