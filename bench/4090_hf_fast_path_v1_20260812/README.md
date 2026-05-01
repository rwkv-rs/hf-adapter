# RTX 4090 unified HF fast-path v1 matrix

This immutable artifact is the first completed card in the superseding
`hf_fast_path_v1` comparison. The run finished on 2026-08-12 on one NVIDIA
GeForce RTX 4090 and passed the fail-closed protocol validator:

- 48/48 Qwen3.5 reference rows pass the official Transformers FLA plus
  Dao-AILab `causal_conv1d` contract;
- 48/48 RWKV-7 candidate rows pass the `native_jit` fair-lane contract with
  CUDA Graph disabled;
- `validation.json` reports 96/96 rows, one runtime signature and no errors;
- `qwen_official_fast_path_status.json` reports
  `main_table_eligible=true` and `fallback_attempted=false`.

Passing the protocol validator means the rows are eligible for the unified
table. It is separate from passing a speed threshold. Under this fair lane,
RWKV wins all 48 raw and parameter-adjusted Decode cells, but only 26/48
parameter-adjusted Prefill cells. The previous 4090 best-optimized HF rows are
therefore retained as a separate historical/optimized appendix and are not
mixed into this result.

## Fixed protocol and runtime

- Models: RWKV-7 g1d/g1i 0.4B/1.5B/2.9B/7.2B versus Qwen3.5
  0.8B/2B/4B/9B.
- Dense FP16; batch 1/8; prompt 128/512/2048; decode 128/512; prefill chunk
  512; 3 warmups and 7 measured runs; median per cell.
- Python 3.10.18, PyTorch 2.8.0+cu128, CUDA 12.8, Triton 3.4.0,
  Transformers 5.12.1, FLA 0.5.1 and causal-conv1d 1.6.2.post1.
- Repository commit: `e61e9a78797071ffc2bec6159e382367b6bb7b30`.
- FLA source commit: `2e38c1fab332174d056928feaf29f8c5fd5ac550`.
- causal-conv1d source commit:
  `4f6ae4e26ae5fe8af9372f8d312ab25cc4595223`.
- Extension build architectures: `sm_86`, `sm_89` and `sm_120`.
- Exact GPU: RTX 4090 `sm_89`, driver 550.142, 24,564 MiB, 450 W limit.

## Results

Throughputs are medians of the six cells for each model/batch row. Ratio
columns report the minimum and median of the six matched cell-level ratios.
`Adj P pass` counts parameter-adjusted Prefill cells above Qwen.

| Pair | Batch | RWKV P / D tok/s | Qwen P / D tok/s | Raw P min / med | Adjusted P min / med | Raw D min / med | Adjusted D min / med | Adj P pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4B / 0.8B | B1 | 6,945.618 / 59.652 | 10,778.643 / 35.512 | 0.642x / 0.646x | 1.072x / 1.079x | 1.675x / 1.683x | 2.797x / 2.809x | 6/6 |
| 0.4B / 0.8B | B8 | 55,212.087 / 442.485 | 68,760.256 / 268.928 | 0.652x / 0.669x | 1.089x / 1.116x | 1.638x / 1.646x | 2.733x / 2.747x | 6/6 |
| 1.5B / 2B | B1 | 6,796.889 / 58.080 | 10,556.757 / 35.115 | 0.631x / 0.647x | 0.777x / 0.797x | 1.652x / 1.655x | 2.035x / 2.039x | 0/6 |
| 1.5B / 2B | B8 | 53,535.344 / 435.433 | 37,337.034 / 268.261 | 0.663x / 1.289x | 0.817x / 1.588x | 1.620x / 1.623x | 1.996x / 2.000x | 4/6 |
| 2.9B / 4B | B1 | 5,138.012 / 43.787 | 7,626.521 / 25.443 | 0.663x / 0.675x | 0.946x / 0.963x | 1.717x / 1.721x | 2.449x / 2.455x | 0/6 |
| 2.9B / 4B | B8 | 27,751.609 / 326.172 | 15,026.464 / 195.923 | 0.690x / 1.711x | 0.985x / 2.442x | 1.654x / 1.665x | 2.360x / 2.375x | 4/6 |
| 7.2B / 9B | B1 | 5,112.890 / 43.962 | 7,476.292 / 25.499 | 0.664x / 0.681x | 0.826x / 0.846x | 1.720x / 1.724x | 2.139x / 2.144x | 0/6 |
| 7.2B / 9B | B8 | 9,601.733 / 324.181 | 8,525.349 / 197.472 | 1.012x / 1.126x | 1.258x / 1.401x | 1.638x / 1.640x | 2.037x / 2.040x | 6/6 |

Across all 48 matched cells:

- raw Prefill: minimum `0.630649x`, median `0.674730x`, 14/48 above Qwen;
- parameter-adjusted Prefill: minimum `0.776985x`, median `1.077582x`, 26/48
  above Qwen;
- raw Decode: minimum `1.619831x`, median `1.660445x`, 48/48 above Qwen;
- parameter-adjusted Decode: minimum `1.995698x`, median `2.266103x`, 48/48
  above Qwen.

The worst adjusted-Prefill cell is 1.5B/2B B1, prompt 2048, decode 128 at
`0.776985x`. The 7.2B/9B B8 lane passes all six raw and adjusted Prefill cells;
its minimum ratios are `1.011829x` raw and `1.258444x` adjusted.

## Corrected Qwen Decode baseline

The old 4090 values below used the superseded mixed convolution/runtime
contract. The unified values are medians over the full six-cell batch slice.

| Qwen3.5 | Old B1 / B8 tok/s | Unified B1 / B8 tok/s |
|---|---:|---:|
| 0.8B | 28.522 / 215.563 | **35.512 / 268.928** |
| 2B | 28.968 / 219.174 | **35.115 / 268.261** |
| 4B | 20.552 / 160.095 | **25.443 / 195.923** |
| 9B | not measured / 201.751 | **25.499 / 197.472** |

The old 0.8B/2B/4B references are invalid for the new main table because their
effective convolution backend was `fla_triton`. The new rows verify live
FLA chunk Prefill, FLA fused-recurrent Decode, fused gated normalization and
official causal-conv1d prefill/update bindings on every linear-attention layer.

## Evidence map

- `main_table.jsonl`: all 96 validated and ordered result rows.
- `validation.json`: row-count, shape, runtime and backend-contract validator.
- `qwen_official_fast_path_status.json`: fail-closed Qwen admission result.
- `environment.json`, `runtime-lock.json`, `pip-freeze.txt`, `system.csv`:
  locked software and exact-card environment.
- `extension_build_manifest.json`: exact extension versions, source commits
  and CUDA architectures.
- `model_hashes.sha256`: recursive config/safetensors integrity hashes for all
  eight model directories.
- `formal.log` and `exit_code.txt`: final validator output and process status.
- `artifact_sha256.txt`: server-side SHA-256 values used to verify the download.

The downloaded files were checked against the server with SHA-256; every hash
matched before this artifact was committed.
