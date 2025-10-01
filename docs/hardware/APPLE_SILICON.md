# Apple Silicon / MPS / MLX adaptation plan

This document tracks the HF adapter work needed for Apple Silicon. It is a
separate hardware-adaptation lane from the CUDA / Albatross performance route:
CUDA fused kernels remain the production-speed target for NVIDIA cards, while
Apple Silicon uses the FLA-free native PyTorch path today plus an optional MLX
reference backend. An initial opt-in MLX/Metal WKV custom-kernel seam now
exists; production fused MLX/Metal WKV/projection/packed-quant kernels remain
the next Apple performance layer.

## Current status

| Area | Status | Evidence / entry point |
|---|---|---|
| Install without CUDA/FLA | supported by packaging | Base dependencies no longer require `flash-linear-attention`; CUDA users can install `.[fla]` / `.[cuda]`. |
| Tiny Apple smoke | pass on local M-series | `tests/test_apple_silicon_smoke.py` passes on MacBook Air / Apple M5 / 16GB / macOS 26.5 / PyTorch 2.12.1 MPS; see `bench/results_apple_silicon_m5_20260704.jsonl`. |
| Converted-model Apple smoke | 0.1B, 0.4B, and 1.5B pass on local M-series | `scripts/run_apple_silicon_smoke.sh` loads `rwkv7-g1d-0.1b-hf`, `rwkv7-g1d-0.4b-hf`, and `rwkv7-g1g-1.5b-hf` through `RWKV7_NATIVE_MODEL=1` on MPS; 0.4B has fp32/fp16 short-generate rows and 1.5B has fp16 short-generate + prompt sweep rows. |
| HF API coverage | partial | Load + forward + `generate(use_cache=True)` through the native backend; tiny native backward and Trainer paths pass; real 0.1B and 0.4B PEFT LoRA, HF Trainer, and TRL SFT/DPO/GRPO paths on MPS are covered. 0.4B also has fp32/fp16 generation length sweep rows and 2-step Trainer/TRL rows. 1.5B has fp16 MPS inference/sweep rows through prompt 512 / decode 8, MLX prompt8192/decode512 rows, plus fp32 PEFT LoRA manual, HF Trainer, and TRL SFT/DPO/GRPO 1/2/3/5/10/12-step rows with finite trainable updates. |
| Quantization | broader functional native smoke + initial MLX/Metal packed speed path | `bitsandbytes` W8/W4 is CUDA-oriented and is not the Apple path. Native MM8/MM4 config-driven module replacement now runs on MPS for tiny, 0.1B, 0.4B, and 1.5B smoke rows. MLX now has a packed W8/W4 affine dequant-matmul projection path (`--quantization mm8/mm4`), an opt-in fused MLX/Metal dequant-projection seam (`--quant-backend metal`), and a conservative backend router (`--quant-backend auto`) with 0.1B short smoke and 0.4B/1.5B prompt128/256 decode4/8 plus prompt512/1024 decode16 and W4 prompt2048/decode128 pressure rows. Same-shape fp16 Metal baselines are now recorded for the prompt512/1024 decode16 ratio gate, and quant+Metal session-batch pressure rows now cover 0.4B 6-session repeat=3 plus 1.5B 5-session repeat=2. Memory drops, but W8/W4 do not yet beat fp16 end to end. Production-speed Apple quant still needs longer repeat/session pressure, more Apple GPU coverage, and fused speed work. |
| Production speed | not claimed | PyTorch MPS is a compatibility path, not the final Apple performance backend. |
| MLX recurrent backend / Metal backend | initial MLX recurrent + Metal WKV + Metal packed quant seams | Optional `.[mlx]` install, `rwkv7_hf.mlx_bridge`, `rwkv7_hf.mlx_model`, `rwkv7_hf.mlx_quant`, `rwkv7_hf.mlx_wkv`, `scripts/convert_hf_to_mlx.py`, `scripts/mlx_generate.py`, `scripts/mlx_session_smoke.py`, `scripts/mlx_session_batch_smoke.py`, `scripts/mlx_generation_sweep.py`, `scripts/mlx_quant_projection_bench.py`, `scripts/run_apple_silicon_mlx_smoke.sh`, `scripts/run_apple_silicon_mlx_model_smoke.sh`, `scripts/run_apple_silicon_mlx_session_smoke.sh`, `scripts/run_apple_silicon_mlx_session_batch_smoke.sh`, and `scripts/run_apple_silicon_mlx_generation_sweep.sh` now validate HF safetensor → MLX array/export, tiny torch/MLX recurrent parity, MLX state-cache select/chunked-prefill/session behavior, tokenizer-backed prompt smoke, dynamic-batch state select, reusable MLX text generate, prefill-once/session decode, interleaved multi-session decode, optional equal-round `--session-backend batched|auto` MLX session batching, prompt/decode sweeps, optional MLX packed W8/W4 projection rows, opt-in `--wkv-backend metal|auto`, opt-in `--quant-backend metal` fused dequant-projection rows, and `--quant-backend auto` backend-routing rows. `scripts/mlx_quant_projection_bench.py` now isolates dense/affine/Metal/auto projection speed so fused WKV+quant work has a reproducible microbench. Production fused Metal still needs longer pressure and cross-device tuning. |

## Why the Apple path is native / no-FLA by default

The default optimized CUDA wrapper depends on `flash-linear-attention` and
Triton/CUDA kernels. Those are not the right baseline for macOS. Apple Silicon
smoke should use:

```bash
export RWKV7_NATIVE_MODEL=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export RWKV7_FAST_FORWARD=0
export RWKV7_FAST_CACHE=0
export RWKV7_FAST_TOKEN_BACKEND=native_jit
```

`RWKV7_NATIVE_MODEL=1` routes `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)` into the FLA-free native PyTorch backend. MPS fallback
keeps unsupported individual ops from aborting the run while the dedicated
MLX/Metal path grows from the current opt-in WKV seam into a production backend.

## Current local evidence

Local smoke on 2026-07-04:

| Machine | Memory | macOS | PyTorch | Device | Test | Result |
|---|---:|---|---|---|---|---|
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | tiny native RWKV-7 `generate()` | PASS (`elapsed_s=0.1121`, 2 generated tokens) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.1b-hf` load + forward + `generate()` | PASS (`elapsed_s=0.2406`, 11 prompt tokens + 2 generated tokens) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | tiny native MM8/MM4 quant smoke | PASS (config-driven from_pretrained; MM8 footprint ratio=0.391615, MM4 footprint ratio=0.267734; decode backend=eager) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.1b-hf` native MM8/MM4 quant smoke | PASS (`MIN_PARAMS=8000000` lm_head replacement; `MIN_PARAMS=1000000` replaces 25 FFN/lm_head modules; `MIN_PARAMS=500000` replaces 73 attention/FFN/lm_head modules; MM8 footprint ratio≈0.253433, MM4 footprint ratio≈0.128433; 1-token generate) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.4b-hf` native MM8/MM4 quant sweep | PASS (`MIN_PARAMS=4000000` replaces 49 FFN/lm_head modules; MM8 footprint ratio≈0.252327, MM4 footprint ratio≈0.127327; 1-token generate) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1g-1.5b-hf` native MM8/MM4 quant smoke | PASS (`MIN_PARAMS=8000000` replaces 49 FFN/lm_head modules; MM8 footprint ratio≈0.251190, MM4 footprint ratio≈0.126190; 1-token generate; MM4 driver memory≈14.8GB, so 16GB is functional smoke only) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | tiny MLX tensor save/load + matmul smoke | PASS (`axis=apple_silicon_mlx_tiny`, `elapsed_s=0.032803`, output shape `[1, 24]`) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.1b-hf` HF safetensor → MLX projection matmul | PASS (`axis=apple_silicon_mlx_projection_smoke`, tensor `model.layers.0.attn.r_proj.weight`, fp16 `[1, 768]`, selected tensor bytes=1179648) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.1b-hf` selected HF safetensor → MLX safetensors export | PASS (`axis=mlx_hf_export`, tensor count=1, fp16 bytes=1179648, manifest `mlx_manifest.json`) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | MLX packed W8/W4 affine quant projection path | PASS (`rwkv7-g1d-0.1b-hf` lm_head W8/W4 footprint≈0.502635/0.252635, prompt32/decode2 W8 min decode≈86.07 tok/s and W4≈32.72 tok/s; `rwkv7-g1d-0.4b-hf` 49 FFN/lm_head modules W8/W4 footprint≈0.502327/0.252327, prompt16/decode1 W8 min decode≈22.09 tok/s and W4≈8.14 tok/s; `rwkv7-g1g-1.5b-hf` 49 FFN/lm_head modules W8/W4 footprint≈0.501190/0.251190, prompt32/decode4 with chunk16 W8 min prefill/decode≈5.63/4.65 tok/s and W4≈2.15/1.79 tok/s; current backend=`affine`; see the `--quant-backend metal` row below for the first fused speed seam) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | MLX/Metal packed W8/W4 fused dequant-projection seam | PASS (`--quant-backend metal --wkv-backend metal`; `rwkv7-g1d-0.1b-hf` lm_head-only prompt32/decode2: W8 footprint≈0.502635, peak≈338MB, prefill≈88.01 tok/s, decode≈110.59 tok/s; W4 footprint≈0.252635, peak≈313MB, prefill≈105.80 tok/s, decode≈75.27 tok/s. `rwkv7-g1d-0.4b-hf` 49 FFN/lm_head modules prompt16/decode1: W8 footprint≈0.502327, peak≈642MB, prefill≈49.48 tok/s, decode≈49.65 tok/s; W4 footprint≈0.252327, peak≈508MB, prefill≈47.99 tok/s, decode≈49.38 tok/s. `rwkv7-g1g-1.5b-hf` 49 FFN/lm_head modules prompt16/decode1: W8 footprint≈0.501190, peak≈2134MB, prefill≈19.47 tok/s, decode≈21.59 tok/s; W4 footprint≈0.251190, peak≈1664MB, prefill≈18.63 tok/s, decode≈24.04 tok/s. Pressure matrix rows now cover 0.4B/1.5B prompt128/256 decode4/8 chunk64, prompt512/1024 decode16 chunk256, and prompt2048/decode128 chunk512: 0.4B prompt2048 W8 peak≈649MB prefill/decode≈43.54/42.03 tok/s, 0.4B prompt2048 W4 peak≈515MB prefill/decode≈50.18/49.84 tok/s, 1.5B prompt2048 W8 peak≈2147MB prefill/decode≈21.03/18.59 tok/s, and 1.5B prompt2048 W4 peak≈1677MB prefill/decode≈20.97/19.74 tok/s. The optimized W4 auto route records `metal=202885` and improves prompt2048/decode128 to 0.4B peak≈515MB prefill/decode≈60.61/59.73 tok/s and 1.5B peak≈1677MB prefill/decode≈27.64/20.42 tok/s. New prompt4096/decode256 chunk1024 rows keep chunked/full max_abs=0.0 and record 0.4B fp16 vs W4 auto prefill/decode≈94.08/75.38 vs 62.01/55.29 tok/s, and 1.5B fp16 vs W8/W4≈35.34/33.21 vs 22.52/20.54 and 27.40/25.46 tok/s; W8/W4 peak remains≈0.70x/0.54x fp16 and speed is still below fp16 at this longer shape. New 1.5B prompt8192/decode512 chunk2048 rows keep max_abs=0.0 and record fp16≈27.97/26.02 tok/s vs W4 auto≈22.77/21.20 tok/s with peak≈1677MB(≈0.54x) and `metal=811525`. Repeat/session pressure and stable fp16-beating gates remain open.) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | MLX/Metal W8/W4 vs fp16 ratio gate | GAP (`--quant-backend metal --wkv-backend metal` compared with same-shape `--quantization none --wkv-backend metal`. prompt512/1024 decode16: 0.4B fp16 peak≈929MB, decode≈52.37 tok/s, W8/W4 peak ratios≈0.71x/0.57x and decode ratios≈0.79x/0.81x; 1.5B fp16 peak≈3110MB, decode≈23.15 tok/s, W8/W4 peak ratios≈0.70x/0.55x and decode ratios≈0.75x/0.84x. New prompt2048/decode128: 0.4B fp16 peak≈916MB, decode≈47.97 tok/s, W8/W4 peak ratios≈0.71x/0.56x and decode ratios≈0.88x/1.04x; 1.5B fp16 peak≈3084MB, decode≈27.20 tok/s, W8/W4 peak ratios≈0.70x/0.54x and decode ratios≈0.68x/0.73x. The optimized W4 auto route improves prompt2048/decode128 to 0.4B W4 decode≈1.25x fp16 with peak≈0.56x and 1.5B W4 decode≈0.75x fp16 with peak≈0.54x. At prompt4096/decode256 W4 auto remains below fp16 (0.4B decode≈0.73x, 1.5B decode≈0.77x) while keeping peak≈0.56x/0.54x; 1.5B W8/Metal at the same shape records decode≈0.62x fp16 with peak≈0.70x. At prompt8192/decode512, 1.5B W4 auto records decode≈0.81x fp16 with peak≈0.54x. Memory target is moving in the right direction and 0.4B W4 has stronger fp16-beating medium-long decode evidence, but stable W8/W4 speed > fp16 across sizes/modes is still open.) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | MLX quant projection microbench | PASS (`scripts/mlx_quant_projection_bench.py --rows 1,4 --bits 4,8 --groups 3 --in-features 2048 --out-features 2048`; W4 footprint≈0.252x, W8≈0.502x. Isolated single-projection rows show W4/Metal can beat dense fp16 at rows=4 in one run (`1.11x`, auto `1.38x`) but remains below dense at rows=1, while W8/Metal improves over affine but remains below dense and W8 auto stays affine by default. Grouped R/K/V-style microbench rows prepack grouped weights and preserve exactness vs separate Metal (`max_abs_vs_separate_metal=0.0`): W8 group rows=1 reaches≈1.12x dense and≈1.10x separate Metal, rows=4 remains≈0.58x dense but≈1.08x separate; W4 group rows remain below dense and do not beat separate Metal. The full MLX model has a default-off integration seam, `RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1`; its default mode is now `direct`, routing three distinct R/K/V inputs plus their existing packed weights through one Metal launch without duplicating a grouped weight cache. Conclusion: W8 benefits from grouped launch fusion, but W4 needs deeper WKV/projection fusion before production speed claims.) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | MLX grouped R/K/V quant model A/B | PASS (`RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1`, broader quant thresholds that include R/K/V, `--wkv-backend metal`; all rows keep chunked/full prefill `max_abs=0.0` and grouped fallback=0. Old packed-cache A/B showed positive speed but higher peak memory: 0.4B W4 auto prompt128/decode8 baseline→packed grouped≈39.33/38.68→44.33/41.38 tok/s with group hits=12672, and 1.5B W4≈19.03/18.37→20.18/19.03 tok/s with group hits=6336. New default `direct` mode removes the duplicated grouped-weight cache while keeping positive W4 rows: 0.4B W4 direct prompt128/decode8≈43.30/42.54 tok/s, peak≈365MB, group hits=6336; 1.5B W4 direct prompt128/decode8≈20.95/19.52 tok/s, peak≈1075MB, group hits=6336; longer prompt512/decode16 direct rows pass at 0.4B W4≈45.83/45.17 tok/s, peak≈365MB, group hits=24960 and 1.5B W4≈20.69/19.28 tok/s, peak≈1075MB, group hits=24960. The same direct W8/Metal path now has prompt512/decode16 rows: 0.4B≈44.50/41.50 tok/s, peak≈549MB, group hits=24960; 1.5B≈19.81/19.27 tok/s, peak≈1746MB, group hits=24960. Broader-threshold direct prompt2048/decode128 rows, with R/K/V quantized, also pass with grouped fallback=0 and chunked/full max_abs=0.0: 0.4B W4/W8≈46.50/43.70 and 42.52/41.36 tok/s at peaks≈365/549MB; 1.5B W4/W8≈21.31/19.63 and 20.47/19.78 tok/s at peaks≈1075/1746MB. New direct W4 prompt4096/decode256 rows keep chunked/full max_abs=0.0: 0.4B broad-threshold≈52.05/45.05 tok/s, peak≈365MB, grouped fallback=0, and 1.5B≈21.14/19.98 tok/s, peak≈1075MB, grouped fallback=0; the 0.4B 4M-threshold control falls back for R/K/V and peaks≈515MB. These broadened direct rows strengthen memory evidence but remain below the stable fp16-speed gate. Earlier W8 packed-cache A/B rows also improve (0.4B W8≈40.32/38.93→43.01/42.80 tok/s; 1.5B W8 prompt64/decode4≈17.62/17.22→19.02/17.53 tok/s). Direct grouped session pressure now covers W4 0.4B 4-session rounds4,4, W4/W8 0.4B 6-session rounds8,8 repeat=2, W4/W8 0.4B 8-session rounds8,8 repeat=2, W4/W8 1.5B 5-session rounds4,4, and W4/W8 1.5B 5-session rounds8,8 repeat=2 probes. One-shot token/text/seen-token checks pass for 0.4B 8-session W4/W8 with aggregate round mins≈97.85/91.08 tok/s and peaks≈505/690MB; 1.5B W4/W8 sequential rounds8,8 repeat=2 also passes with aggregate round mins≈19.49/18.35 tok/s and peaks≈1126/1797MB. 1.5B W8 direct batched strict compare matches sequential and one-shot with aggregate round mins≈26.02/25.04 tok/s, but 1.5B W4 direct batched remains a correctness gap: compare-only localizes mismatches to synthetic sessions at first token indices 6 and 9 despite grouped fallback=0. `SESSION_BACKEND=auto` now applies `auto_mm4_metal_batch_exactness_guard` for W4/Metal and falls back to sequential; the guarded 1.5B W4 direct auto row passes rounds8,8 repeat=2 with aggregate round min≈20.67 tok/s, peak≈1126MB, `metal=13632`, and grouped fallback=0.) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | MLX/Metal W8/W4 session-batch pressure | PASS (`scripts/mlx_session_batch_smoke.py --quant-backend metal --wkv-backend metal`; 4 interleaved sessions, rounds=4,4, one-shot token/text match and seen-token checks pass. `rwkv7-g1d-0.4b-hf` W8/W4 repeat=2: peak≈669MB/534MB, min decode≈40.18/41.17 tok/s, WKV counts all Metal. `rwkv7-g1g-1.5b-hf` W8/W4 repeat=1: peak≈2185MB/1716MB, min decode≈19.58/20.38 tok/s, WKV counts all Metal.) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | MLX/Metal W8/W4 higher-concurrency session pressure | PASS (`--quant-backend metal --wkv-backend metal`; one-shot token/text match and seen-token checks pass. `rwkv7-g1d-0.4b-hf` W8/W4 6 sessions repeat=3: peak≈682MB/547MB, cache≈2328MB/2364MB, min decode≈34.33/27.14 tok/s. `rwkv7-g1g-1.5b-hf` W8/W4 5 sessions repeat=2: peak≈2198MB/1728MB, cache≈5672MB/5681MB, min decode≈15.60/18.87 tok/s. New opt-in `SESSION_BACKEND=batched` W4 rows pass with actual `round_backends=["batched","batched"]`: 0.4B 6-session repeat=2 peak≈617MB, cache≈2444MB, per-session min≈19.00 tok/s, aggregate round min≈105.44 tok/s; 1.5B 5-session repeat=1 peak≈1841MB, cache≈5813MB, per-session min≈6.61 tok/s, aggregate round min≈32.38 tok/s. Longer rounds8,8 pressure also passes: 0.4B W4 8-session repeat=2 peak≈656MB, aggregate round min≈103.91 tok/s; 1.5B W4 5-session repeat=2 peak≈1841MB, aggregate round min≈29.63 tok/s; 1.5B W8 auto 5-session repeat=2 peak≈2198MB, aggregate round min≈18.38 tok/s with the W8 guard. W8/Metal strict batched longer decode still needs an exactness fix, so `SESSION_BACKEND=auto` now records `auto_mm8_metal_batch_exactness_guard` and falls back to sequential; safe auto W8 rows pass for 0.4B 6-session repeat=2 (peak≈682MB, min decode≈39.80 tok/s) and 1.5B 5-session repeat=1 (peak≈2198MB, min decode≈17.43 tok/s). New backend-compare rows show W4 sequential-vs-batched token equality on 0.4B and 1.5B, 1.5B W8 equality in this matrix, and the 0.4B W8 mismatch localized to token index 6 on the short prompt. Optional mismatch-logit tracing shows this W8 mismatch is a low-margin tie case: sequential has tokens 11/261 tied at logit≈8.476562 and `mx.argmax` picks 11, while batched Metal lowers token 11 to≈8.46875 and picks 261 with max-abs logit delta≈0.03125. An explicit `SESSION_BACKEND=batched_stable` low-margin argmax policy restores strict token equality for 0.4B W8/Metal on both 3-session and 6-session compare rows; the 6-session row records batched aggregate round mins≈162.12/163.72 tok/s, peak≈790MB, and `metal=20378`. New longer stable rows also pass: 1.5B W8/Metal rounds8,8 strict compare matches, 0.4B W8/Metal 8-session rounds8,8 repeat=2 matches one-shot with aggregate round min≈184.62 tok/s, and 1.5B W8/Metal 5-session rounds8,8 repeat=4 still matches one-shot with aggregate round min≈26.11 tok/s, peak≈2311MB, and `metal=50728`; the matching 1.5B W4 auto 5-session rounds8,8 repeat=4 row records aggregate round min≈30.94 tok/s, peak≈1841MB, and `metal=50728`. These repeat=4 rows show correctness survives sustained pressure while throughput drops on the local M5/16GB machine. Default W8/Metal auto stays guarded, but `RWKV7_MLX_SESSION_AUTO_W8_STABLE=1` opts `SESSION_BACKEND=auto` into this stable policy; a 0.4B W8/Metal auto row passes with aggregate round min≈90.73 tok/s and `metal=5126`. `--quant-backend auto` now adds safe backend routing: 0.4B W4 auto uses the Metal path (`metal=4913`) and passes a 3-session strict compare with batched aggregate round mins≈78.68/69.17 tok/s, while 0.4B W8 auto defaults to affine and now batches under `SESSION_BACKEND=auto` with aggregate round min≈49.76 tok/s and `affine=5126`. Stable fp16-beating gates remain open.) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | MLX/Metal WKV custom-kernel seam | PASS (`--wkv-backend metal`; `rwkv7-g1d-0.1b-hf` prompt32/decode2 chunk16 peak≈388MB, prefill≈98.55 tok/s, decode≈116.15 tok/s, chunked/full max_abs=0.0; `rwkv7-g1d-0.4b-hf` prompt16/decode1 peak≈910MB, prefill≈54.34 tok/s, decode≈53.18 tok/s; `rwkv7-g1g-1.5b-hf` prompt16/decode1 peak≈3071MB, prefill≈27.61 tok/s, decode≈26.72 tok/s. This is the initial kernel seam, not the final production fused WKV/projection/packed-quant speed path.) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 / PyTorch 2.12.1 | MLX GPU | tiny full recurrent MLX vs native PyTorch parity | PASS (`axis=apple_silicon_mlx_recurrent_tiny_parity`, batch=2, seq=4, max_abs=0.00282228, argmax match) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | tiny MLX recurrent state cache + chunked prefill + session decode | PASS (`axis=apple_silicon_mlx_state_cache_tiny`, chunked/full max_abs=0.0, select-batch decode max_abs=0.0014168; `axis=apple_silicon_mlx_session_tiny`, step_sizes=2,2, one-shot token/text match) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.1b-hf` full MLX recurrent prefill + greedy decode | PASS (`axis=apple_silicon_mlx_recurrent_model_smoke`, fp16 full 399 tensors, prompt=4, generated=1, chunked/full max_abs=0.0, bytes=382069248) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 / PyTorch 2.12.1 | MLX GPU + CPU compare | `rwkv7-g1d-0.1b-hf` MLX recurrent vs HF native PyTorch | PASS (`axis=apple_silicon_mlx_recurrent_model_smoke`, fp32, torch_compare_max_abs=0.01374531, argmax match, bytes=764138496) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.1b-hf` tokenizer prompt + dynamic-batch MLX recurrent smoke | PASS (`prompt="The quick brown fox"`, fp16, prompt=4, generated=2, prefill≈132.05 tok/s, decode≈167.99 tok/s, dynamic select max_abs=0.046875, argmax match) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.4b-hf` tokenizer prompt + dynamic-batch MLX recurrent smoke | PASS (fp16 full 795 tensors, bytes=901535744, prompt=4, generated=1, prefill≈62.95 tok/s, decode≈83.74 tok/s, dynamic select max_abs=0.03125, argmax match) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1g-1.5b-hf` tokenizer prompt + dynamic-batch MLX recurrent smoke | PASS (fp16 full 795 tensors, bytes=3054809088, prompt=4, generated=1, prefill≈10.38 tok/s, decode≈29.33 tok/s, dynamic select max_abs=0.046875, argmax match) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `scripts/mlx_generate.py` reusable text-generate API | PASS (`rwkv7-g1d-0.1b-hf` 8 tokens decode≈95.29 tok/s, peak≈389MB; `rwkv7-g1d-0.4b-hf` 8 tokens decode≈53.02 tok/s, peak≈914MB; `rwkv7-g1g-1.5b-hf` 4 tokens decode≈28.97 tok/s, peak≈3080MB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `scripts/mlx_generation_sweep.py` prompt/decode sweep + chunked-prefill/repeat check | PASS (`rwkv7-g1d-0.1b-hf` prompt16/64 decode2/4 chunk=32 peak≈397MB, repeat pressure prompt32/decode2 x3 peak≈397MB, and longer prompt128/256 decode4/8 repeat=2 chunk=64 peak≈397MB / min prefill≈187.23 tok/s / min decode≈153.35 tok/s; `rwkv7-g1d-0.4b-hf` prompt16/64 decode2 peak≈934MB, prompt128/256 decode4/8 repeat=1 chunk=64 peak≈934MB / min prefill≈49.77 tok/s / min decode≈33.51 tok/s, prompt256/512 decode16/32 repeat=1 chunk=128 peak≈934MB / min prefill≈54.76 tok/s / min decode≈43.90 tok/s, prompt1024/decode64 chunk=256 peak≈921MB / prefill≈55.20 tok/s / decode≈49.32 tok/s, and prompt4096/decode256 chunk=1024 peak≈916MB / prefill≈94.08 tok/s / decode≈75.38 tok/s; `rwkv7-g1g-1.5b-hf` prompt16/64 decode2 peak≈3119MB, prompt128/256 decode4/8 repeat=1 chunk=64 peak≈3119MB / min prefill≈22.00 tok/s / min decode≈18.19 tok/s, prompt256/512 decode16/32 repeat=1 chunk=128 peak≈3119MB / min prefill≈24.05 tok/s / min decode≈22.36 tok/s, prompt1024/decode64 chunk=256 peak≈3093MB / prefill≈23.60 tok/s / decode≈19.55 tok/s, prompt4096/decode256 chunk=1024 peak≈3084MB / prefill≈35.34 tok/s / decode≈33.21 tok/s, and prompt8192/decode512 chunk=2048 peak≈3084MB / prefill≈27.97 tok/s / decode≈26.02 tok/s; all chunked/full max_abs=0.0) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `MLXGenerationSession` prefill-once + chunked decode smoke | PASS (`rwkv7-g1d-0.1b-hf` step_sizes=4,4 token/text match vs one-shot, decode≈60.43 tok/s, peak≈392MB; `rwkv7-g1d-0.4b-hf` step_sizes=4,4 match, decode≈54.13 tok/s, peak≈921MB; `rwkv7-g1g-1.5b-hf` step_sizes=2,2 match, decode≈27.50 tok/s, peak≈3093MB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `MLXGenerationSessionBatch` interleaved multi-session decode smoke | PASS (`rwkv7-g1d-0.1b-hf` 2 sessions rounds=2,2 match peak≈394MB; 3 sessions rounds=2,2 repeat=2 match peak≈397MB / cache≈7.4MB / min decode≈94.56 tok/s; `rwkv7-g1d-0.4b-hf` 3 sessions rounds=2,2 repeat=2 match peak≈934MB / cache≈12.8MB / min decode≈40.74 tok/s, 4 sessions rounds=4,4 repeat=4 match peak≈940MB / cache≈12.8MB / min decode≈37.76 tok/s, and 6 sessions rounds=4,4 repeat=5 match peak≈953MB / cache≈12.8MB / min decode≈41.27 tok/s; `rwkv7-g1g-1.5b-hf` 3 sessions rounds=2,2 repeat=2 match peak≈3119MB / cache≈25.7MB / min decode≈23.86 tok/s, 4 sessions rounds=4,4 repeat=4 match peak≈3132MB / cache≈25.7MB / min decode≈11.64 tok/s, and 5 sessions rounds=4,4 repeat=2 match peak≈3145MB / cache≈25.7MB / min decode≈21.55 tok/s) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.4b-hf` load + forward + `generate()` | PASS (`elapsed_s=0.4699`, 11 prompt tokens + 1 generated token, MPS driver memory≈2171MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.4b-hf` fp16 load + forward + `generate()` | PASS (`elapsed_s=1.2837`, 11 prompt tokens + 1 generated token, MPS driver memory≈1083MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.4b-hf` fp32/fp16 prompt-length sweep | PASS (fp32 prompt tokens 16/64/128; fp16 prompt tokens 16/64/128/256/512; 4 generated tokens; fp16 peak driver_mem≈1219MiB, fp32 peak driver_mem≈2203MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1g-1.5b-hf` fp16 load + forward + `generate()` | PASS (`elapsed_s=1.6407`, 11 prompt tokens + 1 generated token, MPS driver memory≈3283MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1g-1.5b-hf` fp16 prompt-length/decode sweep | PASS (prompt tokens 16/64/128/256/512; 2/4/8 generated tokens; peak driver_mem≈3547MiB; prompt512/new8 prefill 29.753 tok/s, decode 0.453 tok/s) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / PEFT 0.19.1 | MPS | tiny native train + PEFT LoRA train | PASS (`loss=3.870411`, LoRA trainable params=1792) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 | MPS | tiny native Trainer + PEFT LoRA Trainer | PASS (`training_loss=3.877832`, native `changed_l1=6.063786`, LoRA `changed_l1=0.891996`) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 | MPS | `rwkv7-g1d-0.1b-hf` PEFT LoRA train + HF Trainer | PASS (`loss=2.70401`, LoRA params=663552, Trainer `changed_l1=26.63627`, driver_mem≈2466MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 | MPS | `rwkv7-g1d-0.4b-hf` PEFT LoRA train + HF Trainer | PASS (`loss=2.22734`, LoRA params=1769472, Trainer `changed_l1=67.165365`, driver_mem≈4651MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 | MPS | `rwkv7-g1g-1.5b-hf` fp32 PEFT LoRA manual backward | PASS (`loss=1.976301`, LoRA params=3538944, `grad_l1=10386.947289`, `changed_l1=137.084609`, driver_mem≈6843MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 | MPS | `rwkv7-g1g-1.5b-hf` fp32 PEFT LoRA HF Trainer | PASS (1/2/3/5/10/12-step rows; 12-step `training_loss=1.398190`, `changed_l1=907.843804`, 0.770 steps/s, driver_mem≈6875MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 / TRL 1.7.0 | MPS | `rwkv7-g1d-0.1b-hf` TRL SFTTrainer + PEFT LoRA | PASS (`training_loss=2.70401`, `changed_l1=26.620176`, 1.446 steps/s) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 / TRL 1.7.0 | MPS | `rwkv7-g1d-0.1b-hf` TRL GRPOTrainer + PEFT LoRA | PASS (`training_loss=0.0`, `changed_l1=10.454315`, 3.098 steps/s) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 / TRL 1.7.0 | MPS | `rwkv7-g1d-0.1b-hf` TRL DPOTrainer + PEFT LoRA | PASS (`training_loss=0.693147`, `changed_l1=28.315877`, 1.518 steps/s) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 / TRL 1.7.0 | MPS | `rwkv7-g1d-0.4b-hf` TRL SFT/DPO/GRPO + PEFT LoRA | PASS (1-step and 2-step rows; 2-step SFT `training_loss=3.140634`, DPO `training_loss=0.692913`, GRPO `training_loss=0.0`, peak driver_mem≈4980MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 / TRL 1.7.0 | MPS | `rwkv7-g1g-1.5b-hf` TRL SFT/DPO/GRPO + PEFT LoRA | PASS (1/2/3/5/10/12-step rows; 12-step SFT `training_loss=1.411032`, DPO `training_loss=0.374775`, GRPO `training_loss=0.0`, peak driver_mem≈8604MiB) |

Memory-pressure note after the latest 1.5B MLX prompt8192/decode512 and 12-step
Trainer/TRL runs: `vm_stat` reported free pages≈3.6k, inactive pages≈352k,
speculative pages≈54k, wired pages≈185k, compressor pages≈13k. Swap counters
are cumulative macOS counters, so the table relies on per-row MPS memory for
run-local memory evidence.

Commands:

```bash
PYTHONPATH=. python tests/test_apple_silicon_smoke.py \
  --device auto \
  --dtype fp32 \
  --max-new-tokens 2 \
  --results bench/results_apple_silicon_m5_20260704.jsonl

PYTHONPATH=. python tests/test_apple_silicon_smoke.py \
  --device auto \
  --dtype fp32 \
  --max-new-tokens 2 \
  --skip-tiny \
  --model /path/to/rwkv7-g1d-0.1b-hf

REQUIRE_PEFT=1 RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_training_smoke.sh

REQUIRE_PEFT=1 RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_trainer_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
REQUIRE_PEFT=1 RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
REQUIRE_PEFT=1 REQUIRE_TRL=1 RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_trl_sft_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
REQUIRE_PEFT=1 REQUIRE_TRL=1 RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_rl_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
SKIP_TINY=1 \
MAX_NEW_TOKENS=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
SKIP_TINY=1 \
DTYPE=fp16 \
MAX_NEW_TOKENS=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
PROMPT_LENGTHS=16,64,128 \
MAX_NEW_TOKENS=4 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
DTYPE=fp16 \
PROMPT_LENGTHS=256,512 \
MAX_NEW_TOKENS=4 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
REQUIRE_PEFT=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_trl_sft_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_rl_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
MAX_LENGTH=16 MAX_STEPS=2 DATASET_REPEATS=3 \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_rl_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
SKIP_TINY=1 \
DTYPE=fp16 \
MAX_NEW_TOKENS=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
MAX_NEW_TOKENS=2 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp16 \
PROMPT_LENGTHS=64,128,256,512 \
MAX_NEW_TOKENS=4 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp16 \
PROMPT_LENGTHS=512 \
MAX_NEW_TOKENS=8 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp32 \
MAX_LENGTH=8 \
BATCH_SIZE=1 \
MAX_STEPS=1 \
DATASET_REPEATS=2 \
BACKEND=manual \
REQUIRE_PEFT=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp32 \
MAX_LENGTH=8 \
BATCH_SIZE=1 \
MAX_STEPS=10 \
DATASET_REPEATS=12 \
BACKEND=trainer \
REQUIRE_PEFT=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp32 \
MAX_LENGTH=8 \
BATCH_SIZE=1 \
MAX_STEPS=10 \
DATASET_REPEATS=12 \
BACKEND=trl_sft \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp32 \
MAX_LENGTH=8 \
BATCH_SIZE=1 \
MAX_STEPS=10 \
DATASET_REPEATS=12 \
BACKEND=trl_dpo \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp32 \
MAX_LENGTH=8 \
BATCH_SIZE=1 \
MAX_STEPS=10 \
DATASET_REPEATS=12 \
GRPO_MAX_COMPLETION_LENGTH=1 \
BACKEND=trl_grpo \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

# Apple native MM8/MM4 quant, tiny-only.
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_quant_smoke.sh

# Apple native MM8/MM4 quant on converted 0.1B.
# MIN_PARAMS_LIST sweeps from lm_head-only into broader projection groups.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 MIN_PARAMS_LIST=8000000,1000000,500000 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_quant_smoke.sh

# 0.4B quant sweep: MIN_PARAMS=4000000 covers FFN key/value + lm_head modules.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 MIN_PARAMS_LIST=4000000 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_quant_smoke.sh

# Apple MLX bridge, tiny-only.
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_smoke.sh

# Apple MLX bridge on one real 0.1B projection tensor.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_smoke.sh

# Export selected HF safetensors into an MLX-readable bundle.
python scripts/convert_hf_to_mlx.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  /tmp/rwkv7-g1d-0.1b-mlx \
  --dtype fp16 \
  --include model.layers.0.attn.r_proj.weight \
  --copy-metadata \
  --results bench/results_apple_silicon_m5_20260704.jsonl

# Full MLX recurrent reference backend: tiny parity/cache plus optional 0.1B row.
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_model_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
CHUNK_SIZE=2 \
MAX_NEW_TOKENS=2 \
DYNAMIC_BATCH=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_model_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
CHUNK_SIZE=2 \
MAX_NEW_TOKENS=1 \
DYNAMIC_BATCH=1 \
SKIP_TINY=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_model_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
CHUNK_SIZE=2 \
MAX_NEW_TOKENS=1 \
DYNAMIC_BATCH=1 \
SKIP_TINY=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_model_smoke.sh

# Reusable tokenizer-integrated MLX text generation API / CLI.
python scripts/mlx_generate.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  --prompt "The quick brown fox" \
  --max-new-tokens 8 \
  --dtype fp16 \
  --results bench/results_apple_silicon_m5_20260704.jsonl

# Prompt/decode sweep with chunked-prefill correctness and memory telemetry.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16,64 \
DECODE_LENGTHS=2,4 \
CHUNK_SIZE=32 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=2 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_generation_sweep.sh

# Serving-shaped MLX session: prefill once, decode in chunks, compare with one-shot.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
STEP_SIZES=4,4 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_session_smoke.sh

# Optional stronger real-checkpoint parity against HF native PyTorch on CPU.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp32 \
TOKENS=1,2,3,4 \
CHUNK_SIZE=2 \
MAX_NEW_TOKENS=1 \
COMPARE_TORCH=1 \
TORCH_COMPARE_TOLERANCE=0.05 \
SKIP_TINY=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_model_smoke.sh
```

The Trainer wrapper calls `tests/test_apple_silicon_trainer_smoke.py` directly. The 0.1B/0.4B/1.5B model-training, TRL SFT, and TRL RL wrappers call `tests/test_apple_silicon_model_training_smoke.py`. The generation sweep wrapper calls `tests/test_apple_silicon_model_sweep.py`. The native quant wrapper calls `tests/test_apple_silicon_quant_smoke.py`. The MLX bridge wrapper calls `tests/test_apple_silicon_mlx_smoke.py`; the full recurrent MLX wrapper calls `tests/test_apple_silicon_mlx_model_smoke.py`; the reusable MLX generation CLI is `scripts/mlx_generate.py`; the MLX prompt/decode sweep CLI is `scripts/mlx_generation_sweep.py`; the isolated quant projection microbench is `scripts/mlx_quant_projection_bench.py`; the serving-style prefill-once/session-decode CLI is `scripts/mlx_session_smoke.py` with wrapper `scripts/run_apple_silicon_mlx_session_smoke.sh`; the interleaved multi-session CLI is `scripts/mlx_session_batch_smoke.py` with wrapper `scripts/run_apple_silicon_mlx_session_batch_smoke.sh` and `SESSION_BACKEND=batched|auto` for equal-round MLX batching; and the HF→MLX exporter is `scripts/convert_hf_to_mlx.py`.

Recorded rows: [`../../bench/results_apple_silicon_m5_20260704.jsonl`](../../bench/results_apple_silicon_m5_20260704.jsonl).

## Minimal Apple environment

Use an isolated environment. On Apple Silicon the base package should install
without FLA:

```bash
cd /path/to/rwkv7-hf-adapter
python3 -m venv .venv-apple-torch
source .venv-apple-torch/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install accelerate
```

For MLX bridge/export validation on Apple Silicon, install the optional MLX
extra:

```bash
python -m pip install -e '.[mlx]'
```

If `pip install -e .` is not desired, the lightweight fallback is:

```bash
python -m pip install torch torchvision torchaudio transformers safetensors accelerate
export PYTHONPATH=/path/to/rwkv7-hf-adapter:${PYTHONPATH:-}
```

CUDA users who want the optimized default backend should install the optional
extra instead:

```bash
python -m pip install -e '.[fla]'
# or, for CUDA/Triton development helpers:
python -m pip install -e '.[cuda]'
```

## Verify PyTorch MPS

```bash
python - <<'PY'
import platform
import torch
print('platform', platform.platform())
print('machine', platform.machine())
print('torch', torch.__version__)
print('mps built', torch.backends.mps.is_built())
print('mps available', torch.backends.mps.is_available())
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(torch.ones(1, device=device))
PY
```

## Smoke commands

Tiny native model only, safe to run before downloading model weights:

```bash
PYTHONPATH=. python tests/test_apple_silicon_smoke.py --device auto --dtype fp32
```

Converted model smoke, records a JSONL row:

```bash
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
RESULTS=bench/results_apple_silicon.jsonl \
DEVICE=auto \
DTYPE=fp32 \
MAX_NEW_TOKENS=2 \
scripts/run_apple_silicon_smoke.sh
```

MLX bridge smoke, safe tiny row plus optional converted-model projection row:

```bash
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_mlx.jsonl \
scripts/run_apple_silicon_mlx_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_mlx.jsonl \
scripts/run_apple_silicon_mlx_smoke.sh

python scripts/convert_hf_to_mlx.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  /tmp/rwkv7-g1d-0.1b-mlx \
  --dtype fp16 \
  --include model.layers.0.attn.r_proj.weight \
  --copy-metadata

DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_model_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
DYNAMIC_BATCH=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_model_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
SKIP_TINY=1 \
DYNAMIC_BATCH=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_model_smoke.sh

python scripts/mlx_generate.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  --prompt "The quick brown fox" \
  --max-new-tokens 8 \
  --dtype fp16

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16,64 \
DECODE_LENGTHS=2,4 \
CHUNK_SIZE=32 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=2 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

# Extended 0.4B / 1.5B MLX long-decode matrix: prompt4096 + decode256.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=4096 \
DECODE_LENGTHS=256 \
CHUNK_SIZE=1024 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=4096 \
DECODE_LENGTHS=256 \
CHUNK_SIZE=1024 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

# Initial MLX/Metal WKV custom-kernel seam smoke. This is an opt-in backend
# seam (`rwkv7_hf.mlx_wkv`), not the final production fused speed path.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=32 \
DECODE_LENGTHS=2 \
CHUNK_SIZE=16 \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
DECODE_LENGTHS=1 \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
DECODE_LENGTHS=1 \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
STEP_SIZES=4,4 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_session_smoke.sh

# Higher-pressure interleaved session matrix: 4 sessions, rounds=4,4, repeat=4.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_A="The quick brown fox" \
PROMPT_B="User: Apple Silicon RWKV test. Assistant:" \
PROMPT_C="Repeat pressure prompt for MLX sessions." \
PROMPT_D="Fourth concurrent MLX session for pressure." \
ROUNDS=4,4 \
REPEAT=4 \
SESSION_BACKEND=batched \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_session_batch_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_A="The quick brown fox" \
PROMPT_B="User: Apple Silicon RWKV test. Assistant:" \
PROMPT_C="Repeat pressure prompt for MLX sessions." \
PROMPT_D="Fourth concurrent MLX session for pressure." \
ROUNDS=4,4 \
REPEAT=4 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_session_batch_smoke.sh

# Higher-concurrency interleaved session matrix using synthetic prompt fill.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
SESSION_COUNT=6 \
ROUNDS=4,4 \
REPEAT=5 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_session_batch_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
SESSION_COUNT=5 \
ROUNDS=4,4 \
REPEAT=2 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_session_batch_smoke.sh
```

Tiny native training + optional PEFT LoRA training smoke:

```bash
RESULTS=bench/results_apple_silicon_training.jsonl \
DEVICE=auto \
DTYPE=fp32 \
REQUIRE_PEFT=1 \
scripts/run_apple_silicon_training_smoke.sh
```

The underlying test is [`../../tests/test_apple_silicon_training_smoke.py`](../../tests/test_apple_silicon_training_smoke.py).

If the converted model directory was produced by an older checkout, sync the
current remote-code files first:

```bash
python scripts/sync_hf_adapter_code.py /path/to/rwkv7-g1d-0.1b-hf
```

## Validation matrix to fill

| Machine | Memory | Model | Dtype | Device | Required result |
|---|---:|---|---|---|---|
| M1 / M2 Air | 16GB | tiny native | fp32 | mps or cpu | `APPLE SILICON SMOKE PASS` |
| M1 / M2 Air | 16GB | 0.1B HF | fp32 | mps | load + forward + 2-token generate + PEFT LoRA/Trainer/SFT/DPO/GRPO smoke |
| M-series 16GB+ | 16GB+ | tiny + selected 0.1B HF tensors | fp16 | MLX GPU | `scripts/run_apple_silicon_mlx_smoke.sh` tiny save/load/matmul + HF projection matmul, and optional `scripts/convert_hf_to_mlx.py` export manifest |
| M-series 16GB+ | 16GB+ | tiny + 0.1B/0.4B/1.5B HF | fp16 / fp32 | MLX GPU | `scripts/run_apple_silicon_mlx_model_smoke.sh` tiny MLX/Torch recurrent parity, state-cache select, chunked prefill, tokenizer prompt, dynamic-batch state select, 0.1B/0.4B/1.5B full MLX recurrent prefill/generate, optional HF native PyTorch compare, `scripts/mlx_generate.py` reusable text generation, `scripts/run_apple_silicon_mlx_generation_sweep.sh` prompt/decode sweep plus repeat pressure with chunked-prefill checks, `scripts/mlx_quant_projection_bench.py` dense/affine/Metal/auto quant projection microbench, `scripts/run_apple_silicon_mlx_session_smoke.sh` prefill-once/session decode equality vs one-shot, and `scripts/run_apple_silicon_mlx_session_batch_smoke.sh` interleaved multi-session equality vs one-shot plus repeat-pressure summary telemetry |
| M-series 16GB+ | 16GB+ | 0.4B HF | fp32 / fp16 | mps | load + forward + generate + prompt-length sweep through 512 tokens + PEFT LoRA/Trainer/SFT/DPO/GRPO 1-step/2-step smoke + memory note |
| M-series 16GB+ | 16GB+ | tiny + 0.1B/0.4B/1.5B HF | fp32 native MM8/MM4 | mps | bitsandbytes-free native quant smoke + min-params sweep + packed-footprint ratio + finite forward/generate; 1.5B on 16GB is memory-tight evidence only |
| M-series 16GB+ | 16GB+ | 1.5B HF | fp16 inference / fp32 LoRA smoke | mps | load/generate + MPS prompt sweep through 512 tokens / decode 8 + MLX prompt8192/decode512 + PEFT manual + Trainer/SFT/DPO/GRPO 1/2/3/5/10/12-step + peak memory + finite trainable update |
| M-series Max / Ultra | 64GB+ | 1.5B+ HF | fp16 / bf16 | mps | production-length decode, 12+ step Trainer/TRL rows, peak memory, tok/s |

For every Apple result, include:

- macOS version, chip, memory size;
- Python / PyTorch / Transformers versions;
- `torch.backends.mps.is_built()` and `is_available()`;
- command line and JSONL result row;
- Activity Monitor or `memory_pressure` notes if the run swaps heavily.

Apple harness maintenance note: shared script helpers live in
`tests/apple_silicon_utils.py`. New MPS/MLX smoke rows should use it for
hardware probes, JSONL output, package versions, model-size labels, device/dtype
selection, and MPS memory telemetry instead of copying helper blocks into each
test script.

## Known limitations

- This is not an Albatross-speed path. PyTorch MPS validates HF compatibility on
  Apple hardware but does not replace CUDA fused kernels.
- `bitsandbytes` quantization is not an Apple path. Native MM8/MM4 now has
  MPS functional smoke plus 0.1B/0.4B/1.5B min-params sweeps with
  packed-footprint telemetry. MLX has both the portable packed W8/W4 affine
  dequant-matmul projection path and an opt-in Metal fused dequant-projection
  seam for 0.1B/0.4B/1.5B, now including 0.4B/1.5B prompt128/256 decode4/8
  and prompt512/1024 decode16 pressure rows plus same-shape fp16 Metal baselines.
  The opt-in direct grouped R/K/V seam now also has W8/W4 prompt512/decode16
  rows, broader-threshold prompt2048/decode128 rows, and 0.4B 8-session direct grouped pressure rows with grouped fallback=0. 1.5B direct grouped rounds8,8 now has safe sequential W8/W4 rows and a W8 strict-batched match, while W4 strict-batched remains a documented correctness gap. Quant+Metal session-batch pressure rows now pass
  for 0.4B W8/W4 4-session repeat=2 and 6-session repeat=3, plus 1.5B W8/W4
  4-session repeat=1 and 5-session repeat=2. The opt-in equal-round batched session backend now has
  W4 rows on 0.4B 6-session repeat=2, 0.4B 8-session rounds8,8 repeat=2,
  1.5B 5-session repeat=1, and 1.5B 5-session rounds8,8 repeat=2 with
  one-shot equality and aggregate round tok/s telemetry. Strict W8/Metal batched
  longer decode had a batch-exactness gap; default W8/Metal `SESSION_BACKEND=auto`
  still records `auto_mm8_metal_batch_exactness_guard` and falls back unless
  `RWKV7_MLX_SESSION_AUTO_W8_STABLE=1` opts into the stable argmax policy. Backend-compare rows
  now keep this visible without failing the safe path: 0.4B/1.5B W4 match,
  1.5B W8 matches in this matrix, and 0.4B W8 has a reproducible mismatch at
  token index 6 for the short prompt. Optional mismatch-logit tracing shows the
  divergent W8 token is a near-tie (token 11 vs 261, max-abs delta≈0.03125),
  so this is now localized for the next W8 exactness fix. The explicit
  `SESSION_BACKEND=batched_stable` rows now close the 0.4B compare gate and pass
  longer 0.4B/1.5B rounds8,8 repeat pressure, including 1.5B W8/W4 repeat=4 rows; default W8/Metal auto stays
  guarded, while `RWKV7_MLX_SESSION_AUTO_W8_STABLE=1` enables an opt-in stable
  auto route. `--quant-backend auto` now records backend-count telemetry and
  routes W4 normal prefill/decode rows to Metal while keeping W8 on affine by
  default; 0.4B W4/W8 auto rows pass, and W8 auto can now batch safely when it
  resolves to affine.
  Current long-context ratio evidence shows memory wins and an optimized 0.4B
  W4 prompt2048/decode128 fp16-beating row (decode≈1.25x fp16, peak≈0.56x),
  while prompt4096/decode256 shows W8/W4 below fp16 despite peak≈0.70x/0.54x,
  and prompt8192/decode512 shows 1.5B W4 auto at decode≈0.81x fp16 with peak≈0.54x. Production Apple W8/W4
  still needs longer repeat/session pressure, more M-series coverage, and stable
  speed rows that beat fp16 end to end.
- The MLX path is now a correctness-first recurrent reference backend plus an
  initial opt-in MLX/Metal WKV custom-kernel seam, not a production-speed
  backend. It verifies HF safetensor loading/export, full
  recurrent prefill/decode equations, tokenizer prompt handling/API, state-cache
  select, chunked prefill, dynamic-batch row selection, prefill-once/session
  decode equality vs one-shot, interleaved multi-session equality vs one-shot with an opt-in equal-round batched session backend, 0.1B/0.4B/1.5B 3-session rows plus 0.4B/1.5B 4-session repeat-pressure summary rows and higher-concurrency 0.4B 6-session / 1.5B 5-session rows, prompt/decode sweeps through 4096-token prompts on 0.4B and 8192-token prompts / 512-token decode on 1.5B plus repeat pressure rows,
  0.1B/0.4B/1.5B MLX packed W8/W4 affine quant projection rows, 0.1B/0.4B/1.5B `--quant-backend metal` fused dequant-projection rows with 0.4B/1.5B prompt128/256 decode4/8 plus prompt512/1024 decode16 pressure, quant+Metal higher-concurrency session-batch rows, 0.1B/0.4B/1.5B short greedy decode, and 0.1B/0.4B/1.5B `--wkv-backend metal` smoke rows. Production fused
  WKV/projection, Metal fused quant/dequant, still-larger production prompt/decode pressure,
  and production serving integration are still open.
- Long-running full-size training on MPS is not claimed yet. Tiny native Trainer
  and tiny PEFT LoRA Trainer pass; 0.1B and 0.4B PEFT LoRA backward, HF Trainer,
  TRL SFT, DPO, and GRPO one-step and 2-step smoke pass on a 16GB M5. 1.5B
  fp32 PEFT LoRA manual backward, HF Trainer, and TRL SFT/DPO/GRPO 1/2/3/5/10/12-step
  smoke now pass. Longer MLX 1.5B decode through 512 tokens now passes; longer production-style training/decode and larger Apple machines
  are still open. Native MM8/MM4 functional/min-params smoke through 1.5B and
  initial MLX recurrent reference smoke, MLX packed W8/W4 affine projection smoke, an initial Metal WKV seam, and an initial Metal W8/W4 dequant-projection seam through 0.4B/1.5B prompt512/1024 decode16 are present; repeat/session MLX/Metal WKV/projection acceleration
  and production quant speed gates are still open; the current fp16 ratio gate
  proves W8/W4 memory reduction but not speed parity yet.
- 1.5B fp16 PEFT LoRA on the 16GB M5 produced non-finite gradient/update values
  in one local trial. The training smoke now rejects non-finite or zero
  trainable-gradient/update totals instead of recording false-positive rows.
- 16GB machines should start with tiny / 0.1B, then short 0.4B generate
  before longer sweeps. Close browsers and IDEs before running converted-model
  smoke.

## MLX / Metal references

RafaelUI's Apple-focused RWKV work is the most relevant starting point for the
next backend layer:

- [RafaelUI/metal-wkv7](https://github.com/RafaelUI/metal-wkv7): custom Metal WKV-7 forward/backward kernel.
- [RafaelUI/rwkv-metal](https://github.com/RafaelUI/rwkv-metal): Apple Silicon RWKV-7 training / LoRA / QLoRA direction.
- [RafaelUI/rwkv-mlx](https://github.com/RafaelUI/rwkv-mlx): MLX RWKV-7 pretraining / conversion / LoRA direction.
- [RafaelUI/SwiftRWKV](https://github.com/RafaelUI/SwiftRWKV): Swift + MLX / Apple platform inference direction.

## Next engineering steps

1. Extend 0.4B Apple rows beyond 2 training steps.
2. Extend 1.5B beyond 12-step Trainer/TRL and prompt8192/decode512 MLX sweep to
   longer production-style training/decode and memory-pressure notes.
3. Extend the initial MLX packed W8/W4 affine and Metal quant paths beyond the current 0.4B prompt4096/decode256 and 1.5B prompt8192/decode512 ratio rows, 0.4B 8-session repeat=2, and 1.5B 5-session repeat=4 quant+Metal session-batch rows to still-longer repeat/session pressure, memory-pressure notes, and stable fp16-beating gates.
4. Extend the MLX recurrent reference and `MLXGenerationSession` / batched session backend beyond the current
   0.1B prompt256/decode8, 0.4B prompt4096/decode256 and 1.5B prompt8192/decode512 matrices, 0.4B 8-session repeat=2, and 1.5B 5-session repeat=4 rows to longer prompt distributions,
   stronger memory-pressure telemetry, and longer production-style concurrent session reuse.
5. Extend the initial Metal WKV and W8/W4 dequant-projection seams into a production fused path that can beat fp16 end to end.
6. Decide whether the production Metal WKV-7 kernel belongs in this repo as an
   optional backend or in a sibling `rwkv7-mlx` / `rwkv7-metal` package.
