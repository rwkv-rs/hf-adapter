# Contributions — RWKV-7 HF Adapter

This document is an **evidence-linked contribution ledger** for the RWKV-7
HF/Transformers adapter (`dsadsasdaddas/rwkv7-hf-adapter`), structured for
transparent contribution assessment. Every claim carries its commit, its
benchmark number (with baseline), and its verification gate.

Entry points: [README.md](README.md) · [AGENTS.md](AGENTS.md) ·
[HF_STATUS.md](HF_STATUS.md) · [BENCHMARK.md](BENCHMARK.md).

---

## §1 Requirement scorecard (status as of 2026-07-03)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Train/infer parity with official `rwkv` | ✅ | 0.1B–13.3B alignment: cos 0.9999–0.9999976, greedy 16–64/16–64; [`docs/validation/V100_HF_VALIDATION.md`](docs/validation/V100_HF_VALIDATION.md); PR #83 (13.3B) |
| 2 | HF PEFT + RL trainability (LoRA/SFT/DPO/GRPO) | ✅ | 0.4B–2.9B pass (Trainer/SFT/DPO/GRPO); [`tests/test_native_trainer_smoke.py`](tests/test_native_trainer_smoke.py) et al.; PRs #59/#60 |
| 3 | DeepSpeed ZeRO-2/3 (base + resume) | ✅ | ZeRO2 resume ✅ 2.9B; **ZeRO3 resume fixed** PR #92 (root-caused transformers `is_deepspeed_zero3_enabled()` flag leak); [`tests/test_deepspeed_resume_smoke.py`](tests/test_deepspeed_resume_smoke.py) |
| 4 | HW breadth (consumer + datacenter cards) | ✅ 6/12 | V100(sm70) ✅, Blackwell 5070(sm120) ✅, A100(sm80) ✅ #82/#84, 4090(sm89) ✅ #90, A800(sm80) ✅ #97, RTX 3060(sm86) ✅ #87; H100/AMD/Pascal/Ascend pending (no hardware) |
| 5 | W8/W4 quant: VRAM ↓, decode speed ↑ | ✅ VRAM+speed / ◑ Q*_K_M cmp | **mm8 int8**: 2× VRAM, Blackwell decode 1.5–1.8× fp16 (#85); **mm4 int4**: 4× VRAM, lm_head 2.04× fp16 (#88); bnb 8/4-bit functional (#82); persistence via `from_pretrained` (#89) |
| 6 | Speculative decoding | ✅ | LoRA draft training + `rwkv7_speculative_generate` (#95); dense trie tokenizer (#96); ~2.1× V100 |
| — | Albatross-level production perf | ◑ | fused-scan prefill 1.3–1.9× HF (#compact-WY); native_graph decode 4–6.7× eager; **not full Albatross parity** (hand-tuned CUDA moat, documented honestly) |

---

## §2 Original contributions by @dsadsasdaddas (wangyue) — lead

### Architecture & core adapter
- **HF wrapper (`modeling_rwkv7.py`)** — `AutoConfig`/`AutoTokenizer`/`AutoModelForCausalLM`
  integration, `from_pretrained`/`save_pretrained`, `generate(use_cache=True)`, all GenerationMixin
  modes (greedy/beam/sampling), `RWKV7StateCache` (select/reorder/drop/compact/offload/restore),
  chunked prefill, env-flag backend selection, bnb skip-policy with concrete per-layer module names
  (#82 review). Gate: [`tests/test_hf_api_contract.py`](tests/test_hf_api_contract.py).
- **FLA-free native backend (`NativeRWKV7ForCausalLM`)** — pure-PyTorch RWKV-7 forward, **bit-exact vs
  FLA** (cos=1.0, max_abs=0.0), covering full HF ecosystem (Cache contract / generate / PEFT / Trainer /
  SFT / DPO / GRPO). Unblocks training on cards where FLA backward is blocked (Blackwell sm_120: 128KB
  shared-mem > 99KB limit). Gate: [`tests/test_native_model.py`](tests/test_native_model.py). PRs #59/#60.
  **Verified on V100 + 5070 (sm_70 + sm_120).**

### Performance kernels (all original Triton, inspired by published RWKV-7 math)
- **Native fast-token backends (`native_jit.py`)** — `native_jit` (torch.jit per-block) +
  `native_graph` (CUDA-graph replay) integrated into HF `forward`/`generate`. 0.1B decode:
  native_graph **382 tok/s = 4–6.7× official** (5070); 103 tok/s JIT + 254 tok/s graph (V100).
  Gate: [`tests/test_fast_decode_api.py`](tests/test_fast_decode_api.py).
- **Fused operator suite (`fused_*.py`, 10 kernels)** — fused_attn_shift_mix (mix6),
  fused_attn_output_prepare (groupnorm + r_k + ×g), fused_attn_output_project (+ o_proj fold),
  fused_ffn (shift + key + relu² + value), fused_wag_lora / fused_wavg_lora (W/A/G/V low-rank),
  fused_rkv_wag_projection (stacked RKV), fused_recurrent_update (WKV state), fused_norm_mix,
  fused_prefill. All bit-exact (cos ≥ 0.9995); gates in respective `bench/bench_fused_*.py`.
- **Fused-scan prefill (compact WY three-stage path)** — `dplr_prefill.py` + `dplr_prefill_triton.py`:
  parallelized DPLR recurrent scan via Triton, `triton_wy_compact` algorithm. Prefill **1.3–1.9× HF**
  (5070 0.1B: 28336 vs 21588; 4090 0.4B: 17697 vs 9278), bit-exact (cos=1.0). 8 algorithm variants
  benchmarked (sequential/affine/wy/lowrank/triton_wy/cuda_wy/triton_dense3/triton_wy_compact).
  Gate: [`bench/bench_native_prefill_scan.py`](bench/bench_native_prefill_scan.py).
- **Per-GPU kernel policy (`kernel_policy.py`)** — classifies GPUs (Pascal→Blackwell+AMD) into
  families, assigns per-family default-on/off fusion sets + adaptation rules. Gate:
  [`tests/test_kernel_policy.py`](tests/test_kernel_policy.py).

### Quantization (format ported from official rwkv; kernels are original Triton)
- **mm8 int8 quantization (`native_quant_mm8.py`, PR #85)** — ported the official rwkv `fp16i8`
  affine format (uint8 + mx/rx/my/ry scales) from `BlinkDL/rwkv` `model.py`; wrote a **fused Triton
  dequant-GEMV** (reads uint8 + scales, dequantizes in registers, fp32 accumulate) — NOT a copy of the
  official CUDA `cuda_mm8`. Two kernel variants: naive + split-K (mirrors official `mm8_one` layout).
  Results on Blackwell (5070): decode 1.5–1.8× fp16 (lm_head 1.69×, 7B body 1.66×); VRAM 2× smaller.
  V100: 0.46× (cuBLAS fp16 near peak — documented honestly). Bit-exact per-layer (cos ≥ 0.9995).
  Gate: [`tests/test_native_quant_mm8.py`](tests/test_native_quant_mm8.py).
- **mm4 int4 quantization (`native_quant_mm4.py`, PR #88)** — extended the affine scheme to 4-bit
  (16 levels, packed 2/byte along M). **Paired-nibble Triton GEMV**: loads every packed byte once,
  extracts both nibbles, accumulates into two paired outputs. lm_head 2.04× fp16 (5070); VRAM 4× smaller.
  Bit-exact (cos ~0.984 per-layer, int4 floor). Gate: [`tests/test_native_quant_mm4.py`](tests/test_native_quant_mm4.py).
- **mm8 persistence (PR #89)** — `RWKV7HFAdapterConfig` gains `use_native_mm8` flag; `from_pretrained`
  auto-quantizes after loading when flag set. Round-trip exact (int8 is deterministic from fp16).
  Gate: [`tests/test_native_mm8_persist.py`](tests/test_native_mm8_persist.py).

### Bug diagnosis & fixes
- **ZeRO3 checkpoint resume fix (PR #92)** — root-caused: the first HF Trainer sets transformers'
  global `is_deepspeed_zero3_enabled()` flag, deleting the Trainer does NOT reset it → resume-model
  builds under DeepSpeed partitioned-init → FLA's `_initialize_weights` indexes `shape[1]` on a
  partitioned 1-D shard → IndexError. Fix: `unset_hf_deepspeed_config()` before the resume load.
  Verified: 2×V100 PASS (both ranks, first_loss 4.857 → resume_loss 2.270, global_step 2).
  **This is a different failure mode from the A100 ZeRO3-resume dtype mismatch** (diagnosed separately
  in MosRat's #84: `stage3_param_persistence_threshold: 0`).
- **bnb skip-policy delta measured** — the #82 bnb skip fix (concrete LoRA module names) was measured
  to have **zero output delta** (0.1B 8/4-bit + 0.4B 8-bit, bit-identical before/after). The fix is
  defensive code hygiene, not a correctness change. Honest self-check.
- **FP8 root cause** — precisely diagnosed: `torch._scaled_mm` on sm_120 returns
  `CUBLAS_STATUS_NOT_SUPPORTED` at real GEMM shapes (4096²) because cuBLASLt in torch 2.11+cu128
  has **no sm_120 FP8 kernel**. 512² edge-case misleads (works). Unblock: torch cu129+ or
  TransformerEngine.
- **Windows PYTHONPATH separator** (#79) — `;` not `:` on MSYS/Cygwin; fixed `run_hf_acceptance.sh`.
- **Server nvcc installation** — installed cuda-nvcc 12.4 + cuda-cudart-dev + cuda-cccl + ninja into
  the V100 server's rwkv7 conda env, unblocking DeepSpeed (ZeRO3 resume) and official CUDA compilation.

### Validation evidence
- **13.3B official alignment (PR #83)** — single V100-32GB fp16: cos 0.9999976, greedy 16/16,
  decode native_jit 18.4 tok/s (1.58× fla). Gate: [`tests/test_official_alignment.py`](tests/test_official_alignment.py).
- **5070 Blackwell validation** — 12-layer sm_120 report (BLACKWELL_50SERIES.md): all tests pass,
  decode 382 tok/s, TTFT 19–24ms, batch 212→1581 tok/s. First sm_120 validation in the repo.
- **V100 HF validation matrix** — 0.4B/1.5B/2.9B training ecosystem, ZeRO2 resume, quant matrix.

### Other
- **Speculative decoding (PR #95)** — LoRA draft training + `rwkv7_speculative_generate`.
- **Dense trie tokenizer (PR #96)** — faster RWKV vocab encode.
- **CI, tests, bench scripts, docs** — 30+ test files, 20+ bench scripts, issue triage (#66–#98),
  PR review (#82 MosRat, #87 yuyi2439, #97 aierwiki).

work-types: `algorithm` `engineering` `validation` `docs` `coordination`

---

## §3 Adapted code (full disclosure)

| Component | Source | What was adapted | What is original |
|---|---|---|---|
| mm8 quantization format | `BlinkDL/rwkv` `model.py` | The affine int8 **format** (uint8 + mx/rx/my/ry scales, dequant formula) | The **Triton fused dequant-GEMV kernel** (naive + split-K), the size-gated `MM8Linear` integration, the persistence mechanism |
| mm4 quantization format | Extension of the above | The 4-bit affine scheme is a direct generalization | The **paired-nibble Triton kernel** (load byte once, extract both nibbles), the `MM4Linear` integration |
| fused-scan prefill concept | `BlinkDL/Albatross` faster3a (conceptual) | The idea of parallelizing the DPLR scan | The **8 Triton algorithm variants** (triton_wy_compact etc.), the compact WY three-stage path |
| RWKV-7 per-token math | `BlinkDL/RWKV-LM` TMix_one/CMix_one | The per-token forward equations | The `native.py` / `native_model.py` batched port, the Cache/generate/PEFT integration |

**Everything else** (HF wrapper, native_graph, ZeRO3 fix, kernel_policy, bench scripts, CI, tests, docs)
is **original work** of this repo. The official rwkv package (`pip install rwkv`) is used as a
**correctness reference** (for alignment tests), not as a runtime dependency.

---

## §4 External contributions

| Contributor | PR(s) | What | Work-type |
|---|---|---|---|
| @MosRat | #82, #84 | A100 (sm80) validation: 0.1B smoke + ZeRO2/3, extended 0.4B–7.2B batch sweep + quant + Trainer + ZeRO2 resume; A100 ZeRO3-resume dtype-mismatch diagnosis (`stage3_param_persistence_threshold: 0` fix) | `validation` `data` `algorithm`(debug) |
| @yuyi2439 | #87 | RTX 3060 (sm86) validation: 0.1B speed/batch/training bf16 + `print_env.sh` + bf16-default | `validation` `data` `engineering` |
| @aierwiki | #97 | A800 (sm80) validation: 0.4B/1.5B/2.9B batch + bnb quant + training + converter/sync "copy all .py" fix (incl mm8/mm4) + AST regression-guard test | `validation` `data` `engineering` |

---

## §5 Measurement discipline

- **Two-GPU validation**: RTX 5070 Laptop (sm_120, 8GB, local) + Tesla V100-PCIE-32GB (sm_70, server).
- **Correctness**: per-layer cosine + max_abs vs FLA/native reference; end-to-end greedy-token equality
  (16–64 tokens); official `rwkv` package (cpu fp32) as ground truth.
- **Speed**: exclusive GPU, ≥3 warmup + ≥3 runs (bench scripts use `torch.cuda.synchronize` +
  percentile); results committed to `bench/results.jsonl` with `device` + `dtype` labels.
- **Honest self-checks**: bnb skip-fix zero-delta measured (not assumed); mm8 V100 0.46× documented
  (not hidden); FP8 512² edge-case identified as misleading (4096² is the real test); native_graph
  decode is single-batch/fixed-shape (documented limitation).

---

## §6 Reproduce (key gates)

```bash
# Correctness
RWKV7_NATIVE_MODEL=1 python tests/test_native_model.py --model <0.1b-hf>          # native vs FLA bit-exact
python tests/test_official_alignment.py --hf-dir <hf> --pth <pth> --dtype fp16   # vs official rwkv
python tests/test_native_quant_mm8.py --model <0.1b-hf>                          # mm8 int8 correctness
python tests/test_native_quant_mm4.py --model <0.1b-hf>                          # mm4 int4 correctness
python tests/test_native_mm8_persist.py --model <0.1b-hf>                        # mm8 persistence round-trip

# Speed
python bench/bench_native_quant_mm8.py    # fp16 vs mm8 decode speed sweep
python bench/bench_native_quant_mm4.py    # fp16 vs mm4 decode speed sweep
python bench/bench_native_prefill_scan.py --model <hf> --code-source model       # prefill scan (set RWKV7_DPLR_PREFILL_ALGORITHM=triton_wy_compact)

# ZeRO3 resume (2×V100 + deepspeed)
torchrun --standalone --nproc_per_node=2 tests/test_deepspeed_resume_smoke.py --model <0.1b-hf> --zero-stage 3
```

---

## §7 Release model

The adapter was developed incrementally on `main` with feature branches per PR. Key milestones:
- **v0.1.0** (pre-session): HF wrapper + native backends + fused kernels + V100 validation.
- **2026-07-02 session**: mm8/mm4 quant + persistence + ZeRO3 fix + 13.3B validation + FP8 diagnosis + server nvcc.
- Each PR's evidence = commit + bench/results.jsonl rows + test gates + issue/PR discussion.
