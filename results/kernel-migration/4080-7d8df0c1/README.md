# RTX 4080 optional-kernel v1 evidence

This compact bundle records the first clean-boundary acceptance run. Raw logs, model weights, generated checkpoints, and samples remain on the validation host.

## Identity

- Source SHA: `7d8df0c1` (core optional protocol: `0b26ce24`).
- GPU: NVIDIA GeForce RTX 4080; driver 595.84; CUDA 13.0.
- Torch 2.11.0+cu130; Transformers 5.8.0; Triton 3.6.0.
- FLA pinned commit: `80e494f6c588e091fc8316b612870df29375c5b8`.
- HF wheel SHA256: `07b4f6668c3123a3e996e33d4fab8230c468db23bbd7249c3454a93e2f04338f`.
- Kernel wheel SHA256: `31c0892a5284a26f89790567dbbdf4f6255b996cf5f7a32c14fa2406c15e24c9`.

## Correctness and HF contracts

- Graph route `torch-cuda-graph-reference-v1`: **passed** 12/12 FP16 operator cases and 0.1B/0.4B/1.5B model/cache/64-token-greedy gates.
- Triton route `native-triton-rank1-scan-v1`: operator 12/12, all finite, state/cache/64-token greedy passed. The strict aggregate remains **failed** because 0.4B B1/T17 logits reached max-abs `0.15625`, just above the fixed `0.15` gate.
- Graph and Triton both passed package-free AutoConfig/AutoTokenizer/AutoModel/AutoModelForCausalLM, greedy, beam, save/reload, and model training reference fallback.
- A separate environment with neither `rwkv7-hf` nor `rwkv7-kernels` installed loaded the self-contained 0.1B directory and selected `torch-reference-v1`.
- Clean reference vs FLA 0.4B: operator/state/64-token-greedy gates passed; the overall diagnostic is outside thresholds because one B4/T128 full-model logit case had max-abs `0.1875` (>0.15). This is retained rather than relabeled as a pass.

## Fair eager recurrent operator speed

FP16 inputs, FP32 canonical state, H=2, K=V=64, warmup=2, repeats=5; no model-level CUDA Graph and no `torch.compile`.

| Case | Reference ms | Graph ms | Graph × | Triton ms | Triton × | FLA fused ms | Triton vs fused × | FLA chunk ms | Triton vs chunk × |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b1_t1 | 0.0954 | 0.0709 | 1.35 | 0.0549 | 1.74 | 0.0728 | 1.33 | 0.2380 | 4.34 |
| b1_t17 | 1.0126 | 0.3807 | 2.66 | 0.0499 | 20.29 | 0.0579 | 1.16 | 0.2311 | 4.63 |
| b1_t128 | 7.3088 | 2.5417 | 2.88 | 0.1097 | 66.60 | 0.1238 | 1.13 | 0.2297 | 2.09 |
| b1_t512 | 29.0523 | 10.0192 | 2.90 | 0.3244 | 89.56 | 0.3520 | 1.09 | 0.2297 | 0.71 |
| b4_t1 | 0.3020 | 0.1290 | 2.34 | 0.0385 | 7.85 | 0.0575 | 1.49 | 0.2206 | 5.73 |
| b4_t17 | 4.1217 | 1.3766 | 2.99 | 0.0458 | 90.02 | 0.0589 | 1.29 | 0.2269 | 4.96 |
| b4_t128 | 30.4340 | 10.0303 | 3.03 | 0.1075 | 283.04 | 0.1228 | 1.14 | 0.2352 | 2.19 |
| b4_t512 | 121.0198 | 40.4511 | 2.99 | 0.3255 | 371.85 | 0.3510 | 1.08 | 0.2350 | 0.72 |
| b8_t1 | 0.5691 | 0.2097 | 2.71 | 0.0380 | 14.97 | 0.0542 | 1.43 | 0.2291 | 6.03 |
| b8_t17 | 8.2511 | 2.6476 | 3.12 | 0.0461 | 178.95 | 0.0578 | 1.25 | 0.2194 | 4.76 |
| b8_t128 | 61.3335 | 20.0259 | 3.06 | 0.1073 | 571.71 | 0.1231 | 1.15 | 0.2227 | 2.08 |
| b8_t512 | 243.6704 | 76.6180 | 3.18 | 0.3270 | 745.26 | 0.3524 | 1.08 | 0.2397 | 0.73 |

Interpretation: Triton beats FLA fused recurrent in all 12 measured cases (1.08×–1.49×). FLA chunk is faster at the longest T=512 cases, so this does not establish whole-model or long-prefill superiority.

## Scope still open

- FP32/BF16 matrix, explicit left/right-padding model cases, and full 0.1B/0.4B/1.5B shape expansion.
- Whole-model prefill/decode tables and forward+backward performance.
- Three-way 144-unit `lm_eval` equivalence (`hf-reference` / `hf-optimized` / `fla-rwkv7`).
- Triton numerical gate correction or a documented release-policy decision. Graph remains the conservative exact default.
