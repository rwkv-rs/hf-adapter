# AMD gfx1100 post-rebase validation (2026-07-28)

This artifact replays the AMD production gates after rebasing the branch onto
upstream `main` at `c8a29d9`. The tested branch tree contains the decoupled
`model_runtime_policy.py` and `model_runtime.py` paths from both sides of the
rebase; no FLA model/runtime dependency is restored.

## Environment

- AMD Navi 31, runtime architecture `gfx1100`, 47.98 GiB VRAM.
- ROCm 7.2.1, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`.
- fp16 inference and bf16 Trainer smoke.
- Converted RWKV-7 G1D 0.4B and G1H 1.5B/2.9B/7.2B/13.3B
  checkpoints.

`full_0p4b/environment.log` records the exact launch policy. The runner checks
that full-model quantization and fused prefill remain fail-closed while exact
`gfx1100` receives only the measured fused-decode and output-head MM8/MM4
launch rows.

## Complete native-HF replay

`bench/run_amd_rocm_hf_validation.sh` completed with
`AMD ROCm HF VALIDATION PASS` after the rebase. The replay includes:

- FLA-free import, native checkpoint metadata, HF generation/beam API;
- PEFT LoRA, with 144 trainable tensors receiving nonzero gradients;
- bf16 `Trainer`, six optimizer steps and 144/144 trainables updated;
- heterogeneous dynamic batch/cache reorder, compaction and in-place reorder;
- chunked-prefill correctness and B1 dense/chunk speed measurement;
- B1/B2/B4/B8 dense inference;
- fused decode policy A/B at B1 and B8.

The final source-level regression selection passed **84/84 tests**, covering
the runtime/model split, every `native_jit` split, exact-card kernel and quant
policy, cross-card isolation, benchmark contracts and checkpoint code sync.
An additional **8/8 documentation/contract tests** passed after updating the
large-model status.

| batch | prefill tok/s | decode tok/s |
|---:|---:|---:|
| 1 | 262.5 | 141.1 |
| 2 | 471.4 | 282.6 |
| 4 | 936.1 | 538.8 |
| 8 | 1861.9 | 1062.9 |

Chunk sizes 32/64/128 preserved the greedy output and reached
`1.4685x/1.6343x/1.7176x` the measured full-prefill baseline. Fused decode was
`1.7469x` at B1 and `1.7482x` at B8, with 32/32 and 256/256 greedy matches.

## Post-rebase W8/W4 speed matrix

`quant_speed_matrix.jsonl` is a 24-cell same-process paired matrix:

- model sizes: 0.4B, 1.5B, 2.9B;
- batch sizes: 1, 2, 4, 8;
- formats: native MM8 and MM4;
- prompt/decode: 128/128 tokens, three timing repeats.

Every cell passes all promotion gates: model footprint below fp16, decode not
slower than fp16, exact tested greedy stream, deterministic repeats, and high
final-logit cosine.

| model | footprint range / fp16 | decode range / fp16 | minimum cosine |
|---|---:|---:|---:|
| 0.4B | 0.8886-0.9258 | 1.0797x-1.0922x | 0.99983478 |
| 1.5B | 0.9342-0.9562 | 1.0384x-1.0491x | 0.99980903 |
| 2.9B | 0.9573-0.9716 | 1.0226x-1.0273x | 0.99982536 |

This verifies the output-head speed lane only. It does not promote the known
slower full-model memory lane, other AMD architectures, or fused prefill.

## 7.2B large-checkpoint replay

The pinned 7.2B checkpoint was downloaded with the size/SHA256 gate in
`prepare_rwkv7_g1_validation_models.py`, converted through its low-memory path,
and replayed with the same native remote-code closure. FLA-free import and HF
generation passed.

The dense fused-decode policy reached `1.2291x` at B1 and `1.2911x` at B8,
with 16/16 and 128/128 greedy matches. All eight output-head quant cells
(B1/B2/B4/B8 x MM8/MM4) passed:

| footprint range / fp16 | decode range / fp16 | minimum cosine | greedy |
|---:|---:|---:|---:|
| 0.9720-0.9814 | 1.0272x-1.0362x | 0.99955499 | 8/8 pass |

## 13.3B large-checkpoint replay

The pinned 13.3B checkpoint passed the same size/SHA256 gate, low-memory
conversion, native remote-code sync, FLA-free import and HF generation checks.

The dense fused-decode policy reached `1.2131x` at B1 and `1.2909x` at B8,
with 8/8 and 64/64 greedy matches. Its eight output-head quant cells also
passed:

| footprint range / fp16 | decode range / fp16 | minimum cosine | greedy |
|---:|---:|---:|---:|
| 0.9848-0.9899 | 1.0123x-1.0191x | 0.99978411 | 8/8 pass |

Across all five model sizes, the post-rebase output-head quant replay is
**40/40 passing cells**.

## Files

- `full_0p4b/runner.log`: complete production-runner transcript.
- `full_0p4b/results.jsonl`: dense and chunked-prefill measurements.
- `full_0p4b/decode_policy_ab.jsonl`: fused-decode paired A/B.
- `full_0p4b/*.log`: API, PEFT, Trainer, cache and policy evidence.
- `quant_speed_matrix.jsonl`: canonical post-rebase 24-cell quant matrix.
- `large_7p2b/`: 7.2B FLA-free/generate, dense-policy and eight-cell quant
  replay transcripts.
- `large_13p3b/`: 13.3B FLA-free/generate, dense-policy and eight-cell quant
  replay transcripts.
- `model_preparation.json`: pinned source size/SHA256 and conversion status for
  both large checkpoints; each large-model directory also retains its prepare
  transcript.
- `summary.json`: machine-readable aggregate and gate result.
- `focused_pytest.log`: final 84-test post-rebase regression result.
- `docs_pytest.log`: final eight-test documentation/contract result.
- `SHA256SUMS`: integrity hashes for every retained artifact.
